"""
audio_pipeline.py — the correctness-critical core of the Deadzone WER testbed.

Three functions here are where silent bugs live. Each one produces clean-looking
GARBAGE if you get it subtly wrong, with no error message. They are implemented
carefully and unit-tested in test_pipeline.py.

    1. mix_at_snr()      -- SNR computed on ACTIVE SPEECH energy, not whole-file
    2. apply_rir()       -- convolution with direct-path delay + level compensation
    3. classify_errors() -- aligned edit ops (sub/del/ins), not just a scalar WER

Everything downstream (confidence-gap map, failure fingerprints, predictor)
rides on these three being right. Build and trust this layer FIRST.

Deps: numpy, scipy only. Text alignment is pure-python (cross-check against
`jiwer` if you want, but this returns the typed edits jiwer doesn't hand you).
"""

from __future__ import annotations
import os
import time
import math
import re
import numpy as np
from scipy.signal import fftconvolve


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    """Root-mean-square level of a signal."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def active_speech_mask(speech: np.ndarray, fs: int,
                       frame_ms: float = 20.0,
                       thresh_db: float = -30.0) -> np.ndarray:
    """
    Energy-based voice-activity mask, per sample.

    A frame counts as 'speech' if its energy is within `thresh_db` of the
    loudest frame. This is deliberately simple and dependency-free. For a
    production study swap in webrtcvad or silero-vad -- the CONTRACT (return a
    boolean per-sample mask of active regions) stays identical.

    TRAP THIS AVOIDS: if you measure speech power over the whole file (including
    leading/trailing silence and inter-word gaps), you UNDERSTATE speech power,
    so your requested SNR comes out quieter than asked. Everything calibrates
    off active regions only.
    """
    speech = np.asarray(speech, dtype=np.float64)
    n = len(speech)
    frame = max(1, int(fs * frame_ms / 1000.0))
    # frame energies
    n_frames = int(np.ceil(n / frame))
    padded = np.pad(speech, (0, n_frames * frame - n))
    frames = padded.reshape(n_frames, frame)
    energy = np.mean(frames ** 2, axis=1)
    if np.max(energy) <= 0:
        return np.zeros(n, dtype=bool)
    energy_db = 10.0 * np.log10(energy + 1e-12)
    keep = energy_db >= (np.max(energy_db) + thresh_db)
    mask = np.repeat(keep, frame)[:n]
    return mask


# ----------------------------------------------------------------------------
# TRAP 1 — SNR mixing on active-speech energy
# ----------------------------------------------------------------------------

def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float,
               fs: int, seed: int | None = None) -> np.ndarray:
    """
    Add `noise` to `speech` at a target SNR in dB, measured over active speech.

    Returns speech + scaled_noise, same length as speech.

    TRAPS AVOIDED:
      * SNR is computed on active-speech power (see active_speech_mask), not
        whole-file power. A "10 dB" mix is then actually 10 dB during speech.
      * Noise shorter than speech is tiled; longer noise is randomly cropped so
        you don't always grab the same (possibly silent) head of the file.
    """
    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # match noise length to speech (tile then random-crop)
    if len(noise) < len(speech):
        reps = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, reps)
    if len(noise) > len(speech):
        start = int(rng.integers(0, len(noise) - len(speech) + 1))
        noise = noise[start:start + len(speech)]

    mask = active_speech_mask(speech, fs)
    if not mask.any():          # degenerate: treat whole clip as active
        mask = np.ones(len(speech), dtype=bool)

    speech_power = np.mean(speech[mask] ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power <= 0 or speech_power <= 0:
        return speech.copy()

    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    return speech + scale * noise


def measured_snr_db(clean_speech: np.ndarray, added_noise: np.ndarray,
                    fs: int) -> float:
    """Diagnostic: recover the active-region SNR of a mix, for unit tests."""
    mask = active_speech_mask(clean_speech, fs)
    if not mask.any():
        mask = np.ones(len(clean_speech), dtype=bool)
    sp = np.mean(clean_speech[mask] ** 2)
    npow = np.mean(added_noise ** 2)
    return 10.0 * np.log10(sp / (npow + 1e-12))


# ----------------------------------------------------------------------------
# TRAP 2 — RIR convolution with delay + level compensation
# ----------------------------------------------------------------------------

def apply_rir(speech: np.ndarray, rir: np.ndarray,
              fs: int, preserve_level: bool = True) -> np.ndarray:
    """
    Convolve `speech` with a room impulse response, returning a signal the SAME
    length as the input, time-aligned to the input onset.

    TRAPS AVOIDED:
      * DELAY: a measured RIR has samples before its direct-path peak, so raw
        convolution shifts speech later in time. Downstream WER scoring then
        inherits a pure alignment artifact that looks like model error. We find
        the direct-path peak (argmax|rir|) and trim the leading delay so the
        reverberant output starts where the dry input did.
      * LEVEL: convolution changes signal energy. If you then run SNR mixing,
        your calibrated SNR is silently wrong. We renormalize output RMS back to
        the input's active-speech RMS.
    """
    speech = np.asarray(speech, dtype=np.float64)
    rir = np.asarray(rir, dtype=np.float64)
    if rir.ndim > 1:
        rir = rir[:, 0]                      # take first channel if multichannel

    wet = fftconvolve(speech, rir, mode="full")

    # align: drop samples up to the direct-path peak of the RIR
    direct = int(np.argmax(np.abs(rir)))
    wet = wet[direct:direct + len(speech)]
    if len(wet) < len(speech):               # pad if convolution came up short
        wet = np.pad(wet, (0, len(speech) - len(wet)))

    if preserve_level:
        # Match RMS over the INPUT's active-speech region (wet is onset-aligned
        # to the input, so the same mask applies). Using whole-file RMS instead
        # would be wrong: the reverb tail dumps energy into formerly-silent
        # regions, which would then make the active speech quieter than intended
        # and silently mis-calibrate any downstream SNR mixing.
        mask = active_speech_mask(speech, fs)
        if mask.any():
            ref, cur = rms(speech[mask]), rms(wet[mask])
        else:
            ref, cur = rms(speech), rms(wet)
        if cur > 0:
            wet = wet * (ref / cur)
    return wet


# ----------------------------------------------------------------------------
# TRAP 3 — text normalization + aligned edit classification
# ----------------------------------------------------------------------------

def normalize_text(s: str) -> list[str]:
    """
    Canonicalize a transcript to a token list before scoring.

    TRAP THIS AVOIDS: casing, punctuation, and number formatting swing WER by
    points. Normalize BOTH reference and hypothesis identically or your
    attribution ranks your own formatting bugs, not acoustics. Extend the
    number/symbol map for domain vocab as needed ("2" vs "two", "st" vs "street", ...).
    """
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)           # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()


def classify_errors(reference: str, hypothesis: str) -> dict:
    """
    Word-level alignment (Levenshtein backtrace) returning WER AND the typed
    edit operations -- the foundation of the 'failure fingerprint' analysis.

    Returns dict with:
        wer            float
        n_ref          int  (reference length, denominator)
        counts         {'match','sub','del','ins'} : int
        edits          list of (op, ref_word|None, hyp_word|None)

    'edits' is what lets you say "reverb causes DELETIONS, babble causes
    SUBSTITUTIONS, codec kills PROPER NOUNS" -- a scalar WER cannot.
    """
    r = normalize_text(reference)
    h = normalize_text(hypothesis)
    nr, nh = len(r), len(h)

    # DP edit-distance table (substitution cost 1)
    d = np.zeros((nr + 1, nh + 1), dtype=int)
    d[:, 0] = np.arange(nr + 1)
    d[0, :] = np.arange(nh + 1)
    for i in range(1, nr + 1):
        for j in range(1, nh + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1,          # deletion
                          d[i, j - 1] + 1,          # insertion
                          d[i - 1, j - 1] + cost)   # match/sub

    # backtrace
    i, j = nr, nh
    edits: list[tuple] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i - 1] == h[j - 1] and d[i, j] == d[i - 1, j - 1]:
            edits.append(("match", r[i - 1], h[j - 1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
            edits.append(("sub", r[i - 1], h[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            edits.append(("del", r[i - 1], None)); i -= 1
        else:
            edits.append(("ins", None, h[j - 1])); j -= 1
    edits.reverse()

    counts = {"match": 0, "sub": 0, "del": 0, "ins": 0}
    for op, _, _ in edits:
        counts[op] += 1
    errors = counts["sub"] + counts["del"] + counts["ins"]
    wer = errors / nr if nr else float(nh > 0)
    return {"wer": wer, "n_ref": nr, "counts": counts, "edits": edits}


# ============================================================================
# TRANSCRIPTION ADAPTERS — one shape for every ASR model, so WERs compare.
# ============================================================================
#
# CONTRACT. Every adapter returns exactly these keys, so Nova-3 and Whisper
# outputs are drop-in interchangeable everywhere downstream:
#
#     {
#       "transcript":       str,          # RAW model output (see normalization note)
#       "word_confidences": list[float],  # per-word confidence, in transcript order
#       "mean_conf":        float,        # mean of word_confidences (nan if none)
#       "utterance_conf":   float,        # single utterance-level aggregate signal
#     }
#
# On a persistent Deepgram failure the adapter returns a SENTINEL with the same
# keys but ``transcript is None`` (plus an ``"error"`` string). The grid caller
# checks ``result["transcript"] is None`` and logs+skips that clip instead of
# crashing a several-hundred-clip run. Use ``is_failed()`` for the check.
#
# NORMALIZATION PARITY (why cross-model WER is meaningful).
# ``transcript`` is the model's RAW text. It is never scored as-is: scoring goes
# through classify_errors(), which runs BOTH reference and hypothesis through
# the SAME normalize_text() before aligning. That is the single place text is
# canonicalized, so Deepgram and Whisper are judged on identical rules. Two
# reinforcing guarantees keep them comparable:
#   1. Deepgram formatting is turned OFF (smart_format / punctuate / numerals =
#      False) so its raw output is word-form ("fifty", not "50"), matching how
#      Whisper is scored — normalize_text() strips punctuation/case but does NOT
#      map digits<->words, so a formatted "50" vs "fifty" would survive as a
#      spurious substitution. Killing formatting at the source prevents that.
#   2. Both adapters feed the same classify_errors() path; neither pre-normalizes
#      privately, so no adapter can canonicalize differently from the other.
# ============================================================================


def _empty_confidence_result(transcript=None, error=None) -> dict:
    """The failure sentinel (and shared empty shape). transcript=None => failed."""
    return {
        "transcript": transcript,
        "word_confidences": [],
        "mean_conf": float("nan"),
        "utterance_conf": float("nan"),
        "error": error,
    }


def is_failed(result: dict) -> bool:
    """True if a transcribe adapter returned the skip-me sentinel."""
    return result.get("transcript") is None


# ---------------------------------------------------------------------------
# Deepgram (spine — word confidence is the headline signal)
# ---------------------------------------------------------------------------

def _deepgram_kwargs(model: str) -> dict:
    """
    The exact keyword args handed to the SDK's transcribe_file().

    Factored out so a test can assert, deterministically and offline, that raw
    output is requested. smart_format/punctuate/numerals MUST stay False: we
    score against normalize_text(), which does not reconcile "50" vs "fifty",
    so formatted output would inflate WER as pure formatting noise.
    """
    return {
        "model": model,
        "smart_format": False,
        "punctuate": False,
        "numerals": False,
    }


def _response_to_wire_dict(resp) -> dict:
    """
    Normalize a Deepgram response to the plain wire-JSON dict our parser expects.

    The SDK (v7+) returns a pydantic model; model_dump(by_alias=True) reproduces
    the standard `results.channels[].alternatives[]` JSON. A canned dict (real
    saved response, used in tests) passes straight through. So one pure-dict
    parser serves both the live call and the offline tests.
    """
    if hasattr(resp, "model_dump"):
        return resp.model_dump(by_alias=True, mode="json", exclude_none=True)
    if isinstance(resp, dict):
        return resp
    raise TypeError(f"unexpected Deepgram response type: {type(resp)!r}")


def _parse_deepgram_response(wire: dict) -> dict:
    """Pure: standard Deepgram wire JSON -> the adapter contract dict."""
    alt = wire["results"]["channels"][0]["alternatives"][0]
    words = alt.get("words") or []
    confs = [float(w["confidence"]) for w in words if w.get("confidence") is not None]
    utt = alt.get("confidence")
    return {
        "transcript": alt.get("transcript", ""),
        "word_confidences": confs,
        "mean_conf": float(np.mean(confs)) if confs else float("nan"),
        # utterance-level aggregate: a fallback signal for heavily degraded clips
        # where per-word confidence may be sparse or missing entirely.
        "utterance_conf": float(utt) if utt is not None else float("nan"),
    }


# --- Deepgram SDK path (verified live, don't rediscover) --------------------
# deepgram-sdk 7.6.0 is a FERN-REWRITTEN client — nothing like the 3.x/4.x docs.
# WORKING call path (live-verified against the API):
#     dg = DeepgramClient(api_key=key)
#     resp = dg.listen.v1.media.transcribe_file(request=<bytes>, model="nova-3",
#                                               smart_format=False, ...)
#   * options are KEYWORD ARGS to transcribe_file, not a PrerecordedOptions obj.
#   * resp is a pydantic model; resp.model_dump(by_alias=True) => standard wire
#     JSON (results.channels[].alternatives[].{transcript,confidence,words[]}).
#   * valid model literals include "nova-3" and "nova-2-drivethru" (a real
#     domain-tuned model arm — kept as a swappable literal, not a project focus).
# DEAD — do NOT trust these (they raise ImportError/TypeError on 7.6.0):
#     from deepgram import PrerecordedOptions          # gone
#     dg.listen.prerecorded.v("1").transcribe_file(...) # gone
#     DeepgramClient(key)  # positional arg rejected; use api_key=key
def transcribe_deepgram(audio_path: str, api_key: str | None = None,
                        model: str = "nova-3", max_retries: int = 3,
                        backoff_base: float = 1.0, _transcribe_fn=None) -> dict:
    """
    Transcribe one file with Deepgram Nova-3, returning the adapter contract dict
    (see the TRANSCRIPTION ADAPTERS banner). word_confidences is the headline
    signal for the confidence-gap map.

    Robustness for grid runs:
      * API key comes from the DEEPGRAM_API_KEY env var (or the api_key arg);
        never hardcode it.
      * Up to `max_retries` attempts with exponential backoff
        (backoff_base * 2**attempt seconds) so one flaky call doesn't kill a
        multi-hundred-clip run.
      * On persistent failure returns the sentinel (transcript=None) — check with
        is_failed() and skip — rather than raising.

    Raw output only: smart_format/punctuate/numerals are False (see
    _deepgram_kwargs) so WER isn't polluted by formatting differences vs Whisper.

    `_transcribe_fn` is a test seam: a zero-arg callable returning a response
    (pydantic model or wire dict). Left None, the real SDK call is built.
    """
    api_key = api_key if api_key is not None else os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPGRAM_API_KEY not set (pass api_key= or export the env var)"
        )

    if _transcribe_fn is None:
        def _transcribe_fn():
            from deepgram import DeepgramClient          # lazy import
            dg = DeepgramClient(api_key=api_key)
            with open(audio_path, "rb") as f:
                buf = f.read()
            return dg.listen.v1.media.transcribe_file(
                request=buf, **_deepgram_kwargs(model)
            )

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _transcribe_fn()
            return _parse_deepgram_response(_response_to_wire_dict(resp))
        except Exception as e:          # noqa: BLE001 — grid resilience by design
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff_base * (2 ** attempt))

    return _empty_confidence_result(
        error=f"deepgram failed after {max_retries} attempts: {last_err!r}"
    )


# ---------------------------------------------------------------------------
# Whisper (open baseline — proves the study isn't API-locked)
# ---------------------------------------------------------------------------

def _parse_whisper_result(result: dict) -> dict:
    """
    Pure: an openai-whisper transcribe() result -> the adapter contract dict.

    CONFIDENCE CAVEAT (read before comparing to Deepgram). Whisper exposes no
    calibrated confidence. With word_timestamps=True each word carries a
    `probability` (the decoder's token softmax prob) — we surface those as
    word_confidences. They are NOT on the same scale as Deepgram's acoustic
    confidence, so treat confidence WITHIN a model (Nova-3 vs itself across
    conditions), not as an absolute cross-model number.

    If per-word probabilities are absent (word_timestamps off / unsupported), we
    do NOT fabricate them — word_confidences stays empty and mean_conf falls back
    to the segment-level proxy exp(avg_logprob). utterance_conf is always that
    segment proxy: mean over segments of exp(avg_logprob).
    """
    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []

    word_confs: list[float] = []
    for seg in segments:
        for w in (seg.get("words") or []):
            p = w.get("probability")
            if p is not None:
                word_confs.append(float(p))

    seg_probs = [math.exp(seg["avg_logprob"])
                 for seg in segments if seg.get("avg_logprob") is not None]
    utt = float(np.mean(seg_probs)) if seg_probs else float("nan")

    if word_confs:
        mean_conf = float(np.mean(word_confs))
    else:
        # per-word confidence not exposed -> proxy, don't fake per-word values
        mean_conf = utt

    return {
        "transcript": text,
        "word_confidences": word_confs,
        "mean_conf": mean_conf,
        "utterance_conf": utt,
    }


def transcribe_whisper(audio_path: str, model_name: str = "base",
                       _result=None) -> dict:
    """
    Transcribe one file with local openai-whisper, returning the SAME adapter
    contract dict as transcribe_deepgram (see banner) so the two are directly
    comparable. Read the confidence caveat in _parse_whisper_result.

    Requires `pip install openai-whisper` (heavy: pulls torch); imported lazily
    so the rest of the module stays light. faster-whisper users: adapt the call
    to build the same {"text","segments":[{"avg_logprob","words":[{"probability"}]}]}
    shape and reuse _parse_whisper_result.

    `_result` is a test seam: a canned transcribe() result dict. Left None, the
    model is loaded and run for real.
    """
    if _result is None:
        import whisper                                   # lazy import
        model = whisper.load_model(model_name)
        _result = model.transcribe(audio_path, word_timestamps=True)
    return _parse_whisper_result(_result)


# ---------------------------------------------------------------------------
# Vosk (third arm — offline Kaldi model; a DIFFERENT model FAMILY on purpose)
# ---------------------------------------------------------------------------
# Adding a model = write a `_parse_*` + a `transcribe_*` in this shape, then add
# one line to model_compare.MODEL_REGISTRY. Vosk is CPU-only and exposes per-word
# confidence, so it is a genuine third confidence source — but on its OWN scale,
# NOT comparable in absolute terms to Deepgram's or Whisper's (see model_compare).

def _parse_vosk_result(result: dict) -> dict:
    """
    Pure: a Vosk KaldiRecognizer FinalResult() dict -> the adapter contract dict.

    Vosk (with SetWords(True)) returns
        {"text": "...", "result": [{"word": w, "conf": c, "start":.., "end":..}, ...]}
    We surface each ``conf`` as a word confidence. Vosk has NO separate
    utterance-level aggregate, so utterance_conf reuses the mean of the per-word
    confidences (documented; never fabricated — nan when there are no words).
    """
    words = result.get("result") or []
    confs = [float(w["conf"]) for w in words if w.get("conf") is not None]
    mean_conf = float(np.mean(confs)) if confs else float("nan")
    return {
        "transcript": result.get("text", ""),
        "word_confidences": confs,
        "mean_conf": mean_conf,
        "utterance_conf": mean_conf,     # Vosk exposes no separate utterance conf
    }


def transcribe_vosk(audio_path: str, model_path: str | None = None,
                    _result=None) -> dict:
    """
    Transcribe one file with a local Vosk (Kaldi) model, returning the SAME
    adapter contract dict as the other arms. Offline, CPU-only, dependency-light
    (`pip install vosk`, download a small model). Imported lazily so the module
    stays importable with zero ASR deps.

    Confidence caveat: Vosk word ``conf`` is on its own scale — compare it WITHIN
    Vosk across conditions, never as an absolute number against another family
    (the whole point of model_compare's within-model normalization).

    `_result` is a test seam: a canned FinalResult() dict. Left None, the model is
    loaded and run over the wav's PCM frames for real.
    """
    if _result is None:
        import json                                      # lazy imports
        import wave
        from vosk import Model, KaldiRecognizer
        model = Model(model_path) if model_path else Model(lang="en-us")
        with wave.open(audio_path, "rb") as wf:
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                rec.AcceptWaveform(data)
            _result = json.loads(rec.FinalResult())
    return _parse_vosk_result(_result)
