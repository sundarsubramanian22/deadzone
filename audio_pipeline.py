"""
audio_pipeline.py — the correctness-critical core of the drive-thru WER testbed.

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
    number map for menu vocab as needed ("2" vs "two", "lg" vs "large", ...).
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


# ----------------------------------------------------------------------------
# transcribe() adapter — NOT unit-tested here (needs your Deepgram key).
# The important bit: pull BOTH the transcript and the per-word CONFIDENCE,
# because the confidence-vs-accuracy danger-zone map is the headline finding.
# ----------------------------------------------------------------------------

def transcribe_deepgram(audio_path: str, api_key: str,
                        model: str = "nova-3") -> dict:
    """
    Returns {'transcript': str, 'word_confidences': [float], 'mean_conf': float}.

    Requires `pip install deepgram-sdk`. Kept out of the tested core so the DSP
    layer runs offline. Swap in Whisper with an equivalent return shape for your
    open baseline.
    """
    from deepgram import DeepgramClient, PrerecordedOptions   # lazy import

    dg = DeepgramClient(api_key)
    with open(audio_path, "rb") as f:
        payload = {"buffer": f.read()}
    opts = PrerecordedOptions(model=model, smart_format=True, punctuate=True)
    resp = dg.listen.prerecorded.v("1").transcribe_file(payload, opts)
    alt = resp["results"]["channels"][0]["alternatives"][0]
    words = alt.get("words", [])
    confs = [w.get("confidence", float("nan")) for w in words]
    return {
        "transcript": alt.get("transcript", ""),
        "word_confidences": confs,
        "mean_conf": float(np.nanmean(confs)) if confs else float("nan"),
    }
