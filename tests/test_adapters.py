"""
Offline, deterministic tests for the transcription adapters (Task B2).

These hit models/APIs in production, so here EVERYTHING is mocked: a canned
Deepgram wire JSON, a canned openai-whisper result and CAPTURED REAL ElevenLabs
Scribe responses stand in for the network. No key, no torch, no sockets — run the
moment you clone:

    python3 tests/test_adapters.py

What we prove:
  * every adapter emits the IDENTICAL contract dict schema (so WERs compare),
  * Deepgram is asked for RAW output (smart_format/punctuate/numerals=False),
    and that those kwargs actually reach the SDK call,
  * a simulated API failure RETRIES then returns the skip-me sentinel,
  * retry recovers if a later attempt succeeds,
  * normalization parity: the two models' raw transcripts canonicalize
    identically at scoring time,
  * Whisper confidence is derived honestly (per-word probs when present; a
    documented segment proxy when not — never faked),
  * Scribe's three traps are handled: only `type == "word"` entries become
    confidences (spacing and audio_event entries carry a logprob too), the
    exp(logprob) transform is clipped off exactly 1.0, and an empty transcript
    is a RESULT, not a failure.

THE SCRIBE FIXTURES ARE VERBATIM CAPTURES, not hand-written shapes — captured
2026-08-05 from scribe_v2 on data/recordings/u02.wav and on the pre-existing
degraded clip results/audio/u02__rt60-1_snr-0_babble_g726_roll-1.wav. That
matters: the exact-0.0 logprob these tests clip, the spacing entries that
duplicate their neighbour's logprob, and the audio_event token that appears under
degradation are all things the vendor really did, not things we imagined it might.
"""

# --- repo-root bootstrap -------------------------------------------------
# Makes `deadzone`, `scripts` and `demos` importable when this file is run
# directly (`python tests/test_pipeline.py`) with no install step. Harmless
# when it is imported as a module instead.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# -------------------------------------------------------------------------
import os
import math
import numpy as np

from deadzone.audio_pipeline import (
    transcribe_deepgram, transcribe_whisper, transcribe_elevenlabs, is_failed,
    _deepgram_kwargs, _parse_deepgram_response, _parse_whisper_result,
    _elevenlabs_form, _parse_elevenlabs_response, _CONF_EPS, ScribeSchemaError,
    normalize_text, classify_errors,
)

CONTRACT_KEYS = {"transcript", "word_confidences", "mean_conf", "utterance_conf"}


# --- canned fixtures ---------------------------------------------------------

def canned_deepgram_wire():
    """A real-shaped Deepgram v1 REST response (raw output; numerals off)."""
    return {
        "metadata": {"request_id": "canned", "duration": 2.1},
        "results": {"channels": [{"alternatives": [{
            "transcript": "call maria at four zero five",
            "confidence": 0.981,
            "words": [
                {"word": "call",  "start": 0.10, "end": 0.28, "confidence": 0.995},
                {"word": "maria", "start": 0.28, "end": 0.92, "confidence": 0.951},
                {"word": "at",    "start": 0.92, "end": 1.05, "confidence": 0.889},
                {"word": "four",  "start": 1.05, "end": 1.12, "confidence": 0.972},
                {"word": "zero",  "start": 1.12, "end": 1.44, "confidence": 0.934},
                {"word": "five",  "start": 1.44, "end": 1.80, "confidence": 0.907},
            ],
        }]}]}
    }


def canned_whisper_result():
    """openai-whisper transcribe() result with word_timestamps=True."""
    return {
        "text": " Call Maria, at four zero five.",   # note DG-vs-Whisper formatting
        "segments": [{
            "avg_logprob": -0.15,
            "words": [
                {"word": " Call",   "probability": 0.98},
                {"word": " Maria,", "probability": 0.93},
                {"word": " at",     "probability": 0.90},
                {"word": " four",   "probability": 0.97},
                {"word": " zero",   "probability": 0.95},
                {"word": " five.",  "probability": 0.91},
            ],
        }],
    }


def canned_scribe_wire():
    """
    VERBATIM scribe_v2 response for data/recordings/u02.wav (2026-08-05).

    Three properties of this capture are load-bearing and none of them were
    invented: `words[]` interleaves 11 `word` entries with 10 `spacing` entries
    and EVERY entry carries a `logprob`; the spacing logprobs duplicate the
    following word's in 9 of 10 cases; and the word "at" came back at logprob
    exactly 0.0, i.e. probability exactly 1.0. Punctuation is glued to the word
    token ("seven."), never a separate entry.
    """
    return {
        "language_code": "eng",
        "language_probability": 0.7252284288406372,
        "text": "Call Maria at four zero five nine one two seven seven.",
        "words": [
            {"text": "Call",   "start": 1.779, "end": 1.959, "type": "word",    "logprob": -9.179073458653875e-06},
            {"text": " ",      "start": 1.959, "end": 2.019, "type": "spacing", "logprob": -5.960446742392378e-06},
            {"text": "Maria",  "start": 2.019, "end": 2.579, "type": "word",    "logprob": -5.960446742392378e-06},
            {"text": " ",      "start": 2.579, "end": 2.679, "type": "spacing", "logprob": 0.0},
            {"text": "at",     "start": 2.679, "end": 2.96,  "type": "word",    "logprob": 0.0},
            {"text": " ",      "start": 2.96,  "end": 3.019, "type": "spacing", "logprob": -0.03022843599319458},
            {"text": "four",   "start": 3.019, "end": 3.319, "type": "word",    "logprob": -0.03022843599319458},
            {"text": " ",      "start": 3.319, "end": 3.339, "type": "spacing", "logprob": -0.0003137096355203539},
            {"text": "zero",   "start": 3.339, "end": 3.699, "type": "word",    "logprob": -0.0003137096355203539},
            {"text": " ",      "start": 3.699, "end": 3.759, "type": "spacing", "logprob": -2.7418097943154862e-06},
            {"text": "five",   "start": 3.759, "end": 4.159, "type": "word",    "logprob": -2.7418097943154862e-06},
            {"text": " ",      "start": 4.159, "end": 4.519, "type": "spacing", "logprob": -0.08042978495359421},
            {"text": "nine",   "start": 4.519, "end": 4.799, "type": "word",    "logprob": -0.08042978495359421},
            {"text": " ",      "start": 4.799, "end": 4.839, "type": "spacing", "logprob": -1.4305104514278355e-06},
            {"text": "one",    "start": 4.839, "end": 5.099, "type": "word",    "logprob": -1.4305104514278355e-06},
            {"text": " ",      "start": 5.099, "end": 5.119, "type": "spacing", "logprob": -1.5497195136049413e-06},
            {"text": "two",    "start": 5.119, "end": 5.359, "type": "word",    "logprob": -1.5497195136049413e-06},
            {"text": " ",      "start": 5.359, "end": 5.839, "type": "spacing", "logprob": -3.6954811548639555e-06},
            {"text": "seven",  "start": 5.839, "end": 6.239, "type": "word",    "logprob": -3.6954811548639555e-06},
            {"text": " ",      "start": 6.239, "end": 6.259, "type": "spacing", "logprob": -4.172316494077677e-06},
            {"text": "seven.", "start": 6.259, "end": 6.659, "type": "word",    "logprob": -0.037572756402444916},
        ],
        "transcription_id": "Opw85nmtQAQoK3VpPlHc",
        "audio_duration_secs": 7.0820625,
    }


def canned_scribe_audio_event():
    """
    VERBATIM scribe_v2 response for the harsh degraded clip
    results/audio/u02__rt60-1_snr-0_babble_g726_roll-1.wav with tagging left at
    the vendor DEFAULT (tag_audio_events on).

    This is the reason _elevenlabs_form turns tagging off: the transcript is the
    literal string "[background noise]", carried by a single `audio_event` entry.
    Scored, that is two insertions of words nobody said, in exactly the harsh
    cells the study cares about.
    """
    return {
        "language_code": "eng",
        "language_probability": 0.5091670751571655,
        "text": "[background noise]",
        "words": [
            {"text": "[background noise]", "start": 0.099, "end": 7.0,
             "type": "audio_event", "logprob": -0.012995828227940365},
        ],
        "transcription_id": "B9DJTu5V2XXyF2kLPC7k",
        "audio_duration_secs": 7.082125,
    }


def canned_scribe_empty():
    """
    VERBATIM scribe_v2 response for the SAME harsh clip with the adapter's own
    form (tag_audio_events=false, language_code=eng): text "" and words [].

    An empty transcript is a RESULT — the mute-zone signal Deepgram also produces
    on its worst conditions — and must never be confused with the failure
    sentinel, which is transcript=None.
    """
    return {
        "language_code": "eng",
        "language_probability": 1.0,
        "text": "",
        "words": [],
        "transcription_id": "je4FX4BNYEce1KqIbl0F",
        "audio_duration_secs": 7.082125,
    }


class FakeResponse:
    """Minimal stand-in for a requests.Response (status/json/text only)."""

    def __init__(self, payload, status_code=200, text=""):
        self._payload, self.status_code, self.text = payload, status_code, text

    def json(self):
        return self._payload


# --- 1: raw output requested, kwargs actually reach the SDK call -------------

def test_deepgram_requests_raw_output():
    kw = _deepgram_kwargs("nova-3")
    assert kw["smart_format"] is False, kw
    assert kw["punctuate"] is False, kw
    assert kw["numerals"] is False, kw
    assert kw["model"] == "nova-3", kw

    # ...and prove those kwargs are actually PASSED through to transcribe_file.
    # Patch the SDK client the adapter imports lazily, capture the call kwargs.
    import deepgram
    captured = {}

    class FakeMedia:
        def transcribe_file(self, request=None, **kwargs):
            captured.update(kwargs)
            captured["request"] = request
            return canned_deepgram_wire()

    class FakeClient:
        def __init__(self, *a, **k):
            self.listen = type("L", (), {"v1": type("V", (), {"media": FakeMedia()})()})()

    orig = deepgram.DeepgramClient
    deepgram.DeepgramClient = FakeClient
    try:
        # audio_path is opened for bytes, so use this very file as any-bytes stand-in
        out = transcribe_deepgram(__file__, api_key="fake-key")
    finally:
        deepgram.DeepgramClient = orig

    assert captured.get("smart_format") is False, captured
    assert captured.get("punctuate") is False, captured
    assert captured.get("numerals") is False, captured
    assert isinstance(captured.get("request"), (bytes, bytearray)), type(captured.get("request"))
    assert out["transcript"] == "call maria at four zero five"
    print("OK 1: raw output requested AND smart_format/punctuate/numerals=False reach the SDK call")


# --- 2: Deepgram parse -> full contract, utterance_conf captured -------------

def test_deepgram_parse_contract():
    out = _parse_deepgram_response(canned_deepgram_wire())
    assert set(out.keys()) == CONTRACT_KEYS, out.keys()
    assert out["transcript"] == "call maria at four zero five"
    assert len(out["word_confidences"]) == 6
    assert all(isinstance(c, float) for c in out["word_confidences"])
    assert abs(out["mean_conf"] - float(np.mean([0.995,0.951,0.889,0.972,0.934,0.907]))) < 1e-9
    assert out["utterance_conf"] == 0.981          # top-level alternatives[0].confidence
    print("OK 2: Deepgram parse -> contract dict; utterance_conf captured from alternatives[0].confidence")


# --- 3: Whisper parse -> full contract, per-word probs surfaced --------------

def test_whisper_parse_contract():
    out = _parse_whisper_result(canned_whisper_result())
    assert set(out.keys()) == CONTRACT_KEYS, out.keys()
    assert out["transcript"] == "Call Maria, at four zero five."
    assert len(out["word_confidences"]) == 6
    assert abs(out["mean_conf"] - float(np.mean([0.98,0.93,0.90,0.97,0.95,0.91]))) < 1e-9
    assert abs(out["utterance_conf"] - math.exp(-0.15)) < 1e-9   # exp(avg_logprob)
    print("OK 3: Whisper parse -> contract dict; per-word probabilities surfaced as word_confidences")


# --- 4: identical schema across adapters (the whole point) -------------------

def test_identical_schema():
    dg = _parse_deepgram_response(canned_deepgram_wire())
    wh = _parse_whisper_result(canned_whisper_result())
    el = _parse_elevenlabs_response(canned_scribe_wire())
    for out in (dg, wh, el):
        assert set(out.keys()) == CONTRACT_KEYS, out.keys()
    for k in CONTRACT_KEYS:
        assert type(dg[k]) is type(wh[k]) is type(el[k]), (
            k, type(dg[k]), type(wh[k]), type(el[k]))
    # language_probability is deliberately NOT among the keys: it is
    # LANGUAGE-DETECTION confidence, and a key named anything confidence-ish
    # sitting next to mean_conf is an invitation to average the wrong thing.
    assert "language_probability" not in el and "language_code" not in el, el.keys()
    print("OK 4: Deepgram, Whisper and Scribe return byte-for-byte identical dict schema")


# --- 5: Whisper confidence is honest when per-word probs are absent ----------

def test_whisper_no_word_probs_is_not_faked():
    result = {"text": "four zero five",
              "segments": [{"avg_logprob": -0.30}]}   # no 'words' -> no per-word probs
    out = _parse_whisper_result(result)
    assert out["word_confidences"] == [], out["word_confidences"]   # NOT fabricated
    assert abs(out["mean_conf"] - math.exp(-0.30)) < 1e-9           # proxy, documented
    assert abs(out["utterance_conf"] - math.exp(-0.30)) < 1e-9
    print("OK 5: Whisper without per-word probs -> empty word_confidences + documented proxy (no faking)")


# --- 6: persistent API failure -> retry then sentinel ------------------------

def test_retry_then_sentinel():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise RuntimeError("simulated 503")

    out = transcribe_deepgram("x.wav", api_key="fake-key", max_retries=3,
                              backoff_base=0.0, _transcribe_fn=always_fails)
    assert calls["n"] == 3, calls                     # retried the full budget
    assert is_failed(out), out                        # sentinel: transcript is None
    assert out["transcript"] is None
    assert out["word_confidences"] == []
    assert math.isnan(out["mean_conf"]) and math.isnan(out["utterance_conf"])
    assert "error" in out and "3 attempts" in out["error"]
    print("OK 6: 3 failing attempts -> skip-me sentinel (transcript=None), caller can log+skip")


# --- 7: retry recovers if a later attempt succeeds ---------------------------

def test_retry_then_success():
    calls = {"n": 0}

    def fail_twice_then_ok():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("simulated transient")
        return canned_deepgram_wire()

    out = transcribe_deepgram("x.wav", api_key="fake-key", max_retries=3,
                              backoff_base=0.0, _transcribe_fn=fail_twice_then_ok)
    assert calls["n"] == 3, calls
    assert not is_failed(out)
    assert out["transcript"] == "call maria at four zero five"
    print("OK 7: two transient failures then success -> real result (retry recovered)")


# --- 8: missing key raises before any network work ---------------------------

def test_missing_key_raises():
    saved = os.environ.pop("DEEPGRAM_API_KEY", None)
    try:
        raised = False
        try:
            transcribe_deepgram("x.wav")   # no api_key, no env
        except ValueError:
            raised = True
        assert raised, "should raise ValueError when no key available"
    finally:
        if saved is not None:
            os.environ["DEEPGRAM_API_KEY"] = saved
    print("OK 8: no DEEPGRAM_API_KEY and no api_key arg -> ValueError (never hardcoded)")


# --- 9: normalization parity -> WERs are actually comparable -----------------

def test_normalization_parity():
    dg = _parse_deepgram_response(canned_deepgram_wire())
    wh = _parse_whisper_result(canned_whisper_result())
    # DG "call maria at four zero five" vs Whisper "Call Maria, at four zero five."
    # differ only in case/punctuation -> must canonicalize to the SAME tokens.
    assert normalize_text(dg["transcript"]) == normalize_text(wh["transcript"])

    # and the scoring path proves it: identical WER against the same reference.
    ref = "call maria at four zero five"
    assert classify_errors(ref, dg["transcript"])["wer"] == 0.0
    assert classify_errors(ref, wh["transcript"])["wer"] == 0.0
    print("OK 9: raw DG vs Whisper transcripts canonicalize identically -> WERs comparable")


# --- 10: Scribe parse -> full contract, exp(logprob) transform ---------------

def test_scribe_parse_contract():
    wire = canned_scribe_wire()
    out = _parse_elevenlabs_response(wire)
    assert set(out.keys()) == CONTRACT_KEYS, out.keys()
    assert out["transcript"] == "Call Maria at four zero five nine one two seven seven."

    # exp() transform, checked against the raw logprobs of the WORD entries only.
    word_lp = [w["logprob"] for w in wire["words"] if w["type"] == "word"]
    expect = [min(max(math.exp(lp), _CONF_EPS), 1.0 - _CONF_EPS) for lp in word_lp]
    assert out["word_confidences"] == expect, out["word_confidences"]
    assert all(isinstance(c, float) for c in out["word_confidences"])
    assert abs(out["mean_conf"] - float(np.mean(expect))) < 1e-12
    # Scribe exposes no separate utterance aggregate -> documented reuse of the
    # per-word mean (the Vosk precedent), never a fabricated second signal and
    # never language_probability.
    assert out["utterance_conf"] == out["mean_conf"]
    print(f"OK 10: Scribe parse -> contract dict; {len(expect)} word confidences "
          f"as exp(logprob), utterance_conf = mean_conf (documented reuse)")


# --- 11: TRAP 1 — spacing / audio_event entries are NOT words ----------------

def test_scribe_excludes_non_word_token_types():
    wire = canned_scribe_wire()
    n_word = sum(1 for w in wire["words"] if w["type"] == "word")
    n_space = sum(1 for w in wire["words"] if w["type"] == "spacing")
    assert n_space == 10 and n_word == 11, (n_word, n_space)
    # every spacing entry really does carry a logprob — this is why filtering is
    # required rather than merely tidy.
    assert all("logprob" in w for w in wire["words"])

    out = _parse_elevenlabs_response(wire)
    assert len(out["word_confidences"]) == n_word
    # ...and 1:1 with the transcript's whitespace tokens, which is what keeps
    # align_confidences able to bind them. Punctuation is glued to the word token
    # ("seven."), so it costs no extra token on either side.
    assert len(out["word_confidences"]) == len(out["transcript"].split())

    # The damage from NOT filtering, measured on the real capture: the spacing
    # logprobs duplicate their neighbour's, so an unfiltered mean double-counts
    # most words. Not noise — a weighting nobody chose, on the headline signal.
    naive = float(np.mean([math.exp(w["logprob"]) for w in wire["words"]]))
    assert naive != out["mean_conf"]
    dup = sum(1 for a, b in zip(wire["words"], wire["words"][1:])
              if a["type"] == "spacing" and b["type"] == "word"
              and a["logprob"] == b["logprob"])
    assert dup == 9, dup

    # audio_event entries are excluded too, for a different reason: they are not
    # speech. The transcript still carries the tag text, which is exactly why the
    # adapter switches tagging off at the request (see test 14).
    ev = _parse_elevenlabs_response(canned_scribe_audio_event())
    assert ev["word_confidences"] == [] and ev["transcript"] == "[background noise]"
    assert math.isnan(ev["mean_conf"]) and not is_failed(ev)
    print(f"OK 11: spacing ({n_space}) and audio_event entries excluded; naive "
          f"all-entry mean {naive:.6f} vs word-only {out['mean_conf']:.6f} "
          f"({dup}/10 spacing logprobs duplicate the next word's)")


# --- 12: TRAP 2 — exp(0.0) is exactly 1.0, and gets clipped ------------------

def test_scribe_clips_confidence_off_exactly_one():
    wire = canned_scribe_wire()
    zeros = [w for w in wire["words"] if w["type"] == "word" and w["logprob"] == 0.0]
    assert zeros, "the capture must contain a real logprob == 0.0 (word 'at')"

    out = _parse_elevenlabs_response(wire)
    assert max(out["word_confidences"]) == 1.0 - _CONF_EPS
    assert all(0.0 < c < 1.0 for c in out["word_confidences"])

    # The clip must match the calibrator's, or a confidence that leaves this
    # adapter "safe" can still be at a boundary the logit treats differently.
    from deadzone.calibration import _EPS as CALIB_EPS
    assert _CONF_EPS == CALIB_EPS, (_CONF_EPS, CALIB_EPS)

    # underflow direction: a hugely negative logprob exp()s to 0.0 with no error
    tiny = _parse_elevenlabs_response(
        {"text": "x", "words": [{"text": "x", "type": "word", "logprob": -1e4}]})
    assert tiny["word_confidences"] == [_CONF_EPS]
    print(f"OK 12: exp(0.0)=1.0 clipped to {1.0 - _CONF_EPS} and exp(-1e4)=0.0 "
          f"clipped to {_CONF_EPS} — matches calibration._EPS, so no logit is inf")


# --- 13: empty transcript is a RESULT; missing words[] is not a crash --------

def test_scribe_empty_and_missing_words():
    empty = _parse_elevenlabs_response(canned_scribe_empty())
    assert empty["transcript"] == ""              # NOT None
    assert not is_failed(empty), "an empty transcript is a mute zone, not a failure"
    assert empty["word_confidences"] == []
    assert math.isnan(empty["mean_conf"]) and math.isnan(empty["utterance_conf"])

    # words[] absent entirely (a shape the vendor's abbreviated docs show)
    nowords = _parse_elevenlabs_response({"text": "call maria at"})
    assert nowords["transcript"] == "call maria at"
    assert nowords["word_confidences"] == []      # NOT fabricated from anywhere
    assert math.isnan(nowords["mean_conf"]) and not is_failed(nowords)

    # a genuine failure is a different object entirely
    dead = transcribe_elevenlabs("x.wav", api_key="fake", max_retries=2,
                                 backoff_base=0.0,
                                 _transcribe_fn=lambda: (_ for _ in ()).throw(
                                     RuntimeError("simulated 503")))
    assert is_failed(dead) and dead["transcript"] is None
    assert "2 attempts" in dead["error"]
    print("OK 13: empty transcript and missing words[] -> real rows (transcript "
          "'' / text kept, conf empty, mean nan); only a failure gives transcript=None")


# --- 14: the request form disables tagging and pins the language -------------

def test_scribe_form_disables_tagging_and_reaches_the_call():
    form = _elevenlabs_form("scribe_v2", "eng")
    assert form["model_id"] == "scribe_v2"
    assert form["tag_audio_events"] == "false"     # no "[background noise]" tokens
    assert form["timestamps_granularity"] == "word"  # not per-CHARACTER confidence
    assert form["diarize"] == "false"
    assert form["language_code"] == "eng"          # no silent language switch
    assert _elevenlabs_form("scribe_v2", None).get("language_code") is None

    # ...and prove those fields actually reach the HTTP call.
    import requests
    captured = {}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        captured.update({"url": url, "data": data, "files": files})
        return FakeResponse(canned_scribe_wire())

    orig = requests.post
    requests.post = fake_post
    try:
        # audio_path is opened for bytes, so use this very file as a stand-in
        out = transcribe_elevenlabs(__file__, api_key="fake-key")
    finally:
        requests.post = orig

    assert captured["data"]["tag_audio_events"] == "false", captured["data"]
    assert captured["data"]["language_code"] == "eng", captured["data"]
    assert captured["data"]["model_id"] == "scribe_v2", captured["data"]
    assert "file" in captured["files"]
    assert out["transcript"].startswith("Call Maria at")
    print("OK 14: tag_audio_events=false / language_code / word granularity reach "
          "the request (no audio-event tokens, no silent language switch)")


# --- 15: schema guards raise instead of going quiet --------------------------

def test_scribe_schema_guards_are_loud():
    # an unrecognized type would otherwise be silently dropped from the headline
    # signal (or silently folded into it) with no error anywhere
    raised = False
    try:
        _parse_elevenlabs_response(
            {"text": "hi", "words": [{"text": "hi", "type": "phoneme", "logprob": -0.1}]})
    except ScribeSchemaError as e:
        raised = "phoneme" in str(e)
    assert raised, "an unknown words[] type must raise, naming the type"

    # a POSITIVE logprob means the field is no longer a log-probability. Clipping
    # it would report a supremely confident model; raising says the scale moved.
    raised = False
    try:
        _parse_elevenlabs_response(
            {"text": "hi", "words": [{"text": "hi", "type": "word", "logprob": 0.9}]})
    except ScribeSchemaError:
        raised = True
    assert raised, "a positive logprob must raise, not clip to ~1.0"

    # and the guard does not fire on the real capture (negative control)
    _parse_elevenlabs_response(canned_scribe_wire())
    print("OK 15: unknown token type and positive logprob both RAISE "
          "(a scale change cannot masquerade as rising confidence)")


# --- 16: no key -> ValueError before any network work ------------------------

def test_scribe_missing_key_raises():
    saved = os.environ.pop("ELEVENLABS_API_KEY", None)
    try:
        raised = False
        try:
            transcribe_elevenlabs("x.wav")
        except ValueError:
            raised = True
        assert raised, "should raise ValueError when no key available"
    finally:
        if saved is not None:
            os.environ["ELEVENLABS_API_KEY"] = saved
    print("OK 16: no ELEVENLABS_API_KEY and no api_key arg -> ValueError (never hardcoded)")


if __name__ == "__main__":
    test_deepgram_requests_raw_output()
    test_deepgram_parse_contract()
    test_whisper_parse_contract()
    test_identical_schema()
    test_whisper_no_word_probs_is_not_faked()
    test_retry_then_sentinel()
    test_retry_then_success()
    test_missing_key_raises()
    test_normalization_parity()
    test_scribe_parse_contract()
    test_scribe_excludes_non_word_token_types()
    test_scribe_clips_confidence_off_exactly_one()
    test_scribe_empty_and_missing_words()
    test_scribe_form_disables_tagging_and_reaches_the_call()
    test_scribe_schema_guards_are_loud()
    test_scribe_missing_key_raises()
    print("\nAll adapter tests passed — Deepgram, Whisper & Scribe share one "
          "contract, offline.")
