"""
Offline, deterministic tests for the transcription adapters (Task B2).

These hit models/APIs in production, so here EVERYTHING is mocked: a canned
Deepgram wire JSON and a canned openai-whisper result stand in for the network.
No key, no torch, no sockets — run the moment you clone:

    python3 test_adapters.py

What we prove:
  * both adapters emit the IDENTICAL contract dict schema (so WERs compare),
  * Deepgram is asked for RAW output (smart_format/punctuate/numerals=False),
    and that those kwargs actually reach the SDK call,
  * a simulated API failure RETRIES then returns the skip-me sentinel,
  * retry recovers if a later attempt succeeds,
  * normalization parity: the two models' raw transcripts canonicalize
    identically at scoring time,
  * Whisper confidence is derived honestly (per-word probs when present; a
    documented segment proxy when not — never faked).
"""
import os
import math
import numpy as np

from audio_pipeline import (
    transcribe_deepgram, transcribe_whisper, is_failed,
    _deepgram_kwargs, _parse_deepgram_response, _parse_whisper_result,
    normalize_text, classify_errors,
)

CONTRACT_KEYS = {"transcript", "word_confidences", "mean_conf", "utterance_conf"}


# --- canned fixtures ---------------------------------------------------------

def canned_deepgram_wire():
    """A real-shaped Deepgram v1 REST response (raw output; numerals off)."""
    return {
        "metadata": {"request_id": "canned", "duration": 2.1},
        "results": {"channels": [{"alternatives": [{
            "transcript": "two cheeseburgers and a large fry",
            "confidence": 0.981,
            "words": [
                {"word": "two",          "start": 0.10, "end": 0.28, "confidence": 0.995},
                {"word": "cheeseburgers","start": 0.28, "end": 0.92, "confidence": 0.951},
                {"word": "and",          "start": 0.92, "end": 1.05, "confidence": 0.889},
                {"word": "a",            "start": 1.05, "end": 1.12, "confidence": 0.972},
                {"word": "large",        "start": 1.12, "end": 1.44, "confidence": 0.934},
                {"word": "fry",          "start": 1.44, "end": 1.80, "confidence": 0.907},
            ],
        }]}]}
    }


def canned_whisper_result():
    """openai-whisper transcribe() result with word_timestamps=True."""
    return {
        "text": " Two cheeseburgers, and a large fry.",   # note DG-vs-Whisper formatting
        "segments": [{
            "avg_logprob": -0.15,
            "words": [
                {"word": " Two",           "probability": 0.98},
                {"word": " cheeseburgers,","probability": 0.93},
                {"word": " and",           "probability": 0.90},
                {"word": " a",             "probability": 0.97},
                {"word": " large",         "probability": 0.95},
                {"word": " fry.",          "probability": 0.91},
            ],
        }],
    }


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
    assert out["transcript"] == "two cheeseburgers and a large fry"
    print("OK 1: raw output requested AND smart_format/punctuate/numerals=False reach the SDK call")


# --- 2: Deepgram parse -> full contract, utterance_conf captured -------------

def test_deepgram_parse_contract():
    out = _parse_deepgram_response(canned_deepgram_wire())
    assert set(out.keys()) == CONTRACT_KEYS, out.keys()
    assert out["transcript"] == "two cheeseburgers and a large fry"
    assert len(out["word_confidences"]) == 6
    assert all(isinstance(c, float) for c in out["word_confidences"])
    assert abs(out["mean_conf"] - float(np.mean([0.995,0.951,0.889,0.972,0.934,0.907]))) < 1e-9
    assert out["utterance_conf"] == 0.981          # top-level alternatives[0].confidence
    print("OK 2: Deepgram parse -> contract dict; utterance_conf captured from alternatives[0].confidence")


# --- 3: Whisper parse -> full contract, per-word probs surfaced --------------

def test_whisper_parse_contract():
    out = _parse_whisper_result(canned_whisper_result())
    assert set(out.keys()) == CONTRACT_KEYS, out.keys()
    assert out["transcript"] == "Two cheeseburgers, and a large fry."
    assert len(out["word_confidences"]) == 6
    assert abs(out["mean_conf"] - float(np.mean([0.98,0.93,0.90,0.97,0.95,0.91]))) < 1e-9
    assert abs(out["utterance_conf"] - math.exp(-0.15)) < 1e-9   # exp(avg_logprob)
    print("OK 3: Whisper parse -> contract dict; per-word probabilities surfaced as word_confidences")


# --- 4: identical schema across adapters (the whole point) -------------------

def test_identical_schema():
    dg = _parse_deepgram_response(canned_deepgram_wire())
    wh = _parse_whisper_result(canned_whisper_result())
    assert set(dg.keys()) == set(wh.keys()) == CONTRACT_KEYS, (dg.keys(), wh.keys())
    for k in CONTRACT_KEYS:
        assert type(dg[k]) is type(wh[k]), (k, type(dg[k]), type(wh[k]))
    print("OK 4: Deepgram and Whisper return byte-for-byte identical dict schema")


# --- 5: Whisper confidence is honest when per-word probs are absent ----------

def test_whisper_no_word_probs_is_not_faked():
    result = {"text": "large fry",
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
    assert out["transcript"] == "two cheeseburgers and a large fry"
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
    # DG "two cheeseburgers and a large fry" vs Whisper "Two cheeseburgers, and a large fry."
    # differ only in case/punctuation -> must canonicalize to the SAME tokens.
    assert normalize_text(dg["transcript"]) == normalize_text(wh["transcript"])

    # and the scoring path proves it: identical WER against the same reference.
    ref = "two cheeseburgers and a large fry"
    assert classify_errors(ref, dg["transcript"])["wer"] == 0.0
    assert classify_errors(ref, wh["transcript"])["wer"] == 0.0
    print("OK 9: raw DG vs Whisper transcripts canonicalize identically -> WERs comparable")


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
    print("\nAll adapter tests passed — Deepgram & Whisper share one contract, offline.")
