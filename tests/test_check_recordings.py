"""
Tests for the R1.7 recording gate. Fully synthetic and offline — no real clips,
no API — because a gate that silently passes bad audio is worse than no gate:
it launders a broken corpus into every downstream layer.

Each test plants ONE defect and asserts the checker names it.
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
import csv
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.check_recordings import (
    ClipStats, check_corpus, check_manifest_rows,
    evaluate_clip, evaluate_corpus, measure_clip,
)

FS = 48000
rng = np.random.default_rng(0)


def make_clip(pad_head=0.6, pad_tail=0.6, speech_s=2.5, amp=0.30,
              floor=1e-3, fs=FS):
    """A well-formed clip: room tone, a speech-like burst, room tone."""
    head = floor * rng.standard_normal(int(pad_head * fs))
    tail = floor * rng.standard_normal(int(pad_tail * fs))
    t = np.arange(int(speech_s * fs)) / fs
    # a couple of harmonics so it isn't a pure tone; amplitude sets the peak
    burst = amp * (np.sin(2 * np.pi * 180 * t) + 0.4 * np.sin(2 * np.pi * 540 * t))
    burst *= amp / np.max(np.abs(burst))
    return np.concatenate([head, burst, tail])


def stats_for(x, fs=FS, channels=1, subtype="PCM_16", cid="u01"):
    return measure_clip(cid, x, fs, channels, subtype)


# --- measurement is correct before any rule is applied ----------------------
def test_measure_finds_padding_and_floor():
    x = make_clip(pad_head=0.6, pad_tail=0.4)
    s = stats_for(x)
    assert abs(s.lead_s - 0.6) < 0.05, f"lead {s.lead_s}"
    assert abs(s.tail_s - 0.4) < 0.05, f"tail {s.tail_s}"
    assert s.inherent_snr_db > 30, f"synthetic clip should be clean: {s.inherent_snr_db}"
    assert abs(s.peak_dbfs - 20 * np.log10(0.30)) < 0.5
    print("ok: measure_clip recovers padding, floor and peak")


def test_good_clip_passes_clean():
    err, warn = evaluate_clip(stats_for(make_clip()))
    assert not err, f"well-formed clip should pass: {err}"
    assert not warn, f"well-formed clip should not warn: {warn}"
    print("ok: a well-formed clip produces no findings")


# --- each planted defect must be caught -------------------------------------
def test_stereo_is_an_error():
    err, _ = evaluate_clip(stats_for(make_clip(), channels=2))
    assert any("channels" in e for e in err), err
    print("ok: stereo rejected")


def test_lossy_subtype_is_an_error():
    err, _ = evaluate_clip(stats_for(make_clip(), subtype="MPEG_LAYER_III"))
    assert any("PCM" in e for e in err), err
    print("ok: non-PCM source rejected (uncontrolled codec on the codec factor)")


def test_bad_sample_rate_is_an_error():
    err, _ = evaluate_clip(stats_for(make_clip(fs=44100), fs=44100))
    assert any("44100" in e for e in err), err
    print("ok: off-spec sample rate rejected")


def test_clipping_is_an_error():
    x = make_clip(amp=0.30)
    x[int(1.0 * FS):int(1.0 * FS) + 500] = 1.0        # hard clip
    err, _ = evaluate_clip(stats_for(x))
    assert any("CLIPPED" in e for e in err), err
    print("ok: clipping rejected")


def test_missing_tail_padding_is_an_error():
    x = make_clip(pad_tail=0.02)                      # the real u01 failure mode
    err, _ = evaluate_clip(stats_for(x))
    assert any("tail silence" in e for e in err), err
    print("ok: missing tail padding rejected (the fake reverb-deletions trap)")


def test_missing_lead_padding_is_an_error():
    x = make_clip(pad_head=0.02)
    err, _ = evaluate_clip(stats_for(x))
    assert any("lead-in" in e for e in err), err
    print("ok: missing lead-in padding rejected")


def test_duration_out_of_range_is_an_error():
    err, _ = evaluate_clip(stats_for(make_clip(speech_s=0.2, pad_head=0.3, pad_tail=0.3)))
    assert any("duration" in e for e in err), err
    print("ok: absurd duration rejected")


# --- things that should WARN, not fail --------------------------------------
def test_quiet_peak_warns_but_does_not_fail():
    err, warn = evaluate_clip(stats_for(make_clip(amp=0.05)))     # ~-26 dBFS
    assert not err, f"a quiet clip is usable, not fatal: {err}"
    assert any("peak" in w for w in warn), warn
    print("ok: quiet level warns without failing the gate")


def test_low_inherent_snr_warns():
    # Exercise the RULE directly. Driving it through synthetic audio is not
    # possible: active_speech_mask() only separates silence when it sits >30 dB
    # below the loudest frame, so any clip noisy enough to trip a 20 dB SNR
    # warning is also too noisy for the VAD to find a floor at all (see
    # test_noisy_room_defeats_the_vad, which pins that behaviour).
    s = ClipStats(clip_id="u01", fs=FS, channels=1, subtype="PCM_16",
                  n_samples=FS * 4, duration_s=4.0, peak=0.30, peak_dbfs=-10.5,
                  n_near_clip=0, lead_s=0.6, tail_s=0.6,
                  active_dbfs=-25.0, floor_dbfs=-40.0, inherent_snr_db=15.0)
    err, warn = evaluate_clip(s)
    assert not err, err
    assert any("inherent SNR" in w for w in warn), warn
    print("ok: low inherent SNR warns (snr_db axis under-delivers)")


def test_noisy_room_defeats_the_vad():
    """
    A room loud enough to matter doesn't produce a polite SNR warning — the VAD
    marks the WHOLE clip active, so there is no measurable padding and the gate
    fails on lead-in/tail instead. Worth pinning: the failure is loud, which is
    what we want, but it is not the message you'd first expect.
    """
    x = make_clip(amp=0.30, floor=0.05)               # only ~12 dB below speech
    s = stats_for(x)
    assert s.lead_s == 0.0 and s.tail_s == 0.0, (s.lead_s, s.tail_s)
    err, _ = evaluate_clip(s)
    assert any("lead-in" in e for e in err), err
    assert any("tail silence" in e for e in err), err
    print("ok: a too-noisy room fails as missing padding, not as an SNR warning")


# --- corpus-wide consistency -------------------------------------------------
def test_mixed_sample_rates_is_an_error():
    a = stats_for(make_clip(), cid="u01")
    b = stats_for(make_clip(fs=16000), fs=16000, cid="u02")
    err, _ = evaluate_corpus([a, b])
    assert any("MIXED sample rates" in e for e in err), err
    print("ok: a corpus at two rates is rejected")


def test_level_drift_across_session_warns():
    good = [stats_for(make_clip(amp=0.30), cid=f"u{i:02d}") for i in range(1, 5)]
    drifted = stats_for(make_clip(amp=0.03), cid="u05")           # ~-20 dB quieter
    _, warn = evaluate_corpus(good + [drifted])
    assert any("u05" in w and "median" in w for w in warn), warn
    print("ok: level drift across the session warns")


# --- the manifest itself -----------------------------------------------------
def test_unstable_ground_truth_is_an_error():
    bad = [{"id": "u01", "say_this": "Call Maria.", "ground_truth": "Call Maria."}]
    err = check_manifest_rows(bad)
    assert any("normalization-stable" in e for e in err), err
    print("ok: a ground truth that normalize_text() would change is rejected")


def test_real_manifest_is_normalization_stable():
    rows = list(csv.DictReader(open("recording_manifest.csv")))
    err = check_manifest_rows(rows)
    assert not err, f"the shipped manifest must be clean: {err}"
    print(f"ok: all {len(rows)} shipped manifest rows are normalization-stable")


# --- end-to-end over a temp corpus ------------------------------------------
def test_end_to_end_catches_missing_and_stray_files():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "recordings"
        d.mkdir()
        man = Path(td) / "m.csv"
        with open(man, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "say_this", "ground_truth"])
            w.writerow(["u01", "One.", "one"])
            w.writerow(["u02", "Two.", "two"])
        sf.write(d / "u01.wav", make_clip(), FS, subtype="PCM_16")
        sf.write(d / "u99.wav", make_clip(), FS, subtype="PCM_16")   # stray

        rep = check_corpus(str(man), str(d))
        assert any("MISSING" in e and "u02" in e for e in rep.errors), rep.errors
        assert any("stray" in e and "u99" in e for e in rep.errors), rep.errors
        assert not rep.ok
        print("ok: end-to-end catches a missing clip and a mislabelled export")


def test_end_to_end_clean_corpus_passes():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "recordings"
        d.mkdir()
        man = Path(td) / "m.csv"
        with open(man, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "say_this", "ground_truth"])
            for i in (1, 2):
                w.writerow([f"u{i:02d}", "Ok.", "ok"])
        for i in (1, 2):
            sf.write(d / f"u{i:02d}.wav", make_clip(), FS, subtype="PCM_16")
        rep = check_corpus(str(man), str(d))
        assert rep.ok, rep.errors
        assert len(rep.stats) == 2
        print("ok: a clean corpus passes the gate")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} CHECK-RECORDINGS TESTS PASSED")
