"""
Offline validation for Layer 3 (paralinguistic.py). Synthetic audio with planted
structure; assert the extractor + the degradation-rate analysis recover it.

Planted:
  * pure tones at KNOWN pitches -> the f0 extractor must recover them,
  * a clean tone degraded by an increasing noise schedule -> spectral FLATNESS
    must rise monotonically with the noise level (a feature degrading on a known
    schedule),
  * decoupling: when the feature's degradation curve and a lexical (WER) curve
    have different shapes, the analysis must flag DECOUPLING and say which leads;
    when identical, it must call them COUPLED.

Deterministic (fixed seeds), numpy/scipy only, no network. Run:
    python3 tests/test_paralinguistic.py
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
import tempfile

import numpy as np

from deadzone.paralinguistic import (
    extract_features, extract_features_from_path, summary_vector, FEATURE_KEYS,
    feature_drift, compare_degradation_rates,
)

FS = 16000


def tone(freq, dur=0.5, fs=FS, amp=0.6):
    t = np.arange(int(dur * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


# ---- 1: f0 extractor recovers known pitches --------------------------------

def test_f0_tracks_known_pitches():
    for freq in (110.0, 165.0, 220.0):
        feats = extract_features(tone(freq), FS)
        got = feats["summary"]["f0"]
        assert abs(got - freq) / freq < 0.05, (freq, got)
        assert feats["summary"]["voiced_frac"] > 0.8, feats["summary"]["voiced_frac"]
    print("OK 1: autocorr f0 recovers 110/165/220 Hz within 5%")


# ---- 2: a feature degrades on a KNOWN noise schedule -----------------------

def _noise_schedule(seed=0):
    rng = np.random.default_rng(seed)
    clean = tone(200.0)
    noise = rng.standard_normal(len(clean))
    levels = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8])
    clips = [clean + lv * noise for lv in levels]
    return clean, levels, clips


def test_flatness_rises_with_noise():
    clean, levels, clips = _noise_schedule()
    flat = np.array([extract_features(c, FS)["summary"]["flatness"] for c in clips])
    # spectral flatness of a tone is ~0; noise pushes it up, monotonically
    assert np.all(np.diff(flat) > 0), ("flatness should rise monotonically", flat)
    from scipy.stats import spearmanr
    rho = spearmanr(levels, flat).statistic
    assert rho > 0.95, rho
    print(f"OK 2: spectral flatness rises monotonically with noise (rho={rho:.3f}, "
          f"{flat[0]:.3g} -> {flat[-1]:.3g})")


# ---- 3: degradation-rate analysis — coupled vs decoupled -------------------

def test_compare_degradation_rates():
    levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # coupled: identical curves -> together, no gap
    same = compare_degradation_rates(levels, levels, levels)
    assert same["coupled"] and same["max_abs_gap"] < 1e-9, same

    # decoupled: feature degrades early (linear), lexical late (quartic)
    feat = levels.copy()
    lex = levels ** 4
    dec = compare_degradation_rates(levels, feat, lex)
    assert not dec["coupled"], dec
    assert dec["leads"] == "feature", dec
    assert dec["feature_half_level"] < dec["lexical_half_level"], dec
    print(f"OK 3: coupled case flagged coupled; decoupled case -> {dec['note']}")


# ---- 4: extractor + analysis together on planted audio ---------------------

def test_extractor_feeds_analysis():
    clean, levels, clips = _noise_schedule()
    clean_feats = extract_features(clean, FS)
    degraded_feats = [extract_features(c, FS) for c in clips]
    drift = feature_drift(clean_feats, degraded_feats, key="flatness")

    # plant a lexical curve that only breaks at HIGH noise (late) -> should
    # decouple from flatness, which drifts as soon as any noise appears.
    lexical = np.array([0.0, 0.0, 0.02, 0.05, 0.2, 0.6])
    res = compare_degradation_rates(levels, drift, lexical)
    assert not res["coupled"], res
    assert res["leads"] == "feature", res       # flatness moves before WER does
    print(f"OK 4: extracted flatness drift vs late-breaking WER -> {res['note']}")


# ---- 5: real-audio path is wired (lazy soundfile, temp wav) ----------------

def test_real_audio_path():
    x = tone(180.0)
    with tempfile.TemporaryDirectory() as td:
        import soundfile as sf
        path = os.path.join(td, "clip.wav")
        sf.write(path, x, FS)
        feats = extract_features_from_path(path)
    assert set(FEATURE_KEYS) <= set(feats["summary"])
    assert abs(feats["summary"]["f0"] - 180.0) / 180.0 < 0.05
    assert summary_vector(feats).shape == (len(FEATURE_KEYS),)
    print("OK 5: extract_features_from_path round-trips a real wav (f0 recovered)")


# ===========================================================================
# analysis/decoupling.py — the multi-clip L3 runner (SPEC A.R5.9)
# ===========================================================================
# Everything below builds a FAKE SWEEP ON DISK (tiny wavs + an index + a
# transcript cache) with planted structure, and asserts the runner recovers it.
# No network: the transcript cache is pre-seeded, so the API path never runs.

import json  # noqa: E402

from deadzone.analysis.decoupling import (                    # noqa: E402
    FACTOR_DIRECTION, MIN_LEXICAL_RANGE, baseline_path, run_factor_decoupling,
    severity_sort_key, sweep_levels, verdict_sentence, format_report,
    load_transcript_cache, _cache_key, transcribe_sweep,
)

SWEEP_CLIPS_T = ["c1", "c2", "c3", "c4", "c5"]


def _fake_sweep(root, factor, levels, noise_by_severity, wer_by_severity,
                seed=0):
    """Write a synthetic single-factor sweep: 5 clips x len(levels) rungs.

    `noise_by_severity[i]` is the noise amplitude added at severity rung i, so
    the rms/flatness drift curve is PLANTED. `wer_by_severity[i]` is the WER we
    seed the transcript cache with, so the lexical curve is planted too.
    Returns (index, cache).
    """
    import soundfile as sf
    rng = np.random.default_rng(seed)
    d = os.path.join(root, factor)
    os.makedirs(d, exist_ok=True)

    # severity order -> the level that sits at each rung
    ordered = sorted(levels, key=severity_sort_key(factor))

    entries, cache = [], {}
    for clip in SWEEP_CLIPS_T:
        clean = tone(180.0, dur=0.3)
        sf.write(os.path.join(d, f"{clip}__baseline_raw.wav"), clean, FS)
        cache[_cache_key(factor, clip, None)] = {
            "key": _cache_key(factor, clip, None), "failed": False, "wer": 0.0}
        for i, lv in enumerate(ordered):
            y = clean + noise_by_severity[i] * rng.standard_normal(len(clean))
            fn = os.path.join(d, f"{clip}__{factor}_{lv:g}.wav")
            sf.write(fn, y, FS)
            entries.append({"clip_id": clip, "factor": factor,
                            "level": float(lv), "file": fn,
                            "condition_name": f"{factor}-{lv:g}"})
            k = _cache_key(factor, clip, lv)
            cache[k] = {"key": k, "failed": False,
                        "wer": float(wer_by_severity[i])}
    return {factor: entries}, cache


# ---- 6: severity ordering, not numeric ordering (trap 3) -------------------

def test_severity_ordering():
    rt = [0.2, 0.31, 0.43, 1.0]
    sn = [20.0, 14.3, 5.7, 0.0]
    idx_rt = {"rt60": [{"clip_id": "c1", "level": v} for v in rt]}
    idx_sn = {"snr_db": [{"clip_id": "c1", "level": v} for v in sn]}
    # rt60 degrades as the number RISES; snr_db degrades as it FALLS.
    assert sweep_levels(idx_rt, "rt60") == [0.2, 0.31, 0.43, 1.0]
    assert sweep_levels(idx_sn, "snr_db") == [20.0, 14.3, 5.7, 0.0]
    assert FACTOR_DIRECTION["rt60"] == +1 and FACTOR_DIRECTION["snr_db"] == -1
    print("OK 6: severity ordering — rt60 ascends, snr_db descends (20 dB -> 0 dB)")


# ---- 7: planted FEATURE-leads decoupling ----------------------------------

def test_feature_leads_recovered():
    levels = [0.2, 0.31, 0.43, 0.54, 0.66, 0.77, 0.89, 1.0]
    # feature breaks EARLY and saturates; WER holds then breaks LATE.
    noise = [0.0, 0.06, 0.12, 0.17, 0.19, 0.20, 0.20, 0.20]
    wer = [0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.30, 0.60]
    with tempfile.TemporaryDirectory() as td:
        index, cache = _fake_sweep(td, "rt60", levels, noise, wer)
        r = run_factor_decoupling("rt60", index, cache,
                                  feature_keys=("rms",), root=td)
    v = r["per_feature"]["rms"]
    assert not r["lexical_floor"], "WER range 0.6 must NOT be called degenerate"
    assert v["leads"] == "feature", v
    assert v["half_levels_quotable"], v
    assert v["feature_half_level"] < v["lexical_half_level"], v
    assert r["factor_leads"] == "feature", r["factor_leads"]
    assert v["leads_agreement"] == 1.0 and v["n_clips_lexically_flat"] == 0
    print(f"OK 7: planted feature-leads recovered — rms half at "
          f"{v['feature_half_level']:.3f} vs WER half at "
          f"{v['lexical_half_level']:.3f}, agreement "
          f"{v['leads_agreement']:.0%}")


# ---- 8: planted LEXICAL-leads decoupling (the direction flips) ------------

def test_lexical_leads_recovered():
    levels = [0.2, 0.31, 0.43, 0.54, 0.66, 0.77, 0.89, 1.0]
    # WER breaks EARLY; the feature only starts moving late.
    noise = [0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20]
    wer = [0.0, 0.30, 0.50, 0.60, 0.60, 0.60, 0.60, 0.60]
    with tempfile.TemporaryDirectory() as td:
        index, cache = _fake_sweep(td, "rt60", levels, noise, wer, seed=1)
        r = run_factor_decoupling("rt60", index, cache,
                                  feature_keys=("rms",), root=td)
    v = r["per_feature"]["rms"]
    assert not r["lexical_floor"]
    assert v["leads"] == "lexical", v
    assert v["lexical_half_level"] < v["feature_half_level"], v
    assert "would not notice its ASR had already failed" in v["statement"]
    print(f"OK 8: planted lexical-leads recovered — WER half at "
          f"{v['lexical_half_level']:.3f} precedes rms half at "
          f"{v['feature_half_level']:.3f}")


# ---- 9: TRAP 5 — a flat lexical curve is refused a half-level -------------

def test_flat_lexical_curve_is_refused_a_threshold():
    """The trap that actually fired on the real sweep.

    min-max normalization is scale-free, so a WER curve that wanders from 0.000
    to 0.047 normalizes to full range and gets handed a crisp, entirely
    fabricated half-degradation level. The guard must refuse it.
    """
    levels = [20.0, 17.1, 14.3, 11.4, 8.6, 5.7, 2.9, 0.0]
    noise = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]   # clean ladder
    wer = [0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.02, 0.04]         # never leaves floor
    assert max(wer) - min(wer) < MIN_LEXICAL_RANGE
    with tempfile.TemporaryDirectory() as td:
        index, cache = _fake_sweep(td, "snr_db", levels, noise, wer, seed=2)
        r = run_factor_decoupling("snr_db", index, cache,
                                  feature_keys=("rms",), root=td)
    v = r["per_feature"]["rms"]
    assert r["lexical_floor"], "flat WER curve must be flagged degenerate"
    assert r["verdict"].startswith("LEXICAL FLOOR"), r["verdict"]
    assert not v["half_levels_quotable"], "no threshold may be quoted"
    assert not v["stable"]
    # the feature side is genuinely monotone, so it is still reported as signal
    assert v["trend_reliable"], v["monotonicity"]
    assert "no lexical half-degradation level is defined" in v["statement"]
    # and the report marks the un-quotable numbers rather than hiding them
    txt = format_report({"feature_keys": ["rms"], "gap_tol": 0.2,
                         "factors": {"snr_db": r}})
    assert "MEASUREMENT LIMITATION" in txt and "[" in txt
    print(f"OK 9: flat WER curve (range {r['lexical_degeneracy']['range']:.3f}) "
          f"refused a half-level; rms still reported as real signal "
          f"(rho_sev={v['monotonicity']['spearman_vs_severity']:.2f})")


# ---- 10: a clip whose own WER is flat gets NO vote on `leads` -------------

def test_flat_clip_gets_no_vote():
    levels = [0.2, 0.31, 0.43, 0.54, 0.66, 0.77, 0.89, 1.0]
    noise = [0.0, 0.06, 0.12, 0.17, 0.19, 0.20, 0.20, 0.20]
    wer = [0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.30, 0.60]
    with tempfile.TemporaryDirectory() as td:
        index, cache = _fake_sweep(td, "rt60", levels, noise, wer, seed=3)
        # flatten two clips' lexical curves entirely
        for clip in ("c4", "c5"):
            for lv in levels:
                cache[_cache_key("rt60", clip, lv)]["wer"] = 0.0
        r = run_factor_decoupling("rt60", index, cache,
                                  feature_keys=("rms",), root=td)
    v = r["per_feature"]["rms"]
    assert v["n_clips_lexically_flat"] == 2, v["n_clips_lexically_flat"]
    assert sum(v["leads_votes"].values()) == 3, v["leads_votes"]
    flat = [p for p in v["per_clip"] if p["lexical_flat"]]
    assert all("leads" not in p for p in flat), "a flat clip must not vote"
    assert v["per_clip_feature_half_spread"]["n"] == 5   # spread still reported
    print(f"OK 10: 2/5 lexically-flat clips excluded from the leads vote "
          f"({v['leads_votes']}), per-clip spread still reported over all 5")


# ---- 11: snr_db verdict prose is direction-aware --------------------------

def test_verdict_sentence_direction_aware():
    v = {"coupled": False, "leads": "lexical", "max_abs_gap": 0.5,
         "spearman": 0.9, "feature_half_level": 15.3,
         "lexical_half_level": 11.6,
         "per_clip_feature_half_spread": {"n": 5},
         "monotonicity": {"spearman_vs_severity": 0.9, "n_sign_flips": 0},
         "trend_reliable": True, "lexical_degeneracy": {"degenerate": False}}
    s = verdict_sentence("snr_db", "rolloff", v)
    # degradation RISES as SNR falls; saying "increasing snr_db" would invert it
    assert "falling SNR" in s and "increasing" not in s, s
    assert "dB" in s, s
    assert "15.30 dB" in s and "11.60 dB" in s, s
    print("OK 11: snr_db verdict reads 'falling SNR' with half-levels in dB")


# ---- 12: transcript cache round-trips and dedupes baselines ---------------

def test_transcribe_cache_dedupes_and_is_free_on_rerun():
    levels = [0.2, 0.31]
    with tempfile.TemporaryDirectory() as td:
        index, _ = _fake_sweep(td, "rt60", levels, [0.0, 0.1], [0.0, 0.2])
        cache_path = os.path.join(td, "cache.jsonl")
        manifest = {c: "a b c" for c in SWEEP_CLIPS_T}
        calls = []

        def fake_transcribe(path):
            calls.append(path)
            return {"transcript": "a b c", "word_confidences": [0.9, 0.9, 0.9]}

        st = transcribe_sweep(index, manifest, cache_path, workers=2,
                              transcribe_fn=fake_transcribe)
        # 5 clips x 2 levels + 5 baselines (deduped across factor dirs)
        assert st["calls"] == 15, st
        assert st["n_failed"] == 0
        st2 = transcribe_sweep(index, manifest, cache_path, workers=2,
                               transcribe_fn=fake_transcribe)
        assert st2["calls"] == 0, "a re-run must be free"
        assert len(calls) == 15
        assert len(load_transcript_cache(cache_path)) == 15
    print("OK 12: 15 transcriptions cached (baselines deduped); re-run made 0 calls")


# ---- 13: a duplicated index entry cannot silently swap the audio ----------

def test_duplicate_index_entry_raises():
    """
    `clip_curves` keys the index by level, so a SECOND entry for a level already
    present silently REPLACES the first — and every lookup still finds a file,
    so the drift curve comes out the right length, over the right ladder, built
    from whichever wav the index happened to list last. Same shape, same
    verdict machinery, different audio. Two sweeps written into one directory,
    or a regenerated ladder appended rather than replacing, is all it takes.
    """
    levels = [0.2, 0.31, 0.43, 0.54, 0.66, 0.77, 0.89, 1.0]
    noise = [0.0, 0.06, 0.12, 0.17, 0.19, 0.20, 0.20, 0.20]
    wer = [0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.30, 0.60]
    with tempfile.TemporaryDirectory() as td:
        index, cache = _fake_sweep(td, "rt60", levels, noise, wer)
        run_factor_decoupling("rt60", index, cache,          # clean: no false
                              feature_keys=("rms",), root=td)  # positive

        # the SAME clip/level again, pointing at the mildest rung's audio
        dup = dict(index["rt60"][0])
        dup["file"] = index["rt60"][-1]["file"]
        poisoned = {"rt60": index["rt60"] + [dup]}
        try:
            run_factor_decoupling("rt60", poisoned, cache,
                                  feature_keys=("rms",), root=td)
            raise AssertionError("a duplicated sweep entry was accepted")
        except ValueError as exc:
            assert "duplicate sweep entries" in str(exc), exc
            assert "listed last" in str(exc), exc

        # a single-clip sweep makes every reported sd nan while the mean curves
        # still print as measurements
        one = {"rt60": [e for e in index["rt60"] if e["clip_id"] == "c1"]}
        try:
            run_factor_decoupling("rt60", one, cache, feature_keys=("rms",),
                                  root=td)
            raise AssertionError("a single-clip sweep was accepted")
        except ValueError as exc:
            assert "1 clip(s)" in str(exc) and "nan" in str(exc), exc

        # a non-finite WER in the cache would flow into the degeneracy check,
        # the half-level and the verdict as a quiet nan
        bad_cache = dict(cache)
        k = _cache_key("rt60", "c1", levels[0])
        bad_cache[k] = {**cache[k], "wer": float("nan")}
        try:
            run_factor_decoupling("rt60", index, bad_cache,
                                  feature_keys=("rms",), root=td)
            raise AssertionError("a NaN WER was accepted")
        except ValueError as exc:
            assert "not finite" in str(exc), exc
    print("OK 13: a duplicated sweep entry, a single-clip sweep and a NaN WER "
          "all raise instead of producing a right-shaped curve off wrong data")


if __name__ == "__main__":
    test_f0_tracks_known_pitches()
    test_flatness_rises_with_noise()
    test_compare_degradation_rates()
    test_extractor_feeds_analysis()
    test_real_audio_path()
    test_severity_ordering()
    test_feature_leads_recovered()
    test_lexical_leads_recovered()
    test_flat_lexical_curve_is_refused_a_threshold()
    test_flat_clip_gets_no_vote()
    test_verdict_sentence_direction_aware()
    test_transcribe_cache_dedupes_and_is_free_on_rerun()
    test_duplicate_index_entry_raises()
    print("\nAll paralinguistic tests passed — feature extractor recovers planted "
          "acoustic structure, the multi-clip runner recovers the planted leader "
          "in both directions, and a flat lexical curve is refused a "
          "half-degradation level instead of being handed a fabricated one.")
