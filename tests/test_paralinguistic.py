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
    python3 test_paralinguistic.py
"""
import os
import tempfile

import numpy as np

from paralinguistic import (
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


if __name__ == "__main__":
    test_f0_tracks_known_pitches()
    test_flatness_rises_with_noise()
    test_compare_degradation_rates()
    test_extractor_feeds_analysis()
    test_real_audio_path()
    print("\nAll paralinguistic tests passed — feature extractor recovers planted "
          "acoustic structure and the analysis detects lexical/paralinguistic decoupling.")
