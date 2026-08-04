"""
Offline validation for Layer 2 (calibration.py). No audio, no API — a synthetic
population of word outcomes with a KNOWN planted miscalibration stands in for real
model confidences, and we assert the calibrators recover the correction.

Planted structure:
  * true correctness prob per word: true_p = clip(0.97 - 0.55*rt60n - 0.30*(1-snrn))
    (worse with reverb and noise),
  * OVERCONFIDENCE that GROWS with reverb: over = 0.30 * rt60n, so the reported
    confidence raw = clip(true_p + over) — the model is increasingly, and
    condition-dependently, overconfident,
  * binary outcomes y ~ Bernoulli(true_p) with a fixed seed.

Asserted:
  * the raw confidence is meaningfully miscalibrated (ECE well above 0),
  * temperature scaling (global) reduces ECE,
  * the feature-conditioned calibrator reduces ECE substantially MORE (it can see
    rt60), cutting it to a small fraction of the raw ECE,
  * it recovers the KNOWN correction: its per-word correction correlates with the
    planted overconfidence, and its calibrated confidence tracks the true prob far
    better than raw.
All fit on a train split, evaluated on a held-out test split. Deterministic.
Run:  python3 test_calibration.py
"""
import numpy as np
from scipy.stats import spearmanr

from design import DEFAULT_FACTOR_SPACE
from calibration import (
    TemperatureScaler, FeatureCalibrator, expected_calibration_error,
    reliability_curve, calibration_report,
)

SPACE = DEFAULT_FACTOR_SPACE
OVER_MAX = 0.30            # planted overconfidence at max reverb
N = 8000
NOISE = ("babble", "engine", "road")
CODEC = ("none", "amr", "opus-lowrate")


def make_population(seed=0):
    rng = np.random.default_rng(seed)
    rt60 = rng.uniform(0.2, 1.0, N)
    snr = rng.uniform(0.0, 25.0, N)
    rt60n = (rt60 - 0.2) / 0.8
    snrn = snr / 25.0

    true_p = np.clip(0.97 - 0.55 * rt60n - 0.30 * (1 - snrn), 0.02, 0.98)
    over = OVER_MAX * rt60n                                  # grows with reverb
    raw_conf = np.clip(true_p + over, 0.01, 0.999)
    y = (rng.random(N) < true_p).astype(int)                # observed correctness

    rows = [{"rt60": float(rt60[i]), "snr_db": float(snr[i]),
             "noise_type": NOISE[i % 3], "codec": CODEC[i % 3],
             "mic_rolloff": float(rng_val)}
            for i, rng_val in enumerate(rng.uniform(0, 1, N))]
    return rows, raw_conf, y, true_p, over


def split(n, frac=0.5, seed=1):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]


def _take(seq, idx):
    return [seq[i] for i in idx]


# ---- 1: ECE + reliability curve behave sanely ------------------------------

def test_metrics_sane():
    # a perfectly calibrated toy: conf == P(correct) exactly -> ECE ~ 0
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    ece = expected_calibration_error(p, y, n_bins=15)
    assert ece < 0.02, ece
    curve = reliability_curve(p, y, n_bins=15)
    assert len(curve) >= 10 and all(c["count"] > 0 for c in curve)
    # a constant-overconfident toy: conf=0.9 but only 60% correct -> ECE ~ 0.3
    ece_bad = expected_calibration_error(np.full(5000, 0.9),
                                         (rng.random(5000) < 0.6).astype(int))
    assert abs(ece_bad - 0.3) < 0.03, ece_bad
    print(f"OK 1: ECE ~0 when calibrated ({ece:.3f}); ~0.30 when 0.9-conf/0.6-acc "
          f"({ece_bad:.3f}); reliability curve well-formed")


# ---- 2: temperature scaling reduces ECE (global baseline) ------------------

def test_temperature_reduces_ece():
    rows, raw, y, true_p, over = make_population(seed=0)
    tr, te = split(N)
    ece_raw = expected_calibration_error(raw[te], y[te])
    assert ece_raw > 0.05, ("raw should be clearly miscalibrated", ece_raw)

    temp = TemperatureScaler().fit(raw[tr], y[tr])
    ece_temp = expected_calibration_error(temp.transform(raw[te]), y[te])
    assert temp.T > 1.0, ("overconfident model needs T>1 to soften", temp.T)
    assert ece_temp < ece_raw, (ece_temp, ece_raw)
    print(f"OK 2: temp scaling T={temp.T:.2f} cuts ECE {ece_raw:.3f} -> {ece_temp:.3f}")
    return ece_raw, ece_temp


# ---- 3: feature-conditioned calibrator does substantially better -----------

def test_feature_calibrator_beats_temp():
    rows, raw, y, true_p, over = make_population(seed=0)
    tr, te = split(N)
    ece_raw = expected_calibration_error(raw[te], y[te])
    ece_temp = expected_calibration_error(
        TemperatureScaler().fit(raw[tr], y[tr]).transform(raw[te]), y[te])

    fc = FeatureCalibrator(SPACE).fit(_take(rows, tr), raw[tr], y[tr])
    cal_te = fc.transform(_take(rows, te), raw[te])
    ece_feat = expected_calibration_error(cal_te, y[te])

    assert ece_feat < ece_temp, (ece_feat, ece_temp)
    assert ece_feat < 0.5 * ece_raw, ("should cut raw ECE by >half", ece_feat, ece_raw)
    print(f"OK 3: feature calibrator ECE {ece_feat:.3f} < temp {ece_temp:.3f} "
          f"< raw {ece_raw:.3f} (cut to {ece_feat/ece_raw:.0%} of raw)")


# ---- 4: recovers the KNOWN correction --------------------------------------

def test_recovers_planted_correction():
    rows, raw, y, true_p, over = make_population(seed=0)
    tr, te = split(N)
    fc = FeatureCalibrator(SPACE).fit(_take(rows, tr), raw[tr], y[tr])
    cal_te = fc.transform(_take(rows, te), raw[te])

    # the correction it applied should track the planted overconfidence over(rt60)
    correction = raw[te] - cal_te
    rho, _ = spearmanr(correction, over[te])
    assert rho > 0.8, ("correction should track planted overconfidence", rho)

    # and calibrated confidence should track the TRUE prob far better than raw
    rmse_raw = float(np.sqrt(np.mean((raw[te] - true_p[te]) ** 2)))
    rmse_cal = float(np.sqrt(np.mean((cal_te - true_p[te]) ** 2)))
    assert rmse_cal < 0.5 * rmse_raw, (rmse_cal, rmse_raw)
    print(f"OK 4: correction vs planted overconfidence rho={rho:.2f}; "
          f"RMSE-to-true_p {rmse_raw:.3f} (raw) -> {rmse_cal:.3f} (calibrated)")


# ---- 5: report payload has before/after ECE + reliability data -------------

def test_report_payload():
    rows, raw, y, true_p, over = make_population(seed=0)
    tr, te = split(N)
    fc = FeatureCalibrator(SPACE).fit(_take(rows, tr), raw[tr], y[tr])
    rep = calibration_report(raw[te], fc.transform(_take(rows, te), raw[te]), y[te])
    assert rep["ece_after"] < rep["ece_before"]
    assert rep["reliability_before"] and rep["reliability_after"]
    for c in rep["reliability_after"]:
        assert {"bin_lo", "bin_hi", "conf_mean", "accuracy", "count"} <= set(c)
    print(f"OK 5: report ECE {rep['ece_before']:.3f} -> {rep['ece_after']:.3f} "
          f"with reliability-diagram data for both")


if __name__ == "__main__":
    test_metrics_sane()
    test_temperature_reduces_ece()
    test_feature_calibrator_beats_temp()
    test_recovers_planted_correction()
    test_report_payload()
    print("\nAll calibration tests passed — feature-conditioned calibration "
          "recovers the planted condition-dependent overconfidence, ECE and all.")
