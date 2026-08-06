"""
calibration.py — Layer 2, learned confidence calibration (SPEC §5).

A model's raw confidence is only useful if it means what it says: confidence 0.9
should imply ~90% correct. Under acoustic degradation that breaks — models get
*overconfident*, and (Layer 1's headline) the overconfidence is exactly what makes
a failure silent. This layer learns to CORRECT confidence, conditioned on the
acoustic condition, so a downstream system can trust a calibrated number.

Two calibrators:
  * TemperatureScaler — the standard baseline: one scalar T on the confidence
    logits. Fixes *average* miscalibration but cannot fix miscalibration that
    varies with the condition.
  * FeatureCalibrator — a small feature-conditioned regression calibrator (a
    logistic-regression generalization of Platt scaling) whose inputs are the
    confidence logit PLUS the acoustic condition parameters. It can undo
    miscalibration that grows with a degradation factor — the realistic case.

Metric: Expected Calibration Error (ECE) before vs after, plus reliability-diagram
data (per-bin confidence vs empirical accuracy).

Decoupling: continuous feature columns are read from design.py's factor space
(never redefined) and normalized by its bounds. The module imports numpy / scipy /
sklearn only — no audio, no API.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from deadzone.design import DEFAULT_FACTOR_SPACE, FactorSpace

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Input integrity — the one place a mis-assembled label vector can be caught
# ---------------------------------------------------------------------------
#
# Everything in this module takes (confidence, correctness) as two PARALLEL
# arrays that the caller built by walking an alignment (deadzone/analysis/layers.py's
# `word_records`). Two ways that pairing goes wrong, and neither is visible
# afterwards:
#
#   * LENGTH MISMATCH. numpy will happily broadcast a length-1 array against a
#     length-N one, and `zip`-style truncation upstream binds confidences to the
#     wrong words. Either way a temperature is fitted, an ECE is produced, and
#     the result is a number about a pairing that does not exist.
#   * A NON-FINITE CONFIDENCE. `np.clip(nan, eps, 1-eps)` is still nan, so the
#     NLL is nan for every T, and `minimize_scalar` returns whatever its bounded
#     search happens to land on WITHOUT raising. The reported T is then a
#     property of the optimizer, not of the data.
#
# Both are refused here rather than downstream, because downstream is where they
# stop being detectable.

def _check_pair(conf, correct, *, where: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate a (confidence, correctness) pair and return them as arrays."""
    c = np.asarray(conf, dtype=float).ravel()
    y = np.asarray(correct, dtype=float).ravel()
    if c.size != y.size:
        raise ValueError(
            f"{where}: {c.size} confidences but {y.size} correctness labels. "
            f"These are parallel arrays built by walking one alignment — a "
            f"length mismatch means confidences are bound to the wrong words "
            f"(or numpy will broadcast one of them), and the ECE that comes out "
            f"describes a pairing that never existed. Rebuild the word records "
            f"rather than trimming to the shorter list.")
    if c.size == 0:
        raise ValueError(
            f"{where}: no words to calibrate. An empty input yields ECE nan / a "
            f"default temperature, which reads as 'nothing to fix' rather than "
            f"'nothing was measured'.")
    if not np.all(np.isfinite(c)):
        n = int(np.count_nonzero(~np.isfinite(c)))
        raise ValueError(
            f"{where}: {n} of {c.size} confidences are not finite. NaN survives "
            f"the logit clip, so the fit's objective is nan at every "
            f"temperature and minimize_scalar returns a value WITHOUT raising — "
            f"the reported T would be a property of the optimizer, not of the "
            f"data. A word with no confidence has no place in a calibration "
            f"set; drop it upstream and say how many were dropped.")
    if not np.all(np.isfinite(y)):
        n = int(np.count_nonzero(~np.isfinite(y)))
        raise ValueError(
            f"{where}: {n} of {y.size} correctness labels are not finite. A "
            f"label is 1 (match) or 0 (sub/ins); a NaN there poisons the NLL "
            f"and every bin's empirical accuracy silently.")
    return c, y


# ---------------------------------------------------------------------------
# Metrics: ECE + reliability diagram
# ---------------------------------------------------------------------------

def reliability_curve(conf: Sequence[float], correct: Sequence[float],
                      n_bins: int = 15) -> list[dict]:
    """
    Reliability-diagram data: split [0,1] into `n_bins` equal-width confidence
    bins; per non-empty bin report mean confidence, empirical accuracy, and count.
    A perfectly calibrated model sits on the diagonal (conf == accuracy).
    """
    conf, correct = _check_pair(conf, correct, where="reliability_curve")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf >= lo) & (conf < hi if b < n_bins - 1 else conf <= hi)
        if not np.any(m):
            continue
        out.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "conf_mean": float(conf[m].mean()),
            "accuracy": float(correct[m].mean()),
            "count": int(m.sum()),
        })
    return out


def expected_calibration_error(conf: Sequence[float], correct: Sequence[float],
                               n_bins: int = 15) -> float:
    """
    ECE: the count-weighted average gap |accuracy - confidence| across bins. 0 =
    perfectly calibrated; large = confidence doesn't mean what it says.
    """
    curve = reliability_curve(conf, correct, n_bins)
    n = len(np.asarray(conf))
    if n == 0:
        return float("nan")
    return float(sum(c["count"] / n * abs(c["accuracy"] - c["conf_mean"])
                     for c in curve))


# ---------------------------------------------------------------------------
# Baseline: temperature scaling (single scalar)
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """One-parameter calibration: scale the confidence logits by 1/T, fit to
    minimize negative log-likelihood against the binary correctness labels. T>1
    softens an overconfident model. It is a GLOBAL correction — it cannot depend
    on the acoustic condition (that is what FeatureCalibrator adds)."""

    def __init__(self):
        self.T = 1.0

    def fit(self, conf, correct) -> "TemperatureScaler":
        conf, y = _check_pair(conf, correct, where="TemperatureScaler.fit")
        z = _logit(conf)

        def nll(T):
            p = np.clip(_sigmoid(z / T), _EPS, 1 - _EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        self.T = float(minimize_scalar(nll, bounds=(0.05, 20.0),
                                       method="bounded").x)
        return self

    def transform(self, conf) -> np.ndarray:
        return _sigmoid(_logit(conf) / self.T)


# ---------------------------------------------------------------------------
# Feature-conditioned calibrator
# ---------------------------------------------------------------------------

class FeatureCalibrator:
    """
    Confidence correction conditioned on the acoustic condition. Features = the
    confidence logit + the (bounds-normalized) continuous factor params from
    design.py. A logistic regression fit to the correctness labels; its
    predicted probability IS the calibrated confidence (proper scoring => the
    output is calibrated on the fit distribution). Because rt60/SNR/etc. are
    inputs, it can undo miscalibration that GROWS with a degradation factor —
    which a single temperature cannot.
    """

    def __init__(self, space: FactorSpace = DEFAULT_FACTOR_SPACE):
        self.space = space
        self.cont = [f.name for f in space.factors if f.kind == "continuous"]
        b = np.asarray(space.salib_problem()["bounds"], dtype=float)
        idx = [space.names.index(n) for n in self.cont]
        self._lo = b[idx, 0]
        self._span = np.where(b[idx, 1] > b[idx, 0], b[idx, 1] - b[idx, 0], 1.0)
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=10.0, max_iter=2000),
        )

    def _features(self, rows: Sequence[dict], conf, *, where: str) -> np.ndarray:
        c = np.asarray(conf, dtype=float).ravel()
        # The feature rows and the confidences are a THIRD parallel array: each
        # row is the acoustic condition of the word whose confidence sits at the
        # same index. column_stack would raise on a hard mismatch but says
        # nothing about the cause, and a mismatch introduced upstream (a row
        # list filtered without filtering the confidences) means every word is
        # conditioned on the wrong acoustics while the fit still succeeds.
        if len(rows) != c.size:
            raise ValueError(
                f"{where}: {len(rows)} feature rows but {c.size} confidences. "
                f"They are parallel arrays indexed by hypothesis word — a "
                f"mismatch conditions each word on another word's acoustic "
                f"condition, which still fits and still scores.")
        if c.size == 0:
            raise ValueError(f"{where}: no words to fit.")
        cont = np.array([[float(r[n]) for n in self.cont] for r in rows], dtype=float)
        if not np.all(np.isfinite(cont)):
            n = int(np.count_nonzero(~np.isfinite(cont)))
            raise ValueError(
                f"{where}: {n} non-finite values among the {tuple(self.cont)} "
                f"factor columns. The calibrator is supposed to LEARN the "
                f"condition-dependence of overconfidence; a NaN coordinate "
                f"either blows up inside sklearn with no context or, once "
                f"imputed anywhere upstream, teaches it the wrong condition.")
        cont = (cont - self._lo) / self._span                # bounds-normalized
        return np.column_stack([_logit(c), cont])

    def fit(self, rows: Sequence[dict], conf, correct) -> "FeatureCalibrator":
        c, y = _check_pair(conf, correct, where="FeatureCalibrator.fit")
        X = self._features(rows, c, where="FeatureCalibrator.fit")
        self.model.fit(X, y.astype(int))
        return self

    def transform(self, rows: Sequence[dict], conf) -> np.ndarray:
        X = self._features(rows, conf, where="FeatureCalibrator.transform")
        return self.model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Convenience: before/after report
# ---------------------------------------------------------------------------

def calibration_report(conf_before, conf_after, correct, n_bins: int = 15) -> dict:
    """ECE + reliability data before and after a calibrator — the plot payload."""
    # THREE parallel arrays now. Before/after must describe the SAME words or the
    # "ECE improvement" is a comparison of two different datasets — the single
    # number this layer exists to report.
    n_b = np.asarray(conf_before, dtype=float).size
    n_a = np.asarray(conf_after, dtype=float).size
    if n_b != n_a:
        raise ValueError(
            f"calibration_report: {n_b} confidences before vs {n_a} after. The "
            f"before/after ECE pair is only a calibration result if both "
            f"describe the same words; different lengths make the reported "
            f"improvement a comparison of two different datasets.")
    return {
        "ece_before": expected_calibration_error(conf_before, correct, n_bins),
        "ece_after": expected_calibration_error(conf_after, correct, n_bins),
        "reliability_before": reliability_curve(conf_before, correct, n_bins),
        "reliability_after": reliability_curve(conf_after, correct, n_bins),
    }
