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

from design import DEFAULT_FACTOR_SPACE, FactorSpace

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


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
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
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
        z = _logit(conf)
        y = np.asarray(correct, dtype=float)

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

    def _features(self, rows: Sequence[dict], conf) -> np.ndarray:
        cont = np.array([[float(r[n]) for n in self.cont] for r in rows], dtype=float)
        cont = (cont - self._lo) / self._span                # bounds-normalized
        return np.column_stack([_logit(conf), cont])

    def fit(self, rows: Sequence[dict], conf, correct) -> "FeatureCalibrator":
        X = self._features(rows, conf)
        self.model.fit(X, np.asarray(correct, dtype=int))
        return self

    def transform(self, rows: Sequence[dict], conf) -> np.ndarray:
        return self.model.predict_proba(self._features(rows, conf))[:, 1]


# ---------------------------------------------------------------------------
# Convenience: before/after report
# ---------------------------------------------------------------------------

def calibration_report(conf_before, conf_after, correct, n_bins: int = 15) -> dict:
    """ECE + reliability data before and after a calibrator — the plot payload."""
    return {
        "ece_before": expected_calibration_error(conf_before, correct, n_bins),
        "ece_after": expected_calibration_error(conf_after, correct, n_bins),
        "reliability_before": reliability_curve(conf_before, correct, n_bins),
        "reliability_after": reliability_curve(conf_after, correct, n_bins),
    }
