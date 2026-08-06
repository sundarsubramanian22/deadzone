"""
paralinguistic.py — Layer 3, the parallel acoustic-feature stream (SPEC §5).

Transcription throws away everything that isn't words. But a voice agent also
rides on *paralinguistic* signal — energy, pitch, spectral shape — and the
question this layer opens is: **when acoustic degradation corrupts the audio, do
the paralinguistic features and the lexical accuracy degrade at the same rate, or
do they DECOUPLE?** If a feature collapses long before WER moves (or vice-versa),
that decoupling is itself a finding about what the model is and isn't robust to.

So this runs a lightweight feature extractor ALONGSIDE transcription on the same
audio, and provides the analysis hook that compares the two degradation rates.

Dependency-light on purpose: numpy + scipy only (no librosa, no torch), CPU-only,
fully deterministic. The real-audio entry point lazy-imports soundfile so the
module itself imports with zero audio deps.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Framewise feature extraction
# ---------------------------------------------------------------------------

def _frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _f0_autocorr(frame: np.ndarray, fs: int, fmin=80.0, fmax=400.0,
                 voiced_ratio=0.3) -> float:
    """Autocorrelation pitch for one frame: 0.0 if unvoiced (weak periodicity)."""
    f = frame - frame.mean()
    corr = np.correlate(f, f, mode="full")[len(f) - 1:]
    if corr[0] <= _EPS:
        return 0.0
    lag_min, lag_max = int(fs / fmax), int(fs / fmin)
    seg = corr[lag_min:lag_max]
    if len(seg) == 0:
        return 0.0
    peak = int(np.argmax(seg)) + lag_min
    if corr[peak] < voiced_ratio * corr[0]:
        return 0.0
    return float(fs / peak)


def extract_features(audio: np.ndarray, fs: int, frame_ms: float = 25.0,
                     hop_ms: float = 10.0, rolloff_pct: float = 0.85) -> dict:
    """
    Extract a parallel per-frame acoustic-feature stream:
      rms, spectral centroid / bandwidth / flatness / rolloff, zero-crossing rate,
      and an autocorrelation f0 (+ voiced flag). Returns per-frame arrays plus a
      `summary` of per-feature means (voiced-only for f0). numpy/scipy only.
    """
    x = np.asarray(audio, dtype=np.float64)
    frame = max(8, int(fs * frame_ms / 1000))
    hop = max(1, int(fs * hop_ms / 1000))
    fr = _frames(x, frame, hop)                          # (n_frames, frame)

    rms = np.sqrt(np.mean(fr ** 2, axis=1) + _EPS)
    zcr = np.mean(np.abs(np.diff(np.sign(fr), axis=1)) > 0, axis=1)

    win = np.hanning(frame)
    mag = np.abs(np.fft.rfft(fr * win, axis=1))          # (n_frames, n_bins)
    freqs = np.fft.rfftfreq(frame, 1.0 / fs)
    mag_sum = mag.sum(axis=1) + _EPS

    centroid = (mag * freqs).sum(axis=1) / mag_sum
    bandwidth = np.sqrt((mag * (freqs - centroid[:, None]) ** 2).sum(axis=1) / mag_sum)
    power = mag ** 2
    flatness = (np.exp(np.mean(np.log(power + _EPS), axis=1))
                / (np.mean(power, axis=1) + _EPS))
    cum = np.cumsum(mag, axis=1)
    thresh = rolloff_pct * cum[:, -1:]
    rolloff = freqs[np.argmax(cum >= thresh, axis=1)]

    f0 = np.array([_f0_autocorr(fr[i], fs) for i in range(fr.shape[0])])
    voiced = f0 > 0

    feats = {
        "rms": rms, "zcr": zcr, "centroid": centroid, "bandwidth": bandwidth,
        "flatness": flatness, "rolloff": rolloff, "f0": f0, "voiced": voiced,
    }
    feats["summary"] = {
        "rms": float(rms.mean()), "zcr": float(zcr.mean()),
        "centroid": float(centroid.mean()), "bandwidth": float(bandwidth.mean()),
        "flatness": float(flatness.mean()), "rolloff": float(rolloff.mean()),
        "f0": float(f0[voiced].mean()) if voiced.any() else 0.0,
        "voiced_frac": float(voiced.mean()),
    }
    return feats


def extract_features_from_path(path: str, **kw) -> dict:
    """Real-audio entry point (lazy soundfile import; mono-collapsed)."""
    import soundfile as sf                                # lazy: keep module clean
    x, fs = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return extract_features(x, fs, **kw)


FEATURE_KEYS = ("rms", "zcr", "centroid", "bandwidth", "flatness", "rolloff",
                "f0", "voiced_frac")


def summary_vector(feats: dict) -> np.ndarray:
    """The `summary` dict as an ordered vector (for drift/distance computations)."""
    s = feats["summary"]
    return np.array([s[k] for k in FEATURE_KEYS], dtype=float)


# ---------------------------------------------------------------------------
# Degradation-rate analysis: paralinguistic vs lexical
# ---------------------------------------------------------------------------

def feature_drift(clean_feats: dict, degraded_feats: Sequence[dict],
                  key: str = "flatness") -> np.ndarray:
    """
    Per-degradation-level drift of ONE summary feature from its clean value
    (absolute difference). Give it the clean extraction and the extraction of each
    increasingly-degraded clip; get back the feature's degradation curve.
    """
    base = clean_feats["summary"][key]
    return np.array([abs(d["summary"][key] - base) for d in degraded_feats],
                    dtype=float)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def _half_level(levels: np.ndarray, norm_curve: np.ndarray) -> float:
    """Interpolated level at which a normalized curve first reaches 0.5."""
    for i in range(1, len(norm_curve)):
        if norm_curve[i] >= 0.5:
            x0, x1 = levels[i - 1], levels[i]
            y0, y1 = norm_curve[i - 1], norm_curve[i]
            if y1 == y0:
                return float(x1)
            return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    return float(levels[-1])


def compare_degradation_rates(levels: Sequence[float], feature_curve: Sequence[float],
                              lexical_curve: Sequence[float],
                              gap_tol: float = 0.2) -> dict:
    """
    Do a paralinguistic feature and lexical accuracy (e.g. WER) degrade at the SAME
    rate, or decouple? Each curve is min-max normalized to [0,1] over the swept
    levels; we report:
      * spearman (monotone agreement of ordering),
      * max_abs_gap between the normalized curves,
      * each curve's half-degradation level, and which one leads,
      * coupled: the curves track (small max gap) => degradation hits both together.
    A large gap / large half-level difference = DECOUPLING: one collapses while the
    other holds — the interesting case.
    """
    levels = np.asarray(levels, dtype=float)
    fn, ln = _normalize(feature_curve), _normalize(lexical_curve)
    from scipy.stats import spearmanr
    rho = float(spearmanr(feature_curve, lexical_curve).statistic) \
        if len(levels) >= 3 else float("nan")
    max_gap = float(np.max(np.abs(fn - ln)))
    f_half, l_half = _half_level(levels, fn), _half_level(levels, ln)
    coupled = max_gap <= gap_tol
    lead = ("feature" if f_half < l_half else
            "lexical" if l_half < f_half else "together")
    return {
        "spearman": rho, "max_abs_gap": max_gap, "coupled": bool(coupled),
        "feature_half_level": f_half, "lexical_half_level": l_half,
        "leads": lead,
        "note": ("features and lexical accuracy degrade together"
                 if coupled else
                 f"DECOUPLED: {lead} degrades first "
                 f"(half at {min(f_half, l_half):.3g} vs {max(f_half, l_half):.3g})"),
    }
