"""
active_learning.py — Task D3, the third analysis leg (SPEC §5).

Replaces the earlier flat-"predictor" plan with something sharper: instead of
gridding the whole condition space and running the (expensive) ASR oracle in
every cell, we train a cheap Gaussian-process SURROGATE that predicts WER from
the condition parameters and ACTIVELY chooses which condition to evaluate next.
The claim we get to make: *map the failure boundary in far fewer oracle calls
than random/grid sampling* — the map, not the grid.

Decoupling (same discipline as design.py):
  * this module knows only two things — a FACTOR SPACE (design.py) and a callable
    ``oracle(sample: dict) -> float`` that returns WER for one condition. The
    synthetic oracle (tests) and the real pipeline satisfy the SAME signature.
  * it NEVER imports the transcription/acoustic layer at module scope. The
    real-oracle adapter (`make_pipeline_oracle`) lazily imports conditions +
    audio_pipeline only when you actually build a live oracle, so the surrogate
    core stays importable (and testable) with zero audio/API deps.

Encoding is REUSED from design.py, never redefined: candidates are sampled in
SALib's bounds space (categoricals as [0, n_levels)), decoded to the oracle's
sample dict via ``space.decode``. The GP sees those raw rows normalized to the
unit cube by the same bounds (scaling for the kernel, not a new encoding).

Deps: numpy, scipy (qmc), scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from design import FactorSpace, DEFAULT_FACTOR_SPACE

Oracle = Callable[[dict], float]


# ---------------------------------------------------------------------------
# Encoding glue — bounds come from design.py's SALib problem (single source)
# ---------------------------------------------------------------------------

def _bounds(space: FactorSpace) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(space.salib_problem()["bounds"], dtype=float)
    return b[:, 0].copy(), b[:, 1].copy()


def _unit_to_raw(space: FactorSpace, U: np.ndarray) -> np.ndarray:
    """Unit-cube rows -> raw SALib-space rows (design.py bounds)."""
    lo, hi = _bounds(space)
    return lo + np.asarray(U, dtype=float) * (hi - lo)


def _samples_to_dicts(space: FactorSpace, X_raw: np.ndarray) -> list[dict]:
    """Raw rows -> oracle sample dicts, via design.py's decode (categorical floor)."""
    return [space.decode(row) for row in np.atleast_2d(X_raw)]


def lhs_raw(space: FactorSpace, n: int, seed: int) -> np.ndarray:
    """Latin-hypercube seed set in raw space (space-filling initial design)."""
    U = qmc.LatinHypercube(d=len(space), seed=seed).random(n)
    return _unit_to_raw(space, U)


def sobol_pool_raw(space: FactorSpace, n: int, seed: int) -> np.ndarray:
    """A low-discrepancy Sobol candidate pool the acquisition scans over."""
    m = int(np.ceil(np.log2(max(n, 2))))
    U = qmc.Sobol(d=len(space), scramble=True, seed=seed).random_base2(m)
    return _unit_to_raw(space, U[:n])


def random_raw(space: FactorSpace, n: int, seed: int) -> np.ndarray:
    """Uniform-random rows — the honest passive baseline design."""
    lo, hi = _bounds(space)
    U = np.random.default_rng(seed).random((n, len(space)))
    return lo + U * (hi - lo)


# ---------------------------------------------------------------------------
# The surrogate: a small GP regressor over the (normalized) factor space
# ---------------------------------------------------------------------------

class GPSurrogate:
    """
    A lightweight Gaussian-process WER surrogate. ARD Matérn(nu=2.5) kernel over
    the unit-normalized factor vector + a white-noise term (WER measurements are
    noisy). No GPU, no deep learning — fits in milliseconds on tens of points.

        gp = GPSurrogate(space).fit(X_raw, y)
        mean, std = gp.predict(X_raw, return_std=True)
    """

    def __init__(self, space: FactorSpace, seed: int = 0):
        self.space = space
        self.seed = seed
        self._lo, self._hi = _bounds(space)
        self._span = np.where(self._hi > self._lo, self._hi - self._lo, 1.0)
        self.gp: GaussianProcessRegressor | None = None

    def _norm(self, X_raw: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X_raw) - self._lo) / self._span

    def fit(self, X_raw: np.ndarray, y: Sequence[float]) -> "GPSurrogate":
        d = len(self.space)
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2),
                           nu=2.5)
                  + WhiteKernel(1e-3, (1e-6, 1e-1)))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
            random_state=self.seed,
        ).fit(self._norm(X_raw), np.asarray(y, dtype=float))
        return self

    def predict(self, X_raw: np.ndarray, return_std: bool = True):
        assert self.gp is not None, "fit() before predict()"
        return self.gp.predict(self._norm(X_raw), return_std=return_std)

    def log_marginal_likelihood(self) -> float:
        return float(self.gp.log_marginal_likelihood_value_)


# ---------------------------------------------------------------------------
# Acquisition strategies
# ---------------------------------------------------------------------------

# The failure boundary we care about: the WER contour separating "the model can
# handle this condition" from "it can't". 0.5 is a sensible default; a parameter.
DEFAULT_THRESHOLD = 0.5
STRADDLE_KAPPA = 1.96      # Bryan et al. 2005 level-set active learning


def _dedup_mask(cand_raw: np.ndarray, taken_raw: np.ndarray,
                span: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """False for candidates ~identical (in normalized space) to a taken point."""
    if len(taken_raw) == 0:
        return np.ones(len(cand_raw), dtype=bool)
    c = cand_raw / span
    t = taken_raw / span
    d = np.min(np.linalg.norm(c[:, None, :] - t[None, :, :], axis=2), axis=1)
    return d > tol


def acquire(strategy: str, gp: GPSurrogate, cand_raw: np.ndarray,
            taken_raw: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
            kappa: float = STRADDLE_KAPPA) -> int:
    """
    Pick the index (into cand_raw) of the next condition to evaluate.

    strategy='uncertainty'  — pure exploration: the candidate where the GP is
        LEAST sure (max predictive std). Maps the whole surface evenly.
    strategy='boundary'     — level-set / boundary-seeking: the classic STRADDLE
        score  kappa*std - |mean - threshold|.  It is large where the GP predicts
        WER NEAR the failure contour AND is uncertain — i.e. it spends evaluations
        pinning down exactly where the model tips into failure, which is the map
        we actually want.
    Already-evaluated candidates are masked out so the loop can't stall on a point.
    """
    mean, std = gp.predict(cand_raw, return_std=True)
    if strategy == "uncertainty":
        score = std
    elif strategy == "boundary":
        score = kappa * std - np.abs(mean - threshold)
    else:
        raise ValueError(f"unknown strategy {strategy!r} (uncertainty|boundary)")
    ok = _dedup_mask(cand_raw, taken_raw, gp._span)
    score = np.where(ok, score, -np.inf)
    return int(np.argmax(score))


# ---------------------------------------------------------------------------
# The active loop + the passive baseline
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """Full record of a run — everything a plot or the write-up could want."""
    strategy: str
    X_raw: np.ndarray                    # evaluated points, in acquisition order
    y: np.ndarray                        # their oracle WERs
    n_seed: int
    steps: list[dict] = field(default_factory=list)   # per-acquisition log

    @property
    def n_evals(self) -> int:
        return len(self.y)

    def samples(self, space: FactorSpace) -> list[dict]:
        return _samples_to_dicts(space, self.X_raw)


def active_learn(oracle: Oracle, space: FactorSpace, strategy: str = "boundary",
                 n_seed: int = 15, budget: int = 30, pool_size: int = 600,
                 threshold: float = DEFAULT_THRESHOLD, kappa: float = STRADDLE_KAPPA,
                 seed: int = 0) -> Trajectory:
    """
    Seed with an LHS design, then for `budget` steps: fit the GP, acquire ONE
    condition by `strategy`, evaluate it through the oracle, append, repeat.
    Logs the chosen point, its oracle WER, and GP fit quality each step.
    """
    X = lhs_raw(space, n_seed, seed)
    y = np.array([float(oracle(s)) for s in _samples_to_dicts(space, X)])
    pool = sobol_pool_raw(space, pool_size, seed + 1)

    traj = Trajectory(strategy=strategy, X_raw=X, y=y, n_seed=n_seed)
    for step in range(budget):
        gp = GPSurrogate(space, seed=seed).fit(traj.X_raw, traj.y)
        j = acquire(strategy, gp, pool, traj.X_raw, threshold, kappa)
        x_new = pool[j]
        y_new = float(oracle(space.decode(x_new)))
        traj.X_raw = np.vstack([traj.X_raw, x_new])
        traj.y = np.append(traj.y, y_new)
        mean_j, std_j = (float(v[0]) for v in gp.predict([x_new]))
        traj.steps.append({
            "step": step, "n_evals": len(traj.y),
            "chosen": space.decode(x_new), "oracle_wer": y_new,
            "gp_pred": mean_j, "gp_std": std_j,
            "log_marginal_likelihood": gp.log_marginal_likelihood(),
        })
    return traj


def random_baseline(oracle: Oracle, space: FactorSpace, n_total: int,
                    seed: int = 0) -> Trajectory:
    """Passive baseline: `n_total` uniform-random evaluations, no acquisition."""
    X = random_raw(space, n_total, seed)
    y = np.array([float(oracle(s)) for s in _samples_to_dicts(space, X)])
    return Trajectory(strategy="random", X_raw=X, y=y, n_seed=n_total)


# ---------------------------------------------------------------------------
# The comparison that IS the finding — boundary fidelity vs oracle-call count
# ---------------------------------------------------------------------------

def make_test_set(oracle: Oracle, space: FactorSpace, n: int = 512,
                  seed: int = 999) -> tuple[np.ndarray, np.ndarray]:
    """A fixed held-out Sobol set with TRUE oracle WERs — the yardstick both arms
    are scored against (never used for training)."""
    Xt = sobol_pool_raw(space, n, seed)
    yt = np.array([float(oracle(s)) for s in _samples_to_dicts(space, Xt)])
    return Xt, yt


# Half-width of the band around the failure contour used by the HEADLINE metric.
DEFAULT_BAND = 0.15


def boundary_error(gp: GPSurrogate, X_test: np.ndarray, y_true: np.ndarray,
                   threshold: float = DEFAULT_THRESHOLD) -> float:
    """
    Boundary-reconstruction error (classification view): the fraction of held-out
    points the surrogate puts on the WRONG side of the failure contour. Intuitive
    for the write-up ("% of conditions mislabeled safe/unsafe"), but discrete and
    high-variance at low error — use boundary_rmse() as the quantitative headline.
    """
    mean, _ = gp.predict(X_test, return_std=True)
    return float(np.mean((mean >= threshold) != (y_true >= threshold)))


def boundary_rmse(gp: GPSurrogate, X_test: np.ndarray, y_true: np.ndarray,
                  threshold: float = DEFAULT_THRESHOLD,
                  band: float = DEFAULT_BAND) -> float:
    """
    HEADLINE fidelity metric: surrogate RMSE restricted to the held-out points
    NEAR the true failure contour (|true WER - threshold| <= band). This is where
    "did you find the boundary?" actually lives — active sampling concentrates
    evaluations here, random scatters them across the whole cube and mostly
    misses. Continuous (low-variance), so the active-vs-random gap is stable.
    """
    keep = np.abs(y_true - threshold) <= band
    if not np.any(keep):
        return float("nan")
    mean, _ = gp.predict(X_test[keep], return_std=True)
    return float(np.sqrt(np.mean((mean - y_true[keep]) ** 2)))


def surrogate_rmse(gp: GPSurrogate, X_test: np.ndarray, y_true: np.ndarray) -> float:
    """Global surrogate RMSE over the whole space (context, not the headline —
    the boundary strategy deliberately neglects the bulk, so it can trail random
    here even while winning on the boundary)."""
    mean, _ = gp.predict(X_test, return_std=True)
    return float(np.sqrt(np.mean((mean - y_true) ** 2)))


def learning_curve(traj: Trajectory, space: FactorSpace,
                   X_test: np.ndarray, y_true: np.ndarray,
                   checkpoints: Sequence[int] | None = None,
                   threshold: float = DEFAULT_THRESHOLD,
                   band: float = DEFAULT_BAND, seed: int = 0) -> list[dict]:
    """
    Refit the surrogate on the first-k evaluated points (in acquisition order) for
    each checkpoint k, and score the fidelity metrics on the held-out set. Yields
    the fidelity-vs-oracle-calls curve the arms are compared on. `boundary_rmse`
    is the headline; `boundary_error` and `rmse` are reported for context.
    """
    n = traj.n_evals
    if checkpoints is None:
        start = max(traj.n_seed, 3)
        checkpoints = list(range(start, n + 1, 3))
        if checkpoints[-1] != n:
            checkpoints.append(n)
    curve = []
    for k in checkpoints:
        gp = GPSurrogate(space, seed=seed).fit(traj.X_raw[:k], traj.y[:k])
        curve.append({
            "n_evals": int(k),
            "boundary_rmse": boundary_rmse(gp, X_test, y_true, threshold, band),
            "boundary_error": boundary_error(gp, X_test, y_true, threshold),
            "rmse": surrogate_rmse(gp, X_test, y_true),
        })
    return curve


def best_so_far(curve: Sequence[dict], metric: str = "boundary_rmse") -> list[dict]:
    """
    Monotone view of a learning curve: replace each point's `metric` with the best
    (lowest) value achieved up to that oracle-call count. Held-out error bounces
    as points are added; the cumulative min is what makes "oracle calls to reach
    fidelity X" a well-defined, robust comparison between arms.
    """
    out, best = [], float("inf")
    for pt in curve:
        best = min(best, pt[metric])
        out.append({"n_evals": pt["n_evals"], metric: best})
    return out


def evals_to_target(curve: Sequence[dict], target: float,
                    metric: str = "boundary_rmse") -> float:
    """First oracle-call count at which the curve reaches `target` (<=); inf if
    it never does within the run. Pass a best_so_far() curve for a monotone race."""
    for pt in curve:
        if pt[metric] <= target:
            return float(pt["n_evals"])
    return float("inf")


def compare_arms(oracle: Oracle, space: FactorSpace, *, n_seed: int = 15,
                 budget: int = 30, pool_size: int = 600,
                 threshold: float = DEFAULT_THRESHOLD, band: float = DEFAULT_BAND,
                 seed: int = 0, test_n: int = 2048) -> dict:
    """
    Run three arms to the SAME oracle-call budget — active(boundary),
    active(uncertainty), random — and return their learning curves against one
    shared held-out test set. The headline: active reaches a target boundary
    fidelity in far fewer oracle calls than random.
    """
    X_test, y_true = make_test_set(oracle, space, n=test_n, seed=seed + 777)
    n_total = n_seed + budget

    arms: dict[str, Trajectory] = {
        "active_boundary": active_learn(oracle, space, "boundary", n_seed, budget,
                                        pool_size, threshold, seed=seed),
        "active_uncertainty": active_learn(oracle, space, "uncertainty", n_seed,
                                           budget, pool_size, threshold, seed=seed),
        "random": random_baseline(oracle, space, n_total, seed=seed),
    }
    # ONE shared checkpoint schedule so the arms race on identical oracle-call
    # counts (otherwise the passive arm would only be scored at full budget and
    # the comparison would be unfair to it).
    checkpoints = list(range(n_seed, n_total + 1, 3))
    if checkpoints[-1] != n_total:
        checkpoints.append(n_total)
    curves = {name: learning_curve(t, space, X_test, y_true, checkpoints=checkpoints,
                                   threshold=threshold, band=band, seed=seed)
              for name, t in arms.items()}
    return {"arms": arms, "curves": curves, "X_test": X_test, "y_true": y_true,
            "threshold": threshold, "n_total": n_total}


# ---------------------------------------------------------------------------
# Real-oracle adapter — BUILT, not run in tests (same pattern as B2/B3)
# ---------------------------------------------------------------------------

def make_pipeline_oracle(clean_audio, ref_text: str, assets, fs: int = 16000,
                         transcribe_fn: Callable | None = None,
                         model: str = "nova-3"):
    """
    Wrap the real chain  apply_condition -> transcribe -> classify_errors  into an
    ``oracle(sample) -> WER`` so swapping synthetic->real is one line:

        oracle = make_pipeline_oracle(clean_wav, "call maria at four zero five",
                                      asset_library)
        traj   = active_learn(oracle, DEFAULT_FACTOR_SPACE, strategy="boundary")

    Imports the acoustic + transcription layers LAZILY (only when this factory is
    called), so active_learning.py itself stays decoupled and API-free. `assets`
    is a conditions.AssetLibrary; `transcribe_fn` defaults to the Deepgram adapter
    (inject a fake/Whisper for offline use). Not exercised by the test suite — it
    needs real audio + an API key, exactly like the B2/B3 live paths.
    """
    import os
    import tempfile

    import soundfile as sf

    from conditions import Condition, apply_condition
    from audio_pipeline import classify_errors

    if transcribe_fn is None:
        from audio_pipeline import transcribe_deepgram
        transcribe_fn = lambda path: transcribe_deepgram(path, model=model)

    def oracle(sample: dict) -> float:
        condition = Condition.from_sample(sample)
        degraded = apply_condition(np.asarray(clean_audio, dtype=np.float64),
                                   condition, assets, fs)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, f"{condition.name}.wav")
            sf.write(path, degraded, fs)
            result = transcribe_fn(path)
        return float(classify_errors(ref_text, result["transcript"])["wer"])

    return oracle
