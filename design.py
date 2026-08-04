"""
design.py — experiment design + interaction hunt (Task C / SPEC §6, third leg §5).

The instrument's *statistics* layer, deliberately DECOUPLED from acoustics. It
knows nothing about wav files, Deepgram, or the pipeline. It operates on:

  1. a FACTOR SPACE (the knobs and their ranges) — the single source of truth, and
  2. a callable ``wer_fn(sample: dict) -> float`` that scores one factor setting.

``sample`` is a plain dict {factor_name: value} (floats for continuous factors,
level strings for categorical). Today wer_fn is a synthetic planted function
(see test_design.py); later it is a thin wrapper that builds the acoustic
condition and returns the measured WER. design.py never imports the transcription
layer, so this whole file is testable with zero audio.

Pipeline (SPEC §6):
    Stage 1  screen_factors()          cheap Plackett-Burman main-effect screen
    Stage 2  sobol_indices()           Saltelli + Sobol S1 / ST / S2 (interaction hunt)
             find_counterintuitive_cells()   the "huh" cells the write-up needs

Deps: numpy, SALib.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from SALib.sample import sobol as _sobol_sample     # SALib >= 1.4
from SALib.analyze import sobol as _sobol_analyze


# ============================================================================
# FACTOR SPACE — the single source of truth everything else reads
# ============================================================================

@dataclass(frozen=True)
class Factor:
    """
    One experimental knob.

    kind='continuous' -> (low, high) bounds.
    kind='categorical' -> `levels` (a tuple of level names).
    `degradation` (continuous only) records which direction is *worse* audio:
      'up'   -> larger value = more degraded (rt60, mic rolloff)
      'down' -> smaller value = more degraded (snr_db: higher dB is cleaner)
    It is used only for the human-readable narrative in the counterintuitive
    scan; the variance math is direction-agnostic.
    """
    name: str
    kind: str                       # 'continuous' | 'categorical'
    low: float | None = None
    high: float | None = None
    levels: tuple[str, ...] | None = None
    degradation: str | None = None  # 'up' | 'down' (continuous only)

    def __post_init__(self):
        if self.kind == "continuous":
            assert self.low is not None and self.high is not None and self.high > self.low
        elif self.kind == "categorical":
            assert self.levels and len(self.levels) >= 2
        else:
            raise ValueError(f"unknown factor kind: {self.kind!r}")

    @property
    def n_levels(self) -> int:
        return len(self.levels) if self.kind == "categorical" else 0


class FactorSpace:
    """An ordered collection of Factors + the encode/decode glue for SALib."""

    def __init__(self, factors: Sequence[Factor]):
        self.factors: list[Factor] = list(factors)
        self._by_name = {f.name: f for f in self.factors}
        assert len(self._by_name) == len(self.factors), "duplicate factor names"

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.factors]

    def __len__(self) -> int:
        return len(self.factors)

    def subset(self, names: Sequence[str]) -> "FactorSpace":
        """A new space with only `names` (used to hand survivors to Stage 2)."""
        keep = set(names)
        return FactorSpace([f for f in self.factors if f.name in keep])

    def salib_problem(self) -> dict:
        """
        SALib problem dict. Categorical factors are encoded as a continuous
        [0, n_levels] range that decode() floors to a level index — so SALib's
        variance decomposition treats each categorical as one (discretized)
        variable. Its S1/ST are the variance explained by that choice.
        """
        bounds = []
        for f in self.factors:
            if f.kind == "continuous":
                bounds.append([float(f.low), float(f.high)])
            else:
                bounds.append([0.0, float(f.n_levels)])
        return {"num_vars": len(self.factors), "names": self.names, "bounds": bounds}

    def decode(self, x: np.ndarray) -> dict:
        """Raw SALib sample row -> {name: value} the wer_fn understands."""
        out: dict = {}
        for i, f in enumerate(self.factors):
            v = float(x[i])
            if f.kind == "continuous":
                out[f.name] = v
            else:
                idx = min(int(np.floor(v)), f.n_levels - 1)   # clamp x==high edge
                out[f.name] = f.levels[max(idx, 0)]
        return out

    def axis_values(self, factor: Factor, grid: int) -> list:
        """Grid values for the counterintuitive scan (levels for categoricals)."""
        if factor.kind == "continuous":
            return list(np.linspace(factor.low, factor.high, grid))
        return list(factor.levels)


# The Deadzone factor space (SPEC §4/§6): three in-scope acoustic factor families.
DEFAULT_FACTOR_SPACE = FactorSpace([
    Factor("rt60",        "continuous", low=0.2, high=1.0, degradation="up"),   # reverb/placement
    Factor("snr_db",      "continuous", low=0.0, high=25.0, degradation="down"),# noise level
    Factor("noise_type",  "categorical", levels=("babble", "engine", "road")),  # noise character
    Factor("codec",       "categorical", levels=("none", "amr", "opus-lowrate")),# channel
    Factor("mic_rolloff", "continuous", low=0.0, high=1.0, degradation="up"),   # cheap-mic freq response
])


# ============================================================================
# STAGE 1 — Plackett-Burman main-effect screen
# ============================================================================

SCREEN_CAVEAT = (
    "Plackett-Burman is a RESOLUTION-III design: two-factor interactions are "
    "ALIASED onto main effects. A factor with a weak main effect but a strong "
    "interaction can therefore look dead here and be wrongly dropped. So we drop "
    "CONSERVATIVELY — only factors whose |effect| is near-zero relative to the "
    "largest, and never below `min_survivors`. This is a documented limitation "
    "of screening, not a bug: Stage-2 Sobol is what actually resolves interactions."
)

# Classic cyclic Plackett-Burman generating rows (first row; +1/-1).
_PB_GENERATORS = {
    8:  [1, 1, 1, -1, 1, -1, -1],
    12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
    20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],
    24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1],
}


def plackett_burman(n_factors: int) -> np.ndarray:
    """
    A Plackett-Burman ±1 design matrix (runs x n_factors) for a main-effect
    screen. Picks the smallest run count N (multiple of 4) with N-1 >= n_factors,
    builds N-1 rows by cyclically shifting the generating row, appends an
    all-minus row, and returns the first `n_factors` columns.
    """
    N = next((n for n in sorted(_PB_GENERATORS) if n - 1 >= n_factors), None)
    if N is None:
        raise ValueError(f"no PB generator for {n_factors} factors (max {max(_PB_GENERATORS)-1})")
    gen = _PB_GENERATORS[N]
    rows = [np.roll(gen, -i) for i in range(N - 1)]     # cyclic shifts
    rows.append(np.full(N - 1, -1))                     # closing all-low run
    design = np.array(rows, dtype=int)
    return design[:, :n_factors]


def _screen_level_value(f: Factor, sign: int):
    """Map a ±1 screen level to a concrete factor value."""
    if f.kind == "continuous":
        return f.high if sign > 0 else f.low
    # categorical two-level screen: contrast the endpoint levels (documented approx)
    return f.levels[-1] if sign > 0 else f.levels[0]


def screen_factors(space: FactorSpace, wer_fn: Callable[[dict], float],
                   min_survivors: int = 4, drop_frac: float = 0.05) -> dict:
    """
    Stage-1 screen: rank factors by |main effect| and drop only the dead ones.

    A factor's effect is mean(WER | factor high) - mean(WER | factor low) over
    the Plackett-Burman runs. Factors with |effect| < drop_frac * max|effect| are
    candidates to drop, weakest first, but we never go below `min_survivors`, and
    we stop dropping the moment we hit a non-negligible factor. See SCREEN_CAVEAT.

    Returns dict: effects, ranking (strong->weak), survivors, dropped, design,
    n_runs, caveat.
    """
    n = len(space)
    design = plackett_burman(n)
    # evaluate WER at each screening run
    ys = []
    for run in design:
        sample = {f.name: _screen_level_value(f, s) for f, s in zip(space.factors, run)}
        ys.append(float(wer_fn(sample)))
    Y = np.array(ys)

    effects = {}
    for j, f in enumerate(space.factors):
        col = design[:, j]
        effects[f.name] = float(Y[col > 0].mean() - Y[col < 0].mean())

    abseff = {k: abs(v) for k, v in effects.items()}
    max_eff = max(abseff.values()) or 1.0
    ranking = sorted(space.names, key=lambda k: abseff[k], reverse=True)

    survivors = list(ranking)
    for name in sorted(space.names, key=lambda k: abseff[k]):   # weakest first
        if len(survivors) <= min_survivors:
            break
        if abseff[name] < drop_frac * max_eff:
            survivors.remove(name)
        else:
            break                                               # rest are all >= threshold
    survivors = [n for n in ranking if n in survivors]          # keep strong->weak order
    dropped = [n for n in ranking if n not in survivors]

    return {
        "effects": effects,
        "ranking": ranking,
        "survivors": survivors,
        "dropped": dropped,
        "design": design,
        "n_runs": design.shape[0],
        "caveat": SCREEN_CAVEAT,
    }


# ============================================================================
# STAGE 2 — Saltelli sampling + Sobol indices (the interaction hunt)
# ============================================================================

def sobol_indices(space: FactorSpace, wer_fn: Callable[[dict], float],
                  N: int = 1024, seed: int = 0,
                  num_resamples: int = 100) -> dict:
    """
    Variance-based sensitivity on `space`. Saltelli-samples the factor cube
    (calc_second_order=True), evaluates wer_fn, and runs sobol.analyze.

    Interaction signals returned:
      * interaction_gap: per factor, ST - S1 (bootstrap-CI'd). A large gap means
        the factor acts mostly THROUGH interactions, not on its own.
      * s2_ranked: the second-order S2 matrix flattened to ranked (pair, value)
        entries — which SPECIFIC pairs interact.

    N must be a power of 2 (Saltelli requirement). Total model evaluations =
    N * (2*num_vars + 2). Bootstrap CIs come from sobol.analyze's num_resamples.
    """
    problem = space.salib_problem()
    X = _sobol_sample.sample(problem, N, calc_second_order=True, seed=seed)
    Y = np.array([wer_fn(space.decode(row)) for row in X], dtype=float)
    Si = _sobol_analyze.analyze(problem, Y, calc_second_order=True,
                                num_resamples=num_resamples, seed=seed)

    names = space.names
    S1, ST = np.asarray(Si["S1"]), np.asarray(Si["ST"])
    S1c, STc = np.asarray(Si["S1_conf"]), np.asarray(Si["ST_conf"])
    S2, S2c = np.asarray(Si["S2"]), np.asarray(Si["S2_conf"])

    interaction_gap = sorted(
        ({"factor": names[i], "S1": float(S1[i]), "ST": float(ST[i]),
          "gap": float(ST[i] - S1[i]), "S1_conf": float(S1c[i]),
          "ST_conf": float(STc[i])} for i in range(len(names))),
        key=lambda d: d["gap"], reverse=True,
    )

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            val = S2[i, j]
            if np.isnan(val):
                continue
            pairs.append({"pair": (names[i], names[j]), "S2": float(val),
                          "S2_conf": float(S2c[i, j])})
    s2_ranked = sorted(pairs, key=lambda d: d["S2"], reverse=True)

    return {
        "names": names, "S1": S1, "ST": ST, "S1_conf": S1c, "ST_conf": STc,
        "S2": S2, "S2_conf": S2c,
        "interaction_gap": interaction_gap, "s2_ranked": s2_ranked,
        "n_eval": len(Y), "N": N,
    }


def format_sobol_tables(res: dict) -> str:
    """Readable S1 / ST / (ST-S1) table + ranked S2 pairs, with bootstrap CIs."""
    lines = ["First / total order (±95% bootstrap CI), sorted by interaction gap:",
             f"  {'factor':<12} {'S1':>16} {'ST':>16} {'ST-S1':>8}"]
    for d in res["interaction_gap"]:
        lines.append(
            f"  {d['factor']:<12} {d['S1']:>7.3f}±{d['S1_conf']:<7.3f} "
            f"{d['ST']:>7.3f}±{d['ST_conf']:<7.3f} {d['gap']:>8.3f}"
        )
    lines.append("")
    lines.append("Second-order S2 (interacting pairs, strongest first):")
    lines.append(f"  {'pair':<26} {'S2':>16}")
    for d in res["s2_ranked"]:
        a, b = d["pair"]
        lines.append(f"  {a + ' x ' + b:<26} {d['S2']:>7.3f}±{d['S2_conf']:<7.3f}")
    return "\n".join(lines)


# ============================================================================
# COUNTERINTUITIVE-CELL DETECTOR — the "huh" cells for the write-up
# ============================================================================

def _dense_grid(space: FactorSpace, wer_fn, grid: int):
    """Evaluate wer_fn on a dense factor grid. Returns (axes, W ndarray)."""
    axes = [space.axis_values(f, grid) for f in space.factors]
    shape = tuple(len(a) for a in axes)
    W = np.empty(shape, dtype=float)
    for idx in np.ndindex(shape):
        sample = {f.name: axes[d][idx[d]] for d, f in enumerate(space.factors)}
        W[idx] = wer_fn(sample)
    return axes, W


def find_counterintuitive_cells(space: FactorSpace, wer_fn: Callable[[dict], float],
                                grid: int = 11, top_k: int = 10,
                                eps: float = 1e-4) -> dict:
    """
    Scan the response surface for the two "huh" patterns the write-up needs.

    (a) NON-MONOTONICITY: interior local minima of WER along a single continuous
        factor while every other factor is held fixed — i.e. adding degradation
        LOWERS WER (the "noise nudges toward the training distribution / sweet
        spot" effect). Ranked by dip depth.

    (b) EFFECT REVERSALS (sign flips): a continuous factor's marginal effect on
        WER changes SIGN depending on another factor's level (e.g. a channel/mic
        setting that helps in clean audio but compounds with reverb). Ranked by
        the size of the swing.

    Returns dict: {'non_monotonic': [...], 'reversals': [...], 'grid': grid}, each
    entry carrying full factor coordinates.
    """
    axes, W = _dense_grid(space, wer_fn, grid)
    cont = [(d, f) for d, f in enumerate(space.factors) if f.kind == "continuous"]

    # ---- (a) non-monotonic interior local minima along each continuous factor
    non_monotonic = []
    for d, f in cont:
        m = W.shape[d]
        if m < 3:
            continue
        # iterate over every 1-D line along axis d (all other indices fixed)
        other_axes = [range(W.shape[k]) for k in range(W.ndim) if k != d]
        other_dims = [k for k in range(W.ndim) if k != d]
        for combo in itertools.product(*other_axes):
            index = [slice(None)] * W.ndim
            for k, val in zip(other_dims, combo):
                index[k] = val
            profile = W[tuple(index)]                       # 1-D over factor d
            for p in range(1, m - 1):
                if profile[p] < profile[p - 1] - eps and profile[p] < profile[p + 1] - eps:
                    depth = float(min(profile[p - 1], profile[p + 1]) - profile[p])
                    coord = {space.factors[k].name: axes[k][combo[other_dims.index(k)]]
                             for k in other_dims}
                    coord[f.name] = axes[d][p]
                    non_monotonic.append({
                        "factor": f.name,
                        "at_value": float(axes[d][p]),
                        "wer_at_dip": float(profile[p]),
                        "depth": depth,
                        "coords": coord,
                        "note": f"WER dips as {f.name} moves through {axes[d][p]:.3g} "
                                f"(more degradation -> lower WER)",
                    })
    non_monotonic.sort(key=lambda c: c["depth"], reverse=True)

    # ---- (b) effect reversals: sign flip of factor a's effect across factor b
    reversals = []
    for d, a in cont:
        # marginal effect of a = WER at a's last grid end minus its first end
        # (sign-flip detection is symmetric, so the degradation orientation of a
        # doesn't matter here — only whether the sign changes across moderator b)
        a_last, a_first = W.shape[d] - 1, 0
        for e, b in [(k, f) for k, f in enumerate(space.factors) if k != d]:
            b_levels = range(W.shape[e])
            eff_by_b = []
            for bl in b_levels:
                # slice b at level bl, average over all axes except a
                sl = [slice(None)] * W.ndim
                sl[e] = bl
                sub = W[tuple(sl)]
                # axis of a within `sub` (removing e shifts indices > e down by 1)
                a_axis = d if d < e else d - 1
                prof = np.apply_over_axes(
                    np.mean, sub, [ax for ax in range(sub.ndim) if ax != a_axis]
                ).ravel()
                eff = float(prof[a_last] - prof[a_first])
                eff_by_b.append((bl, eff))
            # look for a sign flip between the extreme b levels
            lo_eff = eff_by_b[0][1]
            hi_eff = eff_by_b[-1][1]
            if lo_eff * hi_eff < 0 and abs(lo_eff) > eps and abs(hi_eff) > eps:
                b_lo_val = axes[e][0]
                b_hi_val = axes[e][-1]
                reversals.append({
                    "factor": a.name, "moderator": b.name,
                    "effect_at_low_moderator": lo_eff,
                    "effect_at_high_moderator": hi_eff,
                    "swing": abs(hi_eff - lo_eff),
                    "note": f"effect of {a.name} on WER flips sign as {b.name} "
                            f"goes {b_lo_val} -> {b_hi_val}",
                })
    reversals.sort(key=lambda c: c["swing"], reverse=True)

    return {
        "non_monotonic": non_monotonic[:top_k],
        "reversals": reversals[:top_k],
        "grid": grid,
    }
