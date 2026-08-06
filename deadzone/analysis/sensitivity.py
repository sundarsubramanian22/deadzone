"""
analysis/sensitivity.py — R4.3 (screen) + R4.5 (Sobol) computed EXACTLY from the
real grid, by functional ANOVA rather than by Saltelli estimation.

WHY THIS FILE EXISTS INSTEAD OF CALLING design.sobol_indices
------------------------------------------------------------
`design.sobol_indices` is a Monte-Carlo estimator: it Saltelli-samples the factor
cube and *estimates* S1/ST/S2 from a finite sample, with all the variance that
implies (SPEC §5 records that S2's bootstrap CI crossed zero even with a strongly
planted interaction at N=1024). That machinery was written before we knew what the
real grid would look like, and it is the right tool when the response surface can
only be probed at scattered points.

It is NOT the right tool here, because of a fact about the data that only became
true after the grid ran:

    176 nova-3 conditions x 40 clips. 144 of those conditions form a COMPLETE
    4 x 4 x 3 x 3 full factorial over rt60 x snr_db x codec x mic_rolloff at
    noise_type='babble', with exactly 40 clips in every single cell and the SAME
    40 clips in every cell.

For a complete factorial with equal cell counts the Sobol/functional-ANOVA
decomposition is **exact, not estimated**. Every term — main effects, all two-way
interactions, all higher-order remainders — is a finite sum over cells that
partitions the total variance with no residual. There is nothing left to sample.

So running Saltelli here would mean fitting a GP surrogate to this same grid and
then Monte-Carlo-sampling the surrogate: strictly weaker evidence than decomposing
the measurements themselves, with two extra error sources (surrogate bias, sampling
noise) bolted on top of data that has neither. We decompose the grid.

The output is deliberately emitted in `design.sobol_indices`' exact return shape
so `analysis.interactions` (the verdict layer) consumes it unmodified.

THE THREE THINGS THAT ARE EASY TO GET WRONG HERE
------------------------------------------------
1. THE INPUT DISTRIBUTION. Sobol indices are not a property of a function alone;
   they are a property of a function *and a distribution over its inputs*. This
   decomposition is with respect to the UNIFORM DISTRIBUTION OVER THE DESIGN
   LEVELS, which is not uniform-on-the-continuous-range. `snr_db`'s levels are
   {0, 5, 10, 20} — unequally spaced — so the implied distribution puts 3/4 of its
   mass below 10 dB. Read against a hypothetical uniform-on-[0,20] prior this
   over-weights the harsh end. That is a statement about what was measured, not a
   defect, but it must be stated (see `INPUT_DISTRIBUTION_CAVEAT`).

2. THE BOOTSTRAP UNIT. The CI must come from resampling CLIPS, not cells. Words
   within a clip are correlated, and — more importantly — the same 40 clips appear
   in all 144 cells, so the cell means are correlated *through the clips*. A cell-
   level bootstrap treats 144 correlated numbers as 144 independent draws and
   understates every interval. We resample the 40 clips with replacement,
   recompute all 144 cell means from the resampled corpus, and redecompose. Every
   replicate is therefore a complete, internally consistent alternate experiment.

3. THE INDEXING. A transposed axis or an off-by-one in the Mobius subtraction
   produces indices that still look plausible — the numbers stay in [0,1] and the
   ranking stays believable. The only cheap check that catches it is that the
   decomposition must SUM TO EXACTLY 1.0, which is asserted on every call
   (`_check_partition`), on every bootstrap replicate's point estimate, and again
   in the tests against planted structure.

CLI:
    python3 -m deadzone.analysis.sensitivity --master results/master.csv \
        --out results/sobol.json --bootstrap 2000

Deps: numpy only (SALib is not needed — that is the point).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from typing import Mapping, Sequence

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.design import DEFAULT_FACTOR_SPACE, SCREEN_CAVEAT, plackett_burman  # noqa: E402


# ============================================================================
# THE CAVEATS THAT TRAVEL WITH THE NUMBERS
# ============================================================================

METHOD = "exact-factorial-anova"

METHOD_NOTE = (
    "Sobol indices computed EXACTLY by functional ANOVA of a complete factorial, "
    "not estimated by Saltelli sampling. With equal cell counts the variance "
    "decomposition is a finite partition of the measured response — main effects, "
    "every two-way interaction and the higher-order remainder sum to the total "
    "variance with zero residual. No surrogate, no Monte-Carlo sampling error."
)

INPUT_DISTRIBUTION_CAVEAT = (
    "Sobol indices are defined relative to a distribution over the inputs. These "
    "are with respect to the UNIFORM distribution over the realised DESIGN LEVELS, "
    "not uniform over the continuous factor ranges. snr_db's levels {0, 5, 10, 20} "
    "are unequally spaced, so the implied prior places 3/4 of its mass at or below "
    "10 dB; rt60's {0.2, 0.45, 0.7, 1.0} are near-uniform on [0.2, 1.0]. Indices "
    "for snr_db are therefore 'share of variance across the conditions we actually "
    "ran', which is the honest quantity for this grid, but they would shift under a "
    "different level allocation."
)

BOOTSTRAP_NOTE = (
    "95% CIs are half-widths from a CLIP-level bootstrap: the 40 recorded clips are "
    "resampled with replacement, all cell means are recomputed from the resampled "
    "corpus, and the full decomposition is redone. Clips are the independent unit — "
    "the same 40 appear in every cell, so resampling cells would treat correlated "
    "numbers as independent and understate every interval. Half-width = 1.96 x the "
    "bootstrap standard deviation, matching SALib's `*_conf` convention so the "
    "result is drop-in compatible with design.sobol_indices' shape; percentile "
    "bounds are supplied alongside as `*_ci_lo` / `*_ci_hi`."
)

# --- the two gap intervals, and why BOTH are published -----------------------
#
# THE DEFECT THIS CONSTANT EXISTS TO PREVENT (found by artifact audit, 2026-08-05):
# `report/writeup.md` §6.3 and the pre-registration blockquote published the
# QUADRATURE interval for ST-S1, while `results/sensitivity_report.txt` printed
# the DIRECT bootstrap interval under the header `gap 95% CI` and
# `results/sobol.json` stored only the direct one. Two different intervals, one
# label, sitting under the headline pre-registration verdict. Both forms clear
# the pre-set 0.020 threshold by >=4.5x so no verdict moved -- but a reader
# diffing the artifacts had no way to know they were looking at two quantities.
#
# So both are computed, both are stored, both are printed, and neither may be
# named `gap_ci_*` without a `_direct` / `_quadrature` suffix.
GAP_CI_NOTE = (
    "TWO 95% intervals are published for the ST-S1 gap and they are NOT "
    "interchangeable. `*_direct` comes from the SAME clip-bootstrap replicates "
    "that produced S1 and ST, so the S1/ST covariance is carried exactly: it is "
    "the CORRECT interval and the tighter one. `*_quadrature` is "
    "sqrt(S1_conf^2 + ST_conf^2) -- the two half-widths combined AS IF S1 and ST "
    "were independent. They are not: on this grid the clip-bootstrap correlation "
    "is strongly POSITIVE (reported per factor as `s1_st_bootstrap_corr`), and "
    "Var(ST-S1) = Var(ST) + Var(S1) - 2*Cov(S1,ST), so dropping the covariance "
    "term inflates the width. The exact identity "
    "`gap_conf_direct^2 == S1_conf^2 + ST_conf^2 - 2*corr*S1_conf*ST_conf` holds "
    "on every row and is asserted in the tests, which is what makes the "
    "over-statement auditable from the artifact rather than merely claimed. "
    "The QUADRATURE form is nonetheless the one the PRE-REGISTERED test uses, "
    "deliberately and for two reasons: (1) analysis/interactions.py consumes a "
    "SALib-shaped result, which exposes per-index half-widths but not the "
    "replicates, so quadrature is the only interval computable there and the "
    "verdict must not depend on which producer filled the file; (2) an over-wide "
    "interval makes CONFIRMED harder to reach, never easier, which is the only "
    "direction we are willing to be wrong in on a registered hypothesis. "
    "DO NOT 'simplify' this by keeping only the tighter interval: that would "
    "silently strengthen a pre-registered test after the fact."
)

# Which interval the pre-registration verdict is read off. Named here so the
# choice is a published fact rather than an implementation detail of whichever
# module happened to format the table.
PREREGISTRATION_CI_FORM = "quadrature"


SCREEN_ALIASING_NOTE = (
    "design.SCREEN_CAVEAT (quoted verbatim above) describes the Resolution-III "
    "aliasing of a Plackett-Burman screen: two-factor interactions are confounded "
    "with main effects, so a factor carrying its influence through an interaction "
    "can look dead and be wrongly dropped. THAT CAVEAT DOES NOT BIND HERE, and the "
    "reason is structural rather than lucky. The main effects below are not read "
    "off 8 PB runs; they are the exact marginal means of a COMPLETE factorial, "
    "where every factor combination is measured and therefore no effect is aliased "
    "onto any other. The caveat is reproduced because the screen is still reported "
    "as SPEC R4.3's Stage-1 deliverable and a reader must be able to see which "
    "limitation was escaped and how. As a check we ALSO run the genuine 8-run PB "
    "design on 8 of the 144 measured cells and compare it to the exact answer: "
    "where the two agree, the aliasing did not bite on this surface; where they "
    "disagree, the difference IS the aliasing, measured rather than assumed."
)


# ============================================================================
# LOADING THE FACTORIAL BLOCK OUT OF THE MASTER TABLE
# ============================================================================

# Ordering matters for the screen's high/low contrast, so categorical level order
# is taken from DEFAULT_FACTOR_SPACE (the single source of truth) rather than from
# whatever order the CSV happens to yield.
_SPACE_LEVELS = {f.name: (list(f.levels) if f.kind == "categorical" else None)
                 for f in DEFAULT_FACTOR_SPACE.factors}
_SPACE_KIND = {f.name: f.kind for f in DEFAULT_FACTOR_SPACE.factors}

# The primary block: the 4 factors that are fully crossed at 4x4x3x3 with babble.
PRIMARY_FACTORS = ("rt60", "snr_db", "codec", "mic_rolloff")
PRIMARY_FIXED = {"noise_type": "babble"}

# The secondary block: noise_type IS crossed here, at coarser resolution on the
# others (3 x 2 x 2 x 2 x 2 = 48 cells). Reported so noise_type is not simply
# absent from the sensitivity story.
SECONDARY_FACTORS = ("noise_type", "rt60", "snr_db", "codec", "mic_rolloff")


def _as_bool(v) -> bool:
    """CSV round-trips booleans as strings and the string 'False' is TRUTHY."""
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "t"}
    return bool(v)


def _lev(name: str, raw: str):
    """Cell coordinate for one factor: float for continuous, str for categorical."""
    return raw if _SPACE_KIND.get(name) == "categorical" else round(float(raw), 6)


def _sorted_levels(name: str, seen) -> list:
    if _SPACE_LEVELS.get(name):
        order = _SPACE_LEVELS[name]
        return [lv for lv in order if lv in seen]
    return sorted(seen)


def load_factorial(rows: Sequence[Mapping], factors: Sequence[str] = PRIMARY_FACTORS,
                   *, model: str = "nova-3", fixed: Mapping | None = None,
                   levels: Mapping[str, Sequence] | None = None,
                   response: str = "wer") -> dict:
    """
    Extract a COMPLETE, clip-balanced factorial block from the master table.

    Returns `W` with shape (n_clips, *cell_shape): the per-clip response in every
    cell. The clip axis is kept (rather than pre-averaging) precisely so the
    bootstrap can resample it — collapsing to cell means here would throw away the
    only thing that makes a clip-level CI computable.

    Raises rather than silently sub-setting if the block is not complete or not
    clip-balanced. A ragged factorial breaks the exactness claim: unequal cell
    counts make the ANOVA terms non-orthogonal, at which point the indices no
    longer partition the variance and the sum-to-1 assertion is the only thing
    that would notice.
    """
    fixed = dict(fixed or {})
    factors = list(factors)

    keep = []
    for r in rows:
        if model is not None and r.get("model") != model:
            continue
        if _as_bool(r.get("failed")):
            continue
        if any(r.get(k) != v for k, v in fixed.items()):
            continue
        y = r.get(response)
        try:
            y = float(y)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(y):
            continue
        keep.append((r, y))

    if not keep:
        raise ValueError(f"no usable rows for model={model!r} fixed={fixed}")

    # Level sets: caller-supplied (to carve a sub-block) or inferred.
    if levels is None:
        seen = {f: set() for f in factors}
        for r, _ in keep:
            for f in factors:
                seen[f].add(_lev(f, r[f]))
        lev = {f: _sorted_levels(f, seen[f]) for f in factors}
    else:
        lev = {f: [_lev(f, str(v)) if not isinstance(v, str) or
                   _SPACE_KIND.get(f) != "continuous" else _lev(f, v)
                   for v in levels[f]] for f in factors}
        lev = {f: [v if _SPACE_KIND.get(f) == "categorical" else round(float(v), 6)
                   for v in levels[f]] for f in factors}

    idx_of = {f: {v: i for i, v in enumerate(lev[f])} for f in factors}
    shape = tuple(len(lev[f]) for f in factors)

    clip_ids = sorted({r["clip_id"] for r, _ in keep})
    clip_idx = {c: i for i, c in enumerate(clip_ids)}

    W = np.full((len(clip_ids),) + shape, np.nan, dtype=float)
    n_used = 0
    for r, y in keep:
        try:
            cell = tuple(idx_of[f][_lev(f, r[f])] for f in factors)
        except KeyError:
            continue                       # outside the requested level sub-block
        W[(clip_idx[r["clip_id"]],) + cell] = y
        n_used += 1

    missing = int(np.isnan(W).sum())
    if missing:
        holes = np.argwhere(np.isnan(W))[:5]
        raise ValueError(
            f"factorial block is INCOMPLETE: {missing} of {W.size} (clip, cell) "
            f"combinations are missing (e.g. {holes.tolist()}). The exact ANOVA "
            f"requires equal cell counts; refusing to decompose a ragged design."
        )

    # The hole check above cannot see a DUPLICATE. The fill loop assigns
    # W[clip, cell] = y, so a second row for a (clip, cell) already written
    # silently overwrites the first and leaves no NaN behind -- the block looks
    # complete and the decomposition runs on whichever row happened to land
    # last. That is this project's signature failure mode: a clean-looking
    # number with no error message. The realizable cause is mundane (a partial
    # re-run appending a fresh transcript for an already-measured cell), so
    # assert the count identity rather than trusting the caller.
    if n_used != W.size:
        raise ValueError(
            f"DUPLICATE (clip, cell) rows in the factorial block: consumed "
            f"{n_used} rows to fill {W.size} slots ({n_used - W.size} extra). "
            f"A later row silently overwrote an earlier measurement; the exact "
            f"ANOVA would then decompose whichever row landed last. Deduplicate "
            f"the table (or rebuild it from the cache) before decomposing."
        )

    return {
        "W": W,                              # (n_clips, *shape)
        "factors": factors,
        "levels": {f: list(lev[f]) for f in factors},
        "shape": shape,
        "clip_ids": clip_ids,
        "n_clips": len(clip_ids),
        "n_cells": int(np.prod(shape)),
        "n_rows": n_used,
        "model": model,
        "fixed": fixed,
        "response": response,
    }


# ============================================================================
# BLOCK INTEGRITY — asserted at the door of every consumer
# ============================================================================

MIN_BOOTSTRAP_CLIPS = 2


def _require_usable_block(block: Mapping, where: str, *,
                          min_clips: int = MIN_BOOTSTRAP_CLIPS) -> np.ndarray:
    """Assert a factorial block is dense, finite and bootstrappable. Returns W.

    `load_factorial` already establishes all of this for a block it built, but
    `decompose` / `screen_from_factorial` / `measured_counterintuitive` are
    public and are handed hand-assembled blocks (the tests do exactly that, and
    so would any caller carving a sub-block). Each of the three failures below
    is silent rather than loud if it is not checked here:

      * A SHAPE/COUNT MISMATCH means the block does not hold what it claims to.
        `W.size` is the number of measurements the decomposition will actually
        use, and `n_rows` is what every report prints as "real transcriptions
        decomposed" — if they disagree, the provenance line is a fiction.
      * A NON-FINITE ENTRY propagates through every ANOVA term, and
        `_check_partition`'s `nan > tol` is False, so the decomposition's one
        real guard would pass and hand back an all-`nan` index table.
      * FEWER THAN TWO CLIPS makes every bootstrap `std(ddof=1)` NaN, i.e. every
        confidence interval in the output silently becomes `nan +/- nan` while
        the point estimates still look fine.
    """
    W = np.asarray(block["W"], dtype=float)
    shape = tuple(block["shape"])
    n_clips = int(block.get("n_clips", W.shape[0] if W.ndim else 0))

    if W.size == 0:
        raise ValueError(
            f"{where}: the factorial block is EMPTY (W.shape={W.shape}). There "
            f"is nothing to decompose; every index would be nan. Check the "
            f"model/fixed filters that built it.")
    if W.shape != (n_clips,) + shape:
        raise ValueError(
            f"{where}: block shape identity broken — W.shape={W.shape} but the "
            f"block declares n_clips={n_clips} and cell shape {shape} "
            f"(expected {(n_clips,) + shape}). The decomposition would run over "
            f"the wrong axes and still return in-range indices.")
    declared = block.get("n_rows")
    if declared is not None and int(declared) != int(W.size):
        raise ValueError(
            f"{where}: row-count identity broken — the block declares "
            f"n_rows={int(declared)} but W holds {int(W.size)} measurements "
            f"({int(declared) - int(W.size)} unaccounted for). Every report "
            f"prints n_rows as 'real transcriptions decomposed', so a mismatch "
            f"means the provenance line describes a different dataset than the "
            f"one the numbers came from.")
    n_bad = int(np.count_nonzero(~np.isfinite(W)))
    if n_bad:
        holes = np.argwhere(~np.isfinite(W))[:5]
        raise ValueError(
            f"{where}: {n_bad} of {W.size} (clip, cell) responses are NOT "
            f"FINITE (e.g. {holes.tolist()}). A NaN propagates into every ANOVA "
            f"term and `_check_partition` cannot see it (`nan > tol` is False), "
            f"so the whole S1/ST table would come back nan with no error. Usual "
            f"cause: failed rows left in the table, or a hand-built block whose "
            f"cells were never all filled.")
    if n_clips < min_clips:
        raise ValueError(
            f"{where}: only {n_clips} clip(s) in the block; the clip-level "
            f"bootstrap needs >= {min_clips} or every std(ddof=1) is nan and "
            f"every CI silently prints as nan while the point estimates still "
            f"look fine.")
    return W


# ============================================================================
# THE EXACT DECOMPOSITION
# ============================================================================

def _subsets(k: int) -> list[tuple[int, ...]]:
    """All non-empty subsets of {0..k-1}, ordered by size then lexicographically.

    Size-ascending order is load-bearing: the Mobius step for a subset subtracts
    every PROPER subset's effect, so those must already exist when it runs.
    """
    out = []
    for size in range(1, k + 1):
        out.extend(itertools.combinations(range(k), size))
    return out


def anova_variance_terms(Y: np.ndarray, k: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Functional-ANOVA variance terms of `Y` over its LAST `k` axes.

    `Y` may carry leading batch axes (used to run the whole bootstrap at once);
    the decomposition is applied independently along them.

    Returns (V_terms, V_total, grand_mean) where V_terms maps a subset tuple ->
    array of that term's variance contribution, shaped EXACTLY like the batch axes
    (a 0-d array when there are none). The reshape is not cosmetic: `f_u` is built
    with keepdims so it broadcasts during the Mobius step, which leaves each term
    with a different trailing rank. Summing those relies on broadcasting and would
    silently produce a partition check over the wrong shape.

    The construction is the standard Mobius inversion:
        m_u  = mean of Y over every axis NOT in u        (the "closed" mean)
        f_u  = m_u - sum over proper subsets v of u of f_v
        V_u  = mean of f_u^2
    and the identity sum_u V_u = Var(Y) holds exactly for a complete equally-
    weighted grid. `_check_partition` asserts it.
    """
    Y = np.asarray(Y, dtype=float)
    b = Y.ndim - k                                   # number of leading batch axes
    if b < 0:
        raise ValueError(f"Y.ndim={Y.ndim} < k={k}")
    ax = lambda d: b + d                             # noqa: E731 — factor axis d

    all_dims = tuple(range(k))
    batch_shape = Y.shape[:b]
    grand = Y.mean(axis=tuple(ax(d) for d in all_dims), keepdims=True)

    f_terms: dict[tuple[int, ...], np.ndarray] = {}
    V_terms: dict[tuple[int, ...], np.ndarray] = {}

    for u in _subsets(k):
        comp = tuple(ax(d) for d in all_dims if d not in u)
        m_u = Y.mean(axis=comp, keepdims=True) if comp else Y
        f_u = m_u - grand
        # subtract every strictly-smaller contained effect (they broadcast: each
        # f_v has size 1 on the axes it does not own)
        for size in range(1, len(u)):
            for v in itertools.combinations(u, size):
                f_u = f_u - f_terms[v]
        f_terms[u] = f_u
        # mean over u's own axes; the complement axes are all size 1 (keepdims),
        # so reshaping to the batch shape is exact, not a squeeze-and-hope.
        V_terms[u] = (f_u ** 2).mean(axis=tuple(ax(d) for d in u)).reshape(batch_shape)

    V_total = ((Y - grand) ** 2).mean(axis=tuple(ax(d) for d in all_dims))
    return V_terms, V_total, np.squeeze(grand, axis=tuple(ax(d) for d in all_dims))


def _check_partition(V_terms: Mapping, V_total: np.ndarray, tol: float = 1e-9) -> float:
    """
    Assert the decomposition partitions the variance exactly. This is the single
    cheapest guard against a transposed axis or a botched Mobius subtraction —
    both of which otherwise yield indices that are in range, plausibly ranked, and
    wrong. Returns the worst observed |sum - 1|.
    """
    total = sum(V_terms.values())
    V_total = np.asarray(V_total, dtype=float)
    denom = np.where(V_total > 0, V_total, 1.0)
    err = np.abs(np.asarray(total) / denom - 1.0)
    worst = float(np.max(err))
    # A NaN anywhere in the response makes every variance term NaN, and
    # `nan > tol` is FALSE — so the partition check, the one guard this whole
    # decomposition leans on, would pass silently and hand back an all-NaN S1/ST
    # table that prints as `nan` rather than as an error. Check finiteness
    # explicitly; it is the same class of hole as a duplicate row that leaves no
    # NaN behind.
    if not np.isfinite(worst):
        n_bad = int(np.sum(~np.isfinite(np.asarray(total))))
        raise AssertionError(
            f"ANOVA partition check is NON-FINITE: worst |sum(V_u)/V_total - 1| "
            f"= {worst} ({n_bad} of {np.asarray(total).size} batch entries are "
            f"not finite). A NaN/inf in the response propagates into every "
            f"variance term, and `nan > tol` is False — so this check would "
            f"otherwise PASS and every Sobol index would come back nan. Usual "
            f"cause: a hole in the factorial block (an unfilled cell, or a "
            f"failed row that was not excluded before decomposing)."
        )
    if worst > tol:
        raise AssertionError(
            f"ANOVA decomposition does not partition the variance: worst "
            f"|sum(V_u)/V_total - 1| = {worst:.3e} > {tol:.1e}. This is an "
            f"indexing bug, not a numerical one — do not proceed."
        )
    return worst


def sobol_from_terms(V_terms: Mapping, V_total: np.ndarray, k: int) -> dict:
    """S1 / ST / S2 from the variance terms. Arrays are (batch..., k) / (batch..., k, k)."""
    V_total = np.asarray(V_total, dtype=float)
    safe = np.where(V_total > 0, V_total, np.nan)
    batch = V_total.shape

    S1 = np.stack([V_terms[(i,)] / safe for i in range(k)], axis=-1)
    ST = np.stack([sum(V for u, V in V_terms.items() if i in u) / safe
                   for i in range(k)], axis=-1)
    S2 = np.full(batch + (k, k), np.nan, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            S2[..., i, j] = V_terms[(i, j)] / safe
    return {"S1": S1, "ST": ST, "S2": S2, "V_total": V_total}


# ============================================================================
# CLIP-LEVEL BOOTSTRAP
# ============================================================================

_Z95 = 1.959963984540054


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation of two bootstrap-replicate vectors, NaN-safe.

    Returns NaN when either side is CONSTANT rather than 0.0. A constant S1 across
    replicates means the bootstrap found no variability at all (identical clips),
    in which case the correlation is undefined -- and 0.0 would read as "S1 and ST
    are independent", which is the single claim this number exists to refute.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    a, b = a[ok], b[ok]
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) * (len(a) / (len(a) - 1))
                 / (sa * sb))


def _cell_means(W: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted clip means -> cell means, for a whole batch of bootstrap replicates.

    `weights` is (B, n_clips) multiplicities from a multinomial resample. Using
    multiplicities x a matmul instead of fancy-indexing W[idx] keeps the working
    set at (B, n_cells) floats instead of (B, n_clips, n_cells) — the difference
    between ~2 MB and ~1 GB at B=2000 — and is numerically identical.
    """
    n_clips = W.shape[0]
    flat = W.reshape(n_clips, -1)                        # (n_clips, n_cells)
    return (weights @ flat) / float(n_clips)             # (B, n_cells)


def decompose(block: Mapping, *, bootstrap: int = 2000, seed: int = 0,
              conf_level: float = 0.95) -> dict:
    """
    The whole R4.5 deliverable for one factorial block, in
    `design.sobol_indices`' return shape plus provenance keys.

    Point estimates come from the real cell means; CIs from `bootstrap` clip-level
    resamples. Both go through the identical code path, so a bug in one is a bug
    in both and the sum-to-1 assertion sees it either way.
    """
    W = _require_usable_block(block, "decompose")
    factors = list(block["factors"])
    k = len(factors)
    shape = tuple(block["shape"])
    n_clips = W.shape[0]
    if bootstrap < MIN_BOOTSTRAP_CLIPS:
        raise ValueError(
            f"decompose: bootstrap={bootstrap} replicates cannot produce a CI "
            f"(std(ddof=1) needs >= {MIN_BOOTSTRAP_CLIPS}). Every *_conf and "
            f"*_ci_* field would come back nan next to a perfectly good point "
            f"estimate, which reads as 'no uncertainty' rather than 'not "
            f"computed'.")

    # ---- point estimate: the measured grid -------------------------------
    Y = W.mean(axis=0)                                   # (*shape) cell means
    V_terms, V_total, _ = anova_variance_terms(Y, k)
    partition_err = _check_partition(V_terms, V_total)
    pt = sobol_from_terms(V_terms, V_total, k)
    S1 = np.asarray(pt["S1"], dtype=float)
    ST = np.asarray(pt["ST"], dtype=float)
    S2 = np.asarray(pt["S2"], dtype=float)

    # ---- clip-level bootstrap -------------------------------------------
    rng = np.random.default_rng(seed)
    S1_b = np.empty((bootstrap, k), dtype=float)
    ST_b = np.empty((bootstrap, k), dtype=float)
    S2_b = np.full((bootstrap, k, k), np.nan, dtype=float)

    chunk = max(1, min(bootstrap, 500))
    done = 0
    while done < bootstrap:
        B = min(chunk, bootstrap - done)
        # multinomial(n_clips, uniform) == counts of a with-replacement resample
        wts = rng.multinomial(n_clips, np.full(n_clips, 1.0 / n_clips), size=B
                              ).astype(float)
        Yb = _cell_means(W, wts).reshape((B,) + shape)
        Vt_b, Vtot_b, _ = anova_variance_terms(Yb, k)
        sb = sobol_from_terms(Vt_b, Vtot_b, k)
        S1_b[done:done + B] = sb["S1"]
        ST_b[done:done + B] = sb["ST"]
        S2_b[done:done + B] = sb["S2"]
        done += B

    lo_q = (1.0 - conf_level) / 2.0 * 100.0
    hi_q = (1.0 + conf_level) / 2.0 * 100.0

    def half_width(arr):
        # SALib's `*_conf` convention: Z x bootstrap std, i.e. a HALF-WIDTH.
        # analysis.interactions builds gap_lo/gap_hi as gap -/+ conf, so anything
        # asymmetric would be silently mis-rendered there.
        return _Z95 * np.nanstd(arr, axis=0, ddof=1)

    # S2's lower triangle is all-NaN BY DESIGN (only i<j is defined), and nanstd
    # over an all-NaN slice warns about degrees of freedom. That warning is noise
    # here, not a signal — suppress it narrowly rather than letting it train the
    # reader to ignore warnings from this module.
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            S1_conf, ST_conf = half_width(S1_b), half_width(ST_b)
            S2_conf = half_width(S2_b)
            S1_lo, S1_hi = np.nanpercentile(S1_b, [lo_q, hi_q], axis=0)
            ST_lo, ST_hi = np.nanpercentile(ST_b, [lo_q, hi_q], axis=0)

    # ST - S1 gap. BOTH published intervals are computed here; see GAP_CI_NOTE for
    # why neither may be dropped and why the pre-registered test reads the WIDER
    # one. (An earlier version of this comment claimed "both are reported" while
    # only the direct form was ever stored -- which is precisely how the write-up
    # and the artifact came to publish different numbers under one label.)
    gap_b = ST_b - S1_b
    gap_conf = _Z95 * np.nanstd(gap_b, axis=0, ddof=1)
    gap_lo, gap_hi = np.nanpercentile(gap_b, [lo_q, hi_q], axis=0)

    # QUADRATURE: the two half-widths combined as if S1 and ST were independent.
    # Identical by construction to analysis.interactions._gap_ci(S1_conf, ST_conf),
    # which is the only form computable from a SALib-shaped result. A test pins
    # the two implementations together so they can never drift apart again.
    #
    # math.hypot, NOT np.hypot: they disagree in the last ULP on some inputs, and
    # a 1-ULP difference between the module that WRITES the artifact and the
    # module that READS it turns "these are the same interval" into "these are
    # nearly the same interval", which is how this whole family of defects starts.
    gap_conf_quad = [math.hypot(float(S1_conf[i]), float(ST_conf[i]))
                     for i in range(k)]
    gap_pt = [float(ST[i]) - float(S1[i]) for i in range(k)]

    # The measured reason the quadrature form is wider. Stored per factor so the
    # over-statement is auditable from the artifact via the exact identity
    #     gap_conf_direct^2 == S1_conf^2 + ST_conf^2 - 2*corr*S1_conf*ST_conf
    # rather than merely asserted in prose.
    s1_st_corr = np.array([_pearson(S1_b[:, i], ST_b[:, i]) for i in range(k)])

    interaction_gap = sorted(
        ({"factor": factors[i], "S1": float(S1[i]), "ST": float(ST[i]),
          "gap": gap_pt[i], "S1_conf": float(S1_conf[i]),
          "ST_conf": float(ST_conf[i]),
          # --- interval A: from the gap's own bootstrap replicates (CORRECT) ---
          "gap_conf_direct": float(gap_conf[i]),
          "gap_ci_lo_direct": float(gap_lo[i]), "gap_ci_hi_direct": float(gap_hi[i]),
          # --- interval B: half-widths added in quadrature (CONSERVATIVE; the
          #     one the pre-registered verdict is read off). Built from the same
          #     Python floats interactions.interaction_gap_table starts from, so
          #     the two agree bit-for-bit rather than approximately.
          "gap_conf_quadrature": gap_conf_quad[i],
          "gap_ci_lo_quadrature": gap_pt[i] - gap_conf_quad[i],
          "gap_ci_hi_quadrature": gap_pt[i] + gap_conf_quad[i],
          # --- why they differ, measured ---
          "s1_st_bootstrap_corr": float(s1_st_corr[i]),
          "gap_conf_ratio_quadrature_over_direct": (
              gap_conf_quad[i] / float(gap_conf[i]) if gap_conf[i] > 0
              else float("inf") if gap_conf_quad[i] > 0 else float("nan")),
          "preregistration_ci_form": PREREGISTRATION_CI_FORM}
         for i in range(k)),
        key=lambda d: d["gap"], reverse=True)

    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            if np.isnan(S2[i, j]):
                continue
            pairs.append({"pair": (factors[i], factors[j]), "S2": float(S2[i, j]),
                          "S2_conf": float(S2_conf[i, j]),
                          "S2_ci_lo": float(np.nanpercentile(S2_b[:, i, j], lo_q)),
                          "S2_ci_hi": float(np.nanpercentile(S2_b[:, i, j], hi_q))})
    s2_ranked = sorted(pairs, key=lambda d: d["S2"], reverse=True)

    # variance accounting, for the report
    order_share = {}
    for order in range(1, k + 1):
        order_share[order] = float(
            sum(V for u, V in V_terms.items() if len(u) == order) / V_total)

    return {
        # --- design.sobol_indices' shape (what analysis.interactions consumes) --
        "names": factors,
        "S1": S1, "ST": ST, "S1_conf": S1_conf, "ST_conf": ST_conf,
        "S2": S2, "S2_conf": S2_conf,
        "interaction_gap": interaction_gap, "s2_ranked": s2_ranked,
        # `N` has no Saltelli meaning here; it is the number of measured cells, and
        # `n_eval` the number of real transcriptions the block rests on. Named the
        # same so downstream formatting works, described in `N_meaning`.
        "N": int(block["n_cells"]), "n_eval": int(block["n_rows"]),
        "N_meaning": ("N = measured factorial cells (not a Saltelli base sample); "
                      "n_eval = real (clip, condition) transcriptions decomposed"),
        # --- provenance ---------------------------------------------------------
        "method": METHOD,
        "method_note": METHOD_NOTE,
        "source": "results/master.csv",
        "model": block["model"],
        "fixed_factors": dict(block["fixed"]),
        "levels": {f: list(block["levels"][f]) for f in factors},
        "n_cells": int(block["n_cells"]),
        "n_clips": int(block["n_clips"]),
        "n_rows": int(block["n_rows"]),
        "clip_ids": list(block["clip_ids"]),
        "bootstrap_reps": int(bootstrap),
        "bootstrap_unit": "clip",
        "bootstrap_note": BOOTSTRAP_NOTE,
        "gap_ci_note": GAP_CI_NOTE,
        "gap_ci_forms": ["direct", "quadrature"],
        "preregistration_ci_form": PREREGISTRATION_CI_FORM,
        "bootstrap_seed": int(seed),
        "variance_explained_check": float(sum(V_terms.values()) / V_total),
        "partition_max_abs_error": float(partition_err),
        "total_variance": float(V_total),
        "variance_share_by_order": order_share,
        "input_distribution_caveat": INPUT_DISTRIBUTION_CAVEAT,
        # extra CI bounds (percentile), alongside the SALib-compatible half-widths
        "S1_ci_lo": S1_lo, "S1_ci_hi": S1_hi,
        "ST_ci_lo": ST_lo, "ST_ci_hi": ST_hi,
    }


# ============================================================================
# R4.3 — THE MAIN-EFFECT SCREEN, DERIVED FROM THE GRID
# ============================================================================

def _screen_endpoints(factor: str, levels: Sequence) -> tuple:
    """(low, high) contrast levels — matching design._screen_level_value's rule."""
    return levels[0], levels[-1]


def screen_from_factorial(block: Mapping, *, bootstrap: int = 2000, seed: int = 0,
                          min_survivors: int = 4, drop_frac: float = 0.05) -> dict:
    """
    SPEC R4.3's Stage-1 screen, computed from the completed factorial instead of
    spending 320 fresh API calls on a Plackett-Burman design.

    Each factor's effect is the difference in mean WER between its high and low
    level, AVERAGED OVER ALL OTHER FACTORS — i.e. the exact marginal contrast. In
    a complete factorial this is the same estimand a PB screen targets, computed
    without aliasing because every combination was actually measured.

    Also runs the genuine 8-run PB design on 8 of the measured cells, so the two
    can be compared: agreement demonstrates the aliasing did not bite on this
    surface; disagreement measures it. See SCREEN_ALIASING_NOTE.
    """
    W = _require_usable_block(block, "screen_from_factorial")
    factors = list(block["factors"])
    levels = block["levels"]
    k = len(factors)
    n_clips = W.shape[0]
    Y = W.mean(axis=0)

    def effects_of(Ycells: np.ndarray) -> np.ndarray:
        out = np.empty(Ycells.shape[:-k] + (k,), dtype=float)
        for i, f in enumerate(factors):
            ax = tuple(Ycells.ndim - k + d for d in range(k) if d != i)
            marg = Ycells.mean(axis=ax)                    # (batch..., n_levels_i)
            out[..., i] = marg[..., -1] - marg[..., 0]
        return out

    eff = effects_of(Y)

    rng = np.random.default_rng(seed)
    eff_b = np.empty((bootstrap, k), dtype=float)
    done = 0
    while done < bootstrap:
        B = min(500, bootstrap - done)
        wts = rng.multinomial(n_clips, np.full(n_clips, 1.0 / n_clips),
                              size=B).astype(float)
        Yb = _cell_means(W, wts).reshape((B,) + tuple(block["shape"]))
        eff_b[done:done + B] = effects_of(Yb)
        done += B
    eff_conf = _Z95 * eff_b.std(axis=0, ddof=1)
    eff_lo, eff_hi = np.percentile(eff_b, [2.5, 97.5], axis=0)

    effects = {f: float(eff[i]) for i, f in enumerate(factors)}
    abseff = {f: abs(v) for f, v in effects.items()}
    max_eff = max(abseff.values()) or 1.0
    ranking = sorted(factors, key=lambda f: abseff[f], reverse=True)

    survivors = list(ranking)
    for f in sorted(factors, key=lambda f: abseff[f]):
        if len(survivors) <= min_survivors:
            break
        if abseff[f] < drop_frac * max_eff:
            survivors.remove(f)
        else:
            break
    survivors = [f for f in ranking if f in survivors]
    dropped = [f for f in ranking if f not in survivors]

    # --- per-level marginal means (what the endpoint contrast summarises) ----
    marginals = {}
    for i, f in enumerate(factors):
        ax = tuple(d for d in range(k) if d != i)
        marginals[f] = {str(lv): float(m)
                        for lv, m in zip(levels[f], Y.mean(axis=ax))}

    # --- the genuine PB screen, run on 8 measured cells ---------------------
    pb = _pb_cross_check(block, Y)

    rows = []
    for f in ranking:
        i = factors.index(f)
        lo, hi = _screen_endpoints(f, levels[f])
        rows.append({
            "factor": f, "effect": effects[f], "effect_conf": float(eff_conf[i]),
            "effect_ci_lo": float(eff_lo[i]), "effect_ci_hi": float(eff_hi[i]),
            "low_level": str(lo), "high_level": str(hi),
            "significant": bool(eff_lo[i] > 0.0 or eff_hi[i] < 0.0),
            "pb_effect": pb["effects"].get(f),
            "pb_minus_exact": (None if pb["effects"].get(f) is None
                               else float(pb["effects"][f] - effects[f])),
        })

    return {
        "method": "exact marginal contrasts from the complete factorial",
        "effects": effects, "ranking": ranking,
        "survivors": survivors, "dropped": dropped,
        "rows": rows, "marginal_means": marginals,
        "n_runs_equivalent": int(block["n_cells"]),
        "n_api_calls_spent": 0,
        "api_calls_saved_vs_fresh_pb": 8 * int(block["n_clips"]),
        "caveat": SCREEN_CAVEAT,                # quoted VERBATIM, as SPEC demands
        "caveat_does_not_bind": SCREEN_ALIASING_NOTE,
        "pb_cross_check": pb,
        "bootstrap_reps": int(bootstrap),
        "min_survivors": min_survivors, "drop_frac": drop_frac,
    }


def pb_alias_structure(design: np.ndarray, factors: Sequence[str]) -> dict:
    """
    Which two-factor interaction is confounded with which main effect, for THIS
    design. Computed from the design matrix, not recited from a textbook.

    SCREEN_CAVEAT says Resolution-III aliasing exists; it does not say WHERE. For a
    contrast column to be aliased, the elementwise product of the two factors'
    columns must equal (+/-) a main-effect column. Anything else lands on an unused
    contrast column and contaminates nothing.

    This matters concretely for SPEC §5's pre-registered pair: with 4 factors in an
    8-run PB the pair columns 0x1 turn out NOT to alias onto any main effect, so a
    fresh PB screen would have estimated rt60 and snr_db cleanly anyway — while
    rt60 x codec DOES land on mic_rolloff. Reporting which is which is the
    difference between "screening has a known limitation" and knowing whether that
    limitation touched this result.
    """
    k = len(factors)
    n = design.shape[0]
    aliases: list[dict] = []
    for i, j in itertools.combinations(range(k), 2):
        prod = design[:, i] * design[:, j]
        onto = None
        sign = 0
        for c in range(k):
            dot = float(np.dot(prod, design[:, c])) / n
            if abs(abs(dot) - 1.0) < 1e-9:
                onto, sign = factors[c], int(np.sign(dot))
                break
        aliases.append({
            "pair": [factors[i], factors[j]],
            "aliased_onto": onto,          # None => lands on an unused column
            "sign": sign,
            "contaminates_a_main_effect": onto is not None,
        })
    clean = [a["pair"] for a in aliases if not a["contaminates_a_main_effect"]]
    dirty = [(a["pair"], a["aliased_onto"]) for a in aliases
             if a["contaminates_a_main_effect"]]
    return {
        "aliases": aliases,
        "unaliased_pairs": clean,
        "confounded_pairs": [{"pair": p, "aliased_onto": o} for p, o in dirty],
        "note": ("pairs listed under `unaliased_pairs` land on unused contrast "
                 "columns of this design and therefore bias NO main effect; pairs "
                 "under `confounded_pairs` bias the named factor's estimate."),
    }


def _pb_cross_check(block: Mapping, Y: np.ndarray) -> dict:
    """
    Run the real Plackett-Burman design on cells that were actually measured, and
    report how far its aliased estimates land from the exact marginal effects.

    This is the empirical version of SCREEN_CAVEAT: rather than asserting the
    aliasing is harmless, we measure it on this surface — and `alias_structure`
    says which main effects were even exposed to it.
    """
    factors = list(block["factors"])
    levels = block["levels"]
    k = len(factors)
    design = plackett_burman(k)
    ys = []
    for run in design:
        idx = []
        for i, f in enumerate(factors):
            idx.append(len(levels[f]) - 1 if run[i] > 0 else 0)
        ys.append(float(Y[tuple(idx)]))
    Yr = np.asarray(ys)
    eff = {factors[j]: float(Yr[design[:, j] > 0].mean() - Yr[design[:, j] < 0].mean())
           for j in range(k)}
    return {
        "n_runs": int(design.shape[0]),
        "effects": eff,
        "ranking": sorted(eff, key=lambda f: abs(eff[f]), reverse=True),
        "alias_structure": pb_alias_structure(design, factors),
        "note": ("a genuine Resolution-III PB screen evaluated on 8 of the measured "
                 "factorial cells; compare `pb_effect` to `effect` per factor to see "
                 "the aliasing bias directly, and `alias_structure` to see which "
                 "main effects were exposed to it at all"),
    }


# ============================================================================
# MEASURED COUNTERINTUITIVE CELLS — found IN the grid, not in a surrogate
# ============================================================================
#
# SPEC A.R5.4 prescribes a two-stage protocol: propose on a surrogate, confirm
# with the oracle. That protocol exists because the dense scan is unaffordable
# against the API. But a complete factorial makes a third option available that is
# strictly better than both: scan the MEASUREMENTS THEMSELVES. Every dip found
# here is already measured on 40 clips, needs zero confirmation calls, and cannot
# be a surrogate artefact because no surrogate was involved.
#
# This does not replace the surrogate protocol — the surrogate can look BETWEEN
# grid levels and the grid cannot. It does mean the write-up's counterintuitive
# findings should lead with these.

def _is_ordered(name: str, levels: Sequence) -> bool:
    """
    Is this factor's level axis a MAGNITUDE (so "non-monotonic" is meaningful) or a
    naming convention (so it is not)?

    Two independent signals, and it must fail safe. DEFAULT_FACTOR_SPACE is the
    authority for the five real factors, but a block can be built over factors it
    has never heard of — in which case `_SPACE_KIND.get` returns None, which is not
    "categorical", and the factor would be scanned as if its level order carried
    meaning. So a non-numeric level is treated as categorical regardless: reporting
    a "dip" across codec none -> g726 -> opus-lowrate would be an artefact of the
    order that tuple happens to be written in, which is the single easiest way to
    manufacture a fake surprise in this whole analysis.
    """
    if _SPACE_KIND.get(name) == "categorical":
        return False
    return all(isinstance(v, (int, float, np.floating, np.integer))
               and not isinstance(v, bool) for v in levels)


def _dips_along(Y: np.ndarray, axis: int, k: int) -> np.ndarray:
    """
    Dip depth at every interior position along `axis`, for a batch of grids.

    depth = min(neighbour_lo, neighbour_hi) - centre. Positive => a genuine
    interior local minimum, i.e. MORE degradation produced LOWER WER.
    Returned shape: (batch..., *other_dims, n_interior).
    """
    b = Y.ndim - k
    ax = b + axis
    m = Y.shape[ax]
    lo = np.take(Y, range(0, m - 2), axis=ax)
    mid = np.take(Y, range(1, m - 1), axis=ax)
    hi = np.take(Y, range(2, m), axis=ax)
    depth = np.minimum(lo, hi) - mid
    return np.moveaxis(depth, ax, -1)


def measured_counterintuitive(block: Mapping, *, bootstrap: int = 2000, seed: int = 0,
                              top_k: int = 10, min_depth: float = 0.0) -> dict:
    """
    Scan the measured factorial for interior local minima of WER along each
    ORDERED factor, with clip-bootstrap CIs on the dip depth.

    Two granularities, because they support different claims:
      * `marginal` — dips in a factor's marginal-mean curve (averaged over every
        other factor). The strong claim: "averaged across the whole grid, going
        from level A to level B REDUCED WER."
      * `cellwise` — dips in an individual 1-D line with all other factors pinned.
        Narrower, but concrete and quotable with full coordinates.

    Categorical factors are SKIPPED. `codec`'s level order is a naming convention,
    not a magnitude, so "non-monotonic in codec" would be an artefact of how the
    tuple happens to be written — the single easiest way to manufacture a fake
    surprise here.
    """
    W = _require_usable_block(block, "measured_counterintuitive")
    factors = list(block["factors"])
    levels = block["levels"]
    shape = tuple(block["shape"])
    k = len(factors)
    n_clips = W.shape[0]
    Y = W.mean(axis=0)

    ordered = [i for i, f in enumerate(factors)
               if _is_ordered(f, levels[f]) and len(levels[f]) >= 3]

    # --- bootstrap replicates of the whole grid, reused by both scans --------
    rng = np.random.default_rng(seed)
    Yb_all = np.empty((bootstrap,) + shape, dtype=float)
    done = 0
    while done < bootstrap:
        B = min(500, bootstrap - done)
        wts = rng.multinomial(n_clips, np.full(n_clips, 1.0 / n_clips),
                              size=B).astype(float)
        Yb_all[done:done + B] = _cell_means(W, wts).reshape((B,) + shape)
        done += B

    # --- (a) marginal-curve dips -------------------------------------------
    marginal = []
    for i in ordered:
        others = tuple(d for d in range(k) if d != i)
        curve = Y.mean(axis=others)                              # (n_levels,)
        curve_b = Yb_all.mean(axis=tuple(1 + d for d in others))  # (B, n_levels)
        d_pt = _dips_along(curve, 0, 1)
        d_b = _dips_along(curve_b, 0, 1)
        for p in range(d_pt.shape[-1]):
            depth = float(d_pt[..., p])
            if depth <= min_depth:
                continue
            reps = d_b[..., p]
            lo95, hi95 = np.percentile(reps, [2.5, 97.5])
            marginal.append({
                "factor": factors[i],
                "level": levels[factors[i]][p + 1],
                "from_level": levels[factors[i]][p],
                "to_level": levels[factors[i]][p + 2],
                "wer_at_dip": float(curve[p + 1]),
                "wer_before": float(curve[p]), "wer_after": float(curve[p + 2]),
                "depth": depth,
                "depth_conf": float(_Z95 * reps.std(ddof=1)),
                "depth_ci_lo": float(lo95), "depth_ci_hi": float(hi95),
                "significant": bool(lo95 > 0.0),
                "n_cells_averaged": int(np.prod([shape[d] for d in others])),
                "n_clips": n_clips,
                "evidence": "MEASURED (grid marginal means, no surrogate)",
                "presentable_as_measured": True,
                "note": (f"WER FALLS from {float(curve[p]):.3f} to "
                         f"{float(curve[p+1]):.3f} as {factors[i]} goes "
                         f"{levels[factors[i]][p]} -> {levels[factors[i]][p+1]}, "
                         f"then rises to {float(curve[p+2]):.3f} — more degradation, "
                         f"lower WER"),
            })
    marginal.sort(key=lambda d: d["depth"], reverse=True)

    # --- (b) cellwise dips --------------------------------------------------
    cellwise = []
    for i in ordered:
        d_pt = _dips_along(Y, i, k)                 # (*others, n_interior)
        d_b = _dips_along(Yb_all, i, k)             # (B, *others, n_interior)
        other_dims = [d for d in range(k) if d != i]
        for idx in np.ndindex(d_pt.shape):
            depth = float(d_pt[idx])
            if depth <= min_depth:
                continue
            reps = d_b[(slice(None),) + idx]
            lo95, hi95 = np.percentile(reps, [2.5, 97.5])
            if lo95 <= 0.0:
                continue                            # CI crosses zero: not evidence
            p = idx[-1]
            coords = {factors[d]: levels[factors[d]][idx[j]]
                      for j, d in enumerate(other_dims)}
            full = dict(coords)
            full[factors[i]] = levels[factors[i]][p + 1]
            cellwise.append({
                "factor": factors[i],
                "level": levels[factors[i]][p + 1],
                "from_level": levels[factors[i]][p],
                "to_level": levels[factors[i]][p + 2],
                "coords": full,
                "wer_triplet": [float(np.take(Y, p + o, axis=i)[idx[:-1]])
                                for o in (0, 1, 2)],
                "depth": depth,
                "depth_conf": float(_Z95 * reps.std(ddof=1)),
                "depth_ci_lo": float(lo95), "depth_ci_hi": float(hi95),
                "significant": True,
                "n_clips": n_clips,
                "evidence": "MEASURED (grid cell means, no surrogate)",
                "presentable_as_measured": True,
            })
    cellwise.sort(key=lambda d: d["depth"], reverse=True)

    return {
        "status": "MEASURED_IN_GRID",
        "presentable_as_measured": True,
        "source": "the 144-cell measured factorial (no surrogate, no extra API calls)",
        "marginal": marginal[:top_k],
        "cellwise": cellwise[:top_k],
        "n_marginal": len(marginal),
        "n_cellwise_significant": len(cellwise),
        "ordered_factors_scanned": [factors[i] for i in ordered],
        "categorical_skipped": [f for f in factors if not _is_ordered(f, levels[f])],
        "skip_note": ("categorical factors are excluded: their level ORDER is a "
                      "naming convention, so a 'dip' across it would be an artefact "
                      "of tuple order rather than a physical non-monotonicity"),
        "bootstrap_reps": int(bootstrap),
        "api_calls_spent": 0,
    }


def format_measured_counterintuitive(res: Mapping) -> str:
    out = ["=" * 78,
           "MEASURED COUNTERINTUITIVE CELLS (in-grid; zero surrogate, zero API calls)",
           "=" * 78, "",
           f"  scanned: {res['ordered_factors_scanned']}   "
           f"skipped (categorical): {res['categorical_skipped']}", ""]
    out += ["  (a) MARGINAL-CURVE DIPS — averaged over every other factor:"]
    if not res["marginal"]:
        out.append("      (none)")
    for d in res["marginal"]:
        out += [f"      {d['factor']} = {d['from_level']} -> {d['level']} -> {d['to_level']}",
                f"        WER {d['wer_before']:.4f} -> {d['wer_at_dip']:.4f} -> "
                f"{d['wer_after']:.4f}",
                f"        dip depth {d['depth']:.4f} "
                f"[{d['depth_ci_lo']:.4f}, {d['depth_ci_hi']:.4f}]  "
                f"{'SIGNIFICANT' if d['significant'] else 'CI crosses 0'}"
                f"  (n={d['n_cells_averaged']} cells x {d['n_clips']} clips)"]
    out += ["", f"  (b) CELLWISE DIPS — {res['n_cellwise_significant']} with a CI "
                f"entirely above zero; top {len(res['cellwise'])}:"]
    if not res["cellwise"]:
        out.append("      (none)")
    for d in res["cellwise"]:
        co = ", ".join(f"{a}={b}" for a, b in d["coords"].items() if a != d["factor"])
        t = d["wer_triplet"]
        out += [f"      {d['factor']} {d['from_level']} -> {d['level']} -> {d['to_level']}"
                f"   [{co}]",
                f"        WER {t[0]:.4f} -> {t[1]:.4f} -> {t[2]:.4f}   depth "
                f"{d['depth']:.4f} [{d['depth_ci_lo']:.4f}, {d['depth_ci_hi']:.4f}]"]
    return "\n".join(out)


# ============================================================================
# ARTIFACT PROVENANCE — every file this module writes says what it is FOR
# ============================================================================
#
# `results/sobol_5factor.json` was flagged by an artifact audit as an ORPHAN:
# generated on every run, internally sound (partition sum 1.0000000000000002),
# and referenced by no analysis module, no dashboard panel, no test and no
# write-up claim. It is not dead output -- it is the ONLY place `noise_type`
# receives a variance share, because the primary block holds noise_type fixed at
# babble in order to be a complete factorial. But "has a purpose" and "states its
# purpose" are different things, and a numerically-plausible JSON with no stated
# scope is exactly the kind of file that gets quoted alongside the headline
# indices by someone who does not know the two blocks measure different grids.
#
# So each written payload carries its own provenance header. The two blocks are
# NOT comparable index-for-index: different factor sets, different level sets,
# different totals (V_total 0.1266 vs 0.0962), so an S1 from one may never be put
# in a table beside an S1 from the other.

PRIMARY_BLOCK_DOC = {
    "block_role": "primary",
    "purpose": ("R4.5's headline sensitivity: the exact functional-ANOVA "
                "decomposition of the complete 4x4x3x3 factorial "
                "(rt60 x snr_db x codec x mic_rolloff) at noise_type=babble. "
                "Every Sobol number quoted in the write-up comes from here."),
    "consumed_by": [
        "deadzone.analysis.interactions (load_sobol_json -> the pre-registration "
        "verdict and the ST-S1 gap table)",
        "dashboard/build.py (the sensitivity panel)",
        "report/writeup.md (the S1/ST table and the pre-registration blockquote)",
    ],
    "quotable": True,
}

SECONDARY_BLOCK_DOC = {
    "block_role": "secondary (noise_type crossed, coarser elsewhere)",
    "purpose": ("The ONLY block in which `noise_type` varies, so it is the only "
                "place that factor can receive a variance share at all. The "
                "primary block pins noise_type='babble' because that is what "
                "makes it a COMPLETE factorial; without this secondary "
                "decomposition noise_type would be silently absent from the "
                "sensitivity story rather than measured and found small."),
    "consumed_by": [],
    "quotable": False,
    "status": ("REFERENCE ONLY. Regenerated on every run of this module and read "
               "by NO analysis module, NO dashboard panel, NO test and NO "
               "write-up claim. Kept because deleting it would remove the only "
               "evidence about noise_type, not because anything depends on it."),
    "do_not": ("Do NOT place an index from this file in a table beside one from "
               "results/sobol.json. The two decompose DIFFERENT grids -- "
               "different factor sets, different level sets (this block is "
               "2-level on rt60/snr_db/codec/mic_rolloff), different total "
               "variance -- so the indices are shares of different denominators "
               "and are not comparable index-for-index."),
}


def tag_artifact(res: Mapping, doc: Mapping, path: str | None = None) -> dict:
    """
    Attach a provenance header to a decomposition payload before it is written.

    Kept separate from `decompose` on purpose: `decompose` computes a variance
    decomposition and should not know which file it is destined for, while a file
    on disk with no stated scope is the thing that gets mis-quoted. The header
    goes FIRST in the dict so it is the first thing visible in the written JSON.
    """
    head = {"artifact": path, **{k: v for k, v in doc.items()}}
    out = dict(head)
    out.update({k: v for k, v in res.items() if k not in head})
    return out


# ============================================================================
# SERIALIZATION
# ============================================================================

# ---------------------------------------------------------------------------
# NON-FINITE ENCODING — why this file emits `null` and not `NaN`
# ---------------------------------------------------------------------------
# `json.dump` writes a bare `NaN` token by default. That is a Python extension,
# NOT JSON: RFC 8259 has no NaN literal, so `JSON.parse`, `serde_json`,
# `encoding/json` and most other parsers REJECT the file outright. Only Python's
# own `json` reads it back. This module's artifacts are advertised as `quotable`
# and are read by the write-up, the dashboard and `analysis.interactions`, so
# shipping a file that half the world cannot open is a defect regardless of who
# happens to be reading it today.
#
# The NaNs are not errors, which is exactly why this went unnoticed: S2 is a
# strictly-upper-triangular matrix (only i<j is a defined pair), so its diagonal
# and lower triangle are *meaningfully absent* — 10 of 16 entries for a 4-factor
# block, 30 of 50 for the 5-factor one.
#
# Three options were available and the trade is real:
#   omit the key        loses the matrix SHAPE; `S2[i][j]` stops being indexable
#                       and every consumer needs a second code path.
#   numeric sentinel    a real number that means "not a number" is the exact
#                       silent-failure pattern this project exists to study.
#   null  <- CHOSEN     valid JSON, keeps the shape, and round-trips EXACTLY:
#                       `normalize_sobol_result` does
#                       `np.asarray(v, dtype=float)`, and numpy maps None -> nan,
#                       so its `np.isnan(S2[i, j])` skip-test is bit-identical to
#                       what it was before. No consumer changes.
#
# `null` alone would still be lossy in one direction: nan, +inf and -inf all
# collapse to the same token. So every substitution is recorded losslessly in a
# `nonfinite_encoding` block (path + original value), which is emitted ONLY when
# a substitution actually happened. That makes the encoding auditable instead of
# implicit, and keeps files with no non-finite values byte-unchanged.
#
# `write_sobol_json` then passes `allow_nan=False`, so this can never silently
# regress: if a future field escapes the conversion, the write RAISES instead of
# quietly producing an invalid file again.

_NONFINITE_NAMES = {float("inf"): "inf", float("-inf"): "-inf"}


def _nonfinite_name(v: float) -> str:
    return _NONFINITE_NAMES.get(v, "nan")


def _strip_nonfinite(obj, path: str, log: list) -> object:
    """Recursively replace non-finite floats with None, logging each one."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            log.append({"path": path, "was": _nonfinite_name(obj)})
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _strip_nonfinite(v, f"{path}/{k}", log) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_nonfinite(v, f"{path}[{i}]", log) for i, v in enumerate(obj)]
    return obj


def to_json(res: Mapping) -> dict:
    """
    Strict-JSON-safe view of a decomposition payload.

    Non-finite floats become `null` (see the block comment above): valid JSON,
    shape-preserving, and `np.asarray(..., dtype=float)` maps it straight back to
    `nan` for `normalize_sobol_result`'s `np.isnan` skip-test. Every substitution
    is itemised under `nonfinite_encoding`, which is present only when something
    was actually substituted — so a payload with no non-finite values (e.g. the
    Plackett-Burman screen) is unchanged, key for key.
    """
    out = {}
    for kk, v in res.items():
        if isinstance(v, np.ndarray):
            out[kk] = v.tolist()
        elif kk == "s2_ranked":
            out[kk] = [{**d, "pair": list(d["pair"])} for d in v]
        elif isinstance(v, dict):
            out[kk] = {str(a): (b.tolist() if isinstance(b, np.ndarray) else b)
                       for a, b in v.items()}
        else:
            out[kk] = v

    log: list = []
    out = _strip_nonfinite(out, "", log)
    if log:
        out["nonfinite_encoding"] = {
            "encoded_as": None,
            "note": ("RFC 8259 has no NaN/Infinity literal, so non-finite values "
                     "are written as `null` to keep this file parseable by every "
                     "JSON reader, not just Python's. `null` round-trips to `nan` "
                     "through numpy (`np.asarray(v, dtype=float)`), which is how "
                     "analysis.interactions.normalize_sobol_result already reads "
                     "it. Nothing here is a measurement that went missing."),
            "why_present": ("S2/S2_conf are strictly upper triangular — only i<j "
                            "names a defined factor pair — so the diagonal and "
                            "lower triangle are absent BY CONSTRUCTION, not by "
                            "failure."),
            "n_substituted": len(log),
            "substitutions": log,
        }
    return out


def write_sobol_json(res: Mapping, path: str) -> str:
    """
    Write a decomposition payload as STRICT JSON.

    `allow_nan=False` is the guard, not the docstring: if any value escapes
    `to_json`'s conversion this raises `ValueError` rather than writing a file
    that only Python can read. An invalid artifact that parses locally and fails
    everywhere else is precisely the silent failure this repo studies.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = to_json(res)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    return path


# ============================================================================
# REPORT
# ============================================================================

def format_sensitivity_report(sob: Mapping, screen: Mapping | None = None) -> str:
    names = list(sob["names"])
    S1 = np.asarray(sob["S1"], dtype=float)
    ST = np.asarray(sob["ST"], dtype=float)
    S1c = np.asarray(sob["S1_conf"], dtype=float)
    STc = np.asarray(sob["ST_conf"], dtype=float)

    out = ["=" * 78,
           "SENSITIVITY — EXACT FUNCTIONAL-ANOVA DECOMPOSITION OF THE REAL GRID",
           "=" * 78, "",
           f"  method            {sob['method']}",
           f"  block             {sob['n_cells']} cells x {sob['n_clips']} clips "
           f"= {sob['n_rows']} real transcriptions",
           f"  model             {sob['model']}",
           f"  held fixed        {sob['fixed_factors'] or '(none)'}",
           f"  bootstrap         {sob['bootstrap_reps']} clip-level resamples",
           f"  partition check   sum(S_u) = {sob['variance_explained_check']:.12f} "
           f"(max abs error {sob['partition_max_abs_error']:.2e})",
           f"  total variance    {sob['total_variance']:.6f} (of condition-mean WER)",
           ""]

    out += ["  levels:"]
    for f in names:
        out.append(f"    {f:<12} {sob['levels'][f]}")
    out += ["", f"  {METHOD_NOTE}", "", f"  CAVEAT: {INPUT_DISTRIBUTION_CAVEAT}", ""]

    if screen is not None:
        out += ["-" * 78,
                "R4.3 — MAIN-EFFECT SCREEN (exact marginal contrasts, 0 API calls)",
                "-" * 78,
                f"  {'factor':<13} {'effect':>9} {'95% CI':>20}  {'PB(8-run)':>10}  "
                f"{'PB-exact':>9}"]
        for r in screen["rows"]:
            pbe = "n/a" if r["pb_effect"] is None else f"{r['pb_effect']:>10.4f}"
            pbd = "n/a" if r["pb_minus_exact"] is None else f"{r['pb_minus_exact']:>9.4f}"
            out.append(f"  {r['factor']:<13} {r['effect']:>9.4f} "
                       f"[{r['effect_ci_lo']:>8.4f},{r['effect_ci_hi']:>8.4f}]  "
                       f"{pbe}  {pbd}")
        out += ["",
                f"  high/low contrast per factor: " +
                ", ".join(f"{r['factor']}: {r['low_level']} -> {r['high_level']}"
                          for r in screen["rows"]),
                f"  survivors: {screen['survivors']}   dropped: {screen['dropped'] or '(none)'}",
                "",
                "  design.SCREEN_CAVEAT (verbatim, as SPEC R4.3 requires):",
                "    " + SCREEN_CAVEAT,
                "",
                "  WHY IT DOES NOT BIND HERE:",
                "    " + SCREEN_ALIASING_NOTE, ""]
        out += ["  marginal mean WER by level:"]
        for f in screen["ranking"]:
            cells = "  ".join(f"{lv}={m:.3f}" for lv, m in screen["marginal_means"][f].items())
            out.append(f"    {f:<13} {cells}")
        out.append("")

    out += ["-" * 78,
            "R4.5 — SOBOL INDICES (exact; CIs from the clip bootstrap)",
            "-" * 78,
            f"  {'factor':<13} {'S1':>18} {'ST':>18} {'ST-S1':>9} "
            f"{'gap 95% CI (direct)':>22}"]
    for d in sob["interaction_gap"]:
        i = names.index(d["factor"])
        out.append(
            f"  {d['factor']:<13} {S1[i]:>8.4f}+/-{S1c[i]:<7.4f} "
            f"{ST[i]:>8.4f}+/-{STc[i]:<7.4f} {d['gap']:>9.4f} "
            f"  [{d['gap_ci_lo_direct']:>8.4f},{d['gap_ci_hi_direct']:>8.4f}]")

    # BOTH published intervals, side by side and named. Printing only one of them
    # under a bare `gap 95% CI` header is the defect GAP_CI_NOTE describes.
    if all("gap_conf_quadrature" in d for d in sob["interaction_gap"]):
        ivl = lambda lo, hi: f"[{lo:>7.4f}, {hi:>7.4f}]"      # noqa: E731
        out += ["",
                "  ST-S1 gap — the TWO published 95% intervals, side by side "
                "(they are DIFFERENT quantities; read the label, never the "
                "bare numbers):",
                f"  {'factor':<13} {'ST-S1':>8} {'direct (correct)':>26} "
                f"{'quadrature (CONSERVATIVE)':>30} {'corr(S1,ST)':>12} "
                f"{'x wider':>8}"]
        for d in sob["interaction_gap"]:
            ratio = d.get("gap_conf_ratio_quadrature_over_direct", float("nan"))
            out.append(
                f"  {d['factor']:<13} {d['gap']:>8.4f} "
                f"{ivl(d['gap_ci_lo_direct'], d['gap_ci_hi_direct']):>26} "
                f"{ivl(d['gap_ci_lo_quadrature'], d['gap_ci_hi_quadrature']):>30} "
                f"{d.get('s1_st_bootstrap_corr', float('nan')):>+12.3f} "
                f"{ratio:>8.2f}")
        out += ["",
                f"  PRE-REGISTERED VERDICT IS READ OFF THE "
                f"{str(sob.get('preregistration_ci_form', PREREGISTRATION_CI_FORM)).upper()} "
                f"INTERVAL.",
                f"  {GAP_CI_NOTE}"]

    out += ["", "  Second-order S2 (WHICH pair only — never a magnitude):",
            f"  {'pair':<30} {'S2':>18} {'95% CI':>20}"]
    for d in sob["s2_ranked"]:
        a, b = d["pair"]
        out.append(f"  {a + ' x ' + b:<30} {d['S2']:>8.4f}+/-{d['S2_conf']:<7.4f} "
                   f"[{d['S2_ci_lo']:>8.4f},{d['S2_ci_hi']:>8.4f}]")

    out += ["", "  variance share by interaction order:"]
    for order, share in sorted(sob["variance_share_by_order"].items(),
                               key=lambda kv: int(kv[0])):
        out.append(f"    order {order}: {float(share):.4f}")
    out += ["", f"  {BOOTSTRAP_NOTE}", ""]
    return "\n".join(out)


# ============================================================================
# CLI
# ============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    from deadzone.analysis.interactions import load_master_rows

    ap = argparse.ArgumentParser(
        description="R4.3 + R4.5 — exact factorial ANOVA sensitivity from the real grid")
    ap.add_argument("--master", default="results/master.csv")
    ap.add_argument("--model", default="nova-3")
    ap.add_argument("--out", default="results/sobol.json")
    ap.add_argument("--out-5factor", default="results/sobol_5factor.json")
    ap.add_argument("--screen-out", default="results/screen.json")
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rows = load_master_rows(args.master)

    block = load_factorial(rows, PRIMARY_FACTORS, model=args.model, fixed=PRIMARY_FIXED)
    print(f"[factorial] {block['n_cells']} cells x {block['n_clips']} clips "
          f"= {block['n_rows']} rows; levels: {block['levels']}")

    sob = decompose(block, bootstrap=args.bootstrap, seed=args.seed)
    screen = screen_from_factorial(block, bootstrap=args.bootstrap, seed=args.seed)

    report = format_sensitivity_report(sob, screen)
    print(report)
    write_sobol_json(tag_artifact(sob, PRIMARY_BLOCK_DOC, args.out), args.out)
    print(f"[wrote] {args.out}")
    # Same strict-JSON writer as the Sobol blocks. The screen payload happens to
    # carry no non-finite values today, so this changes its bytes not at all —
    # which is the point: the guard belongs on the writer, not on the one file
    # currently known to need it.
    write_sobol_json(screen, args.screen_out)
    print(f"[wrote] {args.screen_out}")

    # --- secondary block: noise_type crossed, coarser elsewhere -------------
    if args.out_5factor:
        sub_levels = {"noise_type": ["babble", "engine", "road"],
                      "rt60": [0.45, 1.0], "snr_db": [0.0, 10.0],
                      "codec": ["none", "g726"], "mic_rolloff": [0.0, 1.0]}
        try:
            blk5 = load_factorial(rows, SECONDARY_FACTORS, model=args.model,
                                  levels=sub_levels)
            sob5 = decompose(blk5, bootstrap=args.bootstrap, seed=args.seed)
            write_sobol_json(tag_artifact(sob5, SECONDARY_BLOCK_DOC,
                                          args.out_5factor), args.out_5factor)
            print("\n" + format_sensitivity_report(sob5))
            print(f"[wrote] {args.out_5factor}  "
                  f"[{SECONDARY_BLOCK_DOC['block_role']}] "
                  f"{SECONDARY_BLOCK_DOC['status']}")
        except ValueError as e:
            print(f"[5-factor block] skipped: {e}")

    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[wrote] {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
