"""
analysis/al_savings.py — D3b: the active-learning savings curve (SPEC §5 third
leg, A.R5.5).

`active_learning.py` owns the machinery (the GP surrogate, the straddle
acquisition, the arms, the fidelity metrics). This file owns the *claim* the
write-up makes from it:

    "Straddle-acquisition active learning reaches boundary RMSE <= 0.080 in 24
     oracle evaluations vs 45 for random sampling — a 47% reduction in oracle
     calls (median over 3 seeds, range 21-30)."

That sentence is only honest if two things are true, and this module is built so
that neither can be quietly skipped.

THE TWO TRAPS THIS FILE EXISTS TO AVOID
---------------------------------------
TRAP 1 — THE SINGLE-SEED HEADLINE. Active-vs-random is a comparison of two
stochastic procedures with a large seed variance (SPEC A.R4.7 gotcha). Run three
seeds and quote the best one and you can manufacture almost any speedup factor
you like — including from a method that does nothing. The structural defence:
`multi_seed_curves` REFUSES fewer than `MIN_SEEDS` seeds unless you pass
`allow_single_seed=True`, and that flag is not an escape hatch — it stamps the
whole result `evidence_level="INSUFFICIENT"`, replaces the headline with an
explicit not-evidence string, and makes the formatter print a warning banner
instead of the savings number. Reporting is the seed BAND (median + min/max
across seeds), never the best seed; `best_seed` is not computed anywhere.

TRAP 2 — REBUYING MEASUREMENTS YOU ALREADY OWN. `make_test_set` calls the oracle
n times (default 512). Against the real API that is 512 * n_clips calls, i.e.
thousands of dollars-worth of round trips to build a yardstick that the master
results table ALREADY contains as real, measured (params -> WER) pairs. So the
primary path here is `test_set_from_master`: it makes ZERO oracle calls, and the
resulting test set is *more* honest than a fresh one, because those are real
transcriptions rather than another draw from the same simulator.
`master_split_for_al` additionally keeps the surrogate-oracle's training
conditions DISJOINT from the scoring conditions — score a surrogate oracle on the
points it was fitted to and both arms look perfect and the comparison is dead.

PROVENANCE IS PART OF THE CLAIM. The multi-seed replicates run against a GP
surrogate oracle fitted to the master table (instant, free); typically ONE seed is
confirmed end-to-end against the live API oracle. The report states which seeds
were API-backed in every rendering — an unlabelled "median over 5 seeds" invites
the reader to assume five API-backed runs (SPEC A.R5.5 gotcha).

    python3 -m analysis.al_savings results/master.csv --seeds 0 1 2

Deps: numpy, scikit-learn (via active_learning).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Callable, Mapping, Sequence

import numpy as np

# Allow `python analysis/al_savings.py` as well as `import analysis.al_savings`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from design import DEFAULT_FACTOR_SPACE, FactorSpace          # noqa: E402
from active_learning import (                                  # noqa: E402
    DEFAULT_BAND, DEFAULT_THRESHOLD, GPSurrogate, Trajectory, active_learn,
    best_so_far, boundary_error, boundary_rmse, evals_to_target,
    learning_curve, random_baseline,
)
# D3a owns the master-table -> factor-matrix conversion; reused, not duplicated.
from analysis.interactions import (                            # noqa: E402
    condition_matrix, encode_sample, load_master_rows,
)


# ============================================================================
# CONSTANTS
# ============================================================================

MIN_SEEDS = 3                 # SPEC A.R4.7: "one seed is not evidence. Run >= 3."
METRIC = "boundary_rmse"      # the headline fidelity metric (active_learning.py)
ARMS = ("active_boundary", "active_uncertainty", "random")
ACTIVE_ARM = "active_boundary"
PASSIVE_ARM = "random"

INSUFFICIENT = "INSUFFICIENT"
SUFFICIENT = "SUFFICIENT"

_NOT_EVIDENCE = (
    "NOT EVIDENCE — a single seed. Active-vs-random has large seed variance; a "
    "one-seed speedup is an anecdote. Re-run with >= {min_seeds} seeds before "
    "quoting any number from this run."
)


class InsufficientSeedsError(ValueError):
    """Raised when a savings claim is attempted from fewer than MIN_SEEDS seeds."""


# ============================================================================
# THE TEST SET — from the master table, at ZERO oracle cost
# ============================================================================

def master_split_for_al(rows: Sequence[Mapping],
                        space: FactorSpace = DEFAULT_FACTOR_SPACE, *,
                        model: str | None = None, holdout_frac: float = 0.4,
                        seed: int = 0, use_measured_rt60: bool = True,
                        min_test: int = 8) -> dict:
    """
    Split the measured conditions into a surrogate-ORACLE training set and a
    disjoint held-out TEST set. Zero oracle calls: this is pure bookkeeping over
    rows that were already paid for.

    Why the split is not optional. When the multi-seed replicates run against a GP
    surrogate oracle (the cheap path), fitting that surrogate on the same
    conditions used to score the arms makes the test set a memorized set: the
    oracle reproduces it exactly, every arm's error collapses, and the
    active-vs-random gap is measured on a surface that has no held-out behaviour
    left. Disjoint by construction, checked below.
    """
    mat = condition_matrix(rows, space, model=model,
                           use_measured_rt60=use_measured_rt60)
    n = mat["n_conditions"]
    if n < min_test * 2:
        raise ValueError(
            f"only {n} usable conditions in the master table — too few to split "
            f"into an oracle-training half and a >= {min_test}-point held-out test "
            f"set. Run the main grid (R4.4) first."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(min_test, int(round(holdout_frac * n)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    assert not (set(test_idx.tolist()) & set(train_idx.tolist()))   # disjoint

    X, y = mat["X_raw"], mat["y"]
    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
        "conditions_test": [mat["conditions"][i] for i in test_idx],
        "conditions_train": [mat["conditions"][i] for i in train_idx],
        "n_conditions": n, "n_train": len(train_idx), "n_test": len(test_idx),
        "oracle_calls": 0,
        "source": "master results table (real measurements, already paid for)",
        "n_failed_rows": mat["n_failed_rows"], "model": model,
        "use_measured_rt60": use_measured_rt60,
    }


def test_set_from_master(rows: Sequence[Mapping],
                         space: FactorSpace = DEFAULT_FACTOR_SPACE, *,
                         model: str | None = None, holdout_frac: float = 0.4,
                         seed: int = 0, use_measured_rt60: bool = True,
                         threshold: float = DEFAULT_THRESHOLD,
                         band: float = DEFAULT_BAND) -> dict:
    """
    The held-out yardstick both arms are scored against, built from REAL measured
    rows instead of `active_learning.make_test_set`'s fresh oracle calls.

    Guards against the silent-nonsense case: `boundary_rmse` scores only the test
    points within `band` of the failure contour. If the measured grid never comes
    near that contour the metric is NaN for every arm and the comparison is
    vacuous — so we raise with the diagnosis instead of returning NaNs.
    """
    split = master_split_for_al(rows, space, model=model, holdout_frac=holdout_frac,
                                seed=seed, use_measured_rt60=use_measured_rt60)
    near = int(np.sum(np.abs(split["y_test"] - threshold) <= band))
    if near == 0:
        lo, hi = float(np.min(split["y_test"])), float(np.max(split["y_test"]))
        raise ValueError(
            f"no held-out condition lies within +-{band} of the failure threshold "
            f"{threshold} (measured WER spans [{lo:.3f}, {hi:.3f}]). boundary_rmse "
            f"would be NaN for every arm. Either the grid never crosses the "
            f"boundary (widen the conditions) or the threshold is wrong for this "
            f"model."
        )
    split["n_test_near_boundary"] = near
    split["threshold"] = threshold
    split["band"] = band
    return split


def surrogate_oracle_from_master(rows: Sequence[Mapping] | None = None,
                                 space: FactorSpace = DEFAULT_FACTOR_SPACE, *,
                                 split: Mapping | None = None,
                                 model: str | None = None, seed: int = 0,
                                 holdout_frac: float = 0.4,
                                 use_measured_rt60: bool = True,
                                 ) -> tuple[Callable[[dict], float], dict]:
    """
    A free, instant stand-in oracle for the multi-seed replicates: a GP fitted to
    the master table's TRAINING half. Signature-compatible with the real
    `make_pipeline_oracle` / `make_multiclip_oracle`, so the same `active_learn`
    call runs against either.

    This is a REPLICATION device, not a measurement device. Its output must always
    be labelled `oracle_provenance="surrogate"` downstream — see the module
    docstring on provenance.
    """
    if split is None:
        if rows is None:
            raise ValueError("pass either `rows` or a precomputed `split`")
        split = master_split_for_al(rows, space, model=model,
                                    holdout_frac=holdout_frac, seed=seed,
                                    use_measured_rt60=use_measured_rt60)
    gp = GPSurrogate(space, seed=seed).fit(split["X_train"], split["y_train"])

    def oracle(sample: Mapping) -> float:
        x = encode_sample(space, sample).reshape(1, -1)
        return float(np.asarray(gp.predict(x, return_std=False)).ravel()[0])

    meta = {"provenance": "surrogate", "n_train": split["n_train"],
            "n_test": split["n_test"], "oracle_calls_to_build": 0}
    return oracle, meta


# ============================================================================
# ONE SEED — three arms on ONE shared test set and ONE checkpoint schedule
# ============================================================================

def final_fidelity(traj: Trajectory, space: FactorSpace, X_test: np.ndarray,
                   y_test: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
                   band: float = DEFAULT_BAND, seed: int = 0) -> dict:
    """
    End-of-run fidelity for one arm: fit the surrogate on ALL of that arm's
    evaluations and score it on the held-out set with the same two metrics the
    learning curve uses (`boundary_rmse` headline, `boundary_error` for the
    intuitive "% of conditions mislabelled safe/unsafe" sentence).
    """
    gp = GPSurrogate(space, seed=seed).fit(traj.X_raw, traj.y)
    return {
        "n_evals": int(traj.n_evals),
        "boundary_rmse": boundary_rmse(gp, X_test, y_test, threshold, band),
        "boundary_error": boundary_error(gp, X_test, y_test, threshold),
    }


def run_seed(oracle: Callable[[dict], float], space: FactorSpace, seed: int, *,
             X_test: np.ndarray, y_test: np.ndarray,
             n_seed: int = 15, budget: int = 30, pool_size: int = 600,
             threshold: float = DEFAULT_THRESHOLD, band: float = DEFAULT_BAND,
             ) -> dict:
    """
    Run all three arms at ONE seed to the SAME oracle-call budget, scored on ONE
    shared held-out set at ONE shared checkpoint schedule.

    The held-out set is REQUIRED, and comes from `test_set_from_master` — i.e.
    real measurements already paid for. There is deliberately no "build me a fresh
    test set" path (`active_learning.make_test_set` costs hundreds of extra oracle
    calls per seed): the arms here burn exactly `n_seed + budget` calls each and
    not one more, which is what lets the report state `test_oracle_calls == 0`.
    """
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    n_total = n_seed + budget
    arms: dict[str, Trajectory] = {
        "active_boundary": active_learn(oracle, space, "boundary", n_seed, budget,
                                        pool_size, threshold, seed=seed),
        "active_uncertainty": active_learn(oracle, space, "uncertainty", n_seed,
                                           budget, pool_size, threshold, seed=seed),
        "random": random_baseline(oracle, space, n_total, seed=seed),
    }
    # ONE shared checkpoint schedule, exactly as compare_arms does it: otherwise
    # the passive arm is only ever scored at full budget and the race is unfair.
    checkpoints = list(range(n_seed, n_total + 1, 3))
    if checkpoints[-1] != n_total:
        checkpoints.append(n_total)
    curves = {name: learning_curve(t, space, X_test, y_test,
                                   checkpoints=checkpoints, threshold=threshold,
                                   band=band, seed=seed)
              for name, t in arms.items()}
    return {"seed": seed, "curves": curves, "n_total": n_total,
            "final": {name: final_fidelity(t, space, X_test, y_test, threshold,
                                           band, seed)
                      for name, t in arms.items()},
            "test_set_source": "master results table (held-out real measurements)",
            "test_oracle_calls": 0, "n_test_points": int(len(y_test))}


# ============================================================================
# MULTI-SEED — the band, which is the only presentable form
# ============================================================================

def multi_seed_curves(oracle: Callable[[dict], float],
                      space: FactorSpace = DEFAULT_FACTOR_SPACE, *,
                      X_test: np.ndarray, y_test: np.ndarray,
                      seeds: Sequence[int] = (0, 1, 2),
                      n_seed: int = 15, budget: int = 30, pool_size: int = 600,
                      threshold: float = DEFAULT_THRESHOLD, band: float = DEFAULT_BAND,
                      allow_single_seed: bool = False,
                      oracle_provenance: str = "surrogate",
                      api_confirmed_seeds: Sequence[int] = ()) -> dict:
    """
    Replicate the three-arm race across seeds. This is the entry point; there is
    deliberately no convenience wrapper that runs one seed and returns a headline.

    Fewer than MIN_SEEDS seeds raises `InsufficientSeedsError`. `allow_single_seed`
    exists only so that a smoke test / an end-to-end API confirmation run can
    execute the code path — it does not make the run presentable: the result comes
    back `evidence_level="INSUFFICIENT"` and every renderer downstream refuses to
    print a savings number from it.
    """
    seeds = list(seeds)
    if len(seeds) < MIN_SEEDS and not allow_single_seed:
        raise InsufficientSeedsError(
            f"{len(seeds)} seed(s) given; a savings claim needs >= {MIN_SEEDS} "
            f"(SPEC A.R4.7: seed variance in AL-vs-random is large, one seed is "
            f"not evidence). Pass allow_single_seed=True only for a smoke run — "
            f"the result will be stamped INSUFFICIENT and no headline will print."
        )
    enough = len(seeds) >= MIN_SEEDS

    per_seed = [run_seed(oracle, space, s, X_test=X_test, y_test=y_test,
                         n_seed=n_seed, budget=budget, pool_size=pool_size,
                         threshold=threshold, band=band)
                for s in seeds]

    unbacked = [s for s in seeds if s not in set(api_confirmed_seeds)]
    provenance = {
        "oracle": oracle_provenance,
        "api_confirmed_seeds": list(api_confirmed_seeds),
        "surrogate_only_seeds": unbacked,
        "statement": (
            f"{len(seeds)} seed(s) run against the {oracle_provenance} oracle; "
            + (f"seed(s) {list(api_confirmed_seeds)} additionally confirmed "
               f"end-to-end against the live API oracle."
               if api_confirmed_seeds else
               "NO seed was confirmed end-to-end against the live API oracle — "
               "say so in the write-up rather than letting the reader assume it.")
        ),
    }
    return {
        "seeds": seeds, "n_seeds": len(seeds), "per_seed": per_seed,
        "evidence_level": SUFFICIENT if enough else INSUFFICIENT,
        "min_seeds": MIN_SEEDS,
        "insufficient_reason": None if enough else _NOT_EVIDENCE.format(min_seeds=MIN_SEEDS),
        "n_total": per_seed[0]["n_total"],
        "test_set_source": per_seed[0]["test_set_source"],
        "test_oracle_calls": per_seed[0]["test_oracle_calls"],
        "threshold": threshold, "band": band, "metric": METRIC,
        "provenance": provenance,
    }


def seed_band(multi: Mapping, arm: str, metric: str = METRIC,
              monotone: bool = True) -> list[dict]:
    """
    Collapse the per-seed learning curves of one arm into the BAND that gets
    plotted: median with min/max across seeds at each oracle-call count.

    `monotone=True` runs each seed through `best_so_far` first — held-out error
    bounces as points are added, and "oracle calls to reach fidelity X" is only
    well defined on the cumulative-minimum view.

    Note what is NOT here: any notion of a best seed. The band is the claim.
    """
    curves = []
    for ps in multi["per_seed"]:
        c = ps["curves"][arm]
        curves.append(best_so_far(c, metric) if monotone else
                      [{"n_evals": p["n_evals"], metric: p[metric]} for p in c])
    xs = sorted({p["n_evals"] for c in curves for p in c})
    band = []
    for x in xs:
        vals: list[float] = []
        for c in curves:
            for p in c:
                # NaN means "no held-out point near the contour at this checkpoint";
                # it must not propagate into a median that then reads as a real value.
                if p["n_evals"] == x and not math.isnan(float(p[metric])):
                    vals.append(float(p[metric]))
        if not vals:
            continue
        band.append({
            "n_evals": int(x), "n_seeds": len(vals),
            "median": float(np.median(vals)),
            "lo": float(np.min(vals)), "hi": float(np.max(vals)),
            "values": [float(v) for v in vals],
        })
    return band


# ============================================================================
# THE HEADLINE — median evals-to-target, with the spread attached
# ============================================================================

def _median_or_inf(vals: Sequence[float]) -> float:
    """
    Median over the seeds, keeping `inf` (= "never reached the target within
    budget") in the sample rather than dropping it. Dropping non-reaching seeds
    would compute the median over only the runs that happened to succeed —
    survivorship bias in exactly the direction that flatters the method.
    """
    return float(np.median(np.asarray(list(vals), dtype=float)))


def savings_headline(multi: Mapping, target: float | None = None,
                     metric: str = METRIC, active_arm: str = ACTIVE_ARM,
                     passive_arm: str = PASSIVE_ARM) -> dict:
    """
    "Active reaches boundary RMSE <= X in N_active evals vs N_random — a P%
    reduction", computed as the MEDIAN across seeds with the full per-seed spread
    attached.

    `target` defaults to the median fidelity the PASSIVE arm reaches with its whole
    budget — the equal-accuracy-fewer-calls framing (same target for both arms, so
    the comparison can't be gamed by choosing a target one arm can't reach).

    An INSUFFICIENT (single-seed) run gets no sentence: the field is replaced by
    the not-evidence string. That is the entire point of the flag.
    """
    per_seed_final = {arm: [] for arm in (active_arm, passive_arm)}
    for arm in (active_arm, passive_arm):
        for ps in multi["per_seed"]:
            c = best_so_far(ps["curves"][arm], metric)
            per_seed_final[arm].append(c[-1][metric])

    if target is None:
        finite = [v for v in per_seed_final[passive_arm] if math.isfinite(v)]
        if not finite:
            # Every passive seed ended NaN => no held-out point ever sat near the
            # contour. Silently np.median([]) -> nan would print "target nan" and
            # read like a real (if odd) number; refuse instead.
            raise ValueError(
                f"the {passive_arm} arm's final {metric} is NaN for all "
                f"{multi['n_seeds']} seeds — no held-out condition lies within "
                f"+-{multi['band']} of threshold {multi['threshold']}, so there is "
                f"no target to compare against. Widen the band or the grid."
            )
        target = float(np.median(finite))

    per_seed_evals = {arm: [] for arm in (active_arm, passive_arm)}
    for arm in (active_arm, passive_arm):
        for ps in multi["per_seed"]:
            c = best_so_far(ps["curves"][arm], metric)
            per_seed_evals[arm].append(evals_to_target(c, target, metric))

    med_a = _median_or_inf(per_seed_evals[active_arm])
    med_p = _median_or_inf(per_seed_evals[passive_arm])
    reached_a = sum(1 for v in per_seed_evals[active_arm] if math.isfinite(v))
    reached_p = sum(1 for v in per_seed_evals[passive_arm] if math.isfinite(v))
    reduction = (100.0 * (med_p - med_a) / med_p
                 if math.isfinite(med_a) and math.isfinite(med_p) and med_p > 0
                 else float("nan"))

    def _rng(vals):
        f = [v for v in vals if math.isfinite(v)]
        return (min(f), max(f)) if f else (float("nan"), float("nan"))

    a_lo, a_hi = _rng(per_seed_evals[active_arm])
    p_lo, p_hi = _rng(per_seed_evals[passive_arm])

    sufficient = multi["evidence_level"] == SUFFICIENT
    if not sufficient:
        sentence = multi["insufficient_reason"]
    elif not math.isfinite(med_a) or not math.isfinite(med_p):
        sentence = (
            f"No savings claim: the {metric} target {target:.3f} was reached by "
            f"{reached_a}/{multi['n_seeds']} active seeds and "
            f"{reached_p}/{multi['n_seeds']} random seeds within the "
            f"{multi['n_total']}-evaluation budget. Report the budget, not a ratio."
        )
    else:
        sentence = (
            f"Straddle-acquisition active learning reaches {metric} <= {target:.3f} "
            f"in {med_a:.0f} oracle evaluations vs {med_p:.0f} for random sampling "
            f"— a {reduction:.0f}% reduction in oracle calls (median over "
            f"{multi['n_seeds']} seeds; active range {a_lo:.0f}-{a_hi:.0f}, random "
            f"range {p_lo:.0f}-{p_hi:.0f}). {multi['provenance']['statement']}"
        )

    return {
        "metric": metric, "target": float(target),
        "active_arm": active_arm, "passive_arm": passive_arm,
        "n_seeds": multi["n_seeds"], "evidence_level": multi["evidence_level"],
        "median_evals": {active_arm: med_a, passive_arm: med_p},
        "per_seed_evals": {k: [float(v) for v in v_] for k, v_ in per_seed_evals.items()},
        "range_evals": {active_arm: [a_lo, a_hi], passive_arm: [p_lo, p_hi]},
        "per_seed_final_fidelity": {k: [float(v) for v in v_]
                                    for k, v_ in per_seed_final.items()},
        "n_seeds_reaching_target": {active_arm: reached_a, passive_arm: reached_p},
        "pct_reduction": float(reduction),
        "sentence": sentence,
        "presentable": bool(sufficient),
        "provenance": multi["provenance"],
    }


# ============================================================================
# THE REPORT
# ============================================================================

def al_savings_report(multi: Mapping, target: float | None = None,
                      metric: str = METRIC) -> dict:
    """Bands for every arm + the headline. The only object the write-up needs."""
    bands = {arm: seed_band(multi, arm, metric) for arm in multi["per_seed"][0]["curves"]}
    err_bands = {arm: seed_band(multi, arm, "boundary_error")
                 for arm in multi["per_seed"][0]["curves"]}
    return {
        "bands": bands,                     # headline metric (boundary_rmse)
        "boundary_error_bands": err_bands,  # the intuitive % -mislabelled view
        "headline": savings_headline(multi, target=target, metric=metric),
        "metric": metric, "n_seeds": multi["n_seeds"], "seeds": multi["seeds"],
        "evidence_level": multi["evidence_level"],
        "insufficient_reason": multi["insufficient_reason"],
        "n_total": multi["n_total"], "threshold": multi["threshold"],
        "band_halfwidth": multi["band"],
        "test_set_source": multi["test_set_source"],
        "test_oracle_calls": multi["test_oracle_calls"],
        "provenance": multi["provenance"],
    }


def format_al_savings(res: Mapping) -> str:
    """Printed report. An INSUFFICIENT run prints a banner where the number goes."""
    out = []
    bar = "=" * 78
    if res["evidence_level"] == INSUFFICIENT:
        out += ["!" * 78,
                f"INSUFFICIENT EVIDENCE — {res['n_seeds']} seed(s), need "
                f"{MIN_SEEDS}. No savings number is reported below.",
                f"  {res['insufficient_reason']}",
                "!" * 78, ""]
    else:
        out += [bar, "ACTIVE-LEARNING SAVINGS (D3b)", bar, "",
                res["headline"]["sentence"], ""]

    h = res["headline"]
    out += [f"metric={res['metric']}  threshold={res['threshold']}  "
            f"band=+-{res['band_halfwidth']}  budget={res['n_total']} evals/arm",
            f"test set: {res['test_set_source']} "
            f"({res['test_oracle_calls']} oracle calls to build)",
            f"provenance: {res['provenance']['statement']}",
            ""]

    arms = [a for a in ARMS if a in res["bands"]]
    out += ["-" * 78,
            f"{res['metric']} vs oracle calls — MEDIAN [min-max] across "
            f"{res['n_seeds']} seed(s)",
            "-" * 78]
    header = f"  {'n_evals':>7} | " + " | ".join(f"{a:>26}" for a in arms)
    out.append(header)
    xs = sorted({p["n_evals"] for a in arms for p in res["bands"][a]})
    lookup = {a: {p["n_evals"]: p for p in res["bands"][a]} for a in arms}
    for x in xs:
        cells = []
        for a in arms:
            p = lookup[a].get(x)
            cells.append(f"{p['median']:>8.3f} [{p['lo']:.3f}-{p['hi']:.3f}]"
                         if p else " " * 26)
        out.append(f"  {x:>7} | " + " | ".join(f"{c:>26}" for c in cells))

    if res["evidence_level"] != INSUFFICIENT:
        # NB: the per-seed list is printed for transparency, NOT so a favourable
        # seed can be quoted. The band/median is the claim; no single-seed number
        # is ever presented as the headline (and none is computed as such).
        out += ["", f"per-seed evals to target {h['target']:.3f} "
                    f"(the BAND is the claim — the median, never one seed):"]
        for arm, vals in h["per_seed_evals"].items():
            pretty = ", ".join("inf" if not math.isfinite(v) else f"{v:.0f}"
                               for v in vals)
            med = h["median_evals"][arm]
            med_s = f"{med:.0f}" if math.isfinite(med) else "inf"
            out.append(f"  {arm:<20} [{pretty}]   median={med_s}")
    return "\n".join(out)


def plot_payload(res: Mapping) -> dict:
    """
    JSON-safe payload for the dashboard (E2). STABLE SHAPE — a parallel consumer
    depends on these exact keys; add, never rename or remove:

      metric            str    — the fidelity metric plotted ("boundary_rmse")
      threshold         float  — WER value defining the failure contour
      band_halfwidth    float  — half-width of the scored band around it
      n_seeds           int
      seeds             list[int]
      evidence_level    "SUFFICIENT" | "INSUFFICIENT"
      series            list of one dict PER ARM, each:
                          arm      str  ("active_boundary"|"active_uncertainty"|"random")
                          n_evals  list[int]    — x axis, shared checkpoint grid
                          median   list[float]  — the band centre (THE claim)
                          lo, hi   list[float]  — min/max across seeds, not a CI
                          n_seeds  list[int]    — seeds contributing at each x
                        all five lists are the same length as n_evals.
      target            float  — the equal-fidelity target the headline is quoted at
      headline          str    — publishable sentence, OR the not-evidence string
      presentable       bool   — FALSE means: do not render the headline as a claim
      provenance        dict   — {oracle, api_confirmed_seeds, surrogate_only_seeds,
                                  statement}; `statement` must be shown wherever
                                  the headline is shown.
    """
    return {
        "metric": res["metric"], "threshold": res["threshold"],
        "band_halfwidth": res["band_halfwidth"],
        "n_seeds": res["n_seeds"], "seeds": list(res["seeds"]),
        "evidence_level": res["evidence_level"],
        "series": [
            {"arm": arm,
             "n_evals": [p["n_evals"] for p in band],
             "median": [p["median"] for p in band],
             "lo": [p["lo"] for p in band],
             "hi": [p["hi"] for p in band],
             "n_seeds": [p["n_seeds"] for p in band]}
            for arm, band in res["bands"].items()
        ],
        "target": res["headline"]["target"],
        "headline": res["headline"]["sentence"],
        "presentable": res["headline"]["presentable"],
        "provenance": res["provenance"],
    }


# ============================================================================
# CLI
# ============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="D3b — active-learning savings curve from the master table")
    ap.add_argument("master", nargs="?", default="results/master.csv")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-seed", type=int, default=15)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--holdout-frac", type=float, default=0.4)
    ap.add_argument("--api-confirmed-seeds", type=int, nargs="*", default=[],
                    help="seeds that were ALSO run end-to-end against the live oracle")
    ap.add_argument("--allow-single-seed", action="store_true",
                    help="smoke path only; the result is stamped INSUFFICIENT")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    rows = load_master_rows(args.master)
    split = test_set_from_master(rows, model=args.model,
                                 holdout_frac=args.holdout_frac, seed=0)
    oracle, _ = surrogate_oracle_from_master(split=split, model=args.model, seed=0)
    multi = multi_seed_curves(oracle, DEFAULT_FACTOR_SPACE,
                              X_test=split["X_test"], y_test=split["y_test"],
                              seeds=args.seeds, n_seed=args.n_seed,
                              budget=args.budget,
                              allow_single_seed=args.allow_single_seed,
                              api_confirmed_seeds=args.api_confirmed_seeds)
    res = al_savings_report(multi)
    print(format_al_savings(res))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(plot_payload(res), f, indent=2)
        print(f"\n[wrote] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
