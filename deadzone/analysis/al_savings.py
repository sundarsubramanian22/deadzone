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

    python3 -m deadzone.analysis.al_savings results/master.csv --seeds 0 1 2

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

# Allow `python deadzone/analysis/al_savings.py` as well as `import deadzone.analysis.al_savings`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.design import DEFAULT_FACTOR_SPACE, FactorSpace          # noqa: E402
from deadzone.active_learning import (                                  # noqa: E402
    DEFAULT_BAND, DEFAULT_THRESHOLD, GPSurrogate, Trajectory, active_learn,
    best_so_far, boundary_error, boundary_rmse, evals_to_target,
    learning_curve, random_baseline,
)
# D3a owns the master-table -> factor-matrix conversion; reused, not duplicated.
from deadzone.analysis.interactions import (                            # noqa: E402
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
    # The evaluated points themselves, in acquisition order. Kept because the
    # curve alone cannot show WHERE the active arm spent its calls — and "it
    # concentrated them near the contour" is the mechanism behind whatever the
    # savings number turns out to be (including a null one). Serialization-ready.
    trajectories = {
        name: {
            "strategy": t.strategy, "n_seed": int(t.n_seed),
            "n_evals": int(t.n_evals),
            "samples": t.samples(space),
            "y": [float(v) for v in t.y],
            "steps": [dict(s) for s in t.steps],
        }
        for name, t in arms.items()
    }
    return {"seed": seed, "curves": curves, "n_total": n_total,
            "final": {name: final_fidelity(t, space, X_test, y_test, threshold,
                                           band, seed)
                      for name, t in arms.items()},
            "trajectories": trajectories,
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
    # TRAP 1's blind spot. The seed count is checked below, but nothing checked
    # that the seeds are DISTINCT — and `active_learn`/`random_baseline` are
    # deterministic in the seed, so `seeds=(0, 0, 0)` runs the same trajectory
    # three times and passes the MIN_SEEDS gate. The result is a zero-width seed
    # band, an exactly-reproducible "median over 3 seeds", and a headline that
    # looks MORE robust than a genuine three-seed run because the spread
    # vanished. That is the single-seed anecdote this module exists to prevent,
    # wearing the band's clothes.
    if not seeds:
        raise InsufficientSeedsError(
            "no seeds given: every downstream aggregation indexes per_seed[0] "
            "or medians an empty list. Pass at least one (and >= "
            f"{MIN_SEEDS} for a presentable result).")
    dupes = sorted({s for s in seeds if seeds.count(s) > 1})
    if dupes:
        raise InsufficientSeedsError(
            f"duplicate seed(s) {dupes} in {seeds}: the arms are deterministic "
            f"in the seed, so each repeat re-runs an IDENTICAL trajectory. The "
            f"band would collapse to zero width and print as 'median over "
            f"{len(seeds)} seeds' — a single-seed anecdote presented as the "
            f"strongest possible form of the claim. Pass distinct seeds.")
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

def target_sweep(multi: Mapping, targets: Sequence[float] | None = None,
                 metric: str = METRIC, active_arm: str = ACTIVE_ARM,
                 passive_arm: str = PASSIVE_ARM, n_targets: int = 12) -> list[dict]:
    """
    Median evals-to-target for BOTH arms across a whole range of fidelity
    targets, not just the one the headline quotes.

    Why this exists: "evals to reach X" is a function of X, and the reduction
    percentage can swing from large to negative as X moves. Reporting a single
    target invites (and is indistinguishable from) picking the flattering one.
    The default target in `savings_headline` is chosen by a rule fixed in
    advance (the passive arm's own final fidelity), and this sweep is the
    evidence that the rule was not shopped: the reader sees the whole curve.

    Targets default to a linear span from the best fidelity ANY arm reached to
    the worst final fidelity of the passive arm.
    """
    curves = {arm: [best_so_far(ps["curves"][arm], metric)
                    for ps in multi["per_seed"]]
              for arm in (active_arm, passive_arm)}
    if targets is None:
        finals = [c[-1][metric] for arm in curves for c in curves[arm]]
        finite = sorted(v for v in finals if math.isfinite(v))
        if not finite:
            return []
        lo, hi = finite[0], finite[-1]
        if hi <= lo:
            targets = [lo]
        else:
            targets = list(np.linspace(lo, hi, n_targets))
    out = []
    for t in targets:
        row = {"target": float(t)}
        for arm in (active_arm, passive_arm):
            ev = [evals_to_target(c, float(t), metric) for c in curves[arm]]
            row[arm] = _median_or_inf(ev)
            row[f"{arm}_reached"] = sum(1 for v in ev if math.isfinite(v))
        a, p = row[active_arm], row[passive_arm]
        row["pct_reduction"] = (100.0 * (p - a) / p
                                if math.isfinite(a) and math.isfinite(p) and p > 0
                                else float("nan"))
        out.append(row)
    return out


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
        # the whole target curve, so the headline's single target can be checked
        # against every other one rather than taken on faith
        "target_sweep": target_sweep(multi, metric=metric),
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

        sweep = res.get("target_sweep") or []
        if sweep:
            # Printed ALWAYS, not only when it flatters: "evals to reach X" is a
            # function of X, so one target is unfalsifiable. This is the curve.
            out += ["", "-" * 78,
                    "THE WHOLE TARGET CURVE (the headline quotes ONE row of this; "
                    "the rest is here so it can't be cherry-picked)",
                    "-" * 78,
                    f"  {'target':>8} | {'active':>8} | {'random':>8} | "
                    f"{'reduction':>10} | reached (act/rand of "
                    f"{res['n_seeds']})"]
            a_arm = h["active_arm"]
            p_arm = h["passive_arm"]
            for r in sweep:
                def _f(v):
                    return f"{v:.0f}" if math.isfinite(v) else "inf"
                red = (f"{r['pct_reduction']:+.0f}%"
                       if math.isfinite(r["pct_reduction"]) else "n/a")
                mark = " <- headline" if abs(r["target"] - h["target"]) < 1e-9 else ""
                out.append(
                    f"  {r['target']:>8.3f} | {_f(r[a_arm]):>8} | {_f(r[p_arm]):>8} | "
                    f"{red:>10} | {r[a_arm + '_reached']}/{r[p_arm + '_reached']}{mark}")
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
# THE TWO THINGS THAT MAKE A NULL RESULT READABLE
# ============================================================================
#
# A null ("active did not beat random") is a legitimate outcome and SPEC A.R5.5
# says to report it plainly. But an unqualified null is unreadable: it does not
# distinguish "boundary-seeking has nothing to exploit on this surface" from
# "the yardstick had no resolution left" from "we got one unlucky split". The
# two functions below supply exactly those distinctions, and the report prints
# them next to the headline whichever way the headline goes.


def oracle_fidelity_floor(split: Mapping, space: FactorSpace = DEFAULT_FACTOR_SPACE,
                          seed: int = 0, threshold: float = DEFAULT_THRESHOLD,
                          band: float = DEFAULT_BAND) -> dict:
    """
    THE FLOOR: fit the surrogate to ALL the training conditions at once and score
    it on the held-out set. No arm working from a subset of oracle calls can be
    expected to do better, so this is the fidelity the race is running toward.

    Why it belongs beside the headline. If the floor sits at or above the
    fidelity both arms reach inside the budget, then the arms are separated by
    less than the yardstick can resolve and "active did not win" says nothing
    about acquisition — the comparison simply had no headroom. Reporting a null
    without the floor invites the reader to conclude the method is useless when
    the honest conclusion is that this surface cannot answer the question.
    """
    gp = GPSurrogate(space, seed=seed).fit(split["X_train"], split["y_train"])
    Xte = np.asarray(split["X_test"], dtype=float)
    yte = np.asarray(split["y_test"], dtype=float)
    keep = np.abs(yte - threshold) <= band
    const = (float(np.sqrt(np.mean((threshold - yte[keep]) ** 2)))
             if np.any(keep) else float("nan"))
    return {
        "boundary_rmse": boundary_rmse(gp, Xte, yte, threshold, band),
        "boundary_error": boundary_error(gp, Xte, yte, threshold),
        "global_rmse": float(np.sqrt(np.mean(
            (np.asarray(gp.predict(Xte, return_std=False)).ravel() - yte) ** 2))),
        "y_test_sd": float(np.std(yte)),
        "constant_threshold_predictor_rmse": const,
        "n_train_conditions": int(len(split["y_train"])),
        "n_test_near_boundary": int(np.sum(keep)),
        "note": ("surrogate fitted to ALL training conditions and scored on the "
                 "held-out real measurements — the best boundary fidelity any arm "
                 "could reach with unlimited calls to this oracle."),
    }


def paired_arm_difference(multi: Mapping, metric: str = METRIC,
                          active_arm: str = ACTIVE_ARM,
                          passive_arm: str = PASSIVE_ARM) -> dict:
    """
    Final-fidelity difference (active - passive) PAIRED within each seed, plus a
    sign count.

    Paired, because the two arms share a seed and therefore share the random
    draw that dominates the variance; comparing unpaired medians across a band
    this wide would call anything a tie. Negative difference = active is better
    (lower RMSE). The sign count is the readable summary: "active won k of n
    seeds" is a claim a reader can check, where "the medians differ by 0.03" is
    not, given bands of +-0.09.
    """
    diffs = []
    for ps in multi["per_seed"]:
        a = best_so_far(ps["curves"][active_arm], metric)[-1][metric]
        p = best_so_far(ps["curves"][passive_arm], metric)[-1][metric]
        if math.isfinite(a) and math.isfinite(p):
            diffs.append({"seed": ps["seed"], "active": float(a),
                          "passive": float(p), "diff": float(a - p)})
    d = [x["diff"] for x in diffs]
    n_win = sum(1 for v in d if v < 0)
    return {
        "metric": metric, "active_arm": active_arm, "passive_arm": passive_arm,
        "per_seed": diffs, "n_paired": len(d),
        "median_diff": float(np.median(d)) if d else float("nan"),
        "min_diff": float(np.min(d)) if d else float("nan"),
        "max_diff": float(np.max(d)) if d else float("nan"),
        "n_active_better": n_win,
        "n_passive_better": len(d) - n_win,
        "sign_straddles_zero": bool(d and min(d) < 0 < max(d)),
        "convention": "diff = active - passive; NEGATIVE means active is better",
    }


def acquisition_concentration(multi: Mapping) -> dict:
    """
    What fraction of each arm's evaluations landed NEAR the failure contour.

    This separates the two ways a savings claim can come out null, which no
    fidelity curve can tell apart:

      * the acquisition function did not do its job — its evaluations are no
        more concentrated near the contour than random's, so something is
        broken or mis-parameterised; or
      * it did its job and the job did not pay — the evaluations ARE
        concentrated and the held-out fidelity still did not improve, which
        means the boundary was not the scarce information on this surface.

    `random_frac_near_threshold` doubles as a measure of how WIDE the boundary
    region is: if a uniform draw already lands in the band a third of the time,
    the contour is a thick slab rather than a thin sheet, and there is little
    for a boundary-seeker to find that random sampling would have missed.
    """
    thr, band = multi["threshold"], multi["band"]
    out: dict = {"threshold": thr, "band_halfwidth": band, "arms": {}}
    for arm in multi["per_seed"][0]["trajectories"]:
        all_fr, acq_fr = [], []
        for ps in multi["per_seed"]:
            t = ps["trajectories"][arm]
            y = np.asarray(t["y"], dtype=float)
            if len(y):
                all_fr.append(float(np.mean(np.abs(y - thr) <= band)))
            tail = y[t["n_seed"]:]      # the ACQUIRED points, past the LHS seed set
            if len(tail):
                acq_fr.append(float(np.mean(np.abs(tail - thr) <= band)))
        out["arms"][arm] = {
            "frac_all_evals_near_threshold": float(np.median(all_fr)) if all_fr else float("nan"),
            "frac_acquired_evals_near_threshold": float(np.median(acq_fr)) if acq_fr else float("nan"),
            "n_seeds": len(all_fr),
        }
    a = out["arms"].get(ACTIVE_ARM, {}).get("frac_acquired_evals_near_threshold", float("nan"))
    p = out["arms"].get(PASSIVE_ARM, {}).get("frac_all_evals_near_threshold", float("nan"))
    out["random_frac_near_threshold"] = p
    out["active_acquired_frac_near_threshold"] = a
    out["acquisition_worked"] = bool(math.isfinite(a) and math.isfinite(p) and a > p)
    out["interpretation"] = (
        f"straddle put {a:.0%} of its acquired evaluations within +-{band} of the "
        f"{thr} contour vs {p:.0%} for uniform random"
        + (" — the acquisition function did what it is supposed to do, so any null "
           "in the fidelity curve is about the surface, not about the acquisition."
           if out["acquisition_worked"] else
           " — the acquisition function is NOT concentrating near the contour; "
           "investigate before reading anything into the fidelity curve.")
    )
    return out


def split_robustness(rows: Sequence[Mapping], space: FactorSpace = DEFAULT_FACTOR_SPACE,
                     *, model: str | None = None,
                     split_seeds: Sequence[int] = (0, 1, 2, 3),
                     seeds: Sequence[int] = (0, 1, 2, 3, 4),
                     n_seed: int = 15, budget: int = 30, holdout_frac: float = 0.4,
                     threshold: float = DEFAULT_THRESHOLD, band: float = DEFAULT_BAND,
                     use_measured_rt60: bool = True) -> dict:
    """
    Re-run the whole multi-seed race under several DIFFERENT train/test splits of
    the measured conditions, and report whether the winner is stable.

    This is the check that keeps a null (or a win) from resting on one draw. The
    held-out set here is ~70 conditions of which only ~12 sit near the failure
    contour, so a single split's `boundary_rmse` is a 12-point statistic; which
    12 conditions land in the test half is itself a large source of variance,
    entirely separate from the AL seed variance the seed band already covers.

    Reports every split, never a selected one. Zero oracle calls throughout.
    """
    per_split = []
    for ss in split_seeds:
        split = test_set_from_master(rows, space, model=model,
                                     holdout_frac=holdout_frac, seed=ss,
                                     use_measured_rt60=use_measured_rt60,
                                     threshold=threshold, band=band)
        # `space` MUST be forwarded: the oracle factory encodes each sample with
        # `encode_sample(space, ...)`, so letting it fall back to
        # DEFAULT_FACTOR_SPACE would fit the surrogate oracle in a different
        # coordinate system from the one the arms are sampling and being scored
        # in. With the default space that is a no-op, which is why it went
        # unnoticed; with any custom space whose factor NAMES still match it is
        # silently wrong rather than a KeyError.
        oracle, _ = surrogate_oracle_from_master(split=split, space=space,
                                                 model=model, seed=ss)
        multi = multi_seed_curves(oracle, space, X_test=split["X_test"],
                                  y_test=split["y_test"], seeds=seeds,
                                  n_seed=n_seed, budget=budget,
                                  threshold=threshold, band=band)
        finals = {}
        for arm in multi["per_seed"][0]["curves"]:
            v = [best_so_far(ps["curves"][arm], METRIC)[-1][METRIC]
                 for ps in multi["per_seed"]]
            v = [x for x in v if math.isfinite(x)]
            finals[arm] = {"median": float(np.median(v)) if v else float("nan"),
                           "lo": float(np.min(v)) if v else float("nan"),
                           "hi": float(np.max(v)) if v else float("nan"),
                           "n": len(v)}
        best = min((a for a in finals if math.isfinite(finals[a]["median"])),
                   key=lambda a: finals[a]["median"], default=None)
        per_split.append({
            "split_seed": ss,
            "n_train": split["n_train"], "n_test": split["n_test"],
            "n_test_near_boundary": split["n_test_near_boundary"],
            "floor": oracle_fidelity_floor(split, space, seed=ss,
                                           threshold=threshold, band=band),
            "final_fidelity": finals,
            "paired": paired_arm_difference(multi),
            "lowest_median_arm": best,
        })

    wins = [p["lowest_median_arm"] for p in per_split]
    n_act = sum(1 for w in wins if w == ACTIVE_ARM)
    all_diffs = [d["diff"] for p in per_split for d in p["paired"]["per_seed"]]
    n_pair_act = sum(1 for v in all_diffs if v < 0)
    stable = len(set(wins)) == 1
    # `paired_arm_difference` keeps only the seeds where BOTH arms ended finite,
    # so this list can come back short — or empty, if every seed's final RMSE was
    # NaN. `np.median([])` is nan with a RuntimeWarning, and the verdict sentence
    # interpolates it directly: "median paired difference nan" reads as a
    # measured tie rather than as no measurement at all. The field below already
    # guards; the SENTENCE, which is what gets quoted, did not.
    n_expected = len(split_seeds) * len(seeds)
    if not all_diffs:
        raise ValueError(
            f"no paired runs survived across {len(split_seeds)} split(s) x "
            f"{len(seeds)} seed(s) (expected {n_expected}): every seed ended "
            f"with a non-finite final {METRIC} on both arms, so there is nothing "
            f"to compare. The verdict would report 'median paired difference "
            f"nan', which reads as a measured tie. Usual cause: no held-out "
            f"condition sits within +-{band} of threshold {threshold}.")
    if len(all_diffs) < n_expected:
        print(f"[split_robustness] NOTE: {len(all_diffs)} of {n_expected} "
              f"paired runs are comparable; {n_expected - len(all_diffs)} seed(s) "
              f"ended with a non-finite final {METRIC} on one or both arms and "
              f"are excluded from the median paired difference.")
    verdict = (
        f"Across {len(split_seeds)} independent train/test splits x "
        f"{len(seeds)} AL seeds ({len(all_diffs)} paired runs), "
        f"{ACTIVE_ARM} finished with the lower median {METRIC} in {n_act}/"
        f"{len(split_seeds)} splits and beat {PASSIVE_ARM} in {n_pair_act}/"
        f"{len(all_diffs)} paired runs (median paired difference "
        f"{np.median(all_diffs):+.3f}, negative = active better). "
        + ("The winner is STABLE across splits."
           if stable else
           "The winner FLIPS between splits, so the difference is not resolvable "
           "at this test-set size: report a null, not a direction.")
    )
    return {
        "split_seeds": list(split_seeds), "al_seeds": list(seeds),
        "per_split": per_split,
        "winner_by_split": wins,
        "n_splits_won_by_active": n_act,
        "n_paired_runs": len(all_diffs),
        "n_paired_won_by_active": n_pair_act,
        "median_paired_diff": float(np.median(all_diffs)) if all_diffs else float("nan"),
        "winner_stable_across_splits": stable,
        "verdict": verdict,
        "oracle_calls": 0,
    }


def format_split_robustness(rob: Mapping) -> str:
    """The robustness table, printed under the headline whichever way it went."""
    out = ["-" * 78,
           "SPLIT-SEED ROBUSTNESS — is the winner stable, or an artifact of one "
           "train/test draw?",
           "-" * 78,
           f"  {'split':>5} {'nNear':>5} {'floor':>7} | "
           f"{'active_boundary':>21} {'random':>21} | {'act-rnd':>8}  winner"]
    for p in rob["per_split"]:
        f = p["final_fidelity"]
        def cell(a):
            d = f.get(a, {})
            return (f"{d.get('median', float('nan')):.3f}"
                    f"[{d.get('lo', float('nan')):.3f}-{d.get('hi', float('nan')):.3f}]")
        out.append(
            f"  {p['split_seed']:>5} {p['n_test_near_boundary']:>5} "
            f"{p['floor']['boundary_rmse']:>7.3f} | {cell(ACTIVE_ARM):>21} "
            f"{cell(PASSIVE_ARM):>21} | {p['paired']['median_diff']:>+8.3f}  "
            f"{p['lowest_median_arm']}")
    out += ["", f"  {rob['verdict']}",
            "",
            "  floor = surrogate fitted to ALL training conditions, scored on the "
            "same held-out set:",
            "  no arm can be expected to beat it, so it is the resolution limit of "
            "this comparison."]
    return "\n".join(out)


# ============================================================================
# PERSISTENCE — the four artifacts SPEC A.R4.7 / A.R5.5 name by filename
# ============================================================================

def _json_safe(obj):
    """
    Recursively replace non-finite floats with None.

    `inf` in `per_seed_evals` is meaningful — "this seed never reached the
    target within budget" — and `_median_or_inf` deliberately keeps it in the
    sample. But `json.dump` writes it as the bare token `Infinity`, which is not
    valid JSON and which every strict parser (including the browser's) rejects.
    Dropping those entries instead would be survivorship bias; encoding them as
    `null` keeps the entry, its position, and its meaning ("did not reach"),
    which is what the seed-band table is for.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, (np.floating, np.integer)):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def trajectory_payload(multi: Mapping) -> dict:
    """
    The per-seed, per-arm record of WHICH conditions each arm chose and what the
    oracle said — `results/al_trajectory.json` (SPEC A.R4.7).

    This is the object that lets a reader check the mechanism rather than take
    the headline on trust: if straddle acquisition works, its evaluated WERs
    should cluster around the threshold while random's spread across the whole
    range. If it doesn't work, that is visible here too, which is the point.
    """
    seeds = []
    for ps in multi["per_seed"]:
        arms = {}
        for arm, t in ps["trajectories"].items():
            y = [float(v) for v in t["y"]]
            near = [v for v in y
                    if abs(v - multi["threshold"]) <= multi["band"]]
            arms[arm] = {
                "strategy": t["strategy"], "n_seed": t["n_seed"],
                "n_evals": t["n_evals"],
                "samples": t["samples"], "oracle_wer": y,
                # the mechanism summary, precomputed so nobody has to eyeball it
                "frac_evals_near_threshold": (len(near) / len(y)) if y else float("nan"),
                "steps": t["steps"],
            }
        seeds.append({"seed": ps["seed"], "arms": arms})
    return _json_safe({
        "seeds": seeds,
        "threshold": multi["threshold"], "band_halfwidth": multi["band"],
        "n_total": multi["n_total"],
        "evidence_level": multi["evidence_level"],
        "provenance": multi["provenance"],
        "note": ("oracle_wer values come from the oracle named in `provenance`; "
                 "for the surrogate arm they are GP predictions fitted to the "
                 "master table's TRAINING half, not fresh transcriptions."),
    })


def savings_payload(res: Mapping, extra: Mapping | None = None) -> dict:
    """The whole report, JSON-safe — `results/al_savings.json`."""
    out = dict(res)
    if extra:
        out["run_config"] = dict(extra)
    return _json_safe(out)


def write_outputs(res: Mapping, multi: Mapping, out_dir: str = "results",
                  extra: Mapping | None = None) -> dict:
    """
    Write the four named artifacts and return {name: path}.

    al_trajectory.json — the evaluated points, per seed per arm
    al_curve.json      — the seed-band learning curves (dashboard payload shape)
    al_savings.json    — the full report incl. the headline and provenance
    al_savings.txt     — the printed report, verbatim
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "al_trajectory.json": trajectory_payload(multi),
        "al_curve.json": plot_payload(res),
        "al_savings.json": savings_payload(res, extra),
    }
    written = {}
    for name, payload in paths.items():
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
        written[name] = p
    p = os.path.join(out_dir, "al_savings.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(format_full_report(res) + "\n")
    written["al_savings.txt"] = p
    return written


def format_full_report(res: Mapping) -> str:
    """
    `format_al_savings` plus whatever diagnostics the run attached — the floor,
    the paired per-seed difference, the leakage check and the split-seed
    robustness table.

    These travel WITH the headline rather than in a separate file, because the
    headline is not interpretable without them: a savings number means little
    without knowing the floor it was measured against, and a null means little
    without knowing whether it survives a different train/test draw.
    """
    out = [format_al_savings(res)]
    lk = res.get("leakage_check")
    if lk:
        out += ["", "-" * 78, "LEAKAGE CHECK", "-" * 78,
                f"  {lk['n_train_conditions']} surrogate-oracle training conditions "
                f"vs {lk['n_test_conditions']} held-out "
                f"({lk['n_test_near_boundary']} near the contour); shared factor "
                f"vectors = {lk['shared_factor_vectors']} -> "
                f"{'DISJOINT' if lk['disjoint'] else 'LEAKING'}.",
                f"  method: {lk['how']}"]
    d = res.get("diagnostics") or {}
    if d.get("floor"):
        f = d["floor"]
        out += ["", "-" * 78,
                "FIDELITY FLOOR — the resolution limit of this comparison",
                "-" * 78,
                f"  surrogate fitted to ALL {f['n_train_conditions']} training "
                f"conditions, scored on the same held-out set:",
                f"    boundary_rmse = {f['boundary_rmse']:.3f}   "
                f"boundary_error = {f['boundary_error']:.3f}   "
                f"global_rmse = {f['global_rmse']:.3f} (test-set sd "
                f"{f['y_test_sd']:.3f})",
                f"  {f['note']}"]
    if d.get("acquisition_concentration"):
        ac = d["acquisition_concentration"]
        out += ["", "-" * 78,
                "WHERE THE CALLS WENT — did the acquisition function do its job?",
                "-" * 78]
        for arm, v in ac["arms"].items():
            out.append(f"  {arm:<20} {v['frac_all_evals_near_threshold']:>6.1%} of all "
                       f"evals / {v['frac_acquired_evals_near_threshold']:>6.1%} of "
                       f"ACQUIRED evals within +-{ac['band_halfwidth']} of "
                       f"{ac['threshold']}")
        out += ["", f"  {ac['interpretation']}"]
    if d.get("paired_final_difference"):
        p = d["paired_final_difference"]
        out += ["", f"  paired final {p['metric']} (active - random; NEGATIVE = "
                    f"active better): median {p['median_diff']:+.3f} "
                    f"[{p['min_diff']:+.3f}, {p['max_diff']:+.3f}]",
                f"  active better in {p['n_active_better']}/{p['n_paired']} seeds"
                + ("  — the sign straddles zero across seeds."
                   if p["sign_straddles_zero"] else "")]
    if res.get("split_robustness"):
        out += ["", format_split_robustness(res["split_robustness"])]
    return "\n".join(out)


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
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed for the train/test split of the measured conditions")
    ap.add_argument("--split-seeds", type=int, nargs="*", default=None,
                    help="ALSO re-run the race under these train/test splits and "
                         "report whether the winner is stable (robustness check)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="write al_trajectory.json, al_curve.json, al_savings.json "
                         "and al_savings.txt here (SPEC A.R4.7 / A.R5.5)")
    args = ap.parse_args(argv)

    rows = load_master_rows(args.master)
    split = test_set_from_master(rows, model=args.model,
                                 holdout_frac=args.holdout_frac,
                                 seed=args.split_seed)
    oracle, om = surrogate_oracle_from_master(split=split, model=args.model,
                                              seed=args.split_seed)
    # Leakage check, printed every run: the surrogate oracle must never have been
    # fitted at a condition the arms are scored on, or both arms look perfect and
    # the comparison is dead. Structural (index split) + verified here on the
    # actual matrices, because "disjoint by construction" is a claim, not a check.
    tr = {tuple(np.round(r, 9)) for r in split["X_train"]}
    te = {tuple(np.round(r, 9)) for r in split["X_test"]}
    overlap = len(tr & te)
    assert overlap == 0, f"train/test leakage: {overlap} shared factor vectors"

    multi = multi_seed_curves(oracle, DEFAULT_FACTOR_SPACE,
                              X_test=split["X_test"], y_test=split["y_test"],
                              seeds=args.seeds, n_seed=args.n_seed,
                              budget=args.budget,
                              allow_single_seed=args.allow_single_seed,
                              api_confirmed_seeds=args.api_confirmed_seeds)
    res = al_savings_report(multi)
    floor = oracle_fidelity_floor(split, DEFAULT_FACTOR_SPACE, seed=args.split_seed)
    paired = paired_arm_difference(multi)
    res["diagnostics"] = {"floor": floor, "paired_final_difference": paired,
                          "acquisition_concentration": acquisition_concentration(multi)}
    res["leakage_check"] = {
        "n_train_conditions": split["n_train"], "n_test_conditions": split["n_test"],
        "shared_factor_vectors": overlap, "disjoint": overlap == 0,
        "n_test_near_boundary": split["n_test_near_boundary"],
        "how": ("index split of the per-condition matrix, then verified on the "
                "actual factor matrices (rounded to 9dp) — construction AND check"),
    }

    print(format_al_savings(res))
    print(f"\nleakage check: {split['n_train']} training conditions vs "
          f"{split['n_test']} held-out ({split['n_test_near_boundary']} within "
          f"+-{split['band']} of threshold {split['threshold']}); "
          f"shared factor vectors = {overlap} — DISJOINT.")
    print(f"fidelity floor (surrogate on ALL {floor['n_train_conditions']} training "
          f"conditions, scored on the same held-out set): {METRIC} = "
          f"{floor['boundary_rmse']:.3f}, boundary_error = "
          f"{floor['boundary_error']:.3f}. No arm can be expected to beat this.")
    print(f"paired final {METRIC} (active - random, negative = active better): "
          f"median {paired['median_diff']:+.3f} "
          f"[{paired['min_diff']:+.3f}, {paired['max_diff']:+.3f}]; active better in "
          f"{paired['n_active_better']}/{paired['n_paired']} seeds.")

    if args.split_seeds:
        rob = split_robustness(rows, DEFAULT_FACTOR_SPACE, model=args.model,
                               split_seeds=args.split_seeds, seeds=args.seeds,
                               n_seed=args.n_seed, budget=args.budget,
                               holdout_frac=args.holdout_frac)
        res["split_robustness"] = rob
        print()
        print(format_split_robustness(rob))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(plot_payload(res), f, indent=2)
        print(f"[wrote] {args.json_out}")
    if args.out_dir:
        extra = {
            "master": args.master, "model": args.model,
            "seeds": list(args.seeds), "n_seed": args.n_seed,
            "budget": args.budget, "holdout_frac": args.holdout_frac,
            "split_seed": args.split_seed,
            "n_conditions": split["n_conditions"],
            "n_train_conditions": split["n_train"],
            "n_test_conditions": split["n_test"],
            "n_test_near_boundary": split["n_test_near_boundary"],
            "train_test_shared_factor_vectors": overlap,
            "oracle_meta": dict(om),
            "oracle_calls_spent": {
                "test_set": 0,
                "arms": len(args.seeds) * 3 * (args.n_seed + args.budget),
                "note": "arm calls went to the SURROGATE oracle, not to the API",
            },
        }
        for name, p in write_outputs(res, multi, args.out_dir, extra).items():
            print(f"[wrote] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
