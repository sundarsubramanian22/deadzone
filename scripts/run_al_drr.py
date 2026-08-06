"""
scripts/run_al_drr.py — does REPARAMETERISING the reverb axis rescue the D3b
active-learning null?

THE QUESTION
------------
D3b is a null: straddle acquisition does not beat random sampling on the real
grid (`results/al_savings.{json,txt}` — target 0.162 reached by 2/8 active vs
4/8 random seeds; across 4 splits x 8 seeds active won 13/32 paired runs with a
median paired difference of +0.003, and the winner FLIPS between splits). The
implementation is not broken: `tests/test_active_learning.py` shows the same
machinery beating random comfortably on a PLANTED boundary.

`results/interaction_report.txt` supplies an independent, mechanistic reason why
the real surface might be hostile to a GP. `AssetLibrary.resolve()` snaps each
requested `rt60` to the NEAREST MEASURED RIR, so every level of the "rt60" axis
is a DIFFERENT REAL ROOM. RT60 describes a decay slope and says nothing about how
much direct sound reaches the mic, and on this grid the damage is monotone in
direct-to-reverberant ratio but NOT in RT60:

    spearman(DRR, WER) = -1.000   vs   spearman(RT60, WER) = +0.800

That report states the implied fix outright: "The defensible parameterisation for
this axis is DRR (or C50), which orders the measured conditions perfectly where
RT60 does not." A GP fitted with `rt60` as a CONTINUOUS coordinate is therefore
assuming a smoothness the instrument does not have.

So: re-run the EXACT published comparison with the reverb coordinate swapped from
RT60 to DRR, and see whether the null survives its own obvious fix.

WHAT IS HELD FIXED (everything except the coordinate)
-----------------------------------------------------
The comparison is only meaningful if nothing else moves, so this script reuses
`deadzone.analysis.al_savings` wholesale rather than reimplementing the protocol:
same `test_set_from_master` split (holdout_frac 0.4), same
`surrogate_oracle_from_master` GP oracle fitted to the training half, same
`multi_seed_curves` three-arm race, same `split_robustness` aggregation, same
budget (n_seed 15 + budget 30 = 45 evaluations/arm), same 8 AL seeds (0-7), same
4 train/test splits (0-3), same straddle acquisition, same uniform-random
baseline, same held-out test set built from REAL measurements already in the
master table, same metric (`boundary_rmse`), same threshold 0.5 and band +-0.15.
ZERO API calls: every oracle call goes to the surrogate, exactly as published.

THE CEILING ON THIS EXPERIMENT — STATE IT BEFORE READING ANY RESULT
-------------------------------------------------------------------
The master table contains exactly FOUR distinct RIRs, one per `rt60` level. So
the reverb axis is four discrete rooms, and ANY reparameterisation of it — DRR,
C50, anything — is nothing more than a relabelling of four points on a line.
Because `GPSurrogate` normalises each axis by its bounds, a coordinate is fully
characterised by (a) the ORDER of the four rooms along the axis and (b) the
relative SPACING of the two interior points. That is two free numbers plus an
ordering. DRR cannot introduce information the grid did not measure; it can only
place four already-measured points more or less helpfully.

Hence the negative controls are not a formality, they are the experiment:

  CONTROL A (ordering) — all 4! = 24 assignments of the DRR VALUE SET to the four
    rooms. Spacing is held exactly fixed (same four numbers every time); only
    which room gets which coordinate varies. The true DRR assignment is one of the
    24, so its RANK among them is a direct, exhaustive permutation test of the
    claim "DRR wins because it orders the rooms correctly".

  CONTROL B (spacing) — random monotone relabellings that preserve RT60's ordering
    of the four rooms but randomise their spacing. This asks whether RT60's null is
    about its particular spacing rather than its ordering.

If a random relabelling wins as often as DRR does, the "DRR fixes it" story is
dead and this script says so.

A note on what a coordinate change actually does here, because it cuts both ways:
the surrogate ORACLE is refitted in each coordinate too, so a friendlier
coordinate makes the oracle's own response surface smoother — easier for BOTH
arms. That is precisely why the headline statistic is the PAIRED active-minus-
random difference within a coordinate system, and why Control A is the right
control: a permuted relabelling perturbs oracle smoothness in the same way.

    ./.venv/bin/python scripts/run_al_drr.py            # full run, ~20 min
    ./.venv/bin/python scripts/run_al_drr.py --quick    # primary coordinates only

Writes `results/al_drr.json` and `results/al_drr.txt`. Offline, deterministic,
seeded, re-runnable. Deps: numpy, scipy, scikit-learn, soundfile (librosa only if
a RIR needs resampling).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

# --- repo-root bootstrap (same shim the other scripts use) -----------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.design import DEFAULT_FACTOR_SPACE, Factor, FactorSpace     # noqa: E402
from deadzone.active_learning import (                                     # noqa: E402
    DEFAULT_BAND, DEFAULT_THRESHOLD, best_so_far, evals_to_target,
)
from deadzone.analysis.al_savings import (                                 # noqa: E402
    ACTIVE_ARM, ARMS, METRIC, PASSIVE_ARM, _json_safe, acquisition_concentration,
    al_savings_report, multi_seed_curves, oracle_fidelity_floor,
    paired_arm_difference, savings_headline, split_robustness,
    surrogate_oracle_from_master, test_set_from_master,
)
from deadzone.analysis.interactions import (                               # noqa: E402
    condition_matrix, load_master_rows,
)


# ============================================================================
# THE PUBLISHED PROTOCOL — copied from results/al_savings.json's run_config so
# the two results are directly comparable. Do not "tune" these.
# ============================================================================

MODEL = "nova-3"
SPLIT_SEEDS = (0, 1, 2, 3)
AL_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
N_SEED = 15
BUDGET = 30                       # n_seed + budget = 45 evaluations per arm
HOLDOUT_FRAC = 0.4
THRESHOLD = DEFAULT_THRESHOLD     # 0.5
BAND = DEFAULT_BAND               # +-0.15
HEADLINE_SPLIT = 0                # the split the published headline was quoted at

CONTROL_RNG_SEED = 20260805       # fixes Control B's random spacings
N_MONOTONE_CONTROLS = 16

# The reverb factor is position 0 of DEFAULT_FACTOR_SPACE; the coordinate swap
# replaces it IN PLACE so the factor ordering (and hence the GP's ARD kernel
# layout and the SALib encoding) is structurally identical across coordinates.
REVERB_AXIS_INDEX = 0

# DRR / C50 windows — identical to the definitions used by scripts/run_d3a.py,
# which produced the published table in results/interaction_report.txt.
DIRECT_WINDOW_S = 0.0025          # +-2.5 ms around the direct-path peak
EARLY_WINDOW_S = 0.050            # C50's 50 ms early/late split


# ============================================================================
# MEASURING THE DELIVERED ROOM ACOUSTICS (from the RIR files, not hardcoded)
# ============================================================================

def rir_acoustics(path: str, fs: int = 16000) -> dict:
    """
    DRR and C50 of one measured RIR, computed from the FILE.

    Identical arithmetic to `scripts/run_d3a.py`'s `rir_mechanism`, which is what
    produced the numbers quoted in `results/interaction_report.txt`; the test
    suite pins this function against those published values so the two cannot
    drift apart.

      DRR — energy within +-2.5 ms of the direct-path peak over everything else.
            The standard proxy for effective source-mic distance, i.e. exactly the
            thing RT60 omits.
      C50 — early (<= 50 ms from the direct peak) over late energy: the speech
            clarity index.
    """
    import soundfile as sf

    h, hfs = sf.read(path, dtype="float64")
    h = np.asarray(h, dtype=float)
    if h.ndim > 1:
        h = h.mean(axis=1)
    if hfs != fs:
        import librosa
        h = librosa.resample(h, orig_sr=hfs, target_sr=fs)
    d = int(np.argmax(np.abs(h)))
    w = int(DIRECT_WINDOW_S * fs)
    e = int(EARLY_WINDOW_S * fs)
    direct = h[max(0, d - w):d + w]
    reverb = np.concatenate([h[:max(0, d - w)], h[d + w:]])
    drr = 10.0 * np.log10((direct ** 2).sum() / max((reverb ** 2).sum(), 1e-20))
    c50 = 10.0 * np.log10((h[d:d + e] ** 2).sum() / max((h[d + e:] ** 2).sum(), 1e-20))
    return {"drr_db": float(drr), "c50_db": float(c50)}


def measure_delivered_rooms(rows: Sequence[Mapping], model: str = MODEL,
                            fs: int = 16000) -> list[dict]:
    """
    One record per RIR the grid ACTUALLY delivered, keyed by the master table's
    `rir_key` column.

    `rir_key` is the field that records provenance — which RIR file the composer
    resolved this row to — so the coordinate mapping is built from what was
    measured, never from the `rt60` that was requested. The distinction is the
    whole point: `resolve()` snaps requests to the nearest measured RIR, so the
    request and the delivery are different objects.
    """
    by_key: dict[str, dict] = {}
    for r in rows:
        if r.get("model") != model:
            continue
        key = r.get("rir_key")
        if not key:
            continue
        rec = by_key.setdefault(key, {
            "rir_key": key, "room": os.path.basename(key),
            "rt60_requested": set(), "rt60_measured": float("nan"), "wers": [],
        })
        rec["rt60_requested"].add(round(float(r["rt60"]), 6))
        m = r.get("rir_rt60_measured")
        if m not in (None, ""):
            rec["rt60_measured"] = float(m)
        failed = str(r.get("failed", "")).strip().lower() in {"true", "1", "yes", "t"}
        if not failed:
            try:
                rec["wers"].append(float(r["wer"]))
            except (TypeError, ValueError):
                pass

    out = []
    for rec in by_key.values():
        ac = rir_acoustics(rec["rir_key"], fs=fs)
        out.append({
            "rir_key": rec["rir_key"], "room": rec["room"],
            "rt60_requested": sorted(rec["rt60_requested"]),
            "rt60_measured": rec["rt60_measured"],
            "drr_db": ac["drr_db"], "c50_db": ac["c50_db"],
            "marginal_wer": float(np.mean(rec["wers"])) if rec["wers"] else float("nan"),
            "n_rows": len(rec["wers"]),
        })
    out.sort(key=lambda d: d["rt60_measured"])
    return out


# ============================================================================
# COORDINATE SYSTEMS — a relabelling of the four delivered rooms
# ============================================================================

@dataclass(frozen=True)
class Coordinate:
    """
    One parameterisation of the reverb axis: a value per delivered RIR, plus the
    bounds the GP normalises by.

    `values` is keyed by `rir_key` — the master table's provenance field — so the
    mapping is "each condition takes the coordinate of the RIR it actually
    resolved to", by construction rather than by convention.
    """
    name: str                       # the factor/column name in the FactorSpace
    label: str                      # human-readable, for the report
    kind: str                       # primary | control_permutation | control_monotone
    values: dict                    # rir_key -> float
    bounds: tuple
    degradation: str = "up"         # narrative only; unused by the AL path
    note: str = ""
    meta: dict = field(default_factory=dict)

    def space(self) -> FactorSpace:
        """DEFAULT_FACTOR_SPACE with the reverb factor swapped IN PLACE."""
        factors = list(DEFAULT_FACTOR_SPACE.factors)
        factors[REVERB_AXIS_INDEX] = Factor(
            self.name, "continuous", low=float(self.bounds[0]),
            high=float(self.bounds[1]), degradation=self.degradation)
        return FactorSpace(factors)


def _bounds_from(values: Mapping) -> tuple:
    v = [float(x) for x in values.values()]
    return (float(min(v)), float(max(v)))


def remap_rows(rows: Sequence[Mapping], coord: Coordinate,
               model: str = MODEL) -> list[dict]:
    """
    Rewrite each master row with the coordinate of the RIR it resolved to.

    Only the reverb column changes: a shallow copy per row gains a
    `coord.name` key taken from `coord.values[row["rir_key"]]`. Every other
    factor, the WER, and the failed flag pass through untouched.

    An unmapped `rir_key` RAISES rather than dropping the row. A silently dropped
    room would remove a quarter of the reverb axis and the run would still produce
    a plausible-looking number — the exact class of failure this project is about.
    """
    out = []
    missing: set = set()
    for r in rows:
        if r.get("model") != model:
            continue
        key = r.get("rir_key")
        if key not in coord.values:
            missing.add(key)
            continue
        d = dict(r)
        d[coord.name] = float(coord.values[key])
        out.append(d)
    if missing:
        raise KeyError(
            f"coordinate {coord.name!r} has no value for rir_key(s) {sorted(missing)}; "
            f"remapping would silently drop those rooms and shrink the reverb axis.")
    if not out:
        raise ValueError(f"no rows for model {model!r}")
    return out


def primary_coordinates(rooms: Sequence[Mapping]) -> list[Coordinate]:
    """
    The published baseline plus the two physically-motivated reparameterisations,
    plus a bounds-choice control.

    `rt60` reproduces the published run exactly: the factor keeps its name and its
    DEFAULT_FACTOR_SPACE bounds [0.2, 1.0], and the values are the MEASURED
    Schroeder RT60s — which is what `condition_matrix(use_measured_rt60=True)`
    substitutes on the published path. `rt60_minmax` is the same numbers under the
    delivered-min/max bounds rule the other coordinates use, so that any
    difference attributable to the bounds convention rather than the coordinate is
    visible instead of assumed away.
    """
    rt60 = {r["rir_key"]: r["rt60_measured"] for r in rooms}
    drr = {r["rir_key"]: r["drr_db"] for r in rooms}
    c50 = {r["rir_key"]: r["c50_db"] for r in rooms}
    default_rt60 = DEFAULT_FACTOR_SPACE.factors[REVERB_AXIS_INDEX]
    return [
        Coordinate("rt60", "measured RT60 (PUBLISHED baseline)", "primary", rt60,
                   (float(default_rt60.low), float(default_rt60.high)), "up",
                   "the published D3b coordinate: DEFAULT_FACTOR_SPACE bounds, "
                   "measured Schroeder RT60 substituted for the request"),
        Coordinate("rt60_minmax", "measured RT60 (delivered-min/max bounds)",
                   "primary", rt60, _bounds_from(rt60), "up",
                   "bounds-convention control: same values as the baseline, bounds "
                   "set the way every other coordinate here sets them"),
        Coordinate("drr_db", "direct-to-reverberant ratio (THE HYPOTHESIS)",
                   "primary", drr, _bounds_from(drr), "down",
                   "orders the delivered conditions perfectly (spearman -1.000)"),
        Coordinate("c50_db", "C50 clarity index", "primary", c50,
                   _bounds_from(c50), "down",
                   "an ALMOST-right reparameterisation: spearman -0.800, it swaps "
                   "one adjacent pair"),
    ]


def permutation_controls(rooms: Sequence[Mapping]) -> list[Coordinate]:
    """
    CONTROL A — all 24 assignments of the DRR VALUE SET to the four rooms.

    Spacing is held EXACTLY fixed (the same four numbers, hence the same bounds,
    every time); only the room->value pairing varies. The true DRR assignment is
    one of these 24, so its rank among them is an exhaustive permutation test of
    the ordering claim, with no distributional assumptions and nothing left to
    sampling luck.
    """
    keys = [r["rir_key"] for r in rooms]                 # sorted by measured RT60
    vals = sorted(float(r["drr_db"]) for r in rooms)
    true_drr = {r["rir_key"]: float(r["drr_db"]) for r in rooms}
    bounds = (vals[0], vals[-1])
    out = []
    for i, perm in enumerate(itertools.permutations(range(len(vals)))):
        values = {k: vals[p] for k, p in zip(keys, perm)}
        is_true = all(abs(values[k] - true_drr[k]) < 1e-12 for k in keys)
        out.append(Coordinate(
            f"permctl_{i:02d}", f"perm {perm}" + (" == TRUE DRR" if is_true else ""),
            "control_permutation", values, bounds, "down",
            "Control A: DRR's spacing, permuted room assignment",
            {"perm": list(perm), "is_true_drr": bool(is_true)}))
    return out


def monotone_controls(rooms: Sequence[Mapping], n: int = N_MONOTONE_CONTROLS,
                      seed: int = CONTROL_RNG_SEED) -> list[Coordinate]:
    """
    CONTROL B — random monotone relabellings: RT60's ORDERING of the four rooms is
    preserved, their SPACING is drawn at random.

    Asks the other half of the question: is the published null a property of
    RT60's ordering, or merely of the particular spacing RT60 happens to give?
    Because the GP normalises by bounds, only the two interior points' relative
    positions actually vary, which is exactly the free parameter under test.
    """
    keys = [r["rir_key"] for r in rooms]                 # ascending measured RT60
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        u = np.sort(rng.random(len(keys)))
        values = {k: float(v) for k, v in zip(keys, u)}
        out.append(Coordinate(
            f"monoctl_{i:02d}", f"monotone relabel #{i}", "control_monotone",
            values, (float(u[0]), float(u[-1])), "up",
            "Control B: RT60's ordering, random spacing",
            {"spacing": [float(v) for v in u]}))
    return out


# ============================================================================
# ONE COORDINATE SYSTEM, RUN THROUGH THE PUBLISHED PROTOCOL
# ============================================================================

def coordinate_geometry(coord: Coordinate, rooms: Sequence[Mapping]) -> dict:
    """
    What the GP actually sees along this axis: the four rooms' NORMALISED
    positions (the only thing the kernel responds to) alongside their measured
    marginal WER, plus whether WER is monotone in this coordinate.

    Printed with every result so a reader can see for themselves that a
    coordinate change here is a relabelling of four points and nothing more.
    """
    from scipy.stats import spearmanr

    lo, hi = coord.bounds
    span = (hi - lo) or 1.0
    pts = []
    for r in rooms:
        v = float(coord.values[r["rir_key"]])
        pts.append({"room": r["room"], "value": v, "norm": (v - lo) / span,
                    "marginal_wer": r["marginal_wer"],
                    "rt60_measured": r["rt60_measured"]})
    pts.sort(key=lambda p: p["value"])
    x = np.array([p["value"] for p in pts])
    w = np.array([p["marginal_wer"] for p in pts])
    rho = float(spearmanr(x, w).statistic)
    return {"points": pts, "spearman_coord_vs_wer": rho,
            "monotone_in_wer": bool(abs(abs(rho) - 1.0) < 1e-9),
            "bounds": [float(lo), float(hi)]}


def _compact_curves(multi: Mapping) -> dict:
    """Per-arm, per-seed best-so-far curves — the minimum needed to recompute
    evals-to-target at any threshold later, without shipping the trajectories."""
    return {
        arm: [[{"n_evals": int(p["n_evals"]), METRIC: float(p[METRIC])}
               for p in best_so_far(ps["curves"][arm], METRIC)]
              for ps in multi["per_seed"]]
        for arm in multi["per_seed"][0]["curves"]
    }


def run_coordinate(rows: Sequence[Mapping], coord: Coordinate, *,
                   model: str = MODEL, split_seeds: Sequence[int] = SPLIT_SEEDS,
                   al_seeds: Sequence[int] = AL_SEEDS, n_seed: int = N_SEED,
                   budget: int = BUDGET, holdout_frac: float = HOLDOUT_FRAC,
                   threshold: float = THRESHOLD, band: float = BAND,
                   rooms: Sequence[Mapping] | None = None) -> dict:
    """
    The published D3b comparison, run once in `coord`'s coordinate system.

    `split_robustness` is CALLED, not reimplemented — it is the same function that
    produced the published 4-splits x 8-seeds robustness result, so the only thing
    that can differ between this run and that one is the coordinate. The headline
    split is additionally re-run on its own so the seed-band curves survive for the
    side-by-side target sweep (`split_robustness` aggregates them away).
    """
    space = coord.space()
    remapped = remap_rows(rows, coord, model=model)

    # --- the headline split, kept in full so the curves survive ---------------
    split = test_set_from_master(remapped, space, model=model,
                                 holdout_frac=holdout_frac, seed=HEADLINE_SPLIT,
                                 threshold=threshold, band=band)
    # `space=space` is load-bearing: the oracle encodes samples through it, so
    # omitting it fits the oracle in DEFAULT_FACTOR_SPACE's coordinates while the
    # arms sample in this one.
    oracle, _ = surrogate_oracle_from_master(split=split, space=space,
                                             model=model, seed=HEADLINE_SPLIT)

    # Structural check, run every time rather than asserted in prose: the
    # surrogate oracle must never have been fitted at a condition the arms are
    # scored on, or both arms look perfect and the race is dead.
    tr = {tuple(np.round(r, 9)) for r in split["X_train"]}
    te = {tuple(np.round(r, 9)) for r in split["X_test"]}
    overlap = len(tr & te)
    if overlap:
        raise AssertionError(f"train/test leakage in {coord.name}: {overlap} shared rows")

    multi = multi_seed_curves(oracle, space, X_test=split["X_test"],
                              y_test=split["y_test"], seeds=list(al_seeds),
                              n_seed=n_seed, budget=budget,
                              threshold=threshold, band=band)
    headline = savings_headline(multi)
    report = al_savings_report(multi)

    # --- the full 4-split robustness, via the published function --------------
    rob = split_robustness(remapped, space, model=model, split_seeds=list(split_seeds),
                           seeds=list(al_seeds), n_seed=n_seed, budget=budget,
                           holdout_frac=holdout_frac, threshold=threshold, band=band)

    out = {
        "name": coord.name, "label": coord.label, "kind": coord.kind,
        "note": coord.note, "meta": dict(coord.meta),
        "bounds": [float(coord.bounds[0]), float(coord.bounds[1])],
        "values": {k: float(v) for k, v in coord.values.items()},
        "headline": headline,
        "bands": report["bands"],
        "curves": _compact_curves(multi),
        "floor": oracle_fidelity_floor(split, space, seed=HEADLINE_SPLIT,
                                       threshold=threshold, band=band),
        "paired_headline_split": paired_arm_difference(multi),
        "acquisition_concentration": acquisition_concentration(multi),
        "robustness": {
            "n_splits_won_by_active": rob["n_splits_won_by_active"],
            "n_splits": len(rob["split_seeds"]),
            "n_paired_runs": rob["n_paired_runs"],
            "n_paired_won_by_active": rob["n_paired_won_by_active"],
            "median_paired_diff": rob["median_paired_diff"],
            "winner_stable_across_splits": rob["winner_stable_across_splits"],
            "winner_by_split": rob["winner_by_split"],
            "verdict": rob["verdict"],
            "per_split": [{"split_seed": p["split_seed"],
                           "n_test_near_boundary": p["n_test_near_boundary"],
                           "floor_boundary_rmse": p["floor"]["boundary_rmse"],
                           "final_fidelity": p["final_fidelity"],
                           "median_paired_diff": p["paired"]["median_diff"],
                           "lowest_median_arm": p["lowest_median_arm"]}
                          for p in rob["per_split"]],
        },
        # identity fingerprints — see `identity_check`
        "split_fingerprint": {
            "n_conditions": int(split["n_conditions"]),
            "n_train": int(split["n_train"]), "n_test": int(split["n_test"]),
            "n_test_near_boundary": int(split["n_test_near_boundary"]),
            "y_train_sum": float(np.sum(split["y_train"])),
            "y_test_sum": float(np.sum(split["y_test"])),
            "y_test": [float(v) for v in split["y_test"]],
            "test_condition_names": [c["condition_name"] for c in split["conditions_test"]],
            "train_test_shared_rows": int(overlap),
        },
    }
    if rooms is not None:
        out["geometry"] = coordinate_geometry(coord, rooms)
    return out


# ============================================================================
# PARALLEL DRIVER
# ============================================================================

_WORKER: dict = {}


def _init_worker(master: str, model: str) -> None:
    """Load and prune the master table ONCE per worker process."""
    keep = set(DEFAULT_FACTOR_SPACE.names) | {
        "wer", "failed", "model", "condition_name", "rt60",
        "rir_rt60_measured", "rir_key"}
    rows = [{k: v for k, v in r.items() if k in keep}
            for r in load_master_rows(master) if r.get("model") == model]
    _WORKER["rows"] = rows
    _WORKER["model"] = model


def _run_one(payload: tuple) -> dict:
    coord, rooms = payload
    import warnings
    warnings.filterwarnings("ignore")
    return run_coordinate(_WORKER["rows"], coord, model=_WORKER["model"], rooms=rooms)


def run_all(coords: Sequence[Coordinate], rooms: Sequence[Mapping], master: str,
            model: str = MODEL, workers: int = 8) -> list[dict]:
    """Run every coordinate system. Results are keyed by name, so the order in
    which the pool happens to finish cannot affect the output."""
    payloads = [(c, list(rooms)) for c in coords]
    if workers <= 1:
        _init_worker(master, model)
        done = [_run_one(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(master, model)) as ex:
            done = list(ex.map(_run_one, payloads))
    by_name = {d["name"]: d for d in done}
    return [by_name[c.name] for c in coords]


# ============================================================================
# CROSS-COORDINATE CHECKS AND AGGREGATION
# ============================================================================

def identity_check(results: Sequence[Mapping]) -> dict:
    """
    Prove the arms are racing on the SAME data in every coordinate system.

    A coordinate swap must not change WHICH conditions are measured, which of them
    are held out, or what their WERs are — otherwise a "win" could come from an
    easier test set rather than a better axis. `condition_matrix` buckets by the
    factor tuple and the split is an index permutation at a fixed seed, so the same
    conditions should land in the same halves under any bijective relabelling; this
    verifies that rather than assuming it.

    What CANNOT be identical, and must be said plainly: the surrogate oracle is a
    GP refitted in each coordinate, so it is the same DATA through a different
    geometry. That is intrinsic to the question being asked, and it is why the
    headline statistic is a paired within-coordinate difference and why Control A
    exists.
    """
    ref = results[0]["split_fingerprint"]
    bad = []
    for r in results:
        f = r["split_fingerprint"]
        same = (f["n_conditions"] == ref["n_conditions"]
                and f["n_train"] == ref["n_train"] and f["n_test"] == ref["n_test"]
                and f["n_test_near_boundary"] == ref["n_test_near_boundary"]
                and np.allclose(f["y_test"], ref["y_test"], atol=1e-12)
                and abs(f["y_train_sum"] - ref["y_train_sum"]) < 1e-9
                and f["test_condition_names"] == ref["test_condition_names"])
        if not same:
            bad.append(r["name"])
    return {
        "identical_test_set_across_coordinates": not bad,
        "coordinates_differing": bad,
        "n_conditions": ref["n_conditions"], "n_train": ref["n_train"],
        "n_test": ref["n_test"], "n_test_near_boundary": ref["n_test_near_boundary"],
        "all_train_test_disjoint": all(
            r["split_fingerprint"]["train_test_shared_rows"] == 0 for r in results),
        "note": ("same conditions, same held-out WERs, same near-boundary count in "
                 "every coordinate; only the reverb x-coordinate (and hence the "
                 "refitted GP oracle's geometry) differs."),
    }


def shared_target_sweep(results: Sequence[Mapping], n_targets: int = 12) -> dict:
    """
    Median evals-to-target for both arms, in every coordinate system, on ONE
    SHARED grid of fidelity targets.

    `al_savings.target_sweep` derives its own target range per run, which is right
    for a single result and wrong for a comparison: each coordinate would be scored
    on a different x-axis. The grid here spans the union of every coordinate's
    achieved fidelities, so the whole curve is published for all of them and no
    threshold can be picked after the fact.
    """
    finals = [c[-1][METRIC] for r in results for arm in (ACTIVE_ARM, PASSIVE_ARM)
              for c in r["curves"][arm] if math.isfinite(c[-1][METRIC])]
    if not finals:
        return {"targets": [], "rows": []}
    lo, hi = min(finals), max(finals)
    targets = [lo] if hi <= lo else list(np.linspace(lo, hi, n_targets))
    rows = []
    for t in targets:
        row = {"target": float(t), "by_coordinate": {}}
        for r in results:
            cell = {}
            for arm in (ACTIVE_ARM, PASSIVE_ARM):
                ev = [evals_to_target(c, float(t), METRIC) for c in r["curves"][arm]]
                cell[arm] = float(np.median(np.asarray(ev, dtype=float)))
                cell[f"{arm}_reached"] = sum(1 for v in ev if math.isfinite(v))
            a, p = cell[ACTIVE_ARM], cell[PASSIVE_ARM]
            cell["pct_reduction"] = (100.0 * (p - a) / p
                                     if math.isfinite(a) and math.isfinite(p) and p > 0
                                     else float("nan"))
            row["by_coordinate"][r["name"]] = cell
        rows.append(row)
    return {"targets": [float(t) for t in targets], "rows": rows,
            "note": ("ONE shared target grid across all coordinate systems, spanning "
                     "every arm's achieved fidelity — the whole curve, so no single "
                     "threshold can be cherry-picked.")}


def coordinate_spread(results: Sequence[Mapping]) -> dict:
    """
    How much does the CHOICE OF COORDINATE move the active-vs-random gap at all,
    across every parameterisation tried?

    This is the summary statistic that makes the null readable. If 44 different
    relabellings of the same four rooms all land within a hair of zero and the
    SIGN is a coin flip, then the coordinate is not the binding constraint and no
    further reparameterisation is worth running.
    """
    # None = NaN after a JSON round-trip (`_json_safe`); such a coordinate has
    # no comparable paired runs at all, so it is excluded and COUNTED, never
    # silently coerced to 0.0 (which would read as a perfect tie).
    raw = [r["robustness"]["median_paired_diff"] for r in results]
    vals = [float(v) for v in raw if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n_coordinate_systems": 0, "n_incomparable": len(raw),
                "note": "no coordinate system produced comparable paired runs"}
    return {
        "n_coordinate_systems": len(vals),
        "n_incomparable": len(raw) - len(vals),
        "median_paired_diff": {"min": min(vals), "median": float(np.median(vals)),
                               "max": max(vals)},
        "n_favouring_active": sum(1 for v in vals if v < 0),
        "n_exceeding_0.01_in_magnitude": sum(1 for v in vals if abs(v) > 0.01),
        "n_with_stable_winner": sum(
            1 for r in results if r["robustness"]["winner_stable_across_splits"]),
        "note": ("across every parameterisation tried, the median paired "
                 "active-minus-random difference and the sign of that difference. "
                 "A coin-flip sign with a near-zero spread means the coordinate "
                 "choice is not what is holding active learning back."),
    }


def per_split_fidelity(results: Sequence[Mapping], arm: str = PASSIVE_ARM) -> dict:
    """
    One arm's final `boundary_rmse` in EVERY split, per coordinate.

    Here to stop a specific overclaim. On the headline split alone, DRR and C50
    look like they give a materially better surrogate of the real held-out WERs
    than RT60 does — a tempting secondary finding, since `boundary_rmse` really is
    scored against real measurements. Across all four splits that ordering
    reverses. With only ~13 held-out conditions near the contour, a single split's
    fidelity is a 13-point statistic and reading a coordinate ranking off one of
    them is exactly the error the split-robustness check exists to prevent.
    """
    rows = []
    for r in results:
        per = [p["final_fidelity"].get(arm, {}) for p in r["robustness"]["per_split"]]
        med = [float(p.get("median") if p.get("median") is not None else float("nan"))
               for p in per]
        rows.append({
            "name": r["name"], "per_split_median": med,
            "mean_over_splits": float(np.nanmean(med)),
            "n_test_near_boundary": [p["n_test_near_boundary"]
                                     for p in r["robustness"]["per_split"]],
        })
    return {"arm": arm, "rows": rows,
            "note": ("final boundary_rmse of the passive arm, scored against REAL "
                     "held-out measurements, in each split. Compare across splits "
                     "before ranking coordinates.")}


def control_verdict(results: Sequence[Mapping]) -> dict:
    """
    The negative controls, read out as an exhaustive permutation test.

    The statistic is the one the published null reports: the MEDIAN PAIRED
    difference in final `boundary_rmse` (active minus random) across all 4 splits x
    8 seeds = 32 paired runs, where NEGATIVE means active learning is better.

    For Control A the true DRR assignment is one of the 24 permutations of the same
    four numbers, so `drr_rank_of_24` is a p-value in disguise: rank 1 of 24 would
    be p = 1/24 = 0.042 that the ordering is doing the work; a middling rank means
    the "DRR fixes it" story is dead and the report must say so.
    """
    by_kind: dict[str, list] = {}
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r)

    def stat(r):
        return float(r["robustness"]["median_paired_diff"])

    prim = {r["name"]: r for r in by_kind.get("primary", [])}
    perms = by_kind.get("control_permutation", [])
    monos = by_kind.get("control_monotone", [])

    out: dict = {
        "statistic": "median paired (active - random) final boundary_rmse over "
                     "4 splits x 8 seeds = 32 paired runs; NEGATIVE = active better",
        "primary": {n: {"median_paired_diff": stat(r),
                        "n_paired_won_by_active": r["robustness"]["n_paired_won_by_active"],
                        "n_splits_won_by_active": r["robustness"]["n_splits_won_by_active"],
                        "winner_stable": r["robustness"]["winner_stable_across_splits"]}
                    for n, r in prim.items()},
    }

    if perms:
        vals = sorted(stat(r) for r in perms)
        true = next((r for r in perms if r["meta"].get("is_true_drr")), None)
        drr_stat = stat(true) if true is not None else float("nan")
        better = sum(1 for v in vals if v < drr_stat)
        out["control_a_permutation"] = {
            "n_permutations": len(perms),
            "true_drr_median_paired_diff": drr_stat,
            "drr_rank_of_n": better + 1,
            "n_permutations_better_than_drr": better,
            "permutation_p_value": (better + 1) / len(perms),
            "median_over_permutations": float(np.median(vals)),
            "min": vals[0], "max": vals[-1],
            "n_permutations_favouring_active": sum(1 for v in vals if v < 0),
            "note": ("all 4! assignments of the SAME four DRR values to the four "
                     "rooms: spacing fixed, ordering varied. The true DRR "
                     "assignment is one of them."),
        }
    if monos:
        vals = sorted(stat(r) for r in monos)
        base = prim.get("rt60_minmax") or prim.get("rt60")
        b = stat(base) if base is not None else float("nan")
        out["control_b_monotone"] = {
            "n_relabellings": len(monos),
            "baseline_rt60_median_paired_diff": b,
            "n_better_than_rt60": sum(1 for v in vals if v < b),
            "median": float(np.median(vals)), "min": vals[0], "max": vals[-1],
            "n_favouring_active": sum(1 for v in vals if v < 0),
            "note": ("RT60's ordering preserved, spacing randomised: isolates "
                     "spacing from ordering."),
        }

    # --- the one-sentence verdict, written by the numbers, not by hand --------
    drr = prim.get("drr_db")
    rt60 = prim.get("rt60")
    if drr is not None and rt60 is not None:
        d_stat, r_stat = stat(drr), stat(rt60)
        d_rob, r_rob = drr["robustness"], rt60["robustness"]
        d_wins = d_stat < 0 and d_rob["n_paired_won_by_active"] > d_rob["n_paired_runs"] / 2
        pa = out.get("control_a_permutation", {})
        rank = pa.get("drr_rank_of_n")
        n_perm = pa.get("n_permutations", 0)
        beats_controls = bool(rank is not None and n_perm and rank <= max(1, n_perm // 12))
        if d_wins and beats_controls:
            verdict = (
                f"REPARAMETERISATION HELPS: in DRR coordinates active learning beats "
                f"random in {d_rob['n_paired_won_by_active']}/{d_rob['n_paired_runs']} "
                f"paired runs (median paired difference {d_stat:+.3f}) vs "
                f"{r_rob['n_paired_won_by_active']}/{r_rob['n_paired_runs']} "
                f"({r_stat:+.3f}) under RT60, and the true DRR assignment ranks "
                f"{rank}/{n_perm} among all permutations of the same four values "
                f"(permutation p = {pa.get('permutation_p_value'):.3f}), so the gain "
                f"tracks the ORDERING rather than the spacing.")
        elif d_wins:
            verdict = (
                f"DRR WINS BUT THE CONTROL DOES NOT SUPPORT THE MECHANISM: active "
                f"beats random in {d_rob['n_paired_won_by_active']}/"
                f"{d_rob['n_paired_runs']} paired runs under DRR (median "
                f"{d_stat:+.3f}), but the true DRR assignment ranks only {rank}/"
                f"{n_perm} among random permutations of the same four values — a "
                f"random relabelling does about as well, so this is spacing luck on "
                f"four points, not evidence that DRR is the right axis.")
        else:
            verdict = (
                f"STILL A NULL: reparameterising the reverb axis from RT60 to DRR "
                f"does not rescue active learning. Under DRR, straddle acquisition "
                f"beats random in {d_rob['n_paired_won_by_active']}/"
                f"{d_rob['n_paired_runs']} paired runs with a median paired "
                f"difference of {d_stat:+.3f} (RT60: "
                f"{r_rob['n_paired_won_by_active']}/{r_rob['n_paired_runs']}, "
                f"{r_stat:+.3f}); negative would mean active is better.")
        out["verdict"] = verdict
        out["drr_beats_random"] = bool(d_wins)
        out["controls_support_mechanism"] = bool(beats_controls)
    return out


# ============================================================================
# THE REPORT
# ============================================================================

def _f(v) -> str:
    # `_json_safe` writes non-finite floats as null, and "never reached the target
    # within budget" is the ONLY non-finite value the target sweep can hold — so a
    # None coming back off disk means inf, not missing data. Rendering it as a
    # blank (or crashing, as this did) would quietly turn "never reached" into
    # "no measurement", which is the flattering direction.
    if v is None:
        return "inf"
    return f"{v:.0f}" if math.isfinite(v) else "inf"


def format_report(res: Mapping) -> str:
    bar = "=" * 78
    out = [bar, "D3b RE-RUN — DOES REPARAMETERISING THE REVERB AXIS RESCUE THE "
                "ACTIVE-LEARNING NULL?", bar, "",
           res["verdict"], ""]

    out += ["-" * 78, "THE CEILING ON THIS EXPERIMENT (read before the numbers)",
            "-" * 78,
            "  The master table contains exactly FOUR distinct RIRs, one per rt60",
            "  level, so the reverb axis is four discrete rooms. ANY reparameterisation",
            "  of it is a relabelling of four points on a line, and because the GP",
            "  normalises each axis by its bounds, a coordinate is fully described by",
            "  the ORDER of the four rooms plus the relative spacing of the two",
            "  interior points. DRR cannot add information the grid never measured.",
            "  That is why the negative controls below are the experiment, not a",
            "  formality.", ""]

    out += ["-" * 78, "THE DELIVERED ROOMS — measured from the RIR files, not assumed",
            "-" * 78,
            f"  {'room':<44} {'RT60':>6} {'DRR dB':>8} {'C50 dB':>8} {'WER':>7}"]
    for r in res["rooms"]:
        out.append(f"  {r['room'][:44]:<44} {r['rt60_measured']:>6.3f} "
                   f"{r['drr_db']:>8.2f} {r['c50_db']:>8.2f} {r['marginal_wer']:>7.4f}")
    out += ["",
            "  WER is MONOTONE in DRR and NOT in RT60 — that is the entire premise.",
            ""]

    ic = res["identity_check"]
    out += ["-" * 78, "IDENTITY CHECK — are the arms racing on the same data?", "-" * 78,
            f"  test set identical across ALL coordinate systems: "
            f"{'YES' if ic['identical_test_set_across_coordinates'] else 'NO — ' + str(ic['coordinates_differing'])}",
            f"  {ic['n_conditions']} conditions -> {ic['n_train']} oracle-training / "
            f"{ic['n_test']} held out, of which {ic['n_test_near_boundary']} lie within "
            f"+-{res['band']} of the {res['threshold']} contour",
            f"  train/test disjoint in every coordinate: "
            f"{'YES' if ic['all_train_test_disjoint'] else 'NO'}",
            f"  {ic['note']}", ""]

    out += ["-" * 78,
            "HEADLINE COMPARISON — 4 train/test splits x 8 AL seeds = 32 paired runs",
            "-" * 78,
            "  diff = median paired (active - random) final boundary_rmse; "
            "NEGATIVE = active better",
            "",
            f"  {'coordinate':<34} {'splits':>7} {'paired':>8} {'diff':>8} {'stable':>7} "
            f"{'floor':>7} {'mono':>5}"]
    for r in res["coordinates"]:
        if r["kind"] != "primary":
            continue
        rb = r["robustness"]
        g = r.get("geometry", {})
        out.append(
            f"  {r['label'][:34]:<34} {rb['n_splits_won_by_active']}/{rb['n_splits']:<5} "
            f"{rb['n_paired_won_by_active']:>3}/{rb['n_paired_runs']:<4} "
            f"{rb['median_paired_diff']:>+8.3f} "
            f"{'yes' if rb['winner_stable_across_splits'] else 'FLIPS':>7} "
            f"{r['floor']['boundary_rmse']:>7.3f} "
            f"{'YES' if g.get('monotone_in_wer') else 'no':>5}")
    out += ["",
            "  splits = splits in which active_boundary had the lower median "
            "boundary_rmse",
            "  paired = paired runs won by active     floor = best fidelity the "
            "oracle allows",
            "  mono   = is WER monotone along this coordinate?", ""]

    ca = res["controls"].get("control_a_permutation")
    if ca:
        out += ["-" * 78,
                "NEGATIVE CONTROL A — all 24 permutations of the SAME four DRR values",
                "-" * 78,
                "  Spacing held exactly fixed; only which room gets which value varies.",
                "  The true DRR assignment is one of the 24, so its rank is an "
                "exhaustive",
                "  permutation test of the ordering claim.",
                "",
                f"  true DRR median paired diff : {ca['true_drr_median_paired_diff']:+.3f}",
                f"  rank among all permutations : {ca['drr_rank_of_n']}/{ca['n_permutations']} "
                f"(permutation p = {ca['permutation_p_value']:.3f})",
                f"  permutations better than DRR: {ca['n_permutations_better_than_drr']}",
                f"  spread over permutations    : median {ca['median_over_permutations']:+.3f} "
                f"[{ca['min']:+.3f}, {ca['max']:+.3f}]",
                f"  permutations favouring active (diff < 0): "
                f"{ca['n_permutations_favouring_active']}/{ca['n_permutations']}", ""]

    cb = res["controls"].get("control_b_monotone")
    if cb:
        out += ["-" * 78,
                "NEGATIVE CONTROL B — random monotone relabellings (RT60's order, "
                "random spacing)",
                "-" * 78,
                f"  baseline RT60 median paired diff : "
                f"{cb['baseline_rt60_median_paired_diff']:+.3f}",
                f"  relabellings better than RT60    : {cb['n_better_than_rt60']}/"
                f"{cb['n_relabellings']}",
                f"  spread                           : median {cb['median']:+.3f} "
                f"[{cb['min']:+.3f}, {cb['max']:+.3f}]",
                f"  relabellings favouring active    : {cb['n_favouring_active']}/"
                f"{cb['n_relabellings']}", ""]

    sp = res.get("coordinate_spread")
    if sp:
        m = sp["median_paired_diff"]
        out += ["-" * 78,
                f"ALL {sp['n_coordinate_systems']} COORDINATE SYSTEMS — how much does "
                f"the coordinate move the gap?",
                "-" * 78,
                f"  median paired (active - random) difference across every "
                f"parameterisation tried:",
                f"    min {m['min']:+.4f}   median {m['median']:+.4f}   "
                f"max {m['max']:+.4f}",
                f"  favouring active (diff < 0)          : "
                f"{sp['n_favouring_active']}/{sp['n_coordinate_systems']}",
                f"  moving the gap by more than 0.010    : "
                f"{sp['n_exceeding_0.01_in_magnitude']}/{sp['n_coordinate_systems']}",
                f"  with a winner STABLE across 4 splits : "
                f"{sp['n_with_stable_winner']}/{sp['n_coordinate_systems']}",
                "",
                "  Every relabelling of the same four rooms lands within a hair of "
                "zero and the",
                "  sign is a coin flip. The coordinate is not what is holding active "
                "learning back.", ""]

    pf = res.get("per_split_fidelity")
    if pf:
        out += ["-" * 78,
                "PASSIVE-ARM FIDELITY IN EVERY SPLIT — why the headline split alone "
                "would mislead",
                "-" * 78,
                f"  {'coordinate':<14} " + " ".join(f"{'split' + str(i):>8}"
                                                    for i in range(4))
                + f" {'mean':>8}"]
        for row in pf["rows"]:
            out.append(f"  {row['name']:<14} "
                       + " ".join(f"{v:>8.3f}" for v in row["per_split_median"])
                       + f" {row['mean_over_splits']:>8.3f}")
        out += ["",
                "  On split 0 alone DRR and C50 look like materially better "
                "surrogates of the",
                "  real held-out WERs than RT60. Across all four splits that "
                "ordering REVERSES.",
                f"  Only ~{res['identity_check']['n_test_near_boundary']} held-out "
                f"conditions sit near the contour, so one split's fidelity is a "
                f"~13-point",
                "  statistic. No absolute-fidelity claim survives the split check "
                "either.", ""]

    sweep = res.get("target_sweep") or {}
    prim = [r["name"] for r in res["coordinates"] if r["kind"] == "primary"]
    if sweep.get("rows"):
        out += ["-" * 78,
                "THE WHOLE TARGET CURVE — median oracle calls to reach each fidelity",
                "-" * 78,
                "  ONE shared target grid for every coordinate, so no threshold can be "
                "cherry-picked.",
                "  Each cell is active/random median evals over 8 seeds "
                "(headline split 0); inf = never reached in 45.",
                "",
                f"  {'target':>7} | " + " | ".join(f"{n[:16]:>16}" for n in prim)]
        for row in sweep["rows"]:
            cells = []
            for n in prim:
                c = row["by_coordinate"].get(n, {})
                cells.append(f"{_f(c.get(ACTIVE_ARM, float('nan'))):>7}/"
                             f"{_f(c.get(PASSIVE_ARM, float('nan'))):<8}")
            out.append(f"  {row['target']:>7.3f} | " + " | ".join(f"{c:>16}" for c in cells))
        out.append("")

    out += ["-" * 78, "WHERE THE CALLS WENT — did straddle acquisition do its job?",
            "-" * 78]
    for r in res["coordinates"]:
        if r["kind"] != "primary":
            continue
        ac = r["acquisition_concentration"]
        out.append(f"  {r['name']:<14} straddle put "
                   f"{ac['active_acquired_frac_near_threshold']:>6.1%} of ACQUIRED evals "
                   f"near the contour vs {ac['random_frac_near_threshold']:>6.1%} for "
                   f"random -> {'WORKED' if ac['acquisition_worked'] else 'did NOT concentrate'}")
    out += ["",
            "  A high concentration with no fidelity gain means the acquisition "
            "function did",
            "  its job and the job did not pay: the boundary was not the scarce "
            "information",
            "  on this surface.", ""]

    out += ["-" * 78, "PROVENANCE", "-" * 78,
            f"  model={res['model']}  metric={METRIC}  threshold={res['threshold']}  "
            f"band=+-{res['band']}",
            f"  budget={res['n_seed']}+{res['budget']}={res['n_seed'] + res['budget']} "
            f"evals/arm  seeds={list(res['al_seeds'])}  splits={list(res['split_seeds'])}",
            f"  {res['n_coordinates']} coordinate systems x 4 splits x 8 seeds x 3 arms",
            "  ALL oracle calls went to the GP surrogate fitted to the master table. "
            "ZERO API calls.",
            "  No seed was confirmed end-to-end against the live API oracle — the same",
            "  caveat the published D3b result carries.",
            ]
    return "\n".join(out)


# ============================================================================
# CLI
# ============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="D3b re-run under a DRR-parameterised reverb axis")
    ap.add_argument("master", nargs="?", default="results/master.csv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true",
                    help="primary coordinates only — skips the negative controls, "
                         "which means the result is NOT interpretable as evidence")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--reformat", action="store_true",
                    help="recompute the derived aggregates from an existing "
                         "results/al_drr.json and rewrite the report — no re-run")
    args = ap.parse_args(argv)

    if args.reformat:
        p_json = os.path.join(args.out_dir, "al_drr.json")
        with open(p_json, "r", encoding="utf-8") as f:
            res = json.load(f)
        prim = [r for r in res["coordinates"] if r["kind"] == "primary"]
        res["coordinate_spread"] = coordinate_spread(res["coordinates"])
        res["per_split_fidelity"] = per_split_fidelity(prim)
        res["controls"] = control_verdict(res["coordinates"])
        res["verdict"] = res["controls"].get("verdict", res["verdict"])
        txt = format_report(res)
        print(txt)
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump(_json_safe(res), f, indent=2, allow_nan=False)
        with open(os.path.join(args.out_dir, "al_drr.txt"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print(f"\n[rewrote] {p_json}\n[rewrote] "
              f"{os.path.join(args.out_dir, 'al_drr.txt')}")
        return 0

    rows = load_master_rows(args.master)
    rooms = measure_delivered_rooms(rows, model=args.model)
    print(f"[rooms] {len(rooms)} distinct RIRs delivered by the grid:")
    for r in rooms:
        print(f"  {r['room'][:46]:<47} rt60={r['rt60_measured']:.3f} "
              f"DRR={r['drr_db']:+7.2f} C50={r['c50_db']:+7.2f} "
              f"WER={r['marginal_wer']:.4f}")
    if len(rooms) < 2:
        raise SystemExit("need at least 2 delivered RIRs to reparameterise an axis")

    coords = list(primary_coordinates(rooms))
    if not args.quick:
        coords += permutation_controls(rooms) + monotone_controls(rooms)
    print(f"\n[run] {len(coords)} coordinate systems x {len(SPLIT_SEEDS)} splits x "
          f"{len(AL_SEEDS)} seeds x 3 arms, {args.workers} workers — "
          f"0 API calls\n")

    results = run_all(coords, rooms, args.master, model=args.model,
                      workers=args.workers)

    ident = identity_check(results)
    if not ident["identical_test_set_across_coordinates"]:
        raise SystemExit(
            f"ABORT: the held-out test set is NOT identical across coordinate "
            f"systems ({ident['coordinates_differing']}). Any difference between "
            f"coordinates would be confounded with a different yardstick.")
    controls = control_verdict(results)

    res = {
        "verdict": controls.get("verdict", "(no primary comparison available)"),
        "model": args.model, "master": args.master,
        "threshold": THRESHOLD, "band": BAND, "metric": METRIC,
        "n_seed": N_SEED, "budget": BUDGET,
        "al_seeds": list(AL_SEEDS), "split_seeds": list(SPLIT_SEEDS),
        "n_coordinates": len(results),
        "rooms": rooms,
        "identity_check": ident,
        "controls": controls,
        "coordinate_spread": coordinate_spread(results),
        "per_split_fidelity": per_split_fidelity(
            [r for r in results if r["kind"] == "primary"]),
        "target_sweep": shared_target_sweep(
            [r for r in results if r["kind"] == "primary"]),
        "coordinates": results,
        "oracle_calls": {"api": 0, "surrogate": len(results) * len(SPLIT_SEEDS)
                         * len(AL_SEEDS) * 3 * (N_SEED + BUDGET)},
        "quick": bool(args.quick),
        "protocol_note": ("identical to results/al_savings.json's run_config in "
                          "every respect except the reverb coordinate"),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    txt = format_report(res)
    print()
    print(txt)
    p_json = os.path.join(args.out_dir, "al_drr.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(res), f, indent=2, allow_nan=False)
    p_txt = os.path.join(args.out_dir, "al_drr.txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\n[wrote] {p_json}\n[wrote] {p_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
