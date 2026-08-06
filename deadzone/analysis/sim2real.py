"""
analysis/sim2real.py — D4: the sim-vs-real gap (SPEC §4, §5.4, A.R5.6).

Everything else in Deadzone is measured with REAL ingredients: measured RIRs,
recorded noise, only the assembly controlled (SPEC §4's realism principle). This
layer asks the question that any reader who builds a *synthetic-only* testbed
actually cares about:

    If you had simulated the rooms with pyroomacoustics instead of measuring
    them, would you have reached the same conclusions?

It consumes two master results tables — the real-RIR run and the sim-RIR run,
same clips, same condition list, only the reverb ingredient swapped — and answers
in two numbers plus a set:

  1. LEVEL: the mean signed WER gap (sim - real) with a bootstrap CI.
     "Simulation under/over-estimates WER by X points [CI a, b]."
  2. ORDER: the SPEARMAN rank correlation across conditions. This is the more
     useful claim. Absolute WER offsets are easy to explain away (different
     rooms, different mic distances); what a sim-only benchmark really needs is
     for the *ordering* of conditions to survive. A high rank correlation with a
     large level offset is the genuinely quotable result: "use synthetic RIRs to
     RANK conditions, not to predict absolute WER."
  3. DECISION: is the DEAD-ZONE SET (D1's headline output) the same under sim
     RIRs? Rank correlation can be high while the confidently-wrong set differs,
     and the dead-zone set is what a practitioner actually acts on.

THE TRAP THIS FILE EXISTS TO AVOID (SPEC A.R5.6 gotcha). Conditions are paired on
the **measured** Schroeder RT60 of the RIR each run actually used
(`rir_rt60_measured`), NEVER on the requested `rt60` factor value. Two reasons,
both fatal if ignored:
  * `AssetLibrary.pick_rir` snaps a requested rt60 to the CLOSEST available
    measured RIR, and the two libraries snap differently;
  * pyroomacoustics' realized RT60 diverges from the Sabine target it was asked
    for by up to ~30% (see make_sim_rirs.py).
Match on the request and you are comparing, say, a 0.9 s real room with a 1.2 s
simulated one — and reporting the reverb mismatch you introduced yourself as a
sim2real finding. Every join here goes through `rir_rt60_measured` and every pair
is bounded by `rt60_tol`; anything outside it is reported UNMATCHED, not fudged.

THE SECOND HALF OF THE SAME TRAP — THE CLIP SET. "Same clips" is half of the
premise above and it is the half that is easy to lose, because nothing about the
two tables looks wrong when it breaks. The real arm ran all 40 utterances; the
sim arm was deliberately run on the 10-clip AL subset to save API spend. Average
a condition's WER over 40 clips on one side and over a *different-sized* 10-clip
set on the other and the difference you report is (RIR provenance) + (clip
difficulty), with no way to separate them afterwards. On this project's own grid
that confound was worth 7.8 WER points out of a 19.9-point "finding" — i.e. 39%
of the headline number was the corpus, not the simulator. Counterfactual
isolation is the entire premise of the instrument (SPEC §1), so the one layer
whose job is to validate the simulation must not be the layer that breaks it.

So: both arms are restricted to the INTERSECTION of their clip sets before any
aggregation (`clip_intersection`), and the restriction is REPORTED — clip counts
per arm, the common set, and the rows dropped — in the payload and in the
formatted block. A partial mismatch is the expected situation and is handled by
restriction, not by an exception; only an empty or implausibly small intersection
raises (`ClipSetMismatchError`). What must never happen is that it is silent.

THE THIRD TRAP — WHICH WER THE DEAD ZONES ARE FLAGGED ON. A clip whose transcript
comes back EMPTY scores WER 1.0 with 100% deletions and carries NO per-word
confidence, so it inflates a condition's mean WER while contributing nothing to
its `mean_conf`. Thresholding one against the other compares two populations (see
`analysis/__init__.py`, the second trap, and `analysis/confidence_gap.py`). D4's
dead-zone sets are therefore flagged on `wer_spoke` — the WER over exactly the
clips that returned a confidence — via `confidence_gap.condition_flags`, which is
also what D1 and L1 use, so all three layers flag the same cells by construction.

The LEVEL and ORDER claims above are deliberately NOT affected: `gap`, the
Spearman and the Kendall compare two arms' CORPUS severity (`wer` over every
clip) and contain no confidence term at all, so all-clips is the right estimand
there and restricting it to the spoke subset would silently discount each arm's
worst clips. Only the dead-zone SET moves.

AND THE SET IS SCOPED TO THE COMMON CLIP SUBSET. Both arms are restricted to the
clips they share (10 on this project's grid, vs D1's 40), so D4's dead-zone sets
are computed WITHIN that subset and will not coincide with the 40-clip D1 table.
That is correct — a 10-clip and a 40-clip dead zone are different measurements —
and it is stated in the payload (`clip_scope`) and in the formatted block so the
two can never be read as the same set.

Reuse, not reimplementation: the master-table schema/failure handling comes from
`deadzone/analysis/__init__.py`, and dead zones come from
`confidence_gap.condition_flags` over `model_compare.dead_zone_flags` (so D4
flags exactly the cells D1 and L1 flag).

    ./.venv/bin/python -m deadzone.analysis.sim2real results/master.csv results_sim/master_sim.csv

Note the SECOND path: `results_sim/`, not `results/`. The simulated arm keeps its
own results directory on purpose. The run cache is keyed on
`(clip_id, condition_name, model)` and does NOT encode which RIR library produced
the row, so a shared cache would be 100% false hits and would report a sim-vs-real
gap of exactly zero (SPEC B.2 item 7). Pointing this at `results/master_sim.csv`
finds nothing; pointing it at the real table twice reports no gap at all.

Deps: numpy, scipy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np

# Allow `python deadzone/analysis/sim2real.py` as well as `import deadzone.analysis.sim2real`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.analysis import (                       # noqa: E402  (after the path shim)
    as_float, failure_summary, is_silent_row, load_master_table, silence_summary,
    split_by_model, split_failures,
)
# REUSED, not reimplemented — D4 flags exactly the cells D1/L1 flag, on the same
# estimand (wer_spoke), with mute conditions held out the same way.
from deadzone.analysis.confidence_gap import condition_flags   # noqa: E402
from scipy.optimize import linear_sum_assignment   # noqa: E402
from scipy.stats import kendalltau, spearmanr      # noqa: E402


# ============================================================================
# CONSTANTS
# ============================================================================

# The join key. Named as a constant so no caller can typo its way onto `rt60`.
MEASURED_RT60 = "rir_rt60_measured"     # what the run actually DELIVERED
REQUESTED_RT60 = "rt60"                 # what the condition ASKED for — never a key

# A pair is only a pair if both arms delivered the same amount of reverb. Matches
# make_sim_rirs.PAIR_TOL_S: the RIR sets were built to satisfy exactly this bound.
RT60_TOL = 0.05

# Everything except reverb must be identical for two conditions to be comparable.
MATCH_FACTORS: tuple[str, ...] = ("snr_db", "noise_type", "codec", "mic_rolloff")

# Dead-zone quadrant. Same defaults as model_compare / D1 — if these drift, D4
# reports a different dead-zone set than the headline layer and the two disagree.
WER_HI = 0.30
CONF_PCT_HI = 0.60

# Verdict thresholds for the rank-agreement claim.
RANK_STRONG = 0.80      # >= : ordering preserved, a sim-only benchmark can rank
RANK_WEAK = 0.50        # <  : ordering not preserved, sim-only rankings unsafe

# The clip-set invariant. A PARTIAL mismatch is the expected case (the sim arm
# was run on the 10-clip AL subset by design) and is handled by restricting both
# arms to the intersection and saying so. Below this many common clips there is
# nothing worth restricting to: a per-condition mean over one or two utterances
# is dominated by which utterances they were, so we refuse instead of publishing
# a gap that is really a two-clip anecdote.
MIN_COMMON_CLIPS = 3


class ClipSetMismatchError(ValueError):
    """
    Raised when the two arms share too few clips to be comparable at all.

    Loud on purpose, and deliberately NOT raised for the ordinary 10-vs-40 case:
    a silent inner join would still produce a full pair table, and every gap in
    it would be a mixture of an RIR-provenance effect and a clip-difficulty
    effect with no way to separate them afterwards. Restriction plus a reported
    clip census is the fix; this exception is only the floor under it.
    """


# ============================================================================
# PER-CONDITION AGGREGATION
# ============================================================================

def _select_model(rows: Sequence[dict], model: str | None) -> list[dict]:
    """
    Restrict to one model. A table holding both Nova-3 and Whisper rows would
    otherwise average two different systems' WERs into one "sim2real gap", and
    the confidence column would mix two incomparable scales (model_compare's
    headline caveat). So: one model or an explicit failure, never a silent mix.
    """
    by_model = split_by_model(rows)
    if model is not None:
        if model not in by_model:
            raise KeyError(f"model {model!r} not in table (have: {sorted(by_model)})")
        return by_model[model]
    if len(by_model) > 1:
        raise ValueError(
            f"table holds {len(by_model)} models {sorted(by_model)} — pass "
            f"model=... explicitly; averaging WER across model families is "
            f"meaningless and their confidences are not on one scale"
        )
    return next(iter(by_model.values())) if by_model else []


def _mean(vals: Sequence[float]) -> float:
    a = np.asarray([as_float(v) for v in vals], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def usable_rows(rows: Sequence[dict], model: str | None = None) -> list[dict]:
    """
    The rows that may contribute a measurement: this model's, minus the failures.

    Both halves matter and both are exclusions, not filters: a `failed=True` row
    is a MISSING measurement carrying WER 1.0 and NaN confidence (see
    analysis/__init__.py), and another model's rows are a different system. This
    is also the population the clip census is taken over, so a clip that only
    ever failed on one arm is correctly not counted as a clip that arm ran.
    """
    ok, _bad = split_failures(_select_model(rows, model))
    return ok


def aggregate_conditions(rows: Sequence[dict], model: str | None = None,
                         clips: Sequence[str] | None = None) -> list[dict]:
    """
    Master-table rows (one per clip x condition) -> one row per CONDITION.

    Failed rows are split out and NOT averaged in (a failure sentinel scores
    WER 1.0 with NaN confidence; averaging it manufactures a fake dead zone —
    see analysis/__init__.py). Each output row carries `wer` and `mean_conf` under
    exactly those names, so it drops straight into `model_compare.dead_zone_flags`.

    `clips`, when given, restricts to exactly that clip set BEFORE averaging.
    Callers on the sim2real path always pass the intersection of the two arms'
    clip sets — see `clip_intersection` for why averaging two different clip sets
    silently folds clip difficulty into the reported gap.
    """
    ok = usable_rows(rows, model)
    if clips is not None:
        keep = set(clips)
        ok = [r for r in ok if str(r.get("clip_id")) in keep]
    by_cond: dict[str, list[dict]] = {}
    for r in ok:
        by_cond.setdefault(str(r.get("condition_name", "?")), []).append(r)

    out: list[dict] = []
    for name, group in sorted(by_cond.items()):
        wer = _mean([r.get("wer") for r in group])
        # THE PAIRED SUBSET (see the module docstring's third trap). `spoke` is
        # exactly the clips whose confidence is inside `mean_conf`, so anything
        # computed under it is on the same estimand as the confidence. `wer`
        # above is over EVERY clip and must never be thresholded against one.
        conf_arr = np.array([as_float(r.get("mean_conf")) for r in group],
                            dtype=float)
        wer_arr = np.array([as_float(r.get("wer")) for r in group], dtype=float)
        spoke = np.isfinite(conf_arr)
        silent = np.array([is_silent_row(r) for r in group], dtype=bool)
        rec = {
            "condition_name": name,
            "n": len(group),
            # --- all-clips estimand: corpus severity, what LEVEL/ORDER use ----
            "wer": wer,
            "wer_all_clips": wer,
            "wer_sd": float(np.std([as_float(r.get("wer")) for r in group])) if group else float("nan"),
            "mean_conf": _mean([r.get("mean_conf") for r in group]),
            # --- paired estimand: the only WER the confidence may be judged on -
            "wer_spoke": (_mean(wer_arr[spoke]) if spoke.any() else float("nan")),
            "n_spoke": int(spoke.sum()),
            "n_silent": int(silent.sum()),
            "silent_frac": float(silent.mean()) if len(group) else float("nan"),
            "n_conf_without_words": int((spoke & silent).sum()),
            "mute": bool(not spoke.any()),
            MEASURED_RT60: _mean([r.get(MEASURED_RT60) for r in group]),
            REQUESTED_RT60: _mean([r.get(REQUESTED_RT60) for r in group]),
            "model": str(group[0].get("model", "unknown")),
            "n_clips": len({str(r.get("clip_id")) for r in group}),
            "rir_key": group[0].get("rir_key"),
            "n_ref": int(np.nansum([as_float(r.get("n_ref")) for r in group])),
        }
        for k in MATCH_FACTORS:
            v = group[0].get(k)
            rec[k] = as_float(v) if k in ("snr_db", "mic_rolloff") else v
        out.append(rec)
    return out


def _match_key(cond: dict) -> tuple:
    """The identity of a condition MINUS reverb (reverb is matched on measurement)."""
    key = []
    for k in MATCH_FACTORS:
        v = cond.get(k)
        key.append(round(float(v), 6) if isinstance(v, (int, float))
                   and not isinstance(v, bool) else str(v))
    return tuple(key)


# ============================================================================
# CLIP CENSUS — the other half of "same clips, same conditions"
# ============================================================================

def clip_intersection(real_rows: Sequence[dict], sim_rows: Sequence[dict],
                      model: str | None = None,
                      min_common: int = MIN_COMMON_CLIPS) -> dict:
    """
    The clip set both arms actually ran, plus the census that makes any
    restriction visible instead of silent.

    WHY THIS EXISTS. This module's premise is "same clips, same condition list,
    only the reverb ingredient swapped". The RT60 machinery below enforces the
    condition half; nothing enforced the clip half, and on this project's own
    tables the real arm ran 40 utterances against the sim arm's 10-clip AL
    subset. Each condition's real mean was then taken over 40 clips and its sim
    mean over 10 *different* ones, so `wer_sim - wer_real` measured

        (simulated vs measured rooms)  +  (the 30 clips only one arm ever saw)

    and the second term is indistinguishable from the first once it is in the
    table. Measured cost on the real grid: 7.8 WER points of a 19.9-point gap.

    REPORT-AND-PROCEED, NOT RAISE. A 10-vs-40 mismatch is the EXPECTED situation
    — the sim arm is subset by design to save API spend — so refusing it would
    just delete a legitimate finding. The fix is to restrict both arms to the
    intersection and state the clip set the numbers were computed on. The only
    raise is the floor: an empty or implausibly small intersection (<
    `min_common`), where a per-condition mean is an anecdote about which two
    utterances happened to overlap.

    The census is taken over `usable_rows` (this model, failures excluded), so a
    clip that only ever failed on one arm is correctly not counted as run there.
    """
    real_ok = usable_rows(real_rows, model)
    sim_ok = usable_rows(sim_rows, model)
    clips_r = {str(r.get("clip_id")) for r in real_ok}
    clips_s = {str(r.get("clip_id")) for r in sim_ok}
    common = clips_r & clips_s

    if len(common) < max(1, int(min_common)):
        raise ClipSetMismatchError(
            f"the two arms share only {len(common)} clip(s) "
            f"(real {len(clips_r)}: {sorted(clips_r)[:6]}..., "
            f"sim {len(clips_s)}: {sorted(clips_s)[:6]}...) — fewer than the "
            f"{min_common} needed for a per-condition mean to describe an "
            f"acoustic condition rather than the particular utterances that "
            f"happened to overlap. Re-run one arm on the other's clip set; do "
            f"NOT compare unmatched clip sets, which folds clip difficulty into "
            f"the reported sim2real gap."
        )

    kept_r = sum(1 for r in real_ok if str(r.get("clip_id")) in common)
    kept_s = sum(1 for r in sim_ok if str(r.get("clip_id")) in common)
    matched = clips_r == clips_s
    return {
        "matched": bool(matched),
        "common": sorted(common),
        "n_common": len(common),
        "n_clips_real": len(clips_r),
        "n_clips_sim": len(clips_s),
        "real_only": sorted(clips_r - clips_s),
        "sim_only": sorted(clips_s - clips_r),
        "n_rows_real": len(real_ok),
        "n_rows_sim": len(sim_ok),
        "n_rows_real_kept": kept_r,
        "n_rows_sim_kept": kept_s,
        "n_rows_real_dropped": len(real_ok) - kept_r,
        "n_rows_sim_dropped": len(sim_ok) - kept_s,
        "min_common": int(min_common),
        "note": (
            f"clip sets identical ({len(common)} clips); no restriction applied"
            if matched else
            f"clip sets DIFFER (real {len(clips_r)}, sim {len(clips_s)}); both "
            f"arms restricted to the {len(common)} common clip(s) before "
            f"aggregation, dropping {len(real_ok) - kept_r} real and "
            f"{len(sim_ok) - kept_s} sim row(s). Every number below is computed "
            f"on that common clip set, not on either arm's full corpus."
        ),
    }


# ============================================================================
# PAIRING — on MEASURED RT60 (the whole methodological point)
# ============================================================================

def pair_conditions(real_rows: Sequence[dict], sim_rows: Sequence[dict],
                    model: str | None = None, rt60_tol: float = RT60_TOL,
                    min_common_clips: int = MIN_COMMON_CLIPS) -> dict:
    """
    Pair each real-run condition with the sim-run condition that delivered the
    same acoustics: identical non-reverb factors AND measured RT60 within
    `rt60_tol`. The requested `rt60` column is deliberately never consulted.

    One-to-one inside each non-reverb group (Hungarian assignment on
    |delta measured RT60|): greedy nearest-match could spend the same simulated
    condition twice and double-count it in the average.

    BOTH ARMS ARE FIRST RESTRICTED TO THEIR COMMON CLIP SET (`clip_intersection`)
    — the condition half of "same clips, same conditions" is enforced by the
    RT60 join below, and this is the clip half. The census is returned under
    `clip_match` and must be surfaced by every caller.
    """
    clip_match = clip_intersection(real_rows, sim_rows, model=model,
                                   min_common=min_common_clips)
    common = clip_match["common"]
    real_agg = aggregate_conditions(real_rows, model, clips=common)
    sim_agg = aggregate_conditions(sim_rows, model, clips=common)
    _require_measured_rt60(real_agg, "real")
    _require_measured_rt60(sim_agg, "sim")

    groups_r: dict[tuple, list[dict]] = {}
    groups_s: dict[tuple, list[dict]] = {}
    for c in real_agg:
        groups_r.setdefault(_match_key(c), []).append(c)
    for c in sim_agg:
        groups_s.setdefault(_match_key(c), []).append(c)

    pairs: list[dict] = []
    unmatched_real: list[dict] = []
    unmatched_sim: list[dict] = []

    for key in sorted(set(groups_r) | set(groups_s), key=str):
        rs = sorted(groups_r.get(key, []), key=lambda c: c[MEASURED_RT60])
        ss = sorted(groups_s.get(key, []), key=lambda c: c[MEASURED_RT60])
        if not rs or not ss:
            unmatched_real += [_unmatched(c, "no sim condition with these "
                                             "non-reverb factors") for c in rs]
            unmatched_sim += [_unmatched(c, "no real condition with these "
                                            "non-reverb factors") for c in ss]
            continue
        cost = np.abs(np.array([c[MEASURED_RT60] for c in rs])[:, None]
                      - np.array([c[MEASURED_RT60] for c in ss])[None, :])
        ri, si = linear_sum_assignment(cost)
        used_r, used_s = set(), set()
        for i, j in zip(ri, si):
            used_r.add(int(i))
            used_s.add(int(j))
            delta = float(ss[j][MEASURED_RT60] - rs[i][MEASURED_RT60])
            if abs(delta) > rt60_tol:
                unmatched_real.append(_unmatched(
                    rs[i], f"closest sim condition differs by {delta:+.3f} s "
                           f"measured RT60 (> {rt60_tol:.3f})"))
                unmatched_sim.append(_unmatched(ss[j], "measured RT60 out of tolerance"))
                continue
            pairs.append(_pair_record(rs[i], ss[j], delta))
        unmatched_real += [_unmatched(c, "unpaired (group size mismatch)")
                           for i, c in enumerate(rs) if i not in used_r]
        unmatched_sim += [_unmatched(c, "unpaired (group size mismatch)")
                          for j, c in enumerate(ss) if j not in used_s]

    pairs.sort(key=lambda p: p["condition_real"])
    return {
        "pairs": pairs,
        "unmatched_real": unmatched_real,
        "unmatched_sim": unmatched_sim,
        "real_table": real_agg,
        "sim_table": sim_agg,
        "clip_match": clip_match,
        "rt60_tol": float(rt60_tol),
        "matched_on": MEASURED_RT60,
        "max_abs_rt60_delta": (float(max(abs(p["rt60_delta"]) for p in pairs))
                               if pairs else float("nan")),
    }


def _require_measured_rt60(table: Sequence[dict], side: str) -> None:
    bad = [c["condition_name"] for c in table if not np.isfinite(c[MEASURED_RT60])]
    if bad:
        raise ValueError(
            f"{side} table has no {MEASURED_RT60} for {len(bad)} condition(s) "
            f"(e.g. {bad[:3]}). That column is the ONLY legitimate join key for "
            f"sim2real — the runner must record the delivered RT60 (SPEC A.R4.2). "
            f"Falling back to the requested {REQUESTED_RT60!r} would compare "
            f"different amounts of reverb."
        )


def _unmatched(cond: dict, why: str) -> dict:
    return {"condition_name": cond["condition_name"],
            MEASURED_RT60: cond[MEASURED_RT60], "wer": cond["wer"], "reason": why}


def _pair_record(r: dict, s: dict, delta: float) -> dict:
    rec = {
        "condition_real": r["condition_name"],
        "condition_sim": s["condition_name"],
        # ALL-CLIPS on both sides: the gap is a corpus-severity comparison with
        # no confidence term in it, so restricting it to the clips each arm
        # spoke on would discount exactly the clips the reverb destroyed.
        "wer_real": r["wer"],
        "wer_sim": s["wer"],
        "gap": float(s["wer"] - r["wer"]),          # sign: + means sim is WORSE
        # the paired estimand travels alongside, because the dead-zone sets are
        # flagged on it and a reader comparing the two tables needs both
        "wer_spoke_real": r["wer_spoke"],
        "wer_spoke_sim": s["wer_spoke"],
        "n_silent_real": r["n_silent"],
        "n_silent_sim": s["n_silent"],
        "conf_real": r["mean_conf"],
        "conf_sim": s["mean_conf"],
        "rt60_measured_real": r[MEASURED_RT60],
        "rt60_measured_sim": s[MEASURED_RT60],
        "rt60_delta": float(delta),
        "rt60_requested_real": r[REQUESTED_RT60],
        "rt60_requested_sim": s[REQUESTED_RT60],
        "n_real": r["n"],
        "n_sim": s["n"],
    }
    for k in MATCH_FACTORS:
        rec[k] = r[k]
    return rec


# ============================================================================
# LEVEL — mean signed gap + bootstrap CI
# ============================================================================

def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """
    Percentile bootstrap CI of the MEAN. Resampling is over CONDITIONS (the
    paired unit), not over clips: clips inside a condition are not independent
    draws, and resampling them would shrink the interval dishonestly.
    """
    x = np.asarray([as_float(v) for v in values], dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    if x.size == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(int(n_boot), x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def gap_summary(pairs: Sequence[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """
    The level claim: mean signed gap (sim - real) in WER points, with CI.
    Positive => the simulation OVERestimates WER (predicts more errors than the
    measured rooms actually produce); negative => it flatters the model.
    """
    gaps = np.asarray([p["gap"] for p in pairs], dtype=float)
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size == 0:
        return {"n": 0, "mean_gap": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "mean_abs_gap": float("nan"),
                "median_gap": float("nan"), "direction": "undetermined",
                "significant": False, "verdict": "no paired conditions"}
    lo, hi = bootstrap_ci(gaps, n_boot=n_boot, seed=seed)
    mean_gap = float(gaps.mean())
    significant = bool((lo > 0) or (hi < 0))
    direction = ("overestimates" if mean_gap > 0 else
                 "underestimates" if mean_gap < 0 else "matches")
    return {
        "n": int(gaps.size),
        "mean_gap": mean_gap,
        "ci_lo": lo, "ci_hi": hi,
        "mean_abs_gap": float(np.abs(gaps).mean()),
        "median_gap": float(np.median(gaps)),
        "direction": direction,
        "significant": significant,
        "verdict": (f"sim {direction} WER by {abs(mean_gap) * 100:.1f} points "
                    f"[95% CI {lo * 100:+.1f}, {hi * 100:+.1f}] over "
                    f"{gaps.size} paired conditions"
                    + ("" if significant else "; CI includes zero")),
    }


# ============================================================================
# ORDER — Spearman rank correlation (the more useful claim)
# ============================================================================

def rank_agreement(pairs: Sequence[dict]) -> dict:
    """
    Does the simulated testbed put the conditions in the SAME ORDER as the
    measured one? Spearman (headline) + Kendall tau (robust second opinion).

    Read this together with `gap_summary`: high rho + a large level offset means
    a sim-only benchmark is trustworthy for RANKING conditions and untrustworthy
    for absolute WER — which is exactly the actionable sentence for anyone who
    builds one.
    """
    a = np.asarray([p["wer_real"] for p in pairs], dtype=float)
    b = np.asarray([p["wer_sim"] for p in pairs], dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    out = {"n": int(a.size), "spearman": float("nan"), "spearman_p": float("nan"),
           "kendall": float("nan"), "kendall_p": float("nan"),
           "verdict": "too few paired conditions to rank"}
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return out
    rho, p = spearmanr(a, b)
    tau, tp = kendalltau(a, b)
    out.update({"spearman": float(rho), "spearman_p": float(p),
                "kendall": float(tau), "kendall_p": float(tp)})
    out["verdict"] = (
        "ordering preserved" if rho >= RANK_STRONG else
        "ordering partially preserved" if rho >= RANK_WEAK else
        "ordering NOT preserved"
    )
    return out


# ============================================================================
# DECISION — is the dead-zone SET the same under sim RIRs?
# ============================================================================

def _flag_map(table: Sequence[dict], wer_hi: float, conf_pct_hi: float,
              wer_key: str) -> dict[str, bool]:
    return dict(zip((c["condition_name"] for c in table),
                    (bool(f) for f in
                     condition_flags(table, wer_hi, conf_pct_hi, wer_key))))


def dead_zone_agreement(paired: dict, wer_hi: float = WER_HI,
                        conf_pct_hi: float = CONF_PCT_HI) -> dict:
    """
    Compare the D1 dead-zone SET (confidently wrong) between the two runs.

    FLAGGED ON `wer_spoke`, NOT ON THE ALL-CLIPS WER. A condition's `mean_conf`
    is averaged only over the clips that emitted words; its all-clips WER is
    inflated by the clips that emitted nothing and could carry no confidence.
    Testing "high WER *and* high confidence" across those two populations
    flags cells whose apparent danger is really their silence — the exact defect
    D1 was rebuilt around. `condition_flags` is D1's own function (over
    `model_compare.dead_zone_flags`), so all three layers flag the same cells,
    and it holds MUTE conditions out: a run that emitted nothing on any clip has
    no confidence, so it cannot be *confidently* wrong. Mute cells are counted
    and reported per arm instead of silently reading as "not a dead zone".

    The within-run confidence percentile is computed over each run's WHOLE
    condition table (never inside the paired subset): confidence is only
    meaningful relative to that run's own distribution, and re-percentiling a
    subset collapses the ranking.

    SCOPE. Both tables have already been restricted to the arms' common clip set
    (`pair_conditions` -> `clip_intersection`), so these sets are dead zones
    WITHIN that subset — 10 clips on this project's grid, against D1's 40. They
    are not the same measurement as D1's table and `clip_scope` says so.

    Sets are labelled by the REAL condition name so they are comparable.

    Both pairings are returned: `jaccard` is the corrected one, and
    `*_all_clips_pairing` preserves what the mismatched pairing reported so the
    published grid-v1 number stays reproducible rather than quietly restated.
    """
    real_tbl, sim_tbl = paired["real_table"], paired["sim_table"]

    def _sets(wer_key: str) -> tuple[set, set]:
        fr = _flag_map(real_tbl, wer_hi, conf_pct_hi, wer_key)
        fs = _flag_map(sim_tbl, wer_hi, conf_pct_hi, wer_key)
        rs, ss = set(), set()
        for p in paired["pairs"]:
            label = p["condition_real"]
            if fr.get(p["condition_real"], False):
                rs.add(label)
            if fs.get(p["condition_sim"], False):
                ss.add(label)
        return rs, ss

    real_set, sim_set = _sets("wer_spoke")
    real_all, sim_all = _sets("wer")
    both = real_set & sim_set
    union = real_set | sim_set
    union_all = real_all | sim_all

    def _n_mute(table: Sequence[dict]) -> int:
        return int(sum(1 for c in table if c.get("mute")))

    def _n_with_silence(table: Sequence[dict]) -> int:
        return int(sum(1 for c in table if as_float(c.get("n_silent"), 0) > 0))

    clips = paired["clip_match"]
    return {
        "n_paired": len(paired["pairs"]),
        "n_real": len(real_set), "n_sim": len(sim_set), "n_both": len(both),
        "jaccard": (len(both) / len(union)) if union else float("nan"),
        "recall": (len(both) / len(real_set)) if real_set else float("nan"),
        "precision": (len(both) / len(sim_set)) if sim_set else float("nan"),
        "same_set": real_set == sim_set,
        "agree": sorted(both),
        "real_only": sorted(real_set - sim_set),   # sim would MISS these dead zones
        "sim_only": sorted(sim_set - real_set),    # sim invents these
        "wer_hi": float(wer_hi), "conf_pct_hi": float(conf_pct_hi),
        "pairing": "same-subset (wer_spoke vs mean_conf, over the same clips)",
        # what the MISMATCHED all-clips pairing reported — kept, labelled, and
        # never the headline
        "all_clips_pairing": {
            "n_real": len(real_all), "n_sim": len(sim_all),
            "n_both": len(real_all & sim_all),
            "jaccard": ((len(real_all & sim_all) / len(union_all))
                        if union_all else float("nan")),
            "real_only": sorted(real_all - sim_all),
            "sim_only": sorted(sim_all - real_all),
        },
        # the categories a dead-zone count is meaningless without
        "silence": {
            "real": {"n_mute": _n_mute(real_tbl),
                     "n_conditions_with_silence": _n_with_silence(real_tbl),
                     "n_silence_driven": len(real_all - real_set)},
            "sim": {"n_mute": _n_mute(sim_tbl),
                    "n_conditions_with_silence": _n_with_silence(sim_tbl),
                    "n_silence_driven": len(sim_all - sim_set)},
        },
        # THE SCOPE, carried with the sets themselves so they can never be
        # quoted as if they were D1's 40-clip table.
        "clip_scope": {
            "n_clips": clips["n_common"],
            "clips": list(clips["common"]),
            "note": (f"dead zones computed WITHIN the {clips['n_common']}-clip "
                     f"set both arms ran; D1's table is over the full corpus, so "
                     f"these sets are a different measurement and are not "
                     f"expected to coincide with it"),
        },
    }


# ============================================================================
# REPORT + PLOT PAYLOAD
# ============================================================================

def sim2real_report(real_rows: Sequence[dict], sim_rows: Sequence[dict],
                    model: str | None = None, rt60_tol: float = RT60_TOL,
                    wer_hi: float = WER_HI, conf_pct_hi: float = CONF_PCT_HI,
                    n_boot: int = 2000, seed: int = 0,
                    min_common_clips: int = MIN_COMMON_CLIPS) -> dict:
    """One call: pair -> level gap -> rank agreement -> dead-zone set -> payload."""
    paired = pair_conditions(real_rows, sim_rows, model=model, rt60_tol=rt60_tol,
                             min_common_clips=min_common_clips)
    # Failures are counted for the MODEL UNDER ANALYSIS only. Counting them over
    # the whole table would report another arm's outages (Whisper timing out, say)
    # as this arm's missing measurements — the header number would be wrong in the
    # one place a reader checks whether the analysis rests on enough data.
    real_rows = _select_model(real_rows, model)
    sim_rows = _select_model(sim_rows, model)
    level = gap_summary(paired["pairs"], n_boot=n_boot, seed=seed)
    order = rank_agreement(paired["pairs"])
    dz = dead_zone_agreement(paired, wer_hi, conf_pct_hi)
    res = {
        "model": model or (paired["real_table"][0]["model"]
                           if paired["real_table"] else "unknown"),
        "pairs": paired["pairs"],
        "unmatched_real": paired["unmatched_real"],
        "unmatched_sim": paired["unmatched_sim"],
        "clip_match": paired["clip_match"],
        "rt60_tol": paired["rt60_tol"],
        "matched_on": paired["matched_on"],
        "max_abs_rt60_delta": paired["max_abs_rt60_delta"],
        "level": level,
        "order": order,
        "dead_zones": dz,
        "failures_real": failure_summary(real_rows),
        "failures_sim": failure_summary(sim_rows),
        "headline": _headline(level, order),
    }
    res["plot"] = plot_payload(res)
    return res


def _headline(level: dict, order: dict) -> dict:
    """
    The combined verdict — the sentence a practitioner quotes. The interesting
    cell of the 2x2 is (order preserved, level offset): sim-only benchmarks are
    then usable for ranking and misleading for absolute numbers.
    """
    rho = order.get("spearman", float("nan"))
    offset = level.get("significant", False)
    if not np.isfinite(rho):
        verdict, advice = "INSUFFICIENT DATA", "too few paired conditions"
    elif rho >= RANK_STRONG and offset:
        verdict = "ORDER PRESERVED, LEVEL OFFSET"
        advice = ("a pyroomacoustics-only testbed ranks conditions like the "
                  "measured one but reads "
                  f"{abs(level['mean_gap']) * 100:.1f} points "
                  f"{'pessimistic' if level['mean_gap'] > 0 else 'optimistic'} "
                  "in absolute WER — use it to rank, not to quote numbers")
    elif rho >= RANK_STRONG:
        verdict = "SIM TRACKS REAL"
        advice = ("ordering AND level agree within the CI — synthetic RIRs "
                  "reproduce the measured picture for this factor set")
    elif rho >= RANK_WEAK:
        verdict = "PARTIAL AGREEMENT"
        advice = ("ordering only partially survives — check which conditions "
                  "move before trusting a sim-only ranking")
    else:
        verdict = "ORDER NOT PRESERVED"
        advice = ("a sim-only testbed reorders the conditions — its rankings do "
                  "not transfer to measured rooms")
    return {"verdict": verdict, "advice": advice,
            "spearman": rho, "mean_gap": level.get("mean_gap", float("nan"))}


def plot_payload(res: dict) -> dict:
    """
    JSON-serializable payload for the dashboard (E2):
      * `scatter`   — one point per paired condition, real WER (x) vs sim WER (y),
                      plus the dead-zone flags, for an identity-line plot;
      * `identity`  — the [lo, hi] range to draw y = x over;
      * `gap_vs_rt60` — signed gap against delivered reverb, which is where a
                      systematic sim bias shows itself;
      * `headline`  — the two numbers and the verdict.
    """
    pairs = res["pairs"]
    dz = res["dead_zones"]
    real_dz, sim_dz = set(dz["agree"]) | set(dz["real_only"]), \
        set(dz["agree"]) | set(dz["sim_only"])
    scatter = [{
        "condition": p["condition_real"],
        "condition_sim": p["condition_sim"],
        "wer_real": p["wer_real"], "wer_sim": p["wer_sim"], "gap": p["gap"],
        "wer_spoke_real": p["wer_spoke_real"], "wer_spoke_sim": p["wer_spoke_sim"],
        "n_silent_real": p["n_silent_real"], "n_silent_sim": p["n_silent_sim"],
        "rt60_measured_real": p["rt60_measured_real"],
        "rt60_measured_sim": p["rt60_measured_sim"],
        "snr_db": p["snr_db"], "noise_type": p["noise_type"], "codec": p["codec"],
        "n": p["n_real"],
        "dead_zone_real": p["condition_real"] in real_dz,
        "dead_zone_sim": p["condition_real"] in sim_dz,
    } for p in pairs]
    vals = [v for p in pairs for v in (p["wer_real"], p["wer_sim"])
            if np.isfinite(v)]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    return {
        "scatter": scatter,
        "identity": [float(lo), float(hi)],
        "gap_vs_rt60": sorted(
            ({"rt60_measured_real": p["rt60_measured_real"], "gap": p["gap"],
              "condition": p["condition_real"]} for p in pairs),
            key=lambda d: d["rt60_measured_real"]),
        "clip_match": res["clip_match"],
        "headline": {
            "model": res["model"],
            "n_pairs": len(pairs),
            "mean_gap": res["level"]["mean_gap"],
            "ci": [res["level"]["ci_lo"], res["level"]["ci_hi"]],
            "spearman": res["order"]["spearman"],
            "kendall": res["order"]["kendall"],
            # flagged on wer_spoke, over the common clip subset — both facts
            # travel with the number so it cannot be read as D1's 40-clip set
            "dead_zone_jaccard": dz["jaccard"],
            "dead_zone_jaccard_all_clips_pairing":
                dz["all_clips_pairing"]["jaccard"],
            "dead_zone_pairing": dz["pairing"],
            "dead_zone_clip_scope": dz["clip_scope"]["note"],
            "n_dead_zones_real": dz["n_real"], "n_dead_zones_sim": dz["n_sim"],
            "n_mute_real": dz["silence"]["real"]["n_mute"],
            "n_mute_sim": dz["silence"]["sim"]["n_mute"],
            "verdict": res["headline"]["verdict"],
            # The clip set the numbers above were computed on. Carried into the
            # headline itself so a consumer that reads nothing else still cannot
            # quote the gap without knowing what it is a gap over.
            "n_clips": res["clip_match"]["n_common"],
            "clips_matched": res["clip_match"]["matched"],
        },
    }


def format_sim2real(res: dict) -> str:
    """The human-readable D4 block — the two numbers plus the interpretation."""
    lv, od, dz = res["level"], res["order"], res["dead_zones"]
    cm = res["clip_match"]
    L = [f"SIM-VS-REAL (D4) — model {res['model']}, {len(res['pairs'])} paired "
         f"conditions",
         f"  matched on : {res['matched_on']} (NOT the requested rt60); "
         f"max |delta| {res['max_abs_rt60_delta']:.3f} s "
         f"(tolerance {res['rt60_tol']:.3f})",
         f"  CLIP SET   : real arm {cm['n_clips_real']} clips, sim arm "
         f"{cm['n_clips_sim']}, common {cm['n_common']} "
         f"-> {'MATCHED' if cm['matched'] else 'RESTRICTED to the intersection'}",
         f"               {cm['note']}"]
    if not cm["matched"]:
        L += [f"               rows dropped: real "
              f"{cm['n_rows_real_dropped']}/{cm['n_rows_real']}, sim "
              f"{cm['n_rows_sim_dropped']}/{cm['n_rows_sim']}",
              f"               computed on clips: "
              f"{', '.join(cm['common'][:12])}"
              + (" ..." if len(cm["common"]) > 12 else "")]
    L += [f"  LEVEL      : {lv['verdict']}",
         f"               mean |gap| {lv['mean_abs_gap'] * 100:.1f} pts, "
         f"median {lv['median_gap'] * 100:+.1f} pts",
         f"  ORDER      : Spearman rho = {od['spearman']:.3f} "
         f"(p={od['spearman_p']:.2g}), Kendall tau = {od['kendall']:.3f} "
         f"-> {od['verdict']}",
         f"  DEAD ZONES : real {dz['n_real']}, sim {dz['n_sim']}, both "
         f"{dz['n_both']} -> Jaccard {dz['jaccard']:.2f}, "
         f"recall {dz['recall']:.2f}",
         f"               flagged on wer_spoke (the clips each arm actually "
         f"spoke on) — the",
         f"               all-clips pairing would report Jaccard "
         f"{dz['all_clips_pairing']['jaccard']:.2f} from real "
         f"{dz['all_clips_pairing']['n_real']} / sim "
         f"{dz['all_clips_pairing']['n_sim']}",
         f"               SCOPE: {dz['clip_scope']['note']}",
         f"               silence: real {dz['silence']['real']['n_mute']} mute / "
         f"{dz['silence']['real']['n_conditions_with_silence']} conds with a "
         f"silent clip / {dz['silence']['real']['n_silence_driven']} "
         f"silence-driven;",
         f"                        sim  {dz['silence']['sim']['n_mute']} mute / "
         f"{dz['silence']['sim']['n_conditions_with_silence']} conds with a "
         f"silent clip / {dz['silence']['sim']['n_silence_driven']} "
         f"silence-driven",
         f"               (mute = no words on ANY clip: no confidence exists, so "
         f"no dead zone can",
         f"                be flagged there — the worst cells are invisible to a "
         f"confidence monitor)"]
    if dz["real_only"]:
        L.append(f"               sim MISSES: {', '.join(dz['real_only'][:5])}"
                 + (" ..." if len(dz["real_only"]) > 5 else ""))
    if dz["sim_only"]:
        L.append(f"               sim INVENTS: {', '.join(dz['sim_only'][:5])}"
                 + (" ..." if len(dz["sim_only"]) > 5 else ""))
    if res["unmatched_real"]:
        L.append(f"  UNMATCHED  : {len(res['unmatched_real'])} real condition(s) "
                 f"had no in-tolerance sim partner "
                 f"(e.g. {res['unmatched_real'][0]['reason']})")
    L += [f"  VERDICT    : {res['headline']['verdict']}",
          f"               {res['headline']['advice']}"]
    fr, fs_ = res["failures_real"], res["failures_sim"]
    L.append(f"  failures   : real {fr['n_failed']}/{fr['n_rows']}, "
             f"sim {fs_['n_failed']}/{fs_['n_rows']} (excluded, not averaged in)")
    return "\n".join(L)


# ============================================================================
# CLI
# ============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D4 — the sim-vs-real gap")
    ap.add_argument("real_table", help="master table from the REAL-RIR run")
    ap.add_argument("sim_table", help="master table from the SIM-RIR run")
    ap.add_argument("--model", default=None,
                    help="model to analyse (required if the tables hold several)")
    ap.add_argument("--rt60-tol", type=float, default=RT60_TOL)
    ap.add_argument("--wer-hi", type=float, default=WER_HI)
    ap.add_argument("--conf-pct-hi", type=float, default=CONF_PCT_HI)
    ap.add_argument("--out", default="results/sim2real.json")
    ap.add_argument("--out-txt", default="results/sim2real.txt",
                    help="the printed report, verbatim (the clip census lives "
                         "here as well as in the JSON)")
    ap.add_argument("--min-common-clips", type=int, default=MIN_COMMON_CLIPS)
    args = ap.parse_args(argv)

    real_rows = load_master_table(args.real_table)
    sim_rows = load_master_table(args.sim_table)
    models = ([args.model] if args.model
              else sorted(set(split_by_model(real_rows)) & set(split_by_model(sim_rows))))
    payloads, blocks = {}, []
    for m in models:
        res = sim2real_report(real_rows, sim_rows, model=m,
                              rt60_tol=args.rt60_tol, wer_hi=args.wer_hi,
                              conf_pct_hi=args.conf_pct_hi,
                              min_common_clips=args.min_common_clips)
        block = format_sim2real(res)
        print(block)
        print()
        blocks.append(block)
        payloads[res["model"]] = res["plot"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payloads, fh, indent=2)
    print(f"wrote plot payload -> {args.out}")
    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
        with open(args.out_txt, "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(blocks) + "\n")
        print(f"wrote report      -> {args.out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
