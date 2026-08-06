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

Reuse, not reimplementation: the master-table schema/failure handling comes from
`analysis/__init__.py`, and dead zones come from `model_compare.dead_zone_flags`
(so D4 flags exactly the cells D1 and L1 flag).

    python3 -m analysis.sim2real results/master.csv results/master_sim.csv

Deps: numpy, scipy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np

# Allow `python analysis/sim2real.py` as well as `import analysis.sim2real`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis import (                       # noqa: E402  (after the path shim)
    as_float, failure_summary, load_master_table, split_by_model, split_failures,
)
from model_compare import dead_zone_flags    # noqa: E402  — REUSED, not reimplemented
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


def aggregate_conditions(rows: Sequence[dict], model: str | None = None) -> list[dict]:
    """
    Master-table rows (one per clip x condition) -> one row per CONDITION.

    Failed rows are split out and NOT averaged in (a failure sentinel scores
    WER 1.0 with NaN confidence; averaging it manufactures a fake dead zone —
    see analysis/__init__.py). Each output row carries `wer` and `mean_conf` under
    exactly those names, so it drops straight into `model_compare.dead_zone_flags`.
    """
    ok, _bad = split_failures(_select_model(rows, model))
    by_cond: dict[str, list[dict]] = {}
    for r in ok:
        by_cond.setdefault(str(r.get("condition_name", "?")), []).append(r)

    out: list[dict] = []
    for name, group in sorted(by_cond.items()):
        wer = _mean([r.get("wer") for r in group])
        rec = {
            "condition_name": name,
            "n": len(group),
            "wer": wer,
            "wer_sd": float(np.std([as_float(r.get("wer")) for r in group])) if group else float("nan"),
            "mean_conf": _mean([r.get("mean_conf") for r in group]),
            MEASURED_RT60: _mean([r.get(MEASURED_RT60) for r in group]),
            REQUESTED_RT60: _mean([r.get(REQUESTED_RT60) for r in group]),
            "model": str(group[0].get("model", "unknown")),
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
# PAIRING — on MEASURED RT60 (the whole methodological point)
# ============================================================================

def pair_conditions(real_rows: Sequence[dict], sim_rows: Sequence[dict],
                    model: str | None = None, rt60_tol: float = RT60_TOL) -> dict:
    """
    Pair each real-run condition with the sim-run condition that delivered the
    same acoustics: identical non-reverb factors AND measured RT60 within
    `rt60_tol`. The requested `rt60` column is deliberately never consulted.

    One-to-one inside each non-reverb group (Hungarian assignment on
    |delta measured RT60|): greedy nearest-match could spend the same simulated
    condition twice and double-count it in the average.
    """
    real_agg = aggregate_conditions(real_rows, model)
    sim_agg = aggregate_conditions(sim_rows, model)
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
        "wer_real": r["wer"],
        "wer_sim": s["wer"],
        "gap": float(s["wer"] - r["wer"]),          # sign: + means sim is WORSE
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

def dead_zone_agreement(paired: dict, wer_hi: float = WER_HI,
                        conf_pct_hi: float = CONF_PCT_HI) -> dict:
    """
    Compare the D1 dead-zone SET (confidently wrong) between the two runs.

    `dead_zone_flags` is imported from model_compare, not reimplemented, so D4
    flags exactly the cells D1/L1 flag. The within-run confidence percentile is
    computed over each run's WHOLE condition table (never inside the paired
    subset): confidence is only meaningful relative to that run's own
    distribution, and re-percentiling a subset collapses the ranking.

    Sets are labelled by the REAL condition name so they are comparable.
    """
    real_tbl, sim_tbl = paired["real_table"], paired["sim_table"]
    flags_r = dict(zip((c["condition_name"] for c in real_tbl),
                       dead_zone_flags(real_tbl, wer_hi, conf_pct_hi)))
    flags_s = dict(zip((c["condition_name"] for c in sim_tbl),
                       dead_zone_flags(sim_tbl, wer_hi, conf_pct_hi)))

    real_set, sim_set = set(), set()
    for p in paired["pairs"]:
        label = p["condition_real"]
        if flags_r.get(p["condition_real"], False):
            real_set.add(label)
        if flags_s.get(p["condition_sim"], False):
            sim_set.add(label)

    both = real_set & sim_set
    union = real_set | sim_set
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
    }


# ============================================================================
# REPORT + PLOT PAYLOAD
# ============================================================================

def sim2real_report(real_rows: Sequence[dict], sim_rows: Sequence[dict],
                    model: str | None = None, rt60_tol: float = RT60_TOL,
                    wer_hi: float = WER_HI, conf_pct_hi: float = CONF_PCT_HI,
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """One call: pair -> level gap -> rank agreement -> dead-zone set -> payload."""
    paired = pair_conditions(real_rows, sim_rows, model=model, rt60_tol=rt60_tol)
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
        "headline": {
            "model": res["model"],
            "n_pairs": len(pairs),
            "mean_gap": res["level"]["mean_gap"],
            "ci": [res["level"]["ci_lo"], res["level"]["ci_hi"]],
            "spearman": res["order"]["spearman"],
            "kendall": res["order"]["kendall"],
            "dead_zone_jaccard": dz["jaccard"],
            "verdict": res["headline"]["verdict"],
        },
    }


def format_sim2real(res: dict) -> str:
    """The human-readable D4 block — the two numbers plus the interpretation."""
    lv, od, dz = res["level"], res["order"], res["dead_zones"]
    L = [f"SIM-VS-REAL (D4) — model {res['model']}, {len(res['pairs'])} paired "
         f"conditions",
         f"  matched on : {res['matched_on']} (NOT the requested rt60); "
         f"max |delta| {res['max_abs_rt60_delta']:.3f} s "
         f"(tolerance {res['rt60_tol']:.3f})",
         f"  LEVEL      : {lv['verdict']}",
         f"               mean |gap| {lv['mean_abs_gap'] * 100:.1f} pts, "
         f"median {lv['median_gap'] * 100:+.1f} pts",
         f"  ORDER      : Spearman rho = {od['spearman']:.3f} "
         f"(p={od['spearman_p']:.2g}), Kendall tau = {od['kendall']:.3f} "
         f"-> {od['verdict']}",
         f"  DEAD ZONES : real {dz['n_real']}, sim {dz['n_sim']}, both "
         f"{dz['n_both']} -> Jaccard {dz['jaccard']:.2f}, "
         f"recall {dz['recall']:.2f}"]
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
    args = ap.parse_args(argv)

    real_rows = load_master_table(args.real_table)
    sim_rows = load_master_table(args.sim_table)
    models = ([args.model] if args.model
              else sorted(set(split_by_model(real_rows)) & set(split_by_model(sim_rows))))
    payloads = {}
    for m in models:
        res = sim2real_report(real_rows, sim_rows, model=m,
                              rt60_tol=args.rt60_tol, wer_hi=args.wer_hi,
                              conf_pct_hi=args.conf_pct_hi)
        print(format_sim2real(res))
        print()
        payloads[res["model"]] = res["plot"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payloads, fh, indent=2)
    print(f"wrote plot payload -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
