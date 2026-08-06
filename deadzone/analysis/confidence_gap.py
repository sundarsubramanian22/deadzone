"""
analysis/confidence_gap.py — D1, THE HEADLINE: the confidence–accuracy gap, a.k.a.
the silent-failure map (SPEC §5.1, A.R5.1).

An aggregate WER says the model is wrong somewhere. It does not say the thing a
deployed voice agent needs: *does the model know it is wrong?* A model returning
0.45 confidence on a botched utterance is SAFE (re-prompt, escalate). A model
returning 0.93 on the same utterance is DANGEROUS: the system commits. Both sit at
the same WER; only the confidence axis separates them. So the deliverable is not a
number, it is a REGION — the conditions in the dead-zone quadrant (high confidence
AND high WER), named with exact factor values, ranked, written to
results/dead_zones.csv.

Four rules this module holds to:
  * dead-zone detection is `model_compare.dead_zone_flags` /
    `within_model_conf_percentile` — NOT a second copy, which would drift from L1;
  * never two models on one confidence axis (the scales are unrelated quantities),
    so every function takes one model's rows and `report_by_model` fans out;
  * failed rows are split out and counted, never averaged (a failure sentinel is
    not a low-confidence prediction — see analysis/__init__.py);
  * `where_confidence_tracks` reports the regions where confidence DOES work.
    "Confidence is informative below rt60 0.5" is itself actionable.

Deps: numpy, scipy (via model_compare). No audio, no API.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Sequence

import numpy as np

# Allow `python deadzone/analysis/confidence_gap.py` as well as `import deadzone.analysis.confidence_gap`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.analysis import (                       # noqa: E402  (after the path shim)
    as_float, check_unique_cells, coerce_row, failure_summary,
    load_master_table, split_by_model, split_failures,
)
from deadzone.design import DEFAULT_FACTOR_SPACE, FactorSpace  # noqa: E402
from deadzone.model_compare import (                  # noqa: E402
    confidence_wer_shape, dead_zone_flags, within_model_conf_percentile,
    _bins_for, _region_rows,
)
from scipy.stats import rankdata, spearmanr  # noqa: E402

# Quadrant thresholds. Defaults match model_compare so D1 and L1 flag the SAME
# cells; anything else and the two layers tell different stories.
WER_HI = 0.30          # "wrong": >=30% of reference words are wrong
CONF_PCT_HI = 0.60     # "confident": top 40% of THIS model's own confidence range

FACTOR_KEYS = tuple(DEFAULT_FACTOR_SPACE.names)   # rt60, snr_db, noise_type, ...


def _nanmean(a: np.ndarray) -> float:
    """np.nanmean without the empty-slice warning: no finite values -> NaN."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


# ===========================================================================
# 1. PER-CONDITION AGGREGATION (one point per condition, averaged over clips)
# ===========================================================================

def per_condition_table(rows: Sequence[dict], model: str | None = None) -> list[dict]:
    """
    Collapse clip-level rows to ONE row per condition — the scatter point of the
    headline plot.

      wer        macro mean of per-clip WERs (SPEC's "averaged over the 40 clips");
                 this is the key `dead_zone_flags` reads.
      wer_micro  word-weighted, sum(errors)/sum(n_ref). Reported alongside because
                 the two disagree when clip lengths differ, and quoting one without
                 the other is how benchmark tables get argued about.
      mean_conf  mean of per-clip mean confidences (clips weighted equally, matching
                 the macro WER). NaN-conf rows are excluded, never imputed to 0.5.

    Raises if more than one model is present and `model` was not given: silently
    pooling two models' confidences is the one mistake this layer cannot survive.
    """
    if model is not None:
        rows = [r for r in rows if r.get("model") == model]
    models = {r.get("model") for r in rows}
    if len(models) > 1:
        raise ValueError(
            f"per_condition_table got {len(models)} models {sorted(map(str, models))}; "
            f"pass model=... . Confidence is only comparable WITHIN a model "
            f"(see model_compare.within_model_conf_percentile).")

    ok, failed = split_failures(rows)
    # ONE ROW PER (clip, condition) — asserted, not assumed. A repeated cell is
    # averaged into its condition twice, so a single duplicated hard clip lifts
    # that condition's WER (and can lift it over `wer_hi` into the dead-zone
    # quadrant) while `n_clips` reads one higher and everything else looks
    # normal. This is D1's headline table; there is nothing downstream that
    # could catch it.
    check_unique_cells(
        ok, ("clip_id", "condition_name"), where="per_condition_table",
        cause=("Usual cause: rows for one model concatenated from two runs, or "
               "a master table rebuilt over a cache that logged a re-run cell "
               "under a second run_id."))
    groups: dict[str, list[dict]] = {}
    for r in ok:
        groups.setdefault(str(r["condition_name"]), []).append(r)
    failed_by_cond: dict[str, int] = {}
    for r in failed:
        name = str(r["condition_name"])
        failed_by_cond[name] = failed_by_cond.get(name, 0) + 1

    out: list[dict] = []
    for name, grp in groups.items():
        wer = np.array([as_float(r["wer"]) for r in grp], dtype=float)
        conf = np.array([as_float(r.get("mean_conf")) for r in grp], dtype=float)
        n_ref = np.array([as_float(r.get("n_ref", 0), 0.0) for r in grp], dtype=float)
        n_err = np.array([sum(as_float(r.get(f"n_{e}", 0), 0.0)
                              for e in ("sub", "del", "ins")) for r in grp], dtype=float)
        finite_conf = conf[np.isfinite(conf)]
        meas = [m for m in (as_float(r.get("rir_rt60_measured")) for r in grp)
                if np.isfinite(m)]
        rec = {
            "condition_name": name,
            "model": grp[0].get("model"),
            "n_clips": len(grp),
            "n_ref_total": int(np.nansum(n_ref)),
            "n_failed_excluded": failed_by_cond.get(name, 0),
            "wer": _nanmean(wer),
            "wer_micro": float(n_err.sum() / n_ref.sum()) if n_ref.sum() > 0 else float("nan"),
            "mean_conf": float(finite_conf.mean()) if finite_conf.size else float("nan"),
            "n_conf": int(finite_conf.size),
            # the DELIVERED reverb, when the runner recorded it (SPEC A.R4.2 note 2)
            "rt60_measured": float(np.mean(meas)) if meas else float("nan"),
        }
        rec.update({k: grp[0].get(k) for k in FACTOR_KEYS})
        out.append(rec)

    out.sort(key=lambda r: r["condition_name"])
    return out


# ===========================================================================
# 2. THE GAP METRIC
# ===========================================================================

def add_gap_metrics(cond_rows: Sequence[dict]) -> list[dict]:
    """
    Attach the confidence–accuracy gap to each condition (returns new dicts).

      gap      = mean_conf - clip(1 - wer, 0, 1). "Claimed this much accuracy,
                 delivered that much." POSITIVE = overconfident = silent failure.
                 Raw-scale, so within-model only (vendor confidence is not a
                 calibrated probability — that is the L2 layer's premise).
                 THE CLIP IS LOAD-BEARING: insertions push WER past 1.0, an
                 unclipped 1-wer then goes negative, and an insertion storm
                 outranks a genuine dead zone in a table RANKED BY GAP. Delivered
                 accuracy floors at zero — you cannot get less than none right.
      conf_pct = within-model confidence percentile (model_compare) — scale-free.
      acc_pct  = percentile of accuracy, ranked on the UNclipped 1-wer (a rank only
                 needs ordering; clipping would tie every WER>=1 cell together).
      gap_pct  = conf_pct - acc_pct — the only form meaningful across models.
    """
    rows = [dict(r) for r in cond_rows]
    if not rows:
        return rows
    conf_pct = within_model_conf_percentile(rows)
    acc = np.array([1.0 - as_float(r["wer"]) for r in rows], dtype=float)
    # rankdata's tie-averaging (not argsort) — matched to how conf_pct is built, or
    # tied cells get arbitrary percentiles and gap_pct picks up noise.
    finite = np.isfinite(acc)
    acc_pct = np.zeros(len(acc), dtype=float)
    if finite.any():
        acc_pct[finite] = ((rankdata(acc[finite], method="average") - 1)
                           / max(int(finite.sum()) - 1, 1))
    for i, r in enumerate(rows):
        c = as_float(r.get("mean_conf"))
        delivered = min(max(1.0 - as_float(r["wer"]), 0.0), 1.0)
        r["conf_pct"] = float(conf_pct[i])
        r["acc_pct"] = float(acc_pct[i])
        r["gap"] = float(c - delivered) if np.isfinite(c) else float("nan")
        r["gap_pct"] = float(conf_pct[i] - acc_pct[i])
    return rows


def gap_summary(cond_rows: Sequence[dict]) -> dict:
    """Distribution of the gap over the whole condition set (mean/sd/quantiles)."""
    g = np.array([as_float(r.get("gap")) for r in cond_rows], dtype=float)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return {"n": 0, "mean": float("nan")}
    qs = np.quantile(g, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "n": int(g.size),
        "mean": float(g.mean()),
        "sd": float(g.std(ddof=1)) if g.size > 1 else 0.0,
        "quantiles": {"min": float(qs[0]), "p25": float(qs[1]), "median": float(qs[2]),
                      "p75": float(qs[3]), "p90": float(qs[4]), "max": float(qs[5])},
        "frac_overconfident": float(np.mean(g > 0.0)),
    }


def _populated_regions(rows: Sequence[dict], space: FactorSpace, factor: str,
                       n_bins: int) -> list:
    """
    [(row_indices, span)] for the bins of `factor` that contain rows — EMPTY when
    fewer than two bins are populated.

    A factor the grid held CONSTANT otherwise yields one "region" containing 100% of
    the conditions, printed next to the real regions as though it were a finding. It
    is not a region, it is the whole table with a factor's name on it.

    Binning is model_compare's own `_bins_for`/`_region_rows` on purpose, so D1's
    regions line up exactly with L1's `find_divergence_regions`.
    """
    kind, spec = _bins_for(space, factor, n_bins)
    n_slots = (len(spec) - 1) if kind == "continuous" else len(spec)
    out = []
    for b in range(n_slots):
        idx, span = _region_rows(rows, factor, kind, spec, b)
        if len(idx):
            out.append((idx, span))
    return out if len(out) >= 2 else []


def gap_distribution(cond_rows: Sequence[dict],
                     space: FactorSpace = DEFAULT_FACTOR_SPACE,
                     n_bins: int = 4, wer_hi: float = WER_HI,
                     conf_pct_hi: float = CONF_PCT_HI) -> list[dict]:
    """
    How the gap is distributed across FACTOR SPACE: per factor, per bin/level, the
    mean gap, WER, confidence and dead-zone rate. Ranked by mean gap.

    TRAP: the dead-zone flags are computed ONCE over the whole table and then
    sliced. Recomputing the confidence percentile inside a bin would rank a handful
    of near-identical confidences against each other and manufacture a "top 40%" in
    every bin, including the clean ones.
    """
    rows = list(cond_rows)
    if not rows:
        return []
    flags = dead_zone_flags(rows, wer_hi, conf_pct_hi)
    gaps = np.array([as_float(r.get("gap")) for r in rows], dtype=float)
    wer = np.array([as_float(r["wer"]) for r in rows], dtype=float)
    conf = np.array([as_float(r.get("mean_conf")) for r in rows], dtype=float)

    out: list[dict] = []
    for factor in space.names:
        if factor not in rows[0]:
            continue
        for idx, span in _populated_regions(rows, space, factor, n_bins):
            out.append({
                "factor": factor,
                "span": span,
                "n_conditions": int(len(idx)),
                "n_clips": int(sum(rows[i]["n_clips"] for i in idx)),
                "mean_gap": _nanmean(gaps[idx]),
                "mean_wer": _nanmean(wer[idx]),
                "mean_conf": _nanmean(conf[idx]),
                "dead_zone_rate": float(np.mean(flags[idx])),
            })
    out.sort(key=lambda d: (-d["mean_gap"] if np.isfinite(d["mean_gap"]) else 0.0))
    return out


# ===========================================================================
# 3. THE DEAD-ZONE QUADRANT (the deliverable)
# ===========================================================================

def find_dead_zones(cond_rows: Sequence[dict], wer_hi: float = WER_HI,
                    conf_pct_hi: float = CONF_PCT_HI) -> list[dict]:
    """
    The ranked table of named dead-zone conditions: high WER *while* confident.

    Ranked by the gap, not by raw WER, because that is the "how dangerous is this
    cell" ordering. High WER with LOW confidence is deliberately absent: that is a
    loud failure, and a loud failure is a feature.
    """
    if not cond_rows:
        return []
    rows = ([dict(r) for r in cond_rows] if "gap" in cond_rows[0]
            else add_gap_metrics(cond_rows))
    flags = dead_zone_flags(rows, wer_hi, conf_pct_hi)
    dz = [rows[i] for i in np.where(flags)[0]]
    dz.sort(key=lambda r: (as_float(r.get("gap")) if np.isfinite(as_float(r.get("gap")))
                           else -np.inf), reverse=True)
    return dz


def raw_conf_threshold(cond_rows: Sequence[dict],
                       conf_pct_hi: float = CONF_PCT_HI) -> float:
    """
    The RAW confidence at the percentile threshold — where to draw the quadrant line
    on a plot whose x-axis is raw confidence. NaN if no finite confidences.
    """
    pct = within_model_conf_percentile(cond_rows)
    conf = np.array([as_float(r.get("mean_conf")) for r in cond_rows], dtype=float)
    sel = np.isfinite(conf) & (pct >= conf_pct_hi)
    return float(conf[sel].min()) if sel.any() else float("nan")


# ===========================================================================
# 4. DOES CONFIDENCE TRACK ACCURACY? (including where it DOES — SPEC A.R5.1)
# ===========================================================================

def _verdict(rho: float) -> str:
    """rho = spearman(confidence, WER). Negative means confidence warns."""
    if not np.isfinite(rho):
        return "undetermined"
    if rho <= -0.3:
        return "tracks"
    if rho >= 0.2:
        return "inverted"
    return "blind" if rho >= -0.05 else "weak"


def overall_correlation(rows: Sequence[dict]) -> dict:
    """
    The global confidence-vs-accuracy relationship: `confidence_wer_shape`'s rank
    correlation of within-model confidence percentile against WER. NEGATIVE = the
    model's confidence tracks its own errors (self-aware); ~0 or positive = the
    confidence carries no warning.

    A near-zero global rho is the EXPECTED shape when a model is self-aware in one
    part of the space and blind in another: the two cancel. That is why the
    per-region breakdown exists and why the global number is never the finding.
    """
    rows = [coerce_row(r) for r in rows]     # model_compare's _col cannot take None
    rho = confidence_wer_shape(rows)["spearman"]
    conf = np.array([as_float(r.get("mean_conf")) for r in rows], dtype=float)
    wer = np.array([as_float(r["wer"]) for r in rows], dtype=float)
    verdict = {
        "undetermined": "undetermined (degenerate confidence or WER)",
        "tracks": "confidence tracks error globally (self-aware)",
        "inverted": "INVERTED: confidence RISES with error globally",
    }.get(_verdict(rho),
          "confidence carries little global warning; see per-region breakdown")
    return {"spearman_confpct_vs_wer": rho,
            "n": int((np.isfinite(conf) & np.isfinite(wer)).sum()),
            "verdict": verdict}


def where_confidence_tracks(cond_rows: Sequence[dict],
                            space: FactorSpace = DEFAULT_FACTOR_SPACE,
                            n_bins: int = 4, min_n: int = 4) -> list[dict]:
    """
    Per factor region: does confidence track WER *there*? Sorted best-tracking
    first, so the report leads with the honest good news before the dead zones.

    Raw confidence is fine inside a region: spearman is rank-based and
    `within_model_conf_percentile` is a monotone transform, so the within-region
    rank correlation is identical either way. We rank; we do not re-percentile
    (which would be the trap).
    """
    rows = list(cond_rows)
    out: list[dict] = []
    for factor in space.names:
        if not rows or factor not in rows[0]:
            continue
        for idx, span in _populated_regions(rows, space, factor, n_bins):
            if len(idx) < min_n:
                continue
            sub = [rows[i] for i in idx]
            conf = np.array([as_float(r.get("mean_conf")) for r in sub])
            wer = np.array([as_float(r["wer"]) for r in sub])
            m = np.isfinite(conf) & np.isfinite(wer)
            if m.sum() < min_n or np.ptp(conf[m]) == 0 or np.ptp(wer[m]) == 0:
                continue
            rho = float(spearmanr(conf[m], wer[m])[0])
            out.append({
                "factor": factor, "span": span, "n_conditions": int(m.sum()),
                "spearman_conf_vs_wer": rho, "verdict": _verdict(rho),
                "mean_wer": _nanmean(wer[m]), "mean_conf": _nanmean(conf[m]),
            })
    out.sort(key=lambda d: d["spearman_conf_vs_wer"])
    return out


# ===========================================================================
# 5. THE REPORT
# ===========================================================================

def confidence_gap_report(rows: Sequence[dict], model: str | None = None,
                          space: FactorSpace = DEFAULT_FACTOR_SPACE,
                          wer_hi: float = WER_HI, conf_pct_hi: float = CONF_PCT_HI,
                          n_bins: int = 4) -> dict:
    """
    The whole D1 layer for ONE model, from clip-level master-table rows. Every
    number is computed on rows that passed `split_failures`, and the failure
    accounting travels with the report so a reader can never mistake a
    surviving-subset finding for a whole-grid one.
    """
    if model is not None:
        rows = [r for r in rows if r.get("model") == model]
    # Coerce at the door: a None `mean_conf` is a TypeError inside model_compare,
    # but only on the in-memory path — a CSV-loaded table hides it.
    rows = [coerce_row(r) for r in rows]
    fails = failure_summary(rows)
    cond = add_gap_metrics(per_condition_table(rows, model=model))
    dz = find_dead_zones(cond, wer_hi, conf_pct_hi)
    flags = dead_zone_flags(cond, wer_hi, conf_pct_hi) if cond else np.array([], bool)
    ok_rows, _ = split_failures(rows)
    return {
        "model": model or (cond[0]["model"] if cond else None),
        "n_conditions": len(cond),
        "n_clip_rows_used": len(ok_rows),
        "failures": fails,
        "thresholds": {"wer_hi": wer_hi, "conf_pct_hi": conf_pct_hi,
                       "conf_hi_raw": raw_conf_threshold(cond, conf_pct_hi)},
        "per_condition": cond,
        "dead_zones": dz,
        "dead_zone_rate": float(np.mean(flags)) if len(flags) else float("nan"),
        "gap_summary": gap_summary(cond),
        "gap_distribution": gap_distribution(cond, space, n_bins, wer_hi, conf_pct_hi),
        # correlation at BOTH resolutions: per-condition (the plot's points) and
        # per-clip-row (n is ~40x larger, so statistically stronger). They can
        # disagree, and when they do the disagreement is the finding.
        "correlation": overall_correlation(cond),
        "correlation_clip_level": overall_correlation(ok_rows) if ok_rows else {},
        "region_tracking": where_confidence_tracks(cond, space, n_bins),
    }


def report_by_model(rows: Sequence[dict], **kw) -> dict[str, dict]:
    """One D1 report per model. The reports are NEVER merged (different scales)."""
    return {m: confidence_gap_report(sub, model=m, **kw)
            for m, sub in split_by_model(rows).items()}


# ===========================================================================
# 6. OUTPUTS — csv, plot payload, prose
# ===========================================================================

DEAD_ZONE_CSV_COLUMNS = (
    "condition_name", "model", *FACTOR_KEYS, "rt60_measured",
    "mean_conf", "conf_pct", "wer", "wer_micro", "gap", "gap_pct",
    "n_clips", "n_ref_total", "n_failed_excluded",
)


def write_dead_zones_csv(dead_zones: Sequence[dict],
                         path: str = "results/dead_zones.csv") -> str:
    """
    Write the ranked dead-zone table (SPEC A.R5.1's named deliverable). Exact factor
    values travel so a reader can REPRODUCE the cell, not just read about it.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(DEAD_ZONE_CSV_COLUMNS),
                           extrasaction="ignore")
        w.writeheader()
        for r in dead_zones:
            w.writerow({k: r.get(k) for k in DEAD_ZONE_CSV_COLUMNS})
    return path


def plot_payload(report: dict) -> dict:
    """
    Dashboard contract for the headline scatter — JSON-serializable, no re-analysis
    needed downstream. STABLE SHAPE:

        {"model": str|None,
         "points": [{"condition_name", "x_mean_conf", "y_wer", "conf_pct", "gap",
                     "gap_pct", "n_clips", "n_ref_total", "dead_zone": bool,
                     "label": str, <each factor in FACTOR_KEYS>}, ...],
         "quadrant": {"conf_hi_raw", "wer_hi", "conf_pct_hi",
                      "dead_zone_box": {"x_min","x_max","y_min","y_max"}},
         "quadrant_counts": {"dead_zone_confident_and_wrong", "confident_and_right",
                             "loud_failure_unconfident_and_wrong",
                             "unconfident_but_right", "unplaceable_no_confidence"},
         "axes": {"x", "y"},
         "n_failed_rows_excluded": int}

    The x-axis quadrant line is `conf_hi_raw` — the raw confidence at the
    within-model percentile threshold — precisely so the picture and the dead-zone
    list can never disagree about which points are inside the box. The counts always
    sum to len(points).
    """
    cond = report.get("per_condition", [])
    dz_names = {r["condition_name"] for r in report.get("dead_zones", [])}
    th = report.get("thresholds", {})
    conf_hi = as_float(th.get("conf_hi_raw"))
    wer_hi = as_float(th.get("wer_hi"))

    points = [{
        "condition_name": r["condition_name"],
        "x_mean_conf": as_float(r.get("mean_conf")),
        "y_wer": as_float(r.get("wer")),
        "conf_pct": as_float(r.get("conf_pct")),
        "gap": as_float(r.get("gap")),
        "gap_pct": as_float(r.get("gap_pct")),
        "n_clips": r.get("n_clips"),
        "n_ref_total": r.get("n_ref_total"),
        "dead_zone": r["condition_name"] in dz_names,
        "label": format_dead_zone_sentence(r, report.get("model")),
        **{k: r.get(k) for k in FACTOR_KEYS},
    } for r in cond]

    # A point with no confidence (or no WER) belongs to no quadrant, and if the
    # model returned no confidences at all `conf_hi` is NaN and NOTHING is
    # classifiable. Those get their own bucket: a silent all-zero quadrant table
    # reads as "nothing is wrong" when it means "there is no confidence axis".
    def _placed(p: dict) -> bool:
        return (np.isfinite(p["x_mean_conf"]) and np.isfinite(p["y_wer"])
                and np.isfinite(conf_hi))

    def _q(hi_conf: bool, hi_wer: bool) -> int:
        return sum(1 for p in points if _placed(p)
                   and ((p["x_mean_conf"] >= conf_hi) == hi_conf)
                   and ((p["y_wer"] >= wer_hi) == hi_wer))

    xs = [p["x_mean_conf"] for p in points if np.isfinite(p["x_mean_conf"])]
    ys = [p["y_wer"] for p in points if np.isfinite(p["y_wer"])]
    return {
        "model": report.get("model"),
        "points": points,
        "quadrant": {
            "conf_hi_raw": conf_hi, "wer_hi": wer_hi,
            "conf_pct_hi": as_float(th.get("conf_pct_hi")),
            "dead_zone_box": {"x_min": conf_hi, "x_max": max(xs) if xs else 1.0,
                              "y_min": wer_hi, "y_max": max(ys) if ys else 1.0},
        },
        "quadrant_counts": {
            "dead_zone_confident_and_wrong": _q(True, True),
            "confident_and_right": _q(True, False),
            "loud_failure_unconfident_and_wrong": _q(False, True),
            "unconfident_but_right": _q(False, False),
            "unplaceable_no_confidence": sum(1 for p in points if not _placed(p)),
        },
        "axes": {"x": "mean word confidence (within-model scale)", "y": "WER"},
        "n_failed_rows_excluded": report.get("failures", {}).get("n_failed", 0),
    }


def format_dead_zone_sentence(cond: dict, model: str | None = None) -> str:
    """
    The SPEC A.R5.1 Definition-of-Done sentence. If you cannot print this with real
    numbers, this layer is not done.
    """
    return (f"At rt60 = {as_float(cond.get('rt60')):.2g} s, "
            f"SNR = {as_float(cond.get('snr_db')):.3g} dB, {cond.get('noise_type')}, "
            f"{cond.get('codec')}, mic_rolloff = {as_float(cond.get('mic_rolloff')):.2g}, "
            f"{model or cond.get('model')} returns mean word confidence "
            f"{as_float(cond.get('mean_conf')):.3f} while WER is "
            f"{as_float(cond.get('wer')):.3f} "
            f"(n = {cond.get('n_clips')} clips, {cond.get('n_ref_total')} ref words).")


def format_dead_zone_table(dead_zones: Sequence[dict], top_k: int = 20) -> str:
    """Fixed-width ranked dead-zone table for the terminal / write-up."""
    lines = ["Dead zones — confidently wrong, ranked by confidence-accuracy gap:",
             f"  {'#':>2} {'condition':<34} {'conf':>6} {'pct':>5} {'WER':>6} "
             f"{'gap':>6} {'n':>4}"]
    if not dead_zones:
        lines.append("  (none: no condition is both high-WER and high-confidence)")
        return "\n".join(lines)
    for i, r in enumerate(dead_zones[:top_k], 1):
        lines.append(
            f"  {i:>2} {str(r['condition_name'])[:34]:<34} "
            f"{as_float(r.get('mean_conf')):>6.3f} {as_float(r.get('conf_pct')):>5.2f} "
            f"{as_float(r.get('wer')):>6.3f} {as_float(r.get('gap')):>6.3f} "
            f"{r.get('n_clips', 0):>4}")
    return "\n".join(lines)


def format_report(report: dict) -> str:
    """Full terminal summary: failures first, good news second, dead zones last."""
    f, c = report["failures"], report["correlation"]
    lines = [
        f"D1 confidence-accuracy gap — model {report['model']!r}",
        f"  conditions: {report['n_conditions']}   clip-rows used: {report['n_clip_rows_used']}",
        f"  failed rows EXCLUDED: {f['n_failed']} / {f['n_rows']} "
        f"({f['failure_rate']:.2%})" + (f"; all-failed conditions: "
                                        f"{f['all_failed_conditions']}"
                                        if f["all_failed_conditions"] else ""),
        f"  global spearman(conf_pct, WER) = {c['spearman_confpct_vs_wer']:.3f} "
        f"-> {c['verdict']}",
        f"  gap mean {report['gap_summary'].get('mean', float('nan')):.3f}, "
        f"overconfident in {report['gap_summary'].get('frac_overconfident', float('nan')):.0%} "
        f"of conditions; dead-zone rate {report['dead_zone_rate']:.2%}",
        "",
        "Where confidence DOES track WER (reported first, on purpose):",
    ]
    tracks_well = [d for d in report["region_tracking"] if d["verdict"] == "tracks"]
    if tracks_well:
        for d in tracks_well[:8]:
            lines.append(f"  {d['factor']:<12} {str(d['span']):<22} "
                         f"rho={d['spearman_conf_vs_wer']:+.2f} (n={d['n_conditions']})")
    else:
        lines.append("  (no region reaches rho <= -0.3 — confidence warns nowhere)")
    lines += ["", format_dead_zone_table(report["dead_zones"])]
    if report["dead_zones"]:
        lines += ["", "Headline: " + format_dead_zone_sentence(report["dead_zones"][0],
                                                              report["model"])]
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D1: the confidence-accuracy gap map")
    ap.add_argument("--table", default="results/master.csv",
                    help="master results table (.csv or .parquet)")
    ap.add_argument("--model", default=None,
                    help="model to analyse (default: every model, one report each)")
    ap.add_argument("--out", default="results/dead_zones.csv")
    ap.add_argument("--wer-hi", type=float, default=WER_HI)
    ap.add_argument("--conf-pct-hi", type=float, default=CONF_PCT_HI)
    args = ap.parse_args(argv)

    rows = load_master_table(args.table)
    all_dz: list[dict] = []
    for m in ([args.model] if args.model else sorted(split_by_model(rows))):
        rep = confidence_gap_report(rows, model=m, wer_hi=args.wer_hi,
                                    conf_pct_hi=args.conf_pct_hi)
        print(format_report(rep), "\n")
        all_dz.extend(rep["dead_zones"])
    # one CSV, model column included — rows stay separable, the file is one artifact
    path = write_dead_zones_csv(all_dz, args.out)
    print(f"wrote {len(all_dz)} dead-zone rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
