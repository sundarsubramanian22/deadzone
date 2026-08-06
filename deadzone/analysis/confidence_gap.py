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

Five rules this module holds to:
  * dead-zone detection is `model_compare.dead_zone_flags` /
    `within_model_conf_percentile` — NOT a second copy, which would drift from L1;
  * never two models on one confidence axis (the scales are unrelated quantities),
    so every function takes one model's rows and `report_by_model` fans out;
  * failed rows are split out and counted, never averaged (a failure sentinel is
    not a low-confidence prediction — see analysis/__init__.py);
  * MATCHED ESTIMANDS: a confidence is only ever compared against an accuracy
    measured on THE SAME CLIPS (see the section below);
  * `where_confidence_tracks` reports the regions where confidence DOES work.
    "Confidence is informative below rt60 0.5" is itself actionable.

-----------------------------------------------------------------------------
THE SILENT-CLIP MISMATCH — the defect this module was rebuilt around
-----------------------------------------------------------------------------
A clip whose transcript comes back EMPTY scores WER 1.0 (100% deletions) and
carries NO per-word confidence, so it contributes nothing to `mean_conf`. The
original table averaged WER over all 40 clips and confidence over only the clips
that spoke, then subtracted one from the other. That is a comparison between two
different populations: the confidence describes a strictly easier subset, and the
difference reads as overconfidence that is really just the missing clips.

It was not a rounding error. On the real nova-3 grid 123 of 176 conditions had at
least one silent clip, and the #1 published dead zone
(`rt60-0.7_snr-20_babble_opus-lowrate_roll-1`, 10 of 40 clips silent) reported
mean_conf 0.843 against all-clips WER 0.387 for a gap of +0.230 — while on the 30
clips it actually spoke on its WER was 0.182, i.e. it was 81.8% accurate at 0.843
confidence, a gap of +0.025. Well calibrated, not confidently wrong. Correcting
the pairing REORDERED the table: the largest inflation anywhere was +0.524.

So every condition now carries BOTH pairings, explicitly named, and neither can
be read without the other:

  wer / wer_all_clips  macro WER over every clip — what the condition does to the
                       corpus. The honest severity number. NOT comparable to
                       mean_conf.
  wer_spoke            macro WER over the clips that returned a confidence — the
                       ONLY accuracy `mean_conf` may be subtracted from.
  n_silent/silent_frac clips that emitted no scorable words. First-class, because
                       it is the quantity that separates the two WERs.
  gap (= gap_spoke)    the defensible gap. Drives the ranking.
  gap_all_clips        the mismatched pairing, retained and labelled so the
                       published grid-v1 numbers stay reproducible.
  gap_inflation        gap_all_clips - gap_spoke: how much the mismatch added.

And a condition that is silent on EVERY clip has no confidence at all, so it can
have no gap. Those are the worst conditions in the study and dropping them would
be the same defect a second time, so they get their own category — MUTE ZONE —
and are reported, never filtered out. A dead zone is confidently WRONG; a mute
zone is entirely ABSENT. Different mechanisms, different fixes.

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
    as_float, check_unique_cells, coerce_row, failure_summary, is_silent_row,
    load_master_table, silence_summary, split_by_model, split_failures,
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

# The three condition categories. Every condition lands in exactly one of these
# plus "ok"; nothing is dropped.
CATEGORY_DEAD_ZONE = "dead_zone"          # confidently wrong ON THE CLIPS IT SPOKE
CATEGORY_SILENCE_DRIVEN = "silence_driven"  # only a dead zone under the OLD pairing
CATEGORY_MUTE = "mute_zone"               # no words on ANY clip -> no gap can exist
CATEGORY_OK = "ok"

CATEGORY_MEANING = {
    CATEGORY_DEAD_ZONE: ("DEAD ZONE — confidently wrong: WER on the clips the "
                         "model spoke on is >= wer_hi while its within-model "
                         "confidence is in the top band."),
    CATEGORY_SILENCE_DRIVEN: ("SILENCE-DRIVEN — flagged only by the mismatched "
                              "all-clips pairing. On the clips it spoke on the "
                              "model was NOT wrong enough to qualify; the "
                              "apparent gap comes from clips that vanished. "
                              "Dangerous, but it is a silence failure, not a "
                              "confidently-wrong failure, and the fixes differ."),
    CATEGORY_MUTE: ("MUTE ZONE — the model returned an empty transcript on EVERY "
                    "clip (WER 1.00, 100% deletions). There is no confidence, so "
                    "there is no gap to report. These are the worst conditions in "
                    "the study and are listed, never dropped."),
}


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

      wer            macro mean of per-clip WERs over EVERY clip (SPEC's "averaged
                     over the 40 clips") — what the condition does to the corpus.
      wer_all_clips  the same number under an unambiguous name. `wer` is kept
                     because it is the frozen key L1/D2/the dashboard join on.
      wer_micro      word-weighted, sum(errors)/sum(n_ref). Reported alongside
                     because the two disagree when clip lengths differ, and quoting
                     one without the other is how benchmark tables get argued about.
      mean_conf      mean of per-clip mean confidences (clips weighted equally,
                     matching the macro WER). NaN-conf rows are excluded, never
                     imputed to 0.5.
      n_spoke        HOW MANY clips that mean is over — the confidence's denominator.
      wer_spoke      macro WER over exactly those clips, and nothing else. This is
                     the only accuracy `mean_conf` may be compared against.
      n_silent       clips that emitted no scorable words (`analysis.is_silent_row`)
      silent_frac    n_silent / n_clips.

    n_spoke + n_silent == n_clips on the nova-3 arm. It can fall short by a hair on
    an adapter that returns an utterance-level confidence with no words (Whisper
    does, on 3 of 1757 real rows, two of which transcribed pure punctuation). The
    report counts those separately rather than hiding the discrepancy; the paired
    subset is always defined by "did this clip contribute a confidence", because
    that is what the denominator of `mean_conf` literally is.

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
        # THE PAIRED SUBSET. `spoke` is exactly the set of clips whose confidence
        # is inside `mean_conf`, so every quantity computed under this mask is on
        # the same estimand as the confidence. Anything computed over `grp` is on
        # the whole corpus and must never be subtracted from a confidence.
        spoke = np.isfinite(conf)
        silent = np.array([is_silent_row(r) for r in grp], dtype=bool)
        finite_conf = conf[spoke]
        meas = [m for m in (as_float(r.get("rir_rt60_measured")) for r in grp)
                if np.isfinite(m)]
        wer_all = _nanmean(wer)
        rec = {
            "condition_name": name,
            "model": grp[0].get("model"),
            "n_clips": len(grp),
            "n_ref_total": int(np.nansum(n_ref)),
            "n_failed_excluded": failed_by_cond.get(name, 0),
            # --- all-clips estimand (corpus severity) ------------------------
            "wer": wer_all,
            "wer_all_clips": wer_all,
            "wer_micro": float(n_err.sum() / n_ref.sum()) if n_ref.sum() > 0 else float("nan"),
            # --- silence accounting ------------------------------------------
            "n_silent": int(silent.sum()),
            "silent_frac": float(silent.mean()) if len(grp) else float("nan"),
            "n_conf_without_words": int((spoke & silent).sum()),
            # --- paired (spoke-only) estimand — the confidence's own population -
            "mean_conf": float(finite_conf.mean()) if finite_conf.size else float("nan"),
            "n_conf": int(finite_conf.size),
            "n_spoke": int(spoke.sum()),
            "wer_spoke": _nanmean(wer[spoke]) if spoke.any() else float("nan"),
            "wer_micro_spoke": (float(n_err[spoke].sum() / n_ref[spoke].sum())
                                if spoke.any() and n_ref[spoke].sum() > 0 else float("nan")),
            "n_ref_spoke": int(np.nansum(n_ref[spoke])) if spoke.any() else 0,
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

def _accuracy_percentile(rows: Sequence[dict], wer_key: str) -> np.ndarray:
    """
    Percentile of delivered accuracy, ranked on the UNclipped 1-wer (a rank only
    needs ordering; clipping would tie every WER>=1 cell together). Rows with a
    non-finite WER under this key sink to 0, exactly as NaN confidences do in
    `within_model_conf_percentile`.
    """
    acc = np.array([1.0 - as_float(r.get(wer_key)) for r in rows], dtype=float)
    finite = np.isfinite(acc)
    out = np.zeros(len(acc), dtype=float)
    if finite.any():
        # rankdata's tie-averaging (not argsort) — matched to how conf_pct is
        # built, or tied cells get arbitrary percentiles and gap_pct picks up noise.
        out[finite] = ((rankdata(acc[finite], method="average") - 1)
                       / max(int(finite.sum()) - 1, 1))
    return out


def add_gap_metrics(cond_rows: Sequence[dict]) -> list[dict]:
    """
    Attach BOTH confidence–accuracy gaps to each condition (returns new dicts).

      gap_spoke     = mean_conf - clip(1 - wer_spoke, 0, 1). THE DEFENSIBLE ONE:
                      "on the clips where you spoke, you claimed this much accuracy
                      and delivered that much." Both terms are averages over the
                      same clips, so the subtraction means something. POSITIVE =
                      overconfident = silent failure.
      gap_all_clips = mean_conf - clip(1 - wer, 0, 1). THE MISMATCHED PAIRING that
                      grid-v1 published: a subset confidence minus a whole-corpus
                      accuracy. Retained (a) so the published numbers stay
                      reproducible and (b) so the inflation is visible rather than
                      quietly corrected away. Never rank on it.
      gap           = gap_spoke. The canonical name points at the correct quantity;
                      the mismatched one only exists under an explicit name.
      gap_inflation = gap_all_clips - gap_spoke, i.e. how much of the published gap
                      was silent clips. It is ~equal to mean_conf's overstatement
                      and it grows with silent_frac.

    THE CLIP IN `clip(1 - wer, 0, 1)` IS LOAD-BEARING: insertions push WER past 1.0
    (Whisper hallucinates fluent sentences under heavy noise), an unclipped 1-wer
    then goes negative, and an insertion storm would outrank a genuine dead zone in
    a table RANKED BY GAP. Delivered accuracy floors at zero — you cannot get less
    than none right.

      conf_pct = within-model confidence percentile (model_compare) — scale-free.
      acc_pct  = percentile of 1 - wer_spoke;  acc_pct_all_clips = of 1 - wer.
      gap_pct  = conf_pct - acc_pct — the only form meaningful across models
                 (with gap_pct_all_clips as its mismatched twin).

    A MUTE condition (no confidence on any clip) gets NaN for every gap: it cannot
    be overconfident about words it never emitted. `classify_conditions` picks
    those up as their own category so the NaN is a label, not a disappearance.
    """
    rows = [dict(r) for r in cond_rows]
    if not rows:
        return rows
    conf_pct = within_model_conf_percentile(rows)
    acc_pct = _accuracy_percentile(rows, "wer_spoke")
    acc_pct_all = _accuracy_percentile(rows, "wer")
    for i, r in enumerate(rows):
        c = as_float(r.get("mean_conf"))
        ok = np.isfinite(c)
        deliv_spoke = min(max(1.0 - as_float(r.get("wer_spoke")), 0.0), 1.0)
        deliv_all = min(max(1.0 - as_float(r.get("wer")), 0.0), 1.0)
        gap_spoke = float(c - deliv_spoke) if ok and np.isfinite(
            as_float(r.get("wer_spoke"))) else float("nan")
        gap_all = float(c - deliv_all) if ok else float("nan")
        r["conf_pct"] = float(conf_pct[i])
        r["acc_pct"] = float(acc_pct[i])
        r["acc_pct_all_clips"] = float(acc_pct_all[i])
        r["gap_spoke"] = gap_spoke
        r["gap_all_clips"] = gap_all
        r["gap"] = gap_spoke                      # canonical == defensible
        r["gap_inflation"] = (float(gap_all - gap_spoke)
                              if np.isfinite(gap_all) and np.isfinite(gap_spoke)
                              else float("nan"))
        r["gap_pct"] = float(conf_pct[i] - acc_pct[i]) if ok else float("nan")
        r["gap_pct_all_clips"] = (float(conf_pct[i] - acc_pct_all[i])
                                  if ok else float("nan"))
    return rows


def _gap_stats(g: np.ndarray) -> dict:
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


def gap_summary(cond_rows: Sequence[dict]) -> dict:
    """
    Distribution of the gap over the whole condition set (mean/sd/quantiles), for
    the SAME-SUBSET gap — plus the mismatched all-clips pairing and the inflation
    between them, so "mean gap 0.256" can never again be quoted without the
    estimand it belongs to.
    """
    out = _gap_stats(np.array([as_float(r.get("gap_spoke", r.get("gap")))
                               for r in cond_rows], dtype=float))
    out["pairing"] = "same-subset (mean_conf vs 1 - wer_spoke)"
    out["all_clips_pairing"] = _gap_stats(
        np.array([as_float(r.get("gap_all_clips")) for r in cond_rows], dtype=float))
    infl = np.array([as_float(r.get("gap_inflation")) for r in cond_rows], dtype=float)
    infl = infl[np.isfinite(infl)]
    out["inflation"] = {
        "n": int(infl.size),
        "mean": float(infl.mean()) if infl.size else float("nan"),
        "max": float(infl.max()) if infl.size else float("nan"),
    }
    sf = np.array([as_float(r.get("silent_frac")) for r in cond_rows], dtype=float)
    sf = sf[np.isfinite(sf)]
    out["silence"] = {
        "mean_silent_frac": float(sf.mean()) if sf.size else float("nan"),
        "max_silent_frac": float(sf.max()) if sf.size else float("nan"),
        "n_conditions_with_silence": int((sf > 0).sum()) if sf.size else 0,
    }
    return out


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


def condition_flags(cond_rows: Sequence[dict], wer_hi: float = WER_HI,
                    conf_pct_hi: float = CONF_PCT_HI,
                    wer_key: str = "wer_spoke") -> np.ndarray:
    """
    `model_compare.dead_zone_flags` over a per-condition table, with the MUTE rows
    held out and returned False. Shared by every caller here so D1 owns no second
    copy of "confident".

    Holding the mute rows out is safe *and* required:
      * required, because `wer_spoke` is NaN for them and `dead_zone_flags` refuses
        a non-finite WER by design (a NaN would silently read as "not a dead zone"
        while still diluting every mean(flags));
      * safe, because `within_model_conf_percentile` ranks only the FINITE
        confidences and divides by (n_finite - 1). A mute row's confidence is NaN,
        so removing it changes no surviving row's percentile at all — the flags are
        bit-identical to computing them over the full table. (Asserted in
        tests/test_analysis.py.)
    """
    out = np.zeros(len(cond_rows), dtype=bool)
    idx = [i for i, r in enumerate(cond_rows)
           if np.isfinite(as_float(r.get("mean_conf")))
           and np.isfinite(as_float(r.get(wer_key)))]
    if not idx:
        return out
    out[idx] = dead_zone_flags([cond_rows[i] for i in idx], wer_hi, conf_pct_hi,
                               wer_key=wer_key)
    return out


def gap_distribution(cond_rows: Sequence[dict],
                     space: FactorSpace = DEFAULT_FACTOR_SPACE,
                     n_bins: int = 4, wer_hi: float = WER_HI,
                     conf_pct_hi: float = CONF_PCT_HI) -> list[dict]:
    """
    How the gap is distributed across FACTOR SPACE: per factor, per bin/level, the
    mean gap (both pairings), both WERs, the silent fraction, confidence and
    dead-zone rate. Ranked by mean same-subset gap.

    `mean_silent_frac` travels with every row on purpose: it is the single number
    that says how far apart the two pairings can be in that region, so a reader
    can see at a glance whether the region's gap is about confidence or about
    clips going missing.

    TRAP: the dead-zone flags are computed ONCE over the whole table and then
    sliced. Recomputing the confidence percentile inside a bin would rank a handful
    of near-identical confidences against each other and manufacture a "top 40%" in
    every bin, including the clean ones.
    """
    rows = list(cond_rows)
    if not rows:
        return []
    flags = condition_flags(rows, wer_hi, conf_pct_hi, "wer_spoke")
    flags_all = condition_flags(rows, wer_hi, conf_pct_hi, "wer")
    gaps = np.array([as_float(r.get("gap_spoke", r.get("gap"))) for r in rows], dtype=float)
    gaps_all = np.array([as_float(r.get("gap_all_clips")) for r in rows], dtype=float)
    wer = np.array([as_float(r["wer"]) for r in rows], dtype=float)
    wer_spoke = np.array([as_float(r.get("wer_spoke")) for r in rows], dtype=float)
    silent = np.array([as_float(r.get("silent_frac")) for r in rows], dtype=float)
    conf = np.array([as_float(r.get("mean_conf")) for r in rows], dtype=float)
    mute = ~np.isfinite(conf)

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
                "mean_gap_all_clips": _nanmean(gaps_all[idx]),
                "mean_wer": _nanmean(wer[idx]),
                "mean_wer_spoke": _nanmean(wer_spoke[idx]),
                "mean_silent_frac": _nanmean(silent[idx]),
                "mean_conf": _nanmean(conf[idx]),
                "dead_zone_rate": float(np.mean(flags[idx])),
                "dead_zone_rate_all_clips_pairing": float(np.mean(flags_all[idx])),
                "n_mute": int(mute[idx].sum()),
            })
    out.sort(key=lambda d: (-d["mean_gap"] if np.isfinite(d["mean_gap"]) else 0.0))
    return out


# ===========================================================================
# 3. THE DEAD-ZONE QUADRANT (the deliverable)
# ===========================================================================

def _by_gap(rows: list[dict], key: str = "gap") -> list[dict]:
    return sorted(rows, key=lambda r: (as_float(r.get(key))
                                       if np.isfinite(as_float(r.get(key)))
                                       else -np.inf), reverse=True)


def classify_conditions(cond_rows: Sequence[dict], wer_hi: float = WER_HI,
                        conf_pct_hi: float = CONF_PCT_HI) -> dict:
    """
    Sort every condition into exactly one category, and lose none of them.

      dead_zone       confidently wrong ON THE CLIPS IT SPOKE: wer_spoke >= wer_hi
                      and conf_pct >= conf_pct_hi. The defensible claim, and the
                      only set that earns the phrase "confidently wrong".
      silence_driven  flagged ONLY by the all-clips pairing. The model was not
                      wrong enough on the clips it spoke on; the apparent gap came
                      from clips that returned nothing. Still dangerous — but a
                      SILENCE failure, whose fix (endpointing, VAD, re-prompting,
                      dereverberation) is not the confidently-wrong fix (keyword
                      boosting, entity-aware decoding, confidence recalibration).
      mute_zone       the model emitted no words on ANY clip. No confidence exists,
                      so no gap exists. These are the WORST conditions in the study
                      and they are listed first-class instead of vanishing into a
                      NaN. A mute zone is not a quiet dead zone; it is a loud
                      failure that a confidence-based monitor cannot even see.
      ok              everything else.

    Returns the full annotated condition list plus each category, so a caller can
    never obtain the dead zones without also being handed the other two.
    """
    rows = ([dict(r) for r in cond_rows] if cond_rows and "gap_spoke" in cond_rows[0]
            else add_gap_metrics(cond_rows))
    if not rows:
        return {"conditions": [], "dead_zones": [], "silence_driven": [],
                "mute_zones": [], "counts": {"n_conditions": 0}}
    spoke_flags = condition_flags(rows, wer_hi, conf_pct_hi, "wer_spoke")
    all_flags = condition_flags(rows, wer_hi, conf_pct_hi, "wer")
    for i, r in enumerate(rows):
        mute = not np.isfinite(as_float(r.get("mean_conf")))
        r["mute"] = bool(mute)
        r["dead_zone"] = bool(spoke_flags[i])
        r["dead_zone_all_clips_pairing"] = bool(all_flags[i])
        r["category"] = (CATEGORY_MUTE if mute else
                         CATEGORY_DEAD_ZONE if spoke_flags[i] else
                         CATEGORY_SILENCE_DRIVEN if all_flags[i] else
                         CATEGORY_OK)
    dz = _by_gap([r for r in rows if r["category"] == CATEGORY_DEAD_ZONE])
    sd = _by_gap([r for r in rows if r["category"] == CATEGORY_SILENCE_DRIVEN],
                 "gap_all_clips")
    mz = sorted((r for r in rows if r["category"] == CATEGORY_MUTE),
                key=lambda r: str(r["condition_name"]))
    n = len(rows)
    return {
        "conditions": rows,
        "dead_zones": dz,
        "silence_driven": sd,
        "mute_zones": mz,
        "meaning": CATEGORY_MEANING,
        "counts": {
            "n_conditions": n,
            "n_dead_zones": len(dz),
            "n_silence_driven": len(sd),
            "n_mute_zones": len(mz),
            "dead_zone_rate": len(dz) / n,
            "silence_driven_rate": len(sd) / n,
            "mute_rate": len(mz) / n,
            # what the OLD, mismatched pairing would have reported as dead zones
            "n_dead_zones_all_clips_pairing": int(all_flags.sum()),
            "dead_zone_rate_all_clips_pairing": float(all_flags.mean()),
        },
    }


def find_dead_zones(cond_rows: Sequence[dict], wer_hi: float = WER_HI,
                    conf_pct_hi: float = CONF_PCT_HI) -> list[dict]:
    """
    The ranked table of named dead-zone conditions: wrong ON THE CLIPS IT SPOKE
    *while* confident.

    Ranked by the SAME-SUBSET gap, not by raw WER and not by the all-clips gap,
    because that is the "how dangerous is this cell" ordering and it is the only
    one whose two terms describe the same clips. High WER with LOW confidence is
    deliberately absent: that is a loud failure, and a loud failure is a feature.

    This is a thin view over `classify_conditions` — use that directly if you need
    the silence-driven and mute categories too, which anyone quoting this table
    does.
    """
    return classify_conditions(cond_rows, wer_hi, conf_pct_hi)["dead_zones"]


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


def overall_correlation(rows: Sequence[dict], wer_key: str = "wer") -> dict:
    """
    The global confidence-vs-accuracy relationship: `confidence_wer_shape`'s rank
    correlation of within-model confidence percentile against WER. NEGATIVE = the
    model's confidence tracks its own errors (self-aware); ~0 or positive = the
    confidence carries no warning.

    Rows without a confidence, or without a WER under `wer_key`, are DROPPED before
    correlating rather than sunk to percentile 0. Sinking them in would score a
    condition the model never spoke in as "maximally unconfident and maximally
    wrong" — a free, fake data point at the ideal corner of a negative correlation,
    inflating exactly the number ("the model does know when it is failing") that
    this layer is supposed to test.

    A near-zero global rho is the EXPECTED shape when a model is self-aware in one
    part of the space and blind in another: the two cancel. That is why the
    per-region breakdown exists and why the global number is never the finding.
    """
    rows = [coerce_row(r) for r in rows]     # model_compare's _col cannot take None
    paired = [r for r in rows
              if np.isfinite(as_float(r.get("mean_conf")))
              and np.isfinite(as_float(r.get(wer_key)))]
    rho = confidence_wer_shape(paired, wer_key=wer_key)["spearman"] if paired else float("nan")
    verdict = {
        "undetermined": "undetermined (degenerate confidence or WER)",
        "tracks": "confidence tracks error globally (self-aware)",
        "inverted": "INVERTED: confidence RISES with error globally",
    }.get(_verdict(rho),
          "confidence carries little global warning; see per-region breakdown")
    return {"spearman_confpct_vs_wer": rho,
            "n": len(paired),
            "n_excluded_no_confidence": len(rows) - len(paired),
            "wer_key": wer_key,
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

    The headline `spearman_conf_vs_wer` is against `wer_spoke` — same clips as the
    confidence. The all-clips version travels beside it, because a region where the
    two disagree is a region whose apparent self-awareness is really a silence
    pattern.
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
            wer_all = np.array([as_float(r["wer"]) for r in sub])
            wer = np.array([as_float(r.get("wer_spoke", r.get("wer"))) for r in sub])
            m = np.isfinite(conf) & np.isfinite(wer)
            if m.sum() < min_n or np.ptp(conf[m]) == 0 or np.ptp(wer[m]) == 0:
                continue
            rho = float(spearmanr(conf[m], wer[m])[0])
            ma = np.isfinite(conf) & np.isfinite(wer_all)
            rho_all = (float(spearmanr(conf[ma], wer_all[ma])[0])
                       if ma.sum() >= min_n and np.ptp(conf[ma]) and np.ptp(wer_all[ma])
                       else float("nan"))
            out.append({
                "factor": factor, "span": span, "n_conditions": int(m.sum()),
                "spearman_conf_vs_wer": rho, "verdict": _verdict(rho),
                "spearman_conf_vs_wer_all_clips": rho_all,
                "mean_wer": _nanmean(wer_all[ma]), "mean_wer_spoke": _nanmean(wer[m]),
                "mean_conf": _nanmean(conf[m]),
                "mean_silent_frac": _nanmean(
                    np.array([as_float(r.get("silent_frac")) for r in sub])),
                "n_mute_excluded": int((~np.isfinite(conf)).sum()),
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
    ok_rows, _ = split_failures(rows)
    cls = classify_conditions(per_condition_table(rows, model=model),
                              wer_hi, conf_pct_hi)
    cond = cls["conditions"]
    corr = overall_correlation(cond, "wer_spoke")
    corr["spearman_confpct_vs_wer_all_clips"] = overall_correlation(
        cond, "wer")["spearman_confpct_vs_wer"]
    return {
        "model": model or (cond[0]["model"] if cond else None),
        "n_conditions": len(cond),
        "n_clip_rows_used": len(ok_rows),
        "failures": fails,
        # Silence is reported at the same level as failure, because it is the same
        # KIND of thing: a row that carries no prediction to be confident about.
        "silence": silence_summary(ok_rows),
        "thresholds": {"wer_hi": wer_hi, "conf_pct_hi": conf_pct_hi,
                       "conf_hi_raw": raw_conf_threshold(cond, conf_pct_hi)},
        "per_condition": cond,
        "categories": cls["counts"],
        "category_meaning": CATEGORY_MEANING,
        "dead_zones": cls["dead_zones"],
        "silence_driven": cls["silence_driven"],
        "mute_zones": cls["mute_zones"],
        "dead_zone_rate": cls["counts"].get("dead_zone_rate", float("nan"))
        if cond else float("nan"),
        "gap_summary": gap_summary(cond),
        "gap_distribution": gap_distribution(cond, space, n_bins, wer_hi, conf_pct_hi),
        # correlation at BOTH resolutions: per-condition (the plot's points) and
        # per-clip-row (n is ~40x larger, so statistically stronger). They can
        # disagree, and when they do the disagreement is the finding. Both are
        # computed on the clips that carried a confidence — see overall_correlation.
        "correlation": corr,
        "correlation_clip_level": overall_correlation(ok_rows, "wer") if ok_rows else {},
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
    # WHAT KIND OF CONDITION THIS IS — first column, deliberately. The file holds
    # dead zones, silence-driven cells and mute zones; reading a row without its
    # category is how the previous version of this table got misquoted.
    "category",
    "condition_name", "model", *FACTOR_KEYS, "rt60_measured",
    # the confidence and the population it was averaged over
    "mean_conf", "conf_pct", "n_clips", "n_spoke", "n_silent", "silent_frac",
    # BOTH accuracies, both named. There is no bare "wer" column any more: the
    # whole defect was a reader (and a subtraction) not knowing which one it had.
    "wer_all_clips", "wer_spoke", "wer_micro", "wer_micro_spoke",
    # BOTH gaps, plus how much the mismatched pairing added
    "gap_spoke", "gap_all_clips", "gap_inflation", "gap_pct", "gap_pct_all_clips",
    # `wer` and `gap` are kept as trailing aliases of the CANONICAL quantities
    # (all-clips WER; same-subset gap) so older readers of this file keep working.
    "wer", "gap",
    "n_ref_total", "n_ref_spoke", "n_failed_excluded",
    "dead_zone_all_clips_pairing",
)


def write_dead_zones_csv(rows: Sequence[dict],
                         path: str = "results/dead_zones.csv") -> str:
    """
    Write the ranked silent-failure table (SPEC A.R5.1's named deliverable). Exact
    factor values travel so a reader can REPRODUCE the cell, not just read about it.

    Pass the CONCATENATION of `classify_conditions`' three categories, not just the
    dead zones: a file that silently omits the mute zones omits the worst conditions
    in the study. `category` is column one so nothing in it can be read as a dead
    zone by accident.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(DEAD_ZONE_CSV_COLUMNS),
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rec = {k: r.get(k) for k in DEAD_ZONE_CSV_COLUMNS}
            rec.setdefault("category", r.get("category", CATEGORY_DEAD_ZONE))
            if rec.get("category") is None:
                rec["category"] = CATEGORY_DEAD_ZONE
            w.writerow(rec)
    return path


def flagged_conditions(report: dict) -> list[dict]:
    """Dead zones, then silence-driven cells, then mute zones — the CSV's row order."""
    return (list(report.get("dead_zones", []))
            + list(report.get("silence_driven", []))
            + list(report.get("mute_zones", [])))


def plot_payload(report: dict) -> dict:
    """
    Dashboard contract for the headline scatter — JSON-serializable, no re-analysis
    needed downstream. STABLE SHAPE:

        {"model": str|None,
         "points": [{"condition_name", "x_mean_conf", "y_wer", "y_wer_spoke",
                     "y_wer_all_clips", "conf_pct", "gap", "gap_all_clips",
                     "gap_inflation", "n_clips", "n_spoke", "n_silent",
                     "silent_frac", "n_ref_total", "category", "mute",
                     "dead_zone": bool, "label": str, <FACTOR_KEYS>}, ...],
         "quadrant": {"conf_hi_raw", "wer_hi", "conf_pct_hi",
                      "dead_zone_box": {"x_min","x_max","y_min","y_max"}},
         "quadrant_counts": {"dead_zone_confident_and_wrong", "confident_and_right",
                             "loud_failure_unconfident_and_wrong",
                             "unconfident_but_right", "unplaceable_no_confidence",
                             "silence_driven", "mute_no_words_on_any_clip"},
         "categories": {...}, "mute_conditions": [...],
         "axes": {"x", "y", "y_note"},
         "n_failed_rows_excluded": int}

    `y_wer` IS THE SAME-SUBSET WER, on purpose. The scatter's y-axis is one half of
    a subtraction whose other half is the x-axis, so plotting the all-clips WER
    against a spoke-only confidence would draw the exact mismatch this module was
    rebuilt to remove — and the shaded quadrant would then contain points the
    dead-zone list does not. `y_wer_all_clips` travels alongside for the panels
    that want corpus severity (the factor-space heatmap does).

    The x-axis quadrant line is `conf_hi_raw` — the raw confidence at the
    within-model percentile threshold — precisely so the picture and the dead-zone
    list can never disagree about which points are inside the box. The counts always
    sum to len(points).
    """
    cond = report.get("per_condition", [])
    dz_names = {r["condition_name"] for r in report.get("dead_zones", [])}
    sd_names = {r["condition_name"] for r in report.get("silence_driven", [])}
    mute_names = [r["condition_name"] for r in report.get("mute_zones", [])]
    th = report.get("thresholds", {})
    conf_hi = as_float(th.get("conf_hi_raw"))
    wer_hi = as_float(th.get("wer_hi"))

    points = [{
        "condition_name": r["condition_name"],
        "x_mean_conf": as_float(r.get("mean_conf")),
        # the paired accuracy — the only one the confidence may be compared to
        "y_wer": as_float(r.get("wer_spoke")),
        "y_wer_spoke": as_float(r.get("wer_spoke")),
        "y_wer_all_clips": as_float(r.get("wer")),
        "conf_pct": as_float(r.get("conf_pct")),
        "gap": as_float(r.get("gap_spoke", r.get("gap"))),
        "gap_all_clips": as_float(r.get("gap_all_clips")),
        "gap_inflation": as_float(r.get("gap_inflation")),
        "gap_pct": as_float(r.get("gap_pct")),
        "n_clips": r.get("n_clips"),
        "n_spoke": r.get("n_spoke"),
        "n_silent": r.get("n_silent"),
        "silent_frac": as_float(r.get("silent_frac")),
        "n_ref_total": r.get("n_ref_total"),
        "category": r.get("category", CATEGORY_OK),
        "mute": bool(r.get("mute", False)),
        "dead_zone": r["condition_name"] in dz_names,
        "silence_driven": r["condition_name"] in sd_names,
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
        # not part of the quadrant partition (they'd double-count): reported
        # beside it so the panel can never show a dead-zone count without them.
        "silence_counts": {
            "silence_driven": len(sd_names),
            "mute_no_words_on_any_clip": len(mute_names),
            "conditions_with_any_silent_clip": sum(
                1 for p in points if as_float(p["silent_frac"]) > 0),
        },
        "categories": report.get("categories", {}),
        "category_meaning": report.get("category_meaning", CATEGORY_MEANING),
        "mute_conditions": mute_names,
        "silence": report.get("silence", {}),
        "axes": {"x": "mean word confidence (within-model scale)",
                 "y": "WER on the clips the model spoke on",
                 "y_note": ("paired with the confidence on the x-axis — same clips. "
                            "Corpus-wide WER (all clips, including the silent ones) "
                            "is y_wer_all_clips.")},
        "n_failed_rows_excluded": report.get("failures", {}).get("n_failed", 0),
    }


def _factor_prefix(cond: dict, model: str | None = None) -> str:
    return (f"At rt60 = {as_float(cond.get('rt60')):.2g} s, "
            f"SNR = {as_float(cond.get('snr_db')):.3g} dB, {cond.get('noise_type')}, "
            f"{cond.get('codec')}, mic_rolloff = {as_float(cond.get('mic_rolloff')):.2g}, "
            f"{model or cond.get('model')} ")


def format_dead_zone_sentence(cond: dict, model: str | None = None) -> str:
    """
    The SPEC A.R5.1 Definition-of-Done sentence. If you cannot print this with real
    numbers, this layer is not done.

    THE SENTENCE NAMES ITS OWN DENOMINATOR. The original form put a subset
    confidence next to a whole-corpus WER in one clause and read as a single
    finding; a reader could not tell they were different populations. Now the
    quoted WER is always the one the confidence is comparable to, the number of
    silent clips is stated, and when they disagree the mismatched pairing is shown
    being rejected rather than omitted. A MUTE condition gets its own sentence,
    because "returns mean word confidence nan" is not a sentence.
    """
    n_clips = cond.get("n_clips")
    n_silent = int(as_float(cond.get("n_silent"), 0) or 0)
    n_ref = cond.get("n_ref_total")
    conf = as_float(cond.get("mean_conf"))
    if not np.isfinite(conf):
        return (_factor_prefix(cond, model)
                + f"returned an EMPTY transcript on every one of its {n_clips} clips "
                  f"(WER 1.000, 100% deletions). There is no word confidence here to "
                  f"be wrong about: this is a MUTE ZONE, not a dead zone "
                  f"(n = {n_clips} clips, {n_ref} ref words).")
    wer_spoke = as_float(cond.get("wer_spoke"))
    wer_all = as_float(cond.get("wer"))
    n_spoke = cond.get("n_spoke", n_clips)
    base = (_factor_prefix(cond, model)
            + f"returns mean word confidence {conf:.3f} while WER is "
              f"{wer_spoke:.3f}")
    if n_silent:
        return (base + f" on the {n_spoke} of {n_clips} clips where it emitted words "
                       f"— the only clips that confidence is averaged over. Over all "
                       f"{n_clips} clips ({n_silent} came back empty) WER is "
                       f"{wer_all:.3f}; pairing that with the same confidence would "
                       f"report a gap of {as_float(cond.get('gap_all_clips')):+.3f} "
                       f"instead of {as_float(cond.get('gap_spoke')):+.3f} "
                       f"(n = {n_clips} clips, {n_ref} ref words).")
    return (base + f" (n = {n_clips} clips, {n_ref} ref words; 0 clips came back "
                   f"empty, so the confidence and the WER are averaged over the "
                   f"same clips).")


def format_dead_zone_table(dead_zones: Sequence[dict], top_k: int = 20,
                           title: str | None = None) -> str:
    """Fixed-width ranked dead-zone table for the terminal / write-up."""
    lines = [title or ("Dead zones — confidently wrong ON THE CLIPS THE MODEL SPOKE "
                       "ON, ranked by the same-subset gap:"),
             f"  {'#':>2} {'condition':<34} {'conf':>6} {'pct':>5} {'WERsp':>6} "
             f"{'WERall':>6} {'silent':>7} {'gap':>7} {'gapAll':>7} {'n':>4}"]
    if not dead_zones:
        lines.append("  (none: no condition is both high-WER and high-confidence)")
        return "\n".join(lines)
    for i, r in enumerate(dead_zones[:top_k], 1):
        n_clips = int(as_float(r.get("n_clips"), 0) or 0)
        n_sil = int(as_float(r.get("n_silent"), 0) or 0)
        lines.append(
            f"  {i:>2} {str(r['condition_name'])[:34]:<34} "
            f"{as_float(r.get('mean_conf')):>6.3f} {as_float(r.get('conf_pct')):>5.2f} "
            f"{as_float(r.get('wer_spoke')):>6.3f} {as_float(r.get('wer')):>6.3f} "
            f"{f'{n_sil}/{n_clips}':>7} "
            f"{as_float(r.get('gap_spoke', r.get('gap'))):>+7.3f} "
            f"{as_float(r.get('gap_all_clips')):>+7.3f} "
            f"{n_clips:>4}")
    lines.append("  WERsp = WER on the clips that emitted words (paired with conf); "
                 "WERall = all clips.")
    lines.append("  gap = conf - (1 - WERsp)  [same subset];  "
                 "gapAll = conf - (1 - WERall)  [MISMATCHED — shown to be rejected].")
    return "\n".join(lines)


def format_mute_table(mute_zones: Sequence[dict], top_k: int = 20) -> str:
    """The conditions the model returned nothing for, on every clip."""
    lines = ["MUTE ZONES — the model emitted NO words on ANY clip (WER 1.000, 100% "
             "deletions).",
             "  These are the worst conditions measured. They cannot be dead zones: "
             "there is no",
             "  confidence to be wrong with, so a confidence-based monitor is blind "
             "to them entirely.",
             f"  {'#':>2} {'condition':<44} {'clips':>6} {'WER':>6}"]
    if not mute_zones:
        lines.append("  (none: every condition produced words on at least one clip)")
        return "\n".join(lines)
    for i, r in enumerate(mute_zones[:top_k], 1):
        lines.append(f"  {i:>2} {str(r['condition_name'])[:44]:<44} "
                     f"{r.get('n_clips', 0):>6} {as_float(r.get('wer')):>6.3f}")
    return "\n".join(lines)


def format_silence_driven_table(rows: Sequence[dict], top_k: int = 20) -> str:
    """Cells the OLD all-clips pairing called dead zones and the corrected one does not."""
    lines = ["SILENCE-DRIVEN — flagged ONLY by the mismatched all-clips pairing.",
             "  On the clips it spoke on the model was not wrong enough to qualify; "
             "the apparent",
             "  gap is clips vanishing, not confident errors. Dangerous, different "
             "mechanism, different fix.",
             f"  {'#':>2} {'condition':<34} {'conf':>6} {'WERsp':>6} {'WERall':>6} "
             f"{'silent':>7} {'gap':>7} {'gapAll':>7}"]
    if not rows:
        lines.append("  (none: the two pairings agree on every condition)")
        return "\n".join(lines)
    for i, r in enumerate(rows[:top_k], 1):
        n_clips = int(as_float(r.get("n_clips"), 0) or 0)
        n_sil = int(as_float(r.get("n_silent"), 0) or 0)
        lines.append(
            f"  {i:>2} {str(r['condition_name'])[:34]:<34} "
            f"{as_float(r.get('mean_conf')):>6.3f} "
            f"{as_float(r.get('wer_spoke')):>6.3f} {as_float(r.get('wer')):>6.3f} "
            f"{f'{n_sil}/{n_clips}':>7} "
            f"{as_float(r.get('gap_spoke')):>+7.3f} "
            f"{as_float(r.get('gap_all_clips')):>+7.3f}")
    return "\n".join(lines)


def format_report(report: dict) -> str:
    """Full terminal summary: failures and silence first, good news second,
    dead zones / silence-driven cells / mute zones last."""
    f, c = report["failures"], report["correlation"]
    sil = report.get("silence", {})
    cats = report.get("categories", {})
    gs = report["gap_summary"]
    gsa = gs.get("all_clips_pairing", {})
    lines = [
        f"D1 confidence-accuracy gap — model {report['model']!r}",
        f"  conditions: {report['n_conditions']}   clip-rows used: {report['n_clip_rows_used']}",
        f"  failed rows EXCLUDED: {f['n_failed']} / {f['n_rows']} "
        f"({f['failure_rate']:.2%})" + (f"; all-failed conditions: "
                                        f"{f['all_failed_conditions']}"
                                        if f["all_failed_conditions"] else ""),
        f"  SILENT clip-rows (empty transcript, no confidence): "
        f"{sil.get('n_silent', 0)} / {sil.get('n_rows', 0)} "
        f"({as_float(sil.get('silent_rate'), 0.0):.2%}) across "
        f"{sil.get('n_conditions_with_silence', 0)} conditions; "
        f"{len(sil.get('fully_silent_conditions', []))} conditions silent on EVERY clip",
        f"  global spearman(conf_pct, WER_spoke) = {c['spearman_confpct_vs_wer']:.3f} "
        f"-> {c['verdict']}   [all-clips pairing: "
        f"{as_float(c.get('spearman_confpct_vs_wer_all_clips')):.3f}]",
        f"  gap (same subset) mean {gs.get('mean', float('nan')):+.3f}, "
        f"overconfident in {gs.get('frac_overconfident', float('nan')):.0%} of conditions",
        f"  gap (all-clips pairing, MISMATCHED) mean {gsa.get('mean', float('nan')):+.3f} "
        f"— inflation mean {gs.get('inflation', {}).get('mean', float('nan')):+.3f}, "
        f"max {gs.get('inflation', {}).get('max', float('nan')):+.3f}",
        f"  categories: {cats.get('n_dead_zones', 0)} dead zone "
        f"({as_float(cats.get('dead_zone_rate'), float('nan')):.2%}), "
        f"{cats.get('n_silence_driven', 0)} silence-driven, "
        f"{cats.get('n_mute_zones', 0)} mute"
        f"   [the all-clips pairing alone would have called "
        f"{cats.get('n_dead_zones_all_clips_pairing', 0)} of them dead zones]",
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
    lines += ["", format_silence_driven_table(report.get("silence_driven", []))]
    lines += ["", format_mute_table(report.get("mute_zones", []))]
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
    ap.add_argument("--report-out", default="results/confidence_gap.txt",
                    help="terminal report, saved verbatim as a citable artifact")
    ap.add_argument("--wer-hi", type=float, default=WER_HI)
    ap.add_argument("--conf-pct-hi", type=float, default=CONF_PCT_HI)
    args = ap.parse_args(argv)

    rows = load_master_table(args.table)
    flagged: list[dict] = []
    texts: list[str] = []
    counts = {"dead_zone": 0, "silence_driven": 0, "mute_zone": 0}
    for m in ([args.model] if args.model else sorted(split_by_model(rows))):
        rep = confidence_gap_report(rows, model=m, wer_hi=args.wer_hi,
                                    conf_pct_hi=args.conf_pct_hi)
        text = format_report(rep)
        print(text, "\n")
        texts.append(text)
        flagged.extend(flagged_conditions(rep))
        counts["dead_zone"] += len(rep["dead_zones"])
        counts["silence_driven"] += len(rep["silence_driven"])
        counts["mute_zone"] += len(rep["mute_zones"])
    # one CSV, model column included — rows stay separable, the file is one artifact.
    # All three categories are written: a file that lists only the dead zones drops
    # the mute conditions, which are the worst cells measured.
    path = write_dead_zones_csv(flagged, args.out)
    print(f"wrote {len(flagged)} flagged rows -> {path}   "
          f"({counts['dead_zone']} dead_zone, {counts['silence_driven']} "
          f"silence_driven, {counts['mute_zone']} mute_zone)")
    if args.report_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(texts) + "\n")
        print(f"wrote report -> {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
