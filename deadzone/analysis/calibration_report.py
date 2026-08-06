"""
analysis/calibration_report.py — L2 runner: learned confidence calibration on the
REAL grid, with the numbers made final (SPEC §5 "L2", appendix A.R5.8).

WHAT THIS FILE IS. `calibration.py` is the machinery (TemperatureScaler,
FeatureCalibrator, ECE, reliability curves) and `analysis/layers.py` is the
real-data plumbing (`word_records` -> `grouped_split` -> `run_l2_calibration`).
Neither is re-implemented here. This module is the *reporting* layer: it runs L2
over `results/master.csv`, and it answers the three questions that decide whether
the ECE numbers mean anything.

  1. IS THE SPLIT HONEST? Words from one clip+condition share an acoustic
     realization, a speaker and a noise crop. A random word-level split leaks the
     test conditions into training and reports a *better* ECE — a fake
     improvement that nobody investigates because it points the right way. The
     split here is GROUPED (whole conditions held out; whole clips as a secondary
     protocol), and the grouping is re-asserted at report time rather than
     trusted (`_assert_grouped`). Every ECE is quoted with a SEED BAND, because a
     single grouped split of 176 conditions is itself a sample.

  2. DID THE ALIGNMENT FIX MOVE THE NUMBER? The previously-reported L2 figures
     were fit while 123 rows (1.75%) were being SKIPPED: `edits` are built from
     normalized tokens, `word_confidences` from raw transcript tokens, and the
     two disagree in length whenever normalization changes the token count.
     `audio_pipeline.align_confidences` now carries the confidences through the
     same transformation as the text, so those rows are recovered. This module
     re-fits on the LEGACY subset (the rows the old skip path kept) under the
     identical protocol and reports the delta. A defect that is real but barely
     moves the headline is an honest result and is reported as one — the point of
     the fix is that a `zip()` would have bound confidences to the WRONG words,
     which is wrong invisibly, not that the ECE was far off.

  3. WHAT CANNOT THIS ANALYSIS SEE? Deleted reference words have no hypothesis
     token, hence no confidence, hence no row in any calibration dataset — by
     construction, permanently. On this grid deletions are the DOMINANT error
     (D2), so this is a substantive hole in the headline signal, not a footnote.
     It is quantified three ways: as a fraction of reference words, as a fraction
     of all errors, and as the gap between the accuracy the calibrator calibrates
     (emitted-word accuracy) and the accuracy a deployment actually gets
     (reference-word recovery). No attempt is made to patch around it.

FRAMING. Vendor confidence is not a calibrated probability by construction. That
is the PREMISE of this layer, not a defect it uncovered: Deepgram returns an
acoustic posterior over its own hypothesis, never a promise about word-error
rate. L2 asks whether a thin learned layer on top can turn it into one.

    python3 -m deadzone.analysis.calibration_report                       # defaults
    python3 -m deadzone.analysis.calibration_report --seeds 0,1,2,3,4

Writes results/calibration.json (full payload incl. reliability diagrams) and
results/calibration.txt (the terminal report). Deps: numpy + calibration.py.
No API, no audio.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Mapping, Sequence

import numpy as np

# Allow `python deadzone/analysis/calibration_report.py` as well as `-m`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.analysis.layers import (                                    # noqa: E402
    AlignmentError, SPLIT_MODES, _as_int, _as_list, load_master_csv,
    run_l2_calibration, usable_rows, word_records,
)
from deadzone.audio_pipeline import _COMPOUNDS, _normalize_one            # noqa: E402
from deadzone.design import DEFAULT_FACTOR_SPACE, FactorSpace             # noqa: E402

__all__ = [
    "DEFAULT_SEEDS", "DISCOUNT_AT", "PRIMARY_SPLIT",
    "legacy_split", "misalignment_causes", "alignment_recovery",
    "deletion_blindness", "seed_band", "build_report", "format_report",
    "write_report", "main",
    "build_multi_model_report", "format_multi_report", "resolve_models",
]

DEFAULT_TABLE = "results/master.csv"
DEFAULT_MODEL = "nova-3"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

# L2 IS A WITHIN-MODEL LAYER, so every arm joins — including arms excluded from
# the cross-model WER comparison (`model_compare.WER_INCOMPARABLE_ARMS`). ECE
# asks "does THIS arm's confidence predict THIS arm's own word correctness";
# nothing is subtracted across arms, and each arm's confidence never leaves its
# own scale.
#
# THE ONE THING THAT DOES CARRY OVER, stated because it would otherwise be
# assumed away: the word-level correctness LABEL comes from the same alignment
# that produces WER (`match` -> 1, `sub`/`ins` -> 0). So an arm whose orthography
# disagrees with the reference has some correctly-recognised words labelled
# incorrect, which depresses its measured accuracy and therefore inflates its
# apparent overconfidence. Its ECE is an UPPER BOUND, not a point estimate, and
# the cross-arm ECE column is labelled accordingly. Within one arm the labels are
# wrong in a fixed direction, so the BEFORE/AFTER calibration improvement — which
# is what this layer is for — remains readable.
PRIMARY_MODEL = DEFAULT_MODEL

# Held-out CONDITIONS is the primary protocol: it is the strongest question you
# can ask a calibrator ("generalize to an acoustic condition you have never
# seen"), and it is the protocol the earlier L2 numbers used, so the before/after
# comparison is like-for-like. Held-out CLIPS is reported alongside as the
# robustness check ("generalize to a new utterance in a known condition").
PRIMARY_SPLIT = "condition"
SECONDARY_SPLIT = "clip"

# The thresholds the plain-language statement is read off. rt60=0.7 is the one
# A.R5.8 names; 0.45 is the grid's mid-reverb level, and mic_rolloff=0.5 is
# included because D2 found the channel factors drive deletions.
DISCOUNT_AT = {"rt60": 0.7, "mic_rolloff": 0.5}


# ===========================================================================
# The alignment fix: what it recovered, and whether it moved anything
# ===========================================================================

def _n_hyp_words(row: Mapping) -> int:
    return sum(1 for e in _as_list(row.get("edits"))
               if len(e) >= 3 and e[2] is not None)


def _n_confidences(row: Mapping) -> int:
    return len(_as_list(row.get("word_confidences")))


def legacy_split(rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    """Reproduce the PRE-FIX partition: (rows the old skip path kept, rows it dropped).

    The old `word_records` compared `len(hyp_edits)` against the raw
    `word_confidences` length and dropped the whole row when they disagreed.
    Selecting on exactly that criterion — before any re-alignment — reconstructs
    the dataset the previously-reported ECE was fit on, so the two fits differ in
    their DATA and in nothing else.
    """
    kept, dropped = [], []
    for r in rows:
        (kept if _n_hyp_words(r) == _n_confidences(r) else dropped).append(dict(r))
    return kept, dropped


def misalignment_causes(rows: Sequence[Mapping]) -> dict:
    """Attribute each misaligned row to the normalization event that caused it.

    Names the mechanism instead of reporting a bare count: a raw token that
    SPLITS (``follow-up`` -> two scored tokens, one confidence) makes the
    confidence list too short; a pair that MERGES (``wi fi`` -> ``wifi``) makes
    it too long. Both break the 1:1 assumption; a `zip()` would truncate to the
    shorter list and bind every confidence past the event to the wrong word.
    """
    splits: collections.Counter = collections.Counter()
    merges: collections.Counter = collections.Counter()
    for r in rows:
        toks = str(r.get("transcript") or "").split()
        for t in toks:
            n = len(_normalize_one(t))
            if n != 1:
                splits[f"{t} -> {n} tokens"] += 1
        pieces = [p for t in toks for p in _normalize_one(t)]
        i = 0
        while i < len(pieces):
            if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) in _COMPOUNDS:
                merges[f"{pieces[i]} {pieces[i + 1]} -> "
                       f"{_COMPOUNDS[(pieces[i], pieces[i + 1])]}"] += 1
                i += 2
            else:
                i += 1
    return {"split_events": dict(splits.most_common()),
            "merge_events": dict(merges.most_common())}


def alignment_recovery(rows: Sequence[Mapping]) -> dict:
    """How many rows / hypothesis words the fix brings back, and why they were lost."""
    kept, dropped = legacy_split(rows)
    full = word_records(rows)                       # raises if anything is STILL bad
    legacy = word_records(kept)
    return {
        "n_rows_total": len(rows),
        "n_rows_realigned": full["n_realigned_rows"],
        "n_rows_dropped_by_old_path": len(dropped),
        "frac_rows_dropped_by_old_path": len(dropped) / len(rows) if rows else float("nan"),
        "n_rows_still_misaligned": full["n_misaligned_rows"],
        "n_words_after_fix": full["n_words"],
        "n_words_before_fix": legacy["n_words"],
        "n_words_recovered": full["n_words"] - legacy["n_words"],
        "frac_words_recovered": ((full["n_words"] - legacy["n_words"]) / full["n_words"]
                                 if full["n_words"] else float("nan")),
        "causes": misalignment_causes(dropped),
        "note": ("the old path SKIPPED these rows and counted them (it never "
                 "zipped, so no confidence was ever bound to the wrong word); "
                 "the fix carries confidences through the same split/merge/drop "
                 "transformation as the text, so the rows are scored instead of "
                 "discarded"),
    }


# ===========================================================================
# The blind spot: deletions carry no confidence
# ===========================================================================

def silent_conditions(rows: Sequence[Mapping]) -> list[dict]:
    """Conditions that emitted NO hypothesis words at all, over every clip.

    The extreme of the blind spot and the reason it is not a footnote: a cell
    where the model returns an empty transcript is 100% deletions, contributes
    ZERO rows to the calibration dataset, and is therefore excluded from the
    grouped split entirely. The calibrator is silent about exactly the cells
    where the system fails hardest.
    """
    per: dict[str, dict] = {}
    for r in rows:
        c = per.setdefault(str(r.get("condition_name")),
                           {"condition_name": str(r.get("condition_name")),
                            "n_rows": 0, "n_hyp_words": 0, "wer_sum": 0.0})
        c["n_rows"] += 1
        c["n_hyp_words"] += _n_hyp_words(r)
        try:
            c["wer_sum"] += float(r.get("wer"))
        except (TypeError, ValueError):
            pass
    out = []
    for c in per.values():
        if c["n_hyp_words"] == 0:
            out.append({"condition_name": c["condition_name"],
                        "n_rows": c["n_rows"],
                        "mean_wer": c["wer_sum"] / c["n_rows"] if c["n_rows"] else float("nan")})
    return sorted(out, key=lambda d: d["condition_name"])


def deletion_blindness(rows: Sequence[Mapping]) -> dict:
    """Quantify what confidence calibration structurally cannot see.

    Four numbers, because one is not enough:
      * deletions as a fraction of REFERENCE words — how much of the ground truth
        the calibrated confidence says nothing about;
      * deletions as a fraction of all ERRORS — whether the invisible failure
        mode is the marginal one or the dominant one;
      * emitted-word accuracy vs reference recovery — the two accuracies that a
        calibrated confidence could be read as, and the gap between them. A
        perfectly calibrated confidence converges on the FIRST one, so quoting
        it as if it were the second overstates the system by that gap;
      * the SILENT CONDITIONS — cells with no hypothesis words anywhere, which
        drop out of the dataset entirely (see `silent_conditions`).
    """
    c: collections.Counter = collections.Counter()
    for r in rows:
        for k in ("n_ref", "n_match", "n_sub", "n_del", "n_ins"):
            c[k] += _as_int(r.get(k))
    n_ref, n_match = c["n_ref"], c["n_match"]
    n_emitted = n_match + c["n_sub"] + c["n_ins"]
    n_err = c["n_sub"] + c["n_del"] + c["n_ins"]
    emitted_acc = n_match / n_emitted if n_emitted else float("nan")
    recovery = n_match / n_ref if n_ref else float("nan")
    silent = silent_conditions(rows)
    n_cond = len({str(r.get("condition_name")) for r in rows})
    return {
        "n_conditions": n_cond,
        "n_conditions_in_calibration_set": n_cond - len(silent),
        "n_silent_conditions": len(silent),
        "silent_conditions": silent,
        "n_ref_words": n_ref,
        "n_deletions": c["n_del"],
        "n_substitutions": c["n_sub"],
        "n_insertions": c["n_ins"],
        "n_matches": n_match,
        "n_hypothesis_words": n_emitted,
        "deleted_fraction_of_reference": c["n_del"] / n_ref if n_ref else float("nan"),
        "deleted_fraction_of_errors": c["n_del"] / n_err if n_err else float("nan"),
        "emitted_word_accuracy": emitted_acc,
        "reference_word_recovery": recovery,
        "accuracy_overstatement": emitted_acc - recovery,
        "note": ("a deleted reference word has no hypothesis token and therefore "
                 "no confidence: it can never appear in a calibration dataset. "
                 "A perfectly calibrated confidence converges on emitted-word "
                 "accuracy, NOT on how much of the reference was recovered — "
                 "quoting it as the latter overstates the system by "
                 f"{emitted_acc - recovery:.3f} on this grid. Since deletions "
                 "are the dominant error mode here (D2), this is a substantive "
                 "limit on the headline signal, not a footnote. In the limit, a "
                 "condition where the model returns an empty transcript "
                 "contributes NOTHING to the calibration set and drops out of "
                 f"the grouped split entirely — {len(silent)} of {n_cond} "
                 "conditions on this grid do exactly that, and they are the "
                 "worst cells in it. Measured and stated, never patched around."),
    }


# ===========================================================================
# Fitting: one protocol, several seeds
# ===========================================================================

def _assert_grouped(rows: Sequence[Mapping], rep: Mapping, split_by: str) -> None:
    """Re-verify at report time that no group straddles the split.

    `grouped_split` is tested, but the claim "the split is grouped" is the single
    load-bearing methodological claim of this layer — its violation makes the ECE
    improvement fake and points the *encouraging* way. So it is re-derived here
    from the same seed rather than inherited on trust.
    """
    from deadzone.analysis.layers import grouped_split
    w = word_records(rows)
    tr, te = grouped_split(w["records"], split_by=split_by,
                           frac=rep["frac"], seed=rep["seed"])
    key = "condition_name" if split_by == "condition" else "clip_id"
    g = lambda idx: {str(w["records"][i][key]) for i in idx}      # noqa: E731
    leak = g(tr) & g(te)
    if leak:
        raise AssertionError(
            f"{len(leak)} {key} value(s) appear on BOTH sides of the split "
            f"({sorted(leak)[:3]}...) — the ECE improvement would be leakage")
    if int(len(tr)) != int(rep["n_train_words"]) or int(len(te)) != int(rep["n_test_words"]):
        raise AssertionError("re-derived split does not match the reported one")


def _fit(rows: Sequence[Mapping], space: FactorSpace, split_by: str,
         frac: float, seed: int) -> dict:
    rep = run_l2_calibration(rows, space, split_by=split_by, frac=frac,
                             seed=seed, discount_at=DISCOUNT_AT)
    rep["seed"], rep["frac"] = seed, frac
    return rep


def seed_band(rows: Sequence[Mapping], space: FactorSpace = DEFAULT_FACTOR_SPACE,
              split_by: str = PRIMARY_SPLIT, frac: float = 0.5,
              seeds: Sequence[int] = DEFAULT_SEEDS) -> dict:
    """Fit the same protocol under several grouped splits; report the BAND.

    One grouped split is one draw. Quoting its ECE alone invites the reader to
    treat a split-to-split wobble as a real difference between calibrators, so
    every headline number here carries min/median/max across seeds.
    """
    if split_by not in SPLIT_MODES:
        raise ValueError(f"split_by must be one of {SPLIT_MODES}, got {split_by!r}")
    seeds = list(seeds)
    # THE BAND IS THE CLAIM, so the seeds have to be distinct draws. A repeated
    # seed re-runs the IDENTICAL grouped split, so `[0, 0, 0]` produces three
    # identical fits and a zero-width band reported as "median [min, max] over 3
    # grouped splits" — the strongest-looking form of the number, from one draw.
    if not seeds:
        raise ValueError(
            "seed_band got no seeds; np.median over an empty list is nan and "
            "every ECE would print as nan rather than as an error.")
    dupes = sorted({s for s in seeds if seeds.count(s) > 1})
    if dupes:
        raise ValueError(
            f"duplicate grouped-split seed(s) {dupes} in {seeds}: each repeat "
            f"re-runs the identical split, so the reported band collapses to "
            f"zero width while still being labelled 'median [min, max] over "
            f"{len(seeds)} grouped splits'. Pass distinct seeds.")

    fits = [_fit(rows, space, split_by, frac, s) for s in seeds]

    def band(vals):
        v = [float(x) for x in vals]
        if not v or not all(np.isfinite(v)):
            # An ECE of nan comes from a split whose held-out side had no words
            # in any confidence bin. np.median would carry it into the headline
            # as `nan`, which reads as "no miscalibration measured".
            raise ValueError(
                f"non-finite value in a seed band over {len(v)} seed(s): {v}. "
                f"A nan ECE means that split produced no scorable held-out "
                f"words; medianing it into the band would print the headline "
                f"as nan rather than raising.")
        return {"median": float(np.median(v)), "min": float(np.min(v)),
                "max": float(np.max(v)), "per_seed": v}
    return {
        "split_by": split_by, "frac": frac, "seeds": list(seeds),
        "n_words": fits[0]["n_words"],
        "n_groups": fits[0]["n_groups"],
        "ece_raw": band([f["ece_raw"] for f in fits]),
        "ece_temperature": band([f["temperature"]["ece"] for f in fits]),
        "ece_feature": band([f["feature"]["ece"] for f in fits]),
        "temperature_T": band([f["temperature"]["T"] for f in fits]),
        "discount": {
            k: band([f["discount"][k]["mean_discount"] for f in fits
                     if k in f["discount"]])
            for k in DISCOUNT_AT
            if any(k in f["discount"] for f in fits)
        },
        "headline_fit": fits[0],          # seed 0: full payload incl. reliability
    }


# ===========================================================================
# The report
# ===========================================================================

def build_report(table: str = DEFAULT_TABLE, model: str = DEFAULT_MODEL,
                 space: FactorSpace = DEFAULT_FACTOR_SPACE,
                 seeds: Sequence[int] = DEFAULT_SEEDS,
                 frac: float = 0.5,
                 rows: Sequence[Mapping] | None = None) -> dict:
    """The whole L2 result: ECE bands under both grouped protocols, the alignment
    recovery accounting, the deletion blind spot, and the plain-language statement."""
    raw_rows = list(rows) if rows is not None else load_master_csv(table)
    mine = [r for r in raw_rows if str(r.get("model")) == model]
    if not mine:
        raise ValueError(f"no rows for model {model!r} in {table!r}")
    ok, failed = usable_rows(mine)
    if not ok:
        raise ValueError(f"every row for model {model!r} is failed=True")

    recovery = alignment_recovery(ok)
    primary = seed_band(ok, space, PRIMARY_SPLIT, frac, seeds)
    secondary = seed_band(ok, space, SECONDARY_SPLIT, frac, seeds)
    _assert_grouped(ok, primary["headline_fit"], PRIMARY_SPLIT)
    _assert_grouped(ok, secondary["headline_fit"], SECONDARY_SPLIT)

    # Same protocol, pre-fix data: did recovering the rows move the number?
    legacy_rows, _ = legacy_split(ok)
    legacy = seed_band(legacy_rows, space, PRIMARY_SPLIT, frac, seeds)
    moved = {
        "d_ece_raw": primary["ece_raw"]["median"] - legacy["ece_raw"]["median"],
        "d_ece_temperature": (primary["ece_temperature"]["median"]
                              - legacy["ece_temperature"]["median"]),
        "d_ece_feature": (primary["ece_feature"]["median"]
                          - legacy["ece_feature"]["median"]),
        "legacy": {k: legacy[k] for k in
                   ("n_words", "ece_raw", "ece_temperature", "ece_feature",
                    "temperature_T")},
    }
    moved["max_abs_shift"] = max(abs(moved[k]) for k in
                                 ("d_ece_raw", "d_ece_temperature", "d_ece_feature"))
    moved["verdict"] = ("material" if moved["max_abs_shift"] >= 0.005
                        else "negligible")

    rep = {
        "layer": "L2 — learned confidence calibration (final)",
        "table": table, "model": model,
        "n_rows": len(mine), "n_failed_rows": len(failed),
        "protocol": {
            "split": PRIMARY_SPLIT,
            "secondary_split": SECONDARY_SPLIT,
            "frac_train": frac, "seeds": list(seeds),
            "split_modes_offered": list(SPLIT_MODES),
            "why": ("grouped by whole condition (primary) and whole clip "
                    "(secondary). A random word-level split is not offered by "
                    "the API: words from one clip+condition are highly "
                    "correlated, so splitting over words leaks and reports a "
                    "FAKE ECE improvement. Groups are re-verified disjoint at "
                    "report time."),
            "conf_clipping": ("confidences clipped into (0,1) before any logit — "
                              "the vendor returns exactly 1.0"),
        },
        "primary": primary,
        "secondary": secondary,
        "alignment_recovery": recovery,
        "alignment_effect_on_ece": moved,
        "deletion_blindness": deletion_blindness(ok),
        "premise": ("vendor confidence is not a calibrated probability by "
                    "construction — Deepgram returns an acoustic posterior over "
                    "its own hypothesis, not a promise about word error rate. "
                    "That is the PREMISE of this layer, not a defect it found; "
                    "L2 asks whether a thin learned layer can turn it into one."),
    }
    rep["statement"] = plain_language_statement(rep)
    return rep


def plain_language_statement(rep: Mapping) -> str:
    """The A.R5.8 DoD sentence — what the calibrator learned, in real numbers."""
    p, fit = rep["primary"], rep["primary"]["headline_fit"]
    # The arm's name comes from the report, never from a literal. `--model` is a
    # parameter, so a hardcoded "nova-3" here would attribute a THIRD arm's ECE
    # to the spine — a sentence that is fluent, numerate and about the wrong
    # model, with nothing in the output to contradict it.
    model = rep.get("model") or "the model"
    parts = [
        f"On held-out {p['split_by']}s ({p['n_groups']} groups, "
        f"{fit['n_test_words']} test words of {p['n_words']}), {model}'s raw word "
        f"confidence has ECE {p['ece_raw']['median']:.3f} "
        f"[{p['ece_raw']['min']:.3f}, {p['ece_raw']['max']:.3f}]; temperature "
        f"scaling (T={p['temperature_T']['median']:.2f}) gives "
        f"{p['ece_temperature']['median']:.3f} "
        f"[{p['ece_temperature']['min']:.3f}, {p['ece_temperature']['max']:.3f}] "
        f"and the feature-conditioned calibrator "
        f"{p['ece_feature']['median']:.3f} "
        f"[{p['ece_feature']['min']:.3f}, {p['ece_feature']['max']:.3f}] "
        f"(median [min, max] over {len(p['seeds'])} grouped splits)."
    ]
    for fname, d in (fit.get("discount") or {}).items():
        band = p["discount"].get(fname, {})
        parts.append(
            f"Above {fname} = {d['threshold']:g}, reported confidence must be "
            f"discounted by ~{band.get('median', d['mean_discount']):.2f} to "
            f"become a calibrated probability ({d['mean_raw_conf']:.2f} reported "
            f"vs {d['empirical_accuracy']:.2f} observed accuracy on "
            f"{d['n_words']} held-out words).")
    db = rep["deletion_blindness"]
    parts.append(
        f"Blind spot: {db['n_deletions']} deleted reference words "
        f"({db['deleted_fraction_of_reference']:.1%} of the reference, "
        f"{db['deleted_fraction_of_errors']:.1%} of ALL errors) carry no "
        f"hypothesis token and therefore no confidence, so a perfectly "
        f"calibrated confidence converges on emitted-word accuracy "
        f"({db['emitted_word_accuracy']:.3f}), not on reference recovery "
        f"({db['reference_word_recovery']:.3f}) — an overstatement of "
        f"{db['accuracy_overstatement']:.3f} if read as the latter.")
    if db["n_silent_conditions"]:
        parts.append(
            f"At the limit, {db['n_silent_conditions']} of {db['n_conditions']} "
            f"conditions returned an empty transcript on every clip (WER 1.00, "
            f"100% deletions) and so contribute zero words to the calibration "
            f"set: the calibrator is fit on "
            f"{db['n_conditions_in_calibration_set']} conditions and is silent "
            f"about the worst {db['n_silent_conditions']}.")
    return " ".join(parts)


def _fmt_band(b: Mapping) -> str:
    return f"{b['median']:.4f} [{b['min']:.4f}, {b['max']:.4f}]"


def format_report(rep: Mapping) -> str:
    p, s = rep["primary"], rep["secondary"]
    rec, mv, db = (rep["alignment_recovery"], rep["alignment_effect_on_ece"],
                   rep["deletion_blindness"])
    L = [
        f"{rep['layer']} — model {rep['model']!r}, table {rep['table']}",
        f"  rows {rep['n_rows']} ({rep['n_failed_rows']} failed, excluded)   "
        f"words {p['n_words']}",
        "",
        f"PROTOCOL: grouped by {p['split_by']} ({p['n_groups']} groups), "
        f"train frac {p['frac']}, seeds {p['seeds']}",
        f"  offered split modes: {rep['protocol']['split_modes_offered']} "
        "— a random word-level split is NOT offered (it leaks; symptom is a "
        "better ECE)",
        "",
        "ECE (median [min, max] over seeds), held-out CONDITIONS:",
        f"  raw confidence           {_fmt_band(p['ece_raw'])}",
        f"  + temperature scaling    {_fmt_band(p['ece_temperature'])}"
        f"   (T = {_fmt_band(p['temperature_T'])})",
        f"  + feature-conditioned    {_fmt_band(p['ece_feature'])}",
        "",
        "ECE, held-out CLIPS (robustness check, same machinery):",
        f"  raw {_fmt_band(s['ece_raw'])}   temp {_fmt_band(s['ece_temperature'])}"
        f"   feature {_fmt_band(s['ece_feature'])}",
        "",
        "ALIGNMENT FIX — what it recovered:",
        f"  rows re-aligned by align_confidences: {rec['n_rows_realigned']} / "
        f"{rec['n_rows_total']} ({rec['frac_rows_dropped_by_old_path']:.2%}); "
        f"still misaligned: {rec['n_rows_still_misaligned']}",
        f"  hypothesis words fit on: {rec['n_words_before_fix']} (old skip path) "
        f"-> {rec['n_words_after_fix']} (+{rec['n_words_recovered']}, "
        f"{rec['frac_words_recovered']:.2%})",
    ]
    for k, v in (rec["causes"]["merge_events"] or {}).items():
        L.append(f"    merge  {k}: {v} rows (confidence list one too LONG)")
    for k, v in (rec["causes"]["split_events"] or {}).items():
        L.append(f"    split  {k}: {v} rows (confidence list one too SHORT)")
    L += [
        f"  effect on the headline ECE: raw {mv['d_ece_raw']:+.4f}, "
        f"temperature {mv['d_ece_temperature']:+.4f}, "
        f"feature {mv['d_ece_feature']:+.4f}  -> {mv['verdict'].upper()}",
        "  (the defect was real — a zip() would have bound confidences to the "
        "wrong words — but the old path SKIPPED rather than zipped, so the "
        "reported number was already safe; recovering the rows barely moves it.)",
        "",
        "DELETION BLINDNESS — what this analysis structurally cannot see:",
        f"  deletions {db['n_deletions']} = {db['deleted_fraction_of_reference']:.1%} "
        f"of reference words, {db['deleted_fraction_of_errors']:.1%} of ALL errors",
        f"  emitted-word accuracy {db['emitted_word_accuracy']:.3f} vs reference "
        f"recovery {db['reference_word_recovery']:.3f} "
        f"(overstatement {db['accuracy_overstatement']:.3f})",
        f"  silent conditions (empty transcript on every clip, 0 words in the "
        f"calibration set): {db['n_silent_conditions']} / {db['n_conditions']}",
    ]
    for c in db["silent_conditions"]:
        L.append(f"    {c['condition_name']}  (n={c['n_rows']} clips, "
                 f"mean WER {c['mean_wer']:.2f})")
    L += [
        "",
        "PREMISE: " + rep["premise"],
        "",
        "STATEMENT: " + rep["statement"],
    ]
    return "\n".join(L)


# ===========================================================================
# EVERY ARM — L2 is within-model, so nothing is excluded here
# ===========================================================================

def resolve_models(table: str, spec: str,
                   rows: Sequence[Mapping] | None = None) -> list[str]:
    """`--model` -> the arm list. `all` discovers every arm in the table.

    Discovery is the point: a fixed default is how the previously-published L2
    stayed a one-arm result after a third arm had been paid for and run (the
    same failure `model_arms.discover_arms` exists to prevent).
    """
    spec = str(spec or "").strip()
    if spec and spec.lower() != "all":
        return [m.strip() for m in spec.split(",") if m.strip()]
    rows = list(rows) if rows is not None else load_master_csv(table)
    found = sorted({str(r.get("model")) for r in rows if r.get("model")})
    if not found:
        raise ValueError(f"no `model` values in {table!r}; nothing to calibrate.")
    return found


def build_multi_model_report(table: str = DEFAULT_TABLE,
                             models: Sequence[str] | None = None,
                             seeds: Sequence[int] = DEFAULT_SEEDS,
                             frac: float = 0.5,
                             rows: Sequence[Mapping] | None = None) -> dict:
    """
    One L2 report per arm, plus the cross-arm ECE table, in ONE payload.

    BACKWARD COMPATIBILITY IS DELIBERATE AND IS NOT COSMETIC. The primary arm's
    payload is splatted at the top level, so every key the single-arm artifact
    published (`primary`, `secondary`, `deletion_blindness`, `statement`, ...)
    still resolves to exactly the same arm and exactly the same numbers. Adding
    an arm must not silently redefine what `results/calibration.json`'s
    top-level `ece_raw` refers to — that is a rename disguised as an addition,
    and it is unreadable from the file itself. `models` / `by_model` are additive.
    """
    rows = list(rows) if rows is not None else load_master_csv(table)
    models = list(models or resolve_models(table, "all", rows))

    # ONE ARM MUST NOT ABORT THE LAYER, AND MUST NOT VANISH FROM IT EITHER.
    # `build_report` runs `word_records` in its STRICT mode: a row whose
    # hypothesis-word count disagrees with its confidence-list length after
    # re-alignment raises rather than zipping (a zip would bind confidences to
    # the wrong words — wrong invisibly). That is the correct behaviour and it
    # is not relaxed here. But with N arms it means a defect in ONE arm takes
    # the whole run down, and the tempting fix — falling back to
    # `on_misalign="skip"` — would publish that arm's ECE next to the others as
    # if it were the same measurement, when it was fit on a silently smaller
    # set. So a raising arm is recorded as BLOCKED, with the count, and the
    # arms that are clean are still reported.
    by_model: dict[str, dict] = {}
    blocked: list[dict] = []
    for m in models:
        try:
            by_model[m] = build_report(table, m, seeds=seeds, frac=frac,
                                       rows=rows)
        except AlignmentError as exc:
            mine = [r for r in usable_rows(rows)[0] if str(r.get("model")) == m]
            skipped = word_records(mine, on_misalign="skip")
            blocked.append({
                "model": m,
                "reason": "alignment",
                "error": str(exc),
                "n_rows": len(mine),
                "n_misaligned_rows": skipped["n_misaligned_rows"],
                "n_realigned_rows": skipped["n_realigned_rows"],
                "frac_rows_misaligned": (skipped["n_misaligned_rows"] / len(mine)
                                         if mine else float("nan")),
                "n_words_under_skip_protocol": skipped["n_words"],
                "note": ("This arm is NOT calibrated here. Its rows would fit "
                         "only under the weaker `on_misalign='skip'` protocol, "
                         "which drops the misaligned rows — a different "
                         "estimand from the other arms', and printing it in the "
                         "same column would be a silent protocol change. The "
                         "skip-protocol word count is given so the size of the "
                         "gap is visible, not so the number can be quoted."),
            })
    if not by_model:
        raise AlignmentError(
            f"every requested arm {models} failed word-level alignment; "
            f"there is nothing to calibrate. Details: {blocked}")
    models = list(by_model)
    primary = PRIMARY_MODEL if PRIMARY_MODEL in by_model else models[0]

    from deadzone.model_compare import is_wer_comparable, wer_comparability
    census = wer_comparability(models)
    cross = []
    for m in models:
        p = by_model[m]["primary"]
        cross.append({
            "model": m,
            "n_words": p["n_words"], "n_groups": p["n_groups"],
            "ece_raw": p["ece_raw"]["median"],
            "ece_temperature": p["ece_temperature"]["median"],
            "ece_feature": p["ece_feature"]["median"],
            "temperature_T": p["temperature_T"]["median"],
            "ece_reduction_feature": (p["ece_raw"]["median"]
                                      - p["ece_feature"]["median"]),
            # An arm whose orthography disagrees with the reference has some
            # correct words labelled incorrect, so its ECE is an upper bound.
            "ece_is_upper_bound": not is_wer_comparable(m),
            "deleted_fraction_of_reference":
                by_model[m]["deletion_blindness"]["deleted_fraction_of_reference"],
        })
    return {
        **by_model[primary],
        "primary_model": primary,
        "models": models,
        "models_requested": list(dict.fromkeys(list(models) + [b["model"] for b in blocked])),
        "blocked_arms": blocked,
        "by_model": by_model,
        "cross_arm": cross,
        "wer_comparability": census,
        "multi_model_note": (
            "L2 is WITHIN-MODEL: every arm joins, including arms excluded from "
            "the cross-model WER comparison, because ECE never subtracts one "
            "arm's number from another's. The top-level keys are "
            f"{primary!r}'s, unchanged, so previously published values still "
            "resolve; per-arm results are under `by_model`. CAVEAT on the "
            "cross-arm ECE column: the word-level correctness label comes from "
            "the same alignment as WER, so an arm with an orthography mismatch "
            "has correct words labelled incorrect and its ECE reads HIGH — see "
            "`ece_is_upper_bound`."),
    }


def format_multi_report(rep: Mapping) -> str:
    """Cross-arm ECE table first, then each arm's full single-arm report."""
    by_model = rep.get("by_model") or {}
    if not by_model:
        return format_report(rep)
    L = ["L2 — LEARNED CONFIDENCE CALIBRATION, EVERY ARM",
         "=" * 72, "",
         f"arms: {', '.join(rep['models'])}   (primary, and the arm the "
         f"top-level JSON keys describe: {rep['primary_model']})",
         "",
         "L2 is a WITHIN-MODEL layer. Each arm's confidence is calibrated "
         "against its OWN",
         "word correctness; nothing is subtracted across arms, so an arm "
         "excluded from the",
         "cross-model WER comparison is included here without qualification —",
         "with ONE caveat, marked '^': the correctness LABEL comes from the "
         "same alignment",
         "as WER, so an arm whose orthography disagrees with the reference has "
         "correct words",
         "labelled incorrect. Its ECE is an UPPER BOUND. The raw->calibrated "
         "IMPROVEMENT is",
         "still readable, because within one arm the labels are wrong in a "
         "fixed direction.",
         "",
         f"{'model':<20}{'words':>9}{'ECE raw':>10}{'+temp':>9}{'+feature':>10}"
         f"{'T':>7}{'reduction':>11}",
         ]
    for c in rep.get("cross_arm") or []:
        mark = "^" if c.get("ece_is_upper_bound") else " "
        L.append(f"{c['model']:<20}{c['n_words']:>9}{c['ece_raw']:>9.4f}{mark}"
                 f"{c['ece_temperature']:>9.4f}{c['ece_feature']:>10.4f}"
                 f"{c['temperature_T']:>7.2f}{c['ece_reduction_feature']:>11.4f}")
    L += ["", "NOTE: " + rep.get("multi_model_note", ""), ""]
    for b in rep.get("blocked_arms") or []:
        L += [f"ARM NOT CALIBRATED — {b['model']}  (reason: {b['reason']})",
              f"  {b['n_misaligned_rows']} of {b['n_rows']} rows "
              f"({b['frac_rows_misaligned']:.2%}) still have a hypothesis-word "
              f"count that disagrees with the",
              f"  confidence-list length after re-alignment "
              f"({b['n_realigned_rows']} rows were recovered by "
              f"align_confidences). word_records refuses to zip.",
              f"  {b['note']}",
              f"  first failure: {b['error'][:160]}",
              ""]
    for m in rep["models"]:
        L += ["=" * 72, format_report(by_model[m]), ""]
    return "\n".join(L)


def write_report(rep: Mapping, out_json: str = "results/calibration.json",
                 out_txt: str = "results/calibration.txt") -> tuple[str, str]:
    for path in (out_json, out_txt):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=float)
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write((format_multi_report(rep) if rep.get("by_model")
                  else format_report(rep)) + "\n")
    return out_json, out_txt


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="L2: learned confidence calibration on the real grid")
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--model", default="all",
                    help="arm(s) to calibrate: 'all' (default — every arm in "
                         "the table), or a comma-separated list. L2 is "
                         "within-model, so no arm is excluded.")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="comma-separated grouped-split seeds")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--out-json", default="results/calibration.json")
    ap.add_argument("--out-txt", default="results/calibration.txt")
    args = ap.parse_args(argv)

    seeds = tuple(int(s) for s in str(args.seeds).split(",") if s.strip() != "")
    models = resolve_models(args.table, args.model)
    if len(models) == 1:
        rep = build_report(args.table, models[0], seeds=seeds, frac=args.frac)
        print(format_report(rep))
    else:
        rep = build_multi_model_report(args.table, models, seeds=seeds,
                                       frac=args.frac)
        print(format_multi_report(rep))
    j, t = write_report(rep, args.out_json, args.out_txt)
    print(f"\nwrote {j} and {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
