"""
L1 — multi-model comparison on real data (SPEC A.R5.7).

**N arms, not two.** The layer was built and validated against two — **nova-3**,
the commercial model that exposes per-word confidence (the spine), and
**whisper-base**, the open baseline that shows we benchmark rather than depend on
one vendor's API — and a third (ElevenLabs `scribe_v2`) is expected. Nothing here
may be pairwise: every arm in the table is discovered, compared and reported.

## The two-arm hazard this module was audited for

The dangerous shape is not a crash — it is an arm that quietly does not appear.
`matched_arms` used to default to `(SPINE_MODEL, BASELINE_MODEL)`, so a third arm
sitting in `master.csv` was dropped before any analysis ran and the report came
out complete, well-formed and silently two-armed. Same family as SPEC Appendix E
("a guard whose failure mode is silence"). So:

  * arms are **discovered from the table** unless the caller names them, and the
    set actually compared is printed in the report;
  * the dead-zone overlap is **all pairs plus an all-arm intersection**, never one
    hardcoded pair;
  * the hallucination callout is computed for **every** arm, and no key ever holds
    one arm's numbers under another arm's name;
  * `matched_arms` intersects over **all N arms** — which with three arms is a
    strictly smaller set than with two — and the shrinkage is reported as a
    census, with a floor underneath it (SPEC Appendix C.8: an unenforced matching
    silently absorbed 7.8 WER points of confound in D4).

## The three things this module is careful about

**1. Confidence is only comparable WITHIN a model.** Deepgram returns an acoustic
confidence; Whisper returns the decoder's token softmax probability
(`_parse_whisper_result` in `audio_pipeline.py` says so explicitly). They are
different quantities with different supports, and putting them on one axis would
produce a chart that looks meaningful and means nothing. Every cross-model
confidence claim here goes through `within_model_conf_percentile`, which ranks
each model against its own distribution. That is what `model_compare.py` already
does; this module does not bypass it.

**2. Absolute WER is not comparable under the spine normalization.** Whisper
writes numbers as digits and has no switch to stop it; the Deepgram adapter
disables `smart_format`/`punctuate`/`numerals`, so its output is already
word-form. On a corpus loaded with phone numbers, spelled codes and amounts, that
is a large condition-independent offset — and a constant offset is
indistinguishable from an acoustic effect once it is in the table. So this module
reports **both**: the strict spine WER (what the master table holds) and a
re-scored cross-model WER via `cross_model_norm`, which applies the Whisper
authors' own published normalizer symmetrically to both sides. See that module's
docstring for what it fixes and what residuals survive.

**3. Every arm must cover the same cells.** The Whisper arm was run on the
10-clip AL subset, not all 40. Comparing Whisper-on-10 against nova-3-on-40 would
confound the model with the clip set. This module intersects down to the cells
EVERY arm actually ran and reports the census; it refuses only when the
intersection is a data defect (duplicated cells, an arm with no usable rows) or
has collapsed below the floor where a per-condition mean stops describing an
acoustic condition. Arms legitimately running different clip subsets is the
expected case, not an error — see `arm_intersection`.

**4. A confidence is never subtracted from — or thresholded against — an accuracy
measured on DIFFERENT clips.** This is the same defect D1 was rebuilt around
(`analysis/confidence_gap.py`, `analysis/__init__.py` second trap), and L1 had an
identical instance of it: `condition_table` averaged `wer` over EVERY clip while
averaging `mean_conf` over only the clips that produced words, and then
`dead_zone_flags` tested one against the other. A clip whose transcript comes back
EMPTY scores WER 1.0 with 100% deletions and carries no per-word confidence at all,
so it inflates the WER term while contributing nothing to the confidence term. The
arithmetic is clean, the row count is right, and nothing downstream can see it.

So every condition row now carries BOTH pairings under explicit names —
`wer_all_clips` (corpus severity, the honest "what does this condition do")
and `wer_spoke` (the ONLY accuracy the confidence may be thresholded against) —
plus `n_spoke` / `n_silent` / `silent_frac`, and the dead-zone flags are computed
against `wer_spoke` through `confidence_gap.condition_flags` (which holds the
MUTE conditions out; a condition that emitted nothing on any clip has no
confidence, so it cannot be confidently wrong — it is a different, worse failure
and it is reported as its own category rather than dropped).

Measured effect on the real grid (10-clip common cell set, 176 conditions/arm):
nova-3 had 94 conditions with at least one silent clip and ONE condition flipped
out of the dead-zone set (2 -> 1); whisper-base had 0 flips, because its WERs are
so far above the threshold that the spoke-only subset is still above it.

    ./.venv/bin/python -m deadzone.analysis.model_arms
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from deadzone.analysis import as_float, is_silent_row, silence_summary
from deadzone.analysis.confidence_gap import (
    CATEGORY_DEAD_ZONE, CATEGORY_MUTE, CATEGORY_OK, CATEGORY_SILENCE_DRIVEN,
    CATEGORY_MEANING, condition_flags,
)
from deadzone.cross_model_norm import cross_model_classify_errors, cross_model_normalize
from deadzone.design import DEFAULT_FACTOR_SPACE, FactorSpace
from deadzone.model_compare import (
    WER_INCOMPARABLE_ARMS, WerIncomparableArmError, _bins_for, _region_rows,
    compare_models, confidence_wer_shape, find_divergence_regions,
    is_wer_comparable, wer_comparability, within_model_conf_percentile,
)

MASTER = "results/master.csv"
MANIFEST = "recording_manifest.csv"
OUT_JSON = Path("results/model_arms.json")
OUT_TXT = Path("results/model_arms.txt")

SPINE_MODEL = "nova-3"
# The arm whose hallucination behaviour has a named callout in the write-up. It
# is a LABEL, never a selector: every arm is measured, and this name only decides
# which arm the legacy `whisper_hallucination` key aliases (and only when that arm
# is actually present).
BASELINE_MODEL = "whisper-base"

# TWO FLOORS, because they answer two different questions and only one of them
# is a judgement call.
#
#   MIN_COMMON_CLIPS (3) is the REPORTING floor, applied by `model_arms_report`.
#   Mirrors analysis/sim2real.MIN_COMMON_CLIPS and is the same number for the
#   same reason: below it a per-condition mean stops describing an acoustic
#   condition and starts describing whichever two utterances happened to overlap.
#
#   CORRECTNESS_FLOOR (1) is `arm_intersection`'s default, and it is not a
#   judgement at all. An EMPTY intersection is the case that most needs catching
#   because it defeats every other check: with no common cells every arm is
#   equally empty, so the per-arm row-count identity passes, the equal-size check
#   passes, and the failure only surfaces later as a confusing error in an
#   aggregation step — or as a report over nothing.
#
# The split matters for N arms specifically: the intersection is monotonically
# shrinking in the number of arms, so the first time these floors bite will be
# the day a third arm is added.
MIN_COMMON_CLIPS = 3
CORRECTNESS_FLOOR = 1


class RaggedArmsError(ValueError):
    """
    Raised when the arms do not cover the same (clip, condition) cells *for a
    reason that is a data defect rather than a coverage choice*.

    Loud on purpose. A silent inner join would still produce a comparison table,
    and every number in it would be a mixture of a model effect and a coverage
    effect with no way to separate them afterwards.
    """


class ThinOverlapError(RaggedArmsError):
    """
    Raised when intersecting all N arms leaves too few clips to compare on.

    A subclass so existing `except RaggedArmsError` handlers keep working. It
    exists because the intersection is monotonically shrinking in the number of
    arms: two arms sharing the 10-clip AL subset overlap on 10 clips, but add a
    third arm run on a different subset and the common set can collapse to a
    handful — or to nothing — while every per-arm size check still passes
    (they are all equally empty) and the report still renders.
    """


def _nanmean(a: np.ndarray) -> float:
    """np.nanmean without the empty-slice warning: no finite values -> NaN."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


# ---------------------------------------------------------------------------
# loading

def load_master(path: str = MASTER) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("rt60", "snr_db", "mic_rolloff", "wer", "mean_conf",
                      "utterance_conf", "n_ref", "n_sub", "n_del", "n_ins", "n_match"):
                v = r.get(k, "")
                r[k] = float(v) if v not in ("", None) else float("nan")
            r["failed"] = str(r.get("failed", "")).strip().lower() in ("true", "1", "yes")
            rows.append(r)
    return rows


def load_refs(path: str = MANIFEST) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["id"]: r["ground_truth"] for r in csv.DictReader(fh)}


def _cells(rows: Sequence[Mapping]) -> set[tuple[str, str]]:
    return {(r["clip_id"], r["condition_name"]) for r in rows}


def discover_arms(rows: Sequence[Mapping]) -> list[str]:
    """Every model id present in the table, sorted.

    The default arm set, because the alternative — a hardcoded pair — makes a
    newly-added arm vanish from the analysis without changing anything a reader
    can see. That is the exact failure this layer exists to characterise in ASR
    models, committed in the code that characterises it.
    """
    return sorted({str(r["model"]) for r in rows if r.get("model") is not None})


def arm_intersection(rows: Sequence[Mapping],
                     models: Sequence[str] | None = None,
                     min_common_clips: int = CORRECTNESS_FLOOR,
                     ) -> dict:
    """
    Split by model, intersect to the cells EVERY arm ran, and return the arms
    together with the census that makes the restriction visible.

    `models=None` discovers the arms from the table. Naming them restricts the
    comparison; naming a model with no rows is an error, not a silent omission.

    ## Why the census, and why it is not optional

    The intersection is taken over all N arms at once, so it shrinks as arms are
    added: nova-3 ran 40 clips, whisper-base the 10-clip AL subset, and a third
    arm run on some other subset can cut the common set again. Every per-arm
    check still passes in that situation — the arms are equal-sized because they
    have been made equal-sized — so the only thing that can tell a reader the
    comparison narrowed is a census that says so. SPEC Appendix C.8 records what
    the alternative costs: D4's unenforced clip matching quietly folded 7.8 WER
    points of clip difficulty into a 19.9-point "simulation gap".

    ## Report-and-proceed, with a floor

    Arms running different clip subsets is the DESIGNED state here (the local
    arms are subset to save wall clock and spend), so refusing it would delete a
    legitimate finding. Restriction plus a loud census is the fix. Three things
    still raise, because none of them is a coverage choice:

      * an arm with no usable rows at all (`RaggedArmsError`) — a wholesale arm
        failure reads as "this model is perfect on zero cells";
      * a duplicated cell within an arm (`RaggedArmsError`, below) — a data
        defect that double-weights one clip and leaves the table's shape intact;
      * an intersection below `min_common_clips` (`ThinOverlapError`). The
        DEFAULT here is the correctness floor (`CORRECTNESS_FLOOR`): an empty
        intersection, which passes every other check because it makes all arms
        equally empty. The stricter statistical floor (`MIN_COMMON_CLIPS`) is
        applied by `model_arms_report`, which is where per-condition means are
        actually computed and quoted — splitting arms and reporting on them are
        different jobs and only the second one needs the adequacy argument.

    Drops `failed` rows first: a failure sentinel is not a prediction, and
    counting one as a low-confidence output would corrupt both the dead-zone rate
    and the confidence shape.
    """
    present = discover_arms(rows)
    if models is None:
        models = present
    models = list(models)
    if not models:
        raise RaggedArmsError(
            "no model arms found in the table (no rows carry a `model` value). "
            "L1 compares arms; there is nothing here to compare.")
    unknown = [m for m in models if m not in present]
    if unknown:
        raise RaggedArmsError(
            f"requested arm(s) {unknown} are not in the table (it holds "
            f"{present}). Refusing rather than comparing the arms that happen "
            f"to be present: a comparison silently missing a requested arm is "
            f"indistinguishable from one where that arm did well.")

    by_model: dict[str, list[dict]] = {m: [] for m in models}
    raw_counts = {m: 0 for m in models}
    failed_counts = {m: 0 for m in models}
    for r in rows:
        m = str(r["model"])
        if m not in by_model:
            continue
        raw_counts[m] += 1
        if r["failed"]:
            failed_counts[m] += 1
        else:
            by_model[m].append(dict(r))

    empty = [m for m, v in by_model.items() if not v]
    if empty:
        raise RaggedArmsError(
            f"no usable rows for {empty}. The whisper arm in particular fails "
            f"wholesale if the model weights could not be fetched -- check for "
            f"CERTIFICATE_VERIFY_FAILED in the `error` column before reading "
            f"anything into this."
        )

    per_arm_cells = {m: _cells(v) for m, v in by_model.items()}
    common = set.intersection(*per_arm_cells.values())
    common_clips = sorted({str(c[0]) for c in common})

    # THE FLOOR. An EMPTY intersection is the case that most needs catching: it
    # makes every arm equally empty, so the per-arm row-count identity and the
    # equal-size check below both PASS, and the failure only surfaces much later
    # as a confusing error in an aggregation step (or, worse, as a report over
    # one or two clips that reads like a measurement).
    if len(common_clips) < max(1, int(min_common_clips)):
        detail = ", ".join(
            f"{m}: {len({str(c[0]) for c in per_arm_cells[m]})} clips"
            for m in models)
        raise ThinOverlapError(
            f"intersecting {len(models)} arms {models} leaves "
            f"{len(common_clips)} common clip(s) over {len(common)} cell(s) "
            f"({detail}) — below the floor of {min_common_clips}. A "
            f"per-condition mean over that few clips describes the utterances "
            f"that happened to overlap, not an acoustic condition. Note the "
            f"intersection SHRINKS as arms are added: two arms sharing the "
            f"10-clip AL subset overlap on 10 clips, a third arm run on a "
            f"different subset can cut it to nothing. Re-run one arm on the "
            f"others' clip set, or restrict the comparison with models=[...].")

    out = {m: [r for r in v if (r["clip_id"], r["condition_name"]) in common]
           for m, v in by_model.items()}

    # ROW-COUNT IDENTITY, per arm. `common` is a SET of cells, so a cell that
    # appears twice in one arm's rows contributes one member to the set and TWO
    # rows to that arm — and if both arms happen to carry one duplicate each the
    # size check below still passes. `condition_table` would then weight the
    # duplicated clip twice in that condition's mean, and the arm would still
    # report the right number of conditions with the right number of clips per
    # condition. Nothing downstream can see it, so assert the identity here.
    for m, rows_m in out.items():
        if len(rows_m) != len(common):
            counts: dict[tuple[str, str], int] = {}
            for r in rows_m:
                k = (r["clip_id"], r["condition_name"])
                counts[k] = counts.get(k, 0) + 1
            dupes = sorted(((k, n) for k, n in counts.items() if n > 1),
                           key=lambda kv: -kv[1])[:3]
            raise RaggedArmsError(
                f"arm {m!r} has {len(rows_m)} rows for {len(common)} common "
                f"(clip_id, condition_name) cells ({len(rows_m) - len(common)} "
                f"extra). Duplicate cells: {dupes}. A repeated cell is averaged "
                f"in twice by condition_table, so its clip carries double weight "
                f"in that condition's WER and confidence — and the table still "
                f"has the right shape. Usual cause: master.csv rebuilt from a "
                f"cache that logged the same cell under two run_ids, or two "
                f"result tables concatenated. Deduplicate before comparing.")

    # Post-condition, stated over N arms rather than as a set of sizes. A SET of
    # sizes cannot distinguish "all arms agree" from "all arms are equally
    # wrong" (SPEC Appendix E.1 defect 3 is exactly that shape); the per-arm
    # identity above is the real guard and this is its restatement.
    bad = {m: len(v) for m, v in out.items() if len(v) != len(common)}
    if bad:
        raise RaggedArmsError(
            f"arms still ragged after intersecting {len(models)} arms: "
            f"{bad} rows against {len(common)} common cells")

    census = {
        "models": models,
        "n_arms": len(models),
        "arms_discovered": present,
        "arms_compared": models,
        "arms_excluded": [m for m in present if m not in models],
        "n_common_cells": len(common),
        "n_common_clips": len(common_clips),
        "common_clips": common_clips,
        "n_common_conditions": len({str(c[1]) for c in common}),
        # True only when EVERY arm ran exactly the common cell set — i.e. nothing
        # was dropped. False is the expected state on this project's own tables.
        "cells_matched": all(len(per_arm_cells[m]) == len(common) for m in models),
        "clips_matched": all(
            {str(c[0]) for c in per_arm_cells[m]} == set(common_clips)
            for m in models),
        "per_arm": {
            m: {
                "n_rows_raw": raw_counts[m],
                "n_failed_excluded": failed_counts[m],
                "n_usable": len(by_model[m]),
                "n_cells": len(per_arm_cells[m]),
                "n_clips": len({str(c[0]) for c in per_arm_cells[m]}),
                "n_conditions": len({str(c[1]) for c in per_arm_cells[m]}),
                "n_rows_kept": len(out[m]),
                "n_rows_dropped_by_intersection": len(by_model[m]) - len(out[m]),
            } for m in models
        },
        "n_rows_dropped_by_intersection": sum(
            len(by_model[m]) - len(out[m]) for m in models),
        "min_common_clips": int(min_common_clips),
        "note": ("arms intersected to the cells EVERY arm ran; failed rows "
                 "dropped first. The intersection shrinks as arms are added, so "
                 "adding a third arm can narrow a comparison that two arms "
                 "supported — which is why the per-arm counts are published "
                 "rather than summarised."),
    }
    return {"arms": out, "census": census}


def matched_arms(rows: Sequence[Mapping],
                 models: Sequence[str] | None = None,
                 min_common_clips: int = CORRECTNESS_FLOOR,
                 ) -> dict[str, list[dict]]:
    """
    `arm_intersection`'s arms only, for callers that do not need the census.

    `models` defaults to EVERY arm in the table. It used to default to
    `(SPINE_MODEL, BASELINE_MODEL)`, which silently dropped any third arm before
    a single number was computed — the report was complete, correct-looking and
    missing a model. Anything that reports numbers to a human should call
    `arm_intersection` and print the census instead.
    """
    return arm_intersection(rows, models, min_common_clips)["arms"]


# ---------------------------------------------------------------------------
# re-scoring under the cross-model normalization

def rescore_cross_model(arms: Mapping[str, Sequence[Mapping]],
                        refs: Mapping[str, str]) -> dict[str, list[dict]]:
    """
    Add `wer_xm` and the cross-model edit counts to every row, leaving the spine
    `wer` untouched so both are available side by side.

    EVERY arm is re-scored, not just the ones expected to move. Correcting one
    side only is how a "fix" becomes a bias: nova-3's output is already
    word-form, so its numbers should barely move -- and if they move a lot, that
    is a signal the normalizer is doing something unintended and should be looked
    at before trusting it. That audit (`normalization_shift`) is per arm and is a
    GATE on any new arm: a vendor whose orthography differs from the reference's
    produces a large, condition-independent WER offset that is mathematically
    indistinguishable from an acoustic effect once it is in the table.
    """
    # A clip with no manifest entry would score against an EMPTY reference, and
    # `classify_errors("", hyp)` returns wer=1.0 with n_ref=0 — a perfect-looking
    # total failure attributed to the model, for a row whose ground truth was
    # simply never loaded. That is indistinguishable from a real acoustic
    # collapse once it is in the table, and it drags the arm's mean toward 1.0.
    # Refuse instead of scoring against nothing.
    missing = sorted({str(r["clip_id"]) for rows in arms.values() for r in rows
                      if not (refs.get(r["clip_id"]) or "").strip()})
    if missing:
        raise ValueError(
            f"no ground-truth reference for clip_id(s) {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''} ({len(missing)} total) — "
            f"scoring them would compare against an EMPTY reference, which "
            f"returns wer=1.0 with n_ref=0 and reads as a total model failure "
            f"rather than a missing manifest row. The manifest "
            f"({len(refs)} ids) and the results table disagree; fix the "
            f"manifest path or the clip ids before comparing arms.")

    out: dict[str, list[dict]] = {}
    for model, rows in arms.items():
        new = []
        for r in rows:
            ref = refs[r["clip_id"]]
            res = cross_model_classify_errors(ref, r.get("transcript", "") or "")
            d = dict(r)
            d["wer_xm"] = float(res["wer"])
            d["n_sub_xm"] = res["counts"]["sub"]
            d["n_del_xm"] = res["counts"]["del"]
            d["n_ins_xm"] = res["counts"]["ins"]
            d["n_ref_xm"] = res["n_ref"]
            new.append(d)
        out[model] = new
    return out


def normalization_shift(arms: Mapping[str, Sequence[Mapping]]) -> dict:
    """
    How much the cross-model normalization moved each arm.

    This is the audit that makes the correction trustworthy. The expectation is
    a large shift for whisper-base and a near-zero shift for nova-3; anything
    else means the normalizer is changing more than orthography.

    THE AUDIT RUNS FOR EVERY ARM, INCLUDING THE EXCLUDED ONES — that is the
    point. `wer_comparable` is carried on each row because this table is where a
    reader decides whether to believe the arm's WER at all, and an arm excluded
    from the cross-model comparison must be visible as excluded exactly here,
    beside the shift that got it excluded. A shift is only subtractable if it is
    the same shift every call; see `model_compare.WER_INCOMPARABLE_ARMS`.
    """
    out = {}
    for model, rows in arms.items():
        strict = np.array([r["wer"] for r in rows], dtype=float)
        xm = np.array([r["wer_xm"] for r in rows], dtype=float)
        out[model] = {
            "wer_strict_mean": float(np.nanmean(strict)),
            "wer_crossmodel_mean": float(np.nanmean(xm)),
            "mean_shift": float(np.nanmean(strict - xm)),
            "n_rows": len(rows),
            "wer_comparable": is_wer_comparable(model),
            "wer_incomparable_reason": WER_INCOMPARABLE_ARMS.get(str(model)),
        }
    return out


# ---------------------------------------------------------------------------
# per-condition aggregation (the unit every comparison is made on)

def condition_table(rows: Sequence[Mapping], wer_key: str = "wer") -> list[dict]:
    """
    Collapse clip-level rows to one row per condition.

    The condition is the unit of analysis throughout the project: a dead zone is
    a property of an acoustic condition, not of one utterance, and averaging over
    the clip set is what makes the WER estimate precise enough to separate
    adjacent cells.

    TWO ACCURACIES, BOTH NAMED — see the module docstring's point 4. There is no
    single "the WER" of a condition once some clips come back EMPTY:

      wer / wer_all_clips  macro mean over EVERY clip. Corpus severity. This is
                           the honest answer to "what does this condition do",
                           and it is NOT comparable to `mean_conf`.
      wer_spoke            macro mean over exactly the clips that returned a
                           confidence — the population `mean_conf` is averaged
                           over, and therefore the only accuracy that may be
                           thresholded against it or subtracted from it.
      n_spoke              that population's size, i.e. mean_conf's denominator.
      n_silent/silent_frac clips that emitted no scorable words (`is_silent_row`,
                           shared with D1). First-class, because it is exactly
                           the quantity that separates the two WERs.
      mute                 True when NO clip produced a confidence. Such a
                           condition has `wer_spoke` NaN and no gap can exist for
                           it; `confidence_flags` holds it out and it is reported
                           as its own category.

    `wer` is kept as an alias of `wer_all_clips` because it is the frozen key the
    divergence scan and the dashboard join on.

    THE PAIRED SUBSET IS DEFINED BY "did this clip contribute a confidence"
    (`isfinite(mean_conf)`), not by `is_silent_row`. That is what the denominator
    of `mean_conf` literally is, and the two can disagree: whisper returns an
    utterance-level confidence with no words on a handful of real rows. Those are
    counted separately as `n_conf_without_words` rather than reconciled away.
    """
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for r in rows:
        groups[r["condition_name"]].append(r)

    # One clip contributes at most one row per condition. A repeat makes that
    # clip count twice in the condition's mean WER and mean confidence — the
    # `n_clips` field still looks plausible (it just reads one higher), so a
    # duplicated hard clip can push a condition into the dead-zone quadrant with
    # nothing anywhere to show for it.
    for name, g in groups.items():
        seen: dict[str, int] = {}
        for r in g:
            cid = str(r.get("clip_id"))
            seen[cid] = seen.get(cid, 0) + 1
        dupes = sorted(((c, n) for c, n in seen.items() if n > 1),
                       key=lambda kv: -kv[1])
        if dupes:
            raise ValueError(
                f"condition {name!r} has {len(g)} rows for {len(seen)} distinct "
                f"clips — duplicate clip_id(s) {dupes[:5]}. Those clips are "
                f"averaged in twice, so this condition's WER and mean_conf are "
                f"weighted by which clip happened to repeat. Deduplicate the "
                f"rows (matched_arms asserts this too; a caller passing raw "
                f"master-table rows bypasses that).")

    out = []
    for name, g in sorted(groups.items()):
        first = g[0]
        wer = np.array([as_float(r[wer_key]) for r in g], dtype=float)
        conf = np.array([as_float(r["mean_conf"]) for r in g], dtype=float)
        # THE PAIRED SUBSET. Every quantity computed under `spoke` is on the same
        # estimand as `mean_conf`; anything computed over the whole group is not
        # and must never be thresholded against a confidence.
        spoke = np.isfinite(conf)
        silent = np.array([is_silent_row(r) for r in g], dtype=bool)
        wer_all = _nanmean(wer)
        out.append({
            "condition_name": name,
            "rt60": first["rt60"], "snr_db": first["snr_db"],
            "noise_type": first["noise_type"], "codec": first["codec"],
            "mic_rolloff": first["mic_rolloff"],
            # --- all-clips estimand (corpus severity) -------------------------
            "wer": wer_all,
            "wer_all_clips": wer_all,
            # --- paired (spoke-only) estimand — the confidence's own population -
            "mean_conf": _nanmean(conf),
            "wer_spoke": _nanmean(wer[spoke]) if spoke.any() else float("nan"),
            "n_spoke": int(spoke.sum()),
            # --- silence accounting -------------------------------------------
            "n_silent": int(silent.sum()),
            "silent_frac": float(silent.mean()) if len(g) else float("nan"),
            "n_conf_without_words": int((spoke & silent).sum()),
            "mute": bool(not spoke.any()),
            "n_clips": len(g),
        })
    return out


# ---------------------------------------------------------------------------
# the Whisper-specific failure mode

_WORD = re.compile(r"[a-z']+")


def hallucination_report(rows: Sequence[Mapping], refs: Mapping[str, str],
                         top_k: int = 5) -> dict:
    """
    Whisper hallucinates fluent sentences under heavy degradation. That is a
    QUALITATIVELY different failure from nova-3's, and merging the two into one
    insertion count would describe neither.

    The signature: hypothesis length far exceeding reference length, with the
    excess tokens absent from the reference entirely. We measure the length ratio
    and the foreign-token fraction, and surface the worst real examples so the
    write-up can quote a transcript rather than assert a tendency.
    """
    worst: list[dict] = []
    ratios: list[float] = []
    foreign_fracs: list[float] = []

    for r in rows:
        ref_toks = _WORD.findall(cross_model_normalize(refs.get(r["clip_id"], "")))
        hyp_toks = _WORD.findall(cross_model_normalize(r.get("transcript", "") or ""))
        if not ref_toks:
            continue
        ratio = len(hyp_toks) / len(ref_toks)
        ratios.append(ratio)
        ref_set = set(ref_toks)
        foreign = [t for t in hyp_toks if t not in ref_set]
        frac = len(foreign) / max(len(hyp_toks), 1)
        foreign_fracs.append(frac)
        worst.append({
            "clip_id": r["clip_id"], "condition_name": r["condition_name"],
            "len_ratio": round(ratio, 2), "foreign_frac": round(frac, 3),
            "n_ref": len(ref_toks), "n_hyp": len(hyp_toks),
            "reference": " ".join(ref_toks),
            "transcript": (r.get("transcript", "") or "")[:400],
        })

    worst.sort(key=lambda d: (d["len_ratio"], d["foreign_frac"]), reverse=True)
    return {
        "median_len_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "p95_len_ratio": float(np.percentile(ratios, 95)) if ratios else float("nan"),
        "frac_rows_over_2x": float(np.mean([r > 2.0 for r in ratios])) if ratios else 0.0,
        "mean_foreign_frac": float(np.mean(foreign_fracs)) if foreign_fracs else float("nan"),
        "examples": worst[:top_k],
    }


def edit_signature(rows: Sequence[Mapping], prefix: str = "n_") -> dict:
    """Edit-type composition as a fraction of reference words, per model.

    The zero-denominator case is refused rather than defaulted to 1. With
    `... or 1` a table holding no reference words at all returns
    `{sub: 0, del: 0, ins: 0}` — which reads as "this model made no errors",
    the exact inversion of the truth ("there was nothing to score").
    """
    if not rows:
        raise ValueError("edit_signature got no rows — an empty edit signature "
                         "prints as a clean 0/0/0 composition, which reads as "
                         "'no errors' rather than 'no data'.")
    n_ref = sum(int(r[f"{prefix}ref"]) for r in rows)
    if n_ref <= 0:
        raise ValueError(
            f"edit_signature: total {prefix}ref over {len(rows)} rows is "
            f"{n_ref}, so there is no denominator. Returning 0/0/0 here would "
            f"read as 'this model destroyed no words'; it means the reference "
            f"word count never made it into the table. Check that the "
            f"{prefix}ref column is populated.")
    return {op: sum(int(r[f"{prefix}{op}"]) for r in rows) / n_ref
            for op in ("sub", "del", "ins")}


# ---------------------------------------------------------------------------
# the corrected per-region dead-zone rate
#
# `model_compare.find_divergence_regions` computes its own dead-zone rates
# internally with the default `wer_key="wer"` and does not expose the argument.
# Its headline `wer_gap` is a comparison of two arms' CORPUS severity and is
# correct as it stands — but the dead-zone rate it carries is the mismatched
# pairing. Rather than fork the harness, we recompute that one field here over
# the SAME bins (model_compare's own `_bins_for`/`_region_rows`, so D1, L1 and
# the divergence scan cannot drift apart) and publish the old value under an
# explicit name beside it.

def _region_dead_zone_rates(cond: Mapping[str, Sequence[dict]],
                            space: FactorSpace, n_bins: int,
                            wer_hi: float, conf_pct_hi: float) -> dict:
    flags = {m: condition_flags(t, wer_hi, conf_pct_hi, "wer_spoke")
             for m, t in cond.items()}
    out: dict[tuple, tuple[dict, dict]] = {}
    for factor in space.names:
        if not any(factor in (t[0] if t else {}) for t in cond.values()):
            continue
        kind, spec = _bins_for(space, factor, n_bins)
        n_slots = (len(spec) - 1) if kind == "continuous" else len(spec)
        for b in range(n_slots):
            rates: dict[str, float] = {}
            mutes: dict[str, int] = {}
            span = None
            for m, table in cond.items():
                idx, span = _region_rows(table, factor, kind, spec, b)
                if len(idx) == 0:
                    rates[m], mutes[m] = float("nan"), 0
                    continue
                rates[m] = float(np.mean(flags[m][idx]))
                mutes[m] = int(sum(1 for i in idx if table[i]["mute"]))
            out[(factor, str(span))] = (rates, mutes)
    return out


def augment_divergence_regions(regions: list[dict],
                               cond: Mapping[str, Sequence[dict]],
                               space: FactorSpace = DEFAULT_FACTOR_SPACE,
                               n_bins: int = 4, wer_hi: float = 0.3,
                               conf_pct_hi: float = 0.6) -> list[dict]:
    """
    Replace each region's `dead_zone_rate_by_model` with the same-subset value
    and keep the all-clips one under `dead_zone_rate_by_model_all_clips_pairing`.

    The canonical name points at the correct quantity (the convention D1 fixed:
    `gap` == `gap_spoke`); the mismatched one survives only under a name that
    says what it is, so the published grid-v1 numbers stay reproducible.

    A REGION CAN CARRY TWO DIFFERENT ARM SETS, and it must say so. `wer_gap` /
    `wer_by_model` come from the cross-model WER scan and therefore cover only
    the WER-comparable arms; the dead-zone RATE is within-model (each arm's own
    threshold against its own confidence percentile) so it covers every arm
    `cond` holds. Both lists are written onto the region — a reader who counts
    the models in one dict and assumes the other matches is the reason.
    """
    lookup = _region_dead_zone_rates(cond, space, n_bins, wer_hi, conf_pct_hi)
    for r in regions:
        key = (r["factor"], str(r["span"]))
        if key not in lookup:
            continue
        rates, mutes = lookup[key]
        r["dead_zone_rate_by_model_all_clips_pairing"] = r["dead_zone_rate_by_model"]
        r["dead_zone_rate_by_model"] = rates
        r["dead_zone_rate_by_model_spoke"] = rates
        r["n_mute_by_model"] = mutes
        # `wer_gap` and `wer_by_model` are deliberately left on the ALL-CLIPS
        # WER: that comparison has no confidence term in it, so corpus severity
        # is the right estimand and restricting it to the spoke subset would
        # silently discount each arm's worst clips.
        r["wer_pairing"] = "all_clips"
        r["wer_arms"] = sorted(r.get("wer_by_model") or {})
        r["dead_zone_rate_arms"] = sorted(rates)
    return regions


# ---------------------------------------------------------------------------
# the report

def dead_zone_overlap(sets: Mapping[str, set]) -> dict:
    """
    Do the arms fail silently in the SAME places? Over N arms, not a pair.

    Three quantities, because with more than two arms one number cannot carry the
    answer:

      `jaccard`      the ALL-ARM Jaccard, |intersection of every arm| /
                     |union of every arm|. For exactly two arms this is
                     bit-identical to the pairwise value the two-arm version
                     published, so the key keeps its meaning and grid-v1 stays
                     reproducible; for three it honestly reports that agreement
                     must hold everywhere, not just between the two arms someone
                     happened to hardcode.
      `pairwise`     every unordered pair, each with its own Jaccard and shared
                     set. This is what names WHICH arms agree — an all-arm
                     Jaccard of 0 cannot distinguish "no two arms agree" from
                     "two agree perfectly and the third is disjoint".
      `<model>_only` per arm: dead zones in that arm and in NO other. With two
                     arms this is the set difference; with three it is the
                     difference against the union of the others, which is the
                     only reading that does not silently ignore an arm.

    A key never holds a pair's number under a global name. That substitution is
    how the previous version reported a two-arm Jaccard for a three-arm study.
    """
    models = list(sets)
    if len(models) < 2:
        raise ValueError(
            f"dead_zone_overlap needs >= 2 arms, got {models}. With one arm the "
            f"overlap is trivially the arm itself and reads as perfect agreement.")

    def _j(a: set, b: set) -> float:
        u = a | b
        return (len(a & b) / len(u)) if u else float("nan")

    inter_all = set.intersection(*(sets[m] for m in models))
    union_all = set.union(*(sets[m] for m in models))
    pairwise: dict[str, dict] = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            pairwise[f"{a}|{b}"] = {
                "models": [a, b],
                "jaccard": _j(sets[a], sets[b]),
                "n_shared": len(sets[a] & sets[b]),
                "shared": sorted(sets[a] & sets[b]),
                "n_union": len(sets[a] | sets[b]),
            }

    out = {
        "models": models,
        "jaccard": (len(inter_all) / len(union_all)) if union_all else float("nan"),
        "jaccard_definition": (
            f"all-arm Jaccard over {len(models)} arms: |dead zones flagged by "
            f"EVERY arm| / |dead zones flagged by ANY arm|. Identical to the "
            f"pairwise value when there are exactly two arms."),
        "shared": sorted(inter_all),          # shared by EVERY arm
        "n_union": len(union_all),
        "pairwise": pairwise,
        "n_by_model": {m: len(sets[m]) for m in models},
    }
    for m in models:
        others = set.union(*(sets[k] for k in models if k != m))
        out[f"{m}_only"] = sorted(sets[m] - others)
    return out


def model_arms_report(master_path: str = MASTER,
                      manifest_path: str = MANIFEST,
                      wer_hi: float = 0.3, conf_pct_hi: float = 0.6,
                      models: Sequence[str] | None = None,
                      min_common_clips: int = MIN_COMMON_CLIPS) -> dict:
    """
    The whole L1 layer over EVERY arm in the table (or the ones `models` names).

    `models=None` discovers the arms. That default is the fix for this module's
    worst two-arm assumption: with a hardcoded pair, adding a third arm to the
    grid changed nothing a reader of this report could see.

    This is the path that computes and quotes per-condition means, so it applies
    the STATISTICAL floor (`MIN_COMMON_CLIPS`), not just `arm_intersection`'s
    correctness floor.
    """
    rows = load_master(master_path)
    refs = load_refs(manifest_path)
    matched = arm_intersection(rows, models, min_common_clips)
    arms, census = matched["arms"], matched["census"]
    if len(arms) < 2:
        raise RaggedArmsError(
            f"L1 compares arms and this table holds {len(arms)}: "
            f"{sorted(arms)}. Run the grid for a second model "
            f"(`--models <a>,<b>`) before running this layer.")
    arms = rescore_cross_model(arms, refs)

    shift = normalization_shift(arms)

    # Within-model analyses use the SPINE wer -- they are scale-free by
    # construction (each model is ranked against its own distribution), so
    # re-scoring would change nothing and would only muddy the provenance.
    cond = {m: condition_table(v, "wer") for m, v in arms.items()}
    cond_xm = {m: condition_table(v, "wer_xm") for m, v in arms.items()}

    per_model = {}
    for m, table in cond.items():
        # SAME-SUBSET flags — "confidently wrong" is a claim about the clips the
        # model actually spoke on. `condition_flags` (D1's) holds MUTE conditions
        # out, which is both required (their wer_spoke is NaN and dead_zone_flags
        # refuses a non-finite WER by design) and safe (a NaN confidence is
        # already excluded from the within-model percentile, so removing those
        # rows leaves every surviving percentile bit-identical).
        flags = condition_flags(table, wer_hi, conf_pct_hi, "wer_spoke")
        flags_all = condition_flags(table, wer_hi, conf_pct_hi, "wer")
        pct = within_model_conf_percentile(table)
        mute = np.array([bool(r["mute"]) for r in table], dtype=bool)
        # silence-driven: flagged ONLY by the mismatched pairing. The model was
        # not wrong enough on the clips it spoke on; the flag came from clips
        # that vanished. Dangerous, but a silence failure, and the fix differs.
        sd = flags_all & ~flags & ~mute
        paired = [r for r in table
                  if np.isfinite(as_float(r["mean_conf"]))
                  and np.isfinite(as_float(r["wer_spoke"]))]
        per_model[m] = {
            "n_conditions": len(table),
            "n_clips_per_condition": table[0]["n_clips"] if table else 0,
            # Every WER key below is this arm's OWN number. It is a legitimate
            # within-model quantity (it is what the dead-zone threshold is
            # applied to) and an illegitimate cross-arm one for an arm whose
            # orthography is unstable — so the flag rides along with them rather
            # than being kept in a header the reader may not have reached.
            "wer_comparable": is_wer_comparable(m),
            "wer_incomparable_reason": WER_INCOMPARABLE_ARMS.get(str(m)),
            "wer_mean_strict": float(np.nanmean([r["wer"] for r in table])),
            "wer_mean_crossmodel": float(np.nanmean(
                [r["wer"] for r in cond_xm[m]])),
            "wer_mean_strict_spoke": _nanmean(
                np.array([r["wer_spoke"] for r in table])),
            # --- the corrected headline ---------------------------------------
            "dead_zone_rate": float(np.mean(flags)),
            "n_dead_zones": int(np.sum(flags)),
            "dead_zones": [table[i]["condition_name"]
                           for i in np.flatnonzero(flags)],
            "conf_percentile_of_dead_zones": [float(pct[i])
                                              for i in np.flatnonzero(flags)],
            # --- and the mismatched pairing, kept so grid-v1 stays reproducible -
            "dead_zone_rate_all_clips_pairing": float(np.mean(flags_all)),
            "n_dead_zones_all_clips_pairing": int(np.sum(flags_all)),
            "dead_zones_all_clips_pairing": [table[i]["condition_name"]
                                             for i in np.flatnonzero(flags_all)],
            # --- the other two categories, never dropped ----------------------
            "n_silence_driven": int(np.sum(sd)),
            "silence_driven": [table[i]["condition_name"]
                               for i in np.flatnonzero(sd)],
            "n_mute_zones": int(mute.sum()),
            "mute_zones": [table[i]["condition_name"]
                           for i in np.flatnonzero(mute)],
            "silence": silence_summary(arms[m]),
            "n_conditions_with_silence": int(
                sum(1 for r in table if r["n_silent"] > 0)),
            "mean_silent_frac": _nanmean(
                np.array([r["silent_frac"] for r in table])),
            # SHAPE on the paired subset (mute rows carry no confidence AND no
            # wer_spoke; leaving them in returns a NaN spearman).
            "shape": confidence_wer_shape(paired, wer_key="wer_spoke"),
            "shape_all_clips_pairing": confidence_wer_shape(table, wer_key="wer"),
            # THE SAME WITHIN-MODEL SHAPE, RE-SCORED. Same rows, same
            # confidences, same estimand — only the ORTHOGRAPHY layer of the
            # scoring changes. This is not a cross-arm comparison (nothing is
            # subtracted between arms) and it is the measurement that decides
            # whether an arm's formatting offset is a BIAS or NOISE:
            #   * a constant offset shifts WER without reordering conditions, so
            #     a rank correlation is invariant to it — the shape must not move;
            #   * a per-call draw is measurement error in the WER variable, which
            #     ATTENUATES a rank correlation toward zero — the shape moves,
            #     and the strict-scored figure is a lower bound on |rho|.
            # So this column is the evidence for (or against) the arm's presence
            # in WER_INCOMPARABLE_ARMS, rather than a policy asserted in a
            # comment. It is computed for every arm, including the ones expected
            # not to move: an arm that was supposed to be stable and is not is
            # exactly what this is for.
            "shape_crossmodel": confidence_wer_shape(
                [r for r in cond_xm[m]
                 if np.isfinite(as_float(r["mean_conf"]))
                 and np.isfinite(as_float(r["wer_spoke"]))],
                wer_key="wer_spoke"),
            "edit_signature_strict": edit_signature(arms[m], "n_"),
            "edit_signature_crossmodel": {
                op: sum(int(r[f"n_{op}_xm"]) for r in arms[m])
                    / (sum(int(r["n_ref_xm"]) for r in arms[m]) or 1)
                for op in ("sub", "del", "ins")},
        }

    # THE TWO CROSS-MODEL WER CALL SITES. `exclude_incomparable=True` is typed
    # here, in the open, and nowhere else: it is the whole enforcement surface.
    # Without it both calls RAISE the moment a WER-incomparable arm is in the
    # table, which is the behaviour we want — a new arm cannot slide into a WER
    # ranking by being added to the grid. The census returned by each call is
    # published verbatim so the exclusion is a line in the report, not a
    # property of the code.
    wer_census = wer_comparability(list(cond))
    divergence = augment_divergence_regions(
        find_divergence_regions(cond, wer_hi=wer_hi, conf_pct_hi=conf_pct_hi,
                                exclude_incomparable=True),
        cond, wer_hi=wer_hi, conf_pct_hi=conf_pct_hi)
    combined = compare_models(cond, wer_hi=wer_hi, conf_pct_hi=conf_pct_hi,
                              exclude_incomparable=True)
    # `compare_models` is model_compare's own harness and takes no wer_key, so
    # every dead-zone number inside it is the MISMATCHED all-clips pairing. It is
    # kept (this module uses the shared harness rather than a private fork) but
    # it is labelled, because an unlabelled `dead_zone_rate` under a second key
    # is exactly how the corrected one gets misquoted.
    combined["pairing"] = (
        "all-clips (mean_conf vs WER over EVERY clip, including the silent ones "
        "that carry no confidence). model_compare.compare_models does not accept "
        "a wer_key; the CORRECTED same-subset rates are per_model[*]"
        "['dead_zone_rate'] above, and the all-clips ones are repeated there as "
        "'dead_zone_rate_all_clips_pairing'.")

    # Dead-zone set overlap: do the arms fail silently in the SAME places?
    sets = {m: set(v["dead_zones"]) for m, v in per_model.items()}
    sets_all = {m: set(v["dead_zones_all_clips_pairing"])
                for m, v in per_model.items()}
    overlap = dead_zone_overlap(sets)
    overlap["jaccard_all_clips_pairing"] = dead_zone_overlap(sets_all)["jaccard"]
    overlap["pairing"] = "same-subset (dead zones flagged on wer_spoke)"

    # Hallucination is measured for EVERY arm. It was computed only for the
    # baseline arm, so a new arm that invents fluent text under degradation --
    # the failure mode WER structurally understates -- would have gone
    # unmeasured while the report still carried a confident-looking
    # hallucination section about a different model.
    halluc = {m: hallucination_report(arms[m], refs) for m in arms}

    out = {
        "arms": list(arms),
        "arm_census": census,
        # WHICH ARMS ARE IN THE WER COMPARISON. Separate from `arms` on purpose:
        # `arms` is the within-model arm set (everything), this is the cross-arm
        # WER arm set (a subset), and a payload that carried only one of them
        # would let a reader assume the wrong one.
        "wer_comparability": wer_census,
        "normalization_shift": shift,
        "per_model": per_model,
        "dead_zone_overlap": overlap,
        "divergence_regions": divergence,
        "compare_models": combined,
        "hallucination_by_model": halluc,
        "category_meaning": CATEGORY_MEANING,
        "params": {"wer_hi": wer_hi, "conf_pct_hi": conf_pct_hi},
    }
    # Legacy alias, emitted ONLY when the arm it is named after is present. A key
    # called `whisper_hallucination` must never hold some other arm's numbers;
    # consumers should read `hallucination_by_model`, which names every arm.
    if BASELINE_MODEL in halluc:
        out["whisper_hallucination"] = halluc[BASELINE_MODEL]
    return out


def _wrap(text: str, width: int = 70) -> list[str]:
    import textwrap
    return textwrap.wrap(" ".join(str(text).split()), width=width) or [""]


def _wer_mark(res: Mapping, model: str) -> str:
    """`‡` beside any WER printed for an arm excluded from the WER comparison."""
    excluded = set((res.get("wer_comparability") or {}).get("excluded") or ())
    return "‡" if str(model) in excluded else " "


def format_report(res: Mapping) -> str:
    L: list[str] = []
    add = L.append
    cen = res.get("arm_census") or {}
    wc = res.get("wer_comparability") or {}
    excluded = list(wc.get("excluded") or ())
    add("L1 — MULTI-MODEL COMPARISON (SPEC R5.7)")
    add("=" * 72)
    add("")
    add(f"Arms compared ({len(res['arms'])}): {', '.join(res['arms'])}")
    add("Matched to the cells EVERY arm ran; failed rows dropped.")
    add("")

    # THE WER-COMPARABILITY BLOCK. Printed before any WER, in the same place
    # every time, whether or not anything is excluded — a block that appears
    # only when there is bad news trains the reader to skip it, and its absence
    # then carries information nobody reads.
    add("-- WER comparability (WHICH ARMS MAY BE RANKED AGAINST EACH OTHER) ----")
    add(f"  cross-model WER comparison IN : "
        f"{', '.join(wc.get('comparable') or res['arms'])}")
    add(f"  cross-model WER comparison OUT: "
        f"{', '.join(excluded) if excluded else '(none)'}")
    for m in excluded:
        reason = (wc.get("reasons") or {}).get(m, "")
        add(f"  WHY {m} is excluded:")
        for line in _wrap(reason, width=70):
            add(f"    {line}")
    if excluded:
        add("  An excluded arm is still measured WITHIN itself — dead-zone rate,")
        add("  confidence-vs-WER shape and L2 calibration are computed against")
        add("  that arm's OWN distribution and never touch a cross-arm WER. Its")
        add("  own WER columns below are marked with a dagger (‡): read them as")
        add("  that arm's internal scale, never as a rank against another arm.")
        add("  This is enforced in code, not by convention: the cross-model WER")
        add("  paths (model_compare.find_divergence_regions / compare_models)")
        add("  RAISE on an incomparable arm unless the caller explicitly passes")
        add("  exclude_incomparable=True. There is no flag that includes them.")
    add("")

    # THE CENSUS. Printed before any number, because every number below is a
    # number over this cell set and the restriction is invisible otherwise.
    add("-- arm census (what the comparison is actually over) -----------------")
    if cen.get("arms_excluded"):
        add(f"  ARMS IN THE TABLE BUT NOT COMPARED: {cen['arms_excluded']}")
    add(f"  common cells {cen.get('n_common_cells', '?')} = "
        f"{cen.get('n_common_clips', '?')} clips x "
        f"{cen.get('n_common_conditions', '?')} conditions"
        + ("   (every arm ran exactly these)" if cen.get("cells_matched")
           else "   (RESTRICTED — arms ran different cell sets)"))
    add(f"{'model':<16}{'raw':>7}{'failed':>8}{'usable':>8}{'clips':>7}"
        f"{'conds':>7}{'kept':>8}{'dropped':>9}")
    for m, d in (cen.get("per_arm") or {}).items():
        add(f"{m:<16}{d['n_rows_raw']:>7}{d['n_failed_excluded']:>8}"
            f"{d['n_usable']:>8}{d['n_clips']:>7}{d['n_conditions']:>7}"
            f"{d['n_rows_kept']:>8}{d['n_rows_dropped_by_intersection']:>9}")
    if not cen.get("clips_matched", True):
        add("  NOTE: the arms ran different CLIP sets, so the intersection above")
        add("  is smaller than any single arm. This is expected (local arms are")
        add("  subset to save wall clock) but every WER below is over the common")
        add("  clips only — it is NOT each arm's corpus-wide number.")
    add("  The intersection shrinks as arms are added: a third arm on a different")
    add("  clip subset narrows a comparison two arms supported.")
    add("")

    add("-- normalization audit ------------------------------------------------")
    add("Cross-model WER re-scores EVERY arm. An arm that is already word-form")
    add("(nova-3) should shift ~0; a large shift there would mean the normalizer")
    add("is changing more than orthography. This is a GATE on any new arm.")
    add("A shift can only be SUBTRACTED if it is the same shift on every call.")
    add(f"{'model':<16}{'WER strict':>12}{'WER x-model':>13}{'shift':>9}{'n':>8}"
        f"  {'WER-comparable':<14}")
    for m, d in res["normalization_shift"].items():
        add(f"{m:<16}{d['wer_strict_mean']:>12.3f}{d['wer_crossmodel_mean']:>13.3f}"
            f"{d['mean_shift']:>9.3f}{d['n_rows']:>8}  "
            f"{'yes' if d.get('wer_comparable', True) else 'NO — EXCLUDED':<14}")
    add("")

    add("-- per-model, per-condition ------------------------------------------")
    add("WERall = macro WER over EVERY clip (corpus severity).  WERsp = over only")
    add("the clips that emitted words, i.e. the population mean_conf is averaged")
    add("over — the ONLY accuracy a confidence may be thresholded against. Dead")
    add("zones are flagged on WERsp; the all-clips pairing is shown to be rejected.")
    if excluded:
        add("A '‡' marks an arm EXCLUDED from the cross-model WER comparison: its WER")
        add("columns are its own internal scale and must not be ranked against another")
        add("arm's. Its dead-zone rate and confidence shape ARE comparable in kind,")
        add("being computed within the arm — with the caveat that the dead-zone")
        add("threshold (wer_hi) is an ABSOLUTE WER, so an arm carrying an orthography")
        add("offset crosses it for non-acoustic reasons too.")
    add(f"{'model':<16}{'conds':>7}{'WERall':>9}{'WERsp':>9}{'WERxm':>9}"
        f"{'deadzone%':>11}{'n_dz':>6}{'n_dz(all)':>10}")
    for m, d in res["per_model"].items():
        k = _wer_mark(res, m)
        add(f"{m:<16}{d['n_conditions']:>7}{d['wer_mean_strict']:>8.3f}{k}"
            f"{d['wer_mean_strict_spoke']:>8.3f}{k}{d['wer_mean_crossmodel']:>8.3f}{k}"
            f"{100 * d['dead_zone_rate']:>10.2f}%{d['n_dead_zones']:>6}"
            f"{d['n_dead_zones_all_clips_pairing']:>10}")
    add("")

    add("-- silence accounting (the quantity that separates the two WERs) ------")
    add("A clip whose transcript comes back EMPTY scores WER 1.0 with 100% deletions")
    add("and carries NO per-word confidence. It inflates WERall while contributing")
    add("nothing to mean_conf, so the two describe different populations.")
    add(f"{'model':<16}{'silent rows':>13}{'rate':>8}{'conds w/ silence':>18}"
        f"{'silence-driven':>16}{'mute':>6}")
    for m, d in res["per_model"].items():
        s = d["silence"]
        add(f"{m:<16}{s['n_silent']:>6}/{s['n_rows']:<6}"
            f"{100 * s['silent_rate']:>7.1f}%"
            f"{d['n_conditions_with_silence']:>10}/{d['n_conditions']:<7}"
            f"{d['n_silence_driven']:>16}{d['n_mute_zones']:>6}")
    add("  silence-driven = flagged ONLY by the all-clips pairing (a silence "
        "failure,")
    add("                   not a confidently-wrong one; different fix).")
    add("  mute           = NO words on ANY clip, so no confidence and no gap can")
    add("                   exist. The worst conditions measured, and invisible to")
    add("                   a confidence-based monitor. Listed, never dropped.")
    for m, d in res["per_model"].items():
        if d["mute_zones"]:
            add(f"  {m} mute: " + ", ".join(d["mute_zones"][:6])
                + (" ..." if len(d["mute_zones"]) > 6 else ""))
        if d["silence_driven"]:
            add(f"  {m} silence-driven: " + ", ".join(d["silence_driven"][:6])
                + (" ..." if len(d["silence_driven"]) > 6 else ""))
    add("")

    add("-- confidence-vs-WER shape (within-model; scales are NOT comparable) --")
    add("  correlated against WERsp (same clips as the confidence); the all-clips")
    add("  pairing is shown beside it — a large disagreement means the apparent")
    add("  self-awareness is really a silence pattern.")
    for m, d in res["per_model"].items():
        s, sa = d["shape"], d["shape_all_clips_pairing"]
        add(f"  {m:<18} spearman={s['spearman']:+.3f}, n={s['n']}"
            f"   [all-clips: {sa['spearman']:+.3f}, n={sa['n']}]")
    add("")
    add("  BIAS OR NOISE? The same shape, same rows, re-scored under the")
    add("  cross-model orthography normalization. A rank correlation is INVARIANT")
    add("  to a constant WER offset (it reorders nothing) and is ATTENUATED by a")
    add("  per-call one (measurement error in the WER variable pulls |rho| toward")
    add("  zero). So an arm that MOVES here has formatting noise, not formatting")
    add("  bias, and its strict-scored rho is a LOWER BOUND on |rho|. This is the")
    add("  evidence behind the WER-comparability verdict at the top — measured,")
    add("  not asserted.")
    add(f"  {'model':<18}{'rho strict':>12}{'rho x-model':>13}{'shift':>9}"
        f"   verdict")
    for m, d in res["per_model"].items():
        s, sx = d["shape"], d.get("shape_crossmodel") or {}
        rx = as_float(sx.get("spearman"))
        shift = rx - s["spearman"] if np.isfinite(rx) else float("nan")
        verdict = ("(n/a)" if not np.isfinite(shift) else
                   "NOISE — |rho| attenuated by formatting" if abs(shift) >= 0.05
                   else "bias only — rank order unaffected")
        add(f"  {m:<18}{s['spearman']:>12.3f}{rx:>13.3f}{shift:>9.3f}   {verdict}")
    add("")

    add("-- edit signature (fraction of reference words) ----------------------")
    if excluded:
        add("  THE EDIT COMPOSITION IS THE SAME MEASUREMENT AS WER, TYPED. It is a")
        add("  cross-arm comparison and it inherits the comparability verdict above:")
        add("  an orthography mismatch scores `405-912-77` against `four zero five`")
        add("  as SUBSTITUTIONS, so a '‡' arm reading high on `sub` is not evidence")
        add("  that it substitutes more. Rows are marked, not suppressed, because the")
        add("  number is real within the arm — it just does not mean what the column")
        add("  header implies when read across rows.")
    add(f"{'model':<16}{'sub':>9}{'del':>9}{'ins':>9}")
    for m, d in res["per_model"].items():
        e, k = d["edit_signature_crossmodel"], _wer_mark(res, m)
        add(f"{m:<16}{e['sub']:>8.3f}{k}{e['del']:>8.3f}{k}{e['ins']:>8.3f}{k}")
    if excluded:
        add(f"  NOT CLAIMABLE from this table: any statement of the form "
            f"'{excluded[0]} substitutes/deletes more than X'.")
    add("")

    ov = res["dead_zone_overlap"]
    add("-- do the arms fail silently in the SAME places? ---------------------")
    add(f"  dead zones flagged by EVERY arm : {len(ov['shared'])}")
    add(f"  all-arm jaccard                 : {ov['jaccard']:.3f}   "
        f"[all-clips pairing: {ov['jaccard_all_clips_pairing']:.3f}]")
    add(f"  ({ov.get('jaccard_definition', '')})")
    for k in sorted(k for k in ov if k.endswith("_only")):
        add(f"  {k:<32}: {len(ov[k])}   (flagged by this arm and NO other)")
    if len(ov.get("models", [])) > 2:
        add("  pairwise — which arms agree with which:")
        for key, d in ov.get("pairwise", {}).items():
            add(f"    {key:<32} jaccard {d['jaccard']:.3f}  "
                f"shared {d['n_shared']}/{d['n_union']}")
    add("")

    add("-- divergence regions (ranked by all-clips WER gap) ------------------")
    add(f"  WER arms: {', '.join(wc.get('comparable') or res['arms'])}"
        + (f"   (EXCLUDED: {', '.join(excluded)})" if excluded else ""))
    add("  A region's dead_zone_rate_by_model covers EVERY arm (within-model);")
    add("  its wer_by_model covers only the WER-comparable ones. The two dicts")
    add("  can therefore have different lengths — see wer_arms /")
    add("  dead_zone_rate_arms on each region.")
    add("  wer_gap/wer_by_model are ALL-CLIPS (corpus severity — no confidence")
    add("  term, so restricting to the spoke subset would discount each arm's")
    add("  worst clips). dead_zone_rate_by_model is the SAME-SUBSET rate.")
    add("  models_absent is printed ONLY when an arm has no rows in the region —")
    add("  'these arms diverge here' means something different when a third arm")
    add("  was never measured in that slice.")
    for d in res["divergence_regions"][:8]:
        hide = {"gap", "detail", "wer_pairing", "dead_zone_rate_by_model_spoke",
                "models_compared",
                # The full census (with its multi-sentence reason) is printed
                # ONCE in the header block above. Repeating it on every region
                # line buries the numbers the block exists to show — it stays in
                # the JSON, where a consumer that copies a single region still
                # gets it.
                "wer_comparability"}
        if not d.get("models_absent"):
            hide.add("models_absent")
        if not d.get("models_excluded_wer_incomparable"):
            hide |= {"models_excluded_wer_incomparable", "wer_arms",
                     "dead_zone_rate_arms"}
        bits = " ".join(f"{k}={v}" for k, v in d.items() if k not in hide)
        add(f"  {bits}")
    add("")

    halluc = res.get("hallucination_by_model") or {}
    add("-- hallucination, PER ARM (a failure mode WER structurally understates)")
    add("  Inventing fluent text is qualitatively different from acoustic")
    add("  confusion, and it is unbounded where WER caps at one error per")
    add("  reference word. Measured for every arm, so a new arm cannot be")
    add("  described by an older arm's number.")
    add(f"{'model':<16}{'median':>9}{'p95':>8}{'>2x ref':>10}{'foreign':>9}")
    for m, h in halluc.items():
        add(f"{m:<16}{h['median_len_ratio']:>9.2f}{h['p95_len_ratio']:>8.2f}"
            f"{100 * h['frac_rows_over_2x']:>9.1f}%{h['mean_foreign_frac']:>9.3f}")
    worst = max(halluc.items(), key=lambda kv: kv[1]["frac_rows_over_2x"],
                default=(None, None))
    if worst[0] is not None and worst[1]["examples"]:
        add("")
        add(f"  worst arm: {worst[0]}")
        for ex in worst[1]["examples"][:3]:
            add("")
            add(f"  [{worst[0]} {ex['clip_id']} @ {ex['condition_name']}]  "
                f"{ex['n_ref']} ref words -> {ex['n_hyp']} hyp words")
            add(f"    REF: {ex['reference']}")
            add(f"    HYP: {ex['transcript']}")
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L1 multi-model comparison")
    ap.add_argument("--master", default=MASTER)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--wer-hi", type=float, default=0.3)
    ap.add_argument("--conf-pct-hi", type=float, default=0.6)
    ap.add_argument("--models", default=None,
                    help="comma-separated arms to compare (default: EVERY arm "
                         "in the table). Naming a subset is a deliberate act "
                         "and is printed in the report's arm census.")
    a = ap.parse_args(argv)

    models = ([m.strip() for m in a.models.split(",") if m.strip()]
              if a.models else None)
    res = model_arms_report(a.master, a.manifest, a.wer_hi, a.conf_pct_hi,
                            models=models)
    txt = format_report(res)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    OUT_TXT.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nwrote {OUT_JSON} and {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
