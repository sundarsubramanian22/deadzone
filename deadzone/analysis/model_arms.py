"""
L1 — multi-model comparison on real data (SPEC A.R5.7).

Two arms: **nova-3**, the commercial streaming model that exposes per-word
confidence (the spine), and **whisper-base**, the open baseline that shows we
benchmark rather than depend on one vendor's API.

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

**3. The two arms must cover the same cells.** The Whisper arm was run on the
10-clip AL subset, not all 40. Comparing Whisper-on-10 against nova-3-on-40 would
confound the model with the clip set. This module intersects down to the cells
both arms actually ran, and refuses to proceed if the intersection is ragged.

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
    _bins_for, _region_rows, compare_models, confidence_wer_shape,
    find_divergence_regions, within_model_conf_percentile,
)

MASTER = "results/master.csv"
MANIFEST = "recording_manifest.csv"
OUT_JSON = Path("results/model_arms.json")
OUT_TXT = Path("results/model_arms.txt")

SPINE_MODEL = "nova-3"
BASELINE_MODEL = "whisper-base"


class RaggedArmsError(ValueError):
    """
    Raised when the two arms do not cover the same (clip, condition) cells.

    Loud on purpose. A silent inner join would still produce a comparison table,
    and every number in it would be a mixture of a model effect and a coverage
    effect with no way to separate them afterwards.
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


def matched_arms(rows: Sequence[Mapping],
                 models: Sequence[str] = (SPINE_MODEL, BASELINE_MODEL),
                 ) -> dict[str, list[dict]]:
    """
    Split by model and intersect to the cells every arm actually ran.

    Drops `failed` rows first: a failure sentinel is not a prediction, and
    counting one as a low-confidence output would corrupt both the dead-zone rate
    and the confidence shape.
    """
    by_model: dict[str, list[dict]] = {m: [] for m in models}
    for r in rows:
        if r["model"] in by_model and not r["failed"]:
            by_model[r["model"]].append(dict(r))

    empty = [m for m, v in by_model.items() if not v]
    if empty:
        raise RaggedArmsError(
            f"no usable rows for {empty}. The whisper arm in particular fails "
            f"wholesale if the model weights could not be fetched -- check for "
            f"CERTIFICATE_VERIFY_FAILED in the `error` column before reading "
            f"anything into this."
        )

    common = set.intersection(*(_cells(v) for v in by_model.values()))
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

    sizes = {m: len(v) for m, v in out.items()}
    if len(set(sizes.values())) != 1:
        raise RaggedArmsError(f"arms still ragged after intersection: {sizes}")
    return out


# ---------------------------------------------------------------------------
# re-scoring under the cross-model normalization

def rescore_cross_model(arms: Mapping[str, Sequence[Mapping]],
                        refs: Mapping[str, str]) -> dict[str, list[dict]]:
    """
    Add `wer_xm` and the cross-model edit counts to every row, leaving the spine
    `wer` untouched so both are available side by side.

    Both arms are re-scored, not just Whisper. Correcting one side only is how a
    "fix" becomes a bias: nova-3's output is already word-form, so its numbers
    should barely move -- and if they move a lot, that is a signal the normalizer
    is doing something unintended and should be looked at before trusting it.
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
    return regions


# ---------------------------------------------------------------------------
# the report

def model_arms_report(master_path: str = MASTER,
                      manifest_path: str = MANIFEST,
                      wer_hi: float = 0.3, conf_pct_hi: float = 0.6) -> dict:
    rows = load_master(master_path)
    refs = load_refs(manifest_path)
    arms = matched_arms(rows)
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
            "edit_signature_strict": edit_signature(arms[m], "n_"),
            "edit_signature_crossmodel": {
                op: sum(int(r[f"n_{op}_xm"]) for r in arms[m])
                    / (sum(int(r["n_ref_xm"]) for r in arms[m]) or 1)
                for op in ("sub", "del", "ins")},
        }

    divergence = augment_divergence_regions(
        find_divergence_regions(cond, wer_hi=wer_hi, conf_pct_hi=conf_pct_hi),
        cond, wer_hi=wer_hi, conf_pct_hi=conf_pct_hi)
    combined = compare_models(cond, wer_hi=wer_hi, conf_pct_hi=conf_pct_hi)
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

    # Dead-zone set overlap: do the two models fail silently in the SAME places?
    sets = {m: set(v["dead_zones"]) for m, v in per_model.items()}
    sets_all = {m: set(v["dead_zones_all_clips_pairing"])
                for m, v in per_model.items()}
    a, b = SPINE_MODEL, BASELINE_MODEL

    def _jac(s: dict) -> float:
        union = s[a] | s[b]
        return (len(s[a] & s[b]) / len(union)) if union else float("nan")

    return {
        "arms": list(arms),
        "normalization_shift": shift,
        "per_model": per_model,
        "dead_zone_overlap": {
            "shared": sorted(sets[a] & sets[b]),
            f"{a}_only": sorted(sets[a] - sets[b]),
            f"{b}_only": sorted(sets[b] - sets[a]),
            "jaccard": _jac(sets),
            "jaccard_all_clips_pairing": _jac(sets_all),
            "pairing": "same-subset (dead zones flagged on wer_spoke)",
        },
        "divergence_regions": divergence,
        "compare_models": combined,
        "whisper_hallucination": hallucination_report(arms[BASELINE_MODEL], refs),
        "category_meaning": CATEGORY_MEANING,
        "params": {"wer_hi": wer_hi, "conf_pct_hi": conf_pct_hi},
    }


def format_report(res: Mapping) -> str:
    L: list[str] = []
    add = L.append
    add("L1 — MULTI-MODEL COMPARISON (SPEC R5.7)")
    add("=" * 72)
    add("")
    add("Arms matched to the cells BOTH models ran; failed rows dropped.")
    add("")

    add("-- normalization audit ------------------------------------------------")
    add("Cross-model WER re-scores BOTH arms. nova-3 is already word-form, so its")
    add("shift should be ~0; a large shift there would mean the normalizer is")
    add("changing more than orthography.")
    add(f"{'model':<16}{'WER strict':>12}{'WER x-model':>13}{'shift':>9}{'n':>8}")
    for m, d in res["normalization_shift"].items():
        add(f"{m:<16}{d['wer_strict_mean']:>12.3f}{d['wer_crossmodel_mean']:>13.3f}"
            f"{d['mean_shift']:>9.3f}{d['n_rows']:>8}")
    add("")

    add("-- per-model, per-condition ------------------------------------------")
    add("WERall = macro WER over EVERY clip (corpus severity).  WERsp = over only")
    add("the clips that emitted words, i.e. the population mean_conf is averaged")
    add("over — the ONLY accuracy a confidence may be thresholded against. Dead")
    add("zones are flagged on WERsp; the all-clips pairing is shown to be rejected.")
    add(f"{'model':<16}{'conds':>7}{'WERall':>8}{'WERsp':>8}{'WERxm':>8}"
        f"{'deadzone%':>11}{'n_dz':>6}{'n_dz(all)':>10}")
    for m, d in res["per_model"].items():
        add(f"{m:<16}{d['n_conditions']:>7}{d['wer_mean_strict']:>8.3f}"
            f"{d['wer_mean_strict_spoke']:>8.3f}{d['wer_mean_crossmodel']:>8.3f}"
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
        add(f"  {m:<14} spearman={s['spearman']:+.3f}, n={s['n']}"
            f"   [all-clips: {sa['spearman']:+.3f}, n={sa['n']}]")
    add("")

    add("-- edit signature (fraction of reference words) ----------------------")
    add(f"{'model':<16}{'sub':>8}{'del':>8}{'ins':>8}")
    for m, d in res["per_model"].items():
        e = d["edit_signature_crossmodel"]
        add(f"{m:<16}{e['sub']:>8.3f}{e['del']:>8.3f}{e['ins']:>8.3f}")
    add("")

    ov = res["dead_zone_overlap"]
    add("-- do the two models fail silently in the SAME places? ---------------")
    add(f"  shared dead zones : {len(ov['shared'])}")
    add(f"  jaccard           : {ov['jaccard']:.3f}   "
        f"[all-clips pairing: {ov['jaccard_all_clips_pairing']:.3f}]")
    for k in ov:
        if k.endswith("_only"):
            add(f"  {k:<18}: {len(ov[k])}")
    add("")

    add("-- divergence regions (ranked by all-clips WER gap) ------------------")
    add("  wer_gap/wer_by_model are ALL-CLIPS (corpus severity — no confidence")
    add("  term, so restricting to the spoke subset would discount each arm's")
    add("  worst clips). dead_zone_rate_by_model is the SAME-SUBSET rate.")
    for d in res["divergence_regions"][:8]:
        bits = " ".join(f"{k}={v}" for k, v in d.items()
                        if k not in ("gap", "detail", "wer_pairing",
                                     "dead_zone_rate_by_model_spoke"))
        add(f"  {bits}")
    add("")

    h = res["whisper_hallucination"]
    add("-- whisper hallucination (a DIFFERENT failure mode from nova-3's) ----")
    add(f"  median hyp/ref length ratio : {h['median_len_ratio']:.2f}")
    add(f"  p95 length ratio            : {h['p95_len_ratio']:.2f}")
    add(f"  rows over 2x reference len  : {100 * h['frac_rows_over_2x']:.1f}%")
    add(f"  mean foreign-token fraction : {h['mean_foreign_frac']:.3f}")
    for ex in h["examples"][:3]:
        add("")
        add(f"  [{ex['clip_id']} @ {ex['condition_name']}]  "
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
    a = ap.parse_args(argv)

    res = model_arms_report(a.master, a.manifest, a.wer_hi, a.conf_pct_hi)
    txt = format_report(res)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    OUT_TXT.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nwrote {OUT_JSON} and {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
