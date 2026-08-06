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

from deadzone.cross_model_norm import cross_model_classify_errors, cross_model_normalize
from deadzone.model_compare import (
    compare_models, confidence_wer_shape, dead_zone_flags,
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
        out.append({
            "condition_name": name,
            "rt60": first["rt60"], "snr_db": first["snr_db"],
            "noise_type": first["noise_type"], "codec": first["codec"],
            "mic_rolloff": first["mic_rolloff"],
            "wer": float(np.nanmean([r[wer_key] for r in g])),
            "mean_conf": float(np.nanmean([r["mean_conf"] for r in g])),
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
        flags = dead_zone_flags(table, wer_hi=wer_hi, conf_pct_hi=conf_pct_hi)
        pct = within_model_conf_percentile(table)
        per_model[m] = {
            "n_conditions": len(table),
            "n_clips_per_condition": table[0]["n_clips"] if table else 0,
            "wer_mean_strict": float(np.nanmean([r["wer"] for r in table])),
            "wer_mean_crossmodel": float(np.nanmean(
                [r["wer"] for r in cond_xm[m]])),
            "dead_zone_rate": float(np.mean(flags)),
            "n_dead_zones": int(np.sum(flags)),
            "dead_zones": [table[i]["condition_name"]
                           for i in np.flatnonzero(flags)],
            "conf_percentile_of_dead_zones": [float(pct[i])
                                              for i in np.flatnonzero(flags)],
            "shape": confidence_wer_shape(table),
            "edit_signature_strict": edit_signature(arms[m], "n_"),
            "edit_signature_crossmodel": {
                op: sum(int(r[f"n_{op}_xm"]) for r in arms[m])
                    / (sum(int(r["n_ref_xm"]) for r in arms[m]) or 1)
                for op in ("sub", "del", "ins")},
        }

    divergence = find_divergence_regions(cond, wer_hi=wer_hi,
                                         conf_pct_hi=conf_pct_hi)
    combined = compare_models(cond, wer_hi=wer_hi, conf_pct_hi=conf_pct_hi)

    # Dead-zone set overlap: do the two models fail silently in the SAME places?
    sets = {m: set(v["dead_zones"]) for m, v in per_model.items()}
    a, b = SPINE_MODEL, BASELINE_MODEL
    union = sets[a] | sets[b]
    jaccard = (len(sets[a] & sets[b]) / len(union)) if union else float("nan")

    return {
        "arms": list(arms),
        "normalization_shift": shift,
        "per_model": per_model,
        "dead_zone_overlap": {
            "shared": sorted(sets[a] & sets[b]),
            f"{a}_only": sorted(sets[a] - sets[b]),
            f"{b}_only": sorted(sets[b] - sets[a]),
            "jaccard": jaccard,
        },
        "divergence_regions": divergence,
        "compare_models": combined,
        "whisper_hallucination": hallucination_report(arms[BASELINE_MODEL], refs),
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
    add(f"{'model':<16}{'conds':>7}{'WER':>8}{'WERxm':>8}{'deadzone%':>11}{'n_dz':>6}")
    for m, d in res["per_model"].items():
        add(f"{m:<16}{d['n_conditions']:>7}{d['wer_mean_strict']:>8.3f}"
            f"{d['wer_mean_crossmodel']:>8.3f}"
            f"{100 * d['dead_zone_rate']:>10.2f}%{d['n_dead_zones']:>6}")
    add("")

    add("-- confidence-vs-WER shape (within-model; scales are NOT comparable) --")
    for m, d in res["per_model"].items():
        s = d["shape"]
        bits = ", ".join(f"{k}={v:.3f}" for k, v in s.items()
                         if isinstance(v, (int, float)))
        add(f"  {m:<14} {bits}")
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
    add(f"  jaccard           : {ov['jaccard']:.3f}")
    for k in ov:
        if k.endswith("_only"):
            add(f"  {k:<18}: {len(ov[k])}")
    add("")

    add("-- divergence regions (ranked) ---------------------------------------")
    for d in res["divergence_regions"][:8]:
        bits = " ".join(f"{k}={v}" for k, v in d.items()
                        if k not in ("gap", "detail"))
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
