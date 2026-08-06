"""
analysis/layers.py — the REAL-DATA runners for the three advanced layers
(SPEC §5 "Advanced layers"; appendix A.R5.7 / A.R5.8 / A.R5.9).

WHY THIS FILE EXISTS. `model_compare.py` (L1), `calibration.py` (L2) and
`paralinguistic.py` (L3) are the *machinery*: pure, synthetic-validated
functions that know nothing about where their inputs came from. What was
missing is the part that is actually hard on real data — getting the master
results table (and, for L3, degraded wavs on disk) into the exact shape that
machinery expects, without committing one of the four mistakes below. Each of
those mistakes produces a clean-looking number with no error message, which is
this project's recurring failure mode (SPEC §7, the trap functions).

THE FOUR TRAPS THIS MODULE EXISTS TO AVOID
──────────────────────────────────────────
1. (L1) CROSS-FAMILY CONFIDENCE IS NOT ONE QUANTITY. Deepgram returns an
   acoustic confidence; `audio_pipeline._parse_whisper_result` surfaces the
   decoder's per-token *softmax probability* (and, when word timestamps are
   unavailable, falls back to the segment proxy `exp(avg_logprob)` — a
   different number again); Vosk returns a Kaldi lattice confidence. Putting
   them on one axis produces a ranking of *scale conventions*. Everything here
   goes through `model_compare.within_model_conf_percentile`, so every
   cross-model claim is invariant to any strictly-increasing rescaling of one
   model's confidences. The report says so explicitly and carries the raw
   scales only as a labelled, non-comparable diagnostic.

2. (L1) WHISPER HALLUCINATES FLUENT SENTENCES UNDER HEAVY NOISE. That is a
   qualitatively different failure from Nova-3's acoustic confusion: the model
   invents well-formed text with nothing behind it, which explodes `n_ins`
   while `n_sub`/`n_del` stay ordinary. Averaged into a WER it reads as "a bit
   worse"; it is not, it is a different mechanism and a different product risk
   (a confident, grammatical, entirely invented transcript). `find_hallucinations`
   surfaces those rows on their own so they can be reported as their own finding
   with an example transcript, not smeared into the mean.

3. (L2) A RANDOM WORD-LEVEL TRAIN/TEST SPLIT LEAKS. Words from the same
   clip+condition share an acoustic realization, a speaker, and a noise crop:
   they are anything but independent. Split them at random and the calibrator
   sees test conditions during training, and reports an ECE improvement that
   evaporates on new audio. So the split here is GROUPED — by condition or by
   clip — and a word-level split is not an option the API offers at all
   (`SPLIT_MODES`); asking for one raises.

4. (L3) THE CLEAN REFERENCE MUST BE THE RAW RECORDING. There is no "clean"
   `Condition` — `apply_condition` always convolves an RIR and always mixes
   noise (SPEC A.R3.2). Taking the "near-clean" grid cell as the paralinguistic
   baseline therefore measures drift *from an already-reverberant clip*, which
   shrinks the measured feature drift toward zero at exactly the low end of the
   sweep, and that is where the half-degradation level is read off. The
   half-level is the number the whole decoupling verdict rests on. So the
   baseline is the raw wav, and passing a sweep file as the baseline raises.

WORD-LEVEL CORRECTNESS LABELS (the fiddly bit of L2, spelled out)
────────────────────────────────────────────────────────────────
`classify_errors` returns `edits` as an ORDERED list of `(op, ref_word,
hyp_word)`. Filter to the edits whose `hyp_word is not None` — i.e. `match`,
`sub`, `ins` — and that filtered sequence IS the hypothesis, word for word, in
order. It therefore aligns 1:1 with the adapter's `word_confidences`. Label
`correct = 1` for `match`, `0` for `sub` and `ins`.

DELETIONS ARE INVISIBLE TO THIS ANALYSIS, permanently and by construction: a
deleted word has no hypothesis token, so the model never emitted a confidence
for it and there is nothing to calibrate. Confidence calibration can only tell
you whether the words the model DID say are trustworthy; it can say nothing
about the words it silently dropped. Under reverb — where deletions dominate
(SPEC §5.2) — that is a real blind spot, so `run_l2_calibration` measures it
(`deletion_blindness`) and states it rather than letting a good ECE imply a
coverage it does not have.

Deps: numpy, scipy/sklearn (via calibration), soundfile (lazily, L3 only).
No API calls, no pandas. Imports only model_compare / calibration /
paralinguistic / audio_pipeline / design.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

# Allow `python deadzone/analysis/layers.py` as well as `import deadzone.analysis.layers`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.audio_pipeline import (                  # noqa: E402
    ConfidenceAlignmentError, align_confidences,
)
from deadzone.calibration import (                     # noqa: E402
    FeatureCalibrator, TemperatureScaler, calibration_report,
    expected_calibration_error,
)
from deadzone.design import DEFAULT_FACTOR_SPACE, FactorSpace          # noqa: E402
from deadzone.model_compare import (                   # noqa: E402
    compare_models, dead_zone_flags, within_model_conf_percentile,
)
from deadzone.paralinguistic import (                  # noqa: E402
    FEATURE_KEYS, compare_degradation_rates, extract_features_from_path,
    feature_drift,
)

__all__ = [
    # shared
    "AlignmentError", "load_master_csv", "usable_rows",
    # L1
    "aggregate_by_condition", "find_hallucinations", "raw_confidence_scales",
    "run_l1_model_comparison", "summarize_l1",
    # L2
    "SPLIT_MODES", "word_records", "grouped_split", "run_l2_calibration",
    "summarize_l2",
    # L3
    "SweepPoint", "sweep_from_dir", "run_l3_decoupling", "summarize_l3",
]

# Confidence must be strictly inside (0, 1) before a logit. Vendors DO return
# exactly 1.0 (Deepgram on short high-SNR words), and calibration._logit would
# then be handed a clipped-at-eps value silently; we clip explicitly and up
# front so the clipping is a documented step of this pipeline, not a side
# effect buried in someone else's helper.
CONF_EPS = 1e-6

# The factor columns, imported not re-declared (same discipline as
# run_experiment.MASTER_COLUMNS). If design.py changes, this follows.
_FACTOR_NAMES = tuple(DEFAULT_FACTOR_SPACE.names)


class AlignmentError(ValueError):
    """Hypothesis-word count != word-confidence count for one row.

    Raised, never patched over. A `zip()` would silently drop the tail and hand
    the calibrator confidences bound to the WRONG words — a label-shuffling bug
    that still fits, still produces an ECE, and is undetectable downstream.
    The usual real cause is normalization: `normalize_text` splits on
    punctuation and merges `_COMPOUNDS`, so one vendor token like "3.5" becomes
    two scored tokens (one confidence) and "wi fi" becomes one (two
    confidences). See `word_records(on_misalign=...)` for the reporting escape
    hatch — which SKIPS whole rows and counts them, and still never truncates.
    """


# ===========================================================================
# SHARED — minimal, self-contained row handling
# ===========================================================================
# Deliberately self-contained: this module is loaded by the L1/L2/L3 runners
# only and must not couple to the other Track-D layers. The coercions below
# tolerate both a CSV round trip (everything is a string) and in-memory rows
# (real types), which is what makes the offline tests exercise the same code
# path the real table does.

def _as_float(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(v, default: int = 0) -> int:
    f = _as_float(v)
    return int(f) if math.isfinite(f) else default


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "t"}


def _as_list(v) -> list:
    """`word_confidences` / `edits` are JSON strings on disk, real lists in
    memory (tests, and any in-process runner). Accept both."""
    if v is None or v == "":
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    try:
        out = json.loads(v)
    except (TypeError, ValueError):
        return []
    return list(out) if isinstance(out, (list, tuple)) else []


def load_master_csv(path: str) -> list[dict]:
    """Read the master results table from CSV as plain dict rows (stdlib only).

    No coercion here beyond what the readers below do lazily — every accessor
    in this module goes through `_as_*`, so a CSV row and an in-memory row are
    interchangeable everywhere.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def usable_rows(rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    """Split rows into (usable, failed).

    TRAP AVOIDED: a `failed=True` row is a MISSING measurement, not a confident
    wrong answer. The runner writes it with NULL measurements, and a null WER
    coerces to NaN while a null confidence sinks to the bottom of the
    within-model percentile — i.e. a failure would masquerade as a loud,
    low-confidence failure and drag every per-condition mean. Failures are
    always split out and counted, never dropped in silence.
    """
    ok, bad = [], []
    for r in rows:
        (bad if _as_bool(r.get("failed")) else ok).append(dict(r))
    return ok, bad


def _split_by_model(rows: Sequence[Mapping]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get("model")), []).append(dict(r))
    return out


def _span_str(span) -> str:
    if isinstance(span, (tuple, list)) and len(span) == 2:
        return f"{float(span[0]):.2f}–{float(span[1]):.2f}"
    return str(span)


# ===========================================================================
# L1 — MULTI-MODEL COMPARISON ON REAL DATA (A.R5.7)
# ===========================================================================

def aggregate_by_condition(rows: Sequence[Mapping]) -> list[dict]:
    """Collapse per-(clip, condition) rows to ONE row per condition.

    WHY AGGREGATE. A dead zone is a property of an acoustic CONDITION, not of a
    single utterance: at 40 clips x ~8.5 words the per-clip WER has a standard
    error of ~10 points (SPEC A.R1.1 does this arithmetic at the condition
    level), so per-clip rows would let one unlucky utterance manufacture a dead
    zone. Averaging over the clip set is what makes the confidence-vs-WER shape
    and the divergence regions readable.

    Keeps the factor coordinates (they are constant within a condition), the
    mean WER and mean confidence, plus the summed edit counts and `n_clips`.
    """
    groups: dict[str, list[Mapping]] = {}
    for r in rows:
        groups.setdefault(str(r.get("condition_name")), []).append(r)

    out: list[dict] = []
    for name, grp in groups.items():
        head = grp[0]
        agg = {"condition_name": name,
               "model": str(head.get("model")),
               "n_clips": len(grp)}
        for f in _FACTOR_NAMES:
            v = head.get(f)
            agg[f] = _as_float(v) if isinstance(v, (int, float)) or _is_numeric(v) else v
        agg["wer"] = float(np.nanmean([_as_float(r.get("wer")) for r in grp]))
        confs = np.array([_as_float(r.get("mean_conf")) for r in grp], dtype=float)
        agg["mean_conf"] = float(np.nanmean(confs)) if np.isfinite(confs).any() \
            else float("nan")
        for c in ("n_ref", "n_sub", "n_del", "n_ins", "n_match"):
            agg[c] = int(sum(_as_int(r.get(c)) for r in grp))
        out.append(agg)
    out.sort(key=lambda r: r["condition_name"])
    return out


def _is_numeric(v) -> bool:
    return math.isfinite(_as_float(v))


def raw_confidence_scales(tables: Mapping[str, Sequence[Mapping]]) -> dict:
    """Per-model raw-confidence range — a DIAGNOSTIC, never a comparison.

    Reported only so the write-up can say, with numbers, *why* the raw scales
    cannot be compared (e.g. "Whisper's token softmax sits at 0.30-0.55 where
    Nova-3's acoustic confidence sits at 0.85-0.99; a raw-threshold rule would
    call Whisper permanently unconfident and miss every one of its dead
    zones"). `scales_overlap=False` is the loudest possible form of that
    warning. Nothing downstream branches on these values.
    """
    per: dict[str, dict] = {}
    for m, rows in tables.items():
        c = np.array([_as_float(r.get("mean_conf")) for r in rows], dtype=float)
        c = c[np.isfinite(c)]
        per[m] = {
            "min": float(c.min()) if c.size else float("nan"),
            "median": float(np.median(c)) if c.size else float("nan"),
            "max": float(c.max()) if c.size else float("nan"),
            "n": int(c.size),
        }
    los = [v["min"] for v in per.values() if math.isfinite(v["min"])]
    his = [v["max"] for v in per.values() if math.isfinite(v["max"])]
    overlap = bool(los and his and max(los) <= min(his))
    return {
        "per_model": per,
        "scales_overlap": overlap,
        "note": ("RAW confidences are NOT comparable across model families "
                 "(Deepgram acoustic confidence vs Whisper token softmax vs "
                 "Vosk lattice confidence). Diagnostic only — every claim in "
                 "this report goes through within_model_conf_percentile."),
    }


def find_hallucinations(rows: Sequence[Mapping], ins_ratio: float = 1.5,
                        min_ins: int = 5, top_k: int = 20) -> dict:
    """Surface fluent-hallucination rows: `n_ins` greatly exceeding `n_ref`.

    Whisper under heavy noise does not degrade like Nova-3 does — it invents a
    grammatical sentence. The signature is an insertion count larger than the
    whole reference (you cannot "confuse" your way to 3x more words than were
    spoken), with substitutions and deletions staying unremarkable. Reported
    SEPARATELY because merging it into a WER (or into the fingerprint layer's
    edit-type composition) would describe two mechanisms as one.

    A row is flagged when `n_ins >= ins_ratio * n_ref` AND `n_ins >= min_ins`
    (the second guard keeps a 2-word reference with 3 insertions from being
    called a hallucination).
    """
    flagged: list[dict] = []
    for r in rows:
        n_ins, n_ref = _as_int(r.get("n_ins")), _as_int(r.get("n_ref"))
        if n_ref <= 0 or n_ins < min_ins or n_ins < ins_ratio * n_ref:
            continue
        flagged.append({
            "clip_id": r.get("clip_id"), "condition_name": r.get("condition_name"),
            "model": r.get("model"),
            "n_ins": n_ins, "n_ref": n_ref, "n_sub": _as_int(r.get("n_sub")),
            "n_del": _as_int(r.get("n_del")),
            "ins_over_ref": n_ins / n_ref,
            "wer": _as_float(r.get("wer")),
            "mean_conf": _as_float(r.get("mean_conf")),
            "transcript": r.get("transcript"),
            **{f: r.get(f) for f in _FACTOR_NAMES},
        })
    flagged.sort(key=lambda d: d["ins_over_ref"], reverse=True)
    return {
        "n_flagged": len(flagged),
        "rate": len(flagged) / len(rows) if len(rows) else float("nan"),
        "examples": flagged[:top_k],
        "criterion": {"ins_ratio": ins_ratio, "min_ins": min_ins},
        "note": ("insertions >> reference length = the model invented fluent "
                 "text, a DIFFERENT failure mode from acoustic confusion; "
                 "report it separately, with an example transcript"),
    }


def run_l1_model_comparison(rows: Sequence[Mapping],
                            space: FactorSpace = DEFAULT_FACTOR_SPACE,
                            n_bins: int = 4, wer_hi: float = 0.3,
                            conf_pct_hi: float = 0.6, top_k: int = 10,
                            aggregate: str = "condition",
                            ins_ratio: float = 1.5,
                            min_ins: int = 5) -> dict:
    """L1 on the real master table: split by `model`, drive `compare_models`.

    Returns per-model dead-zone rate + named dead-zone conditions, the
    confidence-vs-WER SHAPE per model, the RANKED divergence regions naming
    where the models' dead zones actually differ, and the hallucination
    callout (trap 2).

    `aggregate="condition"` (default) averages the clip set per condition
    before comparing — see `aggregate_by_condition`. `aggregate="row"` compares
    raw per-clip rows, which is noisier and offered only for diagnostics.

    EVERY cross-model number here is computed from within-model confidence
    percentiles, so the result is invariant to any strictly-increasing
    rescaling of one model's confidence. Raw scales appear once, labelled
    non-comparable.
    """
    if aggregate not in ("condition", "row"):
        raise ValueError(f"aggregate must be 'condition' or 'row', got {aggregate!r}")

    ok, failed = usable_rows(rows)
    by_model_raw = _split_by_model(ok)
    by_model_failed = _split_by_model(failed)
    if len(by_model_raw) < 2:
        raise ValueError(
            "L1 needs >= 2 models in the table; got "
            f"{sorted(by_model_raw)} (run the grid with both arms first)")

    tables = {m: (aggregate_by_condition(r) if aggregate == "condition" else r)
              for m, r in by_model_raw.items()}

    result = compare_models(tables, space, n_bins, wer_hi, conf_pct_hi)

    # Enrich each model's dead zones with the within-model percentile that made
    # them dead zones, and rank them — this is the table the write-up quotes.
    per_model: dict[str, dict] = {}
    for m, trows in tables.items():
        pct = within_model_conf_percentile(trows)
        flags = dead_zone_flags(trows, wer_hi, conf_pct_hi)
        zones = [{**{k: trows[i].get(k) for k in
                     ("condition_name", "n_clips", *_FACTOR_NAMES)},
                  "wer": _as_float(trows[i].get("wer")),
                  "mean_conf": _as_float(trows[i].get("mean_conf")),
                  "conf_percentile": float(pct[i])}
                 for i in np.where(flags)[0]]
        zones.sort(key=lambda z: z["wer"], reverse=True)
        n_raw = len(by_model_raw[m])
        n_fail = len(by_model_failed.get(m, []))
        per_model[m] = {
            **result["per_model"][m],
            "dead_zone_conditions": zones[:top_k],
            "n_rows_used": n_raw,
            "n_failed_rows": n_fail,
            "failure_rate": n_fail / (n_raw + n_fail) if (n_raw + n_fail) else 0.0,
            "hallucination": find_hallucinations(by_model_raw[m], ins_ratio,
                                                 min_ins, top_k),
        }
        # dead_zone_rows from compare_models is the un-enriched copy; drop it so
        # there is exactly one dead-zone listing per model in the payload.
        per_model[m].pop("dead_zone_rows", None)

    regions = result["divergence_regions"][:top_k]
    report = {
        "layer": "L1",
        "models": list(tables.keys()),
        "aggregate": aggregate,
        "thresholds": {"wer_hi": wer_hi, "conf_pct_hi": conf_pct_hi,
                       "n_bins": n_bins},
        "per_model": per_model,
        "divergence_regions": regions,
        "raw_confidence_scales": raw_confidence_scales(tables),
        # The flag a reader (or a dashboard) should branch on. There is no code
        # path in this module that compares raw confidences across models.
        "confidence_comparable_across_models": False,
        "notes": [
            "cross-model confidence claims use within_model_conf_percentile "
            "only; results are invariant to a monotone rescaling of any one "
            "model's confidence",
            "Whisper's per-word 'confidence' is the decoder token softmax "
            "probability (audio_pipeline._parse_whisper_result), falling back "
            "to the segment proxy exp(avg_logprob) when word timestamps are "
            "unavailable — not an acoustic confidence like Deepgram's",
            "hallucination rows (n_ins >> n_ref) are a separate failure mode "
            "and are reported separately from dead zones",
        ],
    }
    report["statement"] = summarize_l1(report)
    return report


def summarize_l1(report: Mapping) -> str:
    """The A.R5.7 DoD sentence, filled in from real numbers."""
    regions = report.get("divergence_regions") or []
    if not regions:
        return "L1: no divergence region found (models agree across the grid)."
    top = regions[0]
    worse, better = top["worse_model"], top["better_model"]
    dz = top["dead_zone_rate_by_model"]
    parts = [
        f"In the {top['factor']} {_span_str(top['span'])} region, "
        f"{worse}'s dead-zone rate is {dz.get(worse, float('nan')):.2f} vs "
        f"{better}'s {dz.get(better, float('nan')):.2f} "
        f"(mean WER {top['wer_by_model'][worse]:.2f} vs "
        f"{top['wer_by_model'][better]:.2f}); the models disagree most at "
        f"{top['factor']}={_span_str(top['span'])}."
    ]
    halluc = [(m, d["hallucination"]) for m, d in report["per_model"].items()
              if d["hallucination"]["n_flagged"]]
    if halluc:
        worst = max(halluc, key=lambda kv: kv[1]["rate"])
        parts.append(
            f"Separately, {worst[0]} hallucinated fluent text on "
            f"{worst[1]['rate']:.1%} of rows (insertions >= "
            f"{worst[1]['criterion']['ins_ratio']}x the reference) — a "
            f"different failure mode, reported on its own.")
    return " ".join(parts)


# ===========================================================================
# L2 — LEARNED CONFIDENCE CALIBRATION ON REAL DATA (A.R5.8)
# ===========================================================================

# The ONLY split modes. A word-level / random split is not offered — see trap 3.
SPLIT_MODES: tuple[str, ...] = ("condition", "clip")


def word_records(rows: Sequence[Mapping], on_misalign: str = "raise") -> dict:
    """Explode master-table rows into WORD-level (features, confidence, label).

    THE RECIPE (A.R5.8), implemented exactly:
      * `edits` is the ordered `(op, ref_word, hyp_word)` backtrace;
      * filter to `hyp_word is not None` (match / sub / ins) — that filtered
        sequence IS the hypothesis, in order, so it aligns 1:1 with
        `word_confidences`;
      * `correct = 1` for `match`, `0` for `sub` and `ins`.

    ALIGNMENT IS ASSERTED, NOT ASSUMED. If the two lengths disagree we raise
    `AlignmentError` (or, with `on_misalign="skip"`, drop the WHOLE row and
    count it). Never `zip()`: that silently truncates to the shorter list and
    binds confidences to the wrong words, which still trains, still scores, and
    is wrong in a way nothing downstream can see.

    DELETIONS ARE NOT REPRESENTED, and cannot be: no hypothesis word means no
    confidence. Their count is returned so the caller can state the blind spot.
    """
    if on_misalign not in ("raise", "skip"):
        raise ValueError("on_misalign must be 'raise' or 'skip'")

    records: list[dict] = []
    conf: list[float] = []
    correct: list[int] = []
    misaligned: list[dict] = []
    n_del = n_ref = 0
    n_rows_used = 0

    n_realigned = 0
    for r in rows:
        edits = _as_list(r.get("edits"))
        confs = [float(c) for c in _as_list(r.get("word_confidences"))]
        hyp_edits = [e for e in edits if len(e) >= 3 and e[2] is not None]
        row_del = sum(1 for e in edits if len(e) >= 3 and e[0] == "del")

        # `confs` came back aligned to the RAW transcript tokens; `hyp_edits` are
        # NORMALIZED tokens. Those agree in length only while normalization
        # happens to preserve token count, which it does not (a hyphen splits,
        # an orthographic compound merges, bare punctuation vanishes). Re-align
        # through the SAME transformation the text went through instead of
        # dropping the row. Measured on the real grid: recovers 1.75% of rows.
        if len(hyp_edits) != len(confs):
            transcript = r.get("transcript")
            if transcript:
                try:
                    confs = align_confidences(transcript, confs)
                    n_realigned += 1
                except ConfidenceAlignmentError:
                    pass          # fall through to the raise/skip path below

        if len(hyp_edits) != len(confs):
            info = {"clip_id": r.get("clip_id"),
                    "condition_name": r.get("condition_name"),
                    "model": r.get("model"),
                    "n_hyp_words": len(hyp_edits), "n_confidences": len(confs)}
            if on_misalign == "raise":
                raise AlignmentError(
                    f"{len(hyp_edits)} hypothesis words but {len(confs)} word "
                    f"confidences for clip={info['clip_id']!r} "
                    f"condition={info['condition_name']!r}. Refusing to zip "
                    "(that would bind confidences to the wrong words). Usual "
                    "cause: normalize_text split or merged a vendor token — "
                    "fix the adapter/normalization pairing, or pass "
                    "on_misalign='skip' to drop and count these rows.")
            misaligned.append(info)
            continue

        n_rows_used += 1
        n_del += row_del
        n_ref += _as_int(r.get("n_ref"))
        base = {f: (_as_float(r.get(f)) if _is_numeric(r.get(f)) else r.get(f))
                for f in _FACTOR_NAMES}
        for i, ((op, ref_w, hyp_w), c) in enumerate(zip(hyp_edits, confs)):
            records.append({**base,
                            "clip_id": r.get("clip_id"),
                            "condition_name": r.get("condition_name"),
                            "model": r.get("model"),
                            "word_index": i, "op": op,
                            "ref_word": ref_w, "hyp_word": hyp_w})
            conf.append(c)
            correct.append(1 if op == "match" else 0)

    # Trap: vendors return exactly 1.0, and calibration._logit assumes strictly
    # (0, 1). Clip HERE so it is an explicit, reported step of this pipeline.
    conf_arr = np.clip(np.asarray(conf, dtype=float), CONF_EPS, 1.0 - CONF_EPS)
    n_clipped = int(np.sum((np.asarray(conf, dtype=float) <= 0.0) |
                           (np.asarray(conf, dtype=float) >= 1.0))) if conf else 0

    return {
        "records": records,
        "conf": conf_arr,
        "correct": np.asarray(correct, dtype=int),
        "n_words": len(records),
        "n_rows_used": n_rows_used,
        "n_misaligned_rows": len(misaligned),
        "n_realigned_rows": n_realigned,
        "misaligned": misaligned,
        "n_conf_clipped": n_clipped,
        "n_deletions": n_del,
        "n_ref_words": n_ref,
    }


def grouped_split(records: Sequence[Mapping], split_by: str = "condition",
                  frac: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indices split over whole GROUPS, never over words.

    `split_by="condition"` holds out entire acoustic conditions (the strongest
    test: can the calibrator generalize to a condition it never saw?);
    `split_by="clip"` holds out entire utterances. Both keep every word of a
    correlated unit on one side of the split.

    Anything else — "word", "random" — raises. That is deliberate: the leaky
    option must not be reachable by typo, because its symptom is a *better*
    ECE, which no one investigates.
    """
    if split_by not in SPLIT_MODES:
        raise ValueError(
            f"split_by must be one of {SPLIT_MODES}, got {split_by!r}. A "
            "random/word-level split is NOT offered: words from the same "
            "clip+condition are highly correlated, so it leaks the test "
            "conditions into training and reports a fake ECE improvement.")

    key = "condition_name" if split_by == "condition" else "clip_id"
    groups = np.array([str(r.get(key)) for r in records])
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError(
            f"need >= 2 distinct {key} values to split by {split_by!r}; "
            f"got {len(uniq)}")
    order = np.random.default_rng(seed).permutation(len(uniq))
    k = max(1, min(len(uniq) - 1, int(round(len(uniq) * frac))))
    train_groups = set(uniq[order[:k]])
    is_train = np.array([g in train_groups for g in groups])
    return np.where(is_train)[0], np.where(~is_train)[0]


def run_l2_calibration(rows: Sequence[Mapping],
                       space: FactorSpace = DEFAULT_FACTOR_SPACE,
                       split_by: str = "condition", frac: float = 0.5,
                       seed: int = 0, n_bins: int = 15,
                       on_misalign: str = "raise",
                       discount_at: Mapping[str, float] | None = None) -> dict:
    """L2 on the real master table: word-level labels -> calibrators -> ECE.

    Fits `TemperatureScaler` (the global baseline) and `FeatureCalibrator` (the
    condition-conditioned model) on a GROUPED train split and evaluates both on
    the held-out groups. Returns ECE before/after for each plus
    reliability-diagram data from `calibration_report`.

    `discount_at` (default `{"rt60": 0.7}`) asks the fitted calibrator the
    plain-language question the write-up needs: above that factor value, by how
    much must reported confidence be discounted to become a real probability?

    Single-model input only — confidence is comparable within a model, never
    across (trap 1). Pass one model's rows; call it twice for two arms.
    """
    ok, failed = usable_rows(rows)
    models = sorted({str(r.get("model")) for r in ok})
    if len(models) > 1:
        raise ValueError(
            f"run_l2_calibration takes ONE model's rows (got {models}). "
            "Confidence is only comparable within a model, and a calibrator "
            "fitted across families would learn the scale difference, not the "
            "miscalibration.")

    w = word_records(ok, on_misalign=on_misalign)
    if w["n_words"] < 2:
        raise ValueError("no word-level records to calibrate "
                         f"(n_words={w['n_words']}, "
                         f"misaligned rows={w['n_misaligned_rows']})")

    recs, conf, y = w["records"], w["conf"], w["correct"]
    tr, te = grouped_split(recs, split_by=split_by, frac=frac, seed=seed)
    if len(te) == 0 or len(tr) == 0:
        raise ValueError("grouped split produced an empty side; "
                         "widen the grid or change `frac`")

    take = lambda idx: [recs[i] for i in idx]      # noqa: E731

    ece_raw = expected_calibration_error(conf[te], y[te], n_bins)

    temp = TemperatureScaler().fit(conf[tr], y[tr])
    conf_temp = temp.transform(conf[te])
    rep_temp = calibration_report(conf[te], conf_temp, y[te], n_bins)

    feat = FeatureCalibrator(space).fit(take(tr), conf[tr], y[tr])
    conf_feat = feat.transform(take(te), conf[te])
    rep_feat = calibration_report(conf[te], conf_feat, y[te], n_bins)

    # What the calibrator learned, as a number a practitioner can act on.
    discount_at = dict(discount_at or {"rt60": 0.7})
    discounts: dict[str, dict] = {}
    for fname, thresh in discount_at.items():
        vals = np.array([_as_float(recs[i].get(fname)) for i in te], dtype=float)
        m = np.isfinite(vals) & (vals >= float(thresh))
        if m.sum() < 10:
            continue
        discounts[fname] = {
            "threshold": float(thresh), "n_words": int(m.sum()),
            "mean_raw_conf": float(np.mean(conf[te][m])),
            "mean_calibrated_conf": float(np.mean(conf_feat[m])),
            "mean_discount": float(np.mean(conf[te][m] - conf_feat[m])),
            "empirical_accuracy": float(np.mean(y[te][m])),
        }

    report = {
        "layer": "L2",
        "model": models[0] if models else None,
        "split_by": split_by, "split_modes_offered": SPLIT_MODES,
        "n_words": w["n_words"], "n_train_words": int(len(tr)),
        "n_test_words": int(len(te)),
        "n_groups": int(len({str(r.get("condition_name" if split_by == "condition"
                                       else "clip_id")) for r in recs})),
        "n_rows_used": w["n_rows_used"], "n_failed_rows": len(failed),
        "n_misaligned_rows": w["n_misaligned_rows"],
        "misaligned": w["misaligned"],
        "n_conf_clipped": w["n_conf_clipped"],
        "ece_raw": ece_raw,
        "temperature": {"T": float(temp.T), "ece": rep_temp["ece_after"],
                        "report": rep_temp},
        "feature": {"ece": rep_feat["ece_after"], "report": rep_feat},
        "ece_reduction_temperature": ece_raw - rep_temp["ece_after"],
        "ece_reduction_feature": ece_raw - rep_feat["ece_after"],
        # DELETIONS ARE INVISIBLE HERE — stated, measured, not hidden.
        "deletion_blindness": {
            "n_deletions": w["n_deletions"],
            "n_ref_words": w["n_ref_words"],
            "deleted_fraction_of_reference":
                (w["n_deletions"] / w["n_ref_words"]) if w["n_ref_words"] else float("nan"),
            "note": ("deleted reference words have no hypothesis token and "
                     "therefore no confidence: confidence calibration cannot "
                     "see them at all. It tells you whether the words the "
                     "model DID emit are trustworthy, not whether it silently "
                     "dropped others — which is exactly the reverb failure "
                     "mode (SPEC §5.2)."),
        },
        "discount": discounts,
        "notes": [
            "vendor confidence is not a calibrated probability by "
            "construction — that is the premise of this layer, not a flaw",
            f"confidences clipped to [{CONF_EPS}, {1 - CONF_EPS}] before any "
            "logit (vendors return exactly 1.0)",
            "train/test split is GROUPED; a random word-level split leaks and "
            "reports a fake improvement, so it is not offered",
        ],
    }
    report["statement"] = summarize_l2(report)
    return report


def summarize_l2(report: Mapping) -> str:
    """The A.R5.8 DoD sentence, filled in from real numbers."""
    parts = [
        f"On held-out {report['split_by']}s ({report['n_test_words']} words), "
        f"raw confidence has ECE {report['ece_raw']:.3f}; temperature scaling "
        f"(T={report['temperature']['T']:.2f}) gives "
        f"{report['temperature']['ece']:.3f} and the feature-conditioned "
        f"calibrator {report['feature']['ece']:.3f}."
    ]
    for fname, d in (report.get("discount") or {}).items():
        parts.append(
            f"Above {fname} = {d['threshold']:g}, reported confidence must be "
            f"discounted by ~{d['mean_discount']:.2f} to become a calibrated "
            f"probability ({d['mean_raw_conf']:.2f} reported vs "
            f"{d['empirical_accuracy']:.2f} observed accuracy, "
            f"n={d['n_words']} words).")
    db = report["deletion_blindness"]
    parts.append(
        f"Blind spot: {db['n_deletions']} deleted reference words "
        f"({db['deleted_fraction_of_reference']:.1%} of the reference) carry "
        f"no confidence and are invisible to this analysis.")
    return " ".join(parts)


# ===========================================================================
# L3 — PARALINGUISTIC DECOUPLING ON REAL AUDIO (A.R5.9)
# ===========================================================================
# The only layer that needs DEGRADED AUDIO ON DISK, not just the table: the
# features are extracted from the waveform, so there is nothing in the master
# table to compute them from. Feed it a sweep directory written by
# `run_experiment.py --save-audio` (one factor over ~8 levels, everything else
# held fixed) plus the WER measured at each of those same levels.

# Features whose drift is worth reporting. `voiced_frac` and `rms` are the
# energy/voicing side, `f0` the pitch side, the rest spectral shape — i.e. the
# stream a paralinguistics-driven agent (emotion, endpointing, barge-in) rides
# on.
DEFAULT_L3_FEATURES: tuple[str, ...] = ("f0", "rms", "voiced_frac", "flatness",
                                        "centroid", "rolloff")

_LEVEL_RE = r"([0-9]+(?:\.[0-9]+)?)"


@dataclass(frozen=True)
class SweepPoint:
    """One rung of a single-factor sweep: the degraded wav + the WER measured
    on that exact same audio."""
    level: float
    audio_path: str
    wer: float


def sweep_from_dir(sweep_dir: str, factor: str,
                   wers: Mapping[float, float] | Sequence[float],
                   pattern: str | None = None) -> list[SweepPoint]:
    """Build the sweep from a directory of degraded wavs named `<factor>_<level>.wav`.

    `wers` is either a mapping level -> WER, or a sequence of WERs in ascending
    level order. Returns points sorted by level.

    The WER must be the one measured on THESE clips (same clip subset, same
    condition), not a nearby grid cell — the whole comparison is two curves
    read off the same x-axis.
    """
    pat = pattern or f"{factor}_*.wav"
    paths = sorted(glob.glob(os.path.join(sweep_dir, pat)))
    if not paths:
        raise FileNotFoundError(f"no sweep wavs matching {pat!r} in {sweep_dir!r}")

    levels: list[float] = []
    for p in paths:
        m = re.search(rf"{re.escape(factor)}[_-]{_LEVEL_RE}", os.path.basename(p))
        if not m:
            raise ValueError(
                f"cannot parse a {factor!r} level out of {os.path.basename(p)!r}; "
                f"expected '<factor>_<level>.wav'")
        levels.append(float(m.group(1)))

    order = np.argsort(levels)
    paths = [paths[i] for i in order]
    levels = [levels[i] for i in order]

    if isinstance(wers, Mapping):
        lut = {float(k): float(v) for k, v in wers.items()}
        try:
            wer_list = [lut[lv] for lv in levels]
        except KeyError as e:
            raise KeyError(f"no WER supplied for {factor}={e.args[0]} "
                           f"(have {sorted(lut)})") from None
    else:
        wer_list = [float(v) for v in wers]
        if len(wer_list) != len(levels):
            raise ValueError(f"{len(wer_list)} WERs for {len(levels)} sweep "
                             "levels — they must line up one-to-one")

    return [SweepPoint(lv, p, w) for lv, p, w in zip(levels, paths, wer_list)]


def _check_clean_reference(clean_raw_path: str, sweep: Sequence[SweepPoint],
                           allow_same_dir: bool) -> None:
    """TRAP 4. The baseline must be the RAW recording.

    There is no clean `Condition`: `apply_condition` always applies an RIR and
    always mixes noise. Using the mildest sweep rung as the baseline measures
    drift from an already-reverberant clip, which collapses the low end of the
    feature curve and moves the half-degradation level — the exact number the
    verdict is read off. So: the baseline may not be one of the sweep files,
    and by default may not even live in the sweep directory.
    """
    clean = os.path.realpath(clean_raw_path)
    sweep_real = {os.path.realpath(p.audio_path) for p in sweep}
    if clean in sweep_real:
        raise ValueError(
            f"clean reference {clean_raw_path!r} is one of the sweep clips. The "
            "baseline must be the RAW recording: every composed condition "
            "already has an RIR and mixed noise on it, so drift measured "
            "against one understates the low end of the curve and shifts the "
            "half-degradation level the verdict depends on.")
    if not allow_same_dir:
        sweep_dirs = {os.path.dirname(p) for p in sweep_real}
        if os.path.dirname(clean) in sweep_dirs:
            raise ValueError(
                f"clean reference {clean_raw_path!r} lives in the sweep "
                "directory. Point this at data/recordings/<id>.wav (the raw "
                "capture), or pass allow_same_dir=True if you have genuinely "
                "staged the raw file alongside the degraded ones.")


def run_l3_decoupling(clean_raw_path: str, sweep: Sequence[SweepPoint],
                      factor: str, feature_keys: Sequence[str] = DEFAULT_L3_FEATURES,
                      gap_tol: float = 0.2, primary_key: str | None = None,
                      allow_same_dir: bool = False) -> dict:
    """L3 on real audio: do paralinguistic features and WER degrade together?

    Extracts features from the RAW clean recording and from each degraded sweep
    clip, takes each feature's drift curve (`feature_drift`) and compares it to
    the lexical curve (WER at the same levels) with
    `compare_degradation_rates`.

    Per feature you get spearman, `max_abs_gap`, both half-degradation levels,
    which curve LEADS (degrades first), and a plain-language reading. The
    headline is the `primary_key` feature (default: the one that decouples
    most, i.e. the largest half-level separation) because that is the strongest
    claim the sweep supports.
    """
    if len(sweep) < 3:
        raise ValueError(f"need >= 3 sweep levels to read a curve, got {len(sweep)}")
    sweep = sorted(sweep, key=lambda p: p.level)
    _check_clean_reference(clean_raw_path, sweep, allow_same_dir)

    levels = np.array([p.level for p in sweep], dtype=float)
    wer_curve = np.array([float(p.wer) for p in sweep], dtype=float)

    unknown = [k for k in feature_keys if k not in FEATURE_KEYS]
    if unknown:
        raise ValueError(f"unknown feature keys {unknown} "
                         f"(available: {list(FEATURE_KEYS)})")

    clean_feats = extract_features_from_path(clean_raw_path)
    deg_feats = [extract_features_from_path(p.audio_path) for p in sweep]

    per_feature: dict[str, dict] = {}
    for key in feature_keys:
        drift = feature_drift(clean_feats, deg_feats, key=key)
        verdict = compare_degradation_rates(levels, drift, wer_curve,
                                            gap_tol=gap_tol)
        verdict["drift_curve"] = [float(v) for v in drift]
        verdict["clean_value"] = float(clean_feats["summary"][key])
        verdict["statement"] = _l3_statement(factor, key, verdict)
        per_feature[key] = verdict

    if primary_key is None:
        primary_key = max(per_feature,
                          key=lambda k: abs(per_feature[k]["feature_half_level"]
                                            - per_feature[k]["lexical_half_level"]))
    headline = per_feature[primary_key]
    n_decoupled = sum(1 for v in per_feature.values() if not v["coupled"])

    return {
        "layer": "L3",
        "factor": factor,
        "levels": [float(v) for v in levels],
        "wer_curve": [float(v) for v in wer_curve],
        "clean_reference": clean_raw_path,
        "sweep_paths": [p.audio_path for p in sweep],
        "per_feature": per_feature,
        "primary_feature": primary_key,
        "headline": headline,
        "verdict": "DECOUPLED" if not headline["coupled"] else "COUPLED",
        "n_features_decoupled": n_decoupled,
        "n_features": len(per_feature),
        "statement": headline["statement"],
        "notes": [
            "clean baseline is the RAW recording, not a near-clean composed "
            "condition (which already carries an RIR and mixed noise)",
            "features come from the DEGRADED audio on disk — this is the only "
            "layer that cannot be computed from the master table alone",
            "coupling is a valid result and is reported as such",
        ],
    }


def _l3_statement(factor: str, key: str, v: Mapping) -> str:
    """Turn one comparison into the sentence the write-up quotes (A.R5.9 DoD)."""
    f_half, l_half = v["feature_half_level"], v["lexical_half_level"]
    if v["coupled"]:
        return (f"Under increasing {factor}, the {key} feature and WER degrade "
                f"together (max normalized gap {v['max_abs_gap']:.2f}, spearman "
                f"{v['spearman']:.2f}; half-degradation at {factor}≈"
                f"{f_half:.2f} vs {l_half:.2f}): COUPLED, so a paralinguistic "
                f"monitor is a usable early warning for lexical failure.")
    if v["leads"] == "lexical":
        return (f"Under increasing {factor}, the {key} feature holds to "
                f"{factor}≈{f_half:.2f} while WER halves at "
                f"≈{l_half:.2f}: lexical accuracy leads, so a "
                f"paralinguistics-driven agent would not notice its ASR had "
                f"already failed.")
    return (f"Under increasing {factor}, the {key} feature collapses at "
            f"{factor}≈{f_half:.2f} while WER only halves at "
            f"≈{l_half:.2f}: the paralinguistic stream leads, so a "
            f"feature-based monitor would alarm before the transcript "
            f"measurably degrades (max normalized gap {v['max_abs_gap']:.2f}).")


def summarize_l3(report: Mapping) -> str:
    """One line per swept factor, plus how many features decoupled."""
    return (f"{report['statement']} "
            f"({report['n_features_decoupled']}/{report['n_features']} features "
            f"decoupled from WER over this {report['factor']} sweep.)")
