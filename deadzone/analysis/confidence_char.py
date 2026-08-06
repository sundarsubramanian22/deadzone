"""
analysis/confidence_char.py — WHAT IS THE CONFIDENCE SCORE? An empirical
characterisation of the number this whole project is built on.

WHY THIS FILE EXISTS. D1 (`confidence_gap.py`) is the headline: it maps where a
model is confident and wrong. L2 (`calibration.py` / `calibration_report.py`)
asks whether that confidence can be turned into a probability. Neither ever
describes the SIGNAL ITSELF. The write-up says only what it is NOT — "not a
calibrated probability by construction" — and a reader is entitled to ask the
obvious follow-up: *then what is it?* This module answers that empirically, from
10,560 measured rows, without speculating about any vendor's decoder internals.

Four questions, in the order a sceptic asks them:

  1. WHAT DOES IT LOOK LIKE WHEN NOTHING IS WRONG? Every dead-zone confidence in
     the project is quoted as a bare number. "0.829" means nothing until you know
     that the same model sits at 0.962 on the mildest grid cell. The clean
     reference distribution is the ruler; without it there is no scale. It is
     selected BY FACTOR SETTINGS, never by WER — see `severity_rank`.

  2. IS THE ARITHMETIC MEAN THE RIGHT STATISTIC? The project aggregates per-word
     confidence with `mean`. For the decision this project is actually about —
     "commit to this transcript, or ask the caller to repeat?" — the mean is a
     strange choice: it lets nine confident function words bury one destroyed
     proper noun. The operative statistics are the MINIMUM, a low percentile, or
     the vendor's own utterance-level score. `separation` ranks them empirically
     by AUROC, per arm, and publishes the ranking with bootstrap intervals rather
     than an assertion.

  3. WHAT IS `utterance_conf`? It is in the frozen master schema
     (SPEC A.R4.2), it is written on every row by every adapter, and outside this
     module NOTHING in `deadzone/` reads it. A second confidence signal, captured
     and unused. `utterance_vs_word_mean` asks whether that was a loss: if it is
     numerically identical to the word mean, "we didn't use it" is a complete
     answer; if it diverges, the divergence is the finding.

  4. DOES IT SATURATE? A score that is exactly 1.0 on a tenth of all words is not
     behaving like a posterior over a hypothesis; it is behaving like a score with
     a ceiling, and a ceiling destroys ordering exactly where a monitor most needs
     it. `saturation` measures the pile-up at both ends, per arm, and checks
     whether the saturated words are actually correct.

-----------------------------------------------------------------------------
THE THREE RULES THIS MODULE MAY NOT BREAK
-----------------------------------------------------------------------------
CONFIDENCE IS NEVER POOLED ACROSS ARMS. Deepgram returns an acoustic confidence,
Whisper the decoder's token softmax (or the segment proxy exp(avg_logprob)), and
Scribe exp(logprob). On this grid those live at medians of 0.963, 0.304 and 0.998
respectively — a raw-threshold rule would call Whisper permanently unconfident and
miss every one of its dead zones. So every function here takes ONE arm's rows and
raises on two (`_one_model`), and every cross-arm quantity reported is either a
census or a rank statistic (AUROC, spearman) that is invariant to any monotone
rescaling of a single arm's confidence — the same discipline
`model_compare.within_model_conf_percentile` enforces for D1/L1.

MATCHED ESTIMANDS (SPEC Appendix G, the project's signature defect). A confidence
exists only for a row that emitted words. A row that came back EMPTY scores WER
1.0 and contributes NOTHING to any confidence statistic. Averaging confidence over
the rows that spoke and WER over all rows, then comparing them, is a subtraction
across two populations; it inflated the published headline gap by +0.109 and cost
four of six dead zones. So: every aggregate here is computed on the SPOKE rows,
every label it is scored against is read off THOSE SAME rows, and `n_spoke` /
`n_silent` travel beside every number (`spoke_and_silent`). There is no code path
in this module that pairs a confidence with an accuracy measured elsewhere.

THE DELETION CEILING IS THE HONEST LIMIT ON EVERYTHING BELOW. A deleted reference
word has no hypothesis token and therefore no confidence — permanently, by
construction. On this grid deletions are the DOMINANT nova-3 error mode, so any
sentence beginning "confidence tells you..." is a sentence about a minority of the
damage. `deletion_ceiling` (reused wholesale from `calibration_report`) quantifies
it and is printed at the TOP of the report, not in a footnote.

-----------------------------------------------------------------------------
WHAT IS REUSED RATHER THAN REBUILT
-----------------------------------------------------------------------------
The word-level confidence-vs-correctness relationship — a reliability curve and
its ECE — ALREADY EXISTS and is not duplicated here:

  * `analysis.layers.word_records`   the alignment: `edits` -> the hypothesis
                                     words in order -> 1:1 with word_confidences,
                                     asserted, never zipped;
  * `calibration.reliability_curve`  the per-bin confidence vs empirical accuracy;
  * `calibration.expected_calibration_error`;
  * `calibration_report.deletion_blindness`  the ceiling above.

`word_reliability` is a thin call into those three. What this module ADDS at the
word level is the shape of the raw distribution (quantiles, saturation, the
correctness of saturated words) — which the calibration layer never reports
because it only ever needs conf-vs-accuracy, not conf's own marginal.

Deps: numpy, scipy. No audio, no API, no pandas. Reads results/master.csv only.

    ./.venv/bin/python -m deadzone.analysis.confidence_char
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Mapping, Sequence

import numpy as np

# Allow `python deadzone/analysis/confidence_char.py` as well as `-m`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.analysis import (                                     # noqa: E402
    as_float, as_int, check_unique_cells, coerce_row, failure_summary,
    is_silent_row, load_master_table, n_hyp_words, split_by_model,
    split_failures,
)
from deadzone.analysis.calibration_report import deletion_blindness  # noqa: E402
from deadzone.analysis.confidence_gap import WER_HI                  # noqa: E402
from deadzone.analysis.layers import word_records                    # noqa: E402
from deadzone.calibration import (                                   # noqa: E402
    expected_calibration_error, reliability_curve,
)
from scipy.stats import spearmanr                                    # noqa: E402

__all__ = [
    "AGGREGATES", "CONTROL_PREDICTORS", "SATURATION_EPS", "N_BOOT", "BOOT_SEED",
    "ArmPoolingError", "DegenerateLabelError",
    "row_confidences", "row_aggregates", "spoke_and_silent",
    "severity_rank", "benign_rows", "clean_reference", "saturation",
    "auroc", "separation", "operating_points", "utterance_vs_word_mean",
    "word_reliability", "deletion_ceiling", "condition_aggregate_table",
    "characterize", "report_by_model", "format_report", "write_report", "main",
]

# ---------------------------------------------------------------------------
# The aggregates under test.
#
# `mean` is what the project uses everywhere today. The others are the statistics
# a commit/re-prompt decision would actually want: the WEAKEST word (`min`), a low
# percentile that is less hostage to one token (`p10`, `p25`), the typical word
# (`median`), and the vendor's own utterance-level number (`utterance_conf`).
# ---------------------------------------------------------------------------
AGGREGATES: tuple[str, ...] = ("mean", "median", "p25", "p10", "min",
                               "utterance_conf")

# CONFIDENCE-FREE CONTROLS, scored in the same table on purpose.
#
# `min` and the low percentiles are LENGTH-SENSITIVE: a longer utterance has more
# chances to contain a weak word, so `min` drifts down with word count for reasons
# that have nothing to do with acoustics. If word count alone separates good rows
# from bad ones, a length-sensitive aggregate inherits that separation for free and
# the ranking below would be reporting utterance length wearing a confidence's
# name. So the count is scored as a predictor in its own right and printed in the
# same table. A control that does WELL is not a bug in the control.
CONTROL_PREDICTORS: tuple[str, ...] = ("n_words",)

# "Exactly saturated" is exact equality, not a tolerance: the question is whether
# the vendor emits the endpoint itself. The near-endpoint mass is reported
# separately with this epsilon so a clipped-at-(1-1e-6) arm (Scribe, whose adapter
# clips exp(logprob)) is not confused with an arm that returns a bare 1.0 (nova-3).
SATURATION_EPS = 1e-3

N_BOOT = 1000
BOOT_SEED = 20260806     # fixed: the artifact must be byte-reproducible

# The mildest observed level of every ORDERED factor, used to select the clean
# reference set BY FACTOR SETTING rather than by outcome (see `severity_rank`).
# `noise_type` is deliberately absent: babble/engine/road have no mild-to-harsh
# ordering, so ranking them would invent one.
_HARSHER_WITH = {"rt60": +1, "snr_db": -1, "mic_rolloff": +1}
_MILDEST_CODEC = "none"


class ArmPoolingError(ValueError):
    """Two models' confidences reached one statistic.

    Loud, with no bypass. Deepgram acoustic confidence, Whisper token softmax and
    Scribe exp(logprob) are three unrelated quantities that happen to share the
    interval [0, 1]; on this grid their word-confidence medians are 0.963, 0.304
    and 0.998. A pooled mean, quantile or AUROC over them is a measurement of
    scale conventions, and it comes out as a plausible number with no warning.
    Call these functions once per arm.
    """


class DegenerateLabelError(ValueError):
    """An AUROC was requested where every row carries the same label.

    Refused rather than returned. With one class absent the ranking has nothing
    to separate, and the conventional answers — 0.5, or NaN — are both worse than
    an exception: 0.5 reads as "this statistic is useless" and NaN reads as "no
    result", when the truth is "the question was not asked". This is the failure
    mode Appendix E catalogues: a guard whose degenerate output is a plausible
    value.
    """


# ===========================================================================
# 1. ROW-LEVEL PARSING AND THE SPOKE/SILENT CENSUS
# ===========================================================================

def _one_model(rows: Sequence[Mapping], where: str) -> str:
    """Assert these rows are one arm; return its name."""
    models = {str(r.get("model")) for r in rows}
    if len(models) > 1:
        raise ArmPoolingError(
            f"{where} got {len(models)} arms {sorted(models)}. Confidence is "
            f"only comparable WITHIN a model — see the module docstring and "
            f"model_compare.within_model_conf_percentile. Split with "
            f"analysis.split_by_model and call this once per arm.")
    return next(iter(models)) if models else ""


def row_confidences(row: Mapping) -> np.ndarray:
    """The per-word confidence list for one row, as a float array (may be empty).

    The column is a JSON string on disk and a real list in memory (tests, the
    in-process runner); both are accepted, and anything unparseable yields an
    EMPTY array — which makes the row silent, i.e. excluded from every confidence
    statistic and counted, rather than silently contributing a zero.
    """
    raw = row.get("word_confidences")
    if raw is None or raw == "":
        return np.zeros(0, dtype=float)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return np.zeros(0, dtype=float)
    if not isinstance(raw, (list, tuple)):
        return np.zeros(0, dtype=float)
    out = np.asarray([as_float(v) for v in raw], dtype=float)
    return out[np.isfinite(out)]


def row_aggregates(row: Mapping) -> dict:
    """Every candidate aggregate for ONE row, plus the length control.

    All of them are computed from the SAME word-confidence list, so nothing in the
    comparison below can be an artifact of one statistic seeing different words
    than another. `utterance_conf` is the exception by definition — it is the
    adapter's stored utterance-level number, not a function of the list — and that
    is exactly what makes it worth testing separately.

    Returns NaN for every aggregate on a row that emitted no words. NaN is the
    right answer there and it is not imputed: an absent confidence is not a low
    confidence (SPEC analysis/__init__, the failure-sentinel trap).
    """
    c = row_confidences(row)
    n = int(c.size)
    if n == 0:
        agg = {k: float("nan") for k in AGGREGATES}
    else:
        agg = {
            "mean": float(c.mean()),
            "median": float(np.median(c)),
            "p25": float(np.percentile(c, 25)),
            "p10": float(np.percentile(c, 10)),
            "min": float(c.min()),
            "utterance_conf": as_float(row.get("utterance_conf")),
        }
    agg["n_words"] = float(n)
    agg["max"] = float(c.max()) if n else float("nan")
    return agg


def spoke_and_silent(rows: Sequence[Mapping]) -> dict:
    """Partition one arm's usable rows into (spoke, silent) and CENSUS them.

    "Spoke" means the row carried at least one per-word confidence — which is
    literally the denominator of every statistic in this module, so it is the only
    defensible definition of the population those statistics describe.

    THE PROJECT HAS TWO DEFINITIONS OF SILENCE AND THEY DISAGREE ON REAL ROWS.
    `analysis.is_silent_row` derives the hypothesis-word count from the EDIT
    ALIGNMENT (n_ref - n_del + n_ins) over `normalize_text`-ed tokens, while the
    confidence list is per RAW vendor token (SPEC G.9). On this grid they disagree
    on 2 of 10,557 usable rows — both Whisper rows that transcribed pure
    punctuation: one vendor token, zero scorable words. Neither definition is
    wrong; they answer different questions. This module partitions on the
    CONFIDENCE list because that is what it aggregates, and reports the
    disagreement count rather than picking one silently.
    """
    spoke, silent = [], []
    n_disagree = 0
    for r in rows:
        c = row_confidences(r)
        (spoke if c.size else silent).append(r)
        if (c.size == 0) != bool(is_silent_row(r)):
            n_disagree += 1
    n = len(rows)
    return {
        "spoke": spoke,
        "silent": silent,
        "n_rows": n,
        "n_spoke": len(spoke),
        "n_silent": len(silent),
        "silent_frac": (len(silent) / n) if n else float("nan"),
        "n_silence_definition_disagreements": n_disagree,
        "note": ("every confidence statistic in this report is over the "
                 f"{len(spoke)} rows that emitted words; the {len(silent)} silent "
                 f"rows carry WER 1.0 and NO confidence, so they are invisible to "
                 f"a confidence-based monitor by construction. Their accuracy is "
                 f"never averaged into anything a confidence is compared against."),
    }


# ===========================================================================
# 2. THE CLEAN REFERENCE DISTRIBUTION — the ruler every other number needs
# ===========================================================================

def _ordered_levels(rows: Sequence[Mapping], factor: str) -> list[float]:
    """Observed levels of an ordered factor, MILDEST FIRST."""
    vals = sorted({as_float(r.get(factor)) for r in rows
                   if np.isfinite(as_float(r.get(factor)))})
    return vals if _HARSHER_WITH[factor] > 0 else vals[::-1]


def severity_rank(rows: Sequence[Mapping]) -> dict[str, int]:
    """condition_name -> severity, an integer distance from the mildest corner.

    Sum over the ordered factors of "how many levels harsher than the mildest is
    this one", plus 1 if any codec at all is applied. Severity 0 is the corner
    where every ordered factor sits at its gentlest level.

    THIS IS THE WHOLE POINT AND IT IS EASY TO GET WRONG. The obvious way to pick
    "the clean conditions" is to take the lowest-WER cells. That is selection on
    the outcome: the resulting confidence distribution would be conditioned on the
    model having done well, and quoting a dead zone against it would be quoting it
    against a baseline chosen to flatter the baseline. This function reads FACTOR
    COLUMNS ONLY — it never sees `wer`, `mean_conf` or any measurement — so the
    reference set is fixed by the experiment's design, not by its results.

    `noise_type` is excluded: babble/engine/road have no mild-to-harsh ordering,
    and imposing one would smuggle a result into the selection.
    """
    levels = {f: _ordered_levels(rows, f) for f in _HARSHER_WITH}
    out: dict[str, int] = {}
    for r in rows:
        name = str(r.get("condition_name"))
        if name in out:
            continue
        sev = 0
        for f, ls in levels.items():
            v = as_float(r.get(f))
            sev += ls.index(v) if (np.isfinite(v) and v in ls) else len(ls)
        sev += 0 if str(r.get("codec")) == _MILDEST_CODEC else 1
        out[name] = sev
    return out


def benign_rows(rows: Sequence[Mapping], max_severity: int = 0) -> list[dict]:
    """Rows whose CONDITION sits within `max_severity` steps of the mildest corner."""
    sev = severity_rank(rows)
    return [dict(r) for r in rows
            if sev.get(str(r.get("condition_name")), 10 ** 6) <= max_severity]


def _dist(a: np.ndarray) -> dict:
    """mean/sd/quantiles of a 1-D sample; NaN-safe, never silently empty."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "quantiles": {}, "min": float("nan"), "max": float("nan")}
    qs = np.percentile(a, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()), "max": float(a.max()),
        "quantiles": {"p1": float(qs[0]), "p5": float(qs[1]), "p10": float(qs[2]),
                      "p25": float(qs[3]), "p50": float(qs[4]), "p75": float(qs[5]),
                      "p90": float(qs[6]), "p95": float(qs[7]), "p99": float(qs[8])},
        "iqr": float(qs[5] - qs[3]),
    }


def clean_reference(rows: Sequence[Mapping], max_severity: int = 0,
                    n_hist_bins: int = 20) -> dict:
    """The confidence distribution WHEN NOTHING MUCH IS WRONG — one arm.

    Reported at two resolutions, because a dead zone is quoted at both:
      * `per_word`   every individual word confidence in the benign set — the
                     shape of the raw signal, which is what saturation is about;
      * `per_clip`   each clip's own mean confidence — the unit `mean_conf` is,
                     and therefore the unit a per-condition headline number is
                     built from.

    The paired accuracy is measured on EXACTLY these rows (`wer_spoke`), never on
    the whole benign set including its silent rows, so "0.962 confidence at WER
    0.008" is a statement about one population.
    """
    model = _one_model(rows, "clean_reference")
    ok, _ = split_failures([coerce_row(dict(r)) for r in rows])
    ben = benign_rows(ok, max_severity)
    census = spoke_and_silent(ben)
    spoke = census["spoke"]
    words = np.concatenate([row_confidences(r) for r in spoke]) if spoke \
        else np.zeros(0)
    clip_means = np.array([as_float(r.get("mean_conf")) for r in spoke], dtype=float)
    wer = np.array([as_float(r.get("wer")) for r in spoke], dtype=float)
    hist, edges = (np.histogram(words, bins=n_hist_bins, range=(0.0, 1.0))
                   if words.size else (np.zeros(n_hist_bins, int),
                                       np.linspace(0, 1, n_hist_bins + 1)))
    conds = sorted({str(r.get("condition_name")) for r in ben})
    return {
        "model": model,
        "max_severity": max_severity,
        "selection": ("BY FACTOR SETTING, never by WER: conditions within "
                      f"{max_severity} level(s) of the mildest corner of every "
                      "ORDERED factor (rt60 lowest, snr_db highest, mic_rolloff "
                      "0, codec none). noise_type has no mild/harsh ordering and "
                      "is not ranked. Selecting on low WER instead would condition "
                      "the baseline on the model having done well."),
        "conditions": conds,
        "n_conditions": len(conds),
        "census": {k: v for k, v in census.items() if k not in ("spoke", "silent")},
        "per_word": _dist(words),
        "per_clip": _dist(clip_means),
        "wer_spoke": float(np.nanmean(wer)) if wer.size else float("nan"),
        "histogram": {"counts": [int(x) for x in hist],
                      "edges": [float(x) for x in edges]},
    }


def raw_capture_reference(path: str = "results/clean_transcripts.jsonl") -> dict:
    """The TRUE zero-degradation baseline: the raw recordings, no condition at all.

    There is no "clean" `Condition` — `apply_condition` ALWAYS convolves an RIR and
    ALWAYS mixes noise (SPEC A.R3.2), so even the mildest grid cell has a real room
    and real babble on it. The only true null is the raw wav, and R1.10 already
    transcribed all 40 of them. Loading that artifact costs nothing and gives the
    benign-corner number something to be read against in turn.

    Returns `{"available": False, ...}` when the artifact is absent — this is a
    supplementary ruler, not a dependency, and the arm that produced it is named
    rather than assumed (it exists for the spine arm only).
    """
    if not os.path.exists(path):
        return {"available": False, "path": path,
                "note": "raw-capture baseline artifact not present"}
    confs: list[float] = []
    clip_means: list[float] = []
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            wc = [as_float(v) for v in (rec.get("word_confidences") or [])]
            wc = [v for v in wc if np.isfinite(v)]
            if not wc:
                continue
            n += 1
            confs.extend(wc)
            clip_means.append(float(np.mean(wc)))
    return {
        "available": True, "path": path, "n_clips": n,
        "per_word": _dist(np.asarray(confs)),
        "per_clip": _dist(np.asarray(clip_means)),
        "note": ("the RAW recordings with NO condition applied — the only true "
                 "null in this project, since apply_condition always adds an RIR "
                 "and noise. Produced by the R1.10 clean-baseline pass, which ran "
                 "on the spine arm only; it is not a per-arm quantity."),
    }


# ===========================================================================
# 3. SATURATION — does the score have a ceiling?
# ===========================================================================

def saturation(rows: Sequence[Mapping], eps: float = SATURATION_EPS) -> dict:
    """Pile-up at the endpoints of one arm's word-confidence distribution.

    Four things, because a bare "8% of words are exactly 1.0" invites the wrong
    reading:

      * EXACT endpoint mass (== 1.0, == 0.0). Exact equality on purpose: the
        question is whether the vendor emits the endpoint itself.
      * NEAR-endpoint mass (>= 1-eps, <= eps), which separates an arm that returns
        a bare 1.0 from one whose adapter CLIPS into (0,1) — Scribe's
        `_prob_from_logprob` clips exp(logprob) at 1-1e-6, so it can never be
        exactly 1.0 while still being saturated in every way that matters.
      * `n_distinct` and `top_mass`: how much resolution is actually left at the
        confident end. A ceiling does not merely compress the scale, it DESTROYS
        ORDERING — every saturated word is tied, so no threshold, percentile or
        ranking can separate them, and a monitor built on the top of the range has
        nothing to work with there.
      * `accuracy_at_ceiling` — are the saturated words right? A ceiling that is
        always correct is a harmless encoding; a ceiling that is wrong 1 word in
        20 is a silent-failure generator, since 1.0 admits no discount at all.
    """
    model = _one_model(rows, "saturation")
    ok, _ = split_failures([coerce_row(dict(r)) for r in rows])
    words = [row_confidences(r) for r in ok]
    a = np.concatenate(words) if words else np.zeros(0)
    if a.size == 0:
        return {"model": model, "n_words": 0,
                "note": "no word confidences in this arm"}

    # Correctness of the ceiling words, via the SAME alignment L2 uses — never a
    # second copy of the edits->hypothesis-words recipe.
    w = word_records(ok, on_misalign="skip")
    wc, corr = np.asarray(w["conf"], dtype=float), np.asarray(w["correct"])
    # `word_records` clips into (0, 1) before returning, so an exactly-1.0 vendor
    # confidence arrives as 1-CONF_EPS. Match on the near-endpoint band, which is
    # the same set of words either way, and say so rather than testing == 1.0 and
    # silently finding none.
    hi = wc >= (1.0 - eps)
    lo = wc <= eps
    return {
        "model": model,
        "n_words": int(a.size),
        "exactly_one": int((a == 1.0).sum()),
        "frac_exactly_one": float((a == 1.0).mean()),
        "exactly_zero": int((a == 0.0).sum()),
        "frac_exactly_zero": float((a == 0.0).mean()),
        "eps": eps,
        "frac_within_eps_of_one": float((a >= 1.0 - eps).mean()),
        "frac_within_eps_of_zero": float((a <= eps).mean()),
        "observed_max": float(a.max()), "observed_min": float(a.min()),
        "n_distinct": int(np.unique(a).size),
        "n_distinct_above_p90": int(np.unique(a[a >= np.percentile(a, 90)]).size),
        "top_mass_note": ("every word inside the ceiling band is TIED, so no "
                          "threshold or percentile can order them: saturation "
                          "removes resolution exactly where a commit/re-prompt "
                          "rule needs it most"),
        "accuracy_at_ceiling": {
            "n_words": int(hi.sum()),
            "accuracy": float(corr[hi].mean()) if hi.any() else float("nan"),
            "n_words_at_floor": int(lo.sum()),
            "accuracy_at_floor": float(corr[lo].mean()) if lo.any() else float("nan"),
            "overall_accuracy": float(corr.mean()) if corr.size else float("nan"),
            "n_rows_skipped_misaligned": int(w["n_misaligned_rows"]),
            "note": ("correctness comes from analysis.layers.word_records — the "
                     "same alignment L2 calibrates on. `word_records` clips "
                     "confidences into (0,1), so the ceiling is matched on the "
                     "near-endpoint band rather than on == 1.0, which would find "
                     "nothing and read as 'no saturated words'."),
        },
    }


# ===========================================================================
# 4. WHICH AGGREGATE SEPARATES GOOD FROM BAD? (the headline of this module)
# ===========================================================================

def auroc(score: Sequence[float], bad: Sequence[bool]) -> float:
    """P(score of a randomly chosen BAD row < score of a randomly chosen GOOD row).

    Orientation is fixed and deliberate: LOW confidence should indicate a bad
    utterance, so 1.0 means the statistic separates perfectly in the useful
    direction and 0.5 means it carries no information. Ties are handled by average
    ranks (the Mann-Whitney identity), which matters enormously here — `min` on a
    saturated arm produces huge tie blocks, and an argsort-based AUROC would score
    those tie blocks as if they were ordered and inflate every saturated arm.

    Raises `DegenerateLabelError` when one class is empty, rather than returning
    0.5 or NaN. See that class for why both are worse than an exception.
    """
    s = np.asarray(score, dtype=float)
    y = np.asarray(bad, dtype=bool)
    if s.size != y.size:
        raise ValueError(f"auroc: {s.size} scores but {y.size} labels — these are "
                         f"parallel arrays over the same rows.")
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise DegenerateLabelError(
            f"auroc: {n_pos} bad and {n_neg} good rows among {s.size} finite "
            f"scores — one class is empty, so there is nothing to separate. "
            f"Returning 0.5 would read as 'this statistic is useless' and NaN as "
            f"'no result'; both are wrong, the question was never asked.")
    from scipy.stats import rankdata
    r = rankdata(s)                       # average ranks: ties share a rank
    # High score should mean GOOD, so a perfect separator puts every bad row at
    # the bottom of the ranking.
    auc_good_high = (r[~y].sum() - n_neg * (n_neg + 1) / 2.0) / (n_neg * n_pos)
    return float(auc_good_high)


def _pctl_ci(vals: Sequence[float], n_boot: int) -> tuple[float, float]:
    """95% percentile interval, or (NaN, NaN) if too few replicates survived.

    A degenerate CI must print as ABSENT, never as a tight interval around a
    handful of surviving replicates — that is the flattering failure.
    """
    v = [x for x in vals if np.isfinite(x)]
    if len(v) < max(20, n_boot // 10):
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def _boot_auroc_and_delta(score: np.ndarray, base: np.ndarray, bad: np.ndarray,
                          clip: np.ndarray, n_boot: int, seed: int) -> dict:
    """Clip-bootstrap CIs for AUROC(score) and for AUROC(score) - AUROC(base).

    CLIPS ARE THE RESAMPLING UNIT, not rows. The same utterance appears once per
    condition, and its difficulty (its entities, its length, how the speaker said
    it) is shared across all of those rows. Resampling rows treats them as
    independent draws, which they are not, and the interval comes out far too
    narrow — the same argument `analysis/sensitivity.py` makes for bootstrapping
    the 40 clips rather than the cells.

    THE DELTA IS PAIRED: both statistics are recomputed on the SAME resampled
    clips inside one loop, so the dominant source of variation (which clips were
    drawn) cancels. Two independent loops would give an unpaired interval, which
    is much wider and would call every difference non-significant — and it would
    also cost twice the compute for a worse answer.

    A replicate that happens to draw a single label class is SKIPPED, not scored
    as 0.5.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clip, return_inverse=True)
    by_clip = [np.where(inv == i)[0] for i in range(len(uniq))]
    a_vals: list[float] = []
    d_vals: list[float] = []
    same = score is base
    for _ in range(n_boot):
        pick = rng.integers(0, len(by_clip), len(by_clip))
        idx = np.concatenate([by_clip[i] for i in pick])
        try:
            a = auroc(score[idx], bad[idx])
            b = a if same else auroc(base[idx], bad[idx])
        except DegenerateLabelError:
            continue
        a_vals.append(a)
        d_vals.append(a - b)
    return {"auroc_ci": list(_pctl_ci(a_vals, n_boot)),
            "delta_ci": list(_pctl_ci(d_vals, n_boot))}


def separation(rows: Sequence[Mapping], wer_hi: float = WER_HI,
               baseline: str = "mean", n_boot: int = N_BOOT,
               seed: int = BOOT_SEED) -> dict:
    """RANK THE AGGREGATES by how well each separates good rows from bad ones.

    THE QUESTION, stated operationally: a voice agent has just received one
    utterance's transcript and must decide *commit, or ask the caller to repeat*.
    It has the word confidences and nothing else. Which single number off that
    list best tells it which situation it is in?

    So the unit is the UTTERANCE (one clip in one condition), not the condition —
    a deployed system never gets to average over 40 clips before deciding. A row
    is BAD when its own WER is >= `wer_hi`.

    Every score and every label is read off the SAME spoke rows, so the estimand
    rule holds by construction: there is no path here that can pair a confidence
    with an accuracy measured on a different population. Silent rows are excluded
    and counted, and the exclusion is the ceiling — a monitor reading confidence
    cannot see them at all.

    Metric is AUROC: rank-based, therefore invariant to any monotone rescaling,
    therefore safe to compare `min` against `mean` against a vendor's utterance
    score even though those three live on different parts of the interval. Every
    AUROC carries a clip-bootstrap CI, and every non-baseline aggregate also
    carries a PAIRED CI on its difference from `mean` — because "min beats mean"
    is a claim about a difference and a difference needs its own interval.
    """
    model = _one_model(rows, "separation")
    ok, _ = split_failures([coerce_row(dict(r)) for r in rows])
    check_unique_cells(ok, ("clip_id", "condition_name"), where="separation",
                       cause=("Usual cause: two runs concatenated. A duplicated "
                              "row is double-weighted in every AUROC below and "
                              "the table still looks complete."))
    census = spoke_and_silent(ok)
    spoke = census["spoke"]
    if len(spoke) < 10:
        raise ValueError(f"separation: only {len(spoke)} rows emitted words for "
                         f"{model!r}; nothing to rank.")

    aggs = [row_aggregates(r) for r in spoke]
    wer = np.array([as_float(r.get("wer")) for r in spoke], dtype=float)
    bad = wer >= wer_hi
    clip = np.array([str(r.get("clip_id")) for r in spoke])

    keys = list(AGGREGATES) + list(CONTROL_PREDICTORS)
    scores = {k: np.array([a.get(k, float("nan")) for a in aggs], dtype=float)
              for k in keys}
    base = scores[baseline]

    per: list[dict] = []
    for i, k in enumerate(keys):
        s = scores[k]
        finite = np.isfinite(s)
        entry: dict = {
            "aggregate": k,
            "is_control": k in CONTROL_PREDICTORS,
            "n": int(finite.sum()),
            "n_non_finite": int((~finite).sum()),
            "mean_value": float(np.nanmean(s)),
        }
        try:
            entry["auroc"] = auroc(s, bad)
        except DegenerateLabelError as exc:
            entry["auroc"] = float("nan")
            entry["error"] = str(exc)
            per.append(entry)
            continue
        boot = _boot_auroc_and_delta(s, base, bad, clip, n_boot, seed + i)
        entry["auroc_ci"] = boot["auroc_ci"]
        m = finite & np.isfinite(wer)
        entry["spearman_vs_wer"] = (float(spearmanr(s[m], wer[m])[0])
                                    if m.sum() > 2 and np.ptp(s[m]) > 0
                                    else float("nan"))
        if k != baseline:
            entry["d_auroc_vs_" + baseline] = entry["auroc"] - auroc(base, bad)
            dlo, dhi = boot["delta_ci"]
            entry["d_auroc_ci"] = [dlo, dhi]
            # "Beats the baseline" is a claim about a DIFFERENCE, so it is decided
            # by the paired interval, never by the point estimate. On this grid
            # several aggregates are within 0.005 AUROC of `mean` with intervals
            # straddling zero; calling any of those a winner would be reading
            # noise.
            entry["beats_" + baseline] = bool(np.isfinite(dlo) and dlo > 0.0)
            entry["loses_to_" + baseline] = bool(np.isfinite(dhi) and dhi < 0.0)
        per.append(entry)

    ranked = sorted(per, key=lambda d: (-d["auroc"] if np.isfinite(d["auroc"])
                                        else np.inf))
    winners = [d for d in ranked if not d["is_control"] and np.isfinite(d["auroc"])]
    best = winners[0]["aggregate"] if winners else None
    base_row = next(d for d in per if d["aggregate"] == baseline)

    # THE VERDICT IS DECIDED BY THE INTERVALS, NOT BY THE ORDERING. Sorting always
    # produces a "winner"; on a table where every aggregate is within a hundredth
    # of `mean` that winner is a coin flip, and printing it as a finding is how a
    # ranking of noise gets quoted. Significance is the paired CI excluding zero.
    sig_better = [d["aggregate"] for d in ranked
                  if d.get("beats_" + baseline) and not d["is_control"]]
    sig_worse = [d["aggregate"] for d in ranked
                 if d.get("loses_to_" + baseline) and not d["is_control"]]
    if sig_better:
        verdict = (f"{sig_better[0]!r} beats the arithmetic mean: paired 95% CI on "
                   f"the AUROC difference excludes zero. The project's aggregator "
                   f"is not the best available one for this arm.")
    else:
        verdict = (f"NOTHING beats the arithmetic mean on this arm — no "
                   f"alternative aggregate's paired 95% CI on the AUROC "
                   f"difference lies above zero"
                   + (f"; {len(sig_worse)} of them "
                      f"({', '.join(sig_worse)}) are significantly WORSE"
                      if sig_worse else "")
                   + ". The `mean` the project already uses is the right choice "
                     "here, and that is now a measured result rather than an "
                     "unexamined default.")
    return {
        "model": model,
        "unit": "utterance (one clip in one condition)",
        "label": f"bad := row WER >= {wer_hi}",
        "wer_hi": wer_hi,
        "baseline": baseline,
        "n_rows_scored": len(spoke),
        "n_bad": int(bad.sum()), "n_good": int((~bad).sum()),
        "bad_rate": float(bad.mean()),
        "census": {k: v for k, v in census.items() if k not in ("spoke", "silent")},
        "n_boot": n_boot, "boot_seed": seed,
        "boot_unit": "clip_id (resampled with replacement; rows within a clip "
                     "share its difficulty and are not independent)",
        "per_aggregate": per,
        "ranking": [d["aggregate"] for d in ranked],
        "best_aggregate": best,
        "baseline_auroc": base_row.get("auroc"),
        "best_auroc": winners[0]["auroc"] if winners else float("nan"),
        "significantly_better_than_baseline": sig_better,
        "significantly_worse_than_baseline": sig_worse,
        "verdict": verdict,
        "controls_note": ("`n_words` is a CONFIDENCE-FREE control. `min` and the "
                          "low percentiles fall with utterance length for purely "
                          "combinatorial reasons, so if the control ranks high, "
                          "part of their advantage is length, not acoustics. It "
                          "is printed in the same table so the reader can see it."),
    }


def operating_points(rows: Sequence[Mapping], aggregate: str,
                     wer_hi: float = WER_HI,
                     flag_rates: Sequence[float] = (0.05, 0.10, 0.20, 0.30)) -> dict:
    """Turn one aggregate into the actual decision rule, at several budgets.

    AUROC is a summary over every threshold; a deployment picks ONE. "Flag the
    lowest-confidence R% of utterances and ask those callers to repeat" is the
    rule a product would ship, so the table reports, per budget R: what confidence
    threshold that is, what fraction of flagged utterances really were bad
    (precision), and what fraction of bad utterances got caught (recall).

    Same spoke-row population as `separation`, same label. The re-prompt budget is
    a business choice; this table is what each choice buys.
    """
    model = _one_model(rows, "operating_points")
    ok, _ = split_failures([coerce_row(dict(r)) for r in rows])
    spoke = spoke_and_silent(ok)["spoke"]
    s = np.array([row_aggregates(r).get(aggregate, float("nan")) for r in spoke],
                 dtype=float)
    wer = np.array([as_float(r.get("wer")) for r in spoke], dtype=float)
    m = np.isfinite(s) & np.isfinite(wer)
    s, wer = s[m], wer[m]
    bad = wer >= wer_hi
    if bad.sum() == 0 or (~bad).sum() == 0:
        raise DegenerateLabelError(
            f"operating_points({aggregate!r}): {int(bad.sum())} bad / "
            f"{int((~bad).sum())} good — no decision rule is measurable.")
    pts = []
    for rate in flag_rates:
        thr = float(np.percentile(s, 100.0 * rate))
        flag = s <= thr
        n_flag = int(flag.sum())
        pts.append({
            "flag_rate_requested": float(rate),
            "flag_rate_realized": float(flag.mean()),
            "threshold": thr,
            "n_flagged": n_flag,
            "precision": float(bad[flag].mean()) if n_flag else float("nan"),
            "recall": float(flag[bad].mean()),
        })
    return {"model": model, "aggregate": aggregate, "wer_hi": wer_hi,
            "n_rows": int(s.size), "base_rate_bad": float(bad.mean()),
            "points": pts,
            "note": ("threshold is a percentile of THIS arm's own aggregate "
                     "distribution — never a shared absolute cut-off, since the "
                     "arms' scales are unrelated. precision is measured against "
                     "the same rows the threshold was taken over.")}


# ===========================================================================
# 5. IS `utterance_conf` REDUNDANT WITH THE WORD MEAN?
# ===========================================================================

def utterance_vs_word_mean(rows: Sequence[Mapping],
                           wer_hi: float = WER_HI) -> dict:
    """Is the stored utterance-level score a second signal, or the same one?

    `utterance_conf` is in the frozen master schema, is written by every adapter,
    and — outside this module — is read by NOTHING in `deadzone/`. Before that can
    be called a missed opportunity it has to be shown to carry information the
    word mean does not, so this measures three things:

      * IDENTITY. Some adapters document that they REUSE the word mean because the
        vendor exposes no separate utterance score (Vosk, Scribe). For those arms
        the column is a copy, "we didn't use it" is a complete answer, and any
        apparent finding would be a tautology. Measured exactly, not assumed:
        `frac_identical`.
      * AGREEMENT. Pearson and Spearman against the word mean, plus the mean
        absolute difference, on the same rows.
      * WHETHER THE DIVERGENCE IS USEFUL. Two AUROCs at the same target, so the
        question "should we have used it?" gets a number rather than a correlation.
    """
    model = _one_model(rows, "utterance_vs_word_mean")
    ok, _ = split_failures([coerce_row(dict(r)) for r in rows])
    spoke = spoke_and_silent(ok)["spoke"]
    aggs = [row_aggregates(r) for r in spoke]
    wm = np.array([a["mean"] for a in aggs], dtype=float)
    uc = np.array([a["utterance_conf"] for a in aggs], dtype=float)
    wer = np.array([as_float(r.get("wer")) for r in spoke], dtype=float)
    m = np.isfinite(wm) & np.isfinite(uc)
    wm_, uc_, wer_ = wm[m], uc[m], wer[m]
    if wm_.size < 3:
        return {"model": model, "n": int(wm_.size),
                "verdict": "not measurable (too few paired rows)"}
    diff = uc_ - wm_
    ident = float(np.mean(np.abs(diff) <= 1e-12))
    pear = float(np.corrcoef(wm_, uc_)[0, 1]) if np.ptp(wm_) and np.ptp(uc_) \
        else float("nan")
    spear = float(spearmanr(wm_, uc_)[0]) if np.ptp(wm_) and np.ptp(uc_) \
        else float("nan")
    bad = wer_ >= wer_hi
    try:
        a_wm, a_uc = auroc(wm_, bad), auroc(uc_, bad)
    except DegenerateLabelError:
        a_wm = a_uc = float("nan")
    if ident >= 0.999:
        verdict = ("REDUNDANT BY CONSTRUCTION — this arm's adapter reuses the word "
                   "mean because the vendor exposes no separate utterance-level "
                   "score. Nothing was missed by not reading the column.")
    elif np.isfinite(pear) and pear >= 0.99:
        verdict = ("REDUNDANT IN PRACTICE — numerically distinct but almost "
                   "perfectly correlated with the word mean; using it would not "
                   "have changed a decision.")
    else:
        verdict = ("DISTINCT — it is not a restatement of the word mean "
                   f"(pearson {pear:.3f}, mean |difference| "
                   f"{float(np.abs(diff).mean()):.3f}). Whether that is USEFUL is "
                   "the AUROC comparison, not the correlation.")
    return {
        "model": model, "n": int(wm_.size),
        "frac_identical": ident,
        "pearson_vs_word_mean": pear,
        "spearman_vs_word_mean": spear,
        "mean_abs_diff": float(np.abs(diff).mean()),
        "mean_signed_diff": float(diff.mean()),
        "max_abs_diff": float(np.abs(diff).max()),
        "utterance_conf_range": [float(uc_.min()), float(uc_.max())],
        "word_mean_range": [float(wm_.min()), float(wm_.max())],
        "auroc_word_mean": a_wm,
        "auroc_utterance_conf": a_uc,
        "d_auroc": (a_uc - a_wm) if np.isfinite(a_uc) and np.isfinite(a_wm)
        else float("nan"),
        "verdict": verdict,
    }


# ===========================================================================
# 6. WORD-LEVEL CONFIDENCE vs CORRECTNESS — REUSED, not rebuilt
# ===========================================================================

def word_reliability(rows: Sequence[Mapping], n_bins: int = 15) -> dict:
    """Per-bin confidence vs observed word accuracy, for one arm.

    THIS IS A THIN WRAPPER OVER MACHINERY THAT ALREADY EXISTS, and it is written
    that way deliberately: `analysis.layers.word_records` owns the
    edits -> hypothesis-words -> confidences alignment (which asserts rather than
    zips), and `calibration.reliability_curve` / `expected_calibration_error` own
    the binning and the metric. Re-deriving either here would give the project two
    reliability curves that could drift apart, and the drift would be invisible.

    What is added is the RAW MARGINAL of the confidences — the calibration layer
    never reports it, because conf-vs-accuracy is all it needs, and it is precisely
    what "what does this number look like" is asking for.
    """
    model = _one_model(rows, "word_reliability")
    ok, _ = split_failures([coerce_row(dict(r)) for r in rows])
    w = word_records(ok, on_misalign="skip")
    conf = np.asarray(w["conf"], dtype=float)
    corr = np.asarray(w["correct"], dtype=int)
    if conf.size == 0:
        return {"model": model, "n_words": 0,
                "note": "no aligned hypothesis words in this arm"}
    return {
        "model": model,
        "n_words": int(conf.size),
        "n_rows_used": int(w["n_rows_used"]),
        "n_rows_skipped_misaligned": int(w["n_misaligned_rows"]),
        "n_conf_clipped": int(w["n_conf_clipped"]),
        "word_accuracy": float(corr.mean()),
        "confidence_distribution": _dist(conf),
        "reliability": reliability_curve(conf, corr, n_bins),
        "ece": expected_calibration_error(conf, corr, n_bins),
        "ece_note": ("FULL-SAMPLE raw ECE over every aligned word — a descriptor "
                     "of the marginal, NOT the L2 headline. L2 "
                     "(results/calibration.json) reports raw ECE on HELD-OUT "
                     "conditions across several grouped splits, which is a "
                     "different protocol and a different number. Quoting the two "
                     "against each other would be comparing an in-sample "
                     "description with an out-of-sample measurement."),
        "reused_from": ["analysis.layers.word_records",
                        "calibration.reliability_curve",
                        "calibration.expected_calibration_error"],
        "note": ("word accuracy here is EMITTED-word accuracy: a deleted "
                 "reference word has no hypothesis token and therefore no "
                 "confidence, so it cannot appear in this curve at all. See "
                 "`deletion_ceiling` for how much of the damage that hides."),
    }


def deletion_ceiling(rows: Sequence[Mapping]) -> dict:
    """The hard ceiling on every claim in this module — reused from L2.

    `calibration_report.deletion_blindness` already computes exactly this and is
    called rather than re-derived. It is surfaced HERE, and printed at the top of
    the report rather than the bottom, because the question this module answers
    ("what is the confidence score?") has an answer that begins with what the
    score is structurally unable to be about.
    """
    ok, _ = split_failures([dict(r) for r in rows])
    return deletion_blindness(ok)


# ===========================================================================
# 7. CONDITION-LEVEL VIEW (the resolution D1 quotes)
# ===========================================================================

def condition_aggregate_table(rows: Sequence[Mapping],
                              wer_hi: float = WER_HI) -> dict:
    """The same aggregate comparison at CONDITION resolution.

    D1 quotes conditions, not utterances, so the ranking is repeated here on the
    unit the headline uses. Each condition's aggregate is the macro mean over the
    clips THAT SPOKE of that clip's own statistic, and the label is `wer_spoke`
    over exactly those clips — the matched-estimand rule at the second resolution.

    n is 176 rather than thousands, so the intervals are wide and this table is
    the corroboration, not the claim. The utterance-level `separation` is the
    operational result.
    """
    model = _one_model(rows, "condition_aggregate_table")
    ok, _ = split_failures([coerce_row(dict(r)) for r in rows])
    groups: dict[str, list[Mapping]] = {}
    for r in ok:
        groups.setdefault(str(r.get("condition_name")), []).append(r)

    conds: list[dict] = []
    for name in sorted(groups):
        grp = groups[name]
        cen = spoke_and_silent(grp)
        spoke = cen["spoke"]
        if not spoke:
            conds.append({"condition_name": name, "n_clips": len(grp),
                          "n_spoke": 0, "n_silent": cen["n_silent"],
                          "mute": True})
            continue
        aggs = [row_aggregates(r) for r in spoke]
        rec = {"condition_name": name, "n_clips": len(grp),
               "n_spoke": cen["n_spoke"], "n_silent": cen["n_silent"],
               "mute": False,
               "wer_spoke": float(np.mean([as_float(r.get("wer")) for r in spoke])),
               "wer_all_clips": float(np.mean([as_float(r.get("wer")) for r in grp]))}
        for k in list(AGGREGATES) + list(CONTROL_PREDICTORS):
            vals = np.array([a.get(k, float("nan")) for a in aggs], dtype=float)
            vals = vals[np.isfinite(vals)]
            rec[k] = float(vals.mean()) if vals.size else float("nan")
        conds.append(rec)

    scored = [c for c in conds if not c["mute"]]
    bad = np.array([c["wer_spoke"] >= wer_hi for c in scored], dtype=bool)
    per = []
    for k in list(AGGREGATES) + list(CONTROL_PREDICTORS):
        s = np.array([c[k] for c in scored], dtype=float)
        try:
            a = auroc(s, bad)
        except DegenerateLabelError:
            a = float("nan")
        per.append({"aggregate": k, "is_control": k in CONTROL_PREDICTORS,
                    "auroc": a})
    per.sort(key=lambda d: (-d["auroc"] if np.isfinite(d["auroc"]) else np.inf))
    return {
        "model": model,
        "unit": "condition (macro mean over the clips that spoke)",
        "n_conditions": len(conds),
        "n_mute_conditions": sum(1 for c in conds if c["mute"]),
        "n_scored": len(scored),
        "n_bad": int(bad.sum()), "n_good": int((~bad).sum()),
        "label": f"bad := wer_spoke >= {wer_hi}",
        "per_aggregate": per,
        "ranking": [d["aggregate"] for d in per],
        "conditions": conds,
        "note": ("mute conditions (no clip emitted a word) carry NO confidence "
                 "and are listed, never scored — a confidence-based monitor is "
                 "structurally blind to them (SPEC Appendix G.5)."),
    }


def dynamic_range(cond_table: Mapping, clean: Mapping,
                  aggregate: str = "mean") -> dict:
    """How far the signal actually MOVES between the mildest and harshest cells.

    The clean reference is only half a ruler. The other half is the far end: if
    every condition on the grid returned a confidence within 0.03 of the clean
    baseline, the signal would be near-useless regardless of any AUROC. This
    reports the observed span in the arm's own units — clean corner, best and
    worst SCORED condition, and the drop between them — so a quoted dead-zone
    confidence can be located inside a measured range instead of floating.

    Mute conditions are excluded from the extremes and counted separately: they
    have no confidence, so including them would mean inventing one for the very
    cells where none exists.
    """
    scored = [c for c in cond_table.get("conditions", []) if not c.get("mute")]
    vals = [(as_float(c.get(aggregate)), c) for c in scored]
    vals = [(v, c) for v, c in vals if np.isfinite(v)]
    if not vals:
        return {"aggregate": aggregate, "n_scored": 0}
    lo_v, lo_c = min(vals, key=lambda t: t[0])
    hi_v, hi_c = max(vals, key=lambda t: t[0])
    clean_v = as_float(clean.get("per_clip", {}).get("mean"))
    return {
        "aggregate": aggregate,
        "n_scored": len(vals),
        "n_mute_excluded": int(cond_table.get("n_mute_conditions", 0)),
        "clean_corner": clean_v,
        "clean_corner_wer": as_float(clean.get("wer_spoke")),
        "highest_condition": {"condition_name": hi_c["condition_name"],
                              "value": hi_v,
                              "wer_spoke": as_float(hi_c.get("wer_spoke"))},
        "lowest_condition": {"condition_name": lo_c["condition_name"],
                             "value": lo_v,
                             "wer_spoke": as_float(lo_c.get("wer_spoke"))},
        "span": float(hi_v - lo_v),
        "drop_from_clean_to_worst": float(clean_v - lo_v),
        "note": ("mute conditions carry no confidence and are excluded from both "
                 "extremes — they are the WORST cells on the grid and the signal "
                 "cannot describe them at all."),
    }


# ===========================================================================
# 8. ONE ARM'S FULL CHARACTERISATION, AND THE REPORT
# ===========================================================================

def characterize(rows: Sequence[Mapping], model: str | None = None,
                 wer_hi: float = WER_HI, n_boot: int = N_BOOT,
                 seed: int = BOOT_SEED, benign_severity: int = 0,
                 benign_band_severity: int = 1) -> dict:
    """Everything this module knows about ONE arm's confidence signal."""
    if model is not None:
        rows = [r for r in rows if str(r.get("model")) == model]
    rows = [coerce_row(dict(r)) for r in rows]
    name = _one_model(rows, "characterize")
    sep = separation(rows, wer_hi, "mean", n_boot, seed)
    best = sep.get("best_aggregate") or "mean"
    corner = clean_reference(rows, benign_severity)
    cond = condition_aggregate_table(rows, wer_hi)
    out = {
        "model": name,
        "n_rows": len(rows),
        "failures": failure_summary(rows),
        "label_caveat": _label_caveat(name),
        "deletion_ceiling": deletion_ceiling(rows),
        "clean_reference_corner": corner,
        "clean_reference_band": clean_reference(rows, benign_band_severity),
        "dynamic_range": dynamic_range(cond, corner, "mean"),
        "saturation": saturation(rows),
        "separation": sep,
        # ONE table when the winner IS the baseline. Printing the identical
        # numbers twice under two headings reads as a comparison and is not one.
        "operating_points_best": operating_points(rows, best, wer_hi),
        "operating_points_mean": (None if best == "mean"
                                  else operating_points(rows, "mean", wer_hi)),
        "utterance_conf": utterance_vs_word_mean(rows, wer_hi),
        "word_reliability": word_reliability(rows),
        "condition_level": cond,
    }
    # "Should we have used utterance_conf?" is a question about a DIFFERENCE, and
    # `separation` has already paid for the paired clip-bootstrap of exactly that
    # difference. Reuse it rather than answering from a correlation — a pearson of
    # 0.93 says the two signals are not the same number; it says nothing about
    # which one predicts a bad transcript better.
    uc_row = next((d for d in sep["per_aggregate"]
                   if d["aggregate"] == "utterance_conf"), None)
    if uc_row is not None:
        out["utterance_conf"]["paired_d_auroc_vs_word_mean"] = uc_row.get(
            "d_auroc_vs_mean", float("nan"))
        out["utterance_conf"]["paired_d_auroc_ci"] = uc_row.get("d_auroc_ci")
        better = bool(uc_row.get("beats_mean"))
        worse = bool(uc_row.get("loses_to_mean"))
        # An arm whose adapter REUSES the word mean makes the AUROC comparison a
        # tautology (delta is identically zero, CI [0, 0]). Reporting that as
        # "no measurable advantage" would dress a definitional identity up as an
        # empirical finding, so the by-construction answer wins here.
        identical = as_float(out["utterance_conf"].get("frac_identical"), 0.0) >= 0.999
        out["utterance_conf"]["answer"] = (
            "Nothing was missed: this arm's adapter REUSES the word mean because "
            "the vendor exposes no separate utterance-level score, so the column "
            "is a copy and the AUROC comparison is a tautology." if identical else
            "It IS a second signal and it IS better: using it would have improved "
            "the headline aggregate." if better else
            "It is a second signal but a WORSE one: the paired CI on the AUROC "
            "difference is entirely below zero, so leaving it unread cost "
            "nothing." if worse else
            "It carries no measurable advantage or disadvantage over the word "
            "mean (the paired CI on the AUROC difference straddles zero), so "
            "leaving it unread cost nothing measurable.")
    out["statement"] = arm_statement(out)
    return out


def _label_caveat(model: str) -> dict:
    """Whether this arm's word-correctness labels are an upper bound on its error.

    The label ('was this emitted word correct?') comes from the SAME alignment as
    WER, so an arm whose orthography disagrees with the corpus has genuinely
    correct words marked wrong. Its accuracy reads low and its apparent
    overconfidence reads high. `model_compare` already owns the registry of which
    arms that applies to and WHY, with the evidence attached — this reads that
    registry rather than re-deciding it, so an arm cannot be silently treated as
    comparable here and incomparable in L1.

    The RANKING in `separation` is far more robust to this than the levels are: an
    orthography penalty that lands on the same words regardless of acoustics
    shifts every aggregate's accuracy together and largely cancels out of a rank
    statistic. Levels (accuracy, ECE) do not survive it and are labelled.
    """
    from deadzone.model_compare import WER_INCOMPARABLE_ARMS, is_wer_comparable
    ok = bool(is_wer_comparable(model))
    return {
        "model": model,
        "wer_comparable": ok,
        "labels_are_upper_bound_on_error": not ok,
        "reason": WER_INCOMPARABLE_ARMS.get(model, ""),
        "note": ("word-correctness labels come from the same alignment as WER. "
                 + ("This arm's orthography is comparable, so the labels are "
                    "taken at face value." if ok else
                    "This arm's orthography is NOT comparable, so some correct "
                    "words are labelled incorrect: its accuracy and ECE are "
                    "UPPER BOUNDS on its error, not point estimates. The AUROC "
                    "RANKING is far more robust — a penalty that lands "
                    "independently of the acoustic condition shifts every "
                    "aggregate together and largely cancels from a rank "
                    "statistic.")),
    }


def arm_statement(rep: Mapping) -> str:
    """The sentence an author can say out loud about this arm's confidence."""
    sep, sat = rep["separation"], rep["saturation"]
    corner, dyn = rep["clean_reference_corner"], rep["dynamic_range"]
    dc, uc = rep["deletion_ceiling"], rep["utterance_conf"]
    ceil_frac = as_float(sat.get("frac_within_eps_of_one"), 0.0)
    ceil_acc = as_float(sat.get("accuracy_at_ceiling", {}).get("accuracy"))
    ceil_txt = (f"saturates — {ceil_frac:.1%} of words within "
                f"{sat.get('eps', SATURATION_EPS):g} of 1.0, and those words are "
                f"{ceil_acc:.3f} correct — and falls"
                if ceil_frac > 0 and np.isfinite(ceil_acc) else
                f"does NOT saturate at the top ({ceil_frac:.1%} of words within "
                f"{sat.get('eps', SATURATION_EPS):g} of 1.0; observed max "
                f"{as_float(sat.get('observed_max')):.4f}), and falls")
    return (
        f"{rep['model']}'s per-word confidence sits at "
        f"{as_float(corner['per_clip'].get('mean')):.3f} (per clip) on the mildest "
        f"grid cell at WER {as_float(corner.get('wer_spoke')):.3f}, {ceil_txt} to "
        f"{as_float(dyn.get('lowest_condition', {}).get('value')):.3f} in the "
        f"harshest condition that still emits words. Aggregated per utterance it "
        f"separates bad transcripts (WER >= {sep['wer_hi']}) from good ones at "
        f"AUROC {as_float(sep.get('baseline_auroc')):.3f} using the arithmetic "
        f"mean; {sep['verdict']} "
        f"utterance_conf: {uc.get('answer', uc.get('verdict', ''))} "
        f"All of that describes only the words the model EMITTED: deletions are "
        f"{as_float(dc.get('deleted_fraction_of_errors'), 0.0):.1%} of its errors "
        f"and carry no confidence at all.")


def report_by_model(rows: Sequence[Mapping], **kw) -> dict:
    """One characterisation per arm. The arms are NEVER merged into one statistic.

    The only cross-arm objects in the payload are a census and a table of RANK
    statistics (AUROC), each computed inside one arm and therefore invariant to
    any monotone rescaling of that arm's confidence. No raw confidence from two
    arms ever meets in the same average.
    """
    by_model = {m: characterize(sub, model=m, **kw)
                for m, sub in sorted(split_by_model(rows).items())}
    cross = []
    for m, rep in by_model.items():
        sat, sep = rep["saturation"], rep["separation"]
        cw = rep["clean_reference_corner"]["per_word"]
        cross.append({
            "model": m,
            "clean_corner_mean_conf": cw.get("mean", float("nan")),
            "clean_corner_median_conf": cw.get("quantiles", {}).get("p50",
                                                                    float("nan")),
            "word_conf_median_all": rep["word_reliability"].get(
                "confidence_distribution", {}).get("quantiles", {}).get("p50",
                                                                        float("nan")),
            "frac_exactly_one": sat.get("frac_exactly_one", float("nan")),
            "frac_within_eps_of_one": sat.get("frac_within_eps_of_one", float("nan")),
            "best_aggregate": sep.get("best_aggregate"),
            "best_auroc": sep.get("best_auroc"),
            "mean_auroc": sep.get("baseline_auroc"),
            "silent_frac": sep.get("census", {}).get("silent_frac", float("nan")),
            "utterance_conf_verdict": rep["utterance_conf"].get("verdict", "")[:40],
        })
    return {
        "layer": "confidence characterisation — what the confidence score IS",
        "models": sorted(by_model),
        "by_model": by_model,
        "cross_arm": cross,
        "cross_arm_note": (
            "RAW confidences are NOT comparable across arms and none of them are "
            "compared here: the clean/median columns are per-arm rulers printed "
            "side by side so the SCALES can be seen to differ, and the AUROC "
            "columns are rank statistics computed strictly inside one arm. On "
            "this grid the word-confidence medians differ by a factor of three "
            "across arms; a shared absolute threshold would be meaningless."),
    }


# ===========================================================================
# 9. FORMATTING
# ===========================================================================

def _f(v, spec: str = ".3f") -> str:
    x = as_float(v)
    return "   n/a" if not np.isfinite(x) else format(x, spec)


def _ci(pair) -> str:
    if not pair or not np.isfinite(as_float(pair[0])):
        return "[     n/a     ]"
    return f"[{as_float(pair[0]):+.3f}, {as_float(pair[1]):+.3f}]"


def format_arm(rep: Mapping) -> str:
    m = rep["model"]
    dc, sat, sep = rep["deletion_ceiling"], rep["saturation"], rep["separation"]
    corner, band = rep["clean_reference_corner"], rep["clean_reference_band"]
    uc, wr, cl = rep["utterance_conf"], rep["word_reliability"], rep["condition_level"]
    dyn, lab = rep["dynamic_range"], rep["label_caveat"]
    L = [
        "=" * 78,
        f"ARM {m!r} — what its confidence score empirically IS",
        "=" * 78,
        "",
        "-- 0. THE CEILING ON EVERYTHING BELOW ---------------------------------",
        f"  deletions {dc['n_deletions']} = {dc['deleted_fraction_of_reference']:.1%} of "
        f"reference words and {dc['deleted_fraction_of_errors']:.1%} of ALL errors.",
        "  A deleted word has no hypothesis token, hence NO confidence: it cannot",
        "  appear in any statistic in this report. Confidence describes the words the",
        f"  model DID emit (accuracy {dc['emitted_word_accuracy']:.3f}), not how much of the",
        f"  reference was recovered ({dc['reference_word_recovery']:.3f}) — reading it as the",
        f"  latter overstates the system by {dc['accuracy_overstatement']:.3f}.",
        f"  Silent rows (no words at all): {sep['census']['n_silent']} / "
        f"{sep['census']['n_rows']} ({sep['census']['silent_frac']:.1%}); "
        f"{cl['n_mute_conditions']} conditions are mute on EVERY clip.",
        f"  LABELS: {'comparable orthography' if lab['wer_comparable'] else 'ORTHOGRAPHY NOT COMPARABLE — accuracy/ECE below are UPPER BOUNDS on error'}.",
        "",
        "-- 1. THE CLEAN REFERENCE DISTRIBUTION (the ruler) ---------------------",
        f"  selected {corner['selection'].split(':')[0]}",
        f"  corner  ({corner['n_conditions']} condition(s), "
        f"{corner['census']['n_spoke']} clips, {corner['per_word']['n']} words, "
        f"WER {_f(corner['wer_spoke'])})",
        f"      per-word   mean {_f(corner['per_word']['mean'])}  "
        f"median {_f(corner['per_word']['quantiles'].get('p50'))}  "
        f"sd {_f(corner['per_word']['sd'])}  "
        f"p10 {_f(corner['per_word']['quantiles'].get('p10'))}  "
        f"p90 {_f(corner['per_word']['quantiles'].get('p90'))}  "
        f"min {_f(corner['per_word']['min'])}",
        f"      per-clip   mean {_f(corner['per_clip']['mean'])}  "
        f"sd {_f(corner['per_clip']['sd'])}  "
        f"min {_f(corner['per_clip']['min'])}  max {_f(corner['per_clip']['max'])}",
        f"  band    ({band['n_conditions']} conditions within 1 level of the corner, "
        f"{band['per_word']['n']} words, WER {_f(band['wer_spoke'])})",
        f"      per-word   mean {_f(band['per_word']['mean'])}  "
        f"median {_f(band['per_word']['quantiles'].get('p50'))}  "
        f"p10 {_f(band['per_word']['quantiles'].get('p10'))}",
        "  DYNAMIC RANGE — where a quoted confidence sits inside the measured span:",
        f"      clean corner                {_f(dyn.get('clean_corner'))}  "
        f"(WER {_f(dyn.get('clean_corner_wer'))})",
        f"      highest scored condition    {_f(dyn.get('highest_condition', {}).get('value'))}  "
        f"{str(dyn.get('highest_condition', {}).get('condition_name'))}",
        f"      lowest  scored condition    {_f(dyn.get('lowest_condition', {}).get('value'))}  "
        f"{str(dyn.get('lowest_condition', {}).get('condition_name'))}",
        f"      drop (clean - worst)        {_f(dyn.get('drop_from_clean_to_worst'))}"
        f"   ({dyn.get('n_mute_excluded', 0)} mute conditions excluded: no confidence exists there)",
        "  READ EVERY DEAD-ZONE CONFIDENCE AGAINST THIS. A bare confidence number is",
        "  uninterpretable until the clean baseline and the span are both on the page.",
        "",
        "-- 2. SATURATION ------------------------------------------------------",
        f"  words {sat['n_words']}   exactly 1.0: {sat['exactly_one']} "
        f"({sat['frac_exactly_one']:.2%})   exactly 0.0: {sat['exactly_zero']} "
        f"({sat['frac_exactly_zero']:.2%})",
        f"  within {sat['eps']:g} of 1.0: {sat['frac_within_eps_of_one']:.2%}   "
        f"of 0.0: {sat['frac_within_eps_of_zero']:.2%}   "
        f"observed range [{_f(sat['observed_min'])}, {_f(sat['observed_max'], '.6f')}]",
        f"  distinct values {sat['n_distinct']} "
        f"({sat['n_distinct_above_p90']} of them above the p90)",
        f"  accuracy of the ceiling words: "
        f"{_f(sat['accuracy_at_ceiling']['accuracy'])} on "
        f"{sat['accuracy_at_ceiling']['n_words']} words "
        f"(arm overall {_f(sat['accuracy_at_ceiling']['overall_accuracy'])})",
        "  Tied words cannot be ordered by any threshold or percentile — saturation",
        "  removes resolution exactly where a commit/re-prompt rule needs it.",
        "",
        "-- 3. WHICH AGGREGATE SEPARATES GOOD FROM BAD? (utterance level) -------",
        f"  {sep['n_rows_scored']} utterances that emitted words; "
        f"{sep['n_bad']} bad / {sep['n_good']} good at {sep['label']}",
        f"  AUROC 1.0 = perfect (low confidence => bad); 0.5 = no information.",
        f"  CIs: {sep['n_boot']} bootstrap replicates over {sep['boot_unit'].split(' ')[0]}.",
        "",
        f"  {'#':>2} {'aggregate':<16} {'AUROC':>7} {'95% CI':>18} "
        f"{'dAUROC vs mean':>16} {'95% CI':>18} {'rho(WER)':>9}",
    ]
    for i, d in enumerate(sorted(sep["per_aggregate"],
                                 key=lambda x: (-x["auroc"]
                                                if np.isfinite(x["auroc"])
                                                else np.inf)), 1):
        tag = d["aggregate"] + (" [control]" if d["is_control"] else "")
        L.append(f"  {i:>2} {tag:<16} {_f(d.get('auroc')):>7} "
                 f"{_ci(d.get('auroc_ci')):>18} "
                 f"{_f(d.get('d_auroc_vs_mean'), '+.3f'):>16} "
                 f"{_ci(d.get('d_auroc_ci')):>18} "
                 f"{_f(d.get('spearman_vs_wer'), '+.3f'):>9}")
    L += [
        f"  top of the ordering: {sep['best_aggregate']!r} at AUROC "
        f"{_f(sep['best_auroc'])} vs the project's `mean` at "
        f"{_f(sep['baseline_auroc'])}.",
        f"  VERDICT (decided by the paired CI, not by the ordering): "
        f"{sep['verdict']}",
        "  " + sep["controls_note"],
        "",
        f"  condition-level corroboration (n={cl['n_scored']} scored, "
        f"{cl['n_mute_conditions']} mute): ranking "
        + " > ".join(cl["ranking"][:4]),
        "",
        "-- 4. THE DECISION RULE THIS BUYS -------------------------------------",
    ]
    for op in (rep["operating_points_best"], rep.get("operating_points_mean")):
        if not op:
            continue
        L.append(f"  flag the lowest-confidence R% by {op['aggregate']!r} "
                 f"(base rate bad {op['base_rate_bad']:.1%}):")
        for p in op["points"]:
            L.append(f"      R={p['flag_rate_requested']:.0%}  "
                     f"threshold {_f(p['threshold'])}  "
                     f"precision {p['precision']:.3f}  recall {p['recall']:.3f}")
    L += [
        "",
        "-- 5. IS `utterance_conf` REDUNDANT? ----------------------------------",
        f"  n {uc.get('n', 0)}   identical to the word mean on "
        f"{as_float(uc.get('frac_identical'), 0.0):.1%} of rows   "
        f"pearson {_f(uc.get('pearson_vs_word_mean'))}   "
        f"mean|diff| {_f(uc.get('mean_abs_diff'))}",
        f"  AUROC: word mean {_f(uc.get('auroc_word_mean'))} vs utterance_conf "
        f"{_f(uc.get('auroc_utterance_conf'))} "
        f"(paired delta {_f(uc.get('paired_d_auroc_vs_word_mean'), '+.3f')} "
        f"{_ci(uc.get('paired_d_auroc_ci'))})",
        f"  VERDICT: {uc.get('verdict', '')}",
        f"  ANSWER TO 'why wasn't it used?': {uc.get('answer', '')}",
        "",
        "-- 6. WORD-LEVEL CONFIDENCE vs CORRECTNESS (reused machinery) ---------",
        f"  {wr.get('n_words', 0)} aligned hypothesis words, emitted-word accuracy "
        f"{_f(wr.get('word_accuracy'))}, ECE {_f(wr.get('ece'), '.4f')}",
        f"  reused: {', '.join(wr.get('reused_from', []))}",
    ]
    for b in wr.get("reliability", []):
        L.append(f"      conf {b['bin_lo']:.2f}-{b['bin_hi']:.2f}  "
                 f"mean {b['conf_mean']:.3f}  accuracy {b['accuracy']:.3f}  "
                 f"n {b['count']}")
    L += ["", "STATEMENT: " + rep.get("statement", "")]
    return "\n".join(L)


def format_report(rep: Mapping) -> str:
    L = [
        "CONFIDENCE CHARACTERISATION — what the confidence score empirically IS",
        "=" * 78,
        "",
        "The project's headline signal is a per-word confidence, and the write-up",
        "only ever says what it is NOT ('not a calibrated probability'). This is",
        "the description: its clean reference distribution, its behaviour at the",
        "extremes, whether the arithmetic mean is the right way to aggregate it,",
        "and whether the utterance-level score the schema already captures adds",
        "anything. No vendor internals are claimed — only measured behaviour.",
        "",
        f"arms: {', '.join(rep['models'])}",
        "",
        "-- cross-arm summary (rank statistics and rulers ONLY) ----------------",
        f"  {'model':<20}{'clean p50':>10}{'all p50':>9}{'==1.0':>8}"
        f"{'~1.0':>8}{'silent':>8}{'best agg':>16}{'AUROC':>7}{'mean':>7}",
    ]
    for c in rep["cross_arm"]:
        L.append(f"  {c['model']:<20}"
                 f"{_f(c['clean_corner_median_conf']):>10}"
                 f"{_f(c['word_conf_median_all']):>9}"
                 f"{as_float(c['frac_exactly_one'], 0.0):>7.1%} "
                 f"{as_float(c['frac_within_eps_of_one'], 0.0):>7.1%} "
                 f"{as_float(c['silent_frac'], 0.0):>7.1%} "
                 f"{str(c['best_aggregate']):>15} "
                 f"{_f(c['best_auroc']):>6} {_f(c['mean_auroc']):>6}")
    L += ["", "  NOTE: " + rep["cross_arm_note"], ""]
    raw = rep.get("raw_capture_reference") or {}
    if raw.get("available"):
        L += [
            "-- the TRUE zero-degradation baseline (supplementary) ----------------",
            f"  {raw['n_clips']} RAW recordings, no condition applied at all: "
            f"per-word mean {_f(raw['per_word'].get('mean'))}, "
            f"per-clip mean {_f(raw['per_clip'].get('mean'))}, "
            f"n={raw['per_word'].get('n')} words.",
            "  There is no 'clean' Condition — apply_condition ALWAYS adds an RIR "
            "and noise — so",
            "  the mildest grid cell is a proxy. Comparing the two says how good a "
            "proxy it is.",
            "  " + raw.get("note", ""),
            "",
        ]
    for m in rep["models"]:
        L += [format_arm(rep["by_model"][m]), ""]
    return "\n".join(L)


def write_report(rep: Mapping, out_json: str = "results/confidence_char.json",
                 out_txt: str = "results/confidence_char.txt") -> tuple[str, str]:
    """Persist both artifacts. Byte-reproducible: no timestamps, fixed seed."""
    for path in (out_json, out_txt):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, sort_keys=True, default=float)
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(format_report(rep) + "\n")
    return out_json, out_txt


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Empirical characterisation of the ASR confidence signal")
    ap.add_argument("--table", default="results/master.csv")
    ap.add_argument("--model", default=None,
                    help="one arm (default: every arm, one section each)")
    ap.add_argument("--wer-hi", type=float, default=WER_HI)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--out-json", default="results/confidence_char.json")
    ap.add_argument("--out-txt", default="results/confidence_char.txt")
    args = ap.parse_args(argv)

    rows = load_master_table(args.table)
    if args.model:
        rows = [r for r in rows if str(r.get("model")) == args.model]
        if not rows:
            raise SystemExit(f"no rows for model {args.model!r} in {args.table!r}")
    rep = report_by_model(rows, wer_hi=args.wer_hi, n_boot=args.n_boot,
                          seed=args.seed)
    rep["raw_capture_reference"] = raw_capture_reference()
    text = format_report(rep)
    print(text)
    j, t = write_report(rep, args.out_json, args.out_txt)
    print(f"\nwrote {j} and {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
