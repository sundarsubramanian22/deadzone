"""
model_compare.py — Layer 1, the multi-model comparison harness (SPEC §5).

Two jobs:

  1. A MODEL REGISTRY: one place that lists the ASR arms to compare, each behind
     the shared adapter contract from audio_pipeline. Adding a model is ONE line.
  2. A cross-model DEAD-ZONE comparison: given a per-model results table, find and
     compare the "confidently wrong" regions across models.

THE CRITICAL SUBTLETY. Confidence scales are NOT comparable across model families
(Deepgram acoustic confidence, Whisper decoder softmax, Vosk Kaldi conf are three
different numbers). So we NEVER compare raw confidence across models. Each model's
confidence is normalized WITHIN that model (to a percentile of its own
distribution), and the cross-model question is about the *shape* of the
confidence-vs-WER relationship — does the model's confidence track its own errors?
— not whose number is bigger.

Decoupling: this module reads the factor space from design.py and operates on
plain results tables (list[dict] rows: the condition sample + 'wer' + 'mean_conf').
It never runs a model itself in tests; the registry's real adapters are lazily
resolved from audio_pipeline (whose SDK imports are themselves lazy), so importing
model_compare pulls in no ASR/network deps.

Deps: numpy, scipy.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from deadzone.design import DEFAULT_FACTOR_SPACE, FactorSpace


# ---------------------------------------------------------------------------
# Model registry — adding a model is ONE line
# ---------------------------------------------------------------------------
# Each entry maps a model id -> a callable(audio_path) -> adapter contract dict.
# The adapters live in audio_pipeline and lazy-import their own SDKs, so building
# this dict costs nothing until a model is actually called.

def _registry() -> dict[str, Callable[[str], dict]]:
    from deadzone.audio_pipeline import (
        transcribe_deepgram, transcribe_whisper, transcribe_vosk,
    )
    return {
        "nova-3":        lambda path: transcribe_deepgram(path, model="nova-3"),
        "whisper-base":  lambda path: transcribe_whisper(path, model_name="base"),
        "vosk":          lambda path: transcribe_vosk(path),
        # add a model HERE, one line:
        # "my-model":    lambda path: transcribe_mymodel(path),
    }


# Names only — importable/inspectable without resolving the adapters (no SDKs).
MODEL_REGISTRY = ("nova-3", "whisper-base", "vosk")


def get_model(name: str) -> Callable[[str], dict]:
    """Resolve one registry arm to its callable adapter (lazy)."""
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown model {name!r} (registry: {sorted(reg)})")
    return reg[name]


# ---------------------------------------------------------------------------
# Within-model normalization + dead-zone detection
# ---------------------------------------------------------------------------

def _col(rows: Sequence[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def within_model_conf_percentile(rows: Sequence[dict],
                                 conf_key: str = "mean_conf") -> np.ndarray:
    """
    Map each row's confidence to a percentile in [0,1] of THIS model's own
    confidence distribution. This is the only fair way to say "the model was
    confident here" across families whose raw scales differ. NaN confidences sink
    to percentile 0 (treated as least-confident).
    """
    c = _col(rows, conf_key)
    finite = np.isfinite(c)
    pct = np.zeros(len(c), dtype=float)
    if finite.sum() == 0:
        return pct
    # rank only the finite values; ties share the average rank
    ranks = rankdata(c[finite], method="average")
    pct[finite] = (ranks - 1) / max(finite.sum() - 1, 1)
    return pct


def dead_zone_flags(rows: Sequence[dict], wer_hi: float = 0.3,
                    conf_pct_hi: float = 0.6,
                    wer_key: str = "wer") -> np.ndarray:
    """
    Boolean per row: a DEAD ZONE = the model is *confidently wrong* — WER at or
    above `wer_hi` while its within-model confidence sits in the top (>= conf_pct_hi)
    of its own range. High WER with LOW self-confidence is a loud failure, not a
    silent one, and is deliberately excluded.

    `wer_key` NAMES THE ESTIMAND, and it is not cosmetic. "Confidently wrong" is a
    claim about the clips the model actually spoke on — the only clips its
    confidence was averaged over. When some clips returned an EMPTY transcript
    they score WER 1.0 but contribute no confidence, so a row's all-clips `wer`
    and its `mean_conf` describe different populations and this test would compare
    them anyway (see analysis/__init__.py, the second trap). Callers holding a
    per-condition table must pass the same-subset key (`wer_spoke`); the default
    stays `wer` for clip-level tables, where one row is one clip and the question
    does not arise.
    """
    wer = _col(rows, wer_key)
    # A NaN confidence is legitimate (the model returned no per-word scores) and
    # is handled deliberately: it sinks to percentile 0. A NaN *WER* is not. It
    # makes `wer >= wer_hi` False, so the row is silently classified NOT a dead
    # zone AND still counted in the denominator of every `mean(flags)` — i.e. an
    # unmeasured condition quietly dilutes the dead-zone rate, which is D1's
    # headline number. Failed rows are supposed to be split out before this
    # (analysis/__init__.split_failures); if one got through, say so.
    if not np.all(np.isfinite(wer)):
        n = int(np.count_nonzero(~np.isfinite(wer)))
        bad = [i for i, v in enumerate(wer) if not np.isfinite(v)][:5]
        raise ValueError(
            f"dead_zone_flags: {n} of {len(wer)} rows have a non-finite WER under "
            f"key {wer_key!r} (row indices {bad}). NaN >= threshold is False, so those rows "
            f"would be counted as 'not a dead zone' and would dilute the "
            f"dead-zone rate without appearing anywhere as missing. Split "
            f"failed rows out first (analysis.split_failures) — a failure "
            f"sentinel is not a measurement. With wer_key='wer_spoke', a "
            f"non-finite value also means a condition where the model spoke on "
            f"NO clip: hold those out and report them as their own category "
            f"(analysis.confidence_gap.classify_conditions), never as a quiet "
            f"False.")
    pct = within_model_conf_percentile(rows)
    return (wer >= wer_hi) & (pct >= conf_pct_hi)


def confidence_wer_shape(rows: Sequence[dict], wer_key: str = "wer") -> dict:
    """
    Characterize the SHAPE of the confidence-vs-WER relationship for one model,
    using within-model confidence percentile (scale-free). A self-aware model has
    a NEGATIVE correlation (more confident -> lower WER). Near-zero or positive
    means confidence fails to track error — the dangerous regime.

    `wer_key`: same estimand argument as `dead_zone_flags` — pass the same-subset
    WER when the confidence being correlated is a subset average.
    """
    wer = _col(rows, wer_key)
    pct = within_model_conf_percentile(rows)
    if len(rows) < 3 or np.allclose(pct, pct[0]) or np.allclose(wer, wer[0]):
        return {"spearman": float("nan"), "n": len(rows)}
    rho, _ = spearmanr(pct, wer)
    return {"spearman": float(rho), "n": len(rows)}


# ---------------------------------------------------------------------------
# Cross-model divergence over the factor space
# ---------------------------------------------------------------------------

def _bins_for(space: FactorSpace, factor_name: str, n_bins: int):
    """Bin edges (continuous) or level list (categorical) for a factor."""
    f = space._by_name[factor_name]
    if f.kind == "continuous":
        edges = np.linspace(float(f.low), float(f.high), n_bins + 1)
        return ("continuous", edges)
    return ("categorical", list(f.levels))


def _region_rows(rows: Sequence[dict], factor: str, kind, spec, b: int):
    """Indices of rows falling in bin b of the factor."""
    vals = np.array([r[factor] for r in rows])
    if kind == "continuous":
        lo, hi = spec[b], spec[b + 1]
        # include right edge in the last bin
        upper = vals <= hi if b == len(spec) - 2 else vals < hi
        return np.where((vals >= lo) & upper)[0], (float(lo), float(hi))
    level = spec[b]
    return np.where(vals == level)[0], level


def find_divergence_regions(tables: dict[str, Sequence[dict]],
                            space: FactorSpace = DEFAULT_FACTOR_SPACE,
                            n_bins: int = 4, wer_hi: float = 0.3,
                            conf_pct_hi: float = 0.6) -> list[dict]:
    """
    Scan every factor axis for the regions where the models diverge MOST — i.e.
    where one model's WER (and dead-zone rate) is much worse than another's in the
    same acoustic cell. Returns regions ranked by the max pairwise WER gap.

    Each region: {factor, span, worse_model, better_model, wer_gap,
    wer_by_model, dead_zone_rate_by_model}. Because dead zones use within-model
    confidence, a region can be flagged even when the worse model's RAW confidence
    is numerically lower than the better model's.
    """
    names = list(tables.keys())
    if len(names) < 2:
        raise ValueError(
            f"find_divergence_regions needs >= 2 model tables to compare, got "
            f"{names}. With one arm every region reports a gap of nothing.")
    # Dead-zone flags are computed over each model's WHOLE distribution (that is
    # where the within-model confidence percentile is meaningful), then aggregated
    # per region. Never recompute the percentile inside a bin — if confidence is
    # constant within the bin it collapses to 0.5 and hides the dead zone.
    full_flags = {m: dead_zone_flags(tables[m], wer_hi, conf_pct_hi) for m in names}
    regions: list[dict] = []
    for factor in space.names:
        kind, spec = _bins_for(space, factor, n_bins)
        n_slots = (len(spec) - 1) if kind == "continuous" else len(spec)
        for b in range(n_slots):
            wer_by: dict[str, float] = {}
            dz_by: dict[str, float] = {}
            n_by: dict[str, int] = {}
            span = None
            for m in names:
                idx, span = _region_rows(tables[m], factor, kind, spec, b)
                n_by[m] = int(len(idx))
                if len(idx) == 0:
                    wer_by[m] = float("nan")
                    dz_by[m] = float("nan")
                    continue
                sub = [tables[m][i] for i in idx]
                wer_by[m] = float(np.mean(_col(sub, "wer")))
                dz_by[m] = float(np.mean(full_flags[m][idx]))
            valid = [m for m in names if np.isfinite(wer_by[m])]
            if len(valid) < 2:
                continue
            # THE RAGGED-COVERAGE TRAP. `wer_gap` subtracts one model's mean over
            # ITS rows in this region from another's mean over ITS rows. If the
            # two arms did not run the same cells here, the difference is a model
            # effect PLUS a coverage effect and there is no way to separate them
            # afterwards — the table is still perfectly well-formed and the
            # ranking still looks meaningful. (A region no arm reached is skipped
            # above; that is legitimately empty, not ragged.)
            sizes = {m: n_by[m] for m in valid}
            if len(set(sizes.values())) != 1:
                raise ValueError(
                    f"ragged coverage in region {factor}={span}: rows per model "
                    f"{sizes}. The WER gap reported for this region would mix a "
                    f"model effect with a coverage effect. Intersect the arms to "
                    f"the cells every model actually ran before comparing "
                    f"(analysis.model_arms.matched_arms does exactly this).")
            worse = max(valid, key=lambda m: wer_by[m])
            better = min(valid, key=lambda m: wer_by[m])
            gap = wer_by[worse] - wer_by[better]
            regions.append({
                "factor": factor, "span": span,
                "worse_model": worse, "better_model": better,
                "wer_gap": gap, "wer_by_model": wer_by,
                "dead_zone_rate_by_model": dz_by,
            })
    regions.sort(key=lambda r: r["wer_gap"], reverse=True)
    return regions


def compare_models(tables: dict[str, Sequence[dict]],
                   space: FactorSpace = DEFAULT_FACTOR_SPACE,
                   n_bins: int = 4, wer_hi: float = 0.3,
                   conf_pct_hi: float = 0.6) -> dict:
    """
    Full cross-model comparison over per-model results tables.

    Per model: overall dead-zone rate, dead-zone rows, and the confidence-vs-WER
    SHAPE (within-model, scale-free). Across models: the factor-space regions where
    they diverge most, ranked. The headline is *where the dead zones differ*, not
    whose confidence number is higher.
    """
    per_model: dict[str, dict] = {}
    for m, rows in tables.items():
        flags = dead_zone_flags(rows, wer_hi, conf_pct_hi)
        per_model[m] = {
            "n": len(rows),
            "dead_zone_rate": float(np.mean(flags)) if len(rows) else float("nan"),
            "dead_zone_rows": [rows[i] for i in np.where(flags)[0]],
            "shape": confidence_wer_shape(rows),
        }
    return {
        "per_model": per_model,
        "divergence_regions": find_divergence_regions(
            tables, space, n_bins, wer_hi, conf_pct_hi),
        "models": list(tables.keys()),
    }
