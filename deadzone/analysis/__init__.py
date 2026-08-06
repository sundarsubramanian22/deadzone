"""
analysis/ — the Track-D findings layers (SPEC §9). Every module in here reads the
SAME master results table and nothing else, so they are independent of each other
and of the acoustic/API stack.

This package __init__ owns the ONE parser for the master-table schema (fixed in
SPEC A.R4.2) so no analysis layer re-derives it and drifts:

    clip_id, condition_name, rt60, snr_db, noise_type, codec, mic_rolloff,
    model, transcript, wer, n_ref, n_sub, n_del, n_ins, n_match,
    mean_conf, utterance_conf, word_confidences (json), edits (json),
    rir_key, rir_rt60_measured, noise_key, failed (bool), error, run_id, ts

THE TRAP THIS FILE EXISTS TO AVOID. A row with ``failed=True`` is a *missing
measurement*, not a prediction: the adapter's failure sentinel carries
``transcript=None`` and ``mean_conf=NaN`` (audio_pipeline._empty_confidence_result),
and the runner writes it with whatever WER a null transcript scores (1.0). Average
those in and you manufacture a fake "loud failure" region — or worse, a fake dead
zone, since NaN confidence sinks to the bottom of the within-model percentile and
a WER of 1.0 sits above every threshold. Failures are therefore split out
EXPLICITLY (`split_failures`) and counted in every report, never dropped silently
and never averaged in.

Deps: stdlib only (+ pandas, lazily, and only for .parquet).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Sequence

# The columns every analysis layer here actually depends on. Checked on load so a
# schema drift fails loudly at the door instead of silently producing NaN tables.
# The FULL frozen schema lives in run_experiment.MASTER_COLUMNS and is not copied
# here — two copies of a "frozen" schema is exactly how one of them goes stale.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "clip_id", "condition_name", "rt60", "snr_db", "noise_type", "codec",
    "mic_rolloff", "model", "wer", "n_ref", "n_sub", "n_del", "n_ins",
    "mean_conf", "edits", "failed",
)

_FLOAT_COLUMNS = ("rt60", "snr_db", "mic_rolloff", "wer", "mean_conf",
                  "utterance_conf", "rir_rt60_measured")
_INT_COLUMNS = ("n_ref", "n_sub", "n_del", "n_ins", "n_match")
_JSON_COLUMNS = ("word_confidences", "edits")


# ---------------------------------------------------------------------------
# Coercion helpers — CSV gives you strings; in-memory rows (tests) give you the
# real types. Everything below tolerates both.
# ---------------------------------------------------------------------------

def as_float(v: Any, default: float = float("nan")) -> float:
    """Best-effort float. Empty string / None / 'nan' -> NaN (never 0.0)."""
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def as_int(v: Any, default: int = 0) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def as_bool(v: Any) -> bool:
    """CSV round-trips booleans as text; accept every form a writer might emit."""
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"true", "t", "yes", "y", "1"}


def coerce_row(row: dict) -> dict:
    """
    Normalize one master-table row's types (returns a new dict).

    NOT optional at the door of an analysis layer. The runner writes a missing
    measurement as `None` (`_finite_or_none`), and model_compare's `_col` does a
    bare `float(row[key])`, which raises TypeError on None — so an in-memory table
    (a --rebuild, a parquet load, a smoke run handing rows straight over) blows up
    where a CSV-loaded one silently works. Coercing at entry makes both paths the
    same table: None/"" -> NaN, never 0.0.
    """
    out = dict(row)
    for k in _FLOAT_COLUMNS:
        if k in out:
            out[k] = as_float(out[k])
    for k in _INT_COLUMNS:
        if k in out:
            out[k] = as_int(out[k])
    if "failed" in out:
        out["failed"] = as_bool(out["failed"])
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_master_table(path: str = "results/master.csv",
                      check_schema: bool = True) -> list[dict]:
    """
    Load the master results table as a list[dict] — the same plain-rows contract
    model_compare.py operates on, so tables flow between layers unchanged.

    `.csv` is read with the stdlib (no pandas needed on the analysis path);
    `.parquet` goes through pandas, which is the only place a heavy dep appears.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"master table not found at {path!r} — run run_experiment.py first "
            f"(SPEC A.R4.2/A.R4.4). Analysis layers never invent data.")
    if path.endswith(".parquet"):
        import pandas as pd                       # lazy: CSV path stays dep-free
        rows = pd.read_parquet(path).to_dict("records")
    else:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    rows = [coerce_row(r) for r in rows]
    if check_schema and rows:
        missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            raise ValueError(
                f"master table {path!r} is missing required columns {missing}. "
                f"The schema is frozen in SPEC A.R4.2 — fix the runner, not this.")
    return rows


# ---------------------------------------------------------------------------
# Failure handling — the explicit split (see module docstring)
# ---------------------------------------------------------------------------

def is_failed_row(row: dict) -> bool:
    """
    A row is a failed MEASUREMENT if the runner marked it, or if the transcript is
    the adapter's null sentinel. Both checks: `failed` is authoritative, but a row
    whose transcript is None never carried a prediction either way.
    """
    if as_bool(row.get("failed", False)):
        return True
    return "transcript" in row and row.get("transcript") is None


def split_failures(rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """
    (usable_rows, failed_rows). Callers MUST report len(failed) — a silent drop is
    as dishonest as a silent average. SPEC A.R4.2: if >2% of rows failed, stop and
    investigate before analysing.
    """
    ok, bad = [], []
    for r in rows:
        (bad if is_failed_row(r) else ok).append(r)
    return ok, bad


def failure_summary(rows: Sequence[dict]) -> dict:
    """Counts + the per-condition breakdown of failures, for every report header."""
    ok, bad = split_failures(rows)
    by_condition: dict[str, int] = {}
    for r in bad:
        key = str(r.get("condition_name", "?"))
        by_condition[key] = by_condition.get(key, 0) + 1
    total_by_condition: dict[str, int] = {}
    for r in rows:
        key = str(r.get("condition_name", "?"))
        total_by_condition[key] = total_by_condition.get(key, 0) + 1
    # Conditions where EVERY row failed have no measurement at all. They must not
    # appear in any WER/confidence table (they would read as WER 1.0), so they get
    # their own list and are named in the report.
    all_failed = sorted(c for c, n in by_condition.items()
                        if n == total_by_condition.get(c, 0))
    return {
        "n_rows": len(rows),
        "n_ok": len(ok),
        "n_failed": len(bad),
        "failure_rate": (len(bad) / len(rows)) if rows else float("nan"),
        "failed_by_condition": by_condition,
        "all_failed_conditions": all_failed,
        "errors": sorted({str(r.get("error")) for r in bad if r.get("error")}),
    }


# ---------------------------------------------------------------------------
# JSON columns
# ---------------------------------------------------------------------------

def parse_edits(row: dict) -> list[tuple[str, str | None, str | None]]:
    """
    The `edits` column -> [(op, ref_word, hyp_word)] for op in {sub, del, ins}.

    The stored column is `classify_errors`'s FULL alignment and does include
    `match` rows; they are filtered out here because every consumer in this
    package is asking "what went wrong", and leaving matches in would make every
    error rate a fraction of the wrong denominator. A caller that needs the
    alignment itself (calibration's word-level labels, SPEC A.R5.8, which walks
    the hypothesis words in order) must read the raw column, not this.
    """
    raw = row.get("edits")
    if raw in (None, "", "[]"):
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    out: list[tuple[str, str | None, str | None]] = []
    for e in raw:
        if isinstance(e, dict):
            op, ref, hyp = e.get("op"), e.get("ref"), e.get("hyp")
        else:
            seq = list(e) + [None, None]
            op, ref, hyp = seq[0], seq[1], seq[2]
        if op in ("sub", "del", "ins"):
            out.append((op, ref, hyp))
    return out


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def split_by_model(rows: Sequence[dict]) -> dict[str, list[dict]]:
    """
    Group rows by model. EVERY confidence computation downstream runs inside one
    of these groups — confidence scales are not comparable across model families
    (model_compare.py's headline caveat), so a cross-model average is meaningless.
    """
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get("model", "unknown")), []).append(r)
    return out
