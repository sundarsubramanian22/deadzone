"""
run_experiment.py — ⋈ JOIN 2: run the grid, emit the master results table
(SPEC §9 Track ⋈2, appendix R4.2).

This is the ONLY place the whole instrument is wired together end to end:

    clean clip  --apply_condition-->  degraded wav  --transcribe_fn-->  hypothesis
                                                    --classify_errors-->  ONE ROW

Everything upstream (audio_pipeline, conditions, design) is already tested in
isolation; everything downstream (D1 confidence gap, D2 fingerprints, D4 sim2real,
L1–L4, the dashboard) reads exactly one artefact — `results/master.parquet` and
its `results/master.csv` mirror. So this file's real job is not "loop over stuff";
it is to make that one artefact CORRECT, COMPLETE and RE-RUNNABLE.

Four things here exist because getting them wrong is expensive and silent:

  1. THE SCHEMA IS FROZEN (`MASTER_COLUMNS`). Column set AND order. Every analysis
     layer indexes it by name; a renamed or reordered column is a silent breakage
     in four modules at once. `_ordered()` raises on any drift.

  2. `rir_rt60_measured` IS NOT `rt60`. `rt60` is what the design ASKED for;
     `AssetLibrary.pick_rir` snaps it to the closest MEASURED asset, so a request
     for 1.0 s can be delivered as 0.45 s. Regress against the delivered value
     (that is the reverb the model actually heard); the requested value is only a
     coordinate in the design. Both are recorded, on purpose.

  3. THE CACHE IS NOT AN OPTIMIZATION. It is the crash contract. A 40k-call run
     is ~45 h of audio and hours of wall clock; dying at 80% with nothing on disk
     costs the whole budget. Every row is appended to `results/cache.jsonl` the
     instant it is produced, and every call checks the cache first. A resumed run
     re-does zero work. The loader also tolerates a torn final line (the exact
     shape a kill mid-write leaves behind).

  4. FAILURES ARE ROWS, NOT SILENCE. A dropped failure turns into a survivorship
     bias exactly where the study is most interesting: the harshest conditions are
     both the most likely to fail AND the ones the dead-zone map is about. Failed
     rows are written with `failed=True`, the error text, and NULL measurements —
     never 0.0 (which reads as a perfect transcription) and never NaN (which
     silently poisons a mean). Downstream MUST filter on `failed`.

Run:
    python3 run_experiment.py --dry-run                  # plan + cost, zero calls
    python3 run_experiment.py --clips smoke --limit 20   # small real run
    python3 run_experiment.py --clips all --workers 12   # the main grid
    python3 run_experiment.py --rebuild                  # master table from cache

Deps: numpy, soundfile (+ pandas/pyarrow for the parquet mirror, optional).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from collections.abc import Mapping          # abc, not typing: used with isinstance
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf

from audio_pipeline import classify_errors, is_failed
from conditions import AssetLibrary, Condition, DiskAssetLibrary, apply_condition
from design import DEFAULT_FACTOR_SPACE


# ============================================================================
# CLIP SUBSETS — fixed once (SPEC appendix R1.1), never chosen ad hoc at runtime
# ============================================================================
# R1.1's DoD is literally "the three subsets are written into run_experiment.py
# as named constants". Why it matters: the AL and Sobol arms are compared against
# each other and against the main grid. If the clip set drifts between runs, a
# change in mean WER is a change in the CORPUS, not in the acoustics — and it is
# unrecoverable after the fact because nothing else records which clips went in.

SMOKE_CLIP = "u02"                      # digits + a name, short, unambiguous (R3)

# 10 clips spanning names / digits / spelled codes / addresses. The AL oracle
# averages WER over these: a single-clip oracle is far too noisy for a GP to fit
# (it happily models the per-clip noise as if it were the response surface).
AL_CLIPS = ["u02", "u05", "u06", "u11", "u17", "u22", "u24", "u33", "u36", "u39"]

# Same ten. Sobol needs thousands of samples, so it pays the clip cost on every
# one of them; the full 40 is reserved for the screen and the main grid.
SOBOL_CLIPS = list(AL_CLIPS)


# ============================================================================
# MASTER TABLE SCHEMA — frozen. Do not add, rename or reorder without updating
# every reader (analysis/*, model_compare, calibration, the dashboard).
# ============================================================================

MASTER_COLUMNS: tuple[str, ...] = (
    # identity of the cell
    "clip_id", "condition_name",
    # the condition's factor coordinates (== DEFAULT_FACTOR_SPACE, see guard below)
    "rt60", "snr_db", "noise_type", "codec", "mic_rolloff",
    "model",
    # what came back
    "transcript",
    # scoring — typed edits, not just a scalar (CLAUDE.md: the fingerprint layer
    # needs sub/del/ins separately)
    "wer", "n_ref", "n_sub", "n_del", "n_ins", "n_match",
    # the headline signal
    "mean_conf", "utterance_conf", "word_confidences", "edits",
    # what was actually DELIVERED (vs requested) — see module docstring, point 2
    "rir_key", "rir_rt60_measured", "noise_key",
    # provenance / failure bookkeeping
    "failed", "error", "run_id", "ts",
)

# Drift guard, same discipline as conditions.py: the factor columns ARE the
# factor space, imported not re-declared. If design.py's factors change, this
# fails loudly at import instead of writing a table whose columns quietly no
# longer match the design that generated them.
assert MASTER_COLUMNS[2:7] == tuple(DEFAULT_FACTOR_SPACE.names), (
    "master-table factor columns drifted from DEFAULT_FACTOR_SPACE "
    f"({MASTER_COLUMNS[2:7]} != {tuple(DEFAULT_FACTOR_SPACE.names)})"
)

# Columns holding a measurement. On a failed row these are ALL None (never 0.0,
# never NaN) so that a downstream mean either errors or drops them — but cannot
# quietly absorb a failure as "the model got it perfectly right".
_MEASUREMENT_COLUMNS = (
    "transcript", "wer", "n_ref", "n_sub", "n_del", "n_ins", "n_match",
    "mean_conf", "utterance_conf", "word_confidences", "edits",
)


# ============================================================================
# COST MODEL (R4.1) — used by --dry-run, never by the run itself
# ============================================================================

# Deepgram pre-recorded Nova-3 list price at time of writing. VENDOR RATES MOVE:
# re-check before quoting a number in the write-up (R4.1's DoD says so).
DEEPGRAM_USD_PER_MIN = 0.0043
EST_CLIP_SECONDS = 5.0          # corpus mean utterance length (SPEC measured ~4 s)

# Which arms cost money. Whisper and Vosk run locally: $0, but ~1–2x realtime on
# CPU, so for those the binding constraint is wall clock, not spend.
BILLED_MODELS = frozenset({"nova-3", "nova-2", "nova-2-drivethru", "nova-3-medical"})

DEFAULT_WORKERS = 8             # R4.1: 8–16, bounded by the account's concurrency

# 16 kHz everywhere: DiskAssetLibrary resamples all assets to one rate and
# apply_condition REFUSES to compose across a rate mismatch (by design).
DEFAULT_FS = 16000


# ============================================================================
# THE MAIN GRID (~60 named conditions, R4.1)
# ============================================================================
# Two blocks, because a full 5-way cross is combinatorially silly and the screen
# exists precisely so we don't run one:
#
#   Block 1 — the acoustic core: reverb x SNR x noise character, channel held
#             clean. This is the block every analysis layer leans on, so it is
#             the one that gets the dense sampling.
#   Block 2 — the channel arm: codec x mic rolloff crossed at three acoustic
#             anchors (benign / mid / harsh), so channel effects can be read
#             both on their own and against a reverb+noise background — which is
#             where the interesting interactions live (SPEC §5's pre-registered
#             rt60 x snr expectation has a channel analogue here).

_CORE_RT60 = (0.2, 0.45, 0.7, 1.0)
_CORE_SNR = (0.0, 10.0, 25.0)
_CORE_NOISE = ("babble", "engine", "road")

# (rt60, snr_db) anchors for the channel arm — deliberately drawn FROM the core
# grid values so block-2 cells share coordinates with block-1 cells and the two
# blocks can be compared without interpolating.
_CHANNEL_ANCHORS = ((0.2, 25.0), (0.45, 10.0), (1.0, 0.0))
_CHANNEL_CODECS = ("none", "g726", "opus-lowrate")
_CHANNEL_ROLLOFF = (0.0, 0.5, 1.0)


def main_grid() -> list[Condition]:
    """The ~60-condition main grid (R4.1). Deterministic order, de-duplicated."""
    conds: list[Condition] = []
    for rt60 in _CORE_RT60:                       # block 1: acoustic core
        for snr in _CORE_SNR:
            for noise in _CORE_NOISE:
                conds.append(Condition(rt60, snr, noise, "none", 0.0))
    for rt60, snr in _CHANNEL_ANCHORS:            # block 2: channel arm
        for codec in _CHANNEL_CODECS:
            for roll in _CHANNEL_ROLLOFF:
                conds.append(Condition(rt60, snr, "babble", codec, roll))
    return dedupe_conditions(conds)


# ---------------------------------------------------------------------------
# THE REBALANCED GRID — allocation driven by a measured pre-grid probe
# ---------------------------------------------------------------------------
#
# WHY main_grid() IS NOT WHAT WE RUN. Before spending the R4 budget we probed
# one clip across the factor space (13 calls). Two measurements changed the
# design:
#
#   1. SNR ALONE BARELY MOVES THE MODEL. At rt60=0.5, codec=none, rolloff=0.3,
#      WER stayed ~0.00 from 0 dB to 25 dB SNR. Even 0 dB babble transcribed
#      perfectly. A grid that spends its budget sweeping SNR at benign channel
#      settings measures a flat surface.
#   2. THE DAMAGE IS AN INTERACTION. At rt60=1.0 + g726 + rolloff=1.0, WER ran
#      0.18-0.46 at EVERY SNR, with confidence still 0.59-0.92 -- and it was
#      NON-MONOTONIC in SNR (WER 0.455 at 25 dB vs 0.273 at 5 dB). That is both
#      the dead-zone signature and a counterintuitive-cell candidate.
#
# main_grid() put 42 of 60 cells at codec="none" and only 2 in the harsh-channel
# region, so it would have spent ~70% of the budget on the flat surface. This
# grid fully crosses the channel factors with reverb instead. main_grid() is kept
# because its test pins it, but interaction_grid() is what R4 runs.
#
# NOTE this does NOT alter the pre-registration. SPEC §5 pre-registered
# rt60 x snr_db and that stands exactly as written -- it gets confirmed or not on
# the real grid. Reallocating where we SAMPLE is not the same as changing what we
# PREDICTED, and the probe is reported alongside the verdict either way.

_IG_RT60 = (0.2, 0.45, 0.7, 1.0)
_IG_SNR = (0.0, 5.0, 10.0, 20.0)
_IG_CODEC = ("none", "g726", "opus-lowrate")
_IG_ROLLOFF = (0.0, 0.5, 1.0)
_IG_NOISE_ARM_RT60 = (0.45, 1.0)
_IG_NOISE_ARM_SNR = (0.0, 10.0)
_IG_NOISE_ARM_CODEC = ("none", "g726")
_IG_NOISE_ARM_ROLLOFF = (0.0, 1.0)


def interaction_grid() -> list[Condition]:
    """
    The grid R4 actually runs: reverb x SNR x codec x rolloff fully crossed on
    babble (144 cells), plus a noise-character arm at the corners (32 cells).

    SNR tops out at 20 dB, not 25: the corpus's measured inherent SNR is
    ~25-28 dB (report/measurements.md), so a 25 dB request under-delivers by
    ~2.5 dB while 20 dB is accurate to ~1 dB. Asking for a condition the capture
    chain cannot physically deliver would put a known-wrong x-coordinate on the
    benign end of every plot.
    """
    conds: list[Condition] = []
    for rt60 in _IG_RT60:                          # block 1: the interaction core
        for snr in _IG_SNR:
            for codec in _IG_CODEC:
                for roll in _IG_ROLLOFF:
                    conds.append(Condition(rt60, snr, "babble", codec, roll))
    for rt60 in _IG_NOISE_ARM_RT60:                # block 2: noise character
        for snr in _IG_NOISE_ARM_SNR:
            for codec in _IG_NOISE_ARM_CODEC:
                for roll in _IG_NOISE_ARM_ROLLOFF:
                    for noise in ("engine", "road"):
                        conds.append(Condition(rt60, snr, noise, codec, roll))
    return dedupe_conditions(conds)


def dedupe_conditions(conds: Sequence[Condition]) -> list[Condition]:
    """Drop repeats, keep first-seen order (blocks 1 and 2 overlap by design)."""
    seen: set[str] = set()
    out: list[Condition] = []
    for c in conds:
        if c.name not in seen:
            seen.add(c.name)
            out.append(c)
    return out


# Any factor absent from a design sample is HELD at its benign end. Stage-2 Sobol
# runs on the screen's survivors only, so its samples are missing the dropped
# factors — they still have to take *some* value in the real world. Holding them
# at the clean end keeps them from injecting variance the design never modelled.
# The write-up must state which factors were held and at what value.
HELD_FACTOR_DEFAULTS: dict = {
    "rt60": 0.2, "snr_db": 25.0, "noise_type": "babble",
    "codec": "none", "mic_rolloff": 0.0,
}


# ============================================================================
# CORPUS LOADING
# ============================================================================

def load_manifest(path: str | Path = "recording_manifest.csv") -> dict[str, str]:
    """clip_id -> normalized ground-truth transcript, read from the manifest.

    TRAP THIS AVOIDS: never type a reference inline anywhere. The manifest is the
    single source of truth for ground truth (R3.1); a transcript typed into a
    script and one typed into the manifest diverge silently and every WER in the
    table inherits a constant offset that looks exactly like an acoustic effect.
    """
    rows: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = (r.get("id") or "").strip()
            if cid:
                rows[cid] = (r.get("ground_truth") or "").strip()
    if not rows:
        raise ValueError(f"manifest {path} has no rows")
    return rows


def load_clip(clip_id: str, recordings_dir: str | Path = "data/recordings",
              target_fs: int = DEFAULT_FS) -> np.ndarray:
    """Load one recording as mono float64 at `target_fs`.

    The composer and the asset library must agree on ONE sample rate —
    apply_condition raises MissingAssetError on a mismatch rather than
    resampling behind your back. Resampling happens here, once, at the edge.
    """
    path = Path(recordings_dir) / f"{clip_id}.wav"
    x, fs = sf.read(str(path), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != target_fs:
        import librosa                                   # lazy: heavy import
        x = librosa.resample(x, orig_sr=fs, target_sr=target_fs)
    return np.asarray(x, dtype=np.float64)


def load_corpus(clip_ids: Sequence[str], recordings_dir: str | Path = "data/recordings",
                target_fs: int = DEFAULT_FS) -> dict[str, np.ndarray]:
    """Preload every clip ONCE.

    TRAP THIS AVOIDS: loading inside the per-cell loop re-reads and re-resamples
    the same 40 files 60x2 times. Worse, it puts disk IO on the thread-pool's
    critical path, so the API concurrency you configured is not the concurrency
    you get.
    """
    return {cid: load_clip(cid, recordings_dir, target_fs) for cid in clip_ids}


def select_clip_ids(spec: str, manifest: Mapping[str, str]) -> list[str]:
    """Resolve a --clips spec ('all' | 'smoke' | 'al' | 'sobol' | 'u02,u05') to ids."""
    named = {"all": sorted(manifest), "smoke": [SMOKE_CLIP],
             "al": list(AL_CLIPS), "sobol": list(SOBOL_CLIPS)}
    ids = named.get(spec) or [s.strip() for s in spec.split(",") if s.strip()]
    missing = [i for i in ids if i not in manifest]
    if missing:
        raise KeyError(f"clip ids not in the manifest: {missing}")
    return ids


# ============================================================================
# APPEND-ONLY JSONL CACHE — the crash contract (R4.2)
# ============================================================================

CacheKey = tuple[str, str, str]         # (clip_id, condition_name, model)


def cache_key(row: Mapping) -> CacheKey:
    return (str(row["clip_id"]), str(row["condition_name"]), str(row["model"]))


class ResultCache:
    """
    Append-only JSONL log of completed rows, keyed by (clip, condition, model).

    Append-only rather than "write the table at the end" because the failure mode
    we are insuring against is the process not reaching the end. One row = one
    line = one flush; a `kill -9` loses at most the row in flight.

    Two deliberate behaviours:
      * LAST WRITE WINS on a repeated key. The file is a log, not a table: a
        re-run of a cell (e.g. after fixing a transient API failure) appends a
        newer row, and the newer row is the one that reaches the master table.
        The older line stays on disk as history.
      * A TORN FINAL LINE IS SURVIVABLE. That is exactly what a crash mid-write
        leaves behind, and a cache that refuses to load after a crash is not a
        crash contract at all. Malformed lines are reported and skipped.
    """

    def __init__(self, path: str | Path = "results/cache.jsonl"):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: dict[CacheKey, dict] = {}
        self._fh = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        bad = 0
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._rows[cache_key(row)] = row      # last write wins
                except (json.JSONDecodeError, KeyError):
                    bad += 1
                    print(f"[cache] WARN: skipping malformed line {i} of "
                          f"{self.path} (torn write from an earlier crash?)")
        if bad:
            print(f"[cache] {bad} malformed line(s) skipped; "
                  f"{len(self._rows)} usable rows loaded")

    # --- lookup ------------------------------------------------------------
    def has(self, key: CacheKey) -> bool:
        return key in self._rows

    def get(self, key: CacheKey) -> dict | None:
        return self._rows.get(key)

    def rows(self) -> list[dict]:
        """Every cached row (arbitrary order; dedup by key already applied)."""
        return list(self._rows.values())

    def __len__(self) -> int:
        return len(self._rows)

    # --- append ------------------------------------------------------------
    def append(self, row: Mapping) -> None:
        """Write one row and FLUSH. Thread-safe (the pool calls this from N threads)."""
        with self._lock:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a", encoding="utf-8")
            # allow_nan=False: keep the log strict JSON so any tool can read it.
            # _finite_or_none() has already mapped non-finite floats to null, so
            # this raises only if a new field sneaks a NaN in — loudly, on the
            # first row, rather than producing a file pandas/jq choke on later.
            self._fh.write(json.dumps(dict(row), allow_nan=False) + "\n")
            self._fh.flush()
            self._rows[cache_key(row)] = dict(row)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


# ============================================================================
# ONE CELL — the unit of work
# ============================================================================

_PEAK_CEILING = 0.99


class RunStats:
    """Thread-safe counters for things that are worth reporting but are not rows."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counts[name] = self.counts.get(name, 0) + n

    def get(self, name: str) -> int:
        return self.counts.get(name, 0)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_or_none(x) -> float | None:
    """NaN/inf -> None. 'No confidence available' must be NULL, not NaN: a NaN in
    a float column silently propagates through every mean downstream, while a
    null is visibly missing."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _ordered(row: Mapping) -> dict:
    """Project a row onto MASTER_COLUMNS, in order. Raises on any schema drift —
    this is the guard that keeps four downstream readers from breaking quietly."""
    missing = [c for c in MASTER_COLUMNS if c not in row]
    extra = [c for c in row if c not in MASTER_COLUMNS]
    if missing or extra:
        raise KeyError(f"row schema drift: missing={missing} extra={extra}")
    return {c: row[c] for c in MASTER_COLUMNS}


def write_degraded_wav(path: str | Path, x: np.ndarray, fs: int) -> bool:
    """
    Write the degraded clip as 16-bit PCM. Returns True if the peak guard fired.

    TRAP THIS AVOIDS: mix_at_snr adds noise ON TOP of speech calibrated to the
    speech's own active-region RMS, so the sum routinely peaks above 1.0 — and a
    PCM_16 write hard-clips anything above full scale. That clipping is a
    broadband nonlinearity applied AFTER the codec stage, i.e. a degradation we
    never modelled, strongest exactly where SNR is lowest. It would read as
    "low SNR destroys the model" and be partly our own writer. So we rescale by a
    single uniform gain, which changes level only and leaves the SNR ratio, the
    reverb and the codec artefacts untouched.
    """
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    fired = peak > _PEAK_CEILING
    if fired:
        x = x * (_PEAK_CEILING / peak)
    sf.write(str(path), x.astype(np.float32), int(fs), subtype="PCM_16")
    return fired


def blank_row(clip_id: str, condition: Condition, model: str, run_id: str) -> dict:
    """A schema-complete row with every measurement NULL. Built in column order."""
    row = {
        "clip_id": clip_id,
        "condition_name": condition.name,
        "rt60": float(condition.rt60),
        "snr_db": float(condition.snr_db),
        "noise_type": condition.noise_type,
        "codec": condition.codec,
        "mic_rolloff": float(condition.mic_rolloff),
        "model": model,
    }
    row.update({c: None for c in _MEASUREMENT_COLUMNS})
    row.update({
        "rir_key": None, "rir_rt60_measured": None, "noise_key": None,
        "failed": False, "error": None, "run_id": run_id, "ts": _now_iso(),
    })
    return row


def run_one(clip_id: str, audio: np.ndarray, ref_text: str, condition: Condition,
            assets: AssetLibrary, fs: int, transcribe_fn: Callable[[str], dict],
            model: str, run_id: str, stats: RunStats | None = None) -> dict:
    """
    One (clip, condition, model) cell -> one master-table row.

    degrade -> temp wav -> transcribe -> score. The temp wav is thrown away on
    purpose: apply_condition is seeded from `condition.name`, so the degraded
    audio is byte-reproducible for free, and 40k wavs is ~45 h of disk for data
    we can regenerate in milliseconds. The TRANSCRIPT is what gets cached.
    """
    row = blank_row(clip_id, condition, model, run_id)
    try:
        # Resolve first so the DELIVERED ingredients are recorded even if a later
        # stage blows up. This repeats the resolve inside apply_condition — it is
        # deterministic and cheap, and having rir_rt60_measured on a failed row is
        # worth more than the microseconds.
        resolved = assets.resolve(condition)
        row["rir_key"] = resolved.rir.key
        row["rir_rt60_measured"] = _finite_or_none(resolved.rir.rt60)
        row["noise_key"] = resolved.noise.key

        degraded = apply_condition(np.asarray(audio, dtype=np.float64),
                                   condition, assets, fs)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, f"{clip_id}__{condition.name}.wav")
            if write_degraded_wav(path, degraded, fs) and stats is not None:
                stats.bump("peak_guard")
            result = transcribe_fn(path)
    except Exception as e:                  # noqa: BLE001 — grid resilience by design
        # Same posture as transcribe_deepgram's own broad catch: one bad cell out
        # of 40,000 must not kill the run. The exception text lands in the row, so
        # nothing is lost and nothing is hidden.
        row["failed"] = True
        row["error"] = f"{type(e).__name__}: {e}"
        if stats is not None:
            stats.bump("failed")
        return _ordered(row)

    if is_failed(result):                   # the adapter's skip-me sentinel
        row["failed"] = True
        row["error"] = result.get("error") or "adapter returned the failure sentinel"
        if stats is not None:
            stats.bump("failed")
        return _ordered(row)

    scored = classify_errors(ref_text, result["transcript"])
    counts = scored["counts"]
    row.update({
        "transcript": result["transcript"],
        "wer": float(scored["wer"]),
        "n_ref": int(scored["n_ref"]),
        "n_sub": int(counts["sub"]),
        "n_del": int(counts["del"]),
        "n_ins": int(counts["ins"]),
        "n_match": int(counts["match"]),
        "mean_conf": _finite_or_none(result.get("mean_conf")),
        "utterance_conf": _finite_or_none(result.get("utterance_conf")),
        # JSON strings, not nested objects: the schema has to survive a CSV
        # round-trip and a parquet column, and both want a flat scalar here.
        "word_confidences": json.dumps([float(c) for c in
                                        (result.get("word_confidences") or [])]),
        "edits": json.dumps(scored["edits"]),
    })
    return _ordered(row)


# ============================================================================
# THE GRID RUNNER
# ============================================================================

def resolve_models(models: Sequence[str] | Mapping[str, Callable[[str], dict]]
                   ) -> dict[str, Callable[[str], dict]]:
    """Model names -> adapter callables via model_compare's registry (one line per
    model, already the single place arms are declared). A mapping passes straight
    through, which is how tests inject a fake. Imported lazily so --dry-run needs
    no ASR dependency at all."""
    if isinstance(models, Mapping):
        return dict(models)
    from model_compare import get_model                  # lazy: no SDKs for a dry run
    return {name: get_model(name) for name in models}


def run_grid(clips: Mapping[str, np.ndarray], refs: Mapping[str, str],
             conditions: Sequence[Condition],
             models: Sequence[str] | Mapping[str, Callable[[str], dict]],
             assets: AssetLibrary, fs: int = DEFAULT_FS,
             workers: int = DEFAULT_WORKERS, cache: ResultCache | None = None,
             run_id: str | None = None, limit: int | None = None,
             progress_every: int = 25, stats: RunStats | None = None) -> dict:
    """
    Run every (clip, condition, model) cell and return the rows for THIS plan.

    Cache hits are returned unchanged, including their original `run_id`/`ts`:
    a resumed run must not relabel measurements it did not make, or the freeze
    manifest (R4.6) can no longer say which run produced which number.

    Row order is the PLAN's order, not completion order — threads finish out of
    sequence, and a table whose row order changes run to run makes every diff and
    every "did anything move?" check useless.
    """
    model_fns = resolve_models(models)
    run_id = run_id or f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    stats = stats if stats is not None else RunStats()

    plan: list[tuple[str, Condition, str]] = [
        (clip_id, cond, model)
        for clip_id in clips for cond in conditions for model in model_fns
    ]
    if limit is not None:
        plan = plan[:limit]

    rows: list[dict | None] = [None] * len(plan)
    todo: list[tuple[int, tuple[str, Condition, str]]] = []
    for i, (clip_id, cond, model) in enumerate(plan):
        hit = cache.get((clip_id, cond.name, model)) if cache is not None else None
        if hit is not None:
            rows[i] = _ordered(hit)
            stats.bump("cached")
        else:
            todo.append((i, (clip_id, cond, model)))

    print(f"[grid] {len(plan)} cells | {stats.get('cached')} cached | "
          f"{len(todo)} to run | {len(model_fns)} model(s) | {workers} workers")

    t0 = time.time()
    done = 0

    def _work(item):
        i, (clip_id, cond, model) = item
        row = run_one(clip_id, clips[clip_id], refs[clip_id], cond, assets, fs,
                      model_fns[model], model, run_id, stats=stats)
        # Append INSIDE the worker, the instant the row exists — not in a batch
        # at the end. The whole point is that a crash loses one row, not N.
        if cache is not None:
            cache.append(row)
        return i, row

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            for i, row in pool.map(_work, todo):
                rows[i] = row
                done += 1
                if progress_every and done % progress_every == 0:
                    rate = done / max(time.time() - t0, 1e-6)
                    eta = (len(todo) - done) / max(rate, 1e-9)
                    print(f"[grid] {done}/{len(todo)}  "
                          f"failed={stats.get('failed')}  "
                          f"{rate:.1f} cells/s  eta {eta/60:.1f} min")

    out = [r for r in rows if r is not None]
    n_failed = sum(1 for r in out if r["failed"])
    fail_rate = n_failed / len(out) if out else 0.0

    print(f"[grid] done: {len(out)} rows in {time.time()-t0:.1f}s "
          f"({stats.get('cached')} from cache, {len(todo)} freshly run)")
    # NEVER quiet about failures — R4.2 says count them, and >2% means stop.
    print(f"[grid] FAILED ROWS: {n_failed} / {len(out)} ({100*fail_rate:.2f}%)")
    if fail_rate > 0.02:
        print("[grid] WARNING: failure rate above 2% — investigate BEFORE "
              "analysing (R4.4). Sample errors:")
        for r in [r for r in out if r["failed"]][:5]:
            print(f"         {r['clip_id']} {r['condition_name']} "
                  f"{r['model']}: {r['error']}")
    if stats.get("peak_guard"):
        print(f"[grid] peak guard rescaled {stats.get('peak_guard')} clip(s) "
              f"before writing (uniform gain; SNR unchanged)")

    return {"rows": out, "run_id": run_id, "n_cached": stats.get("cached"),
            "n_called": len(todo), "n_failed": n_failed,
            "fail_rate": fail_rate, "stats": stats}


# ============================================================================
# THE design.py BRIDGE — wer_fn(sample: dict) -> float
# ============================================================================

def make_grid_wer_fn(clips: Mapping[str, np.ndarray], refs: Mapping[str, str],
                     assets: AssetLibrary, transcribe_fn: Callable[[str], dict],
                     fs: int = DEFAULT_FS, *, model: str = "nova-3",
                     cache: ResultCache | None = None, run_id: str | None = None,
                     held: Mapping | None = None,
                     stats: RunStats | None = None) -> Callable[[dict], float]:
    """
    Wrap the real chain into the EXACT callable shape design.py wants:
    ``wer_fn(sample: dict) -> float``, returning the MEAN WER over the clip set.

    So the design layer plugs straight in with no adapter:

        wer_fn = make_grid_wer_fn(clips, refs, assets, transcribe_fn)
        screen = design.screen_factors(DEFAULT_FACTOR_SPACE, wer_fn)
        sobol  = design.sobol_indices(space.subset(screen["survivors"]), wer_fn, N=128)

    MEAN OVER CLIPS, not one clip: with ~8.5 words per utterance a single-clip WER
    moves in ~12-point steps, and both the Sobol variance decomposition and the GP
    surrogate would spend their budget modelling that quantization as if it were
    the response surface.

    Factors missing from `sample` (Stage-2 runs on the screen's survivors only)
    are filled from `held` / HELD_FACTOR_DEFAULTS — the benign end. Say which in
    the write-up.

    FAILURES ARE EXCLUDED FROM THE MEAN, NOT COUNTED AS ZERO. If EVERY clip fails
    for a sample there is no evidence at all, and we raise rather than return a
    fabricated number: a wer_fn that quietly returns 0.0 (or NaN) in the harshest
    corner of the space is precisely how a sensitivity analysis ends up reporting
    that reverb doesn't matter.
    """
    held_all = dict(HELD_FACTOR_DEFAULTS)
    held_all.update(dict(held or {}))
    clip_ids = list(clips)
    run_id = run_id or f"werfn-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

    def wer_fn(sample: dict) -> float:
        full = dict(held_all)
        full.update({k: v for k, v in sample.items() if k in held_all})
        condition = Condition.from_sample(full)

        wers: list[float] = []
        errors: list[str] = []
        for clip_id in clip_ids:
            key = (clip_id, condition.name, model)
            row = cache.get(key) if cache is not None else None
            if row is None:
                row = run_one(clip_id, clips[clip_id], refs[clip_id], condition,
                              assets, fs, transcribe_fn, model, run_id, stats=stats)
                if cache is not None:
                    cache.append(row)
            if row["failed"]:
                errors.append(f"{clip_id}: {row['error']}")
            else:
                wers.append(float(row["wer"]))

        if not wers:
            raise RuntimeError(
                f"every clip failed for condition {condition.name!r} — refusing to "
                f"invent a WER for it (first errors: {errors[:3]})"
            )
        if errors:
            print(f"[wer_fn] {condition.name}: {len(errors)}/{len(clip_ids)} clips "
                  f"failed, averaging the remaining {len(wers)}")
        return float(np.mean(wers))

    return wer_fn


# ============================================================================
# OUTPUT — parquet + csv mirror
# ============================================================================

def write_master(rows: Sequence[Mapping], results_dir: str | Path = "results",
                 basename: str = "master") -> dict[str, str]:
    """
    Write results/<basename>.parquet AND results/<basename>.csv.

    The CSV is not a convenience copy, it is the grep-able / dashboard-readable
    mirror (R4.2). Parquet is preferred for analysis because CSV loses dtypes —
    `failed` comes back as the string "True", which is truthy for BOTH values and
    would silently include every failed row in an analysis that forgot to cast.
    If pyarrow/pandas is unavailable we degrade to CSV only AND SAY SO, rather
    than crashing a run whose expensive part (the API calls) already succeeded.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = [_ordered(r) for r in rows]

    csv_path = out_dir / f"{basename}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(MASTER_COLUMNS), extrasaction="raise")
        w.writeheader()
        w.writerows(ordered)
    paths = {"csv": str(csv_path)}

    pq_path = out_dir / f"{basename}.parquet"
    try:
        import pandas as pd                              # lazy + optional
        pd.DataFrame(ordered, columns=list(MASTER_COLUMNS)).to_parquet(
            pq_path, index=False)
        paths["parquet"] = str(pq_path)
    except Exception as e:                               # noqa: BLE001
        print(f"[master] parquet NOT written ({type(e).__name__}: {e}). "
              f"Wrote {csv_path} only — `pip install pyarrow` for the typed table.")
    return paths


# ============================================================================
# DRY RUN — the full call plan + a cost estimate, with ZERO calls
# ============================================================================

def is_billed(model: str) -> bool:
    """True for cloud arms that cost money per minute of audio."""
    return model in BILLED_MODELS


def build_plan(clip_ids: Sequence[str], conditions: Sequence[Condition],
               models: Sequence[str], cache: ResultCache | None = None,
               limit: int | None = None) -> dict:
    """Enumerate every planned call and price it. Touches no audio and no assets —
    so --dry-run works on a machine that has neither the recordings nor the RIRs
    downloaded yet, which is exactly when you want to check the budget."""
    calls = [(c, cond, m) for c in clip_ids for cond in conditions for m in models]
    if limit is not None:
        calls = calls[:limit]

    cached = [c for c in calls
              if cache is not None and cache.has((c[0], c[1].name, c[2]))]
    remaining = [c for c in calls
                 if cache is None or not cache.has((c[0], c[1].name, c[2]))]

    per_model = {m: sum(1 for c in remaining if c[2] == m) for m in models}
    billed_calls = sum(n for m, n in per_model.items() if is_billed(m))
    minutes = billed_calls * EST_CLIP_SECONDS / 60.0

    return {
        "clip_ids": list(clip_ids), "conditions": list(conditions),
        "models": list(models), "calls": calls,
        "n_total": len(calls), "n_cached": len(cached), "n_remaining": len(remaining),
        "per_model_remaining": per_model, "n_billed": billed_calls,
        "est_audio_minutes": minutes,
        "est_cost_usd": minutes * DEEPGRAM_USD_PER_MIN,
    }


def format_plan(plan: Mapping, verbose: bool = False) -> str:
    """Human-readable call plan. `verbose` prints every single call."""
    L = ["=" * 74, "DRY RUN — call plan (no audio touched, no API called)", "=" * 74,
         f"clips      : {len(plan['clip_ids'])}  {plan['clip_ids']}",
         f"conditions : {len(plan['conditions'])}",
         f"models     : {plan['models']}", ""]
    L.append("condition list:")
    for i, c in enumerate(plan["conditions"], 1):
        L.append(f"  {i:>3}. {c.name}")
    L += ["", f"total cells          : {plan['n_total']}",
          f"  already cached     : {plan['n_cached']}",
          f"  calls to make      : {plan['n_remaining']}"]
    for m, n in plan["per_model_remaining"].items():
        L.append(f"      {m:<20} {n:>7}  {'(billed)' if is_billed(m) else '(local, $0)'}")
    L += ["",
          f"billed calls         : {plan['n_billed']}",
          f"est. audio           : {plan['est_audio_minutes']:.1f} min "
          f"(@ {EST_CLIP_SECONDS:.0f}s/clip)",
          f"est. cost            : ${plan['est_cost_usd']:.2f} "
          f"(@ ${DEEPGRAM_USD_PER_MIN}/min — RE-CHECK vendor pricing, R4.1)",
          "",
          "note: local arms (whisper/vosk) are $0 but ~1-2x realtime on CPU; the",
          "      binding constraint there is wall clock, not spend.", "=" * 74]
    if verbose:
        L.append("full call list:")
        for clip_id, cond, model in plan["calls"]:
            L.append(f"  {clip_id}  {cond.name}  {model}")
        L.append("=" * 74)
    return "\n".join(L)


# ============================================================================
# CLI
# ============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deadzone JOIN-2 grid runner -> results/master.{parquet,csv}")
    p.add_argument("--manifest", default="recording_manifest.csv")
    p.add_argument("--recordings", default="data/recordings")
    p.add_argument("--data-root", default="data",
                   help="root scanned by DiskAssetLibrary (rirs/, noise/)")
    p.add_argument("--rir-subdir", default="rirs",
                   help="use 'rirs_sim' for the D4 sim2real arm")
    p.add_argument("--results", default="results")
    p.add_argument("--basename", default="master",
                   help="output stem: results/<basename>.{parquet,csv}")
    p.add_argument("--clips", default="all",
                   help="all | smoke | al | sobol | comma-separated ids")
    p.add_argument("--models", default="nova-3",
                   help="comma-separated model ids from model_compare.MODEL_REGISTRY")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--fs", type=int, default=DEFAULT_FS)
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N cells of the plan (smoke runs)")
    p.add_argument("--grid", default="interaction", choices=("interaction", "main"),
                   help="interaction = rebalanced R4 grid (default); main = original 60-cell")
    p.add_argument("--dry-run", action="store_true",
                   help="print the full call plan + cost estimate and exit")
    p.add_argument("--verbose", action="store_true",
                   help="with --dry-run, list every individual call")
    p.add_argument("--rebuild", action="store_true",
                   help="rebuild the master table from the cache; make no calls")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    manifest = load_manifest(args.manifest)
    clip_ids = select_clip_ids(args.clips, manifest)
    conditions = {"interaction": interaction_grid, "main": main_grid}[args.grid]()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cache = ResultCache(Path(args.results) / "cache.jsonl")

    plan = build_plan(clip_ids, conditions, models, cache=cache, limit=args.limit)
    print(format_plan(plan, verbose=args.verbose))
    if args.dry_run:
        return 0

    if args.rebuild:
        # Reconstruct the table for THIS plan from cached rows only. The point of
        # an append-only cache is that the expensive artefact survives the process
        # that produced it.
        rows, missing = [], 0
        for clip_id, cond, model in plan["calls"]:
            hit = cache.get((clip_id, cond.name, model))
            if hit is None:
                missing += 1
            else:
                rows.append(_ordered(hit))
        print(f"[rebuild] {len(rows)} cached rows, {missing} planned cells missing")
        paths = write_master(rows, args.results, args.basename)
        print(f"[rebuild] wrote {paths}")
        return 0

    assets = DiskAssetLibrary(root=args.data_root, target_fs=args.fs,
                              rir_subdir=args.rir_subdir)
    clips = load_corpus(clip_ids, args.recordings, args.fs)
    refs = {cid: manifest[cid] for cid in clip_ids}

    res = run_grid(clips, refs, conditions, models, assets, fs=args.fs,
                   workers=args.workers, cache=cache, limit=args.limit)
    cache.close()

    paths = write_master(res["rows"], args.results, args.basename)
    print(f"[master] {len(res['rows'])} rows -> {paths}")
    print(f"[master] run_id={res['run_id']}  failed={res['n_failed']} "
          f"({100*res['fail_rate']:.2f}%)")
    # Non-zero exit on a bad run so a `nohup`'d overnight job is not mistaken for
    # a clean one (R4.4's gate: failure rate must be under 2%).
    return 1 if res["fail_rate"] > 0.02 else 0


if __name__ == "__main__":
    sys.exit(main())
