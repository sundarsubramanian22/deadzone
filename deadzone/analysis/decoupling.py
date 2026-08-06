"""
analysis/decoupling.py — L3 on real audio, multi-clip (SPEC A.R5.9).

THE QUESTION. A voice agent rides on two parallel streams off the same
microphone: the *lexical* stream (the transcript) and the *paralinguistic*
stream (energy, pitch, spectral shape — what drives endpointing, barge-in,
emotion). Acoustic degradation corrupts the waveform both streams are read
from. Do they degrade at the SAME rate, or do they DECOUPLE? If WER collapses
while the features hold, a paralinguistics-driven agent is blind to the fact
that its ASR has already failed. If the features collapse first, they are a
usable early-warning signal. Either answer is a result.

WHY THIS FILE EXISTS ON TOP OF `analysis/layers.run_l3_decoupling`.
`layers.run_l3_decoupling` scores ONE clip: one raw baseline, one ladder of
degraded wavs, one WER curve. That is the right unit of machinery and it is
synthetic-validated (`test_layers.py`). But the real sweep is 5 clips x 8
levels x 2 factors, and with n=5 a single clip's drift curve is not a
measurement — it is a sample. This module is the multi-clip layer over it:
per-clip curves, averaged, WITH THE SPREAD REPORTED, plus the transcription
pass that produces the lexical curve on the very same audio.

THE FOUR TRAPS THIS MODULE EXISTS TO AVOID
──────────────────────────────────────────
1. THE BASELINE MUST BE THE RAW RECORDING, AT THE SWEEP'S SAMPLE RATE.
   `apply_condition` always applies an RIR and always mixes noise, so there is
   no clean `Condition` and the mildest rung of the sweep is NOT a baseline —
   measuring drift against it understates the low end of the curve and moves
   the half-degradation level the whole verdict is read off (SPEC A.R5.9).
   But the second half of that trap is subtler and unwritten: the raw captures
   in `data/recordings/` are **48 kHz** while every composed sweep clip is
   **16 kHz**. `centroid`, `rolloff` and `flatness` are computed over the whole
   band to Nyquist, so a 48 kHz baseline differs from a 16 kHz degraded clip by
   a large constant that has nothing to do with degradation. We therefore use
   `<clip>__baseline_raw.wav` — the raw capture resampled to 16 kHz by
   `make_audio_sets.py`, which is the raw signal at the sweep's rate — and pass
   `allow_same_dir=True` deliberately, because that guard is aimed at the
   composed-condition mistake, not at this one.

2. THE TWO CURVES MUST COME OFF THE SAME AUDIO AT THE SAME LEVELS. The master
   grid sampled rt60 in {0.2,0.45,0.7,1.0} and snr_db in {0,5,10,20}; the sweep
   ladders are 8 levels each and mostly do not coincide. Reading WER off a
   "nearby" grid cell would silently compare a feature curve and a lexical
   curve measured on different signals. So we transcribe the sweep wavs
   themselves (85 calls, cached).

3. SEVERITY, NOT NUMERIC ORDER. `compare_degradation_rates` walks its curves in
   the order given and reports the level at which each first reaches half its
   range. For `rt60`, degradation increases with the number; for `snr_db` it
   increases as the number FALLS (20 dB -> 0 dB). Sorting snr ascending would
   run the ladder backwards and report a half-degradation "level" that is the
   mirror image of the real one, with `leads` inverted. Every curve here is
   ordered by SEVERITY (`_severity_sorted`), and the reported half-levels stay
   in the factor's own units.

4. n=5 IS A SPREAD, NOT A POINT. Averaging five clips into one curve and
   quoting one half-degradation level hides whether the clips agree. We fit the
   comparison per clip as well as on the mean, and report the per-clip spread of
   half-levels and of `leads`, plus an explicit monotonicity check on each mean
   drift curve. A feature whose clips disagree about which curve leads is
   reported as unstable rather than quietly averaged into a verdict.

5. A FLAT CURVE HAS NO HALF-DEGRADATION LEVEL. This is the trap that actually
   fired on the real sweep, and it is the nastiest one here because it produces
   a confident, precise, entirely fabricated number.
   `compare_degradation_rates` MIN-MAX normalizes both curves before finding
   where each crosses 0.5. Min-max normalization is scale-free by design — it
   maps whatever range it is given onto [0,1] and cannot tell a curve that
   collapsed from 0.0 to 1.0 WER from one that wandered from 0.000 to 0.054.
   On this sweep the lexical curve does the latter: mean WER peaks at 0.054
   (snr) and 0.047 (rt60), and three of five clips are IDENTICALLY ZERO at
   every rung. Normalizing that turns 5 points of sampling noise into a full-
   range "degradation curve" and reports a crisp half-degradation level for it.
   The number is real arithmetic on meaningless input. So every curve here is
   checked for degeneracy FIRST (`_curve_degeneracy`), and a curve that never
   leaves the floor is refused a half-level rather than assigned a fake one.
   The reason the curve is flat is itself the finding, and it is a known one:
   SPEC B.3's pre-grid probe found "SNR alone barely moves the model... the
   damage is an interaction", and `results/master.csv` agrees — at rt60=0.45,
   snr=0 the same five clips score WER 0.971, but BOTH sweeps hold the other
   factor at its benign end (the rt60 ladder pins snr=20 dB, the snr ladder
   pins rt60=0.2), so each one walks along a flat EDGE of the response surface
   and never enters the interaction region where the damage lives.

Deps: numpy, scipy (via paralinguistic), soundfile (lazily). Transcription is
only needed for the `transcribe` phase and is cached to
`results/l3_transcripts.jsonl`.

    ./.venv/bin/python -m deadzone.analysis.decoupling --transcribe   # phase 1 (API)
    ./.venv/bin/python -m deadzone.analysis.decoupling                # phase 2 (offline)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deadzone.analysis.layers import (                       # noqa: E402
    DEFAULT_L3_FEATURES, SweepPoint, run_l3_decoupling,
)
from deadzone.paralinguistic import (                        # noqa: E402
    FEATURE_KEYS, extract_features_from_path, feature_drift,
    compare_degradation_rates,
)

__all__ = [
    "SWEEP_ROOT", "TRANSCRIPT_CACHE", "FACTOR_DIRECTION",
    "load_sweep_index", "baseline_path", "severity_sort_key", "_severity_sorted",
    "load_transcript_cache", "transcribe_sweep",
    "clip_curves", "run_factor_decoupling", "run_all",
    "verdict_sentence", "format_report",
]

SWEEP_ROOT = Path("results/audio/sweep")
HARSH_ROOT = Path("results/audio/sweep/harsh")
HARSH_INDEX = Path("results/audio/sweep/harsh_index.json")
TRANSCRIPT_CACHE = Path("results/l3_transcripts.jsonl")
OUT_JSON = Path("results/l3_decoupling.json")
OUT_TXT = Path("results/l3_decoupling.txt")

# +1: degradation increases with the level. -1: degradation increases as the
# level FALLS. This single sign is what keeps trap 3 from happening.
FACTOR_DIRECTION: dict[str, int] = {"rt60": +1, "snr_db": -1}

# How the factor reads in prose, given its direction.
_FACTOR_PROSE: dict[str, str] = {"rt60": "increasing reverb",
                                 "snr_db": "falling SNR"}


def _axis(factor: str) -> str:
    """The underlying physical axis of a sweep key.

    Round-2 sweeps ladder the SAME axis against different pinned backgrounds,
    so their keys are compound: `rt60@snr0`, `snr_db@g726_roll1`. Direction and
    prose belong to the axis (`rt60`), not to the key — resolving them off the
    full key would silently fall back to +1 and run an snr ladder backwards.
    """
    return factor.split("@", 1)[0]


# ---------------------------------------------------------------------------
# ROUND 2 — the harsh-region sweeps (the ones whose lexical curve can move)
# ---------------------------------------------------------------------------
# Round 1 laddered each factor with every OTHER factor pinned at its benign end,
# which walks a flat EDGE of the response surface: WER never left the floor
# (range 0.047 / 0.054) so no lexical half-degradation level was definable and
# every verdict was refused. These four ladders pin the background in the HARSH
# region instead, chosen from `results/master.csv` so the endpoints are KNOWN IN
# ADVANCE to straddle the floor-to-ceiling transition rather than sit above or
# below it. The `master_wer` field is that prior evidence, recorded here so the
# design is auditable and not a guess.
#
# A and D are deliberately the SAME ladders as round 1 with only the pinned
# background changed, which makes the pair a controlled contrast: same axis,
# same clips, same levels, benign vs harsh surroundings.
RT60_LEVELS = [0.2, 0.31, 0.43, 0.54, 0.66, 0.77, 0.89, 1.0]
SNR_LEVELS = [20.0, 17.1, 14.3, 11.4, 8.6, 5.7, 2.9, 0.0]

# The same 5 clips round 1 used, so the benign-vs-harsh contrast holds the
# speaker and the sentences fixed and only the acoustics change.
SWEEP_CLIPS = ["u02", "u06", "u17", "u24", "u36"]

HARSH_SWEEPS: dict[str, dict] = {
    # A — reverb alone at harsh noise. No codec, no rolloff: the cleanest read
    # on rt60, and the direct counterpart to round 1's flat `rt60` sweep.
    "rt60@snr0": {
        "axis": "rt60", "levels": RT60_LEVELS,
        "fixed": dict(snr_db=0.0, noise_type="babble", codec="none",
                      mic_rolloff=0.0),
        "context": "at 0 dB SNR, babble, no codec, no rolloff",
        "master_wer": {0.2: 0.050, 0.45: 0.971, 0.7: 0.640, 1.0: 1.000},
    },
    # B — reverb across the opus channel. Largest measured range of any rt60
    # ladder (0.971) and the D1 dead-zone region; D2 found opus -> DELETIONS.
    "rt60@opus_roll1": {
        "axis": "rt60", "levels": RT60_LEVELS,
        "fixed": dict(snr_db=10.0, noise_type="babble", codec="opus-lowrate",
                      mic_rolloff=1.0),
        "context": "at 10 dB SNR, babble, opus-lowrate, rolloff 1.0",
        "master_wer": {0.2: 0.029, 0.45: 0.595, 0.7: 0.507, 1.0: 1.000},
    },
    # C — reverb across the g726 channel. D2 found g726 -> SUBSTITUTIONS, so B
    # vs C asks whether the paralinguistic stream tracks the two channel
    # failure modes differently.
    "rt60@g726_roll0.5": {
        "axis": "rt60", "levels": RT60_LEVELS,
        "fixed": dict(snr_db=0.0, noise_type="babble", codec="g726",
                      mic_rolloff=0.5),
        "context": "at 0 dB SNR, babble, g726, rolloff 0.5",
        "master_wer": {0.2: 0.082, 0.45: 0.640, 0.7: 0.836, 1.0: 1.000},
    },
    # D — the SNR axis where WER actually moves. Chosen because it is the only
    # snr ladder in master.csv that is MONOTONE across severity, so the lexical
    # curve traverses smoothly instead of jumping. Counterpart to round 1's
    # flat `snr_db` sweep, where rms trended perfectly against a flat WER.
    "snr_db@g726_roll1": {
        "axis": "snr_db", "levels": SNR_LEVELS,
        "fixed": dict(rt60=0.7, noise_type="babble", codec="g726",
                      mic_rolloff=1.0),
        "context": "at rt60 0.7 s, babble, g726, rolloff 1.0",
        "master_wer": {20.0: 0.102, 10.0: 0.579, 5.0: 0.766, 0.0: 0.978},
    },
}

# Clips must agree this often on which curve leads before we call the feature's
# verdict stable. 3/5 is a bare majority; below it we say the clips disagree.
STABILITY_QUORUM = 0.8

# TRAP 5. WER must move at least this far across a sweep before a
# "half-degradation level" read off its min-max-normalized curve means
# anything. 0.10 = the model has to have lost a tenth of the words somewhere on
# the ladder. Below that we are normalizing sampling noise into a full-range
# curve and reporting a threshold for it.
MIN_LEXICAL_RANGE = 0.10

# A feature drift curve is only allowed to carry a quoted threshold if it
# actually trends with severity. Spearman of drift against severity RANK;
# below this the curve is a wobble and its half-level is not a threshold.
MIN_TREND_RHO = 0.70


# ---------------------------------------------------------------------------
# Sweep layout
# ---------------------------------------------------------------------------

def load_sweep_index(root: str | Path = SWEEP_ROOT) -> dict[str, list[dict]]:
    """`results/audio/sweep/index.json` — every wav's clip / factor / level."""
    p = Path(root) / "index.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found — run `./.venv/bin/python scripts/make_audio_sets.py --sweep`")
    return json.loads(p.read_text())


def baseline_path(factor: str, clip_id: str,
                  root: str | Path = SWEEP_ROOT) -> str:
    """TRAP 1. The paralinguistic baseline for a clip.

    `<clip>__baseline_raw.wav` is the RAW capture resampled to the sweep's
    16 kHz — not a composed condition (which already carries an RIR and mixed
    noise), and not the 48 kHz original (whose band-limited spectral features
    are not comparable with a 16 kHz clip's).
    """
    return str(Path(root) / factor / f"{clip_id}__baseline_raw.wav")


def severity_sort_key(factor: str):
    """Order levels by INCREASING DEGRADATION, not by numeric value (trap 3)."""
    sign = FACTOR_DIRECTION.get(_axis(factor), +1)
    return lambda lv: sign * float(lv)


def _severity_sorted(factor: str, entries: Sequence[Mapping]) -> list[dict]:
    return sorted((dict(e) for e in entries),
                  key=lambda e: severity_sort_key(factor)(e["level"]))


def make_harsh_sweep_audio(clips: Sequence[str] | None = None,
                           root: str | Path = HARSH_ROOT,
                           index_path: str | Path = HARSH_INDEX,
                           fs: int = 16000) -> dict:
    """Generate the round-2 harsh-region sweep audio.

    Reuses `make_audio_sets.py`'s composition logic by importing the same
    primitives it does (`load_clip` / `apply_condition` / `write_degraded_wav`)
    rather than editing that shared file. Writes
    `harsh/<sweep_key>/<clip>__<axis>_<level>.wav` plus the per-clip
    `__baseline_raw.wav` (the raw capture at the sweep's 16 kHz — trap 1) and a
    `harsh_index.json` in the same shape as `make_audio_sets`'s index.

    Pure DSP, no API calls, deterministic: `apply_condition` seeds its noise
    crop from the condition name.
    """
    import soundfile as sf
    from deadzone.conditions import Condition, DiskAssetLibrary, apply_condition
    from scripts.run_experiment import load_clip, write_degraded_wav

    clips = list(clips) if clips is not None else list(SWEEP_CLIPS)
    assets = DiskAssetLibrary(root="data", target_fs=fs)
    root = Path(root)
    index: dict[str, list] = {}
    n = 0

    for key, spec in HARSH_SWEEPS.items():
        axis = spec["axis"]
        d = root / key
        d.mkdir(parents=True, exist_ok=True)
        entries = []
        for clip_id in clips:
            audio = load_clip(clip_id, target_fs=fs)
            base = d / f"{clip_id}__baseline_raw.wav"
            sf.write(base, audio, fs, subtype="PCM_16")
            for lvl in spec["levels"]:
                kw = dict(spec["fixed"])
                kw[axis] = lvl
                cond = Condition(**kw)
                y = apply_condition(audio, cond, assets, fs)
                fn = d / f"{clip_id}__{axis}_{lvl:g}.wav"
                write_degraded_wav(fn, y, fs)
                entries.append({"clip_id": clip_id, "factor": key,
                                "axis": axis, "level": float(lvl),
                                "file": str(fn), "baseline": str(base),
                                "condition_name": cond.name,
                                "context": spec["context"]})
                n += 1
        index[key] = entries

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(index_path).write_text(json.dumps(index, indent=2))
    return {"files": n, "sweeps": list(index), "index": str(index_path)}


def load_harsh_index(path: str | Path = HARSH_INDEX) -> dict[str, list[dict]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found — run `python -m deadzone.analysis.decoupling --make-audio`")
    return json.loads(p.read_text())


def load_all_indexes(sweep_root: str | Path = SWEEP_ROOT,
                     harsh_index: str | Path = HARSH_INDEX) -> dict[str, list[dict]]:
    """Round-1 (benign-background) + round-2 (harsh-background) sweeps.

    Keys never collide: round 1 uses the bare axis (`rt60`), round 2 the
    compound key (`rt60@snr0`), so both rounds coexist in one report and the
    benign-vs-harsh contrast is readable off a single table.
    """
    merged = dict(load_sweep_index(sweep_root))
    if Path(harsh_index).is_file():
        merged.update(load_harsh_index(harsh_index))
    return merged


def sweep_clips(index: Mapping[str, Sequence[Mapping]], factor: str) -> list[str]:
    seen: list[str] = []
    for e in index[factor]:
        if e["clip_id"] not in seen:
            seen.append(e["clip_id"])
    return seen


def sweep_levels(index: Mapping[str, Sequence[Mapping]], factor: str) -> list[float]:
    """The factor's ladder, in severity order, de-duplicated."""
    out: list[float] = []
    for e in _severity_sorted(factor, index[factor]):
        lv = float(e["level"])
        if lv not in out:
            out.append(lv)
    return out


# ---------------------------------------------------------------------------
# Phase 1 — transcribe the sweep (the ONLY part that touches the API)
# ---------------------------------------------------------------------------

def _cache_key(factor: str, clip_id: str, level: float | None) -> str:
    """Baselines are keyed by clip, not by path.

    `<clip>__baseline_raw.wav` is written identically into BOTH factor
    directories, so keying it by path would buy two identical transcriptions of
    the same bytes for every clip — 5 wasted calls out of an 85-call budget.
    """
    return f"baseline::{clip_id}" if level is None else \
        f"{factor}::{clip_id}::{level:g}"


def load_transcript_cache(path: str | Path = TRANSCRIPT_CACHE) -> dict[str, dict]:
    """Append-only JSONL, last write per key wins."""
    p = Path(path)
    if not p.is_file():
        return {}
    cache: dict[str, dict] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "key" in rec:
            cache[rec["key"]] = rec
    return cache


def _append_cache(rec: Mapping, path: str | Path = TRANSCRIPT_CACHE) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _load_env_key() -> str:
    """DEEPGRAM_API_KEY from the environment, else from the gitignored `.env`.

    Never printed, never returned to a caller that logs. Read only here.
    """
    key = os.environ.get("DEEPGRAM_API_KEY")
    if key:
        return key
    env = Path(".env")
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPGRAM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError("DEEPGRAM_API_KEY not set (env var or .env)")


def _work_items(index: Mapping[str, Sequence[Mapping]]) -> list[dict]:
    """Every distinct transcription the sweep needs, deduplicated by cache key."""
    items: dict[str, dict] = {}
    for factor, entries in index.items():
        for e in _severity_sorted(factor, entries):
            k = _cache_key(factor, e["clip_id"], float(e["level"]))
            items.setdefault(k, {"key": k, "factor": factor,
                                 "clip_id": e["clip_id"],
                                 "level": float(e["level"]),
                                 "path": e["file"]})
        for clip_id in sweep_clips(index, factor):
            k = _cache_key(factor, clip_id, None)
            items.setdefault(k, {"key": k, "factor": "baseline",
                                 "clip_id": clip_id, "level": None,
                                 "path": baseline_path(factor, clip_id)})
    return list(items.values())


def transcribe_sweep(index: Mapping[str, Sequence[Mapping]] | None = None,
                     manifest: Mapping[str, str] | None = None,
                     cache_path: str | Path = TRANSCRIPT_CACHE,
                     workers: int = 4, max_calls: int = 150,
                     transcribe_fn=None, dry_run: bool = False) -> dict:
    """Transcribe every sweep wav and score it, caching so a re-run is free.

    TRAP 2: the lexical curve must be measured on the SAME audio at the SAME
    levels as the feature curve, so we transcribe the sweep clips rather than
    reading a nearby cell out of `results/master.csv` (whose rt60/snr ladders
    do not line up with the sweep's).

    `max_calls` is a hard stop, not a suggestion — two other agents may be
    hitting the same account and Deepgram throttled hard past ~10k calls.
    """
    from concurrent.futures import ThreadPoolExecutor

    from deadzone.audio_pipeline import classify_errors, is_failed
    from scripts.run_experiment import load_manifest

    index = index if index is not None else load_sweep_index()
    manifest = manifest if manifest is not None else load_manifest()

    cache = load_transcript_cache(cache_path)
    todo = [it for it in _work_items(index) if it["key"] not in cache]

    if len(todo) > max_calls:
        raise RuntimeError(
            f"{len(todo)} transcriptions needed but max_calls={max_calls}; "
            "refusing to blow the budget")
    if dry_run:
        return {"cached": len(cache), "would_call": len(todo), "calls": 0}

    if transcribe_fn is None:
        from deadzone.audio_pipeline import transcribe_deepgram
        api_key = _load_env_key()

        def transcribe_fn(path: str) -> dict:
            return transcribe_deepgram(path, api_key=api_key)

    calls = 0
    failures: list[str] = []

    def _one(item: dict) -> dict:
        res = transcribe_fn(item["path"])
        ref = manifest[item["clip_id"]]
        rec = dict(item)
        if is_failed(res):
            rec.update(failed=True, transcript=None, wer=None,
                       error=str(res.get("error", "transcription failed")))
            return rec
        scored = classify_errors(ref, res["transcript"])
        confs = res.get("word_confidences") or []
        rec.update(failed=False, transcript=res["transcript"],
                   wer=float(scored["wer"]), n_ref=int(scored["n_ref"]),
                   counts=dict(scored["counts"]),
                   mean_conf=float(np.mean(confs)) if confs else None,
                   n_word_conf=len(confs))
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for rec in pool.map(_one, todo):
                _append_cache(rec, cache_path)
                cache[rec["key"]] = rec
                calls += 1
                if rec.get("failed"):
                    failures.append(rec["key"])

    return {"cached": len(cache) - calls, "calls": calls,
            "failures": failures, "n_failed": len(failures)}


def wer_lookup(cache: Mapping[str, Mapping], factor: str,
               clip_id: str, levels: Sequence[float]) -> list[float]:
    """WER at each level for one clip, in the order given. Raises on a gap."""
    out = []
    for lv in levels:
        k = _cache_key(factor, clip_id, float(lv))
        rec = cache.get(k)
        if rec is None:
            raise KeyError(f"no transcription cached for {k} — run --transcribe")
        if rec.get("failed") or rec.get("wer") is None:
            raise ValueError(f"transcription for {k} failed: {rec.get('error')}")
        out.append(float(rec["wer"]))
    return out


# ---------------------------------------------------------------------------
# Phase 2 — the analysis (offline)
# ---------------------------------------------------------------------------

@dataclass
class ClipCurves:
    """One clip's paired curves over the severity-ordered ladder."""
    clip_id: str
    levels: list[float]
    wer: list[float]
    drift: dict[str, list[float]]        # feature key -> drift-from-raw curve
    clean_summary: dict[str, float]
    baseline_wer: float | None


def clip_curves(factor: str, clip_id: str, levels: Sequence[float],
                index: Mapping[str, Sequence[Mapping]],
                cache: Mapping[str, Mapping],
                feature_keys: Sequence[str] = DEFAULT_L3_FEATURES,
                root: str | Path = SWEEP_ROOT) -> ClipCurves:
    """Extract one clip's feature-drift curves and its WER curve.

    Drift is measured against `<clip>__baseline_raw.wav` — trap 1.
    """
    mine = [e for e in index[factor] if e["clip_id"] == clip_id]
    # `by_level` is a dict keyed by level, so a second index entry for a level
    # already present silently REPLACES the first and the lookup below still
    # finds a file for every rung — the curve comes out the right length, built
    # from whichever wav the index happened to list last. Two sweeps written into
    # one directory (or a re-generated ladder appended rather than replacing) is
    # all it takes. Assert the count identity instead.
    seen_levels: dict[float, int] = {}
    for e in mine:
        lv = float(e["level"])
        seen_levels[lv] = seen_levels.get(lv, 0) + 1
    dupes = sorted(((lv, n) for lv, n in seen_levels.items() if n > 1))
    if dupes:
        raise ValueError(
            f"duplicate sweep entries for clip {clip_id!r} on factor "
            f"{factor!r}: {len(mine)} index entries for {len(seen_levels)} "
            f"distinct levels (repeats at {dupes}). Keying them by level means a "
            f"later entry silently replaces an earlier one and the drift curve "
            f"is built from whichever wav was listed last — same length, same "
            f"shape, different audio. Regenerate the sweep index "
            f"(`--make-audio`) rather than appending to it.")

    by_level = {float(e["level"]): e["file"] for e in mine}
    paths = []
    for lv in levels:
        if float(lv) not in by_level:
            raise KeyError(f"{clip_id} has no {factor}={lv} wav in the index")
        paths.append(by_level[float(lv)])

    # Round-2 entries carry their baseline explicitly (each harsh sweep has its
    # own directory); round-1 entries fall back to the conventional location.
    base_path = next((e["baseline"] for e in mine if e.get("baseline")),
                     None) or baseline_path(factor, clip_id, root)
    clean = extract_features_from_path(base_path)
    deg = [extract_features_from_path(p) for p in paths]

    drift = {k: [float(v) for v in feature_drift(clean, deg, key=k)]
             for k in feature_keys}

    base_rec = cache.get(_cache_key(factor, clip_id, None))
    base_wer = None
    if base_rec and not base_rec.get("failed"):
        base_wer = float(base_rec["wer"]) if base_rec.get("wer") is not None else None

    return ClipCurves(clip_id=clip_id, levels=[float(v) for v in levels],
                      wer=wer_lookup(cache, factor, clip_id, levels),
                      drift=drift,
                      clean_summary={k: float(clean["summary"][k])
                                     for k in feature_keys},
                      baseline_wer=base_wer)


def _monotonicity(curve: Sequence[float]) -> dict:
    """Is this drift curve actually a ladder, or is it noise? (trap 4)

    A curve read off 5 clips can wobble. `spearman` against severity rank says
    whether it trends at all; `n_sign_flips` says how ragged it is. A curve with
    many flips still yields a half-degradation level, but that number should not
    be quoted as if it were a threshold.
    """
    v = np.asarray(curve, dtype=float)
    if len(v) < 3:
        return {"spearman_vs_severity": float("nan"), "n_sign_flips": 0,
                "monotone": False}
    from scipy.stats import spearmanr
    rho = float(spearmanr(np.arange(len(v)), v).statistic)
    d = np.diff(v)
    nz = d[np.abs(d) > 1e-12]
    flips = int(np.sum(np.sign(nz[1:]) != np.sign(nz[:-1]))) if len(nz) > 1 else 0
    return {"spearman_vs_severity": rho, "n_sign_flips": flips,
            "monotone": bool(flips == 0)}


def _curve_degeneracy(curve: Sequence[float], min_range: float,
                      label: str) -> dict:
    """TRAP 5. Did this curve move enough for a half-level to mean anything?

    `compare_degradation_rates` min-max normalizes, which is scale-free: it
    cannot distinguish a curve that collapsed across its whole range from one
    that wandered by a rounding error, and it will report a crisp
    half-degradation level for both. So we ask the question normalization
    throws away — how far did the RAW curve actually travel? — and refuse the
    threshold if the answer is "barely".
    """
    v = np.asarray(curve, dtype=float)
    rng = float(v.max() - v.min()) if v.size else 0.0
    degenerate = rng < min_range
    return {
        "range": rng, "min": float(v.min()) if v.size else 0.0,
        "max": float(v.max()) if v.size else 0.0,
        "min_range_required": float(min_range),
        "degenerate": bool(degenerate),
        "why": (f"{label} never leaves the floor over this sweep "
                f"(range {rng:.4g} < {min_range:g}); a half-degradation level "
                f"read off its normalized curve would be sampling noise "
                f"rescaled to full range, so none is quoted"
                if degenerate else
                f"{label} moves by {rng:.4g} across the sweep, enough for a "
                f"half-degradation level to be meaningful"),
    }


def _half_quality(levels: Sequence[float], curve: Sequence[float],
                  half: float, label: str) -> dict:
    """TRAP 6. Is a half-degradation level READABLE off this curve at all?

    `paralinguistic._half_level` scans for the first rung where the normalized
    curve reaches 0.5 and linearly interpolates to it. That is correct only for
    a curve that STARTS BELOW HALF AND RISES — the shape a degradation curve is
    assumed to have. Hand it a curve that starts at its maximum and wanders
    down (which real drift curves do: f0 drift under g726 peaks at the MILDEST
    rung) and the interpolation runs backwards through a near-zero denominator
    and extrapolates far outside the ladder — rt60 = 9.02 on a ladder that
    stops at 1.0. The arithmetic is right; the shape assumption is not.

    So we check the three ways the number can be meaningless:
      * `flat`        — the curve has no range to be half of;
      * `starts_high` — it is already past half at the first rung, so "first
                        crossing" describes nothing;
      * `out_of_range`— the interpolated level escaped the swept ladder.
    A curve that simply never reaches half is CENSORED, not broken: the honest
    reading is "held past the end of the ladder", and the level is reported as
    a bound rather than a threshold.
    """
    v = np.asarray(curve, dtype=float)
    lvl = np.asarray(levels, dtype=float)
    lo, hi = float(v.min()), float(v.max())
    flat = hi <= lo
    norm = (v - lo) / (hi - lo) if not flat else np.zeros_like(v)
    starts_high = bool(norm[0] >= 0.5) and not flat
    censored = bool(not np.any(norm >= 0.5)) and not flat
    a, b = float(lvl.min()), float(lvl.max())
    out_of_range = not (a - 1e-9 <= half <= b + 1e-9)
    ok = not (flat or starts_high or out_of_range)
    why = ("curve is flat — no range to be half of" if flat else
           f"curve is already past half at the first rung "
           f"(normalized {norm[0]:.2f}), so 'first crossing' describes nothing"
           if starts_high else
           f"interpolated level {half:.3g} escaped the swept ladder "
           f"[{a:g}, {b:g}]" if out_of_range else
           f"never reaches half within the ladder — CENSORED, held past "
           f"{b:g}" if censored else
           f"{label} half-level is readable")
    return {"ok": bool(ok), "flat": bool(flat), "starts_high": starts_high,
            "censored": censored, "out_of_range": bool(out_of_range),
            "ladder": [a, b], "why": why}


def _spread(values: Sequence[float]) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"n": int(v.size), "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def run_factor_decoupling(factor: str,
                          index: Mapping[str, Sequence[Mapping]] | None = None,
                          cache: Mapping[str, Mapping] | None = None,
                          feature_keys: Sequence[str] = DEFAULT_L3_FEATURES,
                          gap_tol: float = 0.2,
                          root: str | Path = SWEEP_ROOT) -> dict:
    """The L3 verdict for one swept factor, averaged over the clips WITH SPREAD.

    Per feature: `compare_degradation_rates` on the clip-averaged curves (the
    headline numbers), the same fit repeated per clip (the spread), a
    monotonicity check on the mean drift curve, and a stability flag saying
    whether the clips agree about which curve leads.
    """
    index = index if index is not None else load_sweep_index()
    cache = cache if cache is not None else load_transcript_cache()

    unknown = [k for k in feature_keys if k not in FEATURE_KEYS]
    if unknown:
        raise ValueError(f"unknown feature keys {unknown} "
                         f"(available: {list(FEATURE_KEYS)})")

    levels = sweep_levels(index, factor)        # severity order — trap 3
    clips = sweep_clips(index, factor)
    # The pinned background this ladder was run against (round 2 only), so the
    # verdict sentence stays self-contained.
    ctx = next((e.get("context") for e in index[factor] if e.get("context")), None)
    curves = [clip_curves(factor, c, levels, index, cache, feature_keys, root)
              for c in clips]

    # Every number below is a mean or an sd over these matrices, and both fail
    # QUIETLY: a NaN anywhere propagates through `mean(axis=0)` into the curve,
    # through `compare_degradation_rates`, and out into the verdict as a
    # printed `nan` half-level; and with a single clip `std(axis=0, ddof=1)` is
    # NaN for every rung, so the per-rung spread — the honest part at n=5 —
    # silently disappears while the mean curve still looks like a measurement.
    if len(curves) < 2:
        raise ValueError(
            f"factor {factor!r} has {len(curves)} clip(s) in the sweep index; "
            f"at least 2 are needed or every reported sd (wer_curve_std, "
            f"drift_curve_std, the per-clip half-level spread) is nan while the "
            f"mean curves still print as measurements. n=5 is the designed "
            f"sweep width — check the index was generated for all of "
            f"{SWEEP_CLIPS}.")

    wer_matrix = np.array([c.wer for c in curves], dtype=float)   # (n_clip, n_lvl)
    if not np.all(np.isfinite(wer_matrix)):
        bad = np.argwhere(~np.isfinite(wer_matrix))[:5]
        raise ValueError(
            f"factor {factor!r}: {int(np.count_nonzero(~np.isfinite(wer_matrix)))} "
            f"of {wer_matrix.size} WER values are not finite (e.g. "
            f"[clip, level] {bad.tolist()}). The lexical curve is the thing the "
            f"feature curves are raced against, so a NaN here makes the "
            f"degeneracy check, the half-level and the verdict all nan-valued "
            f"without raising.")
    mean_wer = wer_matrix.mean(axis=0)

    # TRAP 5, asked once for the whole factor: did WER move at all?
    lex_deg = _curve_degeneracy(mean_wer, MIN_LEXICAL_RANGE, "WER")

    per_feature: dict[str, dict] = {}
    for key in feature_keys:
        drift_matrix = np.array([c.drift[key] for c in curves], dtype=float)
        if not np.all(np.isfinite(drift_matrix)):
            bad = np.argwhere(~np.isfinite(drift_matrix))[:5]
            raise ValueError(
                f"factor {factor!r}, feature {key!r}: "
                f"{int(np.count_nonzero(~np.isfinite(drift_matrix)))} of "
                f"{drift_matrix.size} drift values are not finite (e.g. "
                f"[clip, level] {bad.tolist()}). `feature_drift` returns NaN "
                f"when a feature is undefined on a clip (unvoiced audio has no "
                f"f0), and it would flow straight through the mean curve into "
                f"the spearman, the half-level and the printed verdict as a "
                f"quiet `nan` rather than as a missing measurement.")
        mean_drift = drift_matrix.mean(axis=0)

        with warnings.catch_warnings():       # constant curve -> nan spearman
            warnings.simplefilter("ignore")
            verdict = compare_degradation_rates(levels, mean_drift, mean_wer,
                                                gap_tol=gap_tol)

        mono = _monotonicity(mean_drift)
        # The feature side's own guard: a drift curve is allowed to carry a
        # quoted threshold only if it actually trends with severity. Scale-free
        # relative range is reported alongside for context.
        rho_sev = mono["spearman_vs_severity"]
        clean_mean = float(np.mean([c.clean_summary[key] for c in curves]))
        rel_range = (float(mean_drift.max() - mean_drift.min()) / abs(clean_mean)
                     if clean_mean else float("nan"))
        trend_reliable = bool(np.isfinite(rho_sev) and rho_sev >= MIN_TREND_RHO)
        feat_q = _half_quality(levels, mean_drift,
                               verdict["feature_half_level"], key)
        lex_q = _half_quality(levels, mean_wer,
                              verdict["lexical_half_level"], "WER")

        per_clip = []
        for c, row in zip(curves, drift_matrix):
            # A clip whose WER is constant (3 of 5 are identically zero here)
            # has no lexical curve to compare against; min-max would hand it a
            # fabricated half-level. Exclude it from the vote and SAY SO.
            clip_lex_flat = bool(np.ptp(np.asarray(c.wer, dtype=float))
                                 < MIN_LEXICAL_RANGE)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pc = compare_degradation_rates(levels, row, c.wer,
                                                   gap_tol=gap_tol)
            except Exception as e:                      # degenerate clip curve
                per_clip.append({"clip_id": c.clip_id, "error": str(e),
                                 "lexical_flat": clip_lex_flat})
                continue
            rec = {
                "clip_id": c.clip_id, "spearman": pc["spearman"],
                "max_abs_gap": pc["max_abs_gap"],
                "coupled": pc["coupled"],
                "feature_half_level": pc["feature_half_level"],
                "lexical_half_level": pc["lexical_half_level"],
                "lexical_flat": clip_lex_flat,
                "wer_range": float(np.ptp(np.asarray(c.wer, dtype=float))),
            }
            # Only a clip with a real lexical curve gets a vote on `leads`.
            if not clip_lex_flat:
                rec["leads"] = pc["leads"]
            per_clip.append(rec)

        n_flat_clips = sum(1 for p in per_clip if p.get("lexical_flat"))
        leads_votes = Counter(p["leads"] for p in per_clip if "leads" in p)
        top_lead, top_n = (leads_votes.most_common(1)[0]
                           if leads_votes else (None, 0))
        n_voting = sum(leads_votes.values())
        agreement = (top_n / n_voting) if n_voting else 0.0

        verdict.update({
            "drift_curve_mean": [float(v) for v in mean_drift],
            "drift_curve_std": [float(v) for v in drift_matrix.std(axis=0, ddof=1)],
            "clean_value_mean": clean_mean,
            "monotonicity": mono,
            "lexical_degeneracy": lex_deg,
            "feature_rel_range": rel_range,
            "trend_reliable": trend_reliable,
            # TRAP 6: is each half-level readable off the curve it came from?
            "feature_half_quality": feat_q,
            "lexical_half_quality": lex_q,
            # The threshold is only quotable if BOTH curves are real AND both
            # half-levels are actually readable off them.
            "half_levels_quotable": bool(trend_reliable
                                         and not lex_deg["degenerate"]
                                         and feat_q["ok"] and lex_q["ok"]),
            "n_clips_lexically_flat": n_flat_clips,
            "context": ctx,
            "per_clip": per_clip,
            "per_clip_feature_half_spread": _spread(
                [p["feature_half_level"] for p in per_clip
                 if "feature_half_level" in p]),
            "per_clip_lexical_half_spread": _spread(
                [p["lexical_half_level"] for p in per_clip
                 if "lexical_half_level" in p]),
            "leads_votes": dict(leads_votes),
            "leads_agreement": float(agreement),
            # Stable requires: the clips that HAVE a lexical curve agree, they
            # agree with the mean fit, and the thresholds were quotable at all.
            "stable": bool(agreement >= STABILITY_QUORUM
                           and top_lead == verdict["leads"]
                           and trend_reliable
                           and not lex_deg["degenerate"]),
            "statement": "",       # filled below, needs the factor direction
        })
        verdict["statement"] = verdict_sentence(factor, key, verdict)
        per_feature[key] = verdict

    # Factor-level verdict.
    #
    # When WER never left the floor there is nothing to race the features
    # against: no feature can "lead" a curve that never moved. The honest
    # reading is that the paralinguistic stream degrades while transcription
    # holds — reported as its own verdict rather than squeezed into the
    # feature/lexical/together vocabulary, which presumes both curves moved.
    reliable_keys = [k for k, v in per_feature.items() if v["trend_reliable"]]
    if lex_deg["degenerate"]:
        factor_lead = "feature" if reliable_keys else "neither"
        pool = reliable_keys or list(per_feature)
        # With WER flat, the strongest claim is the feature that trends most
        # cleanly with severity, not the biggest (meaningless) half-level gap.
        primary = max(pool, key=lambda k:
                      (per_feature[k]["trend_reliable"],
                       per_feature[k]["monotonicity"]["spearman_vs_severity"]))
        stable_keys = reliable_keys
    else:
        # Only features whose clips agree with each other get a vote.
        stable_keys = [k for k, v in per_feature.items() if v["stable"]]

        def _sep(k: str) -> float:
            v = per_feature[k]
            return abs(v["feature_half_level"] - v["lexical_half_level"])

        if stable_keys:
            votes = Counter(per_feature[k]["leads"] for k in stable_keys)
            factor_lead, _ = votes.most_common(1)[0]
            # Headline: among features voting with the majority AND stable, the
            # one with the largest half-level separation.
            pool = [k for k in stable_keys
                    if per_feature[k]["leads"] == factor_lead] or stable_keys
            primary = max(pool, key=_sep)
        else:
            # TRAP 6, second half. Ranking by "largest half-level separation"
            # with nothing stable actively SELECTS THE MOST BROKEN CURVE: the
            # feature whose half-level extrapolated furthest outside the ladder
            # wins by construction. (On rt60@g726_roll0.5 that picked f0 at
            # rt60=9.02 as the headline.) Fall back to the curve that is most
            # readable instead — quotable first, then trending, then cleanest
            # trend — and say plainly that nothing was stable.
            quotable = [k for k, v in per_feature.items()
                        if v["half_levels_quotable"]]
            trending = [k for k, v in per_feature.items() if v["trend_reliable"]]
            pool = quotable or trending or list(per_feature)
            votes = Counter(per_feature[k]["leads"] for k in pool)
            factor_lead = votes.most_common(1)[0][0] if quotable or trending \
                else "undetermined"
            primary = max(pool, key=lambda k: (
                per_feature[k]["half_levels_quotable"],
                per_feature[k]["trend_reliable"],
                per_feature[k]["monotonicity"]["spearman_vs_severity"]))

    n_decoupled = sum(1 for v in per_feature.values() if not v["coupled"])
    base_wers = [c.baseline_wer for c in curves if c.baseline_wer is not None]

    return {
        "layer": "L3",
        "factor": factor,
        "axis": _axis(factor),
        "context": ctx,
        "round": 2 if "@" in factor else 1,
        "master_wer_prior": HARSH_SWEEPS.get(factor, {}).get("master_wer"),
        "direction": FACTOR_DIRECTION.get(_axis(factor), +1),
        "severity_order": "increasing level" if FACTOR_DIRECTION.get(_axis(factor), 1) > 0
                          else "decreasing level (degradation rises as it falls)",
        "levels": [float(v) for v in levels],
        "clips": clips,
        "n_clips": len(clips),
        "wer_curve_mean": [float(v) for v in mean_wer],
        "wer_curve_std": [float(v) for v in wer_matrix.std(axis=0, ddof=1)],
        "wer_per_clip": {c.clip_id: c.wer for c in curves},
        "baseline_wer_mean": float(np.mean(base_wers)) if base_wers else None,
        "baseline_wer_per_clip": {c.clip_id: c.baseline_wer for c in curves},
        "per_feature": per_feature,
        "primary_feature": primary,
        "headline": per_feature[primary],
        "factor_leads": factor_lead,
        "lexical_degeneracy": lex_deg,
        "lexical_floor": bool(lex_deg["degenerate"]),
        "n_features_trend_reliable": len(
            [k for k, v in per_feature.items() if v["trend_reliable"]]),
        "thresholds_quotable": bool(per_feature[primary]["half_levels_quotable"]),
        "verdict": ("LEXICAL FLOOR — features move, WER does not"
                    if lex_deg["degenerate"] else
                    "NO SUPPORTABLE THRESHOLD — WER moves, feature curves unreadable"
                    if not per_feature[primary]["half_levels_quotable"] else
                    "DECOUPLED" if not per_feature[primary]["coupled"]
                    else "COUPLED"),
        "n_features_decoupled": n_decoupled,
        "n_features": len(per_feature),
        "n_features_stable": len(stable_keys),
        "statement": per_feature[primary]["statement"],
        "notes": [
            "baseline is the RAW capture resampled to the sweep's 16 kHz "
            "(<clip>__baseline_raw.wav), NOT a composed near-clean condition "
            "(which already carries an RIR and mixed noise) and NOT the 48 kHz "
            "original (whose band-limited spectral features are not comparable "
            "with a 16 kHz clip's)",
            "WER is measured on the SAME wavs as the features, at the SAME "
            "levels — not read off a nearby master-grid cell",
            "curves are ordered by SEVERITY, so snr_db runs 20 dB -> 0 dB; "
            "half-degradation levels stay in the factor's own units",
            f"n={len(clips)} clips: half-levels are quoted with their per-clip "
            "spread, and a feature whose clips disagree about which curve leads "
            "is marked unstable rather than averaged into a verdict",
            "coupling is a valid result and is reported as such",
        ],
    }


def run_all(index: Mapping[str, Sequence[Mapping]] | None = None,
            cache: Mapping[str, Mapping] | None = None,
            feature_keys: Sequence[str] = DEFAULT_L3_FEATURES,
            gap_tol: float = 0.2, root: str | Path = SWEEP_ROOT) -> dict:
    """Every swept factor in the index."""
    index = index if index is not None else load_all_indexes(root)
    cache = cache if cache is not None else load_transcript_cache()
    return {
        "layer": "L3",
        "spec": "A.R5.9 — paralinguistic vs lexical decoupling",
        "feature_keys": list(feature_keys),
        "gap_tol": gap_tol,
        "factors": {f: run_factor_decoupling(f, index, cache, feature_keys,
                                             gap_tol, root)
                    for f in index},
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def verdict_sentence(factor: str, key: str, v: Mapping) -> str:
    """The A.R5.9 deliverable sentence, direction-aware.

    `analysis.layers._l3_statement` hardcodes "under increasing <factor>", which
    is wrong for `snr_db`: degradation there rises as the number FALLS. This
    says "falling SNR" and keeps the half-levels in dB.
    """
    axis = _axis(factor)
    prose = _FACTOR_PROSE.get(axis, f"increasing {axis}")
    ctx = v.get("context")
    if ctx:
        prose = f"{prose} ({ctx})"
    unit = " dB" if axis == "snr_db" else ""
    f_half, l_half = v["feature_half_level"], v["lexical_half_level"]
    gap, rho = v["max_abs_gap"], v["spearman"]
    n_clips = v["per_clip_feature_half_spread"]["n"]
    tail = (f" (spearman {rho:.2f}, max normalized gap {gap:.2f}, n="
            f"{n_clips} clips)")

    # TRAP 5. WER never left the floor: there is no lexical half-degradation
    # level to quote and no race for a feature to win. Say that, rather than
    # reporting the number min-max normalization would happily invent.
    deg = v.get("lexical_degeneracy") or {}
    if deg.get("degenerate"):
        lo, hi, rng = deg["min"], deg["max"], deg["range"]
        mono = v["monotonicity"]
        flat = v.get("n_clips_lexically_flat", 0)
        if v.get("trend_reliable"):
            return (
                f"Under {prose}, WER never leaves the floor "
                f"(mean {lo:.3f} -> {hi:.3f}, range {rng:.3f} across the whole "
                f"ladder; {flat}/{n_clips} clips are flat outright), so no "
                f"lexical half-degradation level is defined. The {key} feature "
                f"meanwhile drifts monotonically with severity (spearman vs "
                f"severity {mono['spearman_vs_severity']:.2f}, "
                f"{mono['n_sign_flips']} sign flips, drift "
                f"{v['drift_curve_mean'][0]:.4g} -> "
                f"{v['drift_curve_mean'][-1]:.4g}): the paralinguistic stream "
                f"degrades while transcription holds — decoupled, but in the "
                f"direction where a feature-based monitor would alarm on audio "
                f"the ASR is still handling correctly.")
        return (
            f"Under {prose}, neither curve supports a threshold: WER stays on "
            f"the floor (range {rng:.3f}) and the {key} drift curve does not "
            f"trend with severity (spearman vs severity "
            f"{mono['spearman_vs_severity']:.2f}, {mono['n_sign_flips']} sign "
            f"flips) — this sweep does not enter a region where either stream "
            f"degrades measurably, so no decoupling claim is made.")

    # TRAP 6. WER moved, but this feature's comparison still isn't readable —
    # either its drift curve doesn't trend, or a half-level fell outside the
    # ladder. Name the reason instead of quoting the number anyway.
    if not v.get("half_levels_quotable", True):
        fq = v.get("feature_half_quality") or {}
        lq = v.get("lexical_half_quality") or {}
        mono = v["monotonicity"]
        why = []
        if not v.get("trend_reliable", True):
            why.append(f"the {key} drift curve does not trend with severity "
                       f"(spearman vs severity "
                       f"{mono['spearman_vs_severity']:.2f}, "
                       f"{mono['n_sign_flips']} sign flips)")
        if not fq.get("ok", True):
            why.append(f"its half-degradation level is unreadable — "
                       f"{fq.get('why')}")
        if not lq.get("ok", True):
            why.append(f"the WER half-degradation level is unreadable — "
                       f"{lq.get('why')}")
        return (f"Under {prose}, WER does traverse the full range "
                f"({v['lexical_degeneracy']['min']:.3f} -> "
                f"{v['lexical_degeneracy']['max']:.3f}), but no threshold "
                f"comparison is supportable for {key}: " + "; ".join(why) +
                ". No half-degradation level is quoted for this feature.")

    if v["coupled"]:
        return (f"Under {prose}, the {key} feature and WER degrade together — "
                f"half-degradation at {axis}≈{f_half:.2f}{unit} vs "
                f"≈{l_half:.2f}{unit}: COUPLED, so a paralinguistic monitor is a "
                f"usable early warning for lexical failure.{tail}")
    if v["leads"] == "lexical":
        return (f"Under {prose}, the {key} feature holds to {axis}≈"
                f"{f_half:.2f}{unit} while WER halves at ≈{l_half:.2f}{unit}: "
                f"lexical accuracy leads, i.e. a paralinguistics-driven agent "
                f"would not notice its ASR had already failed.{tail}")
    return (f"Under {prose}, the {key} feature collapses at {axis}≈"
            f"{f_half:.2f}{unit} while WER only halves at ≈{l_half:.2f}{unit}: "
            f"the paralinguistic stream leads, so a feature-based monitor would "
            f"alarm before the transcript measurably degrades.{tail}")


def _fmt_curve(levels: Sequence[float], vals: Sequence[float],
               std: Sequence[float] | None = None, width: int = 12) -> str:
    head = "  level : " + "".join(f"{lv:>{width}.4g}" for lv in levels)
    body = "  value : " + "".join(f"{v:>{width}.4g}" for v in vals)
    out = head + "\n" + body
    if std is not None:
        out += "\n  sd    : " + "".join(f"{v:>{width}.4g}" for v in std)
    return out


def format_report(report: Mapping) -> str:
    """The human-readable `results/l3_decoupling.txt`."""
    L: list[str] = []
    L.append("=" * 78)
    L.append("L3 — PARALINGUISTIC vs LEXICAL DECOUPLING (SPEC A.R5.9)")
    L.append("=" * 78)
    L.append("")
    L.append("Do the paralinguistic stream (energy/pitch/spectral shape) and the")
    L.append("lexical stream (WER) degrade at the same rate under a controlled")
    L.append("acoustic sweep, or does one collapse while the other holds?")
    L.append("")
    L.append(f"features : {', '.join(report['feature_keys'])}")
    L.append(f"gap_tol  : {report['gap_tol']}  (max normalized gap <= this => COUPLED)")
    L.append("")

    for factor, r in report["factors"].items():
        L.append("")
        L.append("─" * 78)
        L.append(f"FACTOR: {factor}   ({r['n_clips']} clips: {', '.join(r['clips'])})")
        L.append(f"severity order: {r['severity_order']}")
        L.append("─" * 78)
        L.append("")
        # Degeneracy is a PROPERTY OF THE MEASUREMENT, so it goes above the
        # verdict, not in a footnote. A reader must not reach a half-level
        # before knowing whether the curve it came off actually moved.
        lex = r["lexical_degeneracy"]
        if r["lexical_floor"]:
            L.append("!!! MEASUREMENT LIMITATION — LEXICAL CURVE ON THE FLOOR")
            L.append(f"    {lex['why']}")
            L.append(f"    mean WER {lex['min']:.3f} -> {lex['max']:.3f} "
                     f"(range {lex['range']:.3f}; needed >= "
                     f"{lex['min_range_required']:g})")
            L.append("    This sweep holds every OTHER factor at its benign end,")
            L.append("    so it walks a flat EDGE of the response surface. The")
            L.append("    damage in this study lives in the rt60 x snr_db")
            L.append("    INTERACTION (master.csv: rt60=0.45, snr=0 -> WER 0.971")
            L.append("    on these same 5 clips), which a single-factor ladder")
            L.append("    cannot reach. No lexical half-degradation level is")
            L.append("    quoted for this factor; the ones in the table below are")
            L.append("    shown in [brackets] to mark them non-quotable.")
            L.append("")
        L.append(">>> VERDICT")
        L.append(f"    {r['statement']}")
        L.append("")
        L.append(f"    headline feature   : {r['primary_feature']}")
        L.append(f"    factor-level lead  : {r['factor_leads']}  "
                 f"(vote over {r['n_features_stable']}/{r['n_features']} "
                 f"stable features)")
        L.append(f"    features decoupled : {r['n_features_decoupled']}"
                 f"/{r['n_features']}")
        L.append(f"    features trending  : {r['n_features_trend_reliable']}"
                 f"/{r['n_features']}  (spearman vs severity >= "
                 f"{MIN_TREND_RHO}; the rest are too noisy at n="
                 f"{r['n_clips']} to carry a threshold)")
        if r["baseline_wer_mean"] is not None:
            L.append(f"    clean (raw) WER    : {r['baseline_wer_mean']:.4f} "
                     "— the floor both curves start from")
        L.append("")
        L.append("  LEXICAL CURVE (mean WER over clips, severity order)")
        L.append(_fmt_curve(r["levels"], r["wer_curve_mean"], r["wer_curve_std"]))
        L.append("")
        L.append("  PER-CLIP WER (n is small; the spread is the honest part)")
        for cid, w in r["wer_per_clip"].items():
            L.append(f"    {cid} : " + "".join(f"{v:>9.4g}" for v in w))
        L.append("")
        L.append("  PER-FEATURE COMPARISON")
        L.append("  (half-levels in [brackets] are NOT quotable — the curve they")
        L.append("   came off is flat or does not trend with severity)")
        L.append(f"    {'feature':<11}{'rho_sev':>9}{'trend?':>8}"
                 f"{'spearman':>10}{'max_gap':>9}"
                 f"{'feat_half':>13}{'wer_half':>12}{'leads':>10}"
                 f"{'coupled':>9}{'stable':>8}")
        for key, v in r["per_feature"].items():
            q = v["half_levels_quotable"]
            fh = (f"{v['feature_half_level']:.3f}" if q
                  else f"[{v['feature_half_level']:.3f}]")
            lh = (f"{v['lexical_half_level']:.3f}" if q
                  else f"[{v['lexical_half_level']:.3f}]")
            L.append(f"    {key:<11}"
                     f"{v['monotonicity']['spearman_vs_severity']:>9.2f}"
                     f"{str(v['trend_reliable']):>8}"
                     f"{v['spearman']:>10.3f}"
                     f"{v['max_abs_gap']:>9.3f}"
                     f"{fh:>13}{lh:>12}"
                     f"{v['leads']:>10}{str(v['coupled']):>9}"
                     f"{str(v['stable']):>8}")
        L.append("")
        L.append("  WHAT EACH FEATURE'S CURVE ACTUALLY SUPPORTS")
        for key, v in r["per_feature"].items():
            m, s = v["monotonicity"], v["per_clip_feature_half_spread"]
            if v["trend_reliable"]:
                L.append(f"    {key:<11} REAL SIGNAL — drifts with severity "
                         f"(rho={m['spearman_vs_severity']:.2f}, "
                         f"{m['n_sign_flips']} sign flips); drift "
                         f"{v['drift_curve_mean'][0]:.4g} -> "
                         f"{v['drift_curve_mean'][-1]:.4g}")
            elif not v["half_levels_quotable"]:
                fq, lq = v["feature_half_quality"], v["lexical_half_quality"]
                bad = fq if not fq["ok"] else lq
                L.append(f"    {key:<11} TRENDS, BUT ITS HALF-LEVEL IS "
                         f"UNREADABLE — {bad['why']}. Drift "
                         f"{v['drift_curve_mean'][0]:.4g} -> "
                         f"{v['drift_curve_mean'][-1]:.4g}; no threshold quoted.")
            else:
                L.append(f"    {key:<11} UNDERPOWERED AT n={r['n_clips']} — "
                         f"drift does not trend with severity "
                         f"(rho={m['spearman_vs_severity']:.2f}, "
                         f"{m['n_sign_flips']} sign flips); per-clip half-level "
                         f"sd={s['std']:.3g} over [{s['min']:.3g}, "
                         f"{s['max']:.3g}]. This is a POWER limitation, not a "
                         f"finding of stability: no threshold is claimed.")
        L.append("")
        L.append("  PER-CLIP SPREAD OF THE FEATURE HALF-DEGRADATION LEVEL")
        L.append(f"    n={r['n_clips']} clips. Clips whose own WER curve is flat")
        L.append("    get NO vote on `leads` (there is no lexical curve to race);")
        L.append("    the count of those is shown as flat=N.")
        L.append(f"    {'feature':<11}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}"
                 f"   leads-vote (agreement)")
        for key, v in r["per_feature"].items():
            s = v["per_clip_feature_half_spread"]
            votes = ", ".join(f"{k}:{n}" for k, n in
                              sorted(v["leads_votes"].items(), key=lambda t: -t[1]))
            votes = votes or "(no clip has a lexical curve)"
            L.append(f"    {key:<11}{s['mean']:>9.3f}{s['std']:>9.3f}"
                     f"{s['min']:>9.3f}{s['max']:>9.3f}   {votes}  "
                     f"({v['leads_agreement']:.0%}, flat="
                     f"{v['n_clips_lexically_flat']})")
        L.append("")
        L.append("  MEAN FEATURE-DRIFT CURVES (|degraded - raw baseline|)")
        for key, v in r["per_feature"].items():
            mono = v["monotonicity"]
            flag = "" if mono["monotone"] else \
                f"   [NON-MONOTONE: {mono['n_sign_flips']} sign flips, " \
                f"rho_vs_severity={mono['spearman_vs_severity']:.2f}]"
            L.append(f"    {key} (clean baseline {v['clean_value_mean']:.4g})"
                     f"{flag}")
            L.append(_fmt_curve(r["levels"], v["drift_curve_mean"],
                                v["drift_curve_std"]))
        L.append("")
        L.append("  PER-FEATURE STATEMENTS")
        for key, v in r["per_feature"].items():
            L.append(f"    - {v['statement']}")
        L.append("")
        L.append("  NOTES / CAVEATS")
        for n in r["notes"]:
            L.append(f"    - {n}")

    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L3 decoupling (SPEC A.R5.9)")
    ap.add_argument("--make-audio", action="store_true",
                    help="phase 0: generate the round-2 harsh-region sweep audio "
                         "(pure DSP, no API calls)")
    ap.add_argument("--transcribe", action="store_true",
                    help="phase 1: transcribe the sweep wavs (uses the API)")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --transcribe: report the call count, call nothing")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=150)
    ap.add_argument("--gap-tol", type=float, default=0.2)
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-txt", default=str(OUT_TXT))
    a = ap.parse_args(argv)

    if a.make_audio:
        st = make_harsh_sweep_audio()
        print(f"generated {st['files']} wavs for {len(st['sweeps'])} harsh "
              f"sweeps -> {st['index']}")

    index = load_all_indexes()

    if a.transcribe:
        st = transcribe_sweep(index, workers=a.workers, max_calls=a.max_calls,
                              dry_run=a.dry_run)
        if a.dry_run:
            print(f"cached={st['cached']}  would_call={st['would_call']}")
            return 0
        print(f"transcribed: {st['calls']} new call(s), {st['cached']} from cache, "
              f"{st['n_failed']} failure(s)")
        if st["failures"]:
            print("  FAILED:", ", ".join(st["failures"]))
            return 1

    report = run_all(index, gap_tol=a.gap_tol)

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(report, indent=2))
    Path(a.out_txt).write_text(format_report(report), encoding="utf-8")
    print(f"wrote {a.out_json} and {a.out_txt}")
    for f, r in report["factors"].items():
        print(f"\n[{f}] {r['statement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
