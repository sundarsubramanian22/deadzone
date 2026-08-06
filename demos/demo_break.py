#!/usr/bin/env python3
"""
demo_break.py — the 60 seconds that make the finding audible.

One recorded clip, two acoustic conditions:

    1. the RAW recording                       -> the model is right, and confident
    2. a MEASURED dead zone from the real grid -> the model is WRONG, and STILL confident

The whole project's premise is that the second case is the dangerous one. A model
that knows it is struggling can ask the speaker to repeat. A model that is wrong
while still sounding sure of itself commits, and the downstream system has nothing
to defend with. (No number is quoted here on purpose: the exemplar's real
confidence is printed by the demo and is read from the measured grid.)

    ./.venv/bin/python demos/demo_break.py                 # offline (DEFAULT). No key, no network.
    ./.venv/bin/python demos/demo_break.py --no-audio      # same, without playback
    ./.venv/bin/python demos/demo_break.py --clip u11      # a different exemplar
    ./.venv/bin/python demos/demo_break.py --prepare       # rebuild the offline cache (still no network)
    ./.venv/bin/python demos/demo_break.py --live          # opt-in: actually call the API

OFFLINE IS THE DEFAULT, NOT A FLAG YOU HAVE TO REMEMBER. Everything printed in the
default path is read from `results/demo/demo_cache.json`, which is built from the
already-paid-for grid (`results/master.csv`, `results/dead_zones.csv`,
`results/clean_transcripts.jsonl`). No API call is made and no key is read. The
two wavs are composed by the same tested DSP the grid used — `apply_condition` is
seeded from the condition NAME, so the audio you hear is bit-identical to the
audio that produced the number on screen.

`--live` re-transcribes both clips through Deepgram. It exists so you can prove
the cache is not theatre; it is never needed for the demo, and it degrades to the
offline path with a printed message when DEEPGRAM_API_KEY is absent.
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
# Makes `deadzone`, `scripts` and `demos` importable when this file is run
# directly (`python tests/test_pipeline.py`) with no install step. Harmless
# when it is imported as a module instead.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# -------------------------------------------------------------------------

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEMO_DIR = Path("results/demo")
CACHE_PATH = DEMO_DIR / "demo_cache.json"
AUDIO_DIR = DEMO_DIR / "audio"

MASTER = Path("results/master.csv")
DEAD_ZONES = Path("results/dead_zones.csv")
CLEAN_TRANSCRIPTS = Path("results/clean_transcripts.jsonl")
MANIFEST = Path("recording_manifest.csv")

MODEL = "nova-3"
FS = 16000

# The #1-ranked dead zone in results/dead_zones.csv: a moderately reverberant
# room, 0 dB engine noise, G.726 narrowband codec, flat mic.
#
# WHY THIS CELL AND NOT THE ONE THAT USED TO BE HERE. The previous default
# (rt60-0.7_snr-20_babble_opus-lowrate_roll-1) ranked #1 only under a pairing
# defect: its confidence was averaged over the 30 of 40 clips that produced
# words, while its WER was averaged over all 40 -- including 10 that returned an
# EMPTY transcript. On the clips it actually spoke on it was 81.8% accurate at
# 0.843 confidence, i.e. well calibrated. It is now classified `silence_driven`,
# and demoing it would have been demoing the artifact. This cell has ZERO silent
# clips, so its confidence and its WER are averaged over the same 40 clips and
# the claim needs no asterisk.
DEFAULT_CONDITION = "rt60-0.45_snr-0_engine_g726_roll-0"

# Exemplar clips inside that condition. u38 leads because it is a GPS
# coordinate destroyed while the model stays confident: "the coordinates are
# thirty seven north ..." -> "recordings on three seven north ...", WER 0.400 at
# mean confidence 0.815. It is the entity-destruction failure mode with the
# clearest downstream consequence, and its confidence clears the "was the model
# actually confident" bar with margin rather than by a hair.
# u39 (the alphanumeric licence plate) stays available via --clip: the plate is
# a better story but the model is only 0.687 confident on it in this cell, and a
# demo of a *silent* failure has to show a confident one.
DEMO_CLIPS = ["u38", "u03", "u25", "u18", "u39"]
DEFAULT_CLIP = "u38"


# --------------------------------------------------------------------------
# terminal
# --------------------------------------------------------------------------

class Ink:
    """ANSI when we're on a tty, plain text when piped. Same information either way."""

    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def dim(self, s):    return self._w("2", s)
    def bold(self, s):   return self._w("1", s)
    def red(self, s):    return self._w("1;31", s)
    def green(self, s):  return self._w("32", s)
    def yellow(self, s): return self._w("33", s)
    def cyan(self, s):   return self._w("36", s)
    def mag(self, s):    return self._w("35", s)
    def strike(self, s): return self._w("9;35", s)
    def inv(self, s):    return self._w("7", s)


def term_width(default: int = 96) -> int:
    try:
        w = shutil.get_terminal_size((default, 24)).columns
    except Exception:
        w = default
    return max(76, min(w, 110))


def visible_len(s: str) -> int:
    """Length ignoring ANSI escapes, so padding lines up when colour is on."""
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            j = s.find("m", i)
            i = len(s) if j < 0 else j + 1
            continue
        out += 1
        i += 1
    return out


def pad(s: str, w: int) -> str:
    return s + " " * max(0, w - visible_len(s))


# --------------------------------------------------------------------------
# cache construction  (offline: pure DSP + already-measured rows)
# --------------------------------------------------------------------------

def _f(x, default=float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_manifest_refs(path: Path = MANIFEST) -> dict[str, str]:
    return {r["id"]: r["ground_truth"] for r in csv.DictReader(open(path))}


def _master_rows(clips: set[str], condition: str) -> dict[str, dict]:
    """The (clip, condition, model) rows we need -- one streaming pass, no pandas."""
    csv.field_size_limit(10**9)
    want = {}
    with open(MASTER, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] != MODEL or row["condition_name"] != condition:
                continue
            if row["clip_id"] in clips:
                want[row["clip_id"]] = row
            if len(want) == len(clips):
                break
    return want


def load_dead_zones() -> list[dict]:
    """
    The GENUINE dead zones for this demo's model, in rank order.

    Two filters, both load-bearing:
      * `category == "dead_zone"`. results/dead_zones.csv is the full silent-failure
        table and also carries `silence_driven` cells (flagged only by a mismatched
        confidence/WER pairing) and `mute_zone` cells (no words on any clip). They
        belong in the file — dropping the mute zones would hide the worst conditions
        measured — but only the dead zones may be shown on stage under that name.
      * `model == MODEL`. Condition names are shared across arms, so a
        name-keyed lookup over the whole file silently returns whichever model was
        written last. That is how a nova-3 demo ends up quoting a Whisper WER.
    Older CSVs without a `category`/`model` column fall through unfiltered.
    """
    rows = list(csv.DictReader(open(DEAD_ZONES)))
    if rows and "model" in rows[0]:
        rows = [r for r in rows if r["model"] == MODEL]
    if rows and "category" in rows[0]:
        rows = [r for r in rows if r["category"] == "dead_zone"]
    return rows


FACTOR_KEYS = ("rt60", "snr_db", "noise_type", "codec", "mic_rolloff")
_NUMERIC_FACTORS = ("rt60", "snr_db", "mic_rolloff")


def _grid_facts() -> dict:
    """
    The shape of the grid, READ from the measured table rather than asserted.

      n_conditions  the denominator for "N of M conditions are dead zones".
      levels        every value each factor actually took. This is what lets the
                    narration say "the harshest SNR level this grid ran" instead
                    of a literal like "0 dB" -- a literal is correct exactly until
                    the analysis moves the exemplar, and then it is a contradiction
                    printed next to the factor line that disagrees with it. That is
                    not hypothetical: see `narrate_condition`.
    """
    csv.field_size_limit(10**9)
    names: set[str] = set()
    seen: dict[str, set[str]] = {k: set() for k in FACTOR_KEYS}
    with open(MASTER, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] != MODEL:
                continue
            names.add(row["condition_name"])
            for k in FACTOR_KEYS:
                if row.get(k, "") != "":
                    seen[k].add(row[k])
    levels: dict[str, list] = {}
    for k, vs in seen.items():
        levels[k] = (sorted({float(v) for v in vs}) if k in _NUMERIC_FACTORS
                     else sorted(vs))
    return {"n_conditions": len(names), "levels": levels}


def _global_correlation() -> dict:
    """
    The global confidence-vs-accuracy rank correlation, computed by the D1 layer
    itself (`deadzone.analysis.confidence_gap`) rather than restated here.

    It is `spearman(within-model confidence percentile, WER)` over the conditions
    in which the model actually emitted words, with WER taken on **those same
    clips** (`wer_spoke`). Both halves therefore describe one population. Pairing
    a confidence averaged over the speaking clips against a WER averaged over all
    of them is the estimand mismatch of SPEC Appendix G, which is what put a
    non-dead-zone at the top of this demo in the first place; recomputing it here
    by hand would be re-opening that door. Costs ~1 s and runs at `--prepare`
    time, never on stage.
    """
    from deadzone.analysis import load_master_table
    from deadzone.analysis.confidence_gap import overall_correlation, per_condition_table

    cond_rows = per_condition_table(load_master_table(str(MASTER)), model=MODEL)
    res = overall_correlation(cond_rows, wer_key="wer_spoke")
    return {
        "spearman": float(res["spearman_confpct_vs_wer"]),
        "n_conditions": int(res["n"]),
        "n_excluded_no_confidence": int(res["n_excluded_no_confidence"]),
        "wer_key": res["wer_key"],
        "verdict": res["verdict"],
    }


# --------------------------------------------------------------------------
# the narration  (DERIVED from the condition, never a literal)
# --------------------------------------------------------------------------
#
# The sentence this replaces was a literal. It read:
#
#     "Note the SNR: 20 dB is a QUIET room. This is reverb and channel, not
#      loudness."
#
# and it was written for the exemplar that ranked #1 BEFORE the estimand
# mismatch was found (SPEC Appendix G). When the corrected #1 turned out to be a
# 0 dB engine cell, the literal stayed put and the demo began printing
# "SNR 0 dB" and "20 dB is a QUIET room" four lines apart, on the slide the whole
# project is built around. Nothing failed; it just said two things.
#
# So the narration is now computed from the condition being demoed and from the
# levels the grid actually ran. It cannot contradict the factor line above it,
# because it is derived from the same dict. Superlatives ("the harshest SNR level
# this grid ran") are ranked against the measured level set rather than against a
# remembered one, and they degrade to plain phrasing if the level set is absent.

def _rank_label(value: float, levels: list, worst: str) -> str:
    """
    Where this level sits among the levels the grid actually ran.

    `worst` says which end hurts: "low" for SNR (quiet noise is easy), "high" for
    reverb and mic rolloff. Returns "" when the level set is unknown or has a
    single level, because a superlative over one point is not a description.
    """
    try:
        vals = sorted(float(v) for v in levels)
    except (TypeError, ValueError):
        return ""
    if len(set(vals)) < 2 or value != value:
        return ""
    lo, hi = vals[0], vals[-1]
    harsh, mild = (lo, hi) if worst == "low" else (hi, lo)
    if abs(value - harsh) < 1e-9:
        return "harshest"
    if abs(value - mild) < 1e-9:
        return "mildest"
    return "mid-range"


def _f_or_nan(cond: dict, key: str) -> float:
    return _f(cond.get(key))


def narrate_condition(cond: dict, levels: dict | None = None) -> str:
    """
    One derived sentence: which knobs are doing the damage in THIS cell, and
    which are sitting at their benign setting. Pure function of `cond` (+ the
    grid's level sets), so it cannot drift away from the condition on screen.
    """
    levels = levels or {}

    def lv(key):
        return levels.get(key) or []

    damage: list[str] = []
    spared: list[str] = []

    # --- noise -------------------------------------------------------------
    snr = _f_or_nan(cond, "snr_db")
    noise = str(cond.get("noise_type") or "noise")
    r = _rank_label(snr, lv("snr_db"), worst="low")
    if r == "mildest":
        spared.append(f"the {noise} noise is at the quietest SNR this grid ran")
    elif r:
        damage.append(f"{noise} noise at the {r} SNR this grid ran")
    else:
        damage.append(f"{noise} noise")

    # --- reverb ------------------------------------------------------------
    rt60 = _f_or_nan(cond, "rt60")
    r = _rank_label(rt60, lv("rt60"), worst="high")
    if r == "mildest":
        spared.append("the room is the driest this grid ran")
    elif r == "harshest":
        damage.append("the most reverberant room this grid ran")
    elif r:
        damage.append("a mid-range room")
    else:
        damage.append("the room")

    # --- codec -------------------------------------------------------------
    codec = str(cond.get("codec") or "none")
    if codec == "none":
        spared.append("no codec is applied")
    else:
        damage.append(f"the {codec} narrowband codec")

    # --- mic ---------------------------------------------------------------
    roll = _f_or_nan(cond, "mic_rolloff")
    r = _rank_label(roll, lv("mic_rolloff"), worst="high")
    if roll == 0.0:
        spared.append("the mic response is flat")
    elif r == "harshest":
        damage.append("the heaviest mic rolloff this grid ran")
    elif r:
        damage.append("a partial mic rolloff")
    else:
        damage.append("mic rolloff")

    def join(parts: list[str]) -> str:
        if len(parts) <= 1:
            return "".join(parts)
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    out = "Doing the damage: " + join(damage) + "."
    if spared:
        out += "  Left alone: " + join(spared) + "."
    # The point the old literal was reaching for, now conditional on the data.
    noise_spared = any("SNR" in s for s in spared)
    if noise_spared and damage:
        out += "  So this is the room and the channel, not loudness."
    elif len(damage) == 1 and not noise_spared and "noise" in damage[0]:
        out += "  So this is loudness alone."
    return out


def _row_facts(row: dict) -> dict:
    return {
        "transcript": row["transcript"],
        "wer": _f(row["wer"]),
        "mean_conf": _f(row["mean_conf"]),
        "n_ref": int(_f(row["n_ref"], 0)),
        "n_sub": int(_f(row["n_sub"], 0)),
        "n_del": int(_f(row["n_del"], 0)),
        "n_ins": int(_f(row["n_ins"], 0)),
        "edits": json.loads(row["edits"] or "[]"),
        "word_confidences": json.loads(row["word_confidences"] or "[]"),
    }


def build_cache(condition_name: str = DEFAULT_CONDITION,
                clips: list[str] | None = None,
                verbose: bool = True) -> dict:
    """
    Bake every fact and both wavs into results/demo/. Reads the measured grid and
    runs the tested DSP composer. Makes NO network call and reads NO API key.
    """
    # imported lazily: the offline demo path must not need numpy/librosa loaded
    import numpy as np
    import soundfile as sf
    from deadzone.audio_pipeline import classify_errors
    from deadzone.conditions import Condition, DiskAssetLibrary, apply_condition
    from scripts.run_experiment import load_clip, write_degraded_wav

    clips = list(clips or DEMO_CLIPS)
    for p in (MASTER, DEAD_ZONES, CLEAN_TRANSCRIPTS, MANIFEST):
        if not p.is_file():
            raise SystemExit(f"demo_break: missing {p} -- run the grid first (SPEC A.R4)")

    refs = load_manifest_refs()
    cond = Condition.from_name(condition_name)

    dz_rows = load_dead_zones()
    dz_index = {r["condition_name"]: r for r in dz_rows}
    if condition_name not in dz_index:
        raise SystemExit(
            f"demo_break: {condition_name!r} is not a `dead_zone` row for model "
            f"{MODEL!r} in {DEAD_ZONES}. Only measured dead zones may be demoed "
            f"— a `silence_driven` or `mute_zone` row is a different finding "
            f"(see analysis/confidence_gap.py) and demoing one as a dead zone "
            f"would be demoing the thing this project exists to catch."
        )
    agg = dz_index[condition_name]

    clean_tx = {}
    for line in open(CLEAN_TRANSCRIPTS):
        rec = json.loads(line)
        clean_tx[rec["id"]] = rec

    rows = _master_rows(set(clips), condition_name)
    missing = [c for c in clips if c not in rows]
    if missing:
        raise SystemExit(f"demo_break: no {MODEL} rows for {missing} in {condition_name}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    assets = DiskAssetLibrary(root="data", target_fs=FS)

    out_clips = {}
    for cid in clips:
        ref = refs[cid]
        raw = load_clip(cid, target_fs=FS)

        clean_path = AUDIO_DIR / f"{cid}_1_clean.wav"
        sf.write(clean_path, raw, FS, subtype="PCM_16")

        degraded = apply_condition(raw, cond, assets, FS)
        dz_path = AUDIO_DIR / f"{cid}_2_deadzone.wav"
        write_degraded_wav(dz_path, degraded, FS)

        ct = clean_tx.get(cid, {})
        clean_hyp = ct.get("transcript", "")
        clean_scored = classify_errors(ref, clean_hyp)
        cc = clean_scored["counts"]
        clean_facts = {
            "transcript": clean_hyp,
            "wer": float(clean_scored["wer"]),
            "mean_conf": _f(ct.get("mean_conf")),
            "n_ref": int(clean_scored["n_ref"]),
            "n_sub": int(cc.get("sub", 0)),
            "n_del": int(cc.get("del", 0)),
            "n_ins": int(cc.get("ins", 0)),
            "edits": [list(e) for e in clean_scored["edits"]],
            "word_confidences": list(ct.get("word_confidences") or []),
        }

        out_clips[cid] = {
            "ref": ref,
            "duration_s": round(len(raw) / FS, 2),
            "clean": clean_facts,
            "deadzone": _row_facts(rows[cid]),
            "audio_clean": str(clean_path),
            "audio_deadzone": str(dz_path),
        }
        if verbose:
            print(f"  baked {cid}: clean WER {clean_facts['wer']:.2f} "
                  f"-> dead-zone WER {out_clips[cid]['deadzone']['wer']:.2f}")

    cache = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_master": str(MASTER),
        "model": MODEL,
        "fs": FS,
        "default_clip": DEFAULT_CLIP if DEFAULT_CLIP in out_clips else clips[0],
        "condition": {
            "name": condition_name,
            "rt60": cond.rt60,
            "snr_db": cond.snr_db,
            "noise_type": cond.noise_type,
            "codec": cond.codec,
            "mic_rolloff": cond.mic_rolloff,
            "rt60_measured": _f(agg.get("rt60_measured")),
            "rank": 1 + [r["condition_name"] for r in dz_rows].index(condition_name),
            "mean_conf": _f(agg.get("mean_conf")),
            "wer": _f(agg.get("wer")),
            # the paired accuracy + the silence accounting that makes the pairing
            # legitimate; a demo that quotes a gap must be able to show both
            "wer_spoke": _f(agg.get("wer_spoke")),
            "n_silent": int(_f(agg.get("n_silent"), 0)),
            "gap": _f(agg.get("gap")),
            "gap_all_clips": _f(agg.get("gap_all_clips")),
            "category": agg.get("category", "dead_zone"),
            "n_clips": int(_f(agg.get("n_clips"), 0)),
            "n_ref_total": int(_f(agg.get("n_ref_total"), 0)),
        },
        "dead_zones": [
            {"name": r["condition_name"], "mean_conf": _f(r["mean_conf"]),
             "wer": _f(r["wer"]), "gap": _f(r["gap"]),
             "wer_spoke": _f(r.get("wer_spoke")),
             "n_silent": int(_f(r.get("n_silent"), 0)),
             "n_clips": int(_f(r["n_clips"], 0))}
            for r in dz_rows
        ],
        "n_dead_zones": len(dz_rows),
        "clips": out_clips,
    }
    grid = _grid_facts()
    cache["n_conditions"] = grid["n_conditions"]
    # every factor level the grid ran -- the narration ranks against this rather
    # than against a remembered level set (see narrate_condition)
    cache["grid_levels"] = grid["levels"]
    # the closing claim, computed by the D1 layer rather than restated here
    cache["correlation"] = _global_correlation()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))
    if verbose:
        print(f"  wrote {CACHE_PATH}  ({len(out_clips)} clips, "
              f"{2 * len(out_clips)} wavs in {AUDIO_DIR})")
    return cache


def load_cache(auto_build: bool = True, verbose: bool = True) -> dict:
    if CACHE_PATH.is_file():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            if verbose:
                print("demo_break: cache unreadable, rebuilding.", file=sys.stderr)
    if not auto_build:
        raise SystemExit(f"demo_break: no cache at {CACHE_PATH} -- run `make demo-prep`")
    if verbose:
        print("demo_break: no cache yet, baking it (offline, ~2 s)...")
    return build_cache(verbose=verbose)


# --------------------------------------------------------------------------
# the aligned diff
# --------------------------------------------------------------------------

GAP = "···"      # ··· stands in for "nothing on this side"


def diff_columns(edits: list) -> list[tuple[str, str, str]]:
    """One (op, ref_cell, hyp_cell) per aligned slot, straight from classify_errors."""
    cols = []
    for e in edits:
        op = e[0]
        r = e[1] if len(e) > 1 and e[1] is not None else ""
        h = e[2] if len(e) > 2 and e[2] is not None else ""
        if op == "del":
            h = GAP
        elif op == "ins":
            r = GAP
        cols.append((op, r, h))
    return cols


def render_diff(edits: list, width: int, ink: Ink,
                word_confidences: list | None = None) -> list[str]:
    """
    Three stacked rows -- REF / HYP / marker -- wrapped to `width`, columns aligned
    so every substitution reads as a vertical pair. Alignment comes from
    classify_errors' typed edits; nothing is re-derived here.
    """
    cols = diff_columns(edits)
    if not cols:
        return [ink.dim("(no aligned edits recorded)")]

    # per-word confidence lines up 1:1 with the HYPOTHESIS words, i.e. the edits
    # that produced a hypothesis token (match / sub / ins). If the counts disagree
    # -- a known tokenisation hazard, see SPEC B.5(3) -- we simply drop the
    # per-word annotation rather than binding confidences to the wrong words.
    confs: list[float | None] = [None] * len(cols)
    if word_confidences:
        hyp_slots = [i for i, (op, _, _) in enumerate(cols) if op in ("match", "sub", "ins")]
        if len(hyp_slots) == len(word_confidences):
            for slot, c in zip(hyp_slots, word_confidences):
                confs[slot] = float(c)

    label_w = 6
    body_w = max(20, width - label_w)
    lines: list[str] = []
    row_ref, row_hyp, row_mark = "", "", ""
    used = 0

    def flush():
        nonlocal row_ref, row_hyp, row_mark, used
        if not used:
            return
        lines.append(ink.dim("ref  ") + " " + row_ref.rstrip())
        lines.append(ink.dim("hyp  ") + " " + row_hyp.rstrip())
        if row_mark.strip():
            lines.append(" " * label_w + row_mark.rstrip())
        lines.append("")
        row_ref = row_hyp = row_mark = ""
        used = 0

    for i, (op, r, h) in enumerate(cols):
        cell = max(len(r), len(h))
        conf = confs[i]
        if conf is not None and op in ("sub", "ins"):
            cell = max(cell, len(f"{conf:.2f}"))
        if used and used + cell + 1 > body_w:
            flush()

        if op == "match":
            rr, hh, mk = ink.dim(r), ink.dim(h), ""
        elif op == "sub":
            rr, hh, mk = ink.yellow(r), ink.red(h), "^" * len(h)
        elif op == "del":
            rr, hh, mk = ink.strike(r), ink.mag(h), "-" * len(h)
        else:  # ins
            rr, hh, mk = ink.mag(r), ink.cyan(h), "+" * len(h)

        if conf is not None and op in ("sub", "ins"):
            mk = f"{conf:.2f}"

        row_ref += pad(rr, cell) + " "
        row_hyp += pad(hh, cell) + " "
        row_mark += pad(mk, cell) + " "
        used += cell + 1

    flush()
    while lines and lines[-1] == "":
        lines.pop()
    return lines


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def rule(width: int, ch: str = "─") -> str:
    return ch * width


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, max(30, width)) or [""]


def _paired_wer(cond: dict) -> float:
    """
    The WER that may legally be printed next to a mean confidence: the one over
    the clips that produced words. `wer` in the same dict is the all-clips
    corpus WER, and subtracting a confidence from it is the estimand mismatch
    (SPEC Appendix G). Falls back to `wer` only when `wer_spoke` is absent, which
    is an old cache, not a licence to mix populations.
    """
    w = _f(cond.get("wer_spoke"))
    return w if w == w else _f(cond.get("wer"))


def verdict_card(facts: dict, ink: Ink, width: int, danger: bool) -> list[str]:
    conf = facts["mean_conf"]
    wer = facts["wer"]
    conf_s = "n/a" if conf != conf else f"{conf:.3f}"
    accent = ink.red if danger else ink.green
    inner = width - 4
    lines = [
        "┌" + "─" * (width - 2) + "┐",
        "│ " + pad(ink.dim("THE MODEL'S OWN VERDICT"), inner) + " │",
        "│ " + pad("", inner) + " │",
        "│ " + pad(ink.dim("mean word confidence"), inner) + " │",
        "│ " + pad("   " + accent(ink.bold(conf_s)), inner) + " │",
        "│ " + pad("", inner) + " │",
        "│ " + pad(ink.dim("WER we measured"), inner) + " │",
        "│ " + pad("   " + accent(ink.bold(f"{wer:.3f}")), inner) + " │",
        "│ " + pad("", inner) + " │",
        "│ " + pad(ink.dim(f"sub {facts['n_sub']}  del {facts['n_del']}  "
                                f"ins {facts['n_ins']}  / {facts['n_ref']} ref"), inner) + " │",
        "└" + "─" * (width - 2) + "┘",
    ]
    return lines


def two_pane(left: list[str], right: list[str], left_w: int, gutter: int = 2) -> list[str]:
    n = max(len(left), len(right))
    out = []
    for i in range(n):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        out.append(pad(l, left_w) + " " * gutter + r)
    return out


def show_stage(title: str, subtitle: str, facts: dict, ink: Ink, width: int,
               danger: bool) -> None:
    card_w = 34
    left_w = width - card_w - 2
    print()
    print(ink.bold(title))
    print(ink.dim(subtitle))
    print(ink.dim(rule(width)))
    left = render_diff(facts["edits"], left_w, ink, facts.get("word_confidences"))
    right = verdict_card(facts, ink, card_w, danger)
    for line in two_pane(left, right, left_w):
        print(line.rstrip())
    print()


# --------------------------------------------------------------------------
# playback
# --------------------------------------------------------------------------

def find_player() -> list[str] | None:
    for cmd in (["afplay"], ["aplay", "-q"], ["play", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


def play(path: str, ink: Ink, enabled: bool) -> None:
    """Never let a missing/hung audio player take the demo down."""
    if not enabled:
        return
    p = Path(path)
    if not p.is_file():
        print(ink.dim(f"   (audio missing: {p} -- run `make demo-prep`)"))
        return
    player = find_player()
    if player is None:
        print(ink.dim("   (no audio player found -- continuing silently)"))
        return
    print(ink.dim(f"   ♪ playing {p.name} ..."))
    try:
        subprocess.run(player + [str(p)], timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:                       # noqa: BLE001 -- demo must survive
        print(ink.dim(f"   (playback skipped: {type(exc).__name__})"))


# --------------------------------------------------------------------------
# --live
# --------------------------------------------------------------------------

def live_refresh(cache: dict, clip_id: str, ink: Ink) -> dict | None:
    """
    Re-transcribe both wavs through the real API. Returns updated facts, or None
    if we could not (no key / no SDK / call failed) -- in which case the caller
    just keeps the cached numbers and says so.
    """
    if not os.environ.get("DEEPGRAM_API_KEY"):
        print(ink.yellow("--live: DEEPGRAM_API_KEY is not set in the environment."))
        print(ink.dim("        Falling back to the cached offline results, which are the "
                      "same measurements the grid produced."))
        return None
    try:
        from deadzone.audio_pipeline import classify_errors, is_failed, transcribe_deepgram
    except Exception as exc:                        # noqa: BLE001
        print(ink.yellow(f"--live: adapter unavailable ({type(exc).__name__}); using cache."))
        return None

    entry = cache["clips"][clip_id]
    ref = entry["ref"]
    out = {}
    for stage, audio_key in (("clean", "audio_clean"), ("deadzone", "audio_deadzone")):
        path = entry[audio_key]
        print(ink.dim(f"--live: transcribing {Path(path).name} ..."))
        try:
            res = transcribe_deepgram(path)
            if is_failed(res):
                raise RuntimeError(res.get("error", "failed"))
        except Exception as exc:                    # noqa: BLE001
            print(ink.yellow(f"--live: call failed ({type(exc).__name__}); using cache."))
            return None
        scored = classify_errors(ref, res["transcript"])
        c = scored["counts"]
        wc = list(res.get("word_confidences") or [])
        out[stage] = {
            "transcript": res["transcript"],
            "wer": float(scored["wer"]),
            "mean_conf": (sum(wc) / len(wc)) if wc else float("nan"),
            "n_ref": int(scored["n_ref"]), "n_sub": int(c.get("sub", 0)),
            "n_del": int(c.get("del", 0)), "n_ins": int(c.get("ins", 0)),
            "edits": [list(e) for e in scored["edits"]],
            "word_confidences": wc,
        }
    return out


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def preflight(ink: Ink) -> int:
    """
    Is every artifact the demo needs on disk, right now, offline? Run this
    BEFORE you turn wifi off, not after.
    """
    ok = True

    def chk(cond: bool, label: str, hint: str = "") -> None:
        nonlocal ok
        cond = bool(cond)
        mark = ink.green("  ok  ") if cond else ink.red(" MISS ")
        tail = "" if cond or not hint else ink.dim("   -> " + hint)
        print(f"{mark}  {label}{tail}")
        ok = ok and cond

    print()
    print(ink.bold("  Deadzone demo preflight (offline)"))
    print()
    chk(CACHE_PATH.is_file(), str(CACHE_PATH), "make demo-prep")
    if CACHE_PATH.is_file():
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            chk(False, "cache parses as JSON", "make demo-prep")
            cache = None
        if cache:
            for cid in sorted(cache["clips"]):
                e = cache["clips"][cid]
                for key in ("audio_clean", "audio_deadzone"):
                    p = Path(e[key])
                    chk(p.is_file(), f"audio {p.name}", "make demo-prep")
            names = {z["name"] for z in cache["dead_zones"]}
            chk(cache["condition"]["name"] in names,
                "the demo condition is a MEASURED dead zone")
            chk(cache["clips"][cache["default_clip"]]["deadzone"]["wer"] > 0.2,
                "the default exemplar actually fails in it")
            # the narration is derived from these two; without them the demo
            # silently drops its closing claim and loses its superlatives
            chk(bool(cache.get("grid_levels")),
                "the grid's factor levels are cached (narration derives from them)",
                "make demo-prep")
            chk(_f((cache.get("correlation") or {}).get("spearman")) ==
                _f((cache.get("correlation") or {}).get("spearman")),
                "the global confidence-WER correlation is cached", "make demo-prep")
    chk(MASTER.is_file(), str(MASTER), "run the grid (SPEC A.R4)")
    chk(DEAD_ZONES.is_file(), str(DEAD_ZONES), "run analysis/confidence_gap.py")
    chk(Path("dashboard/deadzone.html").is_file(), "dashboard/deadzone.html",
        "make dashboard-build")
    chk(Path("dashboard/DEMO.md").is_file(), "dashboard/DEMO.md")
    chk(Path("requirements.lock.txt").is_file(), "requirements.lock.txt", "make lock")
    chk(find_player() is not None, "an audio player on PATH",
        "demo_break degrades to silent, but you lose the visceral half")

    print()
    print("  " + (ink.green("READY — you can turn wifi off.") if ok
                  else ink.red("NOT READY — fix the MISS lines above.")))
    print()
    return 0 if ok else 1


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(cache: dict, clip_id: str, ink: Ink, width: int, audio: bool,
        pause: float, live: bool) -> int:
    if clip_id not in cache["clips"]:
        raise SystemExit(f"demo_break: no cached clip {clip_id!r}. "
                         f"Have: {', '.join(sorted(cache['clips']))}")
    entry = cache["clips"][clip_id]
    cond = cache["condition"]
    clean, dz = entry["clean"], entry["deadzone"]
    source = "cached measurement (offline)"

    if live:
        fresh = live_refresh(cache, clip_id, ink)
        if fresh:
            clean, dz = fresh["clean"], fresh["deadzone"]
            source = "LIVE API call, just now"

    print()
    print(ink.inv(pad(f"  DEADZONE  —  is the model wrong, or is it wrong AND CONFIDENT?  ", width)))
    print()
    print(f"  clip      {ink.bold(clip_id)}   “{entry['ref']}”")
    print(f"  model     {cache['model']}   ·   source: {source}")
    print(ink.dim(rule(width)))

    # ---- stage 1: the raw recording -------------------------------------
    play(entry["audio_clean"], ink, audio)
    show_stage(
        "  [1]  RAW RECORDING  —  no degradation at all",
        "       The control. There is no 'clean' Condition: apply_condition always "
        "applies a room\n       and always mixes noise, so the only true null is the "
        "untouched file.",
        clean, ink, width, danger=False)
    time.sleep(pause)

    # ---- stage 2: the measured dead zone --------------------------------
    play(entry["audio_deadzone"], ink, audio)
    note = narrate_condition(cond, cache.get("grid_levels"))
    show_stage(
        f"  [2]  DEAD ZONE #{cond['rank']}  —  {cond['name']}",
        f"       rt60 {cond['rt60']:g}s (measured {cond['rt60_measured']:.2f}s) · "
        f"SNR {cond['snr_db']:g} dB · {cond['noise_type']} · "
        f"{cond['codec']} · mic rolloff {cond['mic_rolloff']:g}\n"
        + "\n".join("       " + l for l in _wrap(note, width - 7)),
        dz, ink, width, danger=True)

    # ---- the punchline ---------------------------------------------------
    d_conf = clean["mean_conf"] - dz["mean_conf"]
    d_wer = dz["wer"] - clean["wer"]
    dz_wer_s = ink.red("{:.3f}".format(dz["wer"]))
    dz_conf_s = ink.yellow("{:.3f}".format(dz["mean_conf"]))
    print(ink.dim(rule(width)))
    print(ink.bold("  WHAT JUST HAPPENED"))
    print()
    print("    WER        {:.3f}  ->  {}      ({})".format(
        clean["wer"], dz_wer_s, ink.red("+{:.3f}".format(d_wer))))
    print("    confidence {:.3f}  ->  {}      (only {:.3f} lower)".format(
        clean["mean_conf"], dz_conf_s, d_conf))
    print()
    print("    Accuracy collapsed. The model's self-report barely moved.")

    # the per-word twist, when the alignment held
    wc = dz.get("word_confidences") or []
    cols = diff_columns(dz["edits"])
    hyp_slots = [i for i, (op, _, _) in enumerate(cols) if op in ("match", "sub", "ins")]
    if wc and len(hyp_slots) == len(wc):
        # SUBSTITUTIONS only, and named as such. The verdict card next to this
        # line prints `ins 0` for the default exemplar, so calling a substitution
        # an invention is a claim a presenter would immediately have to walk back.
        # Both sides are shown: "heard X as Y" is checkable against the diff above.
        subs = [(cols[i][1], cols[i][2], wc[k])
                for k, i in enumerate(hyp_slots) if cols[i][0] == "sub"]
        if subs:
            ref_w, hyp_w, c = max(subs, key=lambda t: t[2])
            print('    It heard {} as {} and reported {} confidence in that word.'.format(
                ink.yellow('"' + ref_w + '"'), ink.red('"' + hyp_w + '"'),
                ink.red("{:.2f}".format(c))))
    print()

    # ---- it is not one unlucky clip --------------------------------------
    print(ink.dim(rule(width)))
    print(ink.bold("  AND IT IS NOT ONE UNLUCKY CLIP"))
    print()
    # THE ESTIMAND IS NAMED, AND THE TWO HALVES ARE ON THE SAME CLIPS. `gap` is
    # `mean_conf - (1 - wer_spoke)`, so the WER printed beside the confidence must
    # be `wer_spoke` -- the WER over exactly the clips the confidence is averaged
    # over. Printing the all-clips WER here instead is the SPEC Appendix G
    # mismatch, and it is silent: for the default exemplar the two are equal, so
    # a wrong pairing would look right on stage and only misfire once the
    # exemplar moves to a cell with silent clips.
    n_sil = int(cond.get("n_silent", 0) or 0)
    n_spoke = cond["n_clips"] - n_sil
    wer_paired = _paired_wer(cond)
    if n_sil == 0:
        print(f"    Averaged over all {cond['n_clips']} clips ({cond['n_ref_total']} "
              f"reference words) in this same condition:")
    else:
        print(f"    Averaged over the {n_spoke} of {cond['n_clips']} clips this condition "
              f"produced words on:")
    print("      mean confidence {}   WER {}   confidence-accuracy gap {}".format(
        ink.bold("{:.3f}".format(cond["mean_conf"])),
        ink.bold("{:.3f}".format(wer_paired)),
        ink.bold("{:+.3f}".format(cond["gap"]))))
    if n_sil == 0:
        print(ink.dim("      (all {} clips produced a transcript, so the confidence and the "
                      "WER".format(cond["n_clips"])))
        print(ink.dim("       are averaged over the same clips -- the gap needs no asterisk.)"))
    else:
        print(ink.dim("      ({} of {} clips returned an EMPTY transcript and {} no "
                      "confidence,".format(n_sil, cond["n_clips"],
                                           "carries" if n_sil == 1 else "carry")))
        print(ink.dim("       so they are out of both averages above. Over all {} clips the "
                      "WER is {:.3f}.)".format(cond["n_clips"], cond["wer"])))
    print()
    print(f"    {cache['n_dead_zones']} conditions out of {cache.get('n_conditions', '?')} "
          f"cleared the dead-zone bar (high confidence AND high WER):")
    for i, z in enumerate(cache["dead_zones"], start=1):
        mark = ink.red("▸") if z["name"] == cond["name"] else " "
        sil = int(z.get("n_silent", 0) or 0)
        tail = "" if sil == 0 else ink.dim("  ({}/{} spoke)".format(z["n_clips"] - sil,
                                                                   z["n_clips"]))
        print(f"      {mark} {i}. {pad(z['name'], 46)} conf {z['mean_conf']:.3f}  "
              f"WER {_paired_wer(z):.3f}{tail}")
    print(ink.dim("      (conf and WER are both over the clips the model spoke on)"))
    print()

    # The closing claim, read from the cache the D1 layer filled in. If it is
    # absent the demo says so rather than falling back to a remembered number --
    # a stale constant printed with authority is worse than a missing line.
    corr = cache.get("correlation") or {}
    rho = _f(corr.get("spearman"))
    if rho == rho:
        print(ink.dim("    Across the {} conditions where {} emitted words, spearman(confidence"
                      .format(corr.get("n_conditions", "?"), cache["model"])))
        print(ink.dim("    percentile, WER on those same clips) = {:.3f}: this model mostly DOES"
                      .format(rho)))
        print(ink.dim("    know when it is failing. That is what makes these {} dangerous -- a"
                      .format(cache["n_dead_zones"])))
        print(ink.dim("    system tuned on the model's average self-awareness will trust it"))
        print(ink.dim("    exactly where it should not."))
    else:
        print(ink.dim("    (global confidence-WER correlation not in this cache -- run "
                      "`make demo-prep`)"))
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Play one clip clean, then in a measured dead zone, and show "
                    "the transcript diff beside the model's confidence.")
    ap.add_argument("--clip", default=None, help=f"exemplar clip id (default {DEFAULT_CLIP})")
    ap.add_argument("--condition", default=DEFAULT_CONDITION,
                    help="dead-zone condition name (must exist in results/dead_zones.csv)")
    ap.add_argument("--offline", action="store_true",
                    help="explicit no-op: offline is already the default")
    ap.add_argument("--live", action="store_true",
                    help="opt in to a real API call; falls back to cache without a key")
    ap.add_argument("--no-audio", dest="audio", action="store_false",
                    help="skip playback (still prints everything)")
    ap.add_argument("--prepare", action="store_true",
                    help="rebuild results/demo/ from the measured grid, then exit")
    ap.add_argument("--pause", type=float, default=0.6,
                    help="seconds between the two stages (default 0.6)")
    ap.add_argument("--no-color", dest="color", action="store_false",
                    help="force plain output")
    ap.add_argument("--list-clips", action="store_true", help="show cached exemplars and exit")
    ap.add_argument("--check", action="store_true",
                    help="preflight: verify every demo artifact exists, then exit")
    args = ap.parse_args(argv)

    if args.offline and args.live:
        print("demo_break: --offline and --live are contradictory; offline wins.",
              file=sys.stderr)
        args.live = False

    color = args.color and sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")
    ink = Ink(color)
    width = term_width()

    if args.check:
        return preflight(ink)

    if args.prepare:
        print("demo_break --prepare: baking the offline demo cache (no network)...")
        build_cache(condition_name=args.condition)
        print("demo_break: ready. `make demo-break` now runs with wifi off.")
        return 0

    cache = load_cache()
    if args.list_clips:
        for cid, e in sorted(cache["clips"].items()):
            print(f"{cid}  clean WER {e['clean']['wer']:.2f} conf {e['clean']['mean_conf']:.3f}"
                  f"   dead-zone WER {e['deadzone']['wer']:.2f} conf "
                  f"{e['deadzone']['mean_conf']:.3f}   “{e['ref']}”")
        return 0

    clip = args.clip or cache.get("default_clip", DEFAULT_CLIP)
    return run(cache, clip, ink, width, args.audio, args.pause, args.live)


if __name__ == "__main__":
    sys.exit(main())
