#!/usr/bin/env python3
"""
demo_break.py — the 60 seconds that make the finding audible.

One recorded clip, two acoustic conditions:

    1. the RAW recording                       -> the model is right, and confident
    2. a MEASURED dead zone from the real grid -> the model is WRONG, and STILL confident

The whole project's premise is that the second case is the dangerous one. A model
that knows it is struggling can ask the speaker to repeat. A model that is wrong
at 0.85 confidence commits, and the downstream system has nothing to defend with.

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

# The #1-ranked dead zone in results/dead_zones.csv: reverberant room, GOOD SNR
# (20 dB), cafeteria babble, low-rate Opus, narrow mic. The SNR is the point --
# this is a *quiet* room, just a reverberant one on a bad channel.
DEFAULT_CONDITION = "rt60-0.7_snr-20_babble_opus-lowrate_roll-1"

# Exemplar clips inside that condition, ranked by confidence x WER. u39 leads
# because it is an alphanumeric licence plate: the model rewrites two characters
# of the plate and reports 0.86 / 0.73 confidence on the characters it invented.
DEMO_CLIPS = ["u39", "u11", "u23", "u10", "u07"]
DEFAULT_CLIP = "u39"


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


def _count_conditions() -> int:
    """How many conditions the grid actually measured -- the denominator for
    'N of M conditions are dead zones'. Read, never hardcoded."""
    csv.field_size_limit(10**9)
    names = set()
    with open(MASTER, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] == MODEL:
                names.add(row["condition_name"])
    return len(names)


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

    dz_rows = list(csv.DictReader(open(DEAD_ZONES)))
    dz_index = {r["condition_name"]: r for r in dz_rows}
    if condition_name not in dz_index:
        raise SystemExit(
            f"demo_break: {condition_name!r} is not in {DEAD_ZONES}. "
            f"Only measured dead zones may be demoed."
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
            "gap": _f(agg.get("gap")),
            "n_clips": int(_f(agg.get("n_clips"), 0)),
            "n_ref_total": int(_f(agg.get("n_ref_total"), 0)),
        },
        "dead_zones": [
            {"name": r["condition_name"], "mean_conf": _f(r["mean_conf"]),
             "wer": _f(r["wer"]), "gap": _f(r["gap"]),
             "n_clips": int(_f(r["n_clips"], 0))}
            for r in dz_rows
        ],
        "n_dead_zones": len(dz_rows),
        "n_conditions": _count_conditions(),
        "clips": out_clips,
    }
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
    show_stage(
        f"  [2]  DEAD ZONE #{cond['rank']}  —  {cond['name']}",
        f"       rt60 {cond['rt60']:g}s (measured {cond['rt60_measured']:.2f}s) · "
        f"SNR {cond['snr_db']:g} dB · {cond['noise_type']} · "
        f"{cond['codec']} · mic rolloff {cond['mic_rolloff']:g}\n"
        f"       Note the SNR: 20 dB is a QUIET room. This is reverb and channel, "
        f"not loudness.",
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
        subs = [(cols[i][2], wc[k]) for k, i in enumerate(hyp_slots) if cols[i][0] == "sub"]
        if subs:
            worst = max(subs, key=lambda t: t[1])
            print('    It invented the word {} and reported {} confidence in it.'.format(
                ink.red('"' + worst[0] + '"'), ink.red("{:.2f}".format(worst[1]))))
    print()

    # ---- it is not one unlucky clip --------------------------------------
    print(ink.dim(rule(width)))
    print(ink.bold("  AND IT IS NOT ONE UNLUCKY CLIP"))
    print()
    print(f"    Averaged over all {cond['n_clips']} clips ({cond['n_ref_total']} reference "
          f"words) in this same condition:")
    print("      mean confidence {}   WER {}   confidence-accuracy gap {}".format(
        ink.bold("{:.3f}".format(cond["mean_conf"])),
        ink.bold("{:.3f}".format(cond["wer"])),
        ink.bold("{:+.3f}".format(cond["gap"]))))
    print()
    print(f"    {cache['n_dead_zones']} conditions out of {cache.get('n_conditions', '?')} "
          f"cleared the dead-zone bar (high confidence AND high WER):")
    for i, z in enumerate(cache["dead_zones"], start=1):
        mark = ink.red("▸") if z["name"] == cond["name"] else " "
        print(f"      {mark} {i}. {pad(z['name'], 46)} conf {z['mean_conf']:.3f}  "
              f"WER {z['wer']:.3f}")
    print()
    print(ink.dim("    Globally, spearman(confidence, WER) = -0.957: this model mostly DOES"))
    print(ink.dim("    know when it is failing. That is what makes these 6 dangerous -- a"))
    print(ink.dim("    system tuned on the model's average self-awareness will trust it"))
    print(ink.dim("    exactly where it should not."))
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
