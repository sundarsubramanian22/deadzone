#!/usr/bin/env python3
"""
demo_hero.py — THE demo. One clip, heard and transcribed live, twice.

    ./.venv/bin/python demos/demo_hero.py          # `make demo` runs exactly this

This is the merge of the two beats that used to be separate: `demo_break.py`
made the finding *audible* but read its numbers out of a CSV, and
`demo_live.py` made two real calls but played nothing and was pinned to one
clip. Neither is the whole argument. You have to HEAR the degraded audio to
believe the condition is real, and you have to WATCH the confidences come back
off the wire to believe the model is not hedging. So:

    preamble ->  pick a clip (menu, or random)
      [1]  play the RAW recording      -> ONE live call -> per-word confidences
           pause: ready for the degraded version?  (or replay the clean one)
      [2]  play the DEGRADED recording -> ONE live call -> per-word confidences
           the transcript collapses; the model's self-report barely moves
      ->   the punchline, computed from THIS run's payload
      ->   the same cell in the archived grid, as corroboration
      ->   run another?

EVERY NUMBER ON SCREEN IN THE LIVE PATH COMES FROM THE TWO RESPONSES JUST
RECEIVED. Nothing is read from the cache and presented as if it were live. The
archived grid row is shown afterwards, explicitly labelled, as a reproduction
check — never as a substitute.

THE CURATED SET IS DERIVED, NOT CHOSEN BY TASTE. `select_exemplars()` reads
`results/dead_zones.csv`, `results/master.csv`, `results/clean_transcripts.jsonl`
and `task_specs.json` and applies hard filters plus a documented rank (see
SELECTION below). If the grid is re-run and the dead zones move, the menu moves
with them. There is no hardcoded list of "good clips" anywhere in this file.

THE PUNCHLINE IS A LADDER, NOT A STRING. The sentence this demo wants to say is
`EXACT_CLAIM` below. It is printed **only** when the payload that just arrived
supports it: the most confident wrong word must be a word absent from the
reference (i.e. genuinely invented) AND more confident than every word the model
got right. When the live call does not support that, `punchline_claim()` prints
the strongest sentence it *does* support and says so. A demo about silent
failure is not allowed to state a claim its own data contradicts.

THE FAILURE CONTRACT, which matters more than the happy path:
  * no key, no network, a vendor error, a timeout, a missing SDK -> ONE
    explanatory line, fall through to the archived measurements clearly labelled
    CACHED, and **exit 0**. A conference wifi outage must not take the demo down.
  * `--replay` (alias `--offline`) runs the entire beat from cache with no
    network access at all. That is the rehearsal mode and the instant fallback.
  * stdin is never required. EOF, Ctrl-C, a pipe, or a non-tty each finish the
    beat and exit 0 rather than blocking in front of an audience.

WHAT THIS DELIBERATELY DOES NOT PRINT, and where it went instead. The cost line,
the per-minute vendor rate quoted with a date, `run_id` strings and the
`results/MANIFEST.json` provenance block were all on screen in `demo_live.py`.
They are real and they matter — they just belong in the written record, not on a
projector where they crowd out the two numbers the beat is about. They are
listed in `report/_demo_internal_notes.md`. `tests/test_demo_hero.py` asserts
none of them can come back.

NOVA-3 ONLY. Not a stylistic preference: ElevenLabs Scribe's orthography is
non-deterministic call to call (worth up to 0.727 strict WER on byte-identical
input), so live, in front of an audience, it is a coin flip that forces the
presenter to explain vendor orthography on someone else's cue. Whisper is local,
so there is no wire to watch.

    --prepare            bake the wavs + the curated set (offline, no key)
    --check              preflight: can this run live right now?
    --list               print the curated set and exit
    --clip u08           skip the menu
    --random             skip the menu, pick at random from the set
    --replay / --offline the whole beat from cache, no network
    --once               do not offer "run another?"
    --no-audio           skip playback (everything still prints)
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
# Makes `deadzone`, `scripts` and `demos` importable when this file is run
# directly (`python demos/demo_hero.py`) with no install step. Harmless when it
# is imported as a module instead.
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
import random
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

# Presentation and credential primitives are IMPORTED, never re-implemented.
# Two copies of the diff renderer would drift and the audience would see two
# different projects; two copies of the redactor would mean the one that got
# fixed is not necessarily the one that runs (SPEC J.3).
from demos.demo_break import (
    Ink,
    diff_columns,
    find_player,
    load_dead_zones,
    load_manifest_refs,
    pad,
    play,
    render_diff,
    rule,
    term_width,
    _f,
)
from demos.demo_live import (
    LiveCallFailed,
    conf_bar,
    load_credentials,
    redact,
    say,
    score,
    transcribe_live,
)

MODEL = "nova-3"
FS = 16000

HERO_DIR = Path("results/demo/hero")
CACHE_PATH = HERO_DIR / "hero_cache.json"

MASTER = Path("results/master.csv")
DEAD_ZONES = Path("results/dead_zones.csv")
CLEAN_TRANSCRIPTS = Path("results/clean_transcripts.jsonl")
TASK_SPECS = Path("task_specs.json")

# Per-call wall-clock ceiling. ~5x the observed round trip, so it never fires on
# a working link, and two calls plus the fallback still land inside the budget.
DEFAULT_TIMEOUT = 8.0


# =========================================================================
# SELECTION — the curated set, derived from artifacts
# =========================================================================
#
# HARD FILTERS (a candidate that fails any of these cannot be demoed at all):
#
#   1. The CONDITION is a genuine `dead_zone` row for this model in
#      results/dead_zones.csv. `silence_driven` and `mute_zone` rows are in that
#      file for good reasons and are different findings; demoing one under the
#      name "dead zone" is the exact error SPEC Appendix G is about.
#   2. The clip's CLEAN transcript is exactly right (WER 0.000). Stage 1 is the
#      control. A control that is already wrong destroys the contrast before
#      stage 2 gets to make it — and the corpus has a measured clean-condition
#      floor, so some clips genuinely do not qualify.
#   3. The grid row is not a failure, and its per-word confidences line up 1:1
#      with the hypothesis slots of the edit alignment. Where they do not, no
#      per-word claim can be made without binding a confidence to the wrong word.
#   4. Degraded WER >= MIN_DZ_WER. The collapse has to be visible from the back
#      of the room.
#   5. There is a wrong word (substitution or insertion) whose confidence is
#      >= MIN_WRONG_CONF *and* above the row's own mean. A demo of a SILENT
#      failure cannot lead with an error the model hedged on — if the model
#      flagged it, the model behaved correctly and there is nothing to show.
#
# RANK (descending), applied to whatever survives:
#
#   a. a CRITICAL entity slot was damaged (task_specs.json). This is what makes
#      the failure matter downstream: a destroyed gate number or delivery
#      address is a wrong action, not a wrong word.
#   b. the archived row supports EXACT_CLAIM (see punchline_claim). Ordering by
#      this puts the strongest available sentence at the top of the menu; it
#      does not manufacture it, because the sentence is re-derived live.
#   c. margin = (confidence of the worst wrong word) - (mean confidence of the
#      words it got right). The bigger the margin, the more visible the point.
#
# One entry per clip, N_EXEMPLARS of them.

MIN_DZ_WER = 0.25
MIN_WRONG_CONF = 0.85
N_EXEMPLARS = 6

# The sentence, verbatim. Printed only when the payload supports it. Kept as a
# constant so the test can pin the exact wording and so it is impossible for a
# variant to be mistaken for it.
EXACT_CLAIM = ("more confident in the word it invented than the ones it got "
               "right.")


def aligned_hyp(edits: list, word_confidences: list) -> list[tuple] | None:
    """
    One (op, ref_word, hyp_word, confidence) per HYPOTHESIS word, or None.

    The adapter returns a flat confidence list, not the tokens it came from.
    `classify_errors` returns typed edits, and the edits that produced a
    hypothesis token — match, sub, ins — are exactly the model's output words in
    order. So the two line up 1:1, and if they do not, they are NOT paired: this
    returns None and every consumer degrades to printing the raw list. Binding a
    confidence to the wrong word is precisely the plausible-looking, silent
    error this project exists to catch, and it would land on the one slide the
    whole argument rests on.
    """
    cols = diff_columns(edits)
    slots = [(op, r, h) for op, r, h in cols if op in ("match", "sub", "ins")]
    confs = list(word_confidences or [])
    if not confs or len(slots) != len(confs):
        return None
    return [(op, r, h, float(c)) for (op, r, h), c in zip(slots, confs)]


def critical_tokens(clip_id: str, specs: dict) -> tuple[set[str], list[str]]:
    """Tokens of this clip's CRITICAL task slots, and the slot names."""
    spec = specs.get(clip_id) or {}
    slots = spec.get("slots") or {}
    critical = list(spec.get("critical") or [])
    toks: set[str] = set()
    for name in critical:
        toks |= set(str(slots.get(name, "")).split())
    return toks, sorted(slots)


def punchline_claim(ref_text: str, aligned: list[tuple] | None) -> dict:
    """
    The strongest sentence THIS payload supports. Never more than that.

    Two independent axes, both checked against the response that just arrived:

      INVENTED — the hypothesis token does not appear anywhere in the reference.
        The model emitted a word the speaker never said. A substitution whose
        replacement happens to occur elsewhere in the sentence is a *confusion*,
        not an invention, and calling it one is a claim a presenter would have
        to walk back the moment someone reads the diff.

      LOUDER THAN THE CORRECT WORDS — strictly greater than the maximum
        confidence among the words it got right ("than the ones it got right",
        which is a claim about all of them), or, weaker, greater than their
        mean ("than the average word it got right").

    Returns a dict with `tier`, `supported`, `sentence`, and the numbers behind
    it, so the caller prints and the test asserts against the same structure.
    """
    out = {
        "tier": "none", "supported": False, "sentence": "", "detail": "",
        "wrong": None, "invented": False,
        "max_correct": float("nan"), "mean_correct": float("nan"),
        "n_correct": 0, "n_wrong": 0, "aligned": aligned is not None,
    }
    if aligned is None:
        out["sentence"] = (
            "The per-word confidences did not line up 1:1 with the aligned "
            "words on this run, so no per-word claim is made. The utterance "
            "numbers above are unaffected.")
        out["tier"] = "unaligned"
        return out

    correct = [c for op, _, _, c in aligned if op == "match"]
    wrong = [(op, r, h, c) for op, r, h, c in aligned if op in ("sub", "ins")]
    out["n_correct"], out["n_wrong"] = len(correct), len(wrong)

    if not wrong:
        out["sentence"] = ("Every word the model emitted on this run was "
                           "correct, so there is no wrong word to be confident "
                           "about. Printed because the demo reports what came "
                           "back, not what it hoped for.")
        return out

    best = max(wrong, key=lambda t: t[3])
    op, ref_w, hyp_w, conf = best
    out["wrong"] = list(best)

    ref_tokens = set(str(ref_text).split())
    invented = hyp_w not in ref_tokens
    out["invented"] = invented
    noun = "invented" if invented else "got wrong"

    if not correct:
        out["tier"] = "no-correct-words"
        out["sentence"] = (
            f"It {noun} “{hyp_w}” at {conf:.3f} confidence — and it got "
            f"no words right on this run, so there is nothing to compare that "
            f"against. The comparison this demo usually makes is not available "
            f"here.")
        return out

    mx = max(correct)
    mn = sum(correct) / len(correct)
    out["max_correct"], out["mean_correct"] = mx, mn

    if conf > mx:
        out["tier"] = "exact"
        out["supported"] = True
        tail = EXACT_CLAIM if invented else \
            "more confident in the word it got wrong than the ones it got right."
        out["sentence"] = "The model was " + tail
        out["detail"] = (f"{conf:.3f} on “{hyp_w}” against {mx:.3f}, the "
                         f"highest confidence it reported on any of the "
                         f"{len(correct)} words it got right.")
        return out

    if conf > mn:
        out["tier"] = "average"
        out["supported"] = True
        out["sentence"] = (
            f"The model was more confident in the word it {noun} than in the "
            f"average word it got right.")
        out["detail"] = (f"{conf:.3f} on “{hyp_w}” against a mean of "
                         f"{mn:.3f} over the {len(correct)} words it got right "
                         f"— though {mx:.3f} was its highest, so the stronger "
                         f"form of this claim does not hold on this run and is "
                         f"not made.")
        return out

    out["tier"] = "hedged"
    out["sentence"] = (
        f"On this run the model hedged: the word it {noun} "
        f"(“{hyp_w}”) came back at {conf:.3f}, BELOW its {mn:.3f} mean on "
        f"the words it got right.")
    out["detail"] = ("That is the model behaving correctly, and it is printed "
                     "rather than hidden — a demo about confidently-wrong "
                     "output has to be able to show the model getting it right.")
    return out


def _row_facts(row: dict) -> dict:
    """A master.csv row, in the same shape `score()` returns for a live call."""
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


def _is_failed_row(row: dict) -> bool:
    return str(row.get("failed", "")).strip().lower() in ("true", "1", "yes")


def select_exemplars(dead_zone_rows: list[dict], master_rows: dict,
                     clean_facts: dict, refs: dict, specs: dict,
                     n: int = N_EXEMPLARS) -> list[dict]:
    """
    Rank every (clip, dead-zone condition) pair and return the top `n`.

    Pure function of the four artifacts, so the test can re-derive it with a
    different implementation and the menu cannot drift away from the grid.
    `master_rows` is {condition_name: {clip_id: row}}.
    """
    cands: list[dict] = []
    for cond_row in dead_zone_rows:
        cond = cond_row["condition_name"]
        for clip_id, row in (master_rows.get(cond) or {}).items():
            if _is_failed_row(row):
                continue
            ref = refs.get(clip_id)
            if not ref:
                continue
            clean = clean_facts.get(clip_id)
            if clean is None or clean["wer"] != 0.0:
                continue                              # filter 2: the control must be clean
            wer, conf = _f(row["wer"]), _f(row["mean_conf"])
            if not (wer == wer and conf == conf) or wer < MIN_DZ_WER:
                continue                              # filter 4
            facts = _row_facts(row)
            aligned = aligned_hyp(facts["edits"], facts["word_confidences"])
            if aligned is None:
                continue                              # filter 3
            wrong = [t for t in aligned if t[0] in ("sub", "ins")]
            correct = [c for op, _, _, c in aligned if op == "match"]
            if not wrong or not correct:
                continue
            best = max(wrong, key=lambda t: t[3])
            if not (best[3] >= MIN_WRONG_CONF and best[3] > conf):
                continue                              # filter 5
            crit_toks, slot_names = critical_tokens(clip_id, specs)
            damaged = sorted({t[1] for t in aligned
                              if t[0] == "sub" and t[1] in crit_toks}
                             | {e[1] for e in facts["edits"]
                                if e[0] == "del" and len(e) > 1 and e[1] in crit_toks})
            claim = punchline_claim(ref, aligned)
            mean_correct = sum(correct) / len(correct)
            cands.append({
                "clip_id": clip_id,
                "condition_name": cond,
                "ref": ref,
                "slots": slot_names,
                "critical_damaged": damaged,
                "grid_wer": wer,
                "grid_mean_conf": conf,
                "grid_worst_wrong": list(best),
                "grid_claim_tier": claim["tier"],
                "_key": (1 if damaged else 0,
                         1 if claim["tier"] == "exact" else 0,
                         best[3] - mean_correct),
            })

    cands.sort(key=lambda c: c["_key"], reverse=True)
    picked, seen = [], set()
    for c in cands:
        if c["clip_id"] in seen:
            continue
        seen.add(c["clip_id"])
        c.pop("_key")
        picked.append(c)
        if len(picked) >= n:
            break
    return picked


# =========================================================================
# PREPARE — bake the wavs and the curated set. Offline. No key.
# =========================================================================

def _load_clean_facts(refs: dict) -> dict:
    """Score every clean transcript against its reference, once."""
    from deadzone.audio_pipeline import classify_errors
    out = {}
    if not CLEAN_TRANSCRIPTS.is_file():
        return out
    for line in open(CLEAN_TRANSCRIPTS):
        rec = json.loads(line)
        cid = rec["id"]
        if cid not in refs:
            continue
        s = classify_errors(refs[cid], rec.get("transcript", "") or "")
        c = s["counts"]
        out[cid] = {
            "transcript": rec.get("transcript", "") or "",
            "wer": float(s["wer"]),
            "mean_conf": _f(rec.get("mean_conf")),
            "n_ref": int(s["n_ref"]),
            "n_sub": int(c.get("sub", 0)),
            "n_del": int(c.get("del", 0)),
            "n_ins": int(c.get("ins", 0)),
            "edits": [list(e) for e in s["edits"]],
            "word_confidences": list(rec.get("word_confidences") or []),
        }
    return out


def _master_rows_for(conditions: set[str]) -> dict:
    """One streaming pass over master.csv; no pandas on the demo path."""
    csv.field_size_limit(10 ** 9)
    out: dict[str, dict] = {c: {} for c in conditions}
    with open(MASTER, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] != MODEL:
                continue
            if row["condition_name"] in out:
                out[row["condition_name"]][row["clip_id"]] = row
    return out


def build_cache(n: int = N_EXEMPLARS, verbose: bool = True) -> dict:
    """
    Select the exemplars, compose their audio, and freeze both to disk.

    Writes ONLY wavs and one JSON file, all under `results/demo/hero/`. It
    authors no prose and overwrites no hand-written document (SPEC J.1/J.2:
    a generator that cannot tell regenerable output from an irreplaceable
    record must refuse — this one is never handed the latter).
    """
    for p in (MASTER, DEAD_ZONES, CLEAN_TRANSCRIPTS, Path("recording_manifest.csv")):
        if not p.is_file():
            raise SystemExit(f"demo_hero: missing {p} — run the grid first (SPEC A.R4)")

    refs = load_manifest_refs()
    specs = json.loads(TASK_SPECS.read_text()) if TASK_SPECS.is_file() else {}
    dz_rows = load_dead_zones()
    if not dz_rows:
        raise SystemExit(
            f"demo_hero: no `dead_zone` rows for {MODEL} in {DEAD_ZONES}. "
            "Only measured dead zones may be demoed; a `silence_driven` or "
            "`mute_zone` row is a different finding.")
    dz_index = {r["condition_name"]: r for r in dz_rows}
    master = _master_rows_for(set(dz_index))
    clean_facts = _load_clean_facts(refs)

    picked = select_exemplars(dz_rows, master, clean_facts, refs, specs, n=n)
    if not picked:
        raise SystemExit("demo_hero: no clip passed the selection filters — "
                         "see SELECTION in this module and loosen deliberately, "
                         "not silently.")

    # heavy imports stay off the on-stage path
    import soundfile as sf
    from deadzone.conditions import Condition, DiskAssetLibrary, apply_condition
    from scripts.run_experiment import load_clip, write_degraded_wav

    HERO_DIR.mkdir(parents=True, exist_ok=True)
    assets = DiskAssetLibrary(root="data", target_fs=FS)

    entries = []
    for rank, c in enumerate(picked, start=1):
        cid, cond_name = c["clip_id"], c["condition_name"]
        raw = load_clip(cid, target_fs=FS)
        clean_p = HERO_DIR / f"{cid}_clean.wav"
        if not clean_p.is_file():
            sf.write(str(clean_p), raw, FS, subtype="PCM_16")
        deg_p = HERO_DIR / f"{cid}__{cond_name}.wav"
        if not deg_p.is_file():
            # apply_condition is seeded from the condition NAME, so this file is
            # bit-identical to the one that produced the archived row. That is
            # what makes the reproduction check a reproduction check.
            deg = apply_condition(raw, Condition.from_name(cond_name), assets, FS)
            write_degraded_wav(deg_p, deg, FS)

        cond = Condition.from_name(cond_name)
        agg = dz_index[cond_name]
        entry = dict(c)
        entry.update({
            "rank": rank,
            "duration_s": round(len(raw) / FS, 2),
            "audio_clean": str(clean_p),
            "audio_degraded": str(deg_p),
            "clean": clean_facts[cid],
            "grid": _row_facts(master[cond_name][cid]),
            "condition": {
                "name": cond_name,
                "rt60": cond.rt60, "snr_db": cond.snr_db,
                "noise_type": cond.noise_type, "codec": cond.codec,
                "mic_rolloff": cond.mic_rolloff,
                "rt60_measured": _f(agg.get("rt60_measured")),
                "category": agg.get("category", "dead_zone"),
                "mean_conf": _f(agg.get("mean_conf")),
                "wer_spoke": _f(agg.get("wer_spoke")),
                "wer_all_clips": _f(agg.get("wer_all_clips"), _f(agg.get("wer"))),
                "gap": _f(agg.get("gap")),
                "n_clips": int(_f(agg.get("n_clips"), 0)),
                "n_silent": int(_f(agg.get("n_silent"), 0)),
                "n_ref_total": int(_f(agg.get("n_ref_total"), 0)),
            },
        })
        entries.append(entry)
        if verbose:
            print(f"  {rank}. {cid} in {cond_name}: "
                  f"clean WER {entry['clean']['wer']:.2f} -> "
                  f"grid WER {entry['grid']['wer']:.2f} "
                  f"(claim tier on the archived row: {c['grid_claim_tier']})")

    cache = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "fs": FS,
        "source_master": str(MASTER),
        "n_dead_zone_conditions": len(dz_rows),
        "selection": {
            "min_dz_wer": MIN_DZ_WER,
            "min_wrong_conf": MIN_WRONG_CONF,
            "n_requested": n,
        },
        "default": entries[0]["clip_id"],
        "exemplars": entries,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))
    if verbose:
        print(f"  wrote {CACHE_PATH}  ({len(entries)} exemplars, "
              f"{len(entries) * 2} wavs in {HERO_DIR})")
    return cache


def load_cache(auto_build: bool = True, verbose: bool = True) -> dict:
    if CACHE_PATH.is_file():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            if verbose:
                print("demo_hero: cache unreadable, rebuilding.", file=sys.stderr)
    if not auto_build:
        raise SystemExit(f"demo_hero: no cache at {CACHE_PATH} — run `make demo-prep`")
    if verbose:
        print("demo_hero: no cache yet, baking it (offline, no API key, ~15 s)...")
    return build_cache(verbose=verbose)


# =========================================================================
# input — never hangs, never raises, never loops forever
# =========================================================================

class Turn:
    """Whether stdin is still usable, and why it stopped if it is not."""

    def __init__(self, interactive: bool):
        self.interactive = interactive
        self.eof = False

    @property
    def live(self) -> bool:
        return self.interactive and not self.eof


def ask(prompt: str, turn: Turn) -> str | None:
    """
    One line of input, or None.

    None means "nobody is there": EOF (a pipe, `< /dev/null`, a test), or Ctrl-C
    (the presenter cutting the segment short). Neither may raise and neither may
    leave the caller looping — both flip the turn out of live mode so every
    later prompt short-circuits without touching stdin again.
    """
    if not turn.live:
        return None
    try:
        return input(prompt).strip().lower()
    except EOFError:
        turn.eof = True
        print()
        return None
    except KeyboardInterrupt:
        turn.eof = True
        print()
        return None


# =========================================================================
# panels
# =========================================================================

def preamble(ink: Ink, width: int, live: bool, audio: bool, turn: Turn) -> None:
    """
    What is about to happen, said before anything happens. NO audio plays until
    this returns, which is the point of it: the room gets a warning that sound
    is coming and the presenter gets a beat to set up the argument.
    """
    say("")
    say(ink.inv(pad("  DEADZONE — is the model wrong, or is it wrong AND CONFIDENT?  ", width)))
    say("")
    for line in textwrap.wrap(
            "One recorded sentence, played twice. First the raw recording. Then "
            "the same sentence pushed through a measured dead zone — a real "
            "room, real recorded noise at a chosen level, a real telephony "
            "codec. Each version is transcribed "
            + ("LIVE by Deepgram nova-3, right now" if live else
               "from the archived grid run (replay mode — no network)")
            + ", and you see the per-word confidences it returns.",
            width=width - 4):
        say("  " + line)
    say("")
    say("  " + ink.bold("Watch the two numbers move at different rates.")
        + ink.dim("  The transcript is going to"))
    say(ink.dim("  collapse. The model's own confidence is barely going to notice."))
    say("")
    if audio:
        say("  " + ink.yellow("AUDIO WILL PLAY") + ink.dim(
            " — two short clips, a few seconds each. The second one"))
        say(ink.dim("  is deliberately unpleasant. Check your volume now."))
    else:
        say(ink.dim("  (playback is off: --no-audio)"))
    say("")
    if turn.live:
        ask(ink.bold("  press enter when you are ready  "), turn)
    else:
        say(ink.dim("  (no terminal on stdin — running straight through)"))


def menu(cache: dict, ink: Ink, width: int, turn: Turn,
         rng: random.Random) -> dict | None:
    """
    Pick one exemplar. Returns None only if the presenter asked to quit.

    Every entry is a MEASURED dead-zone cell for this model; the menu is the
    output of `select_exemplars`, so it tracks the grid rather than a list
    somebody typed.
    """
    ex = cache["exemplars"]
    say("")
    say(ink.dim(rule(width)))
    say(ink.bold("  PICK A CLIP") + ink.dim(
        f"   ({len(ex)} clips, each one measured in a dead zone)"))
    say("")
    for i, e in enumerate(ex, start=1):
        dmg = ", ".join(e["critical_damaged"][:3])
        say(f"   {ink.bold(str(i))}  {pad(e['clip_id'], 5)}"
            + pad(ink.dim("/".join(e["slots"])), 22)
            + f"“{e['ref']}”")
        if dmg:
            # the same 33-column prefix as the line above, so the two align
            say(" " * 33 + ink.red(f"destroys: {dmg}"))
    say("")
    say(ink.dim("   r  pick one at random        q  quit"))
    say("")

    default = next((e for e in ex if e["clip_id"] == cache.get("default")), ex[0])
    for _ in range(5):                                # never loop forever
        a = ask(ink.bold(f"  choose [1-{len(ex)}, r, q]  "
                         f"(enter = {default['clip_id']})  "), turn)
        if a is None:
            say(ink.dim(f"  (no input — using {default['clip_id']})"))
            return default
        if a == "":
            return default
        if a == "q":
            return None
        if a == "r":
            return rng.choice(ex)
        if a.isdigit() and 1 <= int(a) <= len(ex):
            return ex[int(a) - 1]
        for e in ex:
            if a == e["clip_id"]:
                return e
        say(ink.dim("  (didn't catch that)"))
    return default


def show_words(facts: dict, ink: Ink, width: int, flag_wrong: bool) -> None:
    """
    The per-word confidences, as returned, one row per word.

    Pairing goes through the EDIT ALIGNMENT, not through `transcript.split()`,
    so each confidence is attached to the slot the scorer actually assigned it
    and the flagged words are exactly the ones the punchline is computed from.
    The screen and the claim therefore cannot disagree. If the alignment does
    not hold, this refuses to pair and prints the raw list instead.
    """
    confs = facts.get("word_confidences") or []
    aligned = aligned_hyp(facts.get("edits") or [], confs)
    say(ink.dim("      per-word confidence, exactly as returned:"))
    if aligned is None:
        say(ink.dim(f"      ({len(confs)} confidences vs the aligned word count — "
                    f"not pairing them; raw list follows)"))
        say("      " + ", ".join(f"{float(c):.4f}" for c in confs))
        return

    cell = max((len(h) for _, _, h, _ in aligned), default=4)
    per_row = max(1, (width - 8) // (cell + 20))
    line: list[str] = []
    for op, _ref_w, hyp_w, c in aligned:
        bad = flag_wrong and op in ("sub", "ins")
        mark = ink.red("!") if bad else " "
        word = ink.red(hyp_w) if bad else hyp_w
        line.append(f"{mark} {pad(word, cell)} {c:.4f} {ink.dim(conf_bar(c))}")
        if len(line) == per_row:
            say("      " + "  ".join(line))
            line = []
    if line:
        say("      " + "  ".join(line))


def show_stage(title: str, subtitle: str, facts: dict, ink: Ink, width: int,
               elapsed: float | None, danger: bool) -> None:
    accent = ink.red if danger else ink.green
    say("")
    say(ink.bold(title))
    if subtitle:
        for ln in subtitle.splitlines():
            say(ink.dim(ln))
    if elapsed is not None:
        say(ink.dim(f"      → one call to Deepgram … {elapsed:.2f} s round trip"))
    say("")
    say("      " + ink.dim("hyp  ")
        + (facts["transcript"] or ink.red("(empty transcript)")))
    say("")
    show_words(facts, ink, width, flag_wrong=danger)
    say("")
    conf = facts["mean_conf"]
    conf_s = "n/a" if conf != conf else f"{conf:.3f}"
    say("      " + ink.dim("WER ") + accent(ink.bold(f"{facts['wer']:.3f}"))
        + ink.dim("     mean word confidence ") + accent(ink.bold(conf_s))
        + ink.dim(f"     sub {facts['n_sub']} del {facts['n_del']} "
                  f"ins {facts['n_ins']} / {facts['n_ref']} ref"))


def collapse_narration(clean: dict, dz: dict) -> list[str]:
    """
    The sentence under the table, DERIVED from the two rows above it.

    The literal this replaces read "The model's self-report barely moved." It
    was written for one exemplar and is simply false for others — on the clip
    that leads the curated set confidence falls 0.293, which nobody would call
    "barely". A caption that can contradict the table directly above it is the
    defect SPEC J.7 found in the demo script, and the fix there was the same:
    compute the narration from the numbers rather than remembering it.

    The comparison that carries no such risk is the one the project is about.
    If confidence behaved like an accuracy estimate it would read about
    `1 - WER`. It does not. The surplus is the finding, and it is arithmetic.
    """
    d_wer = dz["wer"] - clean["wer"]
    d_conf = dz["mean_conf"] - clean["mean_conf"]
    out: list[str] = []
    if abs(d_conf) <= 0.5 * abs(d_wer):
        out.append("The transcript collapsed. The model's self-report moved a "
                   "fraction as far.")
    else:
        out.append("The transcript collapsed, and confidence did fall with it — "
                   "but nothing like as far.")
    implied = min(1.0, max(0.0, 1.0 - dz["wer"]))
    conf = dz["mean_conf"]
    if conf == conf:                                  # not NaN
        surplus = conf - implied
        if surplus > 0:
            out.append("If confidence tracked accuracy it would read about "
                       "{:.3f} on this transcript. It reads {:.3f} — {:+.3f}."
                       .format(implied, conf, surplus))
        else:
            out.append("On this run confidence sits at {:.3f} against an "
                       "accuracy-implied {:.3f}, so it is NOT overstating "
                       "itself here."
                       .format(conf, implied))
    return out


def show_collapse(clean: dict, dz: dict, ink: Ink, width: int) -> None:
    say("")
    say(ink.dim(rule(width)))
    say(ink.bold("  WHAT JUST HAPPENED"))
    say("")
    d_wer = dz["wer"] - clean["wer"]
    d_conf = dz["mean_conf"] - clean["mean_conf"]
    say("    " + pad("", 14) + pad(ink.dim("raw"), 12) + pad(ink.dim("dead zone"), 12)
        + ink.dim("change"))
    say("    " + pad("WER", 14) + pad(f"{clean['wer']:.3f}", 12)
        + pad(ink.red(f"{dz['wer']:.3f}"), 12) + ink.red(f"{d_wer:+.3f}"))
    say("    " + pad("confidence", 14) + pad(f"{clean['mean_conf']:.3f}", 12)
        + pad(ink.yellow(f"{dz['mean_conf']:.3f}"), 12) + ink.yellow(f"{d_conf:+.3f}"))
    say("")
    lines = collapse_narration(clean, dz)
    say("    " + ink.bold(lines[0]))
    for extra in lines[1:]:
        for ln in textwrap.wrap(extra, width=max(40, width - 6)):
            say("    " + ink.dim(ln))
    say("")
    for line in render_diff(dz["edits"], width - 4, ink, dz["word_confidences"]):
        say("    " + line.rstrip())


def show_punchline(ref: str, dz: dict, ink: Ink, width: int) -> dict:
    """
    The sentence, and only the sentence this payload supports.

    Returns the claim dict so a caller (and the test) can assert on the tier
    rather than on the rendered string.
    """
    claim = punchline_claim(ref, aligned_hyp(dz.get("edits") or [],
                                             dz.get("word_confidences") or []))
    say("")
    if claim["wrong"]:
        op, ref_w, hyp_w, conf = claim["wrong"]
        if op == "ins":
            lead = (f'    It emitted {ink.red(chr(8220) + hyp_w + chr(8221))} where the '
                    f'speaker said nothing at all,')
        else:
            lead = (f'    It emitted {ink.red(chr(8220) + hyp_w + chr(8221))} where the '
                    f'speaker said {ink.yellow(chr(8220) + ref_w + chr(8221))},')
        say(lead)
        say(f"    and reported {ink.red(f'{conf:.3f}')} confidence in it.")
        say("")
    accent = ink.bold if claim["supported"] else ink.yellow
    for ln in textwrap.wrap(claim["sentence"], width=max(40, width - 6)):
        say("    " + accent(ln))
    if claim["detail"]:
        say("")
        for ln in textwrap.wrap(claim["detail"], width=max(40, width - 6)):
            say("    " + ink.dim(ln))
    return claim


def show_condition_context(entry: dict, ink: Ink, width: int) -> None:
    """
    The one clip on screen is not the finding; the cell is. One line, with the
    population named — a confidence and a WER may only be printed side by side
    when they are averaged over the SAME clips (SPEC Appendix G).
    """
    cond = entry["condition"]
    n_clips = int(cond.get("n_clips") or 0)
    n_sil = int(cond.get("n_silent") or 0)
    spoke = n_clips - n_sil
    wer = cond.get("wer_spoke")
    if wer != wer:                                    # NaN
        return
    say("")
    say(ink.dim(rule(width)))
    say(ink.bold("  AND IT IS NOT ONE UNLUCKY CLIP"))
    say("")
    if n_sil == 0:
        pop = (f"all {n_clips} clips in this same condition "
               f"({cond['n_ref_total']} reference words)")
    else:
        verb, pron = ("carries", "it") if n_sil == 1 else ("carry", "they")
        pop = (f"the {spoke} of {n_clips} clips this condition produced words on "
               f"({n_sil} returned an empty transcript and {verb} no confidence, "
               f"so {pron} sits out of both averages)")
    for ln in textwrap.wrap(f"Averaged over {pop}:", width=max(40, width - 6)):
        say("    " + ln)
    say("      mean confidence {}   WER {}   gap {}".format(
        ink.bold(f"{cond['mean_conf']:.3f}"), ink.bold(f"{wer:.3f}"),
        ink.bold(f"{cond['gap']:+.3f}")))
    say(ink.dim("      (both averages are over those same clips, so they subtract)"))


# Drift the reproduction check tolerates before it calls the run a divergence.
# Vendor confidences move in the third decimal between calls on identical audio,
# and the archived row was written days earlier. Calling that "a different
# experiment" would make the presenter explain a non-difference under time
# pressure and would cry wolf on the branch that exists for a REAL divergence.
REPRO_WER_TOL = 0.05
REPRO_CONF_TOL = 0.05


def show_reproduction(dz: dict, grid: dict, ink: Ink, width: int, live: bool,
                      fell_back: bool = False) -> None:
    """
    THE LIVE NUMBER IS THE RESULT. The grid row is corroboration, printed second
    and labelled as such. It is never allowed to stand in for the live one —
    that would be a cached number wearing a live number's clothes, which is the
    same move as quoting a summary instead of an artifact.
    """
    say("")
    say(ink.dim(rule(width)))
    say(ink.bold("  DOES IT REPRODUCE?"))
    say("")
    d_wer = dz["wer"] - grid["wer"]
    d_conf = dz["mean_conf"] - grid["mean_conf"]
    tag = ink.bold("live  ") if live else ink.bold("shown ")
    lead = ("← what you just watched arrive" if live
            else "← what stage 2 above showed you, from the archive")
    say("    " + tag + "  WER " + ink.bold("{:.3f}".format(dz["wer"]))
        + "   confidence " + ink.bold("{:.3f}".format(dz["mean_conf"]))
        + "   " + ink.dim(lead))
    say("    grid    WER {:.3f}   confidence {:.3f}   ".format(
        grid["wer"], grid["mean_conf"])
        + ink.dim("← the archived row for this same clip and condition"))
    say("")
    if not live:
        if fell_back:
            say(ink.dim("    The live call did not happen, so stage 2 IS this archived "
                        "row — they"))
            say(ink.dim("    agree by construction and this panel proves nothing today. "
                        "Said out"))
            say(ink.dim("    loud rather than left to look like a reproduction."))
        else:
            say(ink.dim("    Replay mode reads that same archived row, so these agree by "
                        "construction."))
            say(ink.dim("    Run without --replay to make the comparison mean something."))
        return
    if abs(d_wer) <= REPRO_WER_TOL and abs(d_conf) <= REPRO_CONF_TOL:
        # 4 dp, deliberately: at 3 dp a genuine agreement prints as "within
        # 0.000", which reads as a rounded-away difference rather than as the
        # tight agreement it is. This is the one line making a precision claim.
        say(ink.dim(f"    Reproduced — within {abs(d_wer):.4f} WER and "
                    f"{abs(d_conf):.4f} confidence of the archived row."))
        say(ink.dim("    Same bytes, same model literal, days after the grid ran."))
    else:
        say(ink.yellow(f"    MOVED — {d_wer:+.3f} WER, {d_conf:+.3f} confidence "
                       f"against the archive."))
        say(ink.dim("    Worth saying out loud rather than hiding: a commercial model"))
        say(ink.dim("    literal is updated server-side, so a live call is not"))
        say(ink.dim("    automatically the same experiment as the grid. The live number"))
        say(ink.dim("    above is the one on screen; the archive is the corroboration."))


# =========================================================================
# the beat
# =========================================================================

def fallback_notice(reason: str) -> str:
    return (f"LIVE CALL SKIPPED — {reason}. Falling back to the archived "
            "measurements for this exact clip and condition. What you would "
            "have seen live: the same two transcripts and the same per-word "
            "confidences, arriving off the wire instead of off disk. Nothing "
            "about the finding depends on the call.")


def run_one(entry: dict, ink: Ink, width: int, want_live: bool, timeout: float,
            audio: bool, turn: Turn) -> int:
    """One full beat on one exemplar. Always returns 0."""
    ref = entry["ref"]
    cond = entry["condition"]
    clean_facts, dz_facts = entry["clean"], entry["grid"]
    timings: dict = {}
    live_ok, note, provenance = False, None, "not read (replay)"

    if want_live:
        present_key, provenance = load_credentials()
        if not present_key:
            note = fallback_notice("DEEPGRAM_API_KEY not found in the environment "
                                   "or .env")

    say("")
    say(ink.dim(rule(width)))
    say(f"  clip       {ink.bold(entry['clip_id'])}   “{ref}”")
    say(f"  condition  {cond['name']}")
    say(ink.dim(f"             rt60 {cond['rt60']:g}s (measured "
                f"{cond['rt60_measured']:.2f}s) · SNR {cond['snr_db']:g} dB · "
                f"{cond['noise_type']} · {cond['codec']} · mic rolloff "
                f"{cond['mic_rolloff']:g}"))
    say(f"  model      {MODEL}   ·   Deepgram pre-recorded endpoint")
    # FUTURE TENSE, deliberately. This line is printed BEFORE either call is
    # made, so it can only state an intention. An earlier draft asserted "two
    # live calls, nothing cached" here and then printed LIVE CALL SKIPPED four
    # lines below it — a caption contradicting the screen, which is the defect
    # SPEC J.7 found in the demo script. What actually happened is stated where
    # it is known: the round-trip line on each stage, and the reproduction panel.
    say("  calls      " + ("two live calls, about to be made in front of you"
                           if (want_live and not note)
                           else "none — the archived measurements (see above)"
                           if note else "none — the archived measurements (replay)"))
    say(f"  credential {provenance}")
    if note:
        say("")
        for ln in textwrap.wrap(note, width=max(40, width - 4)):
            say("  " + ink.yellow(ln))
    say(ink.dim(rule(width)))

    # ---- stage 1: the raw recording -------------------------------------
    play(entry["audio_clean"], ink, audio)
    if want_live and not note:
        try:
            t0 = time.monotonic()
            res = transcribe_live(Path(entry["audio_clean"]), timeout)
            timings["clean"] = time.monotonic() - t0
            clean_facts = score(ref, res["transcript"], res["word_confidences"])
        except LiveCallFailed as exc:                # already redacted at construction
            note = fallback_notice(str(exc))
        except Exception as exc:                     # noqa: BLE001 — stage must survive
            note = fallback_notice(redact(f"{type(exc).__name__}: {exc}"))
        if note:
            say("")
            for ln in textwrap.wrap(note, width=max(40, width - 4)):
                say("  " + ink.yellow(ln))
            timings.pop("clean", None)
            clean_facts = entry["clean"]

    show_stage(
        "  [1]  RAW RECORDING  —  the control",
        "       No degradation at all. There is no 'clean' Condition: "
        "apply_condition always\n"
        "       applies a room and always mixes noise, so the only true null is "
        "the untouched file.",
        clean_facts, ink, width, timings.get("clean"), danger=False)

    # ---- the pause -------------------------------------------------------
    for _ in range(4):
        a = ask(ink.bold("\n  ready for the degraded version? [enter]")
                + ink.dim("   ·   replay the clean clip [r]  "), turn)
        if a == "r":
            play(entry["audio_clean"], ink, audio)
            continue
        break

    # ---- stage 2: the measured dead zone ---------------------------------
    play(entry["audio_degraded"], ink, audio)
    if want_live and not note:
        try:
            t0 = time.monotonic()
            res = transcribe_live(Path(entry["audio_degraded"]), timeout)
            timings["degraded"] = time.monotonic() - t0
            dz_facts = score(ref, res["transcript"], res["word_confidences"])
            live_ok = True
        except LiveCallFailed as exc:
            note = fallback_notice(str(exc))
        except Exception as exc:                     # noqa: BLE001
            note = fallback_notice(redact(f"{type(exc).__name__}: {exc}"))
        if note:
            say("")
            for ln in textwrap.wrap(note, width=max(40, width - 4)):
                say("  " + ink.yellow(ln))
            timings.pop("degraded", None)
            dz_facts = entry["grid"]

    show_stage(
        "  [2]  DEAD ZONE  —  " + cond["name"],
        "       A real measured room, real recorded noise at that SNR, and a "
        "real codec\n       round-trip — every ingredient real, only the "
        "assembly controlled.",
        dz_facts, ink, width, timings.get("degraded"), danger=True)

    show_collapse(clean_facts, dz_facts, ink, width)
    show_punchline(ref, dz_facts, ink, width)
    show_condition_context(entry, ink, width)
    show_reproduction(dz_facts, entry["grid"], ink, width, live=live_ok,
                      fell_back=bool(want_live and not live_ok))
    say("")
    return 0


def run(cache: dict, ink: Ink, width: int, want_live: bool, timeout: float,
        audio: bool, pick: str | None, use_random: bool, once: bool,
        rng: random.Random) -> int:
    interactive = sys.stdin is not None and sys.stdin.isatty()
    turn = Turn(interactive)
    ex = cache["exemplars"]

    preamble(ink, width, live=want_live, audio=audio, turn=turn)

    while True:
        if pick:
            entry = next((e for e in ex if e["clip_id"] == pick), None)
            if entry is None:
                raise SystemExit(
                    f"demo_hero: {pick!r} is not in the curated set. Have: "
                    + ", ".join(e["clip_id"] for e in ex)
                    + "   (--list shows why each one is there)")
        elif use_random:
            entry = rng.choice(ex)
        else:
            entry = menu(cache, ink, width, turn, rng)
            if entry is None:
                say("")
                return 0

        run_one(entry, ink, width, want_live, timeout, audio, turn)

        if once or not turn.live:
            return 0
        a = ask(ink.bold("  run another? [enter]") + ink.dim("   ·   done [q]  "), turn)
        if a is None or a == "q":
            say("")
            return 0
        pick, use_random = None, False


# =========================================================================
# preflight / listing
# =========================================================================

def show_list(cache: dict, ink: Ink) -> int:
    say("")
    say(ink.bold("  the curated set — every entry is a MEASURED dead-zone cell"))
    say(ink.dim(f"  filters: clean WER 0.000 · degraded WER ≥ "
                f"{MIN_DZ_WER} · worst wrong word ≥ {MIN_WRONG_CONF} "
                f"and above the utterance mean"))
    say("")
    for e in cache["exemplars"]:
        op, ref_w, hyp_w, conf = e["grid_worst_wrong"]
        say(f"  {e['rank']}. {pad(e['clip_id'], 5)} {pad(e['condition_name'], 44)}")
        say(f"       archived: WER {e['grid_wer']:.3f}  conf {e['grid_mean_conf']:.3f}  "
            f"worst wrong word “{hyp_w}” @ {conf:.3f} ({op})  "
            f"claim tier {e['grid_claim_tier']}")
        if e["critical_damaged"]:
            say(ink.dim(f"       critical slot tokens destroyed: "
                        f"{', '.join(e['critical_damaged'])}"))
        say(ink.dim(f"       “{e['ref']}”"))
        say("")
    return 0


def preflight(ink: Ink) -> int:
    """Can this run live right now — and does the archived fallback exist?"""
    ok_live = ok_fallback = True

    def chk(cond: bool, label: str, hint: str = "", fatal_to: str = "live") -> None:
        nonlocal ok_live, ok_fallback
        cond = bool(cond)
        mark = ink.green("  ok  ") if cond else ink.red(" MISS ")
        tail = "" if cond or not hint else ink.dim("   -> " + hint)
        say(f"{mark}  {label}{tail}")
        if not cond:
            if fatal_to in ("live", "both"):
                ok_live = False
            if fatal_to in ("fallback", "both"):
                ok_fallback = False

    say("")
    say(ink.bold("  demo_hero preflight"))
    say("")
    present_key, provenance = load_credentials()
    chk(present_key, f"credential: {provenance}",
        "export DEEPGRAM_API_KEY or put it in .env")
    chk(CACHE_PATH.is_file(), str(CACHE_PATH), "make demo-prep", fatal_to="both")
    cache = None
    if CACHE_PATH.is_file():
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            chk(False, "the curated set parses as JSON", "make demo-prep",
                fatal_to="both")
    if cache:
        chk(bool(cache.get("exemplars")), "the curated set is non-empty",
            "make demo-prep", fatal_to="both")
        for e in cache.get("exemplars") or []:
            for key in ("audio_clean", "audio_degraded"):
                p = Path(e[key])
                chk(p.is_file(), f"audio {p.name}", "make demo-prep", fatal_to="both")
        names = {r["condition_name"] for r in load_dead_zones()} \
            if DEAD_ZONES.is_file() else set()
        stale = sorted({e["condition_name"] for e in (cache.get("exemplars") or [])}
                       - names)
        chk(not stale, "every demoed condition is still a MEASURED dead zone",
            f"stale: {', '.join(stale)} — make demo-prep", fatal_to="both")
    chk(MASTER.is_file(), str(MASTER), "run the grid (SPEC A.R4)", fatal_to="both")
    chk(find_player() is not None, "an audio player on PATH",
        "the demo degrades to silent, but you lose the visceral half",
        fatal_to="neither")

    say("")
    say("  " + (ink.green("READY TO RUN LIVE.") if ok_live else
                ink.yellow("NOT READY FOR THE LIVE CALLS — it will fall back to "
                           "the archive.")))
    say("  " + (ink.green("The archived fallback is present, so the demo runs "
                          "either way.") if ok_fallback else
                ink.red("The archived FALLBACK is missing too — run "
                        "`make demo-prep`.")))
    say("")
    return 0 if ok_fallback else 1


# =========================================================================
# main
# =========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="One clip, played and transcribed live twice: raw, then in "
                    "a measured dead zone, with the per-word confidences as "
                    "they come back.")
    ap.add_argument("--clip", default=None, help="skip the menu and use this clip id")
    ap.add_argument("--random", action="store_true",
                    help="skip the menu and pick at random from the curated set")
    ap.add_argument("--replay", "--offline", "--cached", dest="replay",
                    action="store_true",
                    help="run the whole beat from the archive; no network at all")
    ap.add_argument("--once", action="store_true", help="do not offer 'run another?'")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"per-call deadline in seconds (default {DEFAULT_TIMEOUT:g})")
    ap.add_argument("--no-audio", dest="audio", action="store_false",
                    help="skip playback (everything still prints)")
    ap.add_argument("--prepare", action="store_true",
                    help="select the exemplars and compose their audio, then exit")
    ap.add_argument("--check", action="store_true",
                    help="preflight: can this run live right now?")
    ap.add_argument("--list", dest="listing", action="store_true",
                    help="print the curated set and exit")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the random pick (rehearsal/testing)")
    ap.add_argument("--no-color", dest="color", action="store_false",
                    help="force plain output")
    args = ap.parse_args(argv)

    color = (args.color and sys.stdout.isatty()
             and os.environ.get("TERM") not in (None, "dumb"))
    ink = Ink(color)
    width = term_width()

    if args.prepare:
        print("demo_hero --prepare: selecting exemplars from the measured grid "
              "and composing their audio (offline, no API key)...")
        build_cache()
        print("demo_hero: ready. `make demo` will now spend its whole budget on "
              "the two calls.")
        return 0

    if args.check:
        return preflight(ink)

    try:
        cache = load_cache()
    except SystemExit:
        raise
    except Exception as exc:                          # noqa: BLE001
        say(ink.yellow(redact(f"demo_hero: could not load the curated set — "
                              f"{type(exc).__name__}: {exc}")))
        say(ink.dim("           run `make demo-prep`."))
        return 0

    if args.listing:
        return show_list(cache, ink)

    rng = random.Random(args.seed)
    try:
        return run(cache, ink, width, want_live=not args.replay,
                   timeout=args.timeout, audio=args.audio, pick=args.clip,
                   use_random=args.random, once=args.once, rng=rng)
    except SystemExit:
        raise
    except Exception as exc:                          # noqa: BLE001 — LAST resort
        # Anything that got past the per-stage handling still exits 0. This is
        # the on-stage path; it is never allowed to be the reason `make demo`
        # goes red in front of an audience.
        say("")
        say(ink.yellow(redact(f"demo_hero: stopped early — "
                              f"{type(exc).__name__}: {exc}")))
        say(ink.dim("           The archived path is unaffected: "
                    "`make demo -- --replay`, or `make demo-break`."))
        return 0


if __name__ == "__main__":
    sys.exit(main())
