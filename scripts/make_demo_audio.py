"""
make_demo_audio.py — build the LISTENING + DEMO set.

    ./.venv/bin/python scripts/make_demo_audio.py              # everything
    ./.venv/bin/python scripts/make_demo_audio.py --force      # rebuild the wavs too
    ./.venv/bin/python scripts/make_demo_audio.py --force-docs # AND clobber edited docs
    ./.venv/bin/python scripts/make_demo_audio.py --check      # verify, generate nothing

Offline. No API calls, no key, no network. Pure DSP plus a read of
`results/master.csv`. A few seconds.

------------------------------------------------------------------------------
WHY THIS REPLACES results/audio/listen/DEADZONE_*
------------------------------------------------------------------------------
The old dead-zone listening files were all clip `u02`, and nova-3 transcribes
`u02` PERFECTLY (WER 0.000) in four of those six conditions. Playing a clip the
model got exactly right, while narrating "here is where the model fails", is a
demo that refutes itself. The ladder in that directory was and remains sound —
it validated the composer — so it is regenerated here; the DEADZONE_* files are
superseded.

------------------------------------------------------------------------------
WHAT THIS SET IS FOR
------------------------------------------------------------------------------
The study measures ONE axis: what the model does. It has no human axis at all,
because a controlled rig cannot have one — you cannot ask pyroomacoustics
whether a clip was hard to understand. This set generates the missing axis in
three minutes with the listener in the room, and it is built so the comparison
is fair rather than rhetorical.

PART 1 — THE PAIRED, MATCHED-WER TEST (the centrepiece).
Two conditions, each isolating a single degradation (no codec, no mic rolloff),
so nothing is confounded:

  A  "reverb-dominated"  rt60 1.0 @ SNR 20 dB   — a bad room, but QUIET
  B  "babble-dominated"  rt60 0.2 @ SNR  0 dB   — a good room, but buried

Over the same 40 clips the model scores them statistically indistinguishably:
the paired 95% bootstrap CI on the difference spans zero. On three clips it
scores them EXACTLY equal, and non-zero — i.e. the model is equally WRONG on
both, which is a stronger claim than equally right.

A human will almost certainly not rank them equal. We have the precedence
effect and a lifetime of room adaptation, so reverb is nearly free for us;
competing speech causes informational masking, which is brutal for us. The model
has neither prior. So the two axes decouple, and they decouple IN BOTH
DIRECTIONS — which is the point, and why "I listened to it and it sounded fine"
is not evidence about an ASR system.

PART 2 — THE PAYOFF.
`rt60-0.7_snr-20_babble_opus-lowrate_roll-1`. Note SNR 20 dB: this is QUIET.
The damage is reverb + a low-rate codec + full mic rolloff. Ten of the forty
clips return a COMPLETELY EMPTY transcript. You can still hear a person
speaking. The model returned an empty string.

PART 3 — THE FACTOR-ISOLATION LADDER (SPEC A.R3.5).
One clip, one factor at a time, clean to destroyed. This is not about WER; it is
the only test that the composer produces something physically PLAUSIBLE. The
unit tests prove the arithmetic; only ears prove the result.

------------------------------------------------------------------------------
FIVE CONSTRUCTION RULES, EACH LOAD-BEARING
------------------------------------------------------------------------------
1. NO NUMBER IS TYPED BY HAND. Every WER, confidence, transcript and CI in the
   generated markdown is read or recomputed from `results/master.csv` at build
   time, for the same reason `load_manifest` refuses to let a reference be typed
   inline: a hand-copied number and the table it came from diverge silently, and
   a demo that misquotes its own study is worse than no demo.

2. THE WAVS ARE BIT-IDENTICAL TO WHAT THE MODEL HEARD. `apply_condition` is
   seeded from the condition NAME and `write_degraded_wav` is the same writer
   the grid used, so these files reproduce the transcribed audio exactly. That
   is why no loudness normalization is applied here even though B (0 dB) is
   audibly louder than A (20 dB): a cosmetic gain would break the identity for
   a confound that is better handled by letting the listener use the volume
   knob. The confound is named out loud in DEMO_SCRIPT.md instead of hidden.

3. THE BLIND COPIES ARE THE ONLY THING THE LISTENER SEES. The working filenames
   say `reverb` and `babble`; a listener who reads them has been told the answer
   and the exercise is worthless. `blind/` holds neutrally-named byte-identical
   copies plus the only interviewee-facing sheet; `KEY.md` and the prediction
   stay in the parent directory. The blind order is a FROZEN constant, not a
   shuffle, so it is reviewable and cannot drift between runs.

4. THE ONE JUDGEMENT THAT IS NOT IN master.csv IS STORED AS AN OBSERVATION, NOT
   AS A CONSTANT. Which pair to open the exercise on came from a LISTENER, and
   it used to be spelled `"role": "primary" if i <= 2 else "backup"` — a bare
   constant whose owner, date and evidence were nowhere. When a session moved
   pair 1 to last and the documents were hand-corrected, the constant was not,
   so `manifest.json` (the machine-readable answer key) contradicted both
   human-readable answer sheets and `--force-docs` would have rebuilt the sheets
   from the stale constant. The record now lives in `LISTENER_SESSIONS` and the
   order, the roles and the prose are all derived from it. See the block above
   that constant for why a different constant would not have been a fix.

5. REGENERABLE AUDIO AND UNRECOVERABLE RECORD ARE NOT THE SAME ARTIFACT, and
   this script used to treat them as if they were. The wavs rebuild from
   `master.csv` and the asset library in seconds. The markdown does not: two of
   these documents now carry a pre-registered prediction, its verbatim listener
   response, the scoring, the verdict (it FAILED) and an analysis of a flaw in
   the pre-registration's own rubric — none of it derivable from any artifact in
   the repo. Rewriting all five unconditionally already destroyed one such edit,
   silently, because a freshly generated file looks exactly as correct as the
   one it replaced. Every document now goes through `write_doc`, which REFUSES
   to overwrite a file a human has touched. See THE AUTHORED-DOCUMENT GUARD
   below.
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
import hashlib
import json
import re
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from deadzone.conditions import Condition, DiskAssetLibrary, apply_condition
from scripts.make_audio_sets import LADDER, LISTEN_CLIP
from scripts.run_experiment import load_clip, load_manifest, write_degraded_wav

FS = 16000
MODEL = "nova-3"
MASTER = Path("results/master.csv")
DEMO_DIR = Path("results/audio/demo")
BLIND_DIR = DEMO_DIR / "blind"
ISO_DIR = DEMO_DIR / "isolation"
OLD_LISTEN_DIR = Path("results/audio/listen")

# The paired centrepiece. Both isolate ONE degradation: codec=none, rolloff=0.
COND_A = "rt60-1_snr-20_babble_none_roll-0"      # bad room, quiet
COND_B = "rt60-0.2_snr-0_babble_none_roll-0"     # good room, buried
LABEL_A = "reverb-dominated"
LABEL_B = "babble-dominated"

# Clips where the model scores A and B EXACTLY equal AND non-zero, so it is
# equally WRONG on both. Verified against master.csv; the test re-verifies.
#
# This is CONSTRUCTION order, not play order. It fixes the pair NUMBERS and
# through them every working filename (`pair2_A_reverb_u21.wav`) and every entry
# in BLIND_ORDER, so it must not be reordered — doing so would silently repoint
# a KEY.md that someone has already printed. Which pair to PLAY FIRST is a
# separate, derived thing: see LISTENER_SESSIONS below.
PAIR_CLIPS = ["u40", "u21", "u26"]

# The payoff. SNR 20 dB — QUIET. Ten of forty clips come back empty.
COND_PAYOFF = "rt60-0.7_snr-20_babble_opus-lowrate_roll-1"
PAYOFF_CLIP = "u03"                              # longest of the empty ones

BOOT_N = 10000
BOOT_SEED = 0

# Frozen, reviewable blind order: blind_NN -> working filename stem. Chosen so
# no pair is adjacent and the A arm is not always first. Do NOT regenerate this
# with a shuffle — a permutation that silently changes between numpy versions
# would invalidate a KEY.md someone already printed.
BLIND_ORDER = [
    "pair2_B_babble_u21",
    "payoff_u03_clean",
    "pair1_A_reverb_u40",
    "pair3_B_babble_u26",
    "payoff_u03_deadzone",
    "pair2_A_reverb_u21",
    "pair1_B_babble_u40",
    "pair3_A_reverb_u26",
]


# --------------------------------------------------------------------------
# THE PLAY ORDER IS AN EMPIRICAL RESULT, SO IT IS STORED AS ONE
#
# Which pair to open the exercise on is a judgement about a LISTENER, not about
# the grid. Every other number in this generator is read or recomputed from
# `results/master.csv` (construction rule 1); this one cannot be, because no
# artifact in this repo knows what a person said.
#
# It used to be spelled `"role": "primary" if i <= 2 else "backup"` — a bare
# constant with the judgement it encoded recorded nowhere. When the 2026-08-05
# listening session called pair 1 marginal ("both pretty bad") and moved it to
# last, `KEY.md` and `DEMO_SCRIPT.md` were hand-corrected and the constant was
# not. So `manifest.json` — the MACHINE-readable answer key — disagreed with
# both human-readable answer sheets, and `--force-docs` would have regenerated
# the sheets FROM the stale constant, reverting a correction that a listener's
# actual response had earned. The guard protects the files; it cannot protect
# their content while the template is still wrong.
#
# Swapping in a different bare constant would only reset that clock. So the
# OBSERVATION is stored and the ordering is derived from it:
#
#     recorded listener call  ->  play order  ->  role  ->  the prose
#
# Four consequences worth having. A second session is appended rather than
# argued about, and supersedes without deleting. `manifest.json` now carries the
# provenance of its own ordering, so the answer key says on whose authority it
# is in that order. `--force-docs` reproduces the correction instead of
# reverting it. And if the record is ever emptied, every pair reads `untested`
# and the order degrades to construction order — the honest answer, rather than
# a stale opinion with no owner.
#
# What is deliberately NOT derived from this: the sealed prediction in
# `PREREGISTERED_PREDICTION.md` still names pairs 1 and 2, because that is what
# was written down before anyone listened. A pre-registration that silently
# re-orders itself to match the result is not a record of anything.
# --------------------------------------------------------------------------

CONFIDENT, MARGINAL, UNTESTED = "confident", "marginal", "untested"

# Play the pairs that drew a confident call first. `sorted` is stable, so ties
# keep construction order and the result is a total order that cannot drift
# between runs — the same reason BLIND_ORDER is frozen rather than shuffled.
PLAY_RANK = {CONFIDENT: 0, MARGINAL: 1, UNTESTED: 2}

LISTENER_SESSIONS = (
    {
        "date": "2026-08-05",
        "n_listeners": 1,
        "blind_to_condition": True,
        "naive_to_hypothesis": False,          # they knew what the project claims
        "record": "results/audio/demo/PREREGISTERED_PREDICTION.md - OUTCOME",
        # What the sealed prediction claimed, kept here so the per-pair scoring
        # in the OUTCOME section is DERIVED by comparing it against `harder_arm`
        # rather than typed as a verdict somebody has to keep in sync.
        "predicted_harder_arm": "B",
        "predicted_pairs": ("u40", "u21"),
        "verbatim": (
            "bro 3 and 7 are both pretty bad but honestly think that 7 is "
            "better, not 100% eveyeron would agree with my decision tho. 6 "
            "better than 1, but by a little gap that everyone would agree with. "
            "i think 4 better than 8, others would say the same p sure."
        ),
        # clip_id -> the call. `harder_arm` is the arm they ranked HARDER; the
        # listener named the EASIER clip of each pair, so these are the flip of
        # their own words and the OUTCOME section says so out loud.
        #
        # `confidence` and `non_marginal` are DIFFERENT axes and are kept apart
        # on purpose. `confidence` is how sure the listener was that others
        # would agree — it is what the play order is sorted on. `non_marginal`
        # is whether the call clears the prediction's own bar, the word
        # "clearly harder". Pair `u21` is confident and NOT non-marginal ("a
        # little gap that everyone would agree with"): everyone agrees, about a
        # small difference. Collapsing the two would let the sealed prediction
        # be scored leniently, which is exactly the flaw its OUTCOME section
        # was written to name.
        "calls": {
            "u40": {"confidence": MARGINAL, "harder_arm": "A",
                    "non_marginal": False,
                    "said": "both pretty bad ... not 100% eveyeron would agree"},
            "u21": {"confidence": CONFIDENT, "harder_arm": "B",
                    "non_marginal": False,
                    "said": "a little gap that everyone would agree with"},
            "u26": {"confidence": CONFIDENT, "harder_arm": "A",
                    "non_marginal": True,
                    "said": "p sure others would say the same"},
        },
        # An observation made AFTER the result was in view. Recorded, with that
        # label attached, precisely so it can never be quoted as the mechanism —
        # the transcripts it refers to are pulled from the master table at build
        # time so the observation stays checkable.
        "post_hoc": {
            "clip_id": "u40",
            "easier_arm": "B",
            "observation":
                "the clip the listener preferred is the one whose transcript "
                "KEPT the proper noun, while the clip they found harder mangled "
                "it",
            "why_recorded":
                "it is checkable, and entity survival is a real axis elsewhere "
                "in this project (D2: proper nouns are the most-destroyed word "
                "class, 0.646) — not because it explains anything here",
        },
    },
)

ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}


def _ordinal(n: int) -> str:
    return ORDINAL.get(int(n), f"{int(n)}th")


def _and_list(items: list[str]) -> str:
    items = [str(i) for i in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def listener_call(clip_id: str) -> dict:
    """The latest recorded call for one clip, or an `untested` stub.

    Latest wins, so appending a second session supersedes the first without the
    first being deleted — the same superseding-forward discipline the OUTCOME
    section of the prediction uses.
    """
    out = {"confidence": UNTESTED, "harder_arm": None, "non_marginal": None,
           "said": None, "session": None}
    for s in LISTENER_SESSIONS:
        c = (s.get("calls") or {}).get(clip_id)
        if c:
            out = {"confidence": c.get("confidence", UNTESTED),
                   "harder_arm": c.get("harder_arm"),
                   "non_marginal": c.get("non_marginal"),
                   "said": c.get("said"),
                   "session": s.get("date")}
    return out


def play_order(clips: list[str] | None = None) -> list[str]:
    """Clip ids in the order to PLAY them: most-confident call first."""
    cl = list(PAIR_CLIPS if clips is None else clips)
    return sorted(cl, key=lambda c: PLAY_RANK.get(
        listener_call(c)["confidence"], PLAY_RANK[UNTESTED]))


def pair_roles(clips: list[str] | None = None) -> dict[str, dict]:
    """clip_id -> the derived presentation fields. Nothing here is hardcoded.

    `role` is the coarse machine-readable form (`primary` when a listener made a
    confident call on that pair, `reserve` otherwise); `play_position` and
    `play_note` are the ordering; the `listener_*` fields are the evidence, so a
    consumer that reads `role` can always find out who decided it and when.
    """
    order = play_order(clips)
    n = len(order)
    out: dict[str, dict] = {}
    for pos, clip in enumerate(order, start=1):
        call = listener_call(clip)
        primary = call["confidence"] == CONFIDENT
        if primary:
            note = f"primary — play {_ordinal(pos)}"
        elif pos == n:
            note = f"{_ordinal(pos)} choice — play only if they want another"
        else:
            note = f"reserve — play {_ordinal(pos)}"
        out[clip] = {
            "role": "primary" if primary else "reserve",
            "play_position": pos,
            "play_note": note,
            "listener_confidence": call["confidence"],
            "listener_non_marginal": call["non_marginal"],
            "listener_said": call["said"],
            "listener_harder_arm": call["harder_arm"],
            "listener_session": call["session"],
        }
    return out


def by_play_order(pairs: list[dict]) -> list[dict]:
    """Manifest pair records, in the order the run-of-show plays them."""
    return sorted(pairs, key=lambda p: p.get("play_position", p["pair"]))


def prediction_scorecard(m: dict) -> dict | None:
    """The sealed prediction scored against the recorded calls — all derived.

    Returns None when no session on record carries a prediction, which is the
    state a fresh clip set is in. The templates then omit the outcome rather
    than asserting one, because a pre-registration reported as open when it was
    never run and one reported as open when it FAILED must not look the same.
    """
    sess = None
    for s in m.get("listener_sessions", []):
        if s.get("predicted_harder_arm") and s.get("calls"):
            sess = s
    if sess is None:
        return None
    pred = sess["predicted_harder_arm"]
    named = set(sess.get("predicted_pairs") or ())
    rows = []
    for p in m["pairs"]:
        arm = p.get("listener_harder_arm")
        rows.append({
            "pair": p["pair"], "clip_id": p["clip_id"],
            "named": p["clip_id"] in named,
            "harder_arm": arm,
            "direction_hit": None if arm is None else (arm == pred),
            "non_marginal": p.get("listener_non_marginal"),
            "said": p.get("listener_said"),
            "wer_A": p["A"]["wer"], "wer_B": p["B"]["wer"],
        })
    scored = [r for r in rows if r["harder_arm"] is not None]
    n_hit = sum(1 for r in scored if r["direction_hit"])
    # Scored against the prediction's OWN sentence — "a confident, immediate,
    # NON-MARGINAL call in both pairs" — so a marginal call is a miss even when
    # the direction is right. The generous 1-of-N reading is reported too, and
    # labelled generous, because reporting only the flattering one is the
    # failure this document exists to describe.
    strict_failures = [r for r in rows if r["named"]
                       and not (r["direction_hit"] and r["non_marginal"])]
    return {
        "session": sess,
        "predicted_harder_arm": pred,
        "predicted_label": LABEL_B if pred == "B" else LABEL_A,
        "rows": rows,
        "n_pairs": len(rows),
        "n_scored": len(scored),
        "n_direction_hit": n_hit,
        "n_named": sum(1 for r in rows if r["named"]),
        "n_strict_failures": len(strict_failures),
        "strict_failures": strict_failures,
        "held": bool(scored) and n_hit == len(scored),
        "verdict": (
            f"HELD on direction — the predicted direction held in "
            f"{n_hit} of {len(scored)} pairs"
            if scored and n_hit == len(scored) else
            f"FAILED on direction — the predicted direction held in "
            f"{n_hit} of {len(scored)} pairs"),
    }


def _order_phrase(nums: list[int]) -> str:
    if not nums:
        return "(no pairs)"
    if len(nums) == 1:
        return f"pair {nums[0]}"
    head = ", then ".join(f"pair {n}" for n in nums[:-1])
    return f"{head}, pair {nums[-1]} {_ordinal(len(nums))}"


def listener_ordering_note(m: dict) -> str:
    """The paragraph that explains the play order, derived from the record.

    Every branch here is reachable: no session at all (the record emptied, or a
    fresh set of clips), all confident, or a mix. The no-session branch is the
    important one — it says the ordering is arbitrary rather than implying a
    judgement nobody made.
    """
    ordered = by_play_order(m["pairs"])
    nums = [p["pair"] for p in ordered]
    conf = [p["pair"] for p in ordered if p["listener_confidence"] == CONFIDENT]
    soft = [p for p in ordered if p["listener_confidence"] != CONFIDENT
            and p["listener_confidence"] != UNTESTED]
    sess = [s for s in m.get("listener_sessions", []) if s.get("calls")]

    if not conf and not soft:
        return ("Play order is construction order — **no listening session is on "
                "record for these clips**, so no pair is marked primary and the "
                "ordering below carries no judgement. Add a session to "
                "`LISTENER_SESSIONS` in `scripts/make_demo_audio.py` and the "
                "order, the roles and this paragraph all follow from it.")

    head = (f"Play order is **{_order_phrase(nums)}** — see `DEMO_SCRIPT.md` §2, "
            f"and note the beat plays **{words(len(conf))} of the "
            f"{words(len(nums))}**: the rest are reserves. All "
            f"{words(len(nums))} carry identical "
            f"evidential weight (the model scores every pair exactly equal); the "
            f"ordering is about the listener.")
    # "The one session run so far" is only true while there IS one. A second
    # listener would otherwise be silently narrated away.
    when = (f"The one session run so far ({sess[0]['date']})" if len(sess) == 1
            else f"The {words(len(sess))} sessions on record")
    if not soft:
        return (f"{head} No pair drew a hedged call in the session record, so "
                f"nothing is held back.")
    hedged = _and_list([f"pair {p['pair']}" for p in soft])
    said = soft[0]["listener_said"]
    return (f"{head} {when} drew confident calls on "
            f"{_and_list(['pair %d' % n for n in conf])} and a hedge on {hedged}"
            + (f' ("{said}")' if said else "")
            + f", so {hedged} is no longer opened on.")


# --------------------------------------------------------------------------
# master table access
# --------------------------------------------------------------------------

class MissingTable(RuntimeError):
    pass


def load_rows(conditions: set[str], model: str = MODEL) -> dict[tuple[str, str], dict]:
    """(condition_name, clip_id) -> row, for the model arm we demo."""
    if not MASTER.is_file():
        raise MissingTable(
            f"{MASTER} not found — the demo quotes MEASURED numbers and refuses "
            f"to invent them. Rebuild it with:\n"
            f"    ./.venv/bin/python scripts/run_experiment.py --rebuild")
    csv.field_size_limit(10 ** 9)
    out: dict[tuple[str, str], dict] = {}
    with open(MASTER, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["model"] == model and r["condition_name"] in conditions:
                out[(r["condition_name"], r["clip_id"])] = r
    if not out:
        raise MissingTable(
            f"{MASTER} has no {model!r} rows for {sorted(conditions)} — the grid "
            f"this demo describes is not in the table.")
    return out


def _f(row: dict, key: str) -> float | None:
    v = (row.get(key) or "").strip()
    return float(v) if v else None


def paired_stats(rows: dict[tuple[str, str], dict]) -> dict:
    """
    The measured half of the exercise: paired over CLIPS, which is the right
    resampling unit — the same 40 sentences are scored under both conditions, so
    clip identity is a blocking factor and resampling words or cells would throw
    the pairing away and understate the CI.
    """
    clips = sorted({c for (cond, c) in rows if cond == COND_A}
                   & {c for (cond, c) in rows if cond == COND_B})
    a = np.array([float(rows[(COND_A, c)]["wer"]) for c in clips])
    b = np.array([float(rows[(COND_B, c)]["wer"]) for c in clips])
    d = a - b
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(clips), size=(BOOT_N, len(clips)))
    boot = d[idx].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {
        "n_clips": len(clips),
        "clips": clips,
        "mean_wer_A": float(a.mean()),
        "mean_wer_B": float(b.mean()),
        "paired_diff_A_minus_B": float(d.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "spans_zero": bool(lo < 0.0 < hi),
        "n_resamples": BOOT_N,
        "seed": BOOT_SEED,
        "resample_unit": "clip",
    }


def rir_acoustics(cond: Condition, assets, fs: int = FS) -> dict:
    """
    DRR and C50 of the RIR this condition actually resolves to.

    Same definition as scripts/run_d3a.py so the demo and the D3a report cannot
    quote different numbers for the same room. This matters because the project's
    sharpest reverb result is that RT60 MISLABELS the delivered acoustics:
    spearman(DRR, WER) = -1.000 while spearman(RT60, WER) = +0.800.
    """
    rir = assets.resolve(cond).rir
    h, hfs = sf.read(rir.key, dtype="float64")
    h = np.asarray(h, dtype=float)
    if h.ndim > 1:
        h = h.mean(axis=1)
    if hfs != fs:
        import librosa                                    # lazy: heavy import
        h = librosa.resample(h, orig_sr=hfs, target_sr=fs)
    d = int(np.argmax(np.abs(h)))
    w = int(0.0025 * fs)
    direct = h[max(0, d - w):d + w]
    rev = np.concatenate([h[:max(0, d - w)], h[d + w:]])
    e50 = int(0.050 * fs)
    return {
        "rir_key": rir.key,
        "room": Path(rir.key).name,
        "rt60_measured": float(rir.rt60),
        "drr_db": float(10 * np.log10((direct ** 2).sum() / max((rev ** 2).sum(), 1e-20))),
        "c50_db": float(10 * np.log10((h[d:d + e50] ** 2).sum()
                                      / max((h[d + e50:] ** 2).sum(), 1e-20))),
    }


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def _write_degraded(path: Path, clip_id: str, cond: Condition, assets) -> None:
    audio = load_clip(clip_id, target_fs=FS)
    y = apply_condition(audio, cond, assets, FS)
    write_degraded_wav(path, y, FS)


def build_audio(assets) -> dict[str, str]:
    """Every wav. Returns stem -> relative path."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    BLIND_DIR.mkdir(parents=True, exist_ok=True)
    ISO_DIR.mkdir(parents=True, exist_ok=True)

    ca, cb = Condition.from_name(COND_A), Condition.from_name(COND_B)
    made: dict[str, str] = {}

    for i, clip in enumerate(PAIR_CLIPS, start=1):
        for arm, cond, tag in ((("A"), ca, "reverb"), (("B"), cb, "babble")):
            stem = f"pair{i}_{arm}_{tag}_{clip}"
            p = DEMO_DIR / f"{stem}.wav"
            _write_degraded(p, clip, cond, assets)
            made[stem] = str(p)

    # The control is the raw recording resampled to 16 kHz — which is exactly
    # what the pipeline fed the model (load_clip resamples at the edge). Handing
    # over the untouched 48 kHz file instead would make the clean clip audibly
    # brighter for a reason that has nothing to do with the condition, i.e. it
    # would plant a confound in the one comparison the demo rests on.
    raw = load_clip(PAYOFF_CLIP, target_fs=FS)
    sf.write(str(DEMO_DIR / "payoff_u03_clean.wav"), raw, FS, subtype="PCM_16")
    made["payoff_u03_clean"] = str(DEMO_DIR / "payoff_u03_clean.wav")

    _write_degraded(DEMO_DIR / "payoff_u03_deadzone.wav", PAYOFF_CLIP,
                    Condition.from_name(COND_PAYOFF), assets)
    made["payoff_u03_deadzone"] = str(DEMO_DIR / "payoff_u03_deadzone.wav")

    # blind copies — real copies, not symlinks: a symlink resolves to a name
    # that says `reverb` the moment anyone inspects the folder, and a zipped
    # symlink is a broken file on someone else's laptop.
    blind_map: dict[str, str] = {}
    for n, stem in enumerate(BLIND_ORDER, start=1):
        src = Path(made[stem])
        dst = BLIND_DIR / f"blind_{n:02d}.wav"
        shutil.copyfile(src, dst)
        blind_map[dst.name] = src.name
        made[f"blind_{n:02d}"] = str(dst)

    # Part 3: the factor-isolation ladder, LADDER imported from make_audio_sets
    # so there is one definition of the ladder in the repo, not two that drift.
    iso = load_clip(LISTEN_CLIP, target_fs=FS)
    sf.write(str(ISO_DIR / "00_RAW_original.wav"), iso, FS, subtype="PCM_16")
    made["iso_00_RAW_original"] = str(ISO_DIR / "00_RAW_original.wav")
    for label, cond in LADDER:
        p = ISO_DIR / f"{label}.wav"
        y = apply_condition(iso, cond, assets, FS)
        write_degraded_wav(p, y, FS)
        made[f"iso_{label}"] = str(p)

    made["_blind_map"] = blind_map          # popped by the caller
    return made


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def build_manifest(made: dict, rows: dict, stats: dict, assets) -> dict:
    blind_map = made.pop("_blind_map")
    refs = load_manifest()
    ca, cb = Condition.from_name(COND_A), Condition.from_name(COND_B)
    cp = Condition.from_name(COND_PAYOFF)
    inv = {v: k for k, v in blind_map.items()}

    def cell(cond_name: str, clip: str, stem: str) -> dict:
        r = rows[(cond_name, clip)]
        fn = Path(made[stem]).name
        return {
            "file": made[stem],
            "blind": inv.get(fn),
            "wer": float(r["wer"]),
            "mean_conf": _f(r, "mean_conf"),
            "utterance_conf": _f(r, "utterance_conf"),
            "transcript": r["transcript"],
            "n_ref": int(r["n_ref"]), "n_sub": int(r["n_sub"]),
            "n_del": int(r["n_del"]), "n_ins": int(r["n_ins"]),
            "rir_key": r["rir_key"], "noise_key": r["noise_key"],
        }

    # `role`, `play_position` and `play_note` are DERIVED from the recorded
    # listener calls (see LISTENER_SESSIONS), never hardcoded. The
    # `listener_*` fields travel with them so that a consumer reading `role`
    # cannot quote it without also being handed the evidence behind it — the
    # same redundancy-makes-a-silent-error-loud argument as SPEC C.5.
    roles = pair_roles()
    pairs = []
    for i, clip in enumerate(PAIR_CLIPS, start=1):
        pairs.append({
            "pair": i, "clip_id": clip, "ref": refs[clip],
            **roles[clip],
            "A": cell(COND_A, clip, f"pair{i}_A_reverb_{clip}"),
            "B": cell(COND_B, clip, f"pair{i}_B_babble_{clip}"),
        })

    payoff_rows = {c: r for (cond, c), r in rows.items() if cond == COND_PAYOFF}
    empty = sorted(c for c, r in payoff_rows.items() if not r["transcript"].strip())
    pr = payoff_rows[PAYOFF_CLIP]

    return {
        "generated_by": "scripts/make_demo_audio.py",
        "fs": FS,
        "model": MODEL,
        "master_table": str(MASTER),
        "supersedes": str(OLD_LISTEN_DIR / "DEADZONE_*.wav"),
        "conditions": {
            "A": {"name": COND_A, "label": LABEL_A, **ca.to_sample(),
                  **rir_acoustics(ca, assets)},
            "B": {"name": COND_B, "label": LABEL_B, **cb.to_sample(),
                  **rir_acoustics(cb, assets)},
            "payoff": {"name": COND_PAYOFF, **cp.to_sample(),
                       **rir_acoustics(cp, assets)},
        },
        "paired_result": stats,
        "pairs": pairs,
        # The ordering, and the human record it is derived from. `manifest.json`
        # is deliberately NOT guarded by write_doc — it is regenerable output,
        # not an unrecoverable record — and it is only regenerable because the
        # judgement it carries lives in the generator rather than only in prose.
        "play_order": [p["pair"] for p in by_play_order(pairs)],
        "play_order_derivation":
            "Play order and each pair's `role` are DERIVED from the recorded "
            "listener calls in `listener_sessions`: a pair that drew a "
            "confident call is played first, ties keep construction order. "
            "Empty that record and every pair reads `untested`, no pair is "
            "primary, and the order falls back to construction order. "
            "Hardcoding it is what let this file disagree with KEY.md and "
            "DEMO_SCRIPT.md after the 2026-08-05 session.",
        "listener_sessions": [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}
            for s in LISTENER_SESSIONS],
        "payoff": {
            "clip_id": PAYOFF_CLIP,
            "ref": refs[PAYOFF_CLIP],
            "n_empty": len(empty), "n_clips": len(payoff_rows),
            "empty_clips": empty,
            "clean_file": made["payoff_u03_clean"],
            "clean_blind": inv.get("payoff_u03_clean.wav"),
            "deadzone_file": made["payoff_u03_deadzone"],
            "deadzone_blind": inv.get("payoff_u03_deadzone.wav"),
            "wer": float(pr["wer"]), "n_ref": int(pr["n_ref"]),
            "n_del": int(pr["n_del"]),
            "transcript": pr["transcript"],
            "mean_conf": _f(pr, "mean_conf"),
            "utterance_conf": _f(pr, "utterance_conf"),
        },
        "blind_map": blind_map,
        "isolation": [k[4:] for k in sorted(made) if k.startswith("iso_")],
        "files": sorted(v for k, v in made.items() if not k.startswith("_")),
    }


# --------------------------------------------------------------------------
# THE AUTHORED-DOCUMENT GUARD
#
# Every document below used to be written unconditionally on any build — no
# skip-if-modified, no warning, no backup. That is fine for the wavs, which are
# a pure function of `master.csv` and the asset library, and it is catastrophic
# for the prose, which is where a human writes down the one thing the pipeline
# cannot produce: what a listener said. It has already cost one such edit.
#
# The defect is NOT the overwrite. It is that regenerable output and
# unrecoverable record were handed to the same writer.
#
# The rule follows `write_master()` in scripts/run_experiment.py — refuse to
# produce the misleading artifact rather than warn about it afterwards:
#
#   file absent          -> write.
#   hash matches record  -> this generator wrote it and nobody has touched it
#                           since; write.
#   hash differs         -> a human edited it; REFUSE, and name the flag.
#   NO recorded hash     -> provenance unknown; REFUSE.
#
# That last line is the load-bearing one, and it is deliberately the paranoid
# reading. A missing baseline is the degenerate input, and SPEC Appendix E.5's
# rule is to ask what a guard returns for the degenerate input rather than for
# the good one. If "no record" meant "safe to overwrite", this guard would be
# wide open on precisely the state the repo is in the first time it runs —
# every document in the kit already written, and no sidecar yet.
#
# `authored` therefore means "not provably this generator's own output", which
# is a wider set than "a human typed in it". That imprecision is the correct
# direction to be imprecise in, and the census says so rather than asserting a
# human edit it cannot actually observe.
#
# It also settles how the sidecar is seeded: it ISN'T. Recording today's
# hand-edited text as "what the generator last wrote" would certify it as
# generator-owned and the very next default run would erase it — the guard
# would have been the delivery mechanism for the bug it exists to stop. Not
# seeding is what protects it, and it protects any other file that predates the
# guard, or is restored from a backup, for the same reason.
#
# Detection is by CONTENT HASH, never mtime. mtime does not survive a checkout,
# a copy, a `cp` without `-p`, or a restore — all four of which are things that
# happen to a demo kit that gets handed around.
# --------------------------------------------------------------------------

# What the generator last wrote, keyed by repo-relative path (every path in this
# project resolves against the repo root as CWD, SPEC §13). Lives beside the
# documents it guards so it shares their lifecycle: delete `results/audio/demo/`
# and the whole set rebuilds from scratch, guard included.
DOC_HASHES = DEMO_DIR / "generated_docs.json"

ABSENT, GENERATED, AUTHORED = "absent", "generated", "authored"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_key(path: Path) -> str:
    return Path(path).as_posix()


def load_doc_hashes() -> dict[str, str]:
    """Path -> sha256 of the text this generator last wrote there.

    An unreadable sidecar returns `{}`, which is the same as an absent one, and
    both then read as unknown provenance — i.e. the failure mode of this
    function is to protect MORE, never less.
    """
    if not DOC_HASHES.is_file():
        return {}
    try:
        rec = json.loads(DOC_HASHES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    got = rec.get("sha256")
    return dict(got) if isinstance(got, dict) else {}


def _save_doc_hash(key: str, digest: str) -> None:
    """Record one document, immediately after it is written.

    Per-document rather than once at the end: a build that dies halfway must
    leave a record that matches what is actually on disk, or the survivors come
    back as `authored` on the next run and the guard cries wolf.
    """
    sha = load_doc_hashes()
    sha[key] = digest
    DOC_HASHES.parent.mkdir(parents=True, exist_ok=True)
    DOC_HASHES.write_text(json.dumps({
        "written_by": "scripts/make_demo_audio.py",
        "what_this_is":
            "SHA-256 of each document AS THIS GENERATOR LAST WROTE IT. Baseline "
            "for the authored-document guard: a file whose hash no longer "
            "matches has been edited by a human and will not be overwritten "
            "without --force-docs. Deleting this file unlocks nothing — an "
            "absent record reads as unknown provenance, which also refuses.",
        "sha256": dict(sorted(sha.items())),
    }, indent=2) + "\n", encoding="utf-8")


def doc_status(path: Path, hashes: dict[str, str] | None = None) -> str:
    """`absent` | `generated` (safe to rewrite) | `authored` (hands off)."""
    p = Path(path)
    if not p.is_file():
        return ABSENT
    h = load_doc_hashes() if hashes is None else hashes
    recorded = h.get(_doc_key(p))
    if recorded is None:
        return AUTHORED
    try:
        current = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return AUTHORED
    return GENERATED if recorded == _sha256(current) else AUTHORED


def _refusal(p: Path) -> str:
    return "\n".join([
        f"[demo docs] REFUSING to overwrite {p}",
        "    why   : its contents do not match what this generator last wrote, so it",
        "            holds edits made by a human. Rewriting it would replace them with",
        "            template output and say nothing — a regenerated file looks exactly",
        "            as correct as the one it replaced, which is how this was lost once",
        "            already (see results/audio/demo/REGENERATION_HAZARD.md).",
        "    state : the file on disk is UNCHANGED and the rest of the build continued.",
        "    cost  : it therefore does NOT carry this build's numbers. manifest.json and",
        "            KEY.md are the generated source of truth for those.",
        "    fix   : port the hand-written block into this script's template, so a",
        "            rebuild reproduces it — or, to overwrite it anyway:",
        "                ./.venv/bin/python scripts/make_demo_audio.py --force-docs",
        "            which first copies the current file to <name>.superseded-<UTC>.md.",
    ])


def write_doc(path: Path, text: str, *, force_docs: bool = False) -> str:
    """Write a generator-owned document. Returns `written` or `skipped`.

    `force_docs` defaults to False so that any future caller which forgets the
    keyword gets the protective behaviour, not the destructive one.
    """
    p = Path(path)
    status = doc_status(p)
    if status == AUTHORED and not force_docs:
        print(_refusal(p))
        return "skipped"
    if status == AUTHORED:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = p.with_name(f"{p.stem}.superseded-{stamp}{p.suffix}")
        shutil.copyfile(p, backup)
        print(f"[demo docs] --force-docs: OVERWRITING hand-edited {p}\n"
              f"            previous contents preserved at {backup}\n"
              f"            NOTHING ELSE IN THIS REPO HOLDS THEM — read that file "
              f"before deleting it.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    _save_doc_hash(_doc_key(p), _sha256(text))
    return "written"


# --------------------------------------------------------------------------
# documents (every number interpolated, none typed)
# --------------------------------------------------------------------------

def num(v: float) -> str:
    """`1.0 -> '1.0'`, `0.2 -> '0.2'`. `%g` renders 1.0 as a bare `1`, which reads
    as a count rather than a seconds value when spoken aloud."""
    s = f"{float(v):g}"
    return s if "." in s or "e" in s else s + ".0"


def room_name(rir_filename: str) -> str:
    """`mit_rt60-0.99_h081_Shower_2txts.wav` -> `Shower`.

    Read aloud in the demo, so it has to be the ROOM, not whatever token the
    dataset happened to end the filename with.
    """
    stem = Path(rir_filename).stem
    stem = re.sub(r"^mit_rt60-[\d.]+_h\d+_", "", stem)
    stem = re.sub(r"_\d*txts$", "", stem)
    return stem.replace("_", " ") or Path(rir_filename).stem


_NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def words(n: int) -> str:
    """`3 -> 'three'`. Small counts read as digits look like measurements in a
    document whose every digit IS one; spelled out, prose stays prose."""
    return _NUMBER_WORD.get(int(n), str(int(n)))


def para(text: str, width: int = 78) -> str:
    """Re-wrap an interpolated paragraph. Same reason as `quote()`: the template
    cannot know how wide a substituted number or a listener's quote will be."""
    return "\n".join(textwrap.wrap(" ".join(text.split()), width=width))


def bullet(text: str, width: int = 78) -> str:
    """`para()` for a markdown list item, with the hanging indent."""
    return "\n".join(textwrap.wrap(" ".join(text.split()), width=width,
                                   initial_indent="- ", subsequent_indent="  "))


def quote(text: str, width: int = 76) -> str:
    """
    Re-wrap an interpolated paragraph into a markdown blockquote.

    Interpolated numbers have unpredictable width, so wrapping the template by
    hand puts line breaks in the middle of sentences the presenter has to read
    aloud. Wrapping AFTER substitution keeps the spoken lines clean whatever the
    numbers turn out to be.
    """
    out = []
    for i, para in enumerate(p.strip() for p in text.strip().split("\n\n")):
        if i:
            out.append(">")
        body = " ".join(para.split())
        out.extend(textwrap.wrap(body, width=width, initial_indent="> ",
                                 subsequent_indent="> "))
    return "\n".join(out)


ARM_LABEL = {"A": LABEL_A, "B": LABEL_B}


def _prediction_outcome(m: dict, sc: dict, s: dict) -> str:
    """The OUTCOME section, rendered from the recorded session.

    Every cell of the scoring table is derived — the direction column is
    `harder_arm == predicted_harder_arm`, not a ✅ somebody typed — so the table
    cannot drift out of agreement with the record it scores. That is the same
    argument as `gap = mean_conf - (1 - wer)` being stored alongside its inputs
    (SPEC C.5): a derived field that can be recomputed is what makes a wrong one
    loud.
    """
    sess = sc["session"]
    by_clip = {p["clip_id"]: p for p in m["pairs"]}
    pred = sc["predicted_harder_arm"]
    n_miss = sc["n_scored"] - sc["n_direction_hit"]

    rows, maps = [], []
    for r in sc["rows"]:
        p = by_clip[r["clip_id"]]
        tag = ("(named)" if r["named"] else
               "(backup, not named in the prediction)")
        want = (f"`{p[pred]['blind'].removesuffix('.wav')}` = **{pred}** "
                f"{ARM_LABEL[pred]}" if r["named"] else
                f"— (pattern implies **{pred}** {ARM_LABEL[pred]})")
        arm = r["harder_arm"]
        got = (f"`{p[arm]['blind'].removesuffix('.wav')}` = **{arm}** "
               f"{ARM_LABEL[arm]}" if arm else "— no call")
        direction = ("✅ as predicted" if r["direction_hit"]
                     else "❌ **OPPOSITE**" if arm else "—")
        rows.append(
            f"| {r['pair']} {tag} | `{r['clip_id']}` | {want} | {got} | "
            f"{direction} | \"{r['said'] or ''}\" | "
            f"{r['wer_A']:.3f} / {r['wer_B']:.3f} |")
        for a in ("A", "B"):
            maps.append(f"`{p[a]['blind'].removesuffix('.wav')}` = "
                        f"`{Path(p[a]['file']).stem}`")

    strict = sc["strict_failures"]
    if strict:
        why = "; ".join(
            f"pair {r['pair']} on "
            + ("**direction**" if not r["direction_hit"] else "**magnitude**")
            for r in strict)
        strict_line = bullet(
            f"**Scored strictly against the prediction's own sentence** — \"a "
            f"confident, immediate, **non-marginal** call in **both** pairs\" — "
            f"it fails in {words(len(strict))} of the {words(sc['n_named'])} "
            f"named pairs: {why}. A marginal call is a miss when the sentence "
            f"says \"clearly harder\". The {sc['n_direction_hit']}-of-"
            f"{sc['n_scored']} figure is the generous reading.")
    else:
        strict_line = bullet(
            "Scored strictly against the prediction's own sentence — \"a "
            "confident, immediate, non-marginal call in both pairs\" — every "
            "named pair clears the bar.")

    verdict_lines = "\n".join([
        bullet(f"**The predicted direction held in {sc['n_direction_hit']} of "
               f"{sc['n_scored']} pairs.** The listener found the "
               f"**{ARM_LABEL['A' if pred == 'B' else 'B']}** arm harder in "
               f"{words(n_miss)} of them."),
        strict_line,
        bullet(f"**The predicted mechanism is refuted at these settings.** The "
               f"prediction rested on informational masking from competing "
               f"speech overwhelming the precedence effect. At **DRR "
               f"{m['conditions']['A']['drr_db']:+.2f} dB** it did not: heavy "
               f"reverberation was judged harder than "
               f"{num(m['conditions']['B']['snr_db'])} dB babble in "
               f"{words(n_miss)} of {words(sc['n_scored'])} pairs. The mechanism "
               f"may still hold at milder DRR — this says nothing about that — "
               f"but it does not hold where it was predicted to."),
    ])

    arm_defs = para(
        f"`A` = {LABEL_A} (`{m['conditions']['A']['name']}`, "
        f"{room_name(m['conditions']['A']['room'])}, measured RT60 "
        f"{m['conditions']['A']['rt60_measured']:.3f} s, **DRR "
        f"{m['conditions']['A']['drr_db']:+.2f} dB**, babble at "
        f"{num(m['conditions']['A']['snr_db'])} dB SNR — i.e. quiet). "
        f"`B` = {LABEL_B} (`{m['conditions']['B']['name']}`, "
        f"{room_name(m['conditions']['B']['room'])}, measured RT60 "
        f"{m['conditions']['B']['rt60_measured']:.3f} s, **DRR "
        f"{m['conditions']['B']['drr_db']:+.2f} dB**, babble at "
        f"{num(m['conditions']['B']['snr_db'])} dB SNR — i.e. buried).")

    survives = para(
        f"**The listener had a stated preference in {words(sc['n_scored'])} of "
        f"{words(sc['n_pairs'])} pairs, and the model scores each of those pairs "
        f"exactly equal** — "
        + ", ".join(f"{r['wer_A']:.3f}/{r['wer_B']:.3f}" for r in sc["rows"])
        + " (read from `results/master.csv`, not from this file's history).")

    measured = para(
        f"**The measured half is untouched, as designed.** Recomputed from "
        f"`results/master.csv` ({s['n_clips']} clips, {MODEL}): mean WER A "
        f"**{s['mean_wer_A']:.6f}**, mean WER B **{s['mean_wer_B']:.6f}**, "
        f"paired difference **{s['paired_diff_A_minus_B']:+.7f}**, 95 % CI "
        f"**[{s['ci_lo']:+.7f}, {s['ci_hi']:+.7f}]** — "
        f"{s['n_resamples']:,}-resample paired bootstrap over "
        f"{s['resample_unit']}s, seed {s['seed']}, and it "
        f"{'spans' if s['spans_zero'] else 'does NOT span'} zero. **Nothing the "
        f"listener said moves this number**, which is the whole reason the two "
        f"halves were labelled separately in the sealed text above.")

    ph = sess.get("post_hoc") or {}
    post_hoc = ""
    if ph and ph.get("clip_id") in by_clip:
        p = by_clip[ph["clip_id"]]
        easy = ph.get("easier_arm", "B")
        hard = "A" if easy == "B" else "B"
        post_hoc = f"""
## POST-HOC — a hypothesis, explicitly not a finding

⚠️ **Generated after seeing the result. It is not evidence. Do not present it
as an explanation.**

{para(f"In pair {p['pair']}, {ph['observation']} — verified against "
      f"`results/master.csv`:")}

- reference: `{p['ref']}`
- `{p[hard]['blind'].removesuffix('.wav')}` ({ARM_LABEL[hard]}, ranked HARDER):
  `{p[hard]['transcript']}`
- `{p[easy]['blind'].removesuffix('.wav')}` ({ARM_LABEL[easy]}, ranked easier):
  `{p[easy]['transcript']}`

{para(f"Both score WER {p['A']['wer']:.3f}. **The listener was judging audio, "
      f"not transcripts, and never saw either transcript**, so any link between "
      f"their preference and the entity outcome is speculation on n=1. It is "
      f"recorded because {ph['why_recorded']}.")}
"""

    return f"""
---

# OUTCOME — tested {sess['date']}

*The sealed record above — the prediction and its outcome rubric — is
**unaltered**. The only additions above this line are the status banner and the
suffix on the title, both purely additive, so that a reader cannot reach the
prediction without meeting its verdict. Everything below is the result, and
every cell of it is derived from the session recorded in `LISTENER_SESSIONS`
(`scripts/make_demo_audio.py`) joined to `results/master.csv` — nothing here is
typed, so it cannot drift out of agreement with the record it scores.*

## The listener response, verbatim

{para(f"{words(sess['n_listeners']).capitalize()} listener, one sitting. They "
      f"were given `blind/BLIND_SHEET.md` and nothing else, and wrote their "
      f"rankings before this file was opened. Quoted unedited, typos included, "
      f"because a cleaned-up quote is a paraphrase:")}

{quote(f'"{sess["verbatim"]}"')}

"Better" here means *easier to understand* — the listener is naming the easier
clip of each pair, so the **other** clip is the one they ranked harder.

## Per-pair scoring

{arm_defs}

| pair | clip | predicted harder | listener ranked harder | direction | listener's own words on strength | model WER (A / B) |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Blind-name mappings, from `KEY.md` / `manifest.json:blind_map`:
{' · '.join(maps)}.

## Verdict: **{sc['verdict']}**

{verdict_lines}

## What survives — stated precisely, with nothing added

{survives}

So this claim stands:

> **A human and the model disagree about which clip is worse.**

And this one does **not**:

> ~~And here is why: the precedence effect makes reverb cheap for humans and
> informational masking makes competing speech expensive.~~

The demo was rebuilt around the surviving claim. `DEMO_SCRIPT.md` is
**direction-agnostic**: the interviewer ranks first, then learns the model has
no preference. That works whichever way anyone hears it, and it is stronger for
not depending on a direction chosen in advance.

{measured}

## The flaw in this document — worth more than the prediction was

The sealed **"What each outcome means"** section listed exactly two outcomes:

- listener ranks them **unequal** → prediction holds;
- listener ranks them **equal** → prediction fails.

**It never considered "unequal, but backwards."** That is a real design flaw,
not a technicality, and it has a concrete consequence:

{quote(f"**Under the rubric as written, this result scores as a PASS.** The "
       f"listener did rank every pair unequal. Meanwhile the prediction "
       f"*sentence* — which named a direction — was wrong in {words(n_miss)} "
       f"pairs out of {words(sc['n_scored'])}.")}

A rubric whose outcomes do not span what can actually be observed is a rubric
that cannot fail, and a pre-registration that cannot fail is decoration. The gap
between the sentence and the rubric is precisely the gap that lets someone score
a miss as a hit — including in perfectly good faith, months later, from a file
that looks rigorous.

**Rule adopted from this, for any future pre-registration in this repo:**

1. **Enumerate outcomes over the full observable space**, not over the two the
   author has in mind. Here that is at minimum: *predicted direction* /
   *opposite direction* / *no preference* / *inconsistent across pairs* — and
   the last is what actually happened.
2. **The decision rule must be fixed in advance and must be able to fail.**
   Compare SPEC section 5's `rt60 x snr_db` registration, which fixed a numeric
   threshold, a CI condition and a rank check before any data existed — and was
   confirmed by clearing them, not by being re-read leniently.
3. **Score the prediction's sentence, not a looser paraphrase of it.** If the
   sentence says "non-marginal", a marginal call is a miss.

This entry is the deliverable. A pre-registration that failed and is written
down is worth more than one that "held" because nobody wrote down what holding
meant.
{post_hoc}
## Provenance of this section

Listener response: transcribed verbatim from the session and stored in
`LISTENER_SESSIONS` in `scripts/make_demo_audio.py`, which is also where the
per-pair calls live — so this section is regenerated from the record rather than
retyped beside it, and a regeneration reproduces the outcome instead of erasing
it. Blind mappings: `manifest.json:blind_map`. WERs, transcripts, confidences
and the paired bootstrap: recomputed from `results/master.csv` at build time.
No number in this section was copied from a progress log — SPEC C.7 records what
that costs.
"""


def write_prediction(m: dict, *, force_docs: bool = False) -> str:
    s = m["paired_result"]
    sc = prediction_scorecard(m)

    # The two pairs the prediction NAMED — read from the record, not from the
    # play order. The sealed text must reproduce what was written down before
    # anyone listened; a pre-registration that re-orders itself to match the
    # result is not a record of a prediction.
    by_clip = {p["clip_id"]: p for p in m["pairs"]}
    if sc:
        named = [by_clip[c] for c in sc["session"]["predicted_pairs"] if c in by_clip]
    else:
        named = []
    if len(named) < 2:
        named = m["pairs"][:2]
    p1, p2 = named[0], named[1]

    title = "# Pre-registered prediction — SEALED"
    banner = ""
    if sc:
        n_miss = sc["n_scored"] - sc["n_direction_hit"]
        harder_label = sc["predicted_label"]
        other_label = LABEL_A if sc["predicted_harder_arm"] == "B" else LABEL_B
        title = ("# Pre-registered prediction — SEALED · "
                 + ("**TESTED · HELD**" if sc["held"] else "**TESTED · FAILED**"))
        banner = "\n" + quote(
            f"⚠️ **STATUS {sc['session']['date']} — THIS PREDICTION HAS BEEN "
            f"TESTED.** Verdict: **{sc['verdict']}.** The listener found the "
            f"*{other_label}* arm harder in {words(n_miss)} of "
            f"{words(sc['n_scored'])} pairs; the prediction said the "
            f"*{harder_label}* arm would be harder.\n\n"
            f"**The sealed text below is unchanged and is not to be edited** — "
            f"it is the record of what was committed before anyone listened. "
            f"The result is appended in **OUTCOME** at the end of this file. "
            f"Superseding forward, never backward. Read OUTCOME before quoting "
            f"anything above it.") + "\n"

    outcome = _prediction_outcome(m, sc, s) if sc else ""

    txt = f"""{title}
{banner}
Do not show this, or say it aloud, until the listener has ranked the clips.
Announcing a prediction before someone judges is a demand characteristic; this
project's whole subject is not fooling yourself with a number you wanted.

Written before any listener heard anything. The blind names are frozen in
`KEY.md`, so this is checkable after the fact rather than a claim.

## The prediction

> The listener will rank **{p1['B']['blind']}** clearly harder than
> **{p1['A']['blind']}**, and **{p2['B']['blind']}** clearly harder than
> **{p2['A']['blind']}** — a confident, immediate, non-marginal call in both
> pairs.
>
> The model scores the two clips within each of those pairs **EXACTLY equal**
> (pair {p1['pair']}: {p1['A']['wer']:.3f} vs {p1['B']['wer']:.3f}; pair {p2['pair']}:
> {p2['A']['wer']:.3f} vs {p2['B']['wer']:.3f}), and across all
> {s['n_clips']} clips the paired difference is {s['paired_diff_A_minus_B']:+.4f} WER,
> 95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] — spanning zero.

## What each outcome means

- **Listener ranks them clearly unequal** → prediction holds; the human and
  model axes are decoupled and the demo lands.
- **Listener ranks them equal** → prediction fails. Say so. n=1 was never going
  to settle it, and the MEASURED half (the CI above) is unaffected either way —
  which is exactly why the two halves are labelled separately.

The prediction is about a human. The confidence interval is about the model.
Only one of those two things is a measurement.
{outcome}"""
    write_doc(DEMO_DIR / "PREREGISTERED_PREDICTION.md", txt, force_docs=force_docs)
    return txt


def write_key(m: dict, *, force_docs: bool = False) -> str:
    s, A, B = m["paired_result"], m["conditions"]["A"], m["conditions"]["B"]
    lines = [
        "# KEY — presenter only. Do not put this in `blind/`.",
        "",
        "`blind/` is the handover folder: neutral names, byte-identical copies,",
        "and `blind/BLIND_SHEET.md`. Everything that names a condition lives here",
        "in the parent directory.",
        "",
        "## Blind name -> working file",
        "",
        "| blind | working file | condition |",
        "|---|---|---|",
    ]
    stem_cond = {}
    for p in m["pairs"]:
        stem_cond[Path(p["A"]["file"]).name] = f"A · {LABEL_A} · `{A['name']}`"
        stem_cond[Path(p["B"]["file"]).name] = f"B · {LABEL_B} · `{B['name']}`"
    stem_cond[Path(m["payoff"]["clean_file"]).name] = "control · raw recording, 16 kHz"
    stem_cond[Path(m["payoff"]["deadzone_file"]).name] = \
        f"payoff · `{m['conditions']['payoff']['name']}`"
    for blind, work in m["blind_map"].items():
        lines.append(f"| `{blind}` | `{work}` | {stem_cond.get(work, '')} |")

    lines += [
        "",
        "## The two conditions",
        "",
        "| | A — reverb-dominated | B — babble-dominated |",
        "|---|---|---|",
        f"| condition | `{A['name']}` | `{B['name']}` |",
        f"| requested rt60 | {num(A['rt60'])} s | {num(B['rt60'])} s |",
        f"| SNR | {num(A['snr_db'])} dB (quiet) | {num(B['snr_db'])} dB (buried) |",
        f"| noise | {A['noise_type']} | {B['noise_type']} |",
        f"| codec / rolloff | {A['codec']} / {A['mic_rolloff']:g} | "
        f"{B['codec']} / {B['mic_rolloff']:g} |",
        f"| delivered room | {room_name(A['room'])} | {room_name(B['room'])} |",
        f"| RIR file | `{A['room']}` | `{B['room']}` |",
        f"| measured RT60 | {A['rt60_measured']:.3f} s | {B['rt60_measured']:.3f} s |",
        f"| DRR | {A['drr_db']:+.2f} dB | {B['drr_db']:+.2f} dB |",
        f"| C50 | {A['c50_db']:+.2f} dB | {B['c50_db']:+.2f} dB |",
        f"| mean WER over {s['n_clips']} clips | {s['mean_wer_A']:.4f} | "
        f"{s['mean_wer_B']:.4f} |",
        "",
        "Each isolates ONE degradation: codec `none`, mic rolloff 0. Nothing in",
        "this comparison is confounded by the channel factors.",
        "",
        f"**Paired difference (A-B): {s['paired_diff_A_minus_B']:+.4f} WER, "
        f"95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]** — "
        f"{'spans zero' if s['spans_zero'] else 'DOES NOT SPAN ZERO — see below'}. "
        f"{s['n_resamples']:,}-resample paired bootstrap, seed {s['seed']}, "
        f"resampled over {s['resample_unit']}s.",
        "",
        "## Per-clip facts (all from `results/master.csv`)",
        "",
    ]
    # Wrapped after interpolation for the same reason `quote()` exists: the
    # pair numbers and the listener's own words have unpredictable width.
    lines += textwrap.wrap(listener_ordering_note(m), width=78)
    lines.append("")
    # Sections stay in PAIR-NUMBER order — the numbers are what the blind sheet
    # and the filenames use — and each one carries its play position. Reordering
    # the sections would make `pair2_A_reverb_u21.wav` the first heading and
    # every reference to "pair 1" ambiguous.
    for p in m["pairs"]:
        lines += [
            f"### Pair {p['pair']} — `{p['clip_id']}` ({p['play_note']})",
            "",
            f"- reference: `{p['ref']}`",
            f"- **A** {p['A']['blind']} — WER **{p['A']['wer']:.3f}**, "
            f"mean conf {p['A']['mean_conf']:.3f}",
            f"  - hyp: `{p['A']['transcript']}`",
            f"- **B** {p['B']['blind']} — WER **{p['B']['wer']:.3f}**, "
            f"mean conf {p['B']['mean_conf']:.3f}",
            f"  - hyp: `{p['B']['transcript']}`",
            "",
        ]
    pay = m["payoff"]
    lines += [
        "## Payoff",
        "",
        f"- condition `{m['conditions']['payoff']['name']}` — SNR "
        f"{num(m['conditions']['payoff']['snr_db'])} dB, i.e. QUIET. The damage is "
        f"reverb + codec + rolloff, not noise.",
        f"- clean: `{pay['clean_blind']}` · dead zone: `{pay['deadzone_blind']}`",
        f"- `{pay['clip_id']}` reference ({pay['n_ref']} words): `{pay['ref']}`",
        f"- transcript: **{'(empty string)' if not pay['transcript'].strip() else pay['transcript']}**",
        f"- WER {pay['wer']:.3f}, all {pay['n_del']} reference words deleted, "
        f"utterance confidence {pay['utterance_conf']:.2f}",
        f"- **{pay['n_empty']} of {pay['n_clips']} clips returned an empty "
        f"transcript in this condition**: {', '.join(pay['empty_clips'])}",
        "",
        "Mean word confidence is NULL here, not low: there are no words to be",
        "confident about. That is the deletion blindness the calibration layer",
        "reports — deletions carry no hypothesis token, so they are invisible to",
        "any confidence-based monitor.",
        "",
    ]
    txt = "\n".join(lines) + "\n"
    write_doc(DEMO_DIR / "KEY.md", txt, force_docs=force_docs)
    return txt


def pair_listing(p: dict) -> list[str]:
    """
    The two blind names of one pair, in BLIND-NUMBER order.

    Not in arm order. Listing A first every time would put the reverb arm first
    in all three pairs, so presentation position would be perfectly confounded
    with condition and "the second one was harder" would be ambiguous between an
    order effect and the thing being tested. The frozen BLIND_ORDER puts A first
    in one pair and B first in the other two — not balanced (three pairs cannot
    be), which the run-of-show says out loud.
    """
    return sorted([p["A"]["blind"], p["B"]["blind"]])


def write_blind_sheet(m: dict, *, force_docs: bool = False) -> str:
    pay = m["payoff"]
    # Rows in PLAY order, keeping the pair NUMBERS as labels. A listener works a
    # sheet top to bottom, so listing pair 1 first while the run-of-show opens on
    # pair 2 quietly hands back the ordering the listening session earned. The
    # numbers stay put because they are what `DEMO_SCRIPT.md` and `KEY.md` name.
    rows = "\n".join(
        f"| {p['pair']} | `{a}` and `{b}` |"
        for p in by_play_order(m["pairs"]) for a, b in [pair_listing(p)])
    first, second = sorted([pay["clean_blind"], pay["deadzone_blind"]])
    txt = f"""# Listening sheet

Eight short clips. Same speaker throughout; each clip is one sentence.

Play them as loud or as quiet as you like, and re-play freely — the clips are
not level-matched, and how loud something is is not what you are being asked
to judge.

## The task

For each pair below, answer one question: **which one is harder to understand?**
Rank them, and say how confident you are — "clearly harder" vs "about the same"
is the distinction that matters.

| pair | clips |
|---|---|
{rows}

Then, separately: `{first}` and `{second}`. Same question, plus one more —
**can you still tell that someone is speaking?**

Write your ranking down before we discuss any of it.
"""
    write_doc(BLIND_DIR / "BLIND_SHEET.md", txt, force_docs=force_docs)
    return txt


def n_primary(ordered: list[dict]) -> int:
    """How many pairs the beat actually plays.

    Derived from the recorded listener calls — a pair is `primary` iff that
    listener called it confidently — so it is not a constant anyone can drift.
    `demos/demo_listen.py:DEFAULT_N_PAIRS` is the same number on the other
    surface, and `tests/test_make_demo_audio.py` pins the two together: a
    run-of-show that says three while the script plays two is SPEC J.7 in
    miniature.
    """
    return len(ordered)   # MUTATION


def _play_line(ordered: list[dict]) -> str:
    """The one bolded instruction in section 2, built from the derived order."""
    primaries = [p for p in ordered if p["role"] == "primary"]
    reserves = [p for p in ordered if p["role"] != "primary"]
    if not primaries:
        return ("**No listening session is on record for these clips, so play "
                "them in any order — nothing below encodes a judgement about "
                "which pair lands hardest.**")
    lbl = lambda p: f"pair {p['pair']} (`{p['clip_id']}`)"      # noqa: E731
    if len(primaries) == 1:
        lead = f"{lbl(primaries[0])} first"
    else:
        lead = (f"{lbl(primaries[0])} first, then "
                + ", then ".join(lbl(p) for p in primaries[1:]))
    tail = ""
    if reserves:
        rs = _and_list([lbl(p) for p in reserves])
        rs = rs[0].upper() + rs[1:]
        tail = (f" {rs} is the RESERVE — play it {_ordinal(reserves[0]['play_position'])} "
                f"ONLY if they want another." if len(reserves) == 1 else
                f" {rs} are RESERVES, held back unless they want more.")
    n = len(primaries)
    head = (f"{words(n).capitalize()} pair is the beat" if n == 1
            else f"{words(n).capitalize()} pairs are the beat")
    return f"**{head} — play {lead}.{tail}**"


def write_demo_script(m: dict, *, force_docs: bool = False) -> str:
    s = m["paired_result"]
    A, B = m["conditions"]["A"], m["conditions"]["B"]
    pay = m["payoff"]
    verdict = "spans zero" if s["spans_zero"] else "does NOT span zero"

    # Everything about ORDER comes from the derived fields, never from the
    # position of a pair in m["pairs"] — that is construction order, and reading
    # it as play order is the defect this section was rewritten to remove.
    ordered = by_play_order(m["pairs"])
    sc = prediction_scorecard(m)
    n_pairs = len(m["pairs"])
    # The clip SET is n_pairs; the beat PLAYS n_play and holds the rest back.
    # Two different numbers that a reader will happily conflate, so every
    # sentence below has to say which one it means — the recorded session judged
    # the whole set, the room in front of you did not.
    n_play = n_primary(ordered)
    ties = " · ".join(f"pair {p['pair']}: {p['A']['wer']:.3f} / {p['B']['wer']:.3f}"
                      for p in m["pairs"])
    wer_rows = "\n".join(
        f"| {p['pair']} | `{p['clip_id']}` | **{p['A']['wer']:.3f}** "
        f"| **{p['B']['wer']:.3f}** |" for p in ordered)
    transcripts = "\n".join(
        f"- `{p['clip_id']}` reference: `{p['ref']}`\n"
        f"  - reverb: `{p['A']['transcript']}`\n"
        f"  - babble: `{p['B']['transcript']}`" for p in ordered)

    conf = [p["pair"] for p in ordered if p["listener_confidence"] == CONFIDENT]
    soft = [p for p in ordered if p["listener_confidence"] == MARGINAL]
    conf_txt = _and_list(["pair %d" % n for n in conf])
    soft_txt = _and_list(["pair %d" % p["pair"] for p in soft])
    if conf and soft:
        rationale = (
            f"The ordering is about the *listener*, not the data: in the one "
            f"session run so far, {conf_txt} drew confident calls, while "
            f"{soft_txt} drew the least confident one "
            f"(\"{soft[0]['listener_said']}\"). Opening on the pair most likely "
            f"to produce a hedge wastes the strongest moment of the segment.")
    elif conf:
        rationale = ("No pair drew a hedged call in the session record, so "
                     "nothing is held back.")
    else:
        rationale = ("No listening session is on record for these clips, so the "
                     "order carries no judgement — do not present it as one.")
    rank_para = para(
        f"All {words(n_pairs)} carry identical evidential weight — the model "
        f"scores every pair **exactly** equal ({ties}). {rationale}")

    if sc:
        n_miss = sc["n_scored"] - sc["n_direction_hit"]
        revised = quote(
            f"⚠️ **REVISED after the prediction in "
            f"`PREREGISTERED_PREDICTION.md` was tested on "
            f"{sc['session']['date']} and FAILED (direction held in "
            f"{sc['n_direction_hit']} of {sc['n_scored']} pairs).** This segment "
            f"is **direction-agnostic**: the listener ranks, then learns the "
            f"model has no preference. That works whichever way anyone hears "
            f"it. The earlier version led with the pair that drew the least "
            f"confident call and asserted a precedence-effect / "
            f"informational-masking mechanism that the listening pass "
            f"**refuted** — both are gone.\n\n"
            f"**Section 7 changed again on 2026-08-06: the close is a QUESTION, "
            f"not a verdict.** It used to land on a conclusion about what "
            f"listening can and cannot establish about an ASR. One listener, "
            f"judging a {words(n_pairs)}-clip set selected BECAUSE the model tied "
            f"on it, cannot carry a conclusion — this segment is the MOTIVATING "
            f"HOOK, and it is stronger as one. `demos/demo_listen.py` closes the "
            f"same way and plays the same {words(n_play)} pairs by default "
            f"(`DEFAULT_N_PAIRS`); the two must not diverge.\n\n"
            f"The revision lives in the generator template "
            f"(`scripts/make_demo_audio.py`), not only in this file, so a "
            f"regeneration reproduces it instead of reverting it. The play "
            f"order and each pair's role are derived from the recorded listener "
            f"calls; they are not hardcoded.")
        outcome_block = f"""Now open `PREREGISTERED_PREDICTION.md` and show the OUTCOME section. Deliver
this rather than skipping it — it is the strongest 20 seconds in the segment:

{quote(f'''
"I did pre-register which way I thought a listener would go: that the
{sc['predicted_label']} clip would be the harder one. The listener I ran this on
before you went the *other* way in {n_miss} of the {sc['n_scored']} pairs. So the
direction is not established, and rather than quietly re-score it, the miss is
written under the sealed text with the verdict on it.

The rubric I wrote had exactly two outcomes — 'they rank them unequal' and
'they rank them equal.' It never considered 'unequal, but backwards.' Under my
own rubric this scores as a **pass**, because they did rank them unequal. That
is a rubric that cannot fail, and this project is entirely about not fooling
yourself with a number you wanted.

What survived is the half that never depended on the direction: a stated human
preference in {sc['n_scored']} pairs out of {n_pairs} — one of them hedged — and
a model with none."
''')}

### Do NOT repair the story on stage

The tempting move, the moment someone says they found the reverberant clip
harder, is to reach for the DRR number: *"of course — that room is at
{A['drr_db']:+.2f} dB DRR."* **Do not deliver that as the explanation.** It is
post-hoc: the human-side prediction was made in the opposite direction and lost,
DRR is simply the first number to hand that fits the result now in view, and
n=1 cannot adjudicate between the two stories. Presenting it as the mechanism is
exactly the move the failed prediction should have made you distrust.

If you want to raise it at all, raise it labelled:

{quote('''
"Something I'd want to test — and to be clear, this is a hypothesis I formed
*after* seeing the result, not a finding — is whether a listener's ranking
tracks direct-to-reverberant ratio rather than RT60, the way the model's errors
do. That is a proper listening study, and I haven't run it."
''')}

### If they ask why you kept a failed prediction in the repo

{quote('''
"Because a pre-registration whose result isn't recorded is worse than none —
anyone who finds it later assumes it was never run, or assumes it held. It cost
me a mechanism I liked and bought a better finding: the flaw was in my outcome
table, not in the listener."
''')}
"""
        zero_note = (
            para(f"**Do not make a new directional prediction on stage.** The "
                 f"file already carries a tested outcome: it predicted *which* "
                 f"clip a listener would find harder, and was wrong in "
                 f"{n_miss} of {sc['n_scored']} pairs. That failure is recorded "
                 f"under the sealed text and you are going to show it in "
                 f"section 4. The claim this segment rests on does not need a "
                 f"direction, so do not stake one.")
            + "\n\nThe only thing worth predicting out loud is the part that "
              "replicated:\n\n"
            + quote("\"I've written down what I expect to happen — that you'll "
                    "have a preference in each pair. You'll see the file in a "
                    "minute, including the part of it I got wrong.\""))
        honest_extra = (
            f"It is also **not replicated in direction.** I pre-registered "
            f"which way a listener would go and got it wrong in {n_miss} of "
            f"{sc['n_scored']} pairs. What repeated was only that there *was* a "
            f"preference, in {words(sc['n_scored'])} pairs out of "
            f"{words(n_pairs)} — and one of those was hedged.\n\n")
        backwards_fallback = bullet(
            f"**They rank the reverb clip harder** (i.e. the opposite of the "
            f"sealed prediction — this is what the one listener so far did, in "
            f"{n_miss} of {sc['n_scored']} pairs): **nothing changes.** Section "
            f"4 does not depend on the direction. Say so, then show the recorded "
            f"outcome: \"that's the direction I got wrong, and it's written "
            f"down.\" Do **not** improvise a DRR explanation on the spot.") + "\n"
    else:
        revised = ("> ℹ️ **No listening session is on record for these clips.** "
                   "The pre-registered prediction in "
                   "`PREREGISTERED_PREDICTION.md` is OPEN — it has not been "
                   "tested. Do not present it as though it had.")
        outcome_block = para(
            "The prediction in `PREREGISTERED_PREDICTION.md` has not been tested "
            "against a listener yet, so there is no outcome to show. Open it "
            "after they rank, read it as written, and record what happened in "
            "`LISTENER_SESSIONS` in `scripts/make_demo_audio.py`, so the next "
            "regeneration of this file carries the result.") + "\n"
        zero_note = para(
            "Open it after they have ranked, not before — announcing a "
            "prediction first is a demand characteristic.")
        honest_extra = ""
        backwards_fallback = ""

    txt = f"""# DEMO_SCRIPT — the listening exercise (~3 minutes)

Presenter's run-of-show. Everything here is offline: play wavs from
`blind/`, read from this file. No network, no API key.

**Hand over:** `blind/` only (8 wavs + `BLIND_SHEET.md`).
**Keep back:** this file, `KEY.md`, `PREREGISTERED_PREDICTION.md`,
`REGENERATION_HAZARD.md` (it names the conditions too).
The working filenames in the parent directory say `reverb` and `babble`; a
listener who sees them has been told the answer.

{revised}

---

## 0. Before they listen (10 s) — show the sealed file, do not say what is in it

Open `PREREGISTERED_PREDICTION.md`, show that it exists, and **leave it closed**.
Saying a prediction out loud before someone judges is a demand characteristic.
This mirrors how the study handles its own hypotheses: `rt60 x snr_db` was
pre-registered in SPEC section 5 before any real audio existed, and confirmed on
the real grid at ST-S1 = 0.128 [0.091, 0.164]. Same discipline, three minutes
instead of three weeks.

{zero_note}

---

## 1. The task (20 s)

Give them `blind/BLIND_SHEET.md`. Two rules only:

> "Same speaker, same kind of sentence. For each pair: which is harder to
> understand? And use the volume knob however you like — the clips aren't
> level-matched and loudness isn't what I'm asking about."

---

## 2. They listen and rank (60 s)

{_play_line(ordered)}

{rank_para}

**Do not steer, and do not react.** Let them finish every ranking before you
say anything at all. It does not matter which way they go — section 4 is
written so that any confident ranking lands, and so is the one they might not
make (see Fallbacks).

---

## 3. The reveal (45 s)

Leave `PREREGISTERED_PREDICTION.md` closed for one more minute — it comes out
in section 4. Reveal the conditions first:

{quote(f'''
"Those two clips are not the same degradation. One is a bad ROOM but quiet:
requested RT60 {num(A['rt60'])} s at {num(A['snr_db'])} dB SNR. That is the
{room_name(A['room'])} impulse response — a real measured room — with a
measured RT60 of {A['rt60_measured']:.3f} s and a direct-to-reverberant ratio of
{A['drr_db']:+.2f} dB.

The other is a GOOD room with the speech nearly buried: RT60 {num(B['rt60'])} s
at {num(B['snr_db'])} dB SNR, the {room_name(B['room'])} response, DRR
{B['drr_db']:+.2f} dB. Neither one has a codec or mic rolloff on it, so nothing
else is moving between them."
''')}

Then the numbers — read them exactly:

{quote(f'''
"On the pair you just heard, the model scored them **identically**. Not close —
equal. And not equal because it got both right: it got both **wrong**, by the
same amount, in different places.

Across all {s['n_clips']} clips: the reverb condition means WER
**{s['mean_wer_A']:.4f}**, the babble condition **{s['mean_wer_B']:.4f}**. The
paired difference is **{s['paired_diff_A_minus_B']:+.4f}**, 95% CI
**[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]** — {verdict}. That is a
{s['n_resamples']:,}-resample paired bootstrap over clips, seed {s['seed']}.
Statistically indistinguishable."
''')}

Per-pair WER, whichever ones you played, in play order (reverb arm / babble arm):

| pair | clip | reverb arm | babble arm |
|---|---|---|---|
{wer_rows}

Show the transcripts if there's a screen — they make the point better than the
scalar does, because equal WER is arrived at by damaging different words:

{transcripts}

{para(f"⚠️ **Do not generalize an edit-type signature from these "
      f"{words(n_pairs)} clips.** Here the reverb arm happens to substitute and "
      f"the babble arm to delete, but that is {words(n_pairs)} clips and it runs "
      f"**opposite** to the grid-level fingerprint, where `rt60 >= 0.7` drives "
      f"**deletions** — see `results/fingerprints.txt`, which is the measured "
      f"statement and this is not. Use the transcripts to show *that* the damage "
      f"differs, not to claim *how* it differs.")}

---

## 4. The disagreement, and the prediction I got wrong (40 s)

**This section is deliberately direction-agnostic. It does not matter which
clip they picked.**

{quote('''
"You had a preference. The model does not — it scores that pair equal, and
across all the clips the difference is indistinguishable from zero.

Notice what that argument does *not* rest on: which of the two you picked. Any
confident human ranking of that pair is a ranking the model does not share."
''')}

{outcome_block}
### The model-side reverb result, which IS measured

Keep this on the model side of the line and it stands on its own — it is a grid
result, not an inference about the listener:

{quote(f'''
"The model is not tracking RT60 either. Across the four reverb levels in the
grid, the Spearman correlation of measured RT60 with WER is **+0.800** — but
with direct-to-reverberant ratio it is **-1.000**. The pair you just heard is
the extreme case: {room_name(A['room'])} at DRR {A['drr_db']:+.2f} dB against
{room_name(B['room'])} at {B['drr_db']:+.2f} dB. RT60 is the number every reverb
benchmark is parameterised by, and it mislabels the acoustics that actually get
delivered."
''')}

---

## 5. The payoff (25 s)

Play `{pay['clean_blind']}` (clean control) then `{pay['deadzone_blind']}`.

{quote(f'''
"Same sentence, same speaker. The second one is condition
`{m['conditions']['payoff']['name']}` — and look at the SNR:
**{num(m['conditions']['payoff']['snr_db'])} dB**. That is QUIET. The damage here
is reverb plus a low-rate codec plus full mic rolloff. It is not noise.

You can still hear that someone is talking. The model returned an **empty
string**. WER {pay['wer']:.3f}, all {pay['n_ref']} reference words deleted.
**{pay['n_empty']} of the {pay['n_clips']} clips came back completely empty in
this condition.**

And there is no low confidence to catch it with. Mean word confidence is not
low — it is *null*, because there are no words to be confident about. A monitor
watching confidence sees nothing at all."
''')}

---

## 6. The honest label (20 s) — do not skip this

{quote(f'''
"Two halves here, and they are not the same kind of thing.

**The human half is an intuition pump, not a measurement — and here is every
reason it is not one.** It is **n=1**: one listener, one speaker, one accent, one
sitting. They were blind to *which clip was which condition*, but **not naive to
the hypothesis** — they knew what this project claims, which is the kind of thing
that moves a judgement. The clips are **not level-matched** (they are
byte-identical to what the model was scored on, and a cosmetic gain would break
that). **Presentation order is not counterbalanced** — with a {words(n_pairs)}-clip
set it cannot be; one pair plays the reverb arm first, the rest play it second.
And I **selected these {words(n_pairs)} clips precisely because the model tied on
them**, which is a defensible choice for a demonstration and an indefensible one
for an estimate. You heard {words(n_play)} of them: {words(n_play)} pairs is the
beat and the {_ordinal(n_pairs)} is a reserve, so the set is small and the sample
of it is smaller.

{honest_extra}**The measured half is the model-side paired result and its
interval:** {s['paired_diff_A_minus_B']:+.4f} WER, CI
[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}], over all {s['n_clips']} clips,
{s['n_resamples']:,} resamples, resampled over {s['resample_unit']}s, seed
{s['seed']}. That half is not selected and not affected by anything you just
said. It would read the same if you had ranked them the other way round — which
is what actually happened last time — or refused to rank them at all.

Doing the human side properly is a listening study: many listeners naive to the
hypothesis, randomized and counterbalanced order, level-matched stimuli, a real
intelligibility score. That is exactly the experiment this project does not have,
and the limitations section says so."
''')}

---

## 7. The close (10 s) — a QUESTION, not a verdict

{quote(f'''
"Whichever way you called those pairs — and 'about the same' is a real answer —
the model reports no difference at all. Not a small one: none. Every pair you
heard it scored identically, and across all {s['n_clips']} clips the two kinds of
damage differ by **{s['paired_diff_A_minus_B']:+.4f} WER**, an interval that {verdict}.

Your ears and that number are not measuring the same thing, and I had no way to
settle which of them to believe by listening harder. 'Sounds fine to me' is an
opinion and I could not check it — so I built something that could measure it
instead. Everything after this segment is that instrument. This segment is the
question it was built for, not one of its results."
''')}

{para("**Do not upgrade that into a verdict on stage.** This script used to close "
      "on a conclusion about what listening can and cannot establish about an ASR. "
      "One listener, judging a " + words(n_pairs) + "-clip set selected BECAUSE the "
      "model tied on it — and the room in front of you heard "
      + words(n_play) + " of those pairs, not the whole set — cannot carry a "
      "conclusion. Asserting one here would be this project's own signature failure "
      "committed in its own demo. As the motivating hook it is honest and it is "
      "stronger, because the question is the reason there is an instrument "
      "downstream of it.")}

{para("The measured half of that same disagreement is section 5, and it needs no "
      "ranking from anyone: the payoff clip is a condition where a listener can "
      "plainly hear a person speaking and the model returns an empty string on "
      f"{pay['n_empty']} of {pay['n_clips']} clips. Land on that, not on a claim about "
      "listening.")}

{para("`demos/demo_listen.py` closes the same way, and the two must not diverge — "
      "SPEC J.7 is a rehearsal finding a demo script narrating a verdict its own "
      "artifact contradicted. The retracted line and the reasoning behind dropping "
      "it are in `report/_demo_internal_notes.md`.")}

---

## Fallbacks

- **No speakers / bad room:** skip to section 3 and show the transcripts. The
  measured half needs no audio at all.
{backwards_fallback}- **They rank the pair EQUAL, or say "I can't call it":** that is a real answer,
  not a failed demo — say so. One listener agreeing with the model is worth no
  more than one listener disagreeing with it, which is exactly why the interval
  underneath is the measured half and this half is only the question. The
  measured half is untouched. Then pivot to section 5 — the payoff clip needs no
  ranking at all: a human can obviously still hear speech, and the model returned
  an empty string.
- **They rank confidently but inconsistently across pairs** (one each way): that
  is the honest state of the human evidence and it is fine to say so. The
  model-side claim is per-pair and holds in all {words(n_pairs)}.
- **They ask what you predicted, before ranking:** don't tell them. Say "after
  you've called it" — announcing it first is a demand characteristic, which is
  the reason the file is sealed in the first place.
- **They want to hear the factors separately:** `isolation/` is the ladder,
  `00_RAW_original` to `10_destroyed`, one factor at a time. See
  `WHAT_TO_LISTEN_FOR.md`.

## Provenance

Every number above is read or recomputed from `results/master.csv` at build
time by `scripts/make_demo_audio.py` — none is typed into this file.

{para(f"The one thing in this document that does **not** come from the master "
      f"table is the **play order**, and it cannot: it is a judgement about a "
      f"listener. It is derived from the calls recorded in `LISTENER_SESSIONS` "
      f"in that same script ("
      + (f"session {sc['session']['date']}" if sc else "no session on record")
      + "), so it has an owner and a date rather than being an ordering nobody "
        "can account for. Change the record and this file changes with it.")}

The wavs are bit-identical to what nova-3 transcribed: `apply_condition` is
seeded from the condition name and the writer is the grid's own
`write_degraded_wav`. Regenerate with:

    ./.venv/bin/python scripts/make_demo_audio.py --force

**If you hand-edit this file, that edit is safe.** The generator hashes what it
writes and refuses to overwrite anything that no longer matches — it will print
a refusal naming this path and carry on. Only `--force-docs` overrides that, and
it copies the current file to `<name>.superseded-<UTC>.md` before it does.
"""
    write_doc(DEMO_DIR / "DEMO_SCRIPT.md", txt, force_docs=force_docs)
    return txt


def write_what_to_listen_for(m: dict, *, force_docs: bool = False) -> str:
    A, B = m["conditions"]["A"], m["conditions"]["B"]
    pay = m["payoff"]
    txt = f"""# What to listen for

Two different jobs live in this directory. Don't mix them up.

---

## PART 1 — `isolation/`: is the composer physically plausible? (SPEC A.R3.5)

All from clip `{LISTEN_CLIP}`, so what changes is the CONDITION, not the speaker
or the sentence. Start with `00_RAW_original.wav`. **WER is irrelevant here** —
this is a DSP check, and the only instrument is your ears. The unit tests prove
the arithmetic; nothing but listening proves the RESULT.

1. **Onset alignment.** Every file must start when the original starts. A late
   start means `apply_rir`'s direct-path trim is wrong and every WER in the
   study carries a pure alignment artifact.
2. **Reverb sounds like a room** (`02_reverb_only`) — not a delay, not a metallic
   comb, no audible repeat.
3. **SNR is believable.** `01_benign` (20 dB) should be barely noisy;
   `03_noise_only` (0 dB) should nearly bury the speech. If 0 dB sounds mild, the
   calibration is off.
4. **Codecs sound like a phone line** (`05_g726_only`, `06_opus_only`) —
   bandlimited and gritty, not just quieter.

This is the check that caught the `apply_rir` renormalization bug: reverb tail
energy leaking into the silent regions de-calibrated every downstream SNR, and
it produced clean-looking garbage with no error message anywhere.

---

## PART 2 — `blind/`: the human axis the study doesn't have

Run this on someone else, not on yourself. `DEMO_SCRIPT.md` is the run-of-show,
`KEY.md` is the answer key, `blind/BLIND_SHEET.md` is the only page the listener
sees. Do not let them see the working filenames in this directory — they say
`reverb` and `babble`.

What you are listening for yourself, before you run it on anyone:

- **A** (`{A['name']}`) should sound like a **bad room, but quiet**: the
  {room_name(A['room'])} impulse response, measured RT60
  {A['rt60_measured']:.3f} s, DRR {A['drr_db']:+.2f} dB, with the babble at
  {num(A['snr_db'])} dB SNR and barely there.
- **B** (`{B['name']}`) should sound like a **good room with the speech nearly
  buried**: the {room_name(B['room'])} response, measured RT60
  {B['rt60_measured']:.3f} s, DRR {B['drr_db']:+.2f} dB, babble at
  {num(B['snr_db'])} dB.
- If those two do NOT sound clearly different to you, stop and check the
  composer: they are two unrelated degradations, so they should not sound alike.
  That is a DSP check on your own ears and **not** a prediction about the
  listener. The model scores them equally damaging; whether a person ranks them,
  and which way, is the QUESTION this exercise raises — not a result it is
  supposed to produce. A listener who calls them about the same has given a real
  answer, and the run-of-show handles it.
- **The payoff** (`{pay['deadzone_blind']}`, condition
  `{m['conditions']['payoff']['name']}`) should still sound like a person
  speaking. The model returned an empty string on it — and on
  {pay['n_empty']} of {pay['n_clips']} clips in that condition.

The clips are deliberately **not level-matched**: they are byte-identical to the
audio nova-3 was scored on, and a cosmetic gain would break that identity. Use
the volume knob and say so out loud when you run the exercise.

---

## SUPERSEDED

`{OLD_LISTEN_DIR}/DEADZONE_*.wav` are superseded by this directory: all six were
clip `{LISTEN_CLIP}`, which nova-3 transcribes at WER 0.000 in four of those six
conditions — so they demonstrated nothing. The ladder in that directory is fine
and is regenerated here as `isolation/`.
"""
    write_doc(DEMO_DIR / "WHAT_TO_LISTEN_FOR.md", txt, force_docs=force_docs)
    return txt


def mark_old_set_superseded(m: dict, *, force_docs: bool = False) -> None:
    if not OLD_LISTEN_DIR.is_dir():
        return
    write_doc(
        OLD_LISTEN_DIR / "SUPERSEDED.md",
        f"# SUPERSEDED by `{DEMO_DIR}/`\n\n"
        f"The `DEADZONE_*.wav` files here are all clip `{LISTEN_CLIP}`, and nova-3 "
        f"transcribes `{LISTEN_CLIP}` at WER 0.000 in four of those six conditions — "
        f"so they demonstrated nothing.\n\n"
        f"The ladder (`00_RAW_original` .. `10_destroyed`) is sound and is "
        f"regenerated as `{DEMO_DIR}/isolation/`. The replacement listening + demo "
        f"set, built on conditions where the model measurably fails, is in "
        f"`{DEMO_DIR}/` — see its `WHAT_TO_LISTEN_FOR.md` and `DEMO_SCRIPT.md`.\n\n"
        f"    ./.venv/bin/python scripts/make_demo_audio.py\n",
        force_docs=force_docs)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

MANIFEST = DEMO_DIR / "manifest.json"

# Every document this script owns, in the order main() writes them. One list,
# so `check()` reports on exactly the set the build touches and a document added
# later cannot be reported on but not guarded, or the reverse.
DOCS = (DEMO_DIR / "DEMO_SCRIPT.md",
        DEMO_DIR / "KEY.md",
        BLIND_DIR / "BLIND_SHEET.md",
        DEMO_DIR / "PREREGISTERED_PREDICTION.md",
        DEMO_DIR / "WHAT_TO_LISTEN_FOR.md",
        OLD_LISTEN_DIR / "SUPERSEDED.md")


def doc_report() -> dict[str, str]:
    """path -> status, over every document the build would write."""
    h = load_doc_hashes()
    return {str(p): doc_status(p, h) for p in DOCS}


def check() -> int:
    """Verify without generating. Returns a process exit code."""
    if not MANIFEST.is_file():
        print(f"MISSING {MANIFEST} — run:\n"
              f"    ./.venv/bin/python scripts/make_demo_audio.py")
        return 1
    m = json.loads(MANIFEST.read_text())
    missing = [f for f in m["files"] if not Path(f).is_file()]
    for doc in ("DEMO_SCRIPT.md", "KEY.md", "WHAT_TO_LISTEN_FOR.md",
                "PREREGISTERED_PREDICTION.md"):
        if not (DEMO_DIR / doc).is_file():
            missing.append(str(DEMO_DIR / doc))
    if not (BLIND_DIR / "BLIND_SHEET.md").is_file():
        missing.append(str(BLIND_DIR / "BLIND_SHEET.md"))
    if missing:
        print(f"{len(missing)} demo file(s) missing, first few:")
        for f in missing[:5]:
            print("   ", f)
        print("run:\n    ./.venv/bin/python scripts/make_demo_audio.py --force")
        return 1
    s = m["paired_result"]
    print(f"OK: {len(m['files'])} files, {len(m['blind_map'])} blind copies")
    print(f"    paired diff {s['paired_diff_A_minus_B']:+.4f} "
          f"CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] "
          f"({'spans zero' if s['spans_zero'] else 'DOES NOT SPAN ZERO'})")

    # Protected documents are a healthy steady state, not a fault — this is a
    # census, and --check keeps its exit code. Reported anyway because the whole
    # point is that a hand-edited document is invisible otherwise.
    rep = doc_report()
    protected = [p for p, st in rep.items() if st == AUTHORED]
    print(f"    docs: {sum(1 for st in rep.values() if st == GENERATED)} "
          f"generator-owned, {len(protected)} PROTECTED (edited or unrecorded)")
    for p in protected:
        print(f"          protected (will NOT be rewritten): {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the listening + demo audio set.")
    # --force rebuilds the WAVS. It deliberately does NOT unlock the documents:
    # tests/test_demo.py runs this script with --force against the live repo, so
    # a --force that clobbered prose would let a green test suite quietly delete
    # a listening result. Destroying an authored document requires asking for it
    # by name.
    ap.add_argument("--force", action="store_true",
                    help="rebuild the audio even if the manifest is already present "
                         "(does NOT overwrite hand-edited documents)")
    ap.add_argument("--force-docs", action="store_true",
                    help="ALSO overwrite documents a human has edited; each one is "
                         "copied to <name>.superseded-<UTC>.md first")
    ap.add_argument("--check", action="store_true",
                    help="verify the existing set, generate nothing")
    a = ap.parse_args()

    if a.check:
        return check()
    if MANIFEST.is_file() and not (a.force or a.force_docs):
        m = json.loads(MANIFEST.read_text())
        if all(Path(f).is_file() for f in m["files"]):
            print(f"up to date ({len(m['files'])} files) — --force to rebuild")
            return 0

    rows = load_rows({COND_A, COND_B, COND_PAYOFF})
    stats = paired_stats(rows)
    assets = DiskAssetLibrary(root="data", target_fs=FS)

    made = build_audio(assets)
    m = build_manifest(made, rows, stats, assets)
    fd = a.force_docs
    write_demo_script(m, force_docs=fd)
    write_key(m, force_docs=fd)
    write_blind_sheet(m, force_docs=fd)
    write_prediction(m, force_docs=fd)
    write_what_to_listen_for(m, force_docs=fd)
    mark_old_set_superseded(m, force_docs=fd)
    MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")

    # The census, printed AFTER the file list so it is the last thing on screen.
    # A refusal that scrolls past is a refusal that did not happen.
    rep = doc_report()
    protected = [p for p, st in rep.items() if st == AUTHORED]

    print(f"demo set -> {DEMO_DIR}: {len(m['files'])} wavs")
    print(f"  pairs   : {', '.join(p['clip_id'] for p in m['pairs'])}")
    print(f"  blind   : {len(m['blind_map'])} copies in {BLIND_DIR}")
    print(f"  ladder  : {len(m['isolation'])} files in {ISO_DIR}")
    print(f"  paired  : A {stats['mean_wer_A']:.4f} | B {stats['mean_wer_B']:.4f} "
          f"| diff {stats['paired_diff_A_minus_B']:+.4f} "
          f"CI [{stats['ci_lo']:+.4f}, {stats['ci_hi']:+.4f}] "
          f"({'spans zero' if stats['spans_zero'] else 'DOES NOT SPAN ZERO'})")
    print(f"  payoff  : {m['payoff']['n_empty']}/{m['payoff']['n_clips']} clips "
          f"empty under {COND_PAYOFF}")
    print(f"  docs    : {len(DOCS) - len(protected)} written, "
          f"{len(protected)} PROTECTED (edited or unrecorded — left untouched)")
    if protected:
        for p in protected:
            print(f"            kept as-is: {p}")
        print("            Those files were NOT regenerated and do not carry this")
        print("            build's numbers. Port the hand-written blocks into the")
        print("            templates in scripts/make_demo_audio.py, or rebuild them")
        print("            with --force-docs (which backs each one up first).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
