#!/usr/bin/env python3
"""
demo_listen.py — the interactive beat. A blind listening test, run live.

This is the one segment of the demo kit where the audience does something. The
interviewer puts on headphones, hears pairs of clips, and says which one is
harder to understand. THEY form the opinion, out loud, before they are told
anything. Then they learn that the model scores every pair **exactly equal**.

    ./.venv/bin/python demos/demo_listen.py            # the live beat (~3 min)
    ./.venv/bin/python demos/demo_listen.py --replay   # rehearse it alone
    ./.venv/bin/python demos/demo_listen.py --sheet     # print the paper sheet
    ./.venv/bin/python demos/demo_listen.py --check     # preflight the artifacts

WHY THIS BEAT EXISTS. Every other demo in this kit shows you a number. This one
manufactures a disagreement and puts the human on the wrong side of it -- which
is the only way to make "you cannot QA a voice agent by listening to it" land as
something the listener discovered rather than something they were told.

THE SCREEN IS SAFE TO SHOW until the reveal. Every line printed before the
listener commits is routed through `bprint()`, which runs `find_leaks()` over it
and REDACTS anything naming a condition, a room, an SNR, an RT60 or a WER. The
guarantee is structural, not a promise in a comment: `tests/test_demo_listen.py`
asserts the whole pre-commit half of a real run is clean, and carries a negative
control that plants a condition name and checks the detector catches it.

WHAT IS MEASURED AND WHAT IS NOT -- the script says this out loud and so should
you. The model-side result (paired difference -0.0178 WER, 95% CI spanning zero,
over all 40 clips) is a measurement. The listener half is **n=1, unblinded to the
hypothesis, not counterbalanced, not level-matched, and the three clips were
selected precisely because the model tied on them**. It is an intuition pump. It
is not data, and the script refuses to present it as data.

THE DIRECTION IS NOT THE FINDING. A sealed prediction (results/audio/demo/
PREREGISTERED_PREDICTION.md) said the babble arm would be the harder one; the one
listener run so far went the other way in 2 of 3 pairs. That miss is reported
here rather than quietly re-scored. Nothing in this script tells the listener
which way they "should" hear it -- the claim rests on there being a confident
preference at all, against a model that has none.

EVERY NUMBER IS READ, NOT TYPED. The pairs, transcripts, conditions and the
paired interval come from `results/audio/demo/manifest.json`; the per-clip tie
count and the arm means are recomputed from `results/master.csv` on each run and
cross-checked against the manifest. No network, no API key, nothing cached that
was not derived from the already-paid-for grid.
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
# Makes `deadzone`, `scripts` and `demos` importable when this file is run
# directly (`python demos/demo_listen.py`) with no install step. Harmless when
# it is imported as a module instead.
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
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse the demo kit's terminal + playback primitives rather than growing a
# second set. `play()` in particular already survives a missing file, a missing
# player and a hung player, which is exactly what must not take a live beat down.
from demos.demo_break import Ink, find_player, pad, play, term_width, visible_len  # noqa: F401

DEMO_DIR = Path("results/audio/demo")
MANIFEST = DEMO_DIR / "manifest.json"
BLIND_DIR = DEMO_DIR / "blind"
BLIND_SHEET = BLIND_DIR / "BLIND_SHEET.md"
SESSIONS_DIR = DEMO_DIR / "sessions"
MASTER = Path("results/master.csv")

MODEL = "nova-3"
SESSION_SCHEMA = "deadzone.listen-session/1"

# The first line of the reveal. Everything printed BEFORE the first occurrence
# of this banner is the blind half of the run, and the test slices real stdout on
# it. Changing this string changes what the test checks, so it lives here as a
# constant rather than being typed in two places.
REVEAL_BANNER = "WHAT THE MODEL SCORED"

DEFAULT_N_PAIRS = 2          # the target beat: two pairs, ~3 minutes including talk
MAX_ASK_RETRIES = 5          # never loop forever on unparseable input


# ==========================================================================
# the no-leak guard
# ==========================================================================
#
# A listening test that tells you the answer first is not a listening test. The
# failure is silent in the worst way: the segment still "works", the listener
# still ranks the clips, and the ranking is worthless because it was primed.
# So the guard is a redactor in the print path, not a rule in a docstring.

_STATIC_LEAK_TERMS = (
    # the degradations, by any name
    "reverb", "reverberant", "reverberation", "echo", "babble", "chatter",
    "codec", "opus", "g726", "rolloff", "roll-off", "bandlimit", "narrowband",
    # the parameters
    "rt60", "snr", "signal-to-noise", "drr", "c50", "decibel",
    "impulse response", "direct-to-reverberant",
    # the rooms in this manifest's set (also derived from the manifest below,
    # so a regenerated set with different rooms is still covered)
    "shower", "restaurant", "campground", "dininghall",
    # the model's side of the ledger
    "wer", "word error", "transcript", "transcripts", "mean confidence",
    "word confidence", "dead zone", "deadzone", "silent failure", "spearman",
    "nova-3", "whisper", "master.csv", "condition", "conditions",
)

_STATIC_LEAK_PATTERNS = (
    r"(?<![a-z0-9])\d+(?:\.\d+)?\s*db(?![a-z0-9])",   # "0 dB", "20.0 dB"
    r"(?<![a-z0-9.])\d\.\d{3}(?![a-z0-9])",           # a WER-shaped number
    r"(?<![a-z0-9])u\d{2}(?![a-z0-9])",               # a manifest clip id
    r"(?<![a-z0-9])roll-\d",                          # a condition-name fragment
)


def _bounded(term: str) -> str:
    """
    `term` as a whole token, where `_` and `-` COUNT AS SEPARATORS.

    Python's `\\b` treats `_` as a word character, so `\\bbabble\\b` does not match
    inside `rt60-1_snr-20_babble_none_roll-0` -- i.e. the naive boundary misses
    exactly the strings this guard exists to catch. Plain substring matching has
    the opposite failure: `wer` fires on "answer", and the guard redacts the word
    it needs most in a prompt asking someone for one.
    """
    return r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"


def leak_terms(man: dict | None) -> tuple[str, ...]:
    """
    The static vocabulary plus everything this particular manifest could leak:
    condition names, room-file tokens, and the reference/hypothesis sentences.

    Derived rather than hardcoded so that regenerating the demo set with
    different rooms or different clips does not quietly widen the hole.
    """
    terms = set(_STATIC_LEAK_TERMS)
    if not man:
        return tuple(sorted(terms, key=len, reverse=True))
    for cond in (man.get("conditions") or {}).values():
        if cond.get("name"):
            terms.add(str(cond["name"]).lower())
        if cond.get("label"):
            terms.add(str(cond["label"]).lower())
        room = str(cond.get("room") or "")
        for tok in re.split(r"[^A-Za-z]+", room):
            if len(tok) >= 5 and tok.lower() not in {"txts"}:
                terms.add(tok.lower())
    for pair in man.get("pairs") or []:
        if pair.get("ref"):
            terms.add(str(pair["ref"]).lower())
        for arm in ("A", "B"):
            t = (pair.get(arm) or {}).get("transcript")
            if t:
                terms.add(str(t).lower())
    payoff = man.get("payoff") or {}
    if payoff.get("ref"):
        terms.add(str(payoff["ref"]).lower())
    # Longest first, so redaction blanks the most specific match it can.
    return tuple(sorted(terms, key=len, reverse=True))


def _leak_regexes(terms: tuple[str, ...] | None) -> list[str]:
    src = terms if terms is not None else _STATIC_LEAK_TERMS
    return [_bounded(t) for t in src if t] + list(_STATIC_LEAK_PATTERNS)


def find_leaks(text: str, terms: tuple[str, ...] | None = None) -> list[str]:
    """Every banned term or pattern present in `text`. Empty list == safe."""
    low = text.lower()
    hits = []
    for pat in _leak_regexes(terms):
        for m in re.finditer(pat, low):
            hits.append(m.group(0))
    return sorted(set(hits))


def redact(text: str, terms: tuple[str, ...] | None = None) -> str:
    """Blank out anything that would tell the listener the answer."""
    out = text
    for pat in _leak_regexes(terms):
        out = re.sub(pat, lambda m: "▒" * len(m.group(0)), out, flags=re.IGNORECASE)
    return out


class Blind:
    """
    The print channel for everything the listener may see before they commit.

    A leak is redacted and reported to STDERR rather than raised: crashing in
    front of an audience is a worse outcome than a blanked word, and stderr is
    where the presenter finds out afterwards that there is a bug to fix.
    """

    def __init__(self, terms: tuple[str, ...]):
        self.terms = terms
        self.leaks: list[str] = []

    def __call__(self, line: str = "") -> None:
        hits = find_leaks(line, self.terms)
        if hits:
            self.leaks.extend(hits)
            print(redact(line, self.terms))
            print(f"demo_listen: BUG -- pre-commit line named {hits!r}; redacted.",
                  file=sys.stderr)
            return
        print(line)


# ==========================================================================
# data
# ==========================================================================

def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text())


def ordered_pairs(man: dict, n: int | None = None) -> list[dict]:
    """
    The pairs, in the order the recorded listener session says to play them.

    `play_order` is DERIVED in the manifest from which pairs drew a confident
    call; it is deliberately not hardcoded there and must not be hardcoded here
    either (manifest: `play_order_derivation`). Falls back to construction order
    only if the field is missing.
    """
    by_num = {int(p["pair"]): p for p in man.get("pairs") or []}
    order = [int(x) for x in (man.get("play_order") or sorted(by_num))]
    out = [by_num[i] for i in order if i in by_num]
    return out if n is None else out[:max(1, n)]


def arms_in_play_order(pair: dict) -> list[tuple[str, dict]]:
    """
    The two clips of a pair, ordered by BLIND NAME.

    That is the order `blind/BLIND_SHEET.md` lists them in, so the terminal and
    the paper sheet agree. It also means the within-pair order is a property of
    the frozen blind naming rather than of which arm is which -- one pair happens
    to play the reverberant arm first and two play it second. That is NOT
    counterbalancing, it is just not-systematic, and the honest-label section
    says so out loud.
    """
    return sorted((("A", pair["A"]), ("B", pair["B"])), key=lambda kv: kv[1]["blind"])


def blind_path(blind_name: str) -> Path:
    return BLIND_DIR / blind_name


def measured_from_master(man: dict, path: Path = MASTER) -> dict | None:
    """
    Recompute the two arms' per-clip WERs from the grid table.

    The manifest already carries the paired difference and its bootstrap CI; what
    it does NOT carry is how many individual clips land on exactly the same
    number, which is the most concrete form of the claim ("18 of 40 score exactly
    equal"). Recomputing the means here also gives a free provenance check: if the
    manifest and the table have drifted apart, the run says so instead of reciting
    a stale interval next to fresh audio.
    """
    p = Path(path)
    if not p.is_file():
        return None
    conds = man.get("conditions") or {}
    name_a = (conds.get("A") or {}).get("name")
    name_b = (conds.get("B") or {}).get("name")
    if not name_a or not name_b:
        return None

    csv.field_size_limit(10 ** 9)
    wa: dict[str, float] = {}
    wb: dict[str, float] = {}
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("model") != MODEL:
                continue
            cn = row.get("condition_name")
            if cn not in (name_a, name_b):
                continue
            if str(row.get("failed", "")).strip().lower() == "true":
                continue
            try:
                w = float(row["wer"])
            except (KeyError, TypeError, ValueError):
                continue
            (wa if cn == name_a else wb)[row["clip_id"]] = w

    clips = sorted(set(wa) & set(wb))
    if not clips:
        return None
    a = [wa[c] for c in clips]
    b = [wb[c] for c in clips]
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    ties = [c for c in clips if wa[c] == wb[c]]
    return {
        "n_clips": len(clips),
        "mean_wer_A": mean_a,
        "mean_wer_B": mean_b,
        "paired_diff_A_minus_B": mean_a - mean_b,
        "n_exact_ties": len(ties),
        "exact_tie_clips": ties,
        "source": str(p),
    }


# ==========================================================================
# input — never hangs, never raises, never loops forever
# ==========================================================================

class Turn:
    """Whether we are still taking answers, and why we stopped if we are not."""

    def __init__(self, mode: str):
        self.mode = mode              # "live" | "replay"
        self.eof = False
        self.interrupted = False

    @property
    def live(self) -> bool:
        return self.mode == "live" and not self.eof and not self.interrupted


def ask(prompt: str, turn: Turn) -> str | None:
    """
    One line of input. Returns None on EOF or Ctrl-C, having recorded which.

    EOF is the `< /dev/null`, cron and pytest case; Ctrl-C is the presenter
    cutting the segment short. Neither may raise, and neither may leave the
    caller in a loop -- both flip `turn` out of live mode so every later prompt
    short-circuits without touching stdin again.
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
        turn.interrupted = True
        print()
        return None


# ==========================================================================
# the replayed listener
# ==========================================================================

def recorded_calls(man: dict) -> dict[str, dict]:
    """The one recorded listener session, keyed by clip id."""
    sessions = man.get("listener_sessions") or []
    if not sessions:
        return {}
    return dict(sessions[0].get("calls") or {})


_CONF_LABEL = {"clear": "clear", "slight": "a slight edge", "tossup": "a toss-up"}


def replay_answer(pair: dict, calls: dict) -> dict:
    """
    Convert a recorded call into the same shape a live answer produces.

    `confidence` in the manifest is "confident" | "marginal", with a separate
    `non_marginal` flag for the strongest calls. Mapped here rather than in the
    manifest so the recorded record stays verbatim.
    """
    call = calls.get(pair["clip_id"]) or {}
    arm = call.get("harder_arm")
    if call.get("confidence") == "marginal":
        conf = "tossup"
    elif call.get("non_marginal"):
        conf = "clear"
    else:
        conf = "slight"
    return {
        "harder_arm": arm,
        "answer": "same" if arm is None else "pick",
        "confidence": conf,
        "said": call.get("said"),
        "n_replays": 0,
        "seconds_to_answer": None,
        "source": "recorded 2026-08-05 listener session",
    }


# ==========================================================================
# the blind half — every line here is routed through Blind()
# ==========================================================================

def intro_lines(n_pairs: int, ink: Ink, width: int, mode: str) -> list[str]:
    """The framing. Says nothing about what is being varied."""
    L = [
        "",
        ink.bold("  A LISTENING TEST — you go first"),
        ink.dim("  " + "─" * (width - 4)),
        "",
        f"  {n_pairs} pairs of short clips. Same voice throughout; within a pair,",
        "  the same sentence twice.",
        "",
        "  For each pair, one question:  " + ink.bold("which one is harder to understand?"),
        "",
        ink.dim("  Use the volume however you like — the clips are not level-matched,"),
        ink.dim("  and loudness is not what you are being asked to judge. Replay as"),
        ink.dim("  often as you want. \"About the same\" is a real answer; take it if"),
        ink.dim("  it is true."),
        "",
    ]
    if mode == "replay":
        L += [ink.yellow("  [replay] no live listener — playing back a recorded "
                         "session's answers."), ""]
    return L


def pair_prompt_lines(index: int, total: int, ink: Ink, width: int) -> list[str]:
    """The header shown while the clips play. Deliberately contentless."""
    return [
        "",
        ink.dim("  " + "─" * (width - 4)),
        ink.bold(f"  pair {index} of {total}"),
        "",
    ]


CHOICE_MENU = (
    "    [1] clip 1 is harder     [2] clip 2 is harder     [s] about the same\n"
    "    [r] hear both again      [r1] / [r2] hear one     [q] stop here"
)

CONFIDENCE_MENU = (
    "    [c] clear                [l] a slight edge        [t] honestly a toss-up"
)


# ==========================================================================
# the open half — after they have committed, anything may be said
# ==========================================================================

def _fmt_edits(arm: dict) -> str:
    bits = []
    for key, word in (("n_sub", "substituted"), ("n_del", "deleted"), ("n_ins", "inserted")):
        n = int(arm.get(key) or 0)
        if n:
            bits.append(f"{n} {word}")
    return ", ".join(bits) or "no word-level edits"


def _arm_sentence(cond: dict) -> str:
    """Plain language first, the parameters second. Both, in one line."""
    rt60 = cond.get("rt60")
    snr = cond.get("snr_db")
    room = "a BAD room, long reverberation" if (rt60 or 0) >= 0.7 else \
           "a GOOD room, almost no reverberation"
    bg = "background almost silent" if (snr or 0) >= 15 else "speech nearly buried in babble"
    return (f"{room}, {bg}"
            f"   ({cond.get('label','')}: rt60 {rt60} s, SNR {snr} dB, "
            f"measured RT60 {float(cond.get('rt60_measured', 0)):.3f} s, "
            f"DRR {float(cond.get('drr_db', 0)):+.2f} dB)")


def reveal_lines(pair: dict, order: list[tuple[str, dict]], answer: dict,
                 man: dict, ink: Ink, width: int) -> list[str]:
    """The payoff for one pair. The two identical WERs are the point."""
    conds = man.get("conditions") or {}
    wers = [float(arm["wer"]) for _, arm in order]
    equal = abs(wers[0] - wers[1]) < 1e-12

    said = "no call recorded"
    if answer.get("answer") == "same":
        said = "you called it about the same"
    elif answer.get("harder_arm"):
        pos = 1 + [k for k, _ in order].index(answer["harder_arm"])
        said = (f"you said clip {pos} was harder"
                f" ({_CONF_LABEL.get(answer.get('confidence'), 'unrated')})")
    if answer.get("source"):
        said += ink.dim(f"   [{answer['source']}]")

    inner = width - 6
    eq = "==" if equal else "vs"
    L = [
        "",
        ink.bold("  ╔" + "═" * (width - 6) + "╗"),
        "  ║ " + pad(ink.bold(REVEAL_BANNER), inner - 2) + " ║",
        "  ╚" + "═" * (width - 6) + "╝",
        "",
        f"  {said}",
        "",
        "      " + pad(ink.dim("clip 1"), 34) + ink.dim("clip 2"),
        "      " + pad(ink.bold(ink.red(f"WER {wers[0]:.3f}")), 20)
        + pad(ink.bold(eq), 14) + ink.bold(ink.red(f"WER {wers[1]:.3f}")),
        "",
    ]
    if equal:
        L += [ink.bold(ink.red("      identical.  not close — EQUAL.")), ""]
    else:
        L += [ink.yellow("      (these two are not equal — check the manifest)"), ""]

    L += [
        "  what was actually said:",
        ink.cyan(f'      "{pair["ref"]}"'),
        "",
    ]
    for i, (key, arm) in enumerate(order, start=1):
        cond = conds.get(key) or {}
        heard = arm.get("transcript") or ""
        L += [
            f"  clip {i}  ·  " + ink.yellow(_arm_sentence(cond)),
            ink.dim(f"      {cond.get('name','')}"),
            '      heard:  ' + (ink.mag(f'"{heard}"') if heard else ink.red("(nothing)")),
            ink.dim(f"      {_fmt_edits(arm)} of {arm.get('n_ref','?')} reference words"
                    f"   ·   mean word confidence {float(arm.get('mean_conf', 0)):.3f}"),
            "",
        ]
    L += [
        "  " + ink.bold("Two different degradations. Two different broken words."),
        "  " + ink.bold("One number, the same for both."),
        "",
        ink.dim("  (presenter note: three clips — do not read an edit-type signature"),
        ink.dim("   off them. At grid level rt60 >= 0.7 drives DELETIONS, which runs"),
        ink.dim("   opposite to what these three happen to show.)"),
    ]
    return L


def measured_lines(man: dict, live: dict | None, ink: Ink, width: int) -> list[str]:
    pr = man.get("paired_result") or {}
    conds = man.get("conditions") or {}
    diff = float(pr.get("paired_diff_A_minus_B", float("nan")))
    lo = float(pr.get("ci_lo", float("nan")))
    hi = float(pr.get("ci_hi", float("nan")))
    n_clips = int(pr.get("n_clips", 0))
    mean_a = float(pr.get("mean_wer_A", float("nan")))
    mean_b = float(pr.get("mean_wer_B", float("nan")))

    L = [
        "",
        ink.dim("  " + "─" * (width - 4)),
        ink.bold("  THE MEASURED HALF"),
        "",
        f"  Across all {n_clips} clips, not just the ones you heard:",
        "",
        f"      {conds.get('A',{}).get('label','A'):<22} mean WER  {mean_a:.4f}",
        f"      {conds.get('B',{}).get('label','B'):<22} mean WER  {mean_b:.4f}",
        "",
        "      paired difference   " + ink.bold(ink.red(f"{diff:+.4f}")) +
        "   95% CI " + ink.bold(ink.red(f"[{lo:+.4f}, {hi:+.4f}]")) +
        "   " + ink.bold("← spans zero"),
        ink.dim(f"      {int(pr.get('n_resamples', 0)):,}-resample paired bootstrap, "
                f"resampled over {pr.get('resample_unit', 'clip')}s, seed "
                f"{pr.get('seed', 0)}"),
        "",
    ]
    if live:
        L += [
            "      " + ink.bold(f"{live['n_exact_ties']} of the {live['n_clips']} clips "
                                f"score EXACTLY equal.") +
            ink.dim("   (recomputed from results/master.csv on this run)"),
        ]
        drift = abs(live["paired_diff_A_minus_B"] - diff)
        if drift > 1e-6:
            L += [ink.yellow(f"      ⚠ manifest and master.csv disagree on the paired "
                             f"difference by {drift:.6f} — regenerate the demo set.")]
    else:
        L += [ink.dim("      (results/master.csv not present — the tie count is the "
                      "one number this run cannot recompute)")]
    L += [
        "",
        "  " + ink.bold("You had a preference. The model does not."),
        "  Whichever clip you picked, it is a ranking the model does not share.",
    ]
    return L


def prediction_lines(man: dict, ink: Ink, width: int) -> list[str]:
    """
    The sealed prediction, and the fact that it lost.

    Delivered whichever way the listener went, because the direction is not the
    claim. The temptation on stage is to reach for DRR the moment someone finds
    the reverberant clip harder -- that is post-hoc, n=1 cannot adjudicate it,
    and it is exactly the move the failed prediction should make you distrust.
    """
    sess = (man.get("listener_sessions") or [{}])[0]
    calls = sess.get("calls") or {}
    n_opposite = sum(1 for c in calls.values()
                     if c.get("harder_arm") and c["harder_arm"] != sess.get("predicted_harder_arm"))
    return [
        "",
        ink.dim("  " + "─" * (width - 4)),
        ink.bold("  THE PREDICTION I GOT WRONG"),
        "",
        "  Before anyone listened, I sealed a prediction: that the buried-in-babble",
        "  clip would be the harder one. The listener I ran this on",
        f"  " + ink.bold(f"went the OTHER way in {n_opposite} of {len(calls)} pairs."),
        "",
        ink.dim("      results/audio/demo/PREREGISTERED_PREDICTION.md — OUTCOME"),
        "",
        "  My own rubric had two outcomes: 'they rank them unequal' and 'they rank",
        "  them equal.' It never considered " + ink.bold("'unequal, but backwards'") + " — so the miss",
        "  scored as a pass under a rubric that could not fail. That is the failure",
        "  mode this whole project is about, committed against my own demo.",
        "",
        "  " + ink.bold("What survived is the half that never needed a direction:"),
        "  a confident human preference in 3 pairs out of 3, and a model with none.",
        "",
        ink.dim("  (presenter note: do NOT repair this on stage with the DRR number."),
        ink.dim("   It fits the result now in view and it is post-hoc. The model-side"),
        ink.dim("   DRR result — spearman(RT60, WER) +0.800 vs spearman(DRR, WER)"),
        ink.dim("   −1.000 across the grid — is measured and stands on its own.)"),
    ]


def honesty_lines(man: dict, n_answered: int, ink: Ink, width: int) -> list[str]:
    sess = (man.get("listener_sessions") or [{}])[0]
    pr = man.get("paired_result") or {}
    return [
        "",
        ink.dim("  " + "─" * (width - 4)),
        ink.bold(ink.yellow("  n = 1. THIS HALF IS AN INTUITION PUMP, NOT DATA.")),
        "",
        f"  That was {n_answered} judgement{'' if n_answered == 1 else 's'} from one "
        f"listener, one speaker, one accent, one",
        "  sitting. Blind to which clip was which — but NOT naive to the hypothesis;",
        "  you know what this project claims, and that moves a judgement.",
        "  The clips are not level-matched (they are byte-identical to what the model",
        "  was scored on). Presentation order is not counterbalanced — with three",
        "  pairs it cannot be. And these clips were selected " + ink.bold("because") + " the model tied",
        "  on them: defensible for a demonstration, indefensible as an estimate.",
        "",
        "  " + ink.bold("The measured half is the model-side interval") +
        f" — {float(pr.get('paired_diff_A_minus_B', 0)):+.4f} WER,",
        f"  CI [{float(pr.get('ci_lo', 0)):+.4f}, {float(pr.get('ci_hi', 0)):+.4f}], "
        f"over all {int(pr.get('n_clips', 0))} clips. That half is not selected, and",
        "  it would read the same if you had ranked them the other way round —",
        f"  which is what actually happened on {sess.get('date', 'the recorded session')} —",
        "  or refused to rank them at all.",
        "",
        ink.dim("  Doing the human side properly is a listening study: many listeners"),
        ink.dim("  naive to the hypothesis, randomized and counterbalanced order,"),
        ink.dim("  level-matched stimuli, a real intelligibility score. This project"),
        ink.dim("  does not have that, and its limitations section says so."),
    ]


def takeaway_lines(ink: Ink, width: int) -> list[str]:
    return [
        "",
        ink.dim("  " + "─" * (width - 4)),
        "",
        "  " + ink.bold(ink.red("You cannot QA a voice agent by listening to it.")),
        "",
        "  \"Sounds fine to me\" is not evidence the ASR works, and \"sounds terrible\"",
        "  is not evidence it fails. If your acceptance test is a person with",
        "  headphones, you are measuring the wrong system.",
        "",
    ]


# ==========================================================================
# the beat
# ==========================================================================

def play_pair(order: list[tuple[str, dict]], ink: Ink, audio: bool,
              only: int | None = None) -> None:
    for i, (_key, arm) in enumerate(order, start=1):
        if only is not None and i != only:
            continue
        print(ink.bold(f"    clip {i}"))
        path = blind_path(arm["blind"])
        if audio:
            play(str(path), ink, True)
        else:
            print(ink.dim(f"    (audio off — {path.name})"))


def collect_answer(order: list[tuple[str, dict]], ink: Ink, audio: bool,
                   turn: Turn) -> dict:
    """
    One live judgement. Replays on demand, accepts 'about the same', and gives up
    gracefully rather than looping if the input is not usable.
    """
    t0 = time.monotonic()
    replays = 0
    choice = None
    for _ in range(MAX_ASK_RETRIES):
        print()
        print("  " + ink.bold("Which one is harder to understand?"))
        print(ink.dim(CHOICE_MENU))
        raw = ask("  > ", turn)
        if raw is None:
            return {"answer": "abandoned", "harder_arm": None, "confidence": None,
                    "n_replays": replays,
                    "seconds_to_answer": round(time.monotonic() - t0, 2)}
        if raw in ("1", "2"):
            choice = int(raw)
            break
        if raw in ("s", "same", "equal", "3"):
            choice = 0
            break
        if raw in ("q", "quit", "stop", "skip"):
            return {"answer": "skipped", "harder_arm": None, "confidence": None,
                    "n_replays": replays,
                    "seconds_to_answer": round(time.monotonic() - t0, 2)}
        if raw in ("", "r", "rr", "again", "replay"):
            replays += 1
            play_pair(order, ink, audio)
            continue
        if raw in ("r1", "r2", "1r", "2r"):
            replays += 1
            play_pair(order, ink, audio, only=int(raw.replace("r", "")))
            continue
        print(ink.dim("    (1, 2, s for about-the-same, r to replay, q to stop)"))
    else:
        # Out of retries. Do not loop, do not force a choice.
        return {"answer": "unparsed", "harder_arm": None, "confidence": None,
                "n_replays": replays,
                "seconds_to_answer": round(time.monotonic() - t0, 2)}

    seconds = round(time.monotonic() - t0, 2)

    if choice == 0:
        return {"answer": "same", "harder_arm": None, "confidence": "same",
                "n_replays": replays, "seconds_to_answer": seconds}

    conf = None
    for _ in range(MAX_ASK_RETRIES):
        print()
        print("  " + ink.bold("How clear was that call?"))
        print(ink.dim(CONFIDENCE_MENU))
        raw = ask("  > ", turn)
        if raw is None:
            break
        if raw in ("c", "clear"):
            conf = "clear"
            break
        if raw in ("l", "slight", "s"):
            conf = "slight"
            break
        if raw in ("t", "toss", "tossup", "toss-up"):
            conf = "tossup"
            break
        if raw == "":
            break
        print(ink.dim("    (c, l or t)"))

    return {"answer": "pick", "harder_arm": order[choice - 1][0], "confidence": conf,
            "n_replays": replays, "seconds_to_answer": seconds}


def run_payoff(man: dict, ink: Ink, width: int, audio: bool, turn: Turn,
               blind: Blind) -> dict | None:
    """
    The fourth pair: a clean control against a condition the model answers with an
    empty string. A different question -- not "which is harder" but "can you still
    tell someone is speaking?" -- and it needs no ranking to land.
    """
    payoff = man.get("payoff") or {}
    clean = payoff.get("clean_blind")
    dead = payoff.get("deadzone_blind")
    if not clean or not dead:
        return None

    blind("")
    blind(ink.dim("  " + "─" * (width - 4)))
    blind(ink.bold("  one more — and it is a different question"))
    blind("")
    blind("  Two clips, same sentence. Not 'which is harder'. Just:")
    blind("  " + ink.bold("can you still tell that someone is speaking?"))
    blind("")
    print(ink.bold("    clip 1"))
    play(str(blind_path(clean)), ink, audio) if audio else print(ink.dim("    (audio off)"))
    print(ink.bold("    clip 2"))
    play(str(blind_path(dead)), ink, audio) if audio else print(ink.dim("    (audio off)"))

    ans = None
    if turn.live:
        print()
        print(ink.dim("    [y] yes, still speech    [n] no    [enter] skip"))
        raw = ask("  > ", turn)
        ans = {"y": "yes", "yes": "yes", "n": "no", "no": "no"}.get(raw or "", "skipped")
    else:
        ans = "not asked (replay)"

    n_empty = int(payoff.get("n_empty", 0))
    n_clips = int(payoff.get("n_clips", 0))
    print()
    print(ink.bold("  ╔" + "═" * (width - 6) + "╗"))
    print("  ║ " + pad(ink.bold(REVEAL_BANNER), width - 8) + " ║")
    print("  ╚" + "═" * (width - 6) + "╝")
    print()
    print(f"  reference:  " + ink.cyan(f'"{payoff.get("ref","")}"'))
    print("  the model returned:  " + ink.bold(ink.red("(an empty string)")))
    print(f"  WER {float(payoff.get('wer', 1.0)):.3f} — all {payoff.get('n_ref','?')} "
          f"reference words deleted.")
    print(ink.bold(ink.red(f"  {n_empty} of the {n_clips} clips came back completely empty "
                           f"in that condition.")))
    print()
    print(f"  Condition: {payoff.get('condition_name') or (man.get('conditions',{}).get('payoff',{}) or {}).get('name','')}")
    print(ink.dim("  Note the SNR: 20 dB. That is QUIET. The damage is reverberation"))
    print(ink.dim("  plus a low-rate codec plus full mic rolloff — it is not noise."))
    print()
    print("  " + ink.bold("And there is no low confidence to catch it with."))
    print("  Mean word confidence here is not low — it is " + ink.bold("null") +
          ", because there are")
    print("  no words to be confident about. A monitor watching confidence sees")
    print("  nothing at all.")
    return {"answer": ans, "clean": clean, "degraded": dead,
            "n_empty": n_empty, "n_clips": n_clips}


def run(man: dict, ink: Ink, width: int, *, n_pairs: int, audio: bool,
        mode: str, payoff: bool) -> dict:
    """The whole beat. Returns the session record."""
    terms = leak_terms(man)
    blind = Blind(terms)
    turn = Turn(mode)
    calls = recorded_calls(man)
    pairs = ordered_pairs(man, n_pairs)
    started = time.time()
    t0 = time.monotonic()

    for line in intro_lines(len(pairs), ink, width, mode):
        blind(line)
    if mode == "live" and not sys.stdin.isatty():
        blind(ink.dim("  (stdin is not a terminal: answers are read from it if present,"))
        blind(ink.dim("   otherwise the recorded session takes over and the beat still runs)"))

    responses = []
    for i, pair in enumerate(pairs, start=1):
        order = arms_in_play_order(pair)
        for line in pair_prompt_lines(i, len(pairs), ink, width):
            blind(line)
        play_pair(order, ink, audio)

        if turn.live:
            answer = collect_answer(order, ink, audio, turn)
            if not turn.live and answer.get("answer") == "abandoned":
                why = "Ctrl-C" if turn.interrupted else "no input on stdin"
                print(ink.yellow(f"  ({why} — switching to the recorded session so the "
                                 f"beat still resolves)"))
                answer = replay_answer(pair, calls)
        else:
            answer = replay_answer(pair, calls)
            arm = answer.get("harder_arm")
            pos = (1 + [k for k, _ in order].index(arm)) if arm else None
            said = f"clip {pos} was harder" if pos else "about the same"
            print()
            print(ink.yellow(f"  [recorded listener] {said} — "
                             f"{_CONF_LABEL.get(answer['confidence'], 'unrated')}"))
            if answer.get("said"):
                print(ink.dim(f'  their words: "{answer["said"]}"'))

        for line in reveal_lines(pair, order, answer, man, ink, width):
            print(line)

        responses.append({
            "pair": int(pair["pair"]),
            "clip_id": pair["clip_id"],
            "played_order": [arm["blind"] for _, arm in order],
            "answer": answer.get("answer"),
            "harder_arm": answer.get("harder_arm"),
            "harder_blind": (dict(order).get(answer["harder_arm"], {}) or {}).get("blind")
            if answer.get("harder_arm") else None,
            "confidence": answer.get("confidence"),
            "n_replays": answer.get("n_replays", 0),
            "seconds_to_answer": answer.get("seconds_to_answer"),
            "source": answer.get("source", "live listener"),
            "reveal": {
                "wer": {k: float(a["wer"]) for k, a in order},
                "equal": abs(float(order[0][1]["wer"]) - float(order[1][1]["wer"])) < 1e-12,
                "transcripts": {k: a.get("transcript") for k, a in order},
                "conditions": {k: (man.get("conditions", {}).get(k, {}) or {}).get("name")
                               for k, _ in order},
                "ref": pair["ref"],
            },
        })

    live = measured_from_master(man)
    for line in measured_lines(man, live, ink, width):
        print(line)
    for line in prediction_lines(man, ink, width):
        print(line)

    payoff_rec = None
    if payoff:
        payoff_rec = run_payoff(man, ink, width, audio, turn, blind)

    n_answered = sum(1 for r in responses if r["answer"] in ("pick", "same"))
    for line in honesty_lines(man, n_answered, ink, width):
        print(line)
    for line in takeaway_lines(ink, width):
        print(line)

    return {
        "schema": SESSION_SCHEMA,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "duration_s": round(time.monotonic() - t0, 2),
        "mode": mode,
        "stopped_early": bool(turn.interrupted or turn.eof),
        "stopped_reason": ("interrupt" if turn.interrupted
                           else "eof" if turn.eof else None),
        "argv": list(sys.argv[1:]),
        "model": MODEL,
        "source_manifest": str(MANIFEST),
        "play_order": [int(p["pair"]) for p in pairs],
        "responses": responses,
        "payoff": payoff_rec,
        "measured": {
            "from_manifest": man.get("paired_result"),
            "recomputed_from_master": live,
        },
        "leaks_detected": sorted(set(blind.leaks)),
        "caveats": [
            "n=1 per session: an intuition pump, not data",
            "listener blind to condition, NOT naive to the hypothesis",
            "presentation order not counterbalanced (3 pairs cannot be)",
            "clips not level-matched (byte-identical to what the model scored)",
            "pairs selected BECAUSE the model tied on them",
        ],
    }


# ==========================================================================
# session record
# ==========================================================================

def write_session(record: dict, sessions_dir: Path = SESSIONS_DIR) -> Path:
    """
    One JSON per run, timestamped, never overwriting.

    The last listener session became a real finding and had to be reconstructed
    from a chat message afterwards. This is that record, captured at the moment
    it happens instead. Nothing here touches the hand-authored documents in the
    parent directory.
    """
    d = Path(sessions_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = d / f"{stamp}.json"
    n = 1
    while path.exists():
        path = d / f"{stamp}-{n}.json"
        n += 1
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


# ==========================================================================
# preflight / sheet
# ==========================================================================

def preflight(ink: Ink) -> int:
    ok = True

    def line(good: bool, what: str) -> None:
        nonlocal ok
        ok = ok and good
        print(("  ok    " if good else "  MISS  ") + what)

    print()
    print("  demo_listen preflight — everything below is offline, no API key.")
    print()
    line(MANIFEST.is_file(), str(MANIFEST))
    line(BLIND_SHEET.is_file(), str(BLIND_SHEET))
    if MANIFEST.is_file():
        man = load_manifest()
        for name in sorted((man.get("blind_map") or {})):
            line(blind_path(name).is_file(), str(blind_path(name)))
        line(bool(man.get("play_order")), "manifest carries a derived play_order")
        line(bool(recorded_calls(man)), "manifest carries a recorded listener session "
                                        "(--replay needs it)")
    line(MASTER.is_file(), f"{MASTER} (optional: the exact-tie count)")
    line(find_player() is not None, "an audio player on PATH (afplay/aplay/play/ffplay)")
    print()
    print("  " + ("READY" if ok else "NOT READY — run `make demo-prep`"))
    print()
    return 0 if ok else 1


def print_sheet() -> int:
    if not BLIND_SHEET.is_file():
        print(f"demo_listen: {BLIND_SHEET} not found — run `make demo-prep`.",
              file=sys.stderr)
        return 1
    print(BLIND_SHEET.read_text())
    return 0


# ==========================================================================
# cli
# ==========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="A blind listening test, run live: the listener ranks the "
                    "clips, then learns the model scores them equal.")
    ap.add_argument("--replay", action="store_true",
                    help="run the whole beat with a previously recorded listener's "
                         "answers (rehearsal, or when nobody wants to participate)")
    ap.add_argument("--offline", dest="replay", action="store_true",
                    help="alias for --replay (the whole demo is offline either way)")
    ap.add_argument("--sheet", action="store_true",
                    help="print the paper sheet and exit")
    ap.add_argument("--check", action="store_true",
                    help="preflight: is every artifact on disk? then exit")
    ap.add_argument("--pairs", type=int, default=DEFAULT_N_PAIRS,
                    help=f"how many pairs to play (default {DEFAULT_N_PAIRS}; "
                         f"the third is the reserve)")
    ap.add_argument("--no-payoff", dest="payoff", action="store_false",
                    help="skip the final pair")
    ap.add_argument("--no-audio", dest="audio", action="store_false",
                    help="run the script without playing anything")
    ap.add_argument("--no-color", dest="color", action="store_false",
                    help="force plain output")
    ap.add_argument("--sessions-dir", default=str(SESSIONS_DIR),
                    help=f"where the session record is written (default {SESSIONS_DIR})")
    ap.add_argument("--write", action="store_true",
                    help="write a session record even in --replay (off by default: a "
                         "rehearsal is not a listener)")
    ap.add_argument("--no-write", dest="no_write", action="store_true",
                    help="never write a session record")
    args = ap.parse_args(argv)

    color = args.color and sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")
    ink = Ink(color)
    width = term_width()

    if args.sheet:
        return print_sheet()
    if args.check:
        return preflight(ink)

    if not MANIFEST.is_file():
        print(f"demo_listen: {MANIFEST} not found. Run `make demo-prep` first.",
              file=sys.stderr)
        return 1
    man = load_manifest()

    mode = "replay" if args.replay else "live"
    if mode == "replay" and not recorded_calls(man):
        print("demo_listen: --replay needs a recorded listener session in the "
              "manifest; none found.", file=sys.stderr)
        return 1

    try:
        record = run(man, ink, width, n_pairs=args.pairs, audio=args.audio,
                     mode=mode, payoff=args.payoff)
    except KeyboardInterrupt:
        # Belt and braces: `ask()` already swallows Ctrl-C at the prompt, so this
        # only fires if it lands between prompts. Either way the beat ends at 0.
        print()
        print("  (stopped)")
        return 0

    should_write = not args.no_write and (mode == "live" or args.write)
    if should_write:
        try:
            path = write_session(record, Path(args.sessions_dir))
            print(ink.dim(f"  session recorded: {path}"))
        except OSError as exc:                      # noqa: BLE001 -- never take the beat down
            print(ink.yellow(f"  (could not write the session record: {exc})"))
    elif mode == "replay":
        print(ink.dim("  (rehearsal — no session record written; --write to force)"))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
