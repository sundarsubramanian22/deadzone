#!/usr/bin/env python3
"""
demo_live.py — the same finding as demo_break, but with the network on.

One clip, two conditions, **two real Deepgram calls made in front of you**:

    1. the RAW recording                       -> right, and confident
    2. a MEASURED dead zone from the real grid -> WRONG, and STILL confident

`demo_break.py` makes this point from cached measurements, deliberately: the
offline spine is what survives a conference wifi outage. This file is the
optional flourish that makes the same point with nothing cached at all, because
watching the payload come back off the wire is a different kind of evidence
than reading it out of a CSV.

    ./.venv/bin/python demos/demo_live.py            # LIVE (default). Needs key + network.
    ./.venv/bin/python demos/demo_live.py --offline  # identical presentation, cached. Rehearsable.
    ./.venv/bin/python demos/demo_live.py --prepare  # compose the two wavs once (no network)
    ./.venv/bin/python demos/demo_live.py --check    # preflight: can this run live right now?
    ./.venv/bin/python demos/demo_live.py --raw      # dump the per-word payload verbatim

WHY THIS IS SAFE TO PUT ON STAGE. The grid is not live-demoable — 1760 calls is
167 s of wall clock and thousands of chances to fail. *One* transcription is
~1-2 s. So the live beat is scoped to exactly the thing that is cheap: two calls,
a hard per-call deadline, and a fallback that cannot fail.

THE FAILURE CONTRACT, which matters more than the happy path:
  * no key, no network, vendor error, timeout, missing SDK -> print ONE clear
    line naming the cause, fall through to the cached presentation clearly
    labelled CACHED, and **exit 0**. `make demo` must never break because the
    wifi did.
  * every call runs under a hard deadline on a DAEMON thread, so a socket that
    never returns cannot wedge the process — not even at interpreter exit.
  * nothing this file prints can contain a credential. Every line goes through
    `Redactor`, which is not paranoia: `transcribe_deepgram` folds the
    underlying exception's `repr()` into its error string, and an SDK that ever
    put the key in a request repr would otherwise put it on the projector.

SCREEN-SHARING RULE: this file prints no environment dump, no request headers,
no key, and no prefix of one. It prints only which *variable name* was found.
If you extend it, keep that property — the presenter is mirroring their screen.

NOVA-3 ONLY. DO NOT ADD AN ELEVENLABS SCRIBE ARM HERE, and this is not a
stylistic preference. Scribe's output is non-deterministic call to call: the
same audio comes back under either of two orthographies, worth up to 0.727 WER
on identical input. Cached, that is a measurable, explainable arm. Live, in
front of an audience, it is a coin flip that forces the presenter to explain
vendor orthography unprepared and on someone else's cue. The whole point of a
scripted beat is that you choose the ground you argue on. Whisper is likewise
out: it is a local model, so there is no wire to watch.
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
# Makes `deadzone`, `scripts` and `demos` importable when this file is run
# directly (`python demos/demo_live.py`) with no install step. Harmless when it
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
import re
import sys
import textwrap
import threading
import time
from pathlib import Path

# The presentation primitives are IMPORTED, not re-implemented. demo_break is
# the sibling beat and the two must look identical on screen; a second copy of
# render_diff would drift, and the audience would see two different projects.
# demo_break's module-level imports are all stdlib, so this costs nothing.
from demos.demo_break import (
    DEFAULT_CONDITION,
    Ink,
    diff_columns,
    load_dead_zones,
    load_manifest_refs,
    pad,
    render_diff,
    rule,
    term_width,
)

MODEL = "nova-3"
FS = 16000

LIVE_DIR = Path("results/demo/live")
MASTER = Path("results/master.csv")
CLEAN_TRANSCRIPTS = Path("results/clean_transcripts.jsonl")

# Deepgram Nova-3 pre-recorded, as recorded in results/MANIFEST.json on
# 2026-08-04. Quoted with its date because vendor rates move and a stale price
# stated as current is the same class of error this project is about.
USD_PER_MINUTE = 0.0043
RATE_AS_OF = "2026-08-04"
GRID_CALLS, GRID_USD = 11086, 3.26      # the whole nova-3 spend behind this demo

# Per-call wall-clock ceiling. 8 s is ~5x the observed round trip, so it never
# fires on a working link, and two calls still land inside the 20 s budget with
# room for the fallback path to print.
DEFAULT_TIMEOUT = 8.0

# ---------------------------------------------------------------------------
# WHY THIS CLIP AND THIS CONDITION — chosen from the measured grid, not by taste
# ---------------------------------------------------------------------------
# CONDITION: the #1-ranked `dead_zone` row for nova-3 in results/dead_zones.csv.
# mean_conf 0.829 at wer_spoke 0.306, and **0 of 40 clips silent** — which is
# the load-bearing part. The previous #1 was demoted to `silence_driven` because
# its confidence was averaged over the 30 clips that produced words while its
# WER was averaged over all 40; this cell's two averages are over the same 40
# clips, so the gap needs no asterisk. Verified by the identity that settles
# that file, never by counting columns:
#     gap_spoke = mean_conf - (1 - wer_spoke)
#     0.829418  - (1 - 0.306109) = 0.135528  == the stored gap_spoke
#
# CLIP: u11, and the reason is the per-word numbers, not the utterance mean.
# A condition average does not survive contact with an audience — the clip on
# screen has to fail on its own row, and it has to fail while the model is
# confident about the words it invented. In this cell u11 measures:
#     clean     WER 0.000  mean conf 0.961   (a clean control, exactly)
#     dead zone WER 0.300  mean conf 0.849
#     "eighty" -> "three"  @ 0.826
#     "elm"    -> "l"      @ 0.926
#     "street" -> "three"  @ 0.933
# i.e. it is MORE confident on two of the three words it got wrong than it is
# on the utterance as a whole. A delivery address becomes "three eight l three"
# at 0.93 confidence, which is the entity-destruction fingerprint (D2: proper
# nouns 0.646, spelled letters 0.613) with the clearest downstream consequence.
#
# CLIPS REJECTED, AND WHY — the rejections are the argument:
#   u20  higher utterance conf (0.890) but two of its three errors are
#        insertions at 0.565 / 0.529. An ASR engineer will say "the model DID
#        flag those", and they will be right. A demo of silent failure cannot
#        lead with an error the model hedged on.
#   u38  demo_break's exemplar, and fine there — but its headline substitution
#        ("coordinates" -> "recordings") is only 0.544 confident. Same problem.
#   u34  the money clip ("eight hundred" -> "three hundred euros"): the best
#        story in the corpus, but 0.723 / 0.675 on the errors and WER 0.250.
#   u18  WER 0.500, but all four errors are DELETIONS. Deletions carry no
#        hypothesis token and therefore no confidence at all, so there is
#        nothing to put on screen. That is a real and important failure mode
#        (SPEC: the mute zone a confidence monitor is structurally blind to) —
#        it is just not THIS beat.
#   u07  u26  u31  their CLEAN transcripts are not clean. u07 says "okafar",
#        u26 "koalski", u31 splits "wifi" — the measured clean-condition floor.
#        Stage 1 is the control; a control that is already wrong destroys the
#        contrast before stage 2 gets to make it.
DEFAULT_CLIP = "u11"

# Alternates, all measured in the same cell, in case a clip is unusable on the
# day. Ordered by mean confidence on the substituted words, not by WER.
ALT_CLIPS = ["u11", "u20", "u34", "u38", "u18"]


# ---------------------------------------------------------------------------
# credentials — the value NEVER leaves this module
# ---------------------------------------------------------------------------

class Redactor:
    """
    Replace any known secret VALUE with a placeholder, in every string printed.

    This is a real guard, not a ritual. `transcribe_deepgram` builds its error
    string as `f"... {last_err!r}"`, so whatever the SDK put in the exception is
    what lands on the projector. We do not control that repr, and "the vendor
    will never include the key in an error" is precisely the sort of assumption
    this project exists to stop trusting.

    Short values are deliberately NOT redacted: a 1-2 character secret would
    match everywhere and turn the whole demo into placeholders, which is a
    denial-of-service on the presentation rather than a protection.
    """

    MIN_LEN = 8
    PLACEHOLDER = "<redacted>"
    # Anything whose NAME looks like a credential. Matching on the name means a
    # variable added tomorrow is covered without editing this list.
    NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.I)

    def __init__(self, extra: list[str] | None = None):
        vals = set()
        for name, val in os.environ.items():
            if self.NAME_RE.search(name) and val and len(val) >= self.MIN_LEN:
                vals.add(val)
        for v in extra or []:
            if v and len(v) >= self.MIN_LEN:
                vals.add(v)
        # longest first, so a secret that contains another is masked whole
        self._vals = sorted(vals, key=len, reverse=True)

    def __call__(self, s: str) -> str:
        for v in self._vals:
            if v in s:
                s = s.replace(v, self.PLACEHOLDER)
        return s


_REDACT = Redactor()


def redact(s: str) -> str:
    """Scrub known secret values. Always resolves the CURRENT redactor."""
    return _REDACT(str(s))


def say(*parts) -> None:
    """
    The ONLY print in this file. Backstop redaction, not the primary defence.

    THE BUG THIS COMMENT EXISTS FOR, found while testing this file and fixed
    here rather than papered over. Redaction started out happening only here, at
    print time. `fallback_notice` builds a long sentence around the vendor's
    error string and the presenter has to read it aloud, so it is wrapped with
    `textwrap.wrap` before printing — and wrapping inserted a newline INTO the
    middle of the key. Each wrapped line then contained only half the secret, no
    line matched the value being searched for, and the credential printed in
    full across two lines. Clean-looking output, no error, nothing to notice.

    The lesson is the one this repo keeps relearning: a guard whose failure mode
    is silence is not a guard. Redaction now happens where untrusted text
    ENTERS (LiveCallFailed, and the two catch-all handlers), so no downstream
    transform can split a secret back into view. This pass stays as a second
    line of defence.
    """
    print(redact(" ".join(str(p) for p in parts)))


def load_credentials() -> tuple[bool, str]:
    """
    Make DEEPGRAM_API_KEY available and report WHERE it came from — never what.

    Same convention as scripts/run_experiment.load_env: an exported value wins
    (a deliberate override), `.env` only fills gaps, and only variable NAMES are
    ever returned. Returns (present, human-readable provenance).
    """
    global _REDACT
    already = bool(os.environ.get("DEEPGRAM_API_KEY"))
    loaded: list[str] = []
    try:
        from scripts.run_experiment import load_env
        loaded = load_env(".env")
    except Exception:                                # noqa: BLE001 — never fatal
        loaded = []
    # rebuild AFTER loading .env, so values that only just entered the
    # environment are covered before the first line is printed
    _REDACT = Redactor()
    if not os.environ.get("DEEPGRAM_API_KEY"):
        return False, "not found (looked in the environment and in .env)"
    if already:
        return True, "DEEPGRAM_API_KEY, from the environment"
    if "DEEPGRAM_API_KEY" in loaded:
        return True, "DEEPGRAM_API_KEY, from .env"
    return True, "DEEPGRAM_API_KEY"


# ---------------------------------------------------------------------------
# the deadline — a demo must never hang
# ---------------------------------------------------------------------------

class LiveCallFailed(RuntimeError):
    """
    Anything that stops a live call, carrying a message safe to put on screen.

    The message is redacted HERE, at construction, because this is the single
    chokepoint every vendor-supplied and exception-supplied string passes
    through on its way to the terminal. Redacting only at print time is not
    enough: any transform applied in between — wrapping, padding, truncation —
    can split a secret across the boundary the matcher is looking for. Scrubbing
    at ingestion means the raw value never exists downstream at all.
    """

    def __init__(self, message: str):
        super().__init__(redact(message))


def call_with_deadline(fn, timeout: float, label: str):
    """
    Run `fn()` on a daemon thread and give up after `timeout` seconds.

    WHY A RAW DAEMON THREAD AND NOT ThreadPoolExecutor. The executor registers
    an atexit hook that JOINS its workers at interpreter shutdown. A request
    that never returns would therefore let us print "timed out", exit cleanly
    in our own code, and then hang the process on the way out — after the
    presenter has already started talking again. A daemon thread is abandoned
    at exit, which is the behaviour we actually want: the call is not
    cancellable, so the only real guarantee available is that we stop *waiting*
    on it and never let it hold the process open.
    """
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            box["ok"] = fn()
        except BaseException as exc:                 # noqa: BLE001 — reported, not raised
            box["err"] = exc

    t = threading.Thread(target=worker, name=f"deadzone-live-{label}", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise LiveCallFailed(f"{label}: no response within {timeout:g}s")
    if "err" in box:
        exc = box["err"]
        raise LiveCallFailed(f"{label}: {type(exc).__name__}: {exc}")
    return box["ok"]


# ---------------------------------------------------------------------------
# audio — composed once by the same tested DSP the grid used
# ---------------------------------------------------------------------------

def wav_paths(clip_id: str, condition_name: str) -> tuple[Path, Path]:
    return (LIVE_DIR / f"{clip_id}_clean.wav",
            LIVE_DIR / f"{clip_id}_{condition_name}.wav")


def prepare_audio(clip_id: str, condition_name: str, verbose: bool = True) -> tuple[Path, Path]:
    """
    Write the two wavs the live calls will upload. NO network, NO API key.

    `apply_condition` is seeded from the condition NAME, so the degraded file is
    bit-identical to the one that produced the measured row — which is what lets
    the live result be compared against the grid at all.
    """
    clean_p, dz_p = wav_paths(clip_id, condition_name)
    if clean_p.is_file() and dz_p.is_file():
        return clean_p, dz_p

    # lazy: the offline/cached path must not pay for numpy + librosa
    import soundfile as sf
    from deadzone.conditions import Condition, DiskAssetLibrary, apply_condition
    from scripts.run_experiment import load_clip, write_degraded_wav

    if verbose:
        say("  composing the two wavs (one-time, offline, ~15 s)...")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_clip(clip_id, target_fs=FS)
    sf.write(str(clean_p), raw, FS, subtype="PCM_16")
    assets = DiskAssetLibrary(root="data", target_fs=FS)
    degraded = apply_condition(raw, Condition.from_name(condition_name), assets, FS)
    write_degraded_wav(dz_p, degraded, FS)
    if verbose:
        say(f"  wrote {clean_p} and {dz_p}")
    return clean_p, dz_p


def audio_seconds(path: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / float(info.samplerate)
    except Exception:                                # noqa: BLE001 — cost line only
        return float("nan")


# ---------------------------------------------------------------------------
# facts — one shape, whether they came off the wire or out of the table
# ---------------------------------------------------------------------------

def score(ref: str, transcript: str, word_confidences: list) -> dict:
    """(ref, hypothesis, confidences) -> the dict every panel below renders."""
    from deadzone.audio_pipeline import classify_errors
    s = classify_errors(ref, transcript)
    c = s["counts"]
    wc = [float(x) for x in (word_confidences or [])]
    return {
        "transcript": transcript,
        "wer": float(s["wer"]),
        "mean_conf": (sum(wc) / len(wc)) if wc else float("nan"),
        "n_ref": int(s["n_ref"]),
        "n_sub": int(c.get("sub", 0)),
        "n_del": int(c.get("del", 0)),
        "n_ins": int(c.get("ins", 0)),
        "edits": [list(e) for e in s["edits"]],
        "word_confidences": wc,
    }


def transcribe_live(path: Path, timeout: float) -> dict:
    """
    One real call. Raises LiveCallFailed with a printable message on any problem.

    max_retries=1 on purpose. The grid wants three attempts with backoff so a
    flaky call cannot kill a 7000-row run; a stage demo wants ONE clean attempt
    and a fast, explained failure. Three silent retries inside a deadline the
    audience cannot see is the worst of both.
    """
    try:
        from deadzone.audio_pipeline import is_failed, transcribe_deepgram
    except Exception as exc:                         # noqa: BLE001
        raise LiveCallFailed(f"adapter unavailable: {type(exc).__name__}: {exc}") from None

    def call():
        return transcribe_deepgram(str(path), model=MODEL, max_retries=1)

    res = call_with_deadline(call, timeout, label=path.name)
    if is_failed(res):
        raise LiveCallFailed(f"{path.name}: {res.get('error', 'vendor returned a failure')}")
    if not (res.get("word_confidences") or []):
        # SPEC 12's day-one gate, restated on stage: no per-word confidence means
        # the headline signal is missing, and a beat about confidence has nothing
        # to show. Better to fall back than to narrate an empty list.
        raise LiveCallFailed(f"{path.name}: the response carried no per-word confidences")
    return res


def cached_facts(clip_id: str, condition_name: str, ref: str) -> tuple[dict, dict]:
    """
    The identical presentation, read from artifacts the grid already paid for.

    Sources are the same ones demo_break bakes from — results/master.csv for the
    degraded row and results/clean_transcripts.jsonl for the control — read
    directly rather than through a second cache file, so there is no third copy
    of these numbers to drift out of date.
    """
    csv.field_size_limit(10 ** 9)
    row = None
    with open(MASTER, newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["model"] == MODEL and r["condition_name"] == condition_name
                    and r["clip_id"] == clip_id):
                row = r
                break
    if row is None:
        raise SystemExit(f"demo_live: no {MODEL} row for {clip_id} in {condition_name} "
                         f"({MASTER}) -- has the grid been run?")

    clean_rec = {}
    if CLEAN_TRANSCRIPTS.is_file():
        for line in open(CLEAN_TRANSCRIPTS):
            rec = json.loads(line)
            if rec["id"] == clip_id:
                clean_rec = rec
                break

    clean = score(ref, clean_rec.get("transcript", ""),
                  clean_rec.get("word_confidences") or [])
    dz = score(ref, row["transcript"], json.loads(row["word_confidences"] or "[]"))
    return clean, dz


def measured_row(clip_id: str, condition_name: str) -> dict | None:
    """The grid's own numbers for this exact cell, for the reproduction check."""
    csv.field_size_limit(10 ** 9)
    with open(MASTER, newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["model"] == MODEL and r["condition_name"] == condition_name
                    and r["clip_id"] == clip_id):
                return r
    return None


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def pair_words_with_confidences(transcript: str, confs: list) -> list[tuple[str, float]] | None:
    """
    Line the payload's confidences up with its words -- or refuse to.

    The adapter contract returns a flat list of confidences, not the tokens they
    came from. Pairing them with `transcript.split()` is right whenever the
    vendor's transcript is the concatenation of its own word list (which it is
    with smart_format/punctuate/numerals all False), and WRONG the moment it is
    not. So the count is checked, and on a mismatch we return None and print the
    bare list instead. Binding a confidence to the wrong word is exactly the
    silent, plausible-looking error this project is about -- see the same guard
    in demo_break.render_diff.
    """
    words = transcript.split()
    if not confs or len(words) != len(confs):
        return None
    return list(zip(words, (float(c) for c in confs)))


def conf_bar(c: float, width: int = 10) -> str:
    """A coarse visual for a number that is being read from the back of a room."""
    if c != c:                                       # NaN
        return " " * width
    n = max(0, min(width, int(round(c * width))))
    return "█" * n + "·" * (width - n)


def show_payload(facts: dict, ink: Ink, width: int, wrong: set[str], raw: bool) -> None:
    """
    The per-word confidences, as returned. This is the 'I actually use your
    product' half of the beat, so it shows the payload rather than a summary of
    it -- one row per word, four decimal places, nothing rounded away.
    """
    confs = facts["word_confidences"]
    paired = pair_words_with_confidences(facts["transcript"], confs)
    say(ink.dim("      per-word confidence, exactly as returned:"))
    if paired is None:
        say(ink.dim(f"      ({len(confs)} confidences vs "
                    f"{len(facts['transcript'].split())} transcript tokens -- not "
                    f"pairing them; here is the raw list)"))
        say("      " + ", ".join(f"{float(c):.4f}" for c in confs))
        return

    cell = max((len(w) for w, _ in paired), default=4)
    per_row = max(1, (width - 8) // (cell + 20))
    line: list[str] = []
    for w, c in paired:
        mark = ink.red("!") if w in wrong else " "
        txt = f"{mark} {pad(ink.red(w) if w in wrong else w, cell)} {c:.4f} {ink.dim(conf_bar(c))}"
        line.append(txt)
        if len(line) == per_row:
            say("      " + "  ".join(line))
            line = []
    if line:
        say("      " + "  ".join(line))
    if raw:
        say(ink.dim("      raw word_confidences: ") + json.dumps([round(c, 6) for c in confs]))


def wrong_words(facts: dict) -> set[str]:
    """
    Hypothesis tokens the alignment says are substitutions or insertions.

    Goes through diff_columns rather than indexing the edit tuples directly, for
    the same reason demo_break does: an edit is not guaranteed to be a 3-tuple,
    and a `del` carries no hypothesis token at all.
    """
    return {h for op, _, h in diff_columns(facts["edits"])
            if op in ("sub", "ins") and h}


def show_stage(title: str, subtitle: str, facts: dict, ink: Ink, width: int,
               elapsed: float | None, danger: bool, raw: bool) -> None:
    accent = ink.red if danger else ink.green
    say("")
    say(ink.bold(title))
    if subtitle:
        say(ink.dim(subtitle))
    if elapsed is not None:
        say(ink.dim(f"      -> POST to Deepgram ... {elapsed:.2f} s round trip"))
    say("")
    say("      " + ink.dim("hyp  ") + (facts["transcript"] or ink.red("(empty transcript)")))
    say("")
    show_payload(facts, ink, width, wrong_words(facts) if danger else set(), raw)
    say("")
    conf = facts["mean_conf"]
    conf_s = "n/a" if conf != conf else f"{conf:.3f}"
    say("      " + ink.dim("WER ") + accent(ink.bold(f"{facts['wer']:.3f}"))
        + ink.dim("     mean word confidence ") + accent(ink.bold(conf_s))
        + ink.dim(f"     sub {facts['n_sub']} del {facts['n_del']} "
                  f"ins {facts['n_ins']} / {facts['n_ref']} ref"))


def show_side_by_side(clean: dict, dz: dict, ink: Ink, width: int) -> None:
    say("")
    say(ink.dim(rule(width)))
    say(ink.bold("  SIDE BY SIDE"))
    say("")
    say("    " + ink.dim(pad("", 14)) + pad(ink.dim("raw"), 14) + pad(ink.dim("dead zone"), 14)
        + ink.dim("delta"))
    d_wer = dz["wer"] - clean["wer"]
    d_conf = dz["mean_conf"] - clean["mean_conf"]
    say("    " + pad("WER", 14) + pad(f"{clean['wer']:.3f}", 14)
        + pad(ink.red(f"{dz['wer']:.3f}"), 14) + ink.red(f"{d_wer:+.3f}"))
    say("    " + pad("confidence", 14) + pad(f"{clean['mean_conf']:.3f}", 14)
        + pad(ink.yellow(f"{dz['mean_conf']:.3f}"), 14) + ink.yellow(f"{d_conf:+.3f}"))
    say("")
    say("    " + ink.dim("The transcript collapsed. The model's self-report barely moved."))
    say("")
    for line in render_diff(dz["edits"], width - 4, ink, dz["word_confidences"]):
        say("    " + line.rstrip())

    # The sharpest single number available: the confidence the model reported on
    # a word it invented. Only shown when the alignment held, never zipped blind.
    cols = diff_columns(dz["edits"])
    hyp_slots = [i for i, (op, _, _) in enumerate(cols) if op in ("match", "sub", "ins")]
    wc = dz["word_confidences"]
    if wc and len(hyp_slots) == len(wc):
        subs = [(cols[i][1], cols[i][2], wc[k])
                for k, i in enumerate(hyp_slots) if cols[i][0] == "sub"]
        if subs:
            ref_w, hyp_w, c = max(subs, key=lambda t: t[2])
            say("")
            say("    " + ink.bold("It replaced ") + ink.yellow(f'"{ref_w}"') + ink.bold(" with ")
                + ink.red(f'"{hyp_w}"') + ink.bold(" and reported ") + ink.red(f"{c:.3f}")
                + ink.bold(" confidence in it."))
            if c > dz["mean_conf"]:
                say("    " + ink.dim("That is ABOVE its own mean for this utterance "
                                     f"({dz['mean_conf']:.3f}) -- it was more sure of the "
                                     "word it"))
                say("    " + ink.dim("invented than of the ones it got right."))


def show_cost(n_calls: int, seconds: float, ink: Ink, width: int) -> None:
    minutes = seconds / 60.0
    usd = minutes * USD_PER_MINUTE
    say("")
    say(ink.dim(rule(width)))
    say(ink.bold("  WHAT THAT COST"))
    say("")
    say(f"    {n_calls} calls · {seconds:.1f} s of audio · "
        + ink.bold(f"${usd:.5f}")
        + ink.dim(f"   (nova-3 pre-recorded, ${USD_PER_MINUTE}/min as of {RATE_AS_OF})"))
    say(ink.dim(f"    For scale: the entire {GRID_CALLS:,}-call grid behind this demo "
                f"cost ~${GRID_USD:.2f}."))
    say(ink.dim("    The rate is quoted with a date because vendor rates move;"))
    say(ink.dim("    results/MANIFEST.json holds the one the grid was priced at."))


def show_reproduction(dz: dict, row: dict | None, ink: Ink, width: int, live: bool) -> None:
    """
    Live vs the grid, printed whether or not they agree.

    A commercial model literal is updated server-side, so a live call months
    after the grid is NOT the same experiment (SPEC R4.6) -- and a small
    divergence on stage is only awkward if it is a surprise. Printed every time,
    it is a prepared point about reproducibility instead.
    """
    if row is None:
        return
    say("")
    say(ink.dim(rule(width)))
    say(ink.bold("  AGAINST THE MEASURED GRID"))
    say("")
    m_wer, m_conf = float(row["wer"]), float(row["mean_conf"])
    d_wer, d_conf = dz["wer"] - m_wer, dz["mean_conf"] - m_conf
    say(f"    grid   WER {m_wer:.3f}   conf {m_conf:.3f}   "
        + ink.dim(f"(run_id {row.get('run_id', '?')}, {str(row.get('ts', ''))[:10]})"))
    say(("    live   " if live else "    cached ")
        + f"WER {dz['wer']:.3f}   conf {dz['mean_conf']:.3f}   "
        + ink.dim(f"(delta  WER {d_wer:+.3f}   conf {d_conf:+.3f})"))
    if not live:
        say(ink.dim("    Same row, so these agree by construction."))
        return
    # TOLERANCE, AND WHY IT IS NOT FUDGE. Vendor confidences move in the third
    # decimal between calls on identical audio; the grid row was written a day
    # earlier. Calling a 0.001 confidence drift "a different experiment" would
    # make the presenter explain a non-difference under time pressure, and would
    # cry wolf on the branch that exists for a REAL divergence. WER is held to
    # exact equality because it is the claim on screen; confidence gets 0.02,
    # which is an order of magnitude below the 0.105 drop being demonstrated.
    if abs(d_wer) < 1e-9 and abs(d_conf) <= 0.02:
        say(ink.dim("    Reproduced. Identical WER and confidence to within "
                    f"{abs(d_conf):.3f} — the same"))
        say(ink.dim("    audio through the same model literal, days after the grid ran."))
    else:
        say(ink.dim("    These have MOVED, and that is worth saying out loud rather than"))
        say(ink.dim("    hiding: a commercial model literal is updated server-side, so a"))
        say(ink.dim("    live call is not automatically the same experiment as the grid."))
        say(ink.dim("    That is exactly why the run is frozen in results/MANIFEST.json"))
        say(ink.dim("    with the literal and the date (SPEC R4.6)."))


# ---------------------------------------------------------------------------
# the beat
# ---------------------------------------------------------------------------

def present(clip_id: str, condition_name: str, ref: str, clean: dict, dz: dict,
            ink: Ink, width: int, live: bool, provenance: str, timings: dict,
            raw: bool, note: str | None) -> int:
    cond = None
    try:
        from deadzone.conditions import Condition
        cond = Condition.from_name(condition_name)
    except Exception:                                # noqa: BLE001 — cosmetic only
        pass

    banner = ("  DEADZONE — LIVE  ·  two real API calls, nothing cached  "
              if live else
              "  DEADZONE — CACHED REPLAY  ·  no API call was made  ")
    say("")
    say(ink.inv(pad(banner, width)))
    say("")
    if note:
        # Wrapped, because this is the line the presenter has to READ OUT when
        # the wifi has just died. A 400-character single line wrapped by the
        # terminal at an arbitrary column is not readable at speed.
        for ln in textwrap.wrap(note, width=max(40, width - 4)):
            say("  " + ink.yellow(ln))
        say("")
    say(f"  clip       {ink.bold(clip_id)}   “{ref}”")
    say(f"  model      {MODEL}   ·   Deepgram pre-recorded endpoint")
    say(f"  credential {provenance}")
    say(ink.dim(rule(width)))

    show_stage(
        "  [1]  RAW RECORDING  —  the control",
        "       No degradation at all. There is no 'clean' Condition: apply_condition\n"
        "       always applies a room and always mixes noise, so the only true null is\n"
        "       the untouched file.",
        clean, ink, width, timings.get("clean"), danger=False, raw=raw)

    sub = f"       {condition_name}"
    if cond is not None:
        sub = (f"       rt60 {cond.rt60:g}s · SNR {cond.snr_db:g} dB · {cond.noise_type} · "
               f"{cond.codec} · mic rolloff {cond.mic_rolloff:g}\n"
               f"       The #1 measured dead zone for {MODEL}: mean conf 0.829 at WER 0.306\n"
               f"       over 40 clips, with 0 of 40 silent — both averages over the same clips.")
    show_stage(f"  [2]  DEAD ZONE  —  {condition_name}", sub,
               dz, ink, width, timings.get("deadzone"), danger=True, raw=raw)

    show_side_by_side(clean, dz, ink, width)
    show_reproduction(dz, measured_row(clip_id, condition_name), ink, width, live)
    if live:
        show_cost(2, timings.get("audio_seconds", float("nan")), ink, width)
    say("")
    return 0


def run(clip_id: str, condition_name: str, ink: Ink, width: int, want_live: bool,
        timeout: float, raw: bool) -> int:
    refs = load_manifest_refs()
    if clip_id not in refs:
        raise SystemExit(f"demo_live: {clip_id!r} is not in recording_manifest.csv")
    ref = refs[clip_id]

    live_ok, note = False, None
    clean = dz = None
    timings: dict = {}
    provenance = "not read (offline replay)"

    if want_live:
        present_key, provenance = load_credentials()
        try:
            if not present_key:
                raise LiveCallFailed("DEEPGRAM_API_KEY not found in the environment or .env")
            clean_p, dz_p = prepare_audio(clip_id, condition_name)
            timings["audio_seconds"] = (audio_seconds(clean_p) + audio_seconds(dz_p))
            say("")
            say(ink.dim(f"  calling Deepgram ({MODEL}), {timeout:g}s deadline per call ..."))
            out = {}
            for stage, path in (("clean", clean_p), ("deadzone", dz_p)):
                t0 = time.monotonic()
                res = transcribe_live(path, timeout)
                timings[stage] = time.monotonic() - t0
                out[stage] = score(ref, res["transcript"], res["word_confidences"])
            clean, dz = out["clean"], out["deadzone"]
            live_ok = True
        except LiveCallFailed as exc:
            note = fallback_notice(str(exc))         # already redacted at construction
        except Exception as exc:                     # noqa: BLE001 — stage must survive
            note = fallback_notice(redact(f"{type(exc).__name__}: {exc}"))

    if not live_ok:
        clean, dz = cached_facts(clip_id, condition_name, ref)
        timings = {}
        if want_live:
            provenance = "not used — the live call did not happen (see the note above)"

    return present(clip_id, condition_name, ref, clean, dz, ink, width,
                   live_ok, provenance, timings, raw, note)


def fallback_notice(reason: str) -> str:
    """
    The one clear line. Names the cause, says what the live call WOULD have
    shown, and makes clear the beat is still going to happen from cache.
    """
    return (f"LIVE CALL SKIPPED — {reason}. "
            "Falling back to the cached measurements, which are the same rows the "
            "grid produced. What you would have seen live: the same two transcripts "
            "and the same per-word confidences, arriving off the wire instead of off "
            "disk. Nothing about the finding depends on the call.")


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def preflight(clip_id: str, condition_name: str, ink: Ink) -> int:
    """Can this run live right now — and does the cached fallback exist?"""
    ok_live = True
    ok_fallback = True

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
    say(ink.bold("  demo_live preflight"))
    say("")
    present_key, provenance = load_credentials()
    chk(present_key, f"credential: {provenance}", "export DEEPGRAM_API_KEY or put it in .env")
    clean_p, dz_p = wav_paths(clip_id, condition_name)
    chk(clean_p.is_file() and dz_p.is_file(), f"composed wavs in {LIVE_DIR}",
        "python demos/demo_live.py --prepare  (offline, one-time)")
    chk(MASTER.is_file(), str(MASTER), "run the grid (SPEC A.R4)", fatal_to="both")
    chk(CLEAN_TRANSCRIPTS.is_file(), str(CLEAN_TRANSCRIPTS), "run the clean baseline",
        fatal_to="fallback")
    try:
        names = {r["condition_name"] for r in load_dead_zones()}
    except Exception:                                # noqa: BLE001 — reported below
        names = set()
    chk(condition_name in names, f"{condition_name} is a MEASURED dead zone for {MODEL}",
        "run deadzone.analysis.confidence_gap", fatal_to="both")

    say("")
    if ok_live:
        say("  " + ink.green("READY TO RUN LIVE."))
    else:
        say("  " + ink.yellow("NOT READY FOR THE LIVE CALL — it will fall back to cache."))
    say("  " + (ink.green("The cached fallback is present, so the beat runs either way.")
                if ok_fallback else
                ink.red("The cached FALLBACK is also missing — do not schedule this beat.")))
    say("")
    # Exit 0 unless even the fallback is gone: "cannot run live" is a normal,
    # survivable state for an optional beat, and a red build for it would train
    # whoever runs `make demo-check` to ignore red builds.
    return 0 if ok_fallback else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="One clip, clean then in a measured dead zone, transcribed LIVE "
                    "through Deepgram nova-3, showing the per-word confidences as "
                    "they come back.")
    ap.add_argument("--clip", default=DEFAULT_CLIP,
                    help=f"exemplar clip id (default {DEFAULT_CLIP}; "
                         f"alternates: {', '.join(ALT_CLIPS[1:])})")
    ap.add_argument("--condition", default=DEFAULT_CONDITION,
                    help="dead-zone condition name (must be a measured dead zone)")
    ap.add_argument("--offline", "--cached", dest="offline", action="store_true",
                    help="skip the API entirely; identical presentation from cache")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"per-call deadline in seconds (default {DEFAULT_TIMEOUT:g})")
    ap.add_argument("--prepare", action="store_true",
                    help="compose the two wavs and exit (offline, no key)")
    ap.add_argument("--check", action="store_true",
                    help="preflight: can this run live right now?")
    ap.add_argument("--raw", action="store_true",
                    help="also dump the raw word_confidences list")
    ap.add_argument("--no-color", dest="color", action="store_false",
                    help="force plain output")
    args = ap.parse_args(argv)

    color = args.color and sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")
    ink = Ink(color)
    width = term_width()

    if args.check:
        return preflight(args.clip, args.condition, ink)

    if args.prepare:
        prepare_audio(args.clip, args.condition)
        say("demo_live: ready. `make demo-live` will now spend its whole budget on the calls.")
        return 0

    try:
        return run(args.clip, args.condition, ink, width,
                   want_live=not args.offline, timeout=args.timeout, raw=args.raw)
    except SystemExit:
        raise
    except Exception as exc:                         # noqa: BLE001 — LAST resort
        # Anything at all that got past the per-stage handling still exits 0.
        # This beat is optional by construction; it is never allowed to be the
        # reason `make demo` reports a failure in front of an audience.
        say("")
        say(ink.yellow(redact(f"demo_live: skipped — {type(exc).__name__}: {exc}")))
        say(ink.dim("           The offline demos are unaffected: run `make demo-break`."))
        return 0


if __name__ == "__main__":
    sys.exit(main())
