#!/usr/bin/env python3
"""
test_demo_listen.py — the interactive beat cannot leak, cannot hang, cannot lie.

    ./.venv/bin/python tests/test_demo_listen.py

`demos/demo_listen.py` is the one demo that puts a human on the spot in front of
an audience. Three ways it can fail, and all three are silent:

  * IT TELLS THEM THE ANSWER FIRST. A listening test that primes the listener
    still "works" — they still rank the clips, the ranking is still recorded, and
    it is worthless. So the leak guard is asserted end to end (the whole
    pre-commit half of a real run must name no condition, room, SNR, RT60 or WER)
    AND pinned with negative controls, because a leak detector that matches
    nothing passes the first assertion trivially.

  * IT HANGS. It reads stdin in front of an audience. EOF, Ctrl-C, a non-tty and
    unparseable input must each finish the beat and exit 0 — never block, never
    traceback, never loop.

  * IT QUOTES A NUMBER NOBODY MEASURED. Every figure on screen is re-derived here
    from `results/master.csv` by a deliberately different implementation: the
    per-pair WERs, the arm means, the exact-tie count, and the bootstrap CI.

Fully offline: no audio is ever played (`--no-audio` everywhere), no stdin is
ever inherited, the API key is stripped from every child environment, and every
session record is written to a temporary directory.
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
# Makes `deadzone`, `scripts` and `demos` importable when this file is run
# directly (`python tests/test_demo_listen.py`) with no install step. Harmless
# when it is imported as a module instead.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# -------------------------------------------------------------------------

import contextlib
import copy
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from demos import demo_listen as dl

REPO = Path(_REPO_ROOT)
PY = str(Path(sys.executable))
MANIFEST = REPO / "results" / "audio" / "demo" / "manifest.json"
MASTER = REPO / "results" / "master.csv"

# The documents in results/audio/demo/ are HAND-AUTHORED and carry a record that
# derives from no artifact (SPEC: REGENERATION_HAZARD.md). This demo may read
# them and must never write them.
PROTECTED = [
    "results/audio/demo/KEY.md",
    "results/audio/demo/DEMO_SCRIPT.md",
    "results/audio/demo/PREREGISTERED_PREDICTION.md",
    "results/audio/demo/REGENERATION_HAZARD.md",
    "results/audio/demo/WHAT_TO_LISTEN_FOR.md",
    "results/audio/demo/manifest.json",
    "results/audio/demo/blind/BLIND_SHEET.md",
]


def manifest() -> dict:
    if not MANIFEST.is_file():
        raise unittest.SkipTest(f"{MANIFEST} missing — run `make demo-prep`")
    return json.loads(MANIFEST.read_text())


def run_listen(*args: str, stdin: str | None = None,
               timeout: int = 120) -> subprocess.CompletedProcess:
    """
    demo_listen.py in a child process, with the API key REMOVED and audio OFF.

    `stdin=None` means /dev/null — the EOF case — so no test can ever accidentally
    inherit a terminal and block the suite.
    """
    env = dict(os.environ)
    env.pop("DEEPGRAM_API_KEY", None)
    env["TERM"] = "dumb"                       # force the plain-text branch
    return subprocess.run(
        [PY, "demos/demo_listen.py", "--no-audio", *args],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout,
        input=stdin if stdin is not None else "",
    )


def pair_wers_from_master() -> dict[str, dict[str, float]]:
    """
    Per-clip WER for the two arms, read straight off the grid table.

    Deliberately a different implementation from `demo_listen.measured_from_master`
    (dict-of-dicts keyed by arm letter, no shared helper) so that agreement is
    evidence rather than a tautology.
    """
    man = manifest()
    names = {k: man["conditions"][k]["name"] for k in ("A", "B")}
    csv.field_size_limit(10 ** 9)
    out: dict[str, dict[str, float]] = {"A": {}, "B": {}}
    with open(MASTER, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] != dl.MODEL or row.get("failed", "").lower() == "true":
                continue
            for arm, name in names.items():
                if row["condition_name"] == name:
                    out[arm][row["clip_id"]] = float(row["wer"])
    return out


# ==========================================================================

class NoLeakGuard(unittest.TestCase):
    """
    The guarantee the whole segment rests on, plus the controls that make the
    guarantee mean something.
    """

    @classmethod
    def setUpClass(cls):
        cls.man = manifest()
        cls.terms = dl.leak_terms(cls.man)
        cls.r = run_listen("--replay")

    def test_the_blind_half_of_a_real_run_names_nothing(self):
        self.assertEqual(self.r.returncode, 0, self.r.stderr[-3000:])
        self.assertIn(dl.REVEAL_BANNER, self.r.stdout)
        pre = self.r.stdout[:self.r.stdout.index(dl.REVEAL_BANNER)]
        self.assertGreater(len(pre), 200, "the blind half is suspiciously short — "
                                          "did the reveal move to the top?")
        self.assertEqual(dl.find_leaks(pre, self.terms), [],
                         "the listener was told the answer before they committed")

    def test_the_reveal_DOES_name_the_conditions(self):
        """
        Anti-vacuity. If the terms never appear anywhere in the run, the assertion
        above passes for the wrong reason: the demo would be leak-free because it
        says nothing at all.
        """
        post = self.r.stdout[self.r.stdout.index(dl.REVEAL_BANNER):]
        self.assertNotEqual(dl.find_leaks(post, self.terms), [],
                            "the reveal names no condition — the beat has no payoff")
        for name in (self.man["conditions"]["A"]["name"],
                     self.man["conditions"]["B"]["name"]):
            self.assertIn(name, post)

    def test_negative_control_planted_leaks_are_caught_one_by_one(self):
        plants = [
            self.man["conditions"]["A"]["name"],       # the whole condition name
            "this one is the reverberant clip",        # the mechanism, in prose
            "recorded in the Shower",                  # the room
            "babble at 0 dB SNR",                      # the parameters
            "it scores WER 0.222",                     # the model's answer
            "clip u21 again",                          # the manifest clip id
            self.man["pairs"][0]["ref"],               # the reference sentence
        ]
        for p in plants:
            with self.subTest(plant=p):
                self.assertNotEqual(dl.find_leaks(p, self.terms), [],
                                    f"the guard did not catch {p!r}")

    def test_negative_control_the_print_channel_redacts_a_leak(self):
        blind = dl.Blind(self.terms)
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            blind("  clip 2 is the one with reverb in it")
        got = buf.getvalue()
        self.assertNotIn("reverb", got.lower(), "the leak reached stdout")
        self.assertIn("▒", got, "the line was dropped rather than redacted")
        self.assertIn("BUG", err.getvalue(), "a leak must be reported to stderr")
        self.assertIn("reverb", blind.leaks, "the leak was not recorded in the session")

    def test_a_clean_line_passes_through_untouched(self):
        """The other half of the control: the guard must not eat normal prose."""
        blind = dl.Blind(self.terms)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            blind("  Which one is harder to understand?")
        self.assertEqual(buf.getvalue().strip(), "Which one is harder to understand?")
        self.assertEqual(blind.leaks, [])

    def test_wer_does_not_fire_inside_the_word_answer(self):
        """
        A REAL false positive, caught on the first run: substring matching flagged
        `wer` inside "answer" and redacted the prompt asking the listener for one.
        Both directions are pinned — the bare token must still fire.
        """
        self.assertEqual(dl.find_leaks('"About the same" is a real answer', self.terms), [])
        self.assertIn("wer", dl.find_leaks("its WER is high", self.terms))

    def test_underscore_separated_fragments_are_caught(self):
        """
        `\\b` treats `_` as a word character, so the obvious boundary regex misses
        the fragments inside a condition name — i.e. it misses exactly what the
        guard exists for. The naive matcher is asserted to fail here so the fix is
        pinned to the violation rather than to some incidental property.
        """
        name = self.man["conditions"]["A"]["name"]      # rt60-1_snr-20_babble_..._roll-0
        self.assertIsNone(re.search(r"\bbabble\b", name),
                          "the naive boundary suddenly works; this control is stale")
        hits = dl.find_leaks(f"we used {name}", self.terms)
        self.assertTrue({"babble", "snr"} <= set(hits), hits)

    def test_the_vocabulary_is_derived_from_the_manifest(self):
        """A regenerated demo set with different rooms must not widen the hole."""
        man = copy.deepcopy(self.man)
        man["conditions"]["A"]["room"] = "mit_rt60-0.99_h999_Cathedral_2txts.wav"
        self.assertIn("cathedral", dl.leak_terms(man))
        self.assertNotIn("cathedral", dl.leak_terms(self.man))


class TheBeatRuns(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.man = manifest()
        cls.r = run_listen("--replay")

    def test_replay_exits_zero_and_delivers_every_movement(self):
        self.assertEqual(self.r.returncode, 0, self.r.stderr[-3000:])
        out = self.r.stdout
        for needed in ("which one is harder to understand?",
                       dl.REVEAL_BANNER.lower(),
                       "identical.  not close — equal.",
                       "the measured half",
                       "spans zero",
                       "the prediction i got wrong",
                       "intuition pump, not data",
                       "you cannot qa a voice agent by listening to it"):
            self.assertIn(needed, out.lower(), f"missing beat: {needed!r}")

    def test_replay_never_prompts(self):
        """A rehearsal that stops for input is not a rehearsal you can run alone."""
        self.assertNotIn("[1] clip 1 is harder", self.r.stdout)
        self.assertNotIn("[c] clear", self.r.stdout)

    def test_replay_reproduces_the_recorded_listeners_calls(self):
        calls = dl.recorded_calls(self.man)
        for pair in dl.ordered_pairs(self.man, dl.DEFAULT_N_PAIRS):
            arm = calls[pair["clip_id"]]["harder_arm"]
            order = [k for k, _ in dl.arms_in_play_order(pair)]
            pos = 1 + order.index(arm)
            self.assertIn(f"clip {pos} was harder", self.r.stdout,
                          f"pair {pair['pair']}: the replayed call does not match "
                          f"the recorded session")

    def test_stderr_is_silent_on_a_clean_run(self):
        self.assertEqual(self.r.stderr.strip(), "", self.r.stderr[-2000:])

    def test_sheet_prints_the_paper_sheet_and_exits_zero(self):
        r = run_listen("--sheet")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("which one is harder to understand", r.stdout.lower())

    def test_check_is_green_without_a_key_or_a_network(self):
        r = run_listen("--check")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr[-2000:])
        self.assertIn("READY", r.stdout)

    def test_the_third_pair_and_the_payoff_are_optional(self):
        two = run_listen("--replay").stdout.count(dl.REVEAL_BANNER)
        three = run_listen("--replay", "--pairs", "3").stdout.count(dl.REVEAL_BANNER)
        bare = run_listen("--replay", "--no-payoff").stdout.count(dl.REVEAL_BANNER)
        self.assertEqual(bare, dl.DEFAULT_N_PAIRS)
        self.assertEqual(two, dl.DEFAULT_N_PAIRS + 1)      # + the payoff
        self.assertEqual(three, 4)


class InputCannotHangOrCrash(unittest.TestCase):
    """It reads stdin in front of an audience. Every degenerate case exits 0."""

    def test_eof_finishes_the_beat_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_listen("--sessions-dir", d, stdin="")
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            self.assertIn("switching to the recorded session", r.stdout)
            self.assertIn("you cannot qa a voice agent", r.stdout.lower())
            rec = json.loads(sorted(Path(d).glob("*.json"))[-1].read_text())
            self.assertEqual(rec["stopped_reason"], "eof")
            self.assertTrue(rec["stopped_early"])

    def test_a_non_tty_says_so_rather_than_pretending(self):
        # --no-write, because a test must not leave a session record in the real
        # results/ tree; every other run here is either --replay or tmp-dir'd.
        r = run_listen("--no-write", stdin="")
        self.assertIn("stdin is not a terminal", r.stdout)

    def test_ctrl_c_at_the_prompt_is_not_a_traceback(self):
        script = (
            "import builtins\n"
            "from demos import demo_listen as dl\n"
            "def boom(prompt=''):\n"
            "    raise KeyboardInterrupt\n"
            "builtins.input = boom\n"
            "rec = dl.run(dl.load_manifest(), dl.Ink(False), 90, n_pairs=2,\n"
            "             audio=False, mode='live', payoff=True)\n"
            "print('REASON', rec['stopped_reason'])\n"
            "print('RESPONSES', len(rec['responses']))\n"
        )
        env = dict(os.environ)
        env.pop("DEEPGRAM_API_KEY", None)
        r = subprocess.run([PY, "-c", script], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("REASON interrupt", r.stdout)
        self.assertIn("RESPONSES 2", r.stdout,
                      "Ctrl-C must still resolve the beat, not truncate it")

    def test_an_interrupt_between_prompts_still_exits_zero(self):
        """`main` is the last line of defence if Ctrl-C lands outside `ask()`."""
        real = dl.run
        try:
            dl.run = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = dl.main(["--replay", "--no-audio"])
            self.assertEqual(rc, 0)
            self.assertIn("stopped", buf.getvalue())
        finally:
            dl.run = real

    def test_unparseable_input_gives_up_instead_of_looping(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_listen("--sessions-dir", d, stdin="zzz\n" * 40, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            rec = json.loads(sorted(Path(d).glob("*.json"))[-1].read_text())
            self.assertEqual(rec["responses"][0]["answer"], "unparsed")

    def test_scripted_answers_drive_the_flow(self):
        man = manifest()
        pairs = dl.ordered_pairs(man, dl.DEFAULT_N_PAIRS)
        with tempfile.TemporaryDirectory() as d:
            r = run_listen("--sessions-dir", d, stdin="1\nc\ns\ny\n")
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            rec = json.loads(sorted(Path(d).glob("*.json"))[-1].read_text())
            first, second = rec["responses"]

            self.assertEqual(first["answer"], "pick")
            self.assertEqual(first["confidence"], "clear")
            self.assertEqual(first["harder_arm"],
                             dl.arms_in_play_order(pairs[0])[0][0],
                             "answering '1' must select the clip played first")
            # "about the same" is a first-class answer and skips the confidence
            # question entirely — it must not be coerced into a choice.
            self.assertEqual(second["answer"], "same")
            self.assertIsNone(second["harder_arm"])
            self.assertEqual(rec["payoff"]["answer"], "yes")
            self.assertFalse(rec["stopped_early"])

    def test_replaying_a_clip_is_counted_not_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_listen("--sessions-dir", d, stdin="r\nr2\n2\nt\n\n")
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            rec = json.loads(sorted(Path(d).glob("*.json"))[-1].read_text())
            self.assertEqual(rec["responses"][0]["n_replays"], 2)
            self.assertEqual(rec["responses"][0]["confidence"], "tossup")


class SessionRecord(unittest.TestCase):

    def test_it_is_well_formed_and_self_describing(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_listen("--sessions-dir", d, stdin="1\nc\n2\nl\ny\n")
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            files = sorted(Path(d).glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertRegex(files[0].name, r"^\d{8}T\d{6}Z(-\d+)?\.json$")
            rec = json.loads(files[0].read_text())

            for key in ("schema", "started_utc", "duration_s", "mode", "model",
                        "play_order", "responses", "measured", "caveats",
                        "leaks_detected", "stopped_early"):
                self.assertIn(key, rec)
            self.assertEqual(rec["schema"], dl.SESSION_SCHEMA)
            self.assertEqual(rec["leaks_detected"], [])
            self.assertEqual(len(rec["responses"]), dl.DEFAULT_N_PAIRS)

            for resp in rec["responses"]:
                self.assertEqual(len(resp["played_order"]), 2)
                self.assertTrue(resp["reveal"]["equal"],
                                "a pair the model does not tie on has no business "
                                "in this demo")
                self.assertEqual(len(set(resp["reveal"]["wer"].values())), 1)
                self.assertIsNotNone(resp["seconds_to_answer"])

            joined = " ".join(rec["caveats"]).lower()
            self.assertIn("not data", joined)
            self.assertIn("not counterbalanced", joined)

    def test_a_rehearsal_is_not_recorded_as_a_listener(self):
        with tempfile.TemporaryDirectory() as d:
            run_listen("--replay", "--sessions-dir", d)
            self.assertEqual(list(Path(d).glob("*.json")), [],
                             "--replay wrote a session record; a rehearsal is not "
                             "a listener")
            r = run_listen("--replay", "--write", "--sessions-dir", d)
            self.assertEqual(r.returncode, 0)
            self.assertEqual(len(list(Path(d).glob("*.json"))), 1)

    def test_no_write_suppresses_it_entirely(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_listen("--no-write", "--sessions-dir", d, stdin="s\ns\n\n")
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            self.assertEqual(list(Path(d).glob("*.json")), [])

    def test_two_runs_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as d:
            rec = {"schema": dl.SESSION_SCHEMA}
            a = dl.write_session(rec, Path(d))
            b = dl.write_session(rec, Path(d))
            self.assertNotEqual(a, b)
            self.assertEqual(len(list(Path(d).glob("*.json"))), 2)

    def test_the_default_location_is_a_new_directory_not_an_authored_file(self):
        self.assertEqual(dl.SESSIONS_DIR, Path("results/audio/demo/sessions"))
        self.assertNotIn(str(dl.SESSIONS_DIR) + ".md", PROTECTED)
        for p in PROTECTED:
            self.assertFalse(str(dl.SESSIONS_DIR) in p and p.endswith(".md"))


class NumbersAreMeasuredNotTyped(unittest.TestCase):
    """Everything on screen is re-derived here from the grid table."""

    @classmethod
    def setUpClass(cls):
        if not MASTER.is_file():
            raise unittest.SkipTest(f"{MASTER} missing")
        cls.man = manifest()
        cls.w = pair_wers_from_master()
        cls.r = run_listen("--replay", "--pairs", "3")

    def test_every_demo_pair_is_an_exact_tie_in_master_csv(self):
        for pair in self.man["pairs"]:
            cid = pair["clip_id"]
            with self.subTest(clip=cid):
                self.assertEqual(self.w["A"][cid], self.w["B"][cid],
                                 "the premise of the whole segment is that the "
                                 "model ties on these clips")
                self.assertAlmostEqual(self.w["A"][cid], float(pair["A"]["wer"]), 12)
                self.assertAlmostEqual(self.w["B"][cid], float(pair["B"]["wer"]), 12)
                self.assertIn(f"WER {self.w['A'][cid]:.3f}", self.r.stdout)

    def test_the_arm_means_and_tie_count_match_an_independent_recompute(self):
        clips = sorted(set(self.w["A"]) & set(self.w["B"]))
        mean_a = sum(self.w["A"][c] for c in clips) / len(clips)
        mean_b = sum(self.w["B"][c] for c in clips) / len(clips)
        ties = sum(1 for c in clips if self.w["A"][c] == self.w["B"][c])

        got = dl.measured_from_master(self.man)
        self.assertEqual(got["n_clips"], len(clips))
        self.assertAlmostEqual(got["mean_wer_A"], mean_a, 12)
        self.assertAlmostEqual(got["mean_wer_B"], mean_b, 12)
        self.assertEqual(got["n_exact_ties"], ties)
        self.assertIn(f"{ties} of the {len(clips)} clips score EXACTLY equal",
                      self.r.stdout)

    def test_the_manifest_interval_still_reproduces_from_the_table(self):
        """
        The CI is quoted, not recomputed at run time (a bootstrap on stage is a
        stage risk). So it is re-derived HERE, and this test is what makes quoting
        it honest: if the grid moves, the demo stops matching the table.
        """
        pr = self.man["paired_result"]
        clips = sorted(set(self.w["A"]) & set(self.w["B"]))
        d = np.array([self.w["A"][c] - self.w["B"][c] for c in clips])
        self.assertAlmostEqual(float(d.mean()), pr["paired_diff_A_minus_B"], 9)

        rng = np.random.default_rng(int(pr["seed"]))
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean()
                          for _ in range(int(pr["n_resamples"]))])
        self.assertAlmostEqual(float(np.percentile(boots, 2.5)), pr["ci_lo"], 6)
        self.assertAlmostEqual(float(np.percentile(boots, 97.5)), pr["ci_hi"], 6)
        self.assertLess(pr["ci_lo"], 0.0)
        self.assertGreater(pr["ci_hi"], 0.0, "the interval must span zero — that "
                                             "IS the finding")
        self.assertIn(f"{pr['ci_lo']:+.4f}", self.r.stdout)
        self.assertIn(f"{pr['ci_hi']:+.4f}", self.r.stdout)

    def test_a_drifted_manifest_is_reported_not_recited(self):
        man = copy.deepcopy(self.man)
        man["paired_result"]["paired_diff_A_minus_B"] += 0.05
        live = dl.measured_from_master(man)
        text = "\n".join(dl.measured_lines(man, live, dl.Ink(False), 90))
        self.assertIn("disagree on the paired difference", text)

    def test_the_missing_table_is_reported_not_back_filled(self):
        text = "\n".join(dl.measured_lines(self.man, None, dl.Ink(False), 90))
        self.assertIn("cannot recompute", text)
        self.assertNotRegex(text, r"\d+ of the \d+ clips score EXACTLY")

    def test_play_order_comes_from_the_manifest_not_from_this_file(self):
        man = copy.deepcopy(self.man)
        man["play_order"] = [1, 3, 2]
        self.assertEqual([p["pair"] for p in dl.ordered_pairs(man)], [1, 3, 2])
        man.pop("play_order")
        self.assertEqual([p["pair"] for p in dl.ordered_pairs(man)], [1, 2, 3])

    def test_within_pair_order_matches_the_paper_sheet(self):
        """
        The terminal and `blind/BLIND_SHEET.md` must list the two clips in the same
        order, or a listener following on paper is answering about the other clip.
        """
        sheet = (REPO / "results/audio/demo/blind/BLIND_SHEET.md").read_text()
        for pair in self.man["pairs"]:
            names = [a["blind"] for _, a in dl.arms_in_play_order(pair)]
            with self.subTest(pair=pair["pair"]):
                self.assertIn(f"`{names[0]}` and `{names[1]}`", sheet)


class HonestFraming(unittest.TestCase):
    """
    The claim is the DISAGREEMENT, never the direction. A script that told the
    listener which way to hear it would be manufacturing the result.
    """

    @classmethod
    def setUpClass(cls):
        cls.man = manifest()
        cls.out = run_listen("--replay").stdout

    def test_the_failed_prediction_is_reported_not_buried(self):
        low = self.out.lower()
        self.assertIn("the prediction i got wrong", low)
        self.assertIn("went the other way", low)
        self.assertIn("preregistered_prediction.md", low)

    def test_the_miss_count_is_derived_from_the_recorded_session(self):
        man = copy.deepcopy(self.man)
        sess = man["listener_sessions"][0]
        for call in sess["calls"].values():
            call["harder_arm"] = sess["predicted_harder_arm"]
        text = "\n".join(dl.prediction_lines(man, dl.Ink(False), 90))
        self.assertIn("way in 0 of 3 pairs", text,
                      "the miss count is hardcoded rather than read")

    def test_it_never_tells_the_listener_which_way_to_hear_it(self):
        low = self.out.lower()
        for banned in ("you should have", "you should hear", "as expected",
                       "correctly identified", "the right answer",
                       "obviously harder"):
            self.assertNotIn(banned, low, f"the script steers the listener: {banned!r}")

    def test_the_human_half_is_labelled_n_equals_one(self):
        low = self.out.lower()
        self.assertIn("n = 1", low)
        self.assertIn("intuition pump, not data", low)
        for caveat in ("not naive to the hypothesis", "not level-matched",
                       "not counterbalanced", "selected"):
            self.assertIn(caveat, low)

    def test_the_measured_half_is_labelled_separately(self):
        self.assertIn("THE MEASURED HALF", self.out)
        self.assertIn("the measured half is the model-side interval",
                      self.out.lower())

    def test_the_drr_story_is_flagged_as_post_hoc_for_the_human_side(self):
        low = self.out.lower()
        self.assertIn("post-hoc", low)
        self.assertIn("do not repair", low)


class Hygiene(unittest.TestCase):

    def test_it_never_opens_a_socket(self):
        script = (
            "import socket\n"
            "class Boom(Exception): pass\n"
            "def die(*a, **k): raise Boom('demo_listen tried to open a socket')\n"
            "socket.socket = die\n"
            "socket.create_connection = die\n"
            "from demos import demo_listen as dl\n"
            "dl.run(dl.load_manifest(), dl.Ink(False), 90, n_pairs=2, audio=False,\n"
            "       mode='replay', payoff=True)\n"
            "print('NO_SOCKET_OK')\n"
        )
        env = dict(os.environ)
        env.pop("DEEPGRAM_API_KEY", None)
        r = subprocess.run([PY, "-c", script], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("NO_SOCKET_OK", r.stdout)

    def test_the_hand_authored_documents_are_not_touched(self):
        def digest():
            return {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()
                    for p in PROTECTED if (REPO / p).is_file()}
        before = digest()
        self.assertTrue(before, "no protected documents found — check the paths")
        with tempfile.TemporaryDirectory() as d:
            run_listen("--sessions-dir", d, stdin="1\nc\n2\nc\ny\n")
        self.assertEqual(before, digest(),
                         "the demo rewrote a hand-authored document")

    def test_it_reuses_the_kits_audio_player(self):
        """One playback path for the whole kit; it is the one that survives a
        missing file, a missing player and a hung player."""
        src = (REPO / "demos" / "demo_listen.py").read_text()
        self.assertIn("from demos.demo_break import", src)
        self.assertIn("play", src)

    def test_it_only_ever_plays_from_the_blind_directory(self):
        """The working filenames in the parent directory say `reverb` and
        `babble`; playing one would print the answer."""
        self.assertEqual(dl.blind_path("blind_01.wav"),
                         Path("results/audio/demo/blind/blind_01.wav"))
        src = (REPO / "demos" / "demo_listen.py").read_text()
        self.assertNotIn("pair1_A", src)
        self.assertNotIn("_reverb_", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
