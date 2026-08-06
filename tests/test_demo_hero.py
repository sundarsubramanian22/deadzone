#!/usr/bin/env python3
"""
test_demo_hero.py — the hero demo cannot lie, cannot leak, cannot hang.

    ./.venv/bin/python tests/test_demo_hero.py

THIS SUITE MAKES NO NETWORK CALL AND NEEDS NO API KEY. `demos/demo_hero.py` is
the one demo that is *meant* to call a vendor, which makes it the one whose
tests must be most careful not to. Every live path here runs against a
substituted adapter, and one test proves structurally that the replay path opens
no socket at all.

Four things can go wrong on stage, and every one of them is silent:

  * IT STATES A CLAIM ITS OWN PAYLOAD CONTRADICTS. The demo's punchline is a
    specific sentence about the model being more confident in a word it invented
    than in the words it got right. `punchline_claim()` is a ladder, and the
    tests below construct payloads that support each rung *and* payloads that
    contradict it, asserting the sentence disappears. A demo about confidently-
    wrong output is not allowed to be confidently wrong.

  * IT SHOWS A CACHED NUMBER AS A LIVE ONE. The adapter is patched to return a
    transcript that differs from the archived row, and the printed live figures
    are asserted to be the *patched* ones — with the archived row still on
    screen, labelled, so the test cannot pass by the demo printing nothing.

  * IT GOES RED, OR HANGS, BECAUSE THE WIFI DID. No key, a vendor error, a
    timeout, an empty confidence list, EOF on stdin, a non-tty, garbage input:
    each must print one explanation, fall back to the archive, and exit 0.

  * IT PUTS SOMETHING ON THE PROJECTOR THAT BELONGS IN A DOCUMENT. The cost
    line, the per-minute vendor rate, `run_id` strings and the MANIFEST
    provenance block were deliberately moved out (see
    `report/_demo_internal_notes.md`). The scanner that checks for them is
    itself pinned with a positive control, so a scanner that matches nothing
    cannot pass.

EVERY assertion has a negative control: the violating input is constructed and
the check is shown firing on it. A guard whose failure mode is silence is not a
guard (SPEC Appendix E).
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
# Makes `deadzone`, `scripts` and `demos` importable when this file is run
# directly (`python tests/test_demo_hero.py`) with no install step. Harmless
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
import io
import json
import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

from demos import demo_hero as hero
from demos import demo_live

REPO = Path(_REPO_ROOT)
PY = str(Path(sys.executable))

CACHE = REPO / "results" / "demo" / "hero" / "hero_cache.json"
MASTER = REPO / "results" / "master.csv"
DEAD_ZONES = REPO / "results" / "dead_zones.csv"
CLEAN_TX = REPO / "results" / "clean_transcripts.jsonl"

# A value shaped like a real credential: long enough to clear Redactor.MIN_LEN
# and embedded where a vendor SDK would put it.
FAKE_KEY = "fakekey-0123456789abcdef0123456789abcdef"
# A decoy of the SAME shape that is NOT a secret. It must reach the screen
# untouched, or a redactor that simply blanked everything would pass the leak
# tests and nobody would learn the demo prints nothing useful.
DECOY = "notasecret-fedcba9876543210fedcba98765432"

csv.field_size_limit(10 ** 9)


def load_cache() -> dict:
    if not CACHE.is_file():
        raise unittest.SkipTest(f"{CACHE} missing — run `make demo-prep`")
    return json.loads(CACHE.read_text())


def flat(s: str) -> str:
    """Collapse whitespace. Terminal output is WRAPPED, so a prose assertion on
    raw text fails on line length rather than on behaviour."""
    return " ".join(s.split())


def child(*args: str, key: str | None = None, stdin: str = "",
          timeout: int = 180) -> subprocess.CompletedProcess:
    """Run demo_hero.py in a child process with an explicitly-chosen key state."""
    env = dict(os.environ)
    env.pop("DEEPGRAM_API_KEY", None)
    if key is not None:
        env["DEEPGRAM_API_KEY"] = key
    env["TERM"] = "dumb"                              # force the plain-text branch
    return subprocess.run([PY, "demos/demo_hero.py", *args], cwd=REPO, env=env,
                          input=stdin, capture_output=True, text=True,
                          timeout=timeout)


def in_process(argv: list[str], patch_transcribe=None, key: str | None = None,
               hide_env_file: bool = True) -> tuple[int, str]:
    """
    Run main() here with the adapter substituted, capturing stdout.

    The substitution targets `deadzone.audio_pipeline.transcribe_deepgram`,
    which `demo_live.transcribe_live` (which the hero reuses) imports INSIDE the
    function body — so the patch is picked up at call time. That is the same
    seam the real code uses to stay import-light, exercised rather than worked
    around.

    `hide_env_file` neutralizes the `.env` fallback. Without it, `key=None` does
    NOT mean "no key": `load_credentials` correctly reads `.env`, so on a
    machine that has one the no-key test would silently become a live-call test
    — a test that passes for the wrong reason locally and fails only in CI.
    """
    import deadzone.audio_pipeline as ap
    import scripts.run_experiment as rx
    saved_fn = ap.transcribe_deepgram
    saved_env = rx.load_env
    saved_key = os.environ.get("DEEPGRAM_API_KEY")
    saved_redact = demo_live._REDACT
    try:
        if key is None:
            os.environ.pop("DEEPGRAM_API_KEY", None)
        else:
            os.environ["DEEPGRAM_API_KEY"] = key
        if hide_env_file:
            rx.load_env = lambda path=".env": []
        if patch_transcribe is not None:
            ap.transcribe_deepgram = patch_transcribe
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = hero.main(argv)
        return rc, buf.getvalue()
    finally:
        ap.transcribe_deepgram = saved_fn
        rx.load_env = saved_env
        demo_live._REDACT = saved_redact
        if saved_key is None:
            os.environ.pop("DEEPGRAM_API_KEY", None)
        else:
            os.environ["DEEPGRAM_API_KEY"] = saved_key


def ok_response(transcript: str, confs: list[float]) -> dict:
    """The adapter contract, minimally."""
    return {"transcript": transcript, "word_confidences": list(confs),
            "utterance_conf": (sum(confs) / len(confs)) if confs else None,
            "model": "nova-3"}


# ==========================================================================
# the punchline ladder — the claim may never outrun the payload
# ==========================================================================

def aligned(*rows) -> list[tuple]:
    """(op, ref, hyp, conf) rows, written out so each test reads as a payload."""
    return [tuple(r) for r in rows]


class ThePunchlineLadder(unittest.TestCase):
    """
    The sentence is printed ONLY when the payload supports it.

    Two axes, and each gets a positive case and a negative control:
      INVENTED  — the emitted word is absent from the reference
      LOUDER    — louder than every word it got right ("the ones it got right")
                  or merely louder than their mean ("the average word")
    """

    REF = "our flight to chicago departs from gate twelve b"

    def test_exact_claim_when_the_invented_word_beats_every_correct_word(self):
        c = hero.punchline_claim(self.REF, aligned(
            ("match", "our", "our", 0.53),
            ("match", "flight", "flight", 0.89),
            ("sub", "gate", "should", 0.70),
            ("sub", "b", "depart", 0.98)))
        self.assertEqual(c["tier"], "exact")
        self.assertTrue(c["supported"])
        self.assertTrue(c["invented"], "'depart' is absent from the reference")
        self.assertIn(hero.EXACT_CLAIM, c["sentence"])

    def test_NEGATIVE_one_louder_correct_word_removes_the_exact_claim(self):
        """The same payload with a single correct word raised above the wrong
        one. Nothing else changes; the strong sentence must vanish."""
        c = hero.punchline_claim(self.REF, aligned(
            ("match", "our", "our", 0.53),
            ("match", "flight", "flight", 0.99),      # <- was 0.89
            ("sub", "gate", "should", 0.70),
            ("sub", "b", "depart", 0.98)))
        self.assertEqual(c["tier"], "average")
        self.assertNotIn(hero.EXACT_CLAIM, c["sentence"])
        self.assertIn("average word it got right", c["sentence"])

    def test_NEGATIVE_a_word_present_in_the_reference_is_not_called_invented(self):
        """'twelve' IS in the reference, so the model confused a word rather
        than inventing one. Calling that an invention is a claim a presenter
        would have to walk back the moment somebody reads the diff."""
        c = hero.punchline_claim(self.REF, aligned(
            ("match", "our", "our", 0.53),
            ("match", "flight", "flight", 0.89),
            ("sub", "b", "twelve", 0.98)))
        self.assertEqual(c["tier"], "exact")
        self.assertFalse(c["invented"])
        self.assertNotIn(hero.EXACT_CLAIM, c["sentence"])
        self.assertIn("got wrong", c["sentence"])

    def test_NEGATIVE_a_hedged_wrong_word_makes_no_claim_at_all(self):
        """The model flagged its own error. That is the model behaving
        correctly, and the demo must say so rather than print the punchline."""
        c = hero.punchline_claim(self.REF, aligned(
            ("match", "our", "our", 0.93),
            ("match", "flight", "flight", 0.95),
            ("sub", "b", "depart", 0.31)))
        self.assertEqual(c["tier"], "hedged")
        self.assertFalse(c["supported"])
        self.assertNotIn(hero.EXACT_CLAIM, c["sentence"])
        self.assertIn("hedged", c["sentence"])

    def test_NEGATIVE_no_wrong_word_means_no_punchline(self):
        c = hero.punchline_claim(self.REF, aligned(
            ("match", "our", "our", 0.93), ("match", "flight", "flight", 0.95)))
        self.assertEqual(c["tier"], "none")
        self.assertFalse(c["supported"])
        self.assertNotIn(hero.EXACT_CLAIM, c["sentence"])

    def test_NEGATIVE_no_correct_word_means_no_comparison(self):
        c = hero.punchline_claim(self.REF, aligned(("sub", "b", "depart", 0.98)))
        self.assertEqual(c["tier"], "no-correct-words")
        self.assertFalse(c["supported"])
        self.assertNotIn(hero.EXACT_CLAIM, c["sentence"])

    def test_NEGATIVE_an_unaligned_payload_makes_no_per_word_claim(self):
        c = hero.punchline_claim(self.REF, None)
        self.assertEqual(c["tier"], "unaligned")
        self.assertFalse(c["supported"])
        self.assertNotIn(hero.EXACT_CLAIM, c["sentence"])

    def test_alignment_refuses_to_pair_when_the_counts_disagree(self):
        """Binding a confidence to the wrong word is the silent, plausible
        error this whole project is about, and it would land on the one slide
        the argument rests on."""
        edits = [["match", "a", "a"], ["sub", "b", "c"]]
        self.assertIsNotNone(hero.aligned_hyp(edits, [0.9, 0.8]))
        self.assertIsNone(hero.aligned_hyp(edits, [0.9]))         # too few
        self.assertIsNone(hero.aligned_hyp(edits, [0.9, 0.8, 0.7]))  # too many

    def test_deletions_carry_no_confidence_and_are_not_paired(self):
        """A deletion has no hypothesis token, so it must not consume one of the
        confidences — that off-by-one would shift every later word."""
        edits = [["match", "a", "a"], ["del", "b", None], ["sub", "c", "d"]]
        got = hero.aligned_hyp(edits, [0.9, 0.8])
        self.assertEqual([r[0] for r in got], ["match", "sub"])
        self.assertEqual([r[3] for r in got], [0.9, 0.8])


class TheRenderedPunchlineMatchesTheLadder(unittest.TestCase):
    """The ladder is only useful if what reaches the screen obeys it."""

    REF = ThePunchlineLadder.REF

    def _render(self, rows) -> str:
        edits, confs = [], []
        for op, r, h, c in rows:
            edits.append([op, r if op != "ins" else None, h])
            confs.append(c)
        facts = {"edits": edits, "word_confidences": confs}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hero.show_punchline(self.REF, facts, hero.Ink(False), 96)
        return flat(buf.getvalue())

    def test_the_supported_sentence_is_printed(self):
        out = self._render(aligned(("match", "our", "our", 0.53),
                                   ("match", "flight", "flight", 0.89),
                                   ("sub", "b", "depart", 0.98)))
        self.assertIn(hero.EXACT_CLAIM, out)
        self.assertIn("0.980", out)

    def test_NEGATIVE_the_contradicted_sentence_is_NOT_printed(self):
        """Positive control above proves the string can appear at all, so this
        is not passing because the renderer prints nothing."""
        out = self._render(aligned(("match", "our", "our", 0.53),
                                   ("match", "flight", "flight", 0.99),
                                   ("sub", "b", "depart", 0.98)))
        self.assertNotIn(hero.EXACT_CLAIM, out)
        self.assertIn("average word it got right", out)


class TheCollapseNarrationIsDerived(unittest.TestCase):
    """
    SPEC J.7: the demo script narrated a verdict the panel contradicted, because
    the sentence was a literal written for an exemplar that had moved. This
    caption is computed from the two rows above it, so it cannot.
    """

    def test_a_small_confidence_drop_reads_as_a_small_drop(self):
        lines = hero.collapse_narration({"wer": 0.0, "mean_conf": 0.98},
                                        {"wer": 0.80, "mean_conf": 0.90})
        self.assertIn("a fraction as far", lines[0])

    def test_NEGATIVE_a_large_confidence_drop_does_not(self):
        """The literal this replaces said 'barely moved' unconditionally."""
        lines = hero.collapse_narration({"wer": 0.0, "mean_conf": 0.98},
                                        {"wer": 0.30, "mean_conf": 0.40})
        self.assertNotIn("a fraction as far", lines[0])
        self.assertIn("nothing like as far", lines[0])

    def test_the_surplus_is_arithmetic_on_the_numbers_above_it(self):
        lines = hero.collapse_narration({"wer": 0.0, "mean_conf": 0.98},
                                        {"wer": 0.778, "mean_conf": 0.686})
        joined = " ".join(lines)
        self.assertIn("0.222", joined)                # 1 - 0.778
        self.assertIn("+0.464", joined)               # 0.686 - 0.222

    def test_NEGATIVE_an_underconfident_run_is_not_called_overconfident(self):
        lines = hero.collapse_narration({"wer": 0.0, "mean_conf": 0.98},
                                        {"wer": 0.10, "mean_conf": 0.40})
        joined = " ".join(lines)
        self.assertIn("NOT overstating", joined)
        self.assertNotIn("If confidence tracked accuracy it would read", joined)


# ==========================================================================
# the curated set is DERIVED from artifacts, not typed
# ==========================================================================

class TheCuratedSetTracksTheGrid(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cache = load_cache()
        cls.ex = cls.cache["exemplars"]

    def test_it_is_small_and_has_no_duplicate_clips(self):
        self.assertTrue(2 <= len(self.ex) <= 8, f"{len(self.ex)} exemplars")
        ids = [e["clip_id"] for e in self.ex]
        self.assertEqual(len(ids), len(set(ids)), "a clip appears twice in the menu")

    def test_every_condition_is_a_MEASURED_dead_zone_for_this_model(self):
        """`silence_driven` and `mute_zone` rows live in the same file and are
        different findings. Demoing one under the name 'dead zone' is the exact
        error SPEC Appendix G is about."""
        rows = [r for r in csv.DictReader(open(DEAD_ZONES))
                if r["model"] == hero.MODEL and r["category"] == "dead_zone"]
        names = {r["condition_name"] for r in rows}
        self.assertTrue(names, "no dead-zone rows in the artifact")
        for e in self.ex:
            self.assertIn(e["condition_name"], names)

    def test_NEGATIVE_a_silence_driven_condition_would_be_caught(self):
        """The check above passes trivially if the name set is permissive. It is
        not: a `silence_driven` row is in the same file and is rejected."""
        rows = list(csv.DictReader(open(DEAD_ZONES)))
        others = {r["condition_name"] for r in rows
                  if r["model"] == hero.MODEL and r["category"] != "dead_zone"}
        self.assertTrue(others, "artifact has no non-dead-zone rows to test with")
        good = {r["condition_name"] for r in rows
                if r["model"] == hero.MODEL and r["category"] == "dead_zone"}
        for name in others:
            self.assertNotIn(name, good)

    def test_every_control_clip_really_is_clean(self):
        """Stage 1 is the control. A control that is already wrong destroys the
        contrast before stage 2 gets to make it — and the corpus has a measured
        clean-condition floor, so some clips genuinely do not qualify."""
        from deadzone.audio_pipeline import classify_errors
        refs = hero.load_manifest_refs()
        tx = {json.loads(l)["id"]: json.loads(l) for l in open(CLEAN_TX)}
        for e in self.ex:
            cid = e["clip_id"]
            wer = float(classify_errors(refs[cid],
                                        tx[cid].get("transcript", ""))["wer"])
            self.assertEqual(wer, 0.0, f"{cid} clean WER is {wer}, not 0")

    def test_NEGATIVE_the_corpus_contains_clips_that_would_fail_that(self):
        """Proof the filter has work to do: the measured clean floor is not
        zero, so 'every selected clip is clean' is a real constraint."""
        from deadzone.audio_pipeline import classify_errors
        refs = hero.load_manifest_refs()
        tx = {json.loads(l)["id"]: json.loads(l) for l in open(CLEAN_TX)}
        dirty = [cid for cid, rec in tx.items()
                 if cid in refs
                 and float(classify_errors(refs[cid],
                                           rec.get("transcript", ""))["wer"]) > 0]
        self.assertTrue(dirty, "no clip has a non-zero clean WER — filter is vacuous")
        for cid in dirty:
            self.assertNotIn(cid, [e["clip_id"] for e in self.ex])

    def test_the_selection_reruns_from_the_artifacts_and_agrees(self):
        """Re-derived here from the four artifacts. If the grid moves and the
        cache is stale, this fails rather than letting the menu narrate a
        finding that is no longer true."""
        picked = self._reselect()
        self.assertEqual([c["clip_id"] for c in picked],
                         [e["clip_id"] for e in self.ex])
        self.assertEqual([c["condition_name"] for c in picked],
                         [e["condition_name"] for e in self.ex])

    def test_NEGATIVE_the_selection_moves_when_the_artifact_moves(self):
        """A re-derivation that returned the same answer for any input would
        make the test above meaningless. Drop the top clip's grid row and the
        top of the menu must change."""
        base = self._reselect()
        rows = copy.deepcopy(self._master_rows())
        top = base[0]
        rows[top["condition_name"]].pop(top["clip_id"])
        moved = hero.select_exemplars(self._dz_rows(), rows, self._clean(),
                                      hero.load_manifest_refs(), self._specs())
        self.assertNotEqual([c["clip_id"] for c in moved][:1],
                            [c["clip_id"] for c in base][:1])

    def test_the_cached_grid_numbers_match_master_csv(self):
        """Every number the demo prints has to be re-readable from the artifact
        that produced it — never copied from a summary (SPEC C.7)."""
        want = {(e["clip_id"], e["condition_name"]) for e in self.ex}
        found = {}
        with open(MASTER, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["clip_id"], r["condition_name"])
                if r["model"] == hero.MODEL and key in want:
                    found[key] = r
        for e in self.ex:
            row = found[(e["clip_id"], e["condition_name"])]
            self.assertAlmostEqual(e["grid"]["wer"], float(row["wer"]), places=9)
            self.assertAlmostEqual(e["grid"]["mean_conf"], float(row["mean_conf"]),
                                   places=9)
            self.assertEqual(e["grid"]["transcript"], row["transcript"])

    def test_every_exemplar_actually_breaks_and_the_model_stayed_confident(self):
        """The two things that make a cell demoable at all."""
        for e in self.ex:
            self.assertGreaterEqual(e["grid"]["wer"], hero.MIN_DZ_WER)
            worst = e["grid_worst_wrong"][3]
            self.assertGreaterEqual(worst, hero.MIN_WRONG_CONF)
            self.assertGreater(worst, e["grid"]["mean_conf"],
                               "the model hedged on its worst word — nothing to show")

    def test_both_wav_files_exist_for_every_exemplar(self):
        for e in self.ex:
            for key in ("audio_clean", "audio_degraded"):
                self.assertTrue((REPO / e[key]).is_file(), f"{e[key]} missing")

    # -- helpers: a deliberately separate re-derivation ---------------------

    def _dz_rows(self):
        return [r for r in csv.DictReader(open(DEAD_ZONES))
                if r["model"] == hero.MODEL and r["category"] == "dead_zone"]

    def _master_rows(self):
        names = {r["condition_name"] for r in self._dz_rows()}
        out = {n: {} for n in names}
        with open(MASTER, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["model"] == hero.MODEL and r["condition_name"] in names:
                    out[r["condition_name"]][r["clip_id"]] = r
        return out

    def _clean(self):
        return hero._load_clean_facts(hero.load_manifest_refs())

    def _specs(self):
        return json.loads((REPO / "task_specs.json").read_text())

    def _reselect(self):
        return hero.select_exemplars(self._dz_rows(), self._master_rows(),
                                     self._clean(), hero.load_manifest_refs(),
                                     self._specs())


# ==========================================================================
# what must NOT reach the projector
# ==========================================================================

# Moved to report/_demo_internal_notes.md. Each pattern is checked against a
# positive control below, so a scanner that matches nothing cannot pass.
BANNED = {
    "a dollar amount": r"\$\s?\d",
    "a per-minute rate": r"/\s?min\b",
    "a run_id": r"run_id",
    "the MANIFEST provenance": r"MANIFEST",
    "a cost heading": r"WHAT THAT COST",
    "a USD label": r"\bUSD\b",
}


def banned_hits(text: str) -> list[str]:
    return [name for name, pat in BANNED.items()
            if re.search(pat, text, re.I)]


class NothingOperationalReachesTheScreen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = child("--replay", "--no-color", "--once", "--no-audio")

    def test_the_replay_run_prints_none_of_them(self):
        hits = banned_hits(self.r.stdout)
        self.assertEqual(hits, [], f"internal-only material on screen: {hits}")

    def test_NEGATIVE_the_scanner_catches_every_one_of_them(self):
        """A scanner pinned to nothing passes the assertion above trivially."""
        samples = {
            "a dollar amount": "2 calls, $0.00057",
            "a per-minute rate": "nova-3 pre-recorded, $0.0043/min as of 2026-08-04",
            "a run_id": "(run_id 7f3a21, 2026-08-04)",
            "the MANIFEST provenance": "results/MANIFEST.json holds the rate",
            "a cost heading": "  WHAT THAT COST",
            "a USD label": "total USD 3.26",
        }
        for name, text in samples.items():
            self.assertIn(name, banned_hits(text),
                          f"the scanner would not have caught {name!r}")

    def test_the_live_run_prints_none_of_them_either(self):
        """The stripped material lived on the LIVE path, so a replay-only check
        would miss it entirely."""
        e = load_cache()["exemplars"][0]
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--clip", e["clip_id"]],
            patch_transcribe=lambda *a, **k: ok_response("our flight", [0.9, 0.8]),
            key=FAKE_KEY)
        self.assertEqual(rc, 0)
        self.assertEqual(banned_hits(out), [], out[-1500:])

    def test_the_module_does_not_import_the_pricing_constants(self):
        """Structural, so the numbers cannot creep back in via a helper."""
        src = (REPO / "demos" / "demo_hero.py").read_text()
        for name in ("USD_PER_MINUTE", "RATE_AS_OF", "GRID_USD", "GRID_CALLS",
                     "show_cost"):
            self.assertNotIn(name, src, f"{name} is back in demo_hero.py")

    def test_plain_text_output_has_no_ansi(self):
        self.assertNotIn("\033[", self.r.stdout,
                         "ANSI escapes leaked into a non-tty run")


# ==========================================================================
# the live number is the result; the archive is corroboration
# ==========================================================================

class TheLiveNumberIsNeverReplacedByACachedOne(unittest.TestCase):

    def setUp(self):
        self.e = load_cache()["exemplars"][0]

    def test_the_printed_figures_are_the_ones_that_just_arrived(self):
        """The adapter returns a transcript that differs from the archive, so
        the live and archived numbers are guaranteed to disagree. If the demo
        ever showed the cached number as the live one, this is where it shows."""
        ref = self.e["ref"]
        fake_clean = ok_response(ref, [0.99] * len(ref.split()))
        fake_dz = ok_response("totally different words here",
                              [0.95, 0.94, 0.93, 0.92])
        seq = [fake_clean, fake_dz]

        def patched(*a, **k):
            return seq.pop(0) if seq else fake_dz

        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--clip", self.e["clip_id"]],
            patch_transcribe=patched, key=FAKE_KEY)
        self.assertEqual(rc, 0, out[-2000:])

        live = hero.score(ref, fake_dz["transcript"], fake_dz["word_confidences"])
        grid = self.e["grid"]
        self.assertNotAlmostEqual(live["wer"], grid["wer"], places=3,
                                  msg="fixture failed to make them differ")
        self.assertIn("totally different words here", out)
        # the live figures are on screen, labelled live
        self.assertRegex(flat(out),
                         r"live\s+WER %.3f\s+confidence %.3f"
                         % (live["wer"], live["mean_conf"]))
        # and the archive is there too, labelled, not substituted
        self.assertRegex(flat(out),
                         r"grid\s+WER %.3f\s+confidence %.3f"
                         % (grid["wer"], grid["mean_conf"]))
        self.assertIn("MOVED", out, "a real divergence must be called out")

    def test_NEGATIVE_an_agreeing_call_is_reported_as_reproduced(self):
        """Same code path, agreeing payload. Without this the assertion above
        could be satisfied by a demo that always cries divergence."""
        ref = self.e["ref"]
        grid = self.e["grid"]
        seq = [ok_response(ref, [0.99] * len(ref.split())),
               ok_response(grid["transcript"], grid["word_confidences"])]

        def patched(*a, **k):
            return seq.pop(0) if seq else seq

        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--clip", self.e["clip_id"]],
            patch_transcribe=patched, key=FAKE_KEY)
        self.assertEqual(rc, 0, out[-2000:])
        self.assertIn("Reproduced", out)
        self.assertNotIn("MOVED", out)

    def test_replay_says_so_rather_than_claiming_a_live_call(self):
        r = child("--replay", "--no-color", "--once", "--no-audio")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("archived measurements", r.stdout)
        self.assertNotIn("two live calls", r.stdout)
        self.assertIn("agree by construction", r.stdout)


# ==========================================================================
# populations are always named (SPEC Appendix G)
# ==========================================================================

class EveryAggregateNamesItsPopulation(unittest.TestCase):

    def test_the_condition_block_names_the_clip_count_and_the_silent_ones(self):
        out = child("--replay", "--no-color", "--once", "--no-audio").stdout
        f = flat(out)
        self.assertIn("AND IT IS NOT ONE UNLUCKY CLIP", f)
        self.assertRegex(f, r"Averaged over (all \d+ clips|the \d+ of \d+ clips)")

    def test_the_aggregate_wer_is_the_one_paired_with_the_confidence(self):
        """`wer_spoke`, never the all-clips WER. Subtracting a confidence
        averaged over the speaking clips from a WER averaged over all of them is
        the estimand mismatch that cost this project its first headline."""
        e = load_cache()["exemplars"][0]
        cond = e["condition"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hero.show_condition_context(e, hero.Ink(False), 96)
        out = buf.getvalue()
        self.assertIn("%.3f" % cond["wer_spoke"], out)

    def test_NEGATIVE_the_all_clips_wer_is_not_what_is_printed(self):
        """Only meaningful where the two differ, so the fixture forces it."""
        e = copy.deepcopy(load_cache()["exemplars"][0])
        e["condition"]["wer_spoke"] = 0.311
        e["condition"]["wer_all_clips"] = 0.777
        e["condition"]["n_silent"] = 3
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hero.show_condition_context(e, hero.Ink(False), 96)
        out = buf.getvalue()
        self.assertIn("0.311", out)
        self.assertNotIn("0.777", out)


# ==========================================================================
# the failure contract — it never goes red, never hangs, never leaks
# ==========================================================================

class NoNetwork(unittest.TestCase):
    """Structural proof, not a promise in a docstring."""

    def test_the_replay_path_opens_no_socket(self):
        script = (
            "import sys, os\n"
            "os.environ.pop('DEEPGRAM_API_KEY', None)\n"
            "import socket\n"
            "class Boom(Exception): pass\n"
            "def die(*a, **k): raise Boom('the demo tried to open a socket')\n"
            # connect is blocked rather than socket.socket replaced: ssl
            # subclasses socket.socket at import time, so replacing the class
            # breaks unrelated imports and would prove nothing.
            "socket.socket.connect = die\n"
            "socket.socket.connect_ex = die\n"
            "socket.create_connection = die\n"
            "from demos import demo_hero\n"
            "rc = demo_hero.main(['--replay', '--no-color', '--once', '--no-audio'])\n"
            "print('NO_SOCKET_OK', rc)\n"
        )
        r = self._run(script)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("NO_SOCKET_OK 0", r.stdout)

    def test_NEGATIVE_the_socket_guard_actually_fires(self):
        """Same guard, an explicit connection attempt. Without this the test
        above could pass because the guard was never armed."""
        script = (
            "import socket\n"
            "class Boom(Exception): pass\n"
            "def die(*a, **k): raise Boom('blocked')\n"
            "socket.socket.connect = die\n"
            "socket.create_connection = die\n"
            "try:\n"
            "    socket.create_connection(('127.0.0.1', 9))\n"
            "except Boom:\n"
            "    print('GUARD_FIRES')\n"
        )
        r = self._run(script)
        self.assertIn("GUARD_FIRES", r.stdout)

    def _run(self, script: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.pop("DEEPGRAM_API_KEY", None)
        env["TERM"] = "dumb"
        return subprocess.run([PY, "-c", script], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=240)


class GracefulDegradation(unittest.TestCase):
    """
    Each case constructs the real failure and asserts the same three things:
    exit 0, one line naming the cause, and the beat still delivered from the
    archive. `make demo` must not go red because the wifi did.
    """

    def setUp(self):
        self.e = load_cache()["exemplars"][0]

    def _assert_survived(self, rc, out, cause_fragment):
        self.assertEqual(rc, 0, f"exited {rc}\n{out[-2000:]}")
        f = flat(out)
        self.assertIn("LIVE CALL SKIPPED", f)
        self.assertIn(cause_fragment, f)
        # the beat still happened, from the archive
        self.assertIn("RAW RECORDING", out)
        self.assertIn("DEAD ZONE", out)
        self.assertIn("%.3f" % self.e["grid"]["wer"], out)

    def test_no_key_at_all(self):
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--clip", self.e["clip_id"]],
            key=None)
        self._assert_survived(rc, out, "DEEPGRAM_API_KEY not found")

    def test_a_vendor_error(self):
        def boom(*a, **k):
            raise RuntimeError("vendor said 503")
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--clip", self.e["clip_id"]],
            patch_transcribe=boom, key=FAKE_KEY)
        self._assert_survived(rc, out, "vendor said 503")

    def test_a_failure_sentinel_from_the_adapter(self):
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--clip", self.e["clip_id"]],
            patch_transcribe=lambda *a, **k: {"transcript": None,
                                              "error": "auth rejected",
                                              "word_confidences": []},
            key=FAKE_KEY)
        self._assert_survived(rc, out, "auth rejected")

    def test_a_response_with_no_per_word_confidences(self):
        """SPEC 12's day-one gate, restated on stage: with no per-word
        confidence a beat about confidence has nothing to show, so it falls
        back rather than narrating an empty list."""
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--clip", self.e["clip_id"]],
            patch_transcribe=lambda *a, **k: ok_response("something", []),
            key=FAKE_KEY)
        self._assert_survived(rc, out, "no per-word confidences")

    def test_a_call_that_never_returns_is_abandoned_on_the_deadline(self):
        def hang(*a, **k):
            time.sleep(60)
        t0 = time.monotonic()
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--timeout", "1",
             "--clip", self.e["clip_id"]],
            patch_transcribe=hang, key=FAKE_KEY)
        elapsed = time.monotonic() - t0
        self._assert_survived(rc, out, "no response within 1s")
        self.assertLess(elapsed, 30, "the deadline did not fire")

    def test_a_fallback_does_not_get_presented_as_a_reproduction(self):
        """
        The reproduction panel compares stage 2 against the archive. When the
        live call failed, stage 2 IS the archive, so the two agree trivially —
        and a panel reading 'Reproduced — within 0.0000' would be the single
        most misleading line the demo could print. It has to say so instead.
        """
        def boom(*a, **k):
            raise RuntimeError("vendor said 503")
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--clip", self.e["clip_id"]],
            patch_transcribe=boom, key=FAKE_KEY)
        f = flat(out)
        self.assertEqual(rc, 0)
        self.assertIn("proves nothing today", f)
        self.assertNotIn("Reproduced —", f)
        # and the header must not have asserted a call that never happened
        self.assertNotIn("nothing cached", f)

    def test_NEGATIVE_a_real_call_does_get_the_reproduction_verdict(self):
        """Same panel, a call that actually happened. Without this the check
        above would pass on a demo that never claims a reproduction at all."""
        ref = self.e["ref"]
        grid = self.e["grid"]
        seq = [ok_response(ref, [0.99] * len(ref.split())),
               ok_response(grid["transcript"], grid["word_confidences"])]
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--clip", self.e["clip_id"]],
            patch_transcribe=lambda *a, **k: seq.pop(0), key=FAKE_KEY)
        f = flat(out)
        self.assertEqual(rc, 0)
        self.assertIn("Reproduced —", f)
        self.assertNotIn("proves nothing today", f)

    def test_NEGATIVE_a_working_call_produces_no_skip_notice(self):
        """Without this, a demo that ALWAYS fell back would pass every test
        above."""
        ref = self.e["ref"]
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio", "--clip", self.e["clip_id"]],
            patch_transcribe=lambda *a, **k: ok_response(
                ref, [0.9] * len(ref.split())),
            key=FAKE_KEY)
        self.assertEqual(rc, 0)
        self.assertNotIn("LIVE CALL SKIPPED", out)
        self.assertIn("two live calls", out)


class NeverHangsOnStdin(unittest.TestCase):
    """It reads stdin in front of an audience."""

    def test_eof_finishes_the_beat(self):
        r = child("--replay", "--no-color", "--no-audio", stdin="")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("WHAT JUST HAPPENED", r.stdout)

    def test_garbage_input_does_not_loop_forever(self):
        r = child("--replay", "--no-color", "--no-audio",
                  stdin="\n".join(["zzz"] * 40), timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])

    def test_a_menu_choice_selects_that_clip(self):
        """The interviewer picking from the menu is the feature; assert the
        choice is honoured rather than silently defaulted."""
        cache = load_cache()
        if len(cache["exemplars"]) < 2:
            self.skipTest("need two exemplars to tell a choice from a default")
        want = cache["exemplars"][1]
        rc, out = in_process(["--replay", "--no-color", "--once", "--no-audio",
                              "--clip", want["clip_id"]])
        self.assertEqual(rc, 0)
        self.assertIn(want["ref"], out)
        self.assertNotIn(cache["exemplars"][0]["ref"], out)

    def test_random_stays_inside_the_curated_set(self):
        cache = load_cache()
        refs = {e["ref"] for e in cache["exemplars"]}
        for seed in range(6):
            rc, out = in_process(["--replay", "--no-color", "--once", "--no-audio",
                                  "--random", "--seed", str(seed)])
            self.assertEqual(rc, 0)
            self.assertTrue(any(r in out for r in refs),
                            "random picked something outside the curated set")

    def test_an_unknown_clip_is_refused_by_name(self):
        r = child("--replay", "--no-color", "--once", "--no-audio", "--clip", "u99")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not in the curated set", r.stdout + r.stderr)


class TheInteractiveBeat(unittest.TestCase):
    """
    The menu and the mid-beat pause only run when stdin is a terminal, so a
    piped test never reaches them. They are driven directly here instead —
    otherwise the one feature the interviewer actually touches is the one
    feature nothing covers.
    """

    def setUp(self):
        self.cache = load_cache()
        self.ink = hero.Ink(False)
        import random as _r
        self.rng = _r.Random(0)

    @contextlib.contextmanager
    def _typed(self, *answers):
        import builtins
        saved = builtins.input
        seq = list(answers)

        def fake(prompt=""):
            if not seq:
                raise EOFError
            return seq.pop(0)

        builtins.input = fake
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            builtins.input = saved

    def _menu(self, *answers):
        turn = hero.Turn(interactive=True)
        with self._typed(*answers):
            return hero.menu(self.cache, self.ink, 96, turn, self.rng)

    def test_typing_a_number_selects_that_entry(self):
        if len(self.cache["exemplars"]) < 3:
            self.skipTest("need three entries to tell a choice from a default")
        self.assertEqual(self._menu("3")["clip_id"],
                         self.cache["exemplars"][2]["clip_id"])

    def test_typing_a_clip_id_selects_that_entry(self):
        want = self.cache["exemplars"][-1]["clip_id"]
        self.assertEqual(self._menu(want)["clip_id"], want)

    def test_bare_enter_takes_the_default(self):
        self.assertEqual(self._menu("")["clip_id"], self.cache["default"])

    def test_r_picks_from_inside_the_curated_set(self):
        ids = {e["clip_id"] for e in self.cache["exemplars"]}
        self.assertIn(self._menu("r")["clip_id"], ids)

    def test_q_quits_and_the_caller_exits_zero(self):
        self.assertIsNone(self._menu("q"))
        turn = hero.Turn(interactive=True)
        with self._typed("", "q"):                    # preamble enter, then quit
            with contextlib.redirect_stdout(io.StringIO()):
                rc = hero.run(self.cache, self.ink, 96, want_live=False,
                              timeout=1.0, audio=False, pick=None,
                              use_random=False, once=True, rng=self.rng)
        self.assertEqual(rc, 0)

    def test_NEGATIVE_unparseable_input_does_not_loop_forever(self):
        """Five bad answers then EOF. It must fall through to the default, not
        block a presenter in front of a room."""
        got = self._menu(*(["zzz"] * 12))
        self.assertEqual(got["clip_id"], self.cache["default"])

    def test_the_pause_replays_the_clean_clip_on_r_then_moves_on(self):
        """`r` at the mid-beat pause is the one control the presenter uses when
        somebody asks to hear the clean version again."""
        played: list[str] = []
        saved = hero.play
        hero.play = lambda path, ink, enabled: played.append(Path(path).name)
        try:
            turn = hero.Turn(interactive=True)
            e = self.cache["exemplars"][0]
            with self._typed("r", "r", ""):
                hero.run_one(e, self.ink, 96, want_live=False, timeout=1.0,
                             audio=True, turn=turn)
        finally:
            hero.play = saved
        clean = Path(e["audio_clean"]).name
        degraded = Path(e["audio_degraded"]).name
        self.assertEqual(played.count(clean), 3, f"played: {played}")
        self.assertEqual(played[-1], degraded, "the degraded clip never played")

    def test_NEGATIVE_without_r_the_clean_clip_plays_exactly_once(self):
        """Proof the count above is the replays and not an artefact of the
        instrumentation."""
        played: list[str] = []
        saved = hero.play
        hero.play = lambda path, ink, enabled: played.append(Path(path).name)
        try:
            turn = hero.Turn(interactive=True)
            e = self.cache["exemplars"][0]
            with self._typed(""):
                hero.run_one(e, self.ink, 96, want_live=False, timeout=1.0,
                             audio=True, turn=turn)
        finally:
            hero.play = saved
        self.assertEqual(played.count(Path(e["audio_clean"]).name), 1)


class NoAudioBeforeTheRoomIsReady(unittest.TestCase):
    """
    The preamble exists so nobody is surprised by sound. Structural: the first
    playback call must come after the preamble returns.
    """

    def test_the_preamble_precedes_every_play_call(self):
        src = (REPO / "demos" / "demo_hero.py").read_text()
        run_body = src[src.index("def run(cache"):]
        self.assertLess(run_body.index("preamble("), run_body.index("run_one("),
                        "the beat starts before the room has been warned")
        one = src[src.index("def run_one("):src.index("def run(cache")]
        self.assertIn("play(entry[", one)

    def test_the_preamble_warns_about_audio_and_waits(self):
        out = child("--replay", "--no-color", "--once", stdin="").stdout
        f = flat(out)
        self.assertIn("AUDIO WILL PLAY", f)
        self.assertIn("Check your volume", f)

    def test_NEGATIVE_no_audio_says_playback_is_off_instead(self):
        out = child("--replay", "--no-color", "--once", "--no-audio",
                    stdin="").stdout
        f = flat(out)
        self.assertNotIn("AUDIO WILL PLAY", f)
        self.assertIn("playback is off", f)


class NoCredentialCanReachTheScreen(unittest.TestCase):
    """
    The realistic leak: a vendor error string with the key inside it, folded
    into the demo's own message and then WRAPPED — which is how a previous
    version printed a key across two lines with nothing matching on either
    (SPEC J.3).
    """

    def test_a_key_embedded_in_a_vendor_error_is_redacted(self):
        def boom(*a, **k):
            raise RuntimeError(f"401 unauthorized for token={FAKE_KEY} on /listen")
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio"],
            patch_transcribe=boom, key=FAKE_KEY)
        self.assertEqual(rc, 0)
        squeezed = "".join(out.split())
        self.assertNotIn(FAKE_KEY, squeezed,
                         "the API key reached stdout")
        self.assertIn("<redacted>", out)

    def test_NEGATIVE_a_non_secret_of_the_same_shape_survives_untouched(self):
        """A redactor that blanked everything would pass the test above and the
        demo would print nothing useful."""
        def boom(*a, **k):
            raise RuntimeError(f"503 upstream request id={DECOY}")
        rc, out = in_process(
            ["--no-color", "--once", "--no-audio"],
            patch_transcribe=boom, key=FAKE_KEY)
        self.assertEqual(rc, 0)
        self.assertIn(DECOY, "".join(out.split()))


class Preflight(unittest.TestCase):

    def test_check_exits_zero_when_the_archive_is_present(self):
        r = child("--check", "--no-color")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("archived fallback is present", r.stdout)

    def test_list_prints_the_whole_curated_set_with_its_filters(self):
        r = child("--list", "--no-color")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        for e in load_cache()["exemplars"]:
            self.assertIn(e["clip_id"], r.stdout)
        self.assertIn("clean WER 0.000", r.stdout)


class TheMakefileWiring(unittest.TestCase):
    """`make demo` is the command the candidate types. It has to be this one."""

    def setUp(self):
        self.mk = (REPO / "Makefile").read_text()

    def test_make_demo_runs_the_hero(self):
        body = self.mk[self.mk.index("\ndemo: "):]
        body = body[:body.index("\n\n")]
        self.assertIn("demos/demo_hero.py", body)

    def test_the_old_chain_survives_as_demo_all(self):
        self.assertRegex(self.mk, r"\ndemo-all: .*demo.*demo-al.*dashboard")

    def test_the_fallbacks_are_still_targets(self):
        for target in ("demo-break:", "demo-live:", "demo-replay:"):
            self.assertIn(f"\n{target}", self.mk)
        self.assertIn("demo-break", self.mk[:self.mk.index("help:")],
                      "demo-break dropped out of .PHONY")

    def test_help_lists_the_hero_and_files_the_old_beats_as_fallbacks(self):
        help_text = self.mk[self.mk.index("help:"):self.mk.index("# ----", self.mk.index("help:"))]
        self.assertIn("make demo", help_text)
        self.assertIn("FALLBACKS", help_text)
        # the two superseded beats must not sit in the on-stage block
        on_stage = help_text[:help_text.index("FALLBACKS")]
        self.assertNotIn("make demo-break", on_stage)
        self.assertNotIn("make demo-live", on_stage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
