#!/usr/bin/env python3
"""
test_make_demo_audio.py — the generator must not eat the record.

    ./.venv/bin/python tests/test_make_demo_audio.py

`scripts/make_demo_audio.py` produces two KINDS of artifact from one build, and
until now it wrote both with the same call:

  * the wavs and `manifest.json`, a pure function of `results/master.csv` and the
    asset library — losing them costs three seconds; and
  * five markdown documents, which are where a human writes down the one thing
    the pipeline cannot produce. `PREREGISTERED_PREDICTION.md` now carries a
    pre-registered prediction, the verbatim listener response, the scoring, the
    verdict (it FAILED) and an analysis of a flaw in the pre-registration's own
    rubric. None of that is derivable from any artifact in this repo.

Rewriting the second kind unconditionally already destroyed one such edit, and
it destroyed it SILENTLY — a regenerated document looks exactly as correct as
the one it replaced, so nothing tells you. That is SPEC Appendix E.5's family
("a guard whose failure mode is silence") pointed at the project's own demo kit.

What is asserted here is therefore both halves of the guard, because only the
pair is evidence:

  * it REFUSES to overwrite a document a human has touched, and the original
    bytes are still on disk afterwards — not merely that it returned "skipped";
  * it still writes normally when the document is absent, or is its own
    untouched output. A guard that refuses everything is not a guard, it is an
    outage, and it trains the next person to pass --force-docs by reflex.

The last test in `TheAuthoredRecord` is the concrete regression: it runs the
real generator, at its real default, against a COPY of the real demo directory
and asserts that not one byte of any document moved. The copy is not squeamishness
— a test that exercises the destructive path against the live repo is precisely
the footgun being fixed here, and `tests/test_demo.py` runs this script with
`--force` against the live tree on every `make test`.
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
# Makes `deadzone`, `scripts` and `demos` importable when this file is run
# directly (`python tests/test_make_demo_audio.py`) with no install step.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# -------------------------------------------------------------------------

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import make_demo_audio as mda

REPO = Path(_REPO_ROOT)
PY = str(Path(sys.executable))
DEMO = REPO / "results" / "audio" / "demo"

# The exact strings `results/audio/demo/REGENERATION_HAZARD.md` names as the
# things to check for after any regeneration. They are hardcoded on purpose: if
# a run of this generator makes them disappear, that IS the loss event, and a
# test that re-derived them from the file it is checking could not see it.
OUTCOME_SENTINELS = {
    "PREREGISTERED_PREDICTION.md": ("OUTCOME — tested 2026-08-05",
                                    "FAILED on direction"),
    "DEMO_SCRIPT.md": ("The disagreement, and the prediction I got wrong",),
}


def _build_sandbox(root: Path, script_patches: list[tuple[str, str]] | None = None
                   ) -> Path:
    """A throwaway repo root holding a COPY of `results/audio/demo/`.

    `scripts/` is symlinked when unpatched and copied when a mutation is asked
    for, so the mutation lives only inside the sandbox and the real generator is
    never edited by a test.
    """
    (root / "results" / "audio").mkdir(parents=True)
    for name in ("data", "recording_manifest.csv", "deadzone"):
        os.symlink(REPO / name, root / name)
    if script_patches:
        shutil.copytree(REPO / "scripts", root / "scripts")
        src_path = root / "scripts" / "make_demo_audio.py"
        src = src_path.read_text()
        for old, new in script_patches:
            assert old in src, (
                f"the mutation anchor is gone from make_demo_audio.py, so this "
                f"negative control is no longer reintroducing the defect it "
                f"claims to — update it:\n{old}")
            src = src.replace(old, new, 1)
        src_path.write_text(src)
    else:
        os.symlink(REPO / "scripts", root / "scripts")
    os.symlink(REPO / "results" / "master.csv", root / "results" / "master.csv")
    shutil.copytree(DEMO, root / "results" / "audio" / "demo")
    return root / "results" / "audio" / "demo"


class TheGuard(unittest.TestCase):
    """
    `write_doc()` in its four states. Every refusal is paired with the write it
    must still allow, so each test is pinned to the violation rather than to
    some incidental property of the fixture.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="deadzone-docguard-"))
        self.doc = self.tmp / "DOC.md"
        self._saved = mda.DOC_HASHES
        mda.DOC_HASHES = self.tmp / "generated_docs.json"

    def tearDown(self):
        mda.DOC_HASHES = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- the writes it must still perform (the negative controls) ----------

    def test_it_writes_a_document_that_is_not_there(self):
        self.assertEqual(mda.write_doc(self.doc, "generated v1"), "written")
        self.assertEqual(self.doc.read_text(), "generated v1")

    def test_it_rewrites_its_own_untouched_output(self):
        """
        The alarm-fatigue control. The numbers in these documents change every
        time the grid does, so a guard that also refused its OWN output would
        make --force-docs the habitual invocation and the protection would be
        decoration inside a week.
        """
        mda.write_doc(self.doc, "generated v1")
        self.assertEqual(mda.write_doc(self.doc, "generated v2"), "written")
        self.assertEqual(self.doc.read_text(), "generated v2")

    # --- the refusal --------------------------------------------------------

    def test_it_refuses_to_overwrite_a_hand_edited_document(self):
        mda.write_doc(self.doc, "generated v1")
        self.doc.write_text("generated v1\n\nOUTCOME: the prediction FAILED.")
        edited = self.doc.read_text()

        self.assertEqual(mda.write_doc(self.doc, "generated v2"), "skipped")
        # The bytes, not the return value. A guard that reports "skipped" and
        # writes anyway is the same defect wearing a label.
        self.assertEqual(self.doc.read_text(), edited)
        self.assertIn("OUTCOME: the prediction FAILED.", self.doc.read_text())

    def test_an_unrecorded_document_is_refused_not_adopted(self):
        """
        The degenerate input, and the state this repo was actually in: five
        hand-edited documents and no sidecar yet written. SPEC E.5's rule is to
        ask what a guard returns for the degenerate input — if "no record" meant
        "safe to overwrite", the guard would have been wide open on day one.
        """
        self.doc.write_text("hand-written, provenance unknown")
        self.assertFalse(mda.DOC_HASHES.exists())

        self.assertEqual(mda.write_doc(self.doc, "template output"), "skipped")
        self.assertEqual(self.doc.read_text(), "hand-written, provenance unknown")

        # Control: the same file, once the generator has a record of writing it,
        # is rewritten. So the refusal is caused by the missing record and not by
        # anything else about this fixture.
        mda.write_doc(self.doc, "template output", force_docs=True)
        self.assertEqual(mda.write_doc(self.doc, "template v2"), "written")

    def test_a_corrupt_or_absent_sidecar_fails_closed(self):
        mda.write_doc(self.doc, "generated v1")
        good = mda.DOC_HASHES.read_text()

        for label, broken in (("corrupt", "not json {{{"),
                              ("empty", ""),
                              ("wrong shape", json.dumps({"sha256": "a string"}))):
            with self.subTest(sidecar=label):
                mda.DOC_HASHES.write_text(broken)
                self.assertEqual(mda.write_doc(self.doc, "template output"),
                                 "skipped", f"a {label} sidecar failed OPEN")
                self.assertEqual(self.doc.read_text(), "generated v1")

        mda.DOC_HASHES.unlink()
        self.assertEqual(mda.write_doc(self.doc, "template output"), "skipped")

        # Control: restore the real sidecar and the very same write goes through.
        mda.DOC_HASHES.write_text(good)
        self.assertEqual(mda.write_doc(self.doc, "template output"), "written")

    def test_it_compares_content_and_not_mtime(self):
        """
        mtime does not survive a checkout, a `cp` without `-p`, or a restore from
        backup, and all three happen to a demo kit that gets handed around. Here
        the edit is made and the mtime is then rewound to the moment the
        generator wrote the file; an mtime-based guard sees nothing.
        """
        mda.write_doc(self.doc, "generated v1")
        stamp = self.doc.stat().st_mtime
        self.doc.write_text("generated v1 + a hand-written outcome")
        os.utime(self.doc, (stamp, stamp))
        self.assertEqual(self.doc.stat().st_mtime, stamp)

        self.assertEqual(mda.write_doc(self.doc, "generated v2"), "skipped")
        self.assertIn("hand-written outcome", self.doc.read_text())

    def test_write_doc_protects_by_default(self):
        """`force_docs` defaults to False, so a caller that forgets the keyword
        gets the protective behaviour rather than the destructive one."""
        self.doc.write_text("hand-written")
        self.assertEqual(mda.write_doc(self.doc, "template"), "skipped")  # no kwarg
        self.assertEqual(self.doc.read_text(), "hand-written")

    def test_the_refusal_names_the_file_and_the_exact_override(self):
        """A refusal nobody can act on is a warning, and warnings get ignored."""
        msg = mda._refusal(self.doc)
        self.assertIn(str(self.doc), msg)
        self.assertIn("--force-docs", msg)
        self.assertIn("REFUSING", msg)
        self.assertIn("UNCHANGED", msg)

    # --- the override, which must still work --------------------------------

    def test_force_docs_overwrites_and_the_old_text_is_recoverable(self):
        """
        The kit has to stay rebuildable from scratch, so the override is real.
        It also takes a copy first: the one irreversible operation in this script
        should not be the one with no undo.
        """
        self.doc.write_text("the only copy of a listening result")
        self.assertEqual(mda.write_doc(self.doc, "fresh template",
                                       force_docs=True), "written")
        self.assertEqual(self.doc.read_text(), "fresh template")

        backups = [p for p in self.tmp.glob("DOC.superseded-*.md")]
        self.assertEqual(len(backups), 1, f"no backup was taken: {list(self.tmp.iterdir())}")
        self.assertEqual(backups[0].read_text(),
                         "the only copy of a listening result")

    def test_doc_status_reports_the_three_states(self):
        self.assertEqual(mda.doc_status(self.doc), mda.ABSENT)
        mda.write_doc(self.doc, "generated v1")
        self.assertEqual(mda.doc_status(self.doc), mda.GENERATED)
        self.doc.write_text("edited")
        self.assertEqual(mda.doc_status(self.doc), mda.AUTHORED)

    def test_the_sidecar_records_the_hash_of_what_was_written(self):
        mda.write_doc(self.doc, "generated v1")
        rec = json.loads(mda.DOC_HASHES.read_text())
        want = hashlib.sha256(b"generated v1").hexdigest()
        self.assertEqual(rec["sha256"][self.doc.as_posix()], want)


class NoWriterBypassesTheGuard(unittest.TestCase):
    """
    Structural, because the fix is only worth what the next document is worth.
    Adding a sixth markdown file and reaching for `.write_text()` out of habit
    reintroduces the whole defect for that one file, and nothing would say so.
    """

    def test_every_write_text_call_goes_through_write_doc(self):
        src = (REPO / "scripts" / "make_demo_audio.py").read_text()
        tree = ast.parse(src)
        allowed = {"write_doc",        # the guard itself
                   "_save_doc_hash",   # the sidecar, which is not a document
                   "main"}             # manifest.json, machine data, regenerable
        offenders = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "write_text"
                        and fn.name not in allowed):
                    offenders.append(f"{fn.name}() at line {node.lineno}")
        self.assertEqual(offenders, [],
                         "these write markdown without the authored-document "
                         "guard, so a hand-edited file would be destroyed "
                         "silently — route them through write_doc()")

    def test_the_census_covers_every_document_the_build_writes(self):
        """`DOCS` is what --check and the run summary report on. A document that
        is guarded but absent from the census is invisible in both."""
        names = {Path(p).name for p in mda.DOCS}
        for want in ("DEMO_SCRIPT.md", "KEY.md", "BLIND_SHEET.md",
                     "PREREGISTERED_PREDICTION.md", "WHAT_TO_LISTEN_FOR.md"):
            self.assertIn(want, names)


class TheAuthoredRecord(unittest.TestCase):
    """
    The concrete regression. On 2026-08-05 a regeneration erased a hand-written
    outcome minutes after it was written, and it was caught only because an
    editing tool happened to notice the file had changed underneath it.
    """

    @classmethod
    def setUpClass(cls):
        if not DEMO.is_dir():
            raise unittest.SkipTest(f"{DEMO} not present")

    def test_the_outcome_record_is_on_disk(self):
        for name, sentinels in OUTCOME_SENTINELS.items():
            text = (DEMO / name).read_text(encoding="utf-8")
            for s in sentinels:
                self.assertIn(
                    s, text,
                    f"{name} no longer contains {s!r}. Either the record was "
                    f"destroyed by a regeneration — see "
                    f"results/audio/demo/REGENERATION_HAZARD.md, which names "
                    f"exactly this string as the check — or it was deliberately "
                    f"rewritten, in which case update OUTCOME_SENTINELS here.")

    def test_a_default_run_leaves_every_document_byte_identical(self):
        """
        The real generator, at its real default, against a COPY of the real
        demo directory. Byte-equality rather than a sentinel search, so this
        keeps working when the wording of the record changes.
        """
        master = REPO / "results" / "master.csv"
        if not master.is_file() or not (REPO / "data" / "recordings").is_dir():
            self.skipTest("needs results/master.csv and data/ to run the generator")

        with tempfile.TemporaryDirectory(prefix="deadzone-demo-sandbox-") as td:
            sb = Path(td)
            (sb / "results" / "audio").mkdir(parents=True)
            for name in ("data", "recording_manifest.csv", "deadzone", "scripts"):
                os.symlink(REPO / name, sb / name)
            os.symlink(master, sb / "results" / "master.csv")
            shutil.copytree(DEMO, sb / "results" / "audio" / "demo")

            before = {p.name: p.read_bytes()
                      for p in (sb / "results" / "audio" / "demo").rglob("*.md")}
            n_wavs_before = len(list((sb / "results" / "audio" / "demo").rglob("*.wav")))

            r = subprocess.run([PY, "scripts/make_demo_audio.py", "--force"],
                               cwd=sb, capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])

            after = {p.name: p.read_bytes()
                     for p in (sb / "results" / "audio" / "demo").rglob("*.md")}
            changed = sorted(n for n in before if after.get(n) != before[n])
            self.assertEqual(changed, [],
                             "a default run rewrote hand-edited documents — this "
                             "is the exact regression the guard exists to stop")

            # The control that makes the assertion above mean something: the run
            # really did do its work. A generator that crashed early, or bailed
            # on "up to date", would leave every document untouched too and would
            # pass a test that only checked for damage.
            self.assertIn("REFUSING to overwrite", r.stdout)
            self.assertIn("PROTECTED", r.stdout)
            self.assertEqual(
                len(list((sb / "results" / "audio" / "demo").rglob("*.wav"))),
                n_wavs_before, "the audio half of the build did not run")

    def test_force_docs_can_still_rebuild_the_kit_from_scratch(self):
        """The override is not decorative: without it the kit could never be
        regenerated after the first hand edit."""
        master = REPO / "results" / "master.csv"
        if not master.is_file() or not (REPO / "data" / "recordings").is_dir():
            self.skipTest("needs results/master.csv and data/ to run the generator")

        with tempfile.TemporaryDirectory(prefix="deadzone-demo-force-") as td:
            sb = Path(td)
            (sb / "results" / "audio").mkdir(parents=True)
            for name in ("data", "recording_manifest.csv", "deadzone", "scripts"):
                os.symlink(REPO / name, sb / name)
            os.symlink(master, sb / "results" / "master.csv")
            shutil.copytree(DEMO, sb / "results" / "audio" / "demo")
            demo_sb = sb / "results" / "audio" / "demo"
            before = (demo_sb / "PREREGISTERED_PREDICTION.md").read_bytes()

            r = subprocess.run([PY, "scripts/make_demo_audio.py", "--force-docs"],
                               cwd=sb, capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])

            self.assertNotEqual(
                (demo_sb / "PREREGISTERED_PREDICTION.md").read_bytes(), before,
                "--force-docs did not overwrite, so the kit is not rebuildable")
            backups = list(demo_sb.glob("PREREGISTERED_PREDICTION.superseded-*.md"))
            self.assertEqual(len(backups), 1,
                             "--force-docs overwrote without keeping a copy")
            self.assertEqual(backups[0].read_bytes(), before)

            # And the second run is quiet: the documents are now the generator's
            # own output again, so nothing is refused and nothing is backed up.
            r2 = subprocess.run([PY, "scripts/make_demo_audio.py", "--force"],
                                cwd=sb, capture_output=True, text=True, timeout=600)
            self.assertEqual(r2.returncode, 0, r2.stderr[-3000:])
            self.assertNotIn("REFUSING to overwrite", r2.stdout)


class ThePlayOrderIsDerivedFromTheListenerRecord(unittest.TestCase):
    """
    The second regression of the same family, and it survived the first fix.

    `write_doc` made the DOCUMENTS safe. It did nothing for the TEMPLATES. After
    the 2026-08-05 session called pair 1 marginal and `KEY.md` / `DEMO_SCRIPT.md`
    were hand-corrected to lead with pairs 2 and 3, the generator still carried
    `"role": "primary" if i <= 2 else "backup"` and still printed "Pairs 1 and 2
    are the primary evidence" — so `manifest.json`, the machine-readable answer
    key, disagreed with both human-readable answer sheets, and `--force-docs`
    would have rebuilt the sheets FROM the stale constant. The guard protects the
    file; it cannot protect the content while the template is still wrong.

    So the ordering is now derived from `LISTENER_SESSIONS`, and what is asserted
    here is the DERIVATION, not the current answer: the expectation below is
    recomputed from the record with a deliberately separate implementation, so a
    future session that appends a listener moves the test's expectation with it
    instead of breaking a hardcoded [2, 3, 1].
    """

    # The mutation that reinstates the pre-2026-08-06 behaviour EXACTLY: play
    # order = construction order, and the first two pairs are primary regardless
    # of what anyone said. Applied to a sandbox copy for the negative control.
    OLD_HARDCODED_ROLE = [
        ("    return sorted(cl, key=lambda c: PLAY_RANK.get(\n"
         "        listener_call(c)[\"confidence\"], PLAY_RANK[UNTESTED]))",
         "    return cl                      # OLD: construction order"),
        ('        primary = call["confidence"] == CONFIDENT',
         "        primary = pos <= 2         # OLD: 'primary' if i <= 2"),
    ]

    @classmethod
    def setUpClass(cls):
        if not DEMO.is_dir():
            raise unittest.SkipTest(f"{DEMO} not present")

    # --- the expectation, recomputed from the record -----------------------

    @staticmethod
    def expected_order() -> list[str]:
        """Clip ids in play order, derived from `LISTENER_SESSIONS` here rather
        than by calling `mda.play_order()` — a test that asks the code under test
        what the answer is cannot fail."""
        calls = {}
        for s in mda.LISTENER_SESSIONS:
            calls.update(s.get("calls") or {})
        rank = {mda.CONFIDENT: 0, mda.MARGINAL: 1, mda.UNTESTED: 2}
        return sorted(mda.PAIR_CLIPS,
                      key=lambda c: rank[calls.get(c, {}).get("confidence",
                                                              mda.UNTESTED)])

    @staticmethod
    def recorded_confidence(clip: str) -> str:
        for s in reversed(mda.LISTENER_SESSIONS):
            c = (s.get("calls") or {}).get(clip)
            if c:
                return c["confidence"]
        return mda.UNTESTED

    def assert_kit_agrees_with_the_record(self, demo: Path, *,
                                          check_blind_sheet: bool = True):
        """Every artifact in the kit tells the same story about which pair leads.

        Raises AssertionError on any disagreement — which is what the negative
        control below asserts it does.

        `check_blind_sheet=False` is for the LIVE kit only. `blind/BLIND_SHEET.md`
        on disk predates the play-order fix and is `authored` (no recorded hash),
        so the generator will not touch it without `--force-docs` — and running
        that against the live kit is precisely what the guard exists to prevent
        being done casually. The row-order guarantee is therefore asserted where
        it is actually a guarantee: on a fresh `--force-docs` regeneration.
        """
        want = self.expected_order()
        m = json.loads((demo / "manifest.json").read_text(encoding="utf-8"))
        by_clip = {p["clip_id"]: p for p in m["pairs"]}
        num = {c: by_clip[c]["pair"] for c in want}

        # 1. the machine-readable key
        self.assertEqual(m["play_order"], [num[c] for c in want],
                         "manifest.json play_order disagrees with the record")
        for clip, p in by_clip.items():
            conf = self.recorded_confidence(clip)
            self.assertEqual(p["role"], "primary" if conf == mda.CONFIDENT
                             else "reserve",
                             f"{clip}: role does not follow its recorded call "
                             f"({conf}) — it is hardcoded again")
            self.assertEqual(p["listener_confidence"], conf)
        self.assertEqual(self.recorded_confidence(want[0]), mda.CONFIDENT,
                         "the exercise opens on a pair the listener hedged on")

        # 2. the presenter's answer sheet
        key = (demo / "KEY.md").read_text(encoding="utf-8")
        for clip, p in by_clip.items():
            self.assertIn(f"### Pair {p['pair']} — `{clip}` ({p['play_note']})",
                          key, "KEY.md's per-pair annotation is not the derived "
                               "play_note")
        self.assertIn(f"pair {num[want[0]]}, then pair {num[want[1]]}", key,
                      "KEY.md's stated play order disagrees with the record")

        # 3. the run-of-show. Position within section 2, not mere presence: the
        #    old template named every clip too, just in the wrong order.
        script = (demo / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
        sec2 = script.split("## 2. They listen and rank")[1].split("\n## ")[0]
        seen = [c for c in re.findall(r"u\d\d", sec2)]
        self.assertEqual(seen[:len(want)], want,
                         f"DEMO_SCRIPT section 2 presents the pairs as {seen} — "
                         f"the record says {want}")

        # 4. the listener's own sheet, which they work top to bottom
        if check_blind_sheet:
            sheet = (demo / "blind" / "BLIND_SHEET.md").read_text(encoding="utf-8")
            rows = re.findall(r"^\| (\d) \| `blind", sheet, flags=re.M)
            self.assertEqual([int(r) for r in rows], [num[c] for c in want],
                             "BLIND_SHEET rows are not in play order, so a "
                             "listener working down the page undoes the ordering")

    # --- the live kit ------------------------------------------------------

    def test_the_kit_on_disk_agrees_with_the_record(self):
        self.assert_kit_agrees_with_the_record(DEMO, check_blind_sheet=False)

    def test_emptying_the_record_degrades_to_construction_order(self):
        """The degenerate input (SPEC E.5). With nothing recorded, the honest
        answer is 'no judgement here', not a stale opinion with no owner."""
        saved = mda.LISTENER_SESSIONS
        try:
            mda.LISTENER_SESSIONS = ()
            self.assertEqual(mda.play_order(), list(mda.PAIR_CLIPS))
            roles = mda.pair_roles()
            self.assertEqual({r["role"] for r in roles.values()}, {"reserve"})
            self.assertEqual({r["listener_confidence"] for r in roles.values()},
                             {mda.UNTESTED})
        finally:
            mda.LISTENER_SESSIONS = saved
        # Control: with the record restored, a pair IS primary again — so the
        # `reserve` above is caused by the empty record, not by pair_roles()
        # being unable to produce a primary at all.
        self.assertIn("primary", {r["role"] for r in mda.pair_roles().values()})

    # --- the regression, and its mutation control --------------------------

    def _regenerate(self, patches=None) -> tuple[Path, subprocess.CompletedProcess]:
        td = tempfile.mkdtemp(prefix="deadzone-playorder-")
        self.addCleanup(shutil.rmtree, td, ignore_errors=True)
        demo = _build_sandbox(Path(td), patches)
        r = subprocess.run([PY, "scripts/make_demo_audio.py", "--force-docs"],
                           cwd=td, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        return demo, r

    def test_force_docs_reproduces_the_correction_instead_of_reverting_it(self):
        """
        THE regression. `--force-docs` rebuilds every document from the
        templates, which is exactly the path that would have restored guidance a
        listener's response contradicts. Run against a COPY, never the live kit:
        those files carry a verbatim listener response that derives from no
        artifact, and a regeneration already destroyed that record once.
        """
        master = REPO / "results" / "master.csv"
        if not master.is_file() or not (REPO / "data" / "recordings").is_dir():
            self.skipTest("needs results/master.csv and data/ to run the generator")

        demo, r = self._regenerate()
        self.assertIn("OVERWRITING hand-edited", r.stdout,
                      "--force-docs did not actually rewrite anything, so this "
                      "proves nothing about what the templates produce")
        self.assert_kit_agrees_with_the_record(demo)

        # And the listener-outcome material is REPRODUCED, not merely tolerated:
        # a template that dropped the failed prediction would leave the file
        # reading as an OPEN one, which is worse than no record at all.
        for name, sentinels in OUTCOME_SENTINELS.items():
            text = (demo / name).read_text(encoding="utf-8")
            for sentinel in sentinels:
                self.assertIn(sentinel, text,
                              f"--force-docs regenerated {name} without {sentinel!r}")
        pred = (demo / "PREREGISTERED_PREDICTION.md").read_text(encoding="utf-8")
        self.assertIn("bro 3 and 7 are both pretty bad", pred,
                      "the verbatim listener response did not survive")
        script = (demo / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
        for refuted in ("precedence effect", "informational masking"):
            self.assertNotIn(
                refuted, script,
                f"--force-docs put the refuted {refuted!r} mechanism back into "
                f"the run-of-show — REGENERATION_HAZARD.md names this as the "
                f"post-regeneration check")

    def test_negative_control_the_old_hardcoded_role_fails_this(self):
        """
        The mutation control. Restore the two lines that made the ordering a
        bare constant — nothing else — and the assertions above must fail.
        Without this the test could be passing on some incidental property of
        the fixture rather than on the ordering actually being derived.
        """
        master = REPO / "results" / "master.csv"
        if not master.is_file() or not (REPO / "data" / "recordings").is_dir():
            self.skipTest("needs results/master.csv and data/ to run the generator")

        demo, _ = self._regenerate(self.OLD_HARDCODED_ROLE)
        m = json.loads((demo / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([p["role"] for p in m["pairs"]],
                         ["primary", "primary", "reserve"],
                         "the mutation did not reinstate the old hardcoded role, "
                         "so this control is not testing what it claims")
        with self.assertRaises(AssertionError):
            self.assert_kit_agrees_with_the_record(demo)


if __name__ == "__main__":
    unittest.main(verbosity=2)
