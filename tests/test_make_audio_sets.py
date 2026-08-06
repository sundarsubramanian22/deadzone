#!/usr/bin/env python3
"""
test_make_audio_sets.py — the generator must not eat the listening notes.

    ./.venv/bin/python tests/test_make_audio_sets.py

`scripts/make_audio_sets.py` produces two KINDS of artifact from one run, and
until now it wrote both with the same call:

  * the wavs and `results/audio/sweep/index.json`, a pure function of `data/` and
    the condition names — `apply_condition` seeds its noise crop from the
    condition NAME, so losing them costs seconds; and
  * `results/audio/listen/WHAT_TO_LISTEN_FOR.md`, the instructions page that sits
    in the directory where the listening pass actually HAPPENS. That makes it the
    likeliest file in this repo for a human to annotate with what they heard, and
    in this project a listening note is not a nicety: the listening pass is what
    invalidated the published headline (SPEC Appendix G — the estimand mismatch,
    dead zones 6 -> 2). Nothing in the repo can regenerate a listener's
    observation.

The same hazard in `scripts/make_demo_audio.py` already cost a real edit — a
pre-registered prediction's verbatim listener response, its scoring and its
verdict, erased minutes after being written, and erased SILENTLY, because a
regenerated document looks exactly as correct as the one it replaced. That is
SPEC Appendix E.5's family ("a guard whose failure mode is silence") pointed at
the project's own kit. It was closed in `aea58e2`; this suite closes the twin.

What is asserted here is BOTH halves of the guard, because only the pair is
evidence:

  * it REFUSES to overwrite a document a human has touched, and the original
    bytes are still on disk afterwards — not merely that it returned "skipped";
  * it still writes when the document is absent, or is its own untouched output,
    or has merely been `touch`ed. A guard that refuses everything is not a guard,
    it is an outage, and it trains the next person to pass --force-docs by
    reflex.

`BothGuardsAgree` runs this guard and the sibling's through one scenario matrix
and requires identical answers. Two different guards for one hazard is worse than
one, and the way that happens is drift, not design.

Every destructive check runs against a COPY of the real listening directory. A
test that exercised the destructive path against the live tree would be precisely
the footgun being fixed — `tests/test_demo.py` runs the SIBLING generator with
`--force` against the live tree on every `make test`, which is how a green test
suite became the delivery mechanism the first time.
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
# Makes `deadzone`, `scripts` and `demos` importable when this file is run
# directly (`python tests/test_make_audio_sets.py`) with no install step.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# -------------------------------------------------------------------------

import ast
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import make_audio_sets as mas

REPO = Path(_REPO_ROOT)
PY = str(Path(sys.executable))
LISTEN = REPO / "results" / "audio" / "listen"
DOC_NAME = "WHAT_TO_LISTEN_FOR.md"


def _needs_assets(case: unittest.TestCase) -> None:
    """Skip unless the DSP inputs the real generator reads are present."""
    if not (REPO / "data" / "recordings").is_dir():
        case.skipTest("needs data/recordings to run the generator")
    if not (REPO / "recording_manifest.csv").is_file():
        case.skipTest("needs recording_manifest.csv to run the generator")
    if not LISTEN.is_dir():
        case.skipTest(f"{LISTEN} not present")


def _sandbox(td: str) -> Path:
    """A repo-shaped scratch tree: real inputs symlinked, listen/ COPIED.

    Copied, not linked, because this is the directory the test is allowed to let
    the generator write into. Everything else is read-only to the run.
    """
    sb = Path(td)
    (sb / "results" / "audio").mkdir(parents=True)
    for name in ("data", "recording_manifest.csv", "deadzone", "scripts"):
        os.symlink(REPO / name, sb / name)
    dz = REPO / "results" / "dead_zones.csv"
    if dz.is_file():                       # optional: adds the DEADZONE_* wavs
        os.symlink(dz, sb / "results" / "dead_zones.csv")
    shutil.copytree(LISTEN, sb / "results" / "audio" / "listen")
    return sb


class TheListeningInstructionsGuard(unittest.TestCase):
    """
    `write_doc()` in its four states. Every refusal is paired with the write it
    must still allow, so each test is pinned to the violation rather than to some
    incidental property of the fixture.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="deadzone-listenguard-"))
        self.doc = self.tmp / "DOC.md"
        self._saved = mas.DOC_HASHES
        mas.DOC_HASHES = self.tmp / "generated_docs.json"

    def tearDown(self):
        mas.DOC_HASHES = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- the writes it must still perform (the negative controls) ----------

    def test_it_writes_a_document_that_is_not_there(self):
        self.assertEqual(mas.write_doc(self.doc, "generated v1"), "written")
        self.assertEqual(self.doc.read_text(), "generated v1")

    def test_it_rewrites_its_own_untouched_output(self):
        """
        The alarm-fatigue control. The ladder labels and the DEADZONE_* filenames
        this document describes change whenever the grid does, so a guard that
        also refused its OWN output would make --force-docs the habitual
        invocation and the protection would be decoration inside a week.
        """
        mas.write_doc(self.doc, "generated v1")
        self.assertEqual(mas.write_doc(self.doc, "generated v2"), "written")
        self.assertEqual(self.doc.read_text(), "generated v2")

    def test_a_touch_is_not_an_edit(self):
        """
        mtime-only change: same bytes, new timestamp. That is what a checkout, a
        `cp` without `-p`, a zip round-trip or a restore does to a kit that gets
        handed around, and none of them is an edit. An mtime-based guard would
        refuse every one of them and be abandoned within a week.
        """
        mas.write_doc(self.doc, "generated v1")
        before = self.doc.read_text()
        os.utime(self.doc, (0, 0))
        self.assertEqual(self.doc.stat().st_mtime, 0)

        self.assertEqual(mas.doc_status(self.doc), mas.GENERATED)
        self.assertEqual(mas.write_doc(self.doc, "generated v2"), "written")
        self.assertNotEqual(self.doc.read_text(), before)

    # --- the refusal --------------------------------------------------------

    def test_it_refuses_to_overwrite_a_hand_edited_document(self):
        mas.write_doc(self.doc, "generated v1")
        self.doc.write_text("generated v1\n\nHEARD: 03_noise_only is intelligible.")
        edited = self.doc.read_text()

        self.assertEqual(mas.write_doc(self.doc, "generated v2"), "skipped")
        # The bytes, not the return value. A guard that reports "skipped" and
        # writes anyway is the same defect wearing a label.
        self.assertEqual(self.doc.read_text(), edited)
        self.assertIn("HEARD: 03_noise_only is intelligible.", self.doc.read_text())

    def test_an_unrecorded_document_is_refused_not_adopted(self):
        """
        The degenerate input, and the state the repo is actually in: the document
        already on disk, hand-annotated, and no sidecar yet. SPEC E.5's rule is to
        ask what a guard returns for the DEGENERATE input — if "no record" meant
        "safe to overwrite", the guard would be wide open on precisely the run
        that matters most, its first.
        """
        self.doc.write_text("hand-written, provenance unknown")
        self.assertFalse(mas.DOC_HASHES.exists())

        self.assertEqual(mas.write_doc(self.doc, "template output"), "skipped")
        self.assertEqual(self.doc.read_text(), "hand-written, provenance unknown")

        # Control: the same file, once the generator has a record of writing it,
        # is rewritten. So the refusal is caused by the missing record and not by
        # anything else about this fixture.
        mas.write_doc(self.doc, "template output", force_docs=True)
        self.assertEqual(mas.write_doc(self.doc, "template v2"), "written")

    def test_the_sidecar_is_never_seeded_from_the_file_on_disk(self):
        """
        The decision that makes the guard protect rather than deliver. Recording
        today's bytes as "what the generator last wrote" would certify hand-written
        text as generator-owned, and the next default run would erase it — the
        guard would be the delivery mechanism for the bug it exists to stop.
        Asserted as behaviour: merely ASKING the status of an unrecorded file must
        not create a record for it.
        """
        self.doc.write_text("a listening note nobody has a copy of")
        self.assertEqual(mas.doc_status(self.doc), mas.AUTHORED)
        self.assertEqual(mas.load_doc_hashes(), {})
        self.assertFalse(mas.DOC_HASHES.exists())

        # ...and it is still refused afterwards, i.e. the query did not adopt it.
        self.assertEqual(mas.write_doc(self.doc, "template output"), "skipped")
        self.assertEqual(self.doc.read_text(), "a listening note nobody has a copy of")

    def test_a_corrupt_or_absent_sidecar_fails_closed(self):
        mas.write_doc(self.doc, "generated v1")
        good = mas.DOC_HASHES.read_text()

        for label, broken in (("corrupt", "not json {{{"),
                              ("empty", ""),
                              ("wrong shape", json.dumps({"sha256": "a string"}))):
            with self.subTest(sidecar=label):
                mas.DOC_HASHES.write_text(broken)
                self.assertEqual(mas.write_doc(self.doc, "template output"),
                                 "skipped", f"a {label} sidecar failed OPEN")
                self.assertEqual(self.doc.read_text(), "generated v1")

        mas.DOC_HASHES.unlink()
        self.assertEqual(mas.write_doc(self.doc, "template output"), "skipped")

        # Control: restore the real sidecar and the very same write goes through.
        mas.DOC_HASHES.write_text(good)
        self.assertEqual(mas.write_doc(self.doc, "template output"), "written")

    def test_it_compares_content_and_not_mtime(self):
        """
        The other direction from `test_a_touch_is_not_an_edit`: the content is
        edited and the mtime is then rewound to the moment the generator wrote the
        file. An mtime-based guard sees nothing and destroys the edit.
        """
        mas.write_doc(self.doc, "generated v1")
        stamp = self.doc.stat().st_mtime
        self.doc.write_text("generated v1 + a hand-written listening note")
        os.utime(self.doc, (stamp, stamp))
        self.assertEqual(self.doc.stat().st_mtime, stamp)

        self.assertEqual(mas.write_doc(self.doc, "generated v2"), "skipped")
        self.assertIn("hand-written listening note", self.doc.read_text())

    def test_write_doc_protects_by_default(self):
        """`force_docs` defaults to False, so a caller that forgets the keyword
        gets the protective behaviour rather than the destructive one."""
        self.doc.write_text("hand-written")
        self.assertEqual(mas.write_doc(self.doc, "template"), "skipped")  # no kwarg
        self.assertEqual(self.doc.read_text(), "hand-written")

    def test_the_refusal_names_the_file_and_the_exact_override(self):
        """A refusal nobody can act on is a warning, and warnings get ignored."""
        msg = mas._refusal(self.doc)
        self.assertIn(str(self.doc), msg)
        self.assertIn("REFUSING", msg)
        self.assertIn("UNCHANGED", msg)
        self.assertIn("does NOT carry this build's content", msg)
        self.assertIn("scripts/make_audio_sets.py --force-docs", msg)
        # It must name THIS script's override. Pointing at the sibling's flag
        # would be an instruction that runs, does nothing to this file, and
        # teaches the reader the guard is broken.
        self.assertNotIn("make_demo_audio.py --force-docs", msg)

    # --- the override, which must still work --------------------------------

    def test_force_docs_overwrites_and_the_old_text_is_recoverable(self):
        """
        The set has to stay rebuildable from scratch, so the override is real. It
        also takes a copy first: the one irreversible operation in this script
        should not be the one with no undo.
        """
        self.doc.write_text("the only copy of a listening result")
        self.assertEqual(mas.write_doc(self.doc, "fresh template",
                                       force_docs=True), "written")
        self.assertEqual(self.doc.read_text(), "fresh template")

        backups = list(self.tmp.glob("DOC.superseded-*.md"))
        self.assertEqual(len(backups), 1,
                         f"no backup was taken: {list(self.tmp.iterdir())}")
        self.assertEqual(backups[0].read_text(),
                         "the only copy of a listening result")

    def test_doc_status_reports_the_three_states(self):
        self.assertEqual(mas.doc_status(self.doc), mas.ABSENT)
        mas.write_doc(self.doc, "generated v1")
        self.assertEqual(mas.doc_status(self.doc), mas.GENERATED)
        self.doc.write_text("edited")
        self.assertEqual(mas.doc_status(self.doc), mas.AUTHORED)

    def test_the_sidecar_records_the_hash_of_what_was_written(self):
        mas.write_doc(self.doc, "generated v1")
        rec = json.loads(mas.DOC_HASHES.read_text())
        want = hashlib.sha256(b"generated v1").hexdigest()
        self.assertEqual(rec["sha256"][self.doc.as_posix()], want)
        self.assertEqual(rec["written_by"], "scripts/make_audio_sets.py")


class BothGuardsAgree(unittest.TestCase):
    """
    One hazard, one mechanism. This guard and `scripts/make_demo_audio.py`'s are
    separate code — they must be, because `make_demo_audio` imports LADDER and
    LISTEN_CLIP from `make_audio_sets`, so importing back would cycle, and because
    a refusal has to name the flag of the script the user actually ran. Separate
    code is how two guards for one hazard start answering differently, which is
    strictly worse than having one. So the two are driven through the same
    scenario matrix and required to agree at every step.
    """

    def setUp(self):
        from scripts import make_demo_audio as mda      # noqa: E402  (lazy: heavy)
        self.mda = mda
        for mod in (mas, mda):
            self.assertTrue(
                hasattr(mod, "write_doc") and hasattr(mod, "doc_status"),
                f"{mod.__name__} lost its authored-document guard — the hazard is "
                f"back in that script, and the two are no longer one mechanism")
        self.tmp = Path(tempfile.mkdtemp(prefix="deadzone-guardparity-"))
        self._saved = (mas.DOC_HASHES, mda.DOC_HASHES)
        mas.DOC_HASHES = self.tmp / "a" / "generated_docs.json"
        mda.DOC_HASHES = self.tmp / "b" / "generated_docs.json"

    def tearDown(self):
        mas.DOC_HASHES, self.mda.DOC_HASHES = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, m):
        """Drive one guard through the scenario matrix; return an answer sheet.

        The answers are (status, result-of-write, bytes-on-disk) at each step, so
        parity is checked on what actually happened to the file and not only on a
        returned label.
        """
        doc = m.DOC_HASHES.parent / "DOC.md"
        doc.parent.mkdir(parents=True, exist_ok=True)
        out = {}

        out["absent"] = (m.doc_status(doc), m.write_doc(doc, "v1"), doc.read_text())
        out["own output"] = (m.doc_status(doc), m.write_doc(doc, "v2"),
                             doc.read_text())

        doc.write_text("v2\n\nHEARD: 03_noise_only is intelligible.")
        out["hand-edited"] = (m.doc_status(doc), m.write_doc(doc, "v3"),
                              doc.read_text())

        m.DOC_HASHES.write_text("not json {{{")
        out["corrupt sidecar"] = (m.doc_status(doc), m.write_doc(doc, "v4"),
                                  doc.read_text())

        m.DOC_HASHES.unlink()
        out["absent sidecar"] = (m.doc_status(doc), m.write_doc(doc, "v5"),
                                 doc.read_text())

        out["forced"] = (m.write_doc(doc, "v6", force_docs=True), doc.read_text(),
                         len(list(doc.parent.glob("DOC.superseded-*.md"))))

        os.utime(doc, (0, 0))                      # a touch, not an edit
        out["touched only"] = (m.doc_status(doc), m.write_doc(doc, "v7"),
                               doc.read_text())
        return out

    def test_the_two_guards_answer_identically(self):
        mine, sibling = self._run(mas), self._run(self.mda)
        for step in mine:
            with self.subTest(step=step):
                self.assertEqual(mine[step], sibling[step],
                                 f"the two guards disagree on {step!r} — they have "
                                 f"drifted into two different mechanisms for one "
                                 f"hazard, which is worse than having one")

    def test_the_agreed_answers_are_the_protective_ones(self):
        """
        The control for the parity test above, which would also pass if BOTH
        guards were no-ops. This pins WHAT they agree on.
        """
        got = self._run(mas)
        self.assertEqual(got["absent"][1], "written")
        self.assertEqual(got["own output"][1], "written")
        self.assertEqual(got["hand-edited"][:2], (mas.AUTHORED, "skipped"))
        self.assertIn("HEARD:", got["hand-edited"][2])
        self.assertEqual(got["corrupt sidecar"][1], "skipped")
        self.assertEqual(got["absent sidecar"][1], "skipped")
        self.assertEqual(got["forced"][0], "written")
        self.assertEqual(got["forced"][2], 1, "the override kept no backup")
        self.assertEqual(got["touched only"][:2], (mas.GENERATED, "written"))

    def test_both_use_a_content_hash_sidecar_and_neither_seeds_it(self):
        """The two properties the mechanism IS: hash-keyed record, never seeded."""
        mda = self.mda
        for m in (mas, mda):
            with self.subTest(module=m.__name__):
                p = m.DOC_HASHES.parent / "SEED.md"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("pre-existing hand-written text")
                self.assertEqual(m.doc_status(p), m.AUTHORED)
                self.assertEqual(m.write_doc(p, "template"), "skipped")
                self.assertEqual(p.read_text(), "pre-existing hand-written text")

                m.write_doc(p, "template", force_docs=True)
                rec = json.loads(m.DOC_HASHES.read_text())["sha256"]
                self.assertEqual(rec[p.as_posix()],
                                 hashlib.sha256(b"template").hexdigest())


class NoWriterBypassesTheGuard(unittest.TestCase):
    """
    Structural, because the fix is only worth what the NEXT document is worth.
    Adding a second markdown file and reaching for `.write_text()` out of habit
    reintroduces the whole defect for that one file, and nothing would say so.
    """

    SRC = REPO / "scripts" / "make_audio_sets.py"

    def _write_text_calls(self):
        tree = ast.parse(self.SRC.read_text())
        out = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "write_text"):
                    out.append((fn.name, node))
        return out

    def test_every_write_text_call_goes_through_write_doc(self):
        allowed = {"write_doc",        # the guard itself
                   "_save_doc_hash",   # the sidecar, which is not a document
                   "make_sweeps"}      # index.json: machine data, regenerable
        offenders = [f"{name}() at line {node.lineno}"
                     for name, node in self._write_text_calls()
                     if name not in allowed]
        self.assertEqual(offenders, [],
                         "these write without the authored-document guard, so a "
                         "hand-edited file would be destroyed silently — route "
                         "them through write_doc()")

    def test_no_markdown_path_is_written_outside_the_guard(self):
        """
        The allowlist above is by FUNCTION, so a document added inside an already
        allowed function would slip through. This one is by TARGET: any
        `.write_text` whose path expression mentions a `.md` file must be the
        guard. It is the narrower check and it survives the allowlist growing.
        """
        offenders = []
        for name, node in self._write_text_calls():
            if name == "write_doc":
                continue
            if ".md" in ast.unparse(node.func.value):
                offenders.append(f"{name}() at line {node.lineno}")
        self.assertEqual(offenders, [],
                         "a markdown document is written without the guard")

    def test_the_census_covers_every_document_the_build_writes(self):
        """`DOCS` is what the run summary reports on. A document that is guarded
        but absent from the census is protected invisibly, which is how a refusal
        gets missed."""
        self.assertIn(DOC_NAME, {Path(p).name for p in mas.DOCS})
        # ...and the census must not claim to cover regenerable output, or the
        # "0 written, 1 PROTECTED" line stops meaning anything.
        self.assertTrue(all(str(p).endswith(".md") for p in mas.DOCS), mas.DOCS)


class NoFlagUnlocksDocumentsByAccident(unittest.TestCase):
    """
    The sibling's actual failure mode, which this script must not rebuild through
    a different door. There, `--force` rebuilt the wavs, `tests/test_demo.py`
    passed it against the live tree on every `make test`, and so a GREEN TEST
    SUITE deleted a hand-written record. Here `--listen` and `--sweep` are
    selectors; the override has to be asked for by name.
    """

    def test_make_listening_set_defaults_to_protecting(self):
        sig = inspect.signature(mas.make_listening_set)
        self.assertIs(sig.parameters["force_docs"].default, False)
        self.assertEqual(sig.parameters["force_docs"].kind,
                         inspect.Parameter.KEYWORD_ONLY,
                         "keyword-only, so no positional caller can flip it by "
                         "accident when the signature grows")

    def test_the_help_offers_force_docs_and_disclaims_listen(self):
        r = subprocess.run([PY, "scripts/make_audio_sets.py", "--help"],
                           cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("--force-docs", r.stdout)
        self.assertIn("does NOT overwrite hand-edited documents", r.stdout,
                      "--listen must say out loud that it is not the override")

    def test_a_listen_run_does_not_touch_an_edited_document(self):
        """
        End to end, against a COPY: the real generator, at the flag a future
        `make listen-prep` would use, leaves a hand-edited document byte-identical
        while still rebuilding every wav.
        """
        _needs_assets(self)
        with tempfile.TemporaryDirectory(prefix="deadzone-listen-sandbox-") as td:
            sb = _sandbox(td)
            doc = sb / "results" / "audio" / "listen" / DOC_NAME
            doc.write_text(doc.read_text() +
                           "\n\n## HEARD 2026-08-05\n03_noise_only is intelligible.\n")
            before = doc.read_bytes()
            # Truncate the wavs the ladder is guaranteed to rewrite, so "the
            # document is unchanged" cannot pass by the generator doing nothing.
            # The DEADZONE_* files are deliberately NOT in this set: their names
            # come from results/dead_zones.csv, so stale ones from an earlier
            # dead-zone list are never rewritten and would fail a control that
            # has nothing to do with the guard.
            ladder = ["00_RAW_original.wav"] + [f"{label}.wav"
                                                for label, _ in mas.LADDER]
            for name in ladder:
                (doc.parent / name).write_bytes(b"")

            r = subprocess.run([PY, "scripts/make_audio_sets.py", "--listen"],
                               cwd=sb, capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])

            self.assertEqual(doc.read_bytes(), before,
                             "a --listen run rewrote a hand-edited document — this "
                             "is the exact regression the guard exists to stop")
            # The controls that make the assertion above mean something: the run
            # really did its work and really did say so. A generator that crashed
            # early would leave the document untouched too, and pass a test that
            # only checked for damage.
            self.assertIn("REFUSING to overwrite", r.stdout)
            self.assertIn("PROTECTED", r.stdout)
            for name in ladder:
                self.assertGreater((doc.parent / name).stat().st_size, 0,
                                   f"{name} was not regenerated — the audio half "
                                   f"of the build did not run")

    def test_force_docs_can_still_rebuild_the_set_from_scratch(self):
        """The override is not decorative: without it the document could never be
        regenerated after the first hand edit."""
        _needs_assets(self)
        with tempfile.TemporaryDirectory(prefix="deadzone-listen-force-") as td:
            sb = _sandbox(td)
            doc = sb / "results" / "audio" / "listen" / DOC_NAME

            # PLANT the hand edit rather than relying on the live document
            # already differing from the template. It used to, because the live
            # copy was stale; regenerating it made the two agree and this test
            # failed -- asserting "force changed the bytes" when there was
            # nothing left to change. A test whose premise is that a file on
            # disk has drifted stops testing anything the moment someone fixes
            # the drift, and it fails in the direction that looks like a
            # regression. The planted marker makes the overwrite observable
            # whatever the live document happens to say.
            before = b"# HAND-EDITED\n\nnotes from an actual listening pass\n"
            doc.write_bytes(before)
            pre_existing = {p.name for p in
                            doc.parent.glob("WHAT_TO_LISTEN_FOR.superseded-*.md")}

            r = subprocess.run([PY, "scripts/make_audio_sets.py",
                                "--listen", "--force-docs"],
                               cwd=sb, capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])

            self.assertNotEqual(doc.read_bytes(), before,
                                "--force-docs did not overwrite, so the set is not "
                                "rebuildable")
            self.assertNotIn(b"HAND-EDITED", doc.read_bytes(),
                             "--force-docs left the planted edit in place, so the "
                             "overwrite did not actually happen")
            # Count backups THIS RUN made, not backups in the directory. The
            # sandbox is a copy of the live tree, so any .superseded-* file a
            # human left lying around there was being counted as one of ours --
            # a second real failure this test produced today, and one that
            # accuses the generator of a bug in someone else's housekeeping.
            backups = [p for p in doc.parent.glob("WHAT_TO_LISTEN_FOR.superseded-*.md")
                       if p.name not in pre_existing]
            self.assertEqual(len(backups), 1,
                             "--force-docs overwrote without keeping a copy")
            self.assertEqual(backups[0].read_bytes(), before)

            # And the second run is quiet: the document is now the generator's own
            # output again, so nothing is refused and nothing is backed up.
            r2 = subprocess.run([PY, "scripts/make_audio_sets.py", "--listen"],
                                cwd=sb, capture_output=True, text=True, timeout=600)
            self.assertEqual(r2.returncode, 0, r2.stderr[-3000:])
            self.assertNotIn("REFUSING to overwrite", r2.stdout)
            self.assertEqual(
                len(list(doc.parent.glob("WHAT_TO_LISTEN_FOR.superseded-*.md"))), 1)


class TheLiveDocumentIsProtectedRightNow(unittest.TestCase):
    """
    Not a property of the code but of the repo as it stands: the document on disk
    today predates the guard and has no sidecar record, so it must read as
    unknown provenance. If this ever fails, someone seeded the sidecar from the
    file — which is the one move that turns the guard into the delivery mechanism.
    """

    def test_the_document_on_disk_is_not_overwritable_by_default(self):
        if not (LISTEN / DOC_NAME).is_file():
            self.skipTest(f"{LISTEN / DOC_NAME} not present")
        self.assertEqual(mas.doc_status(LISTEN / DOC_NAME), mas.AUTHORED,
                         "the live listening instructions are marked as this "
                         "generator's own output — a default run would erase any "
                         "listening notes in them")


if __name__ == "__main__":
    unittest.main(verbosity=2)
