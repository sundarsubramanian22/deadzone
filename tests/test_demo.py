#!/usr/bin/env python3
"""
test_demo.py — the demo kit is offline, key-free, and says the right numbers.

    ./.venv/bin/python tests/test_demo.py

The demo is the one artifact that fails in front of an audience, so what is
asserted here is exactly what stage failure looks like:

  * the default path must not require DEEPGRAM_API_KEY  (the subprocess is
    launched with the key STRIPPED from the environment, so a machine that has
    one cannot mask a bug on a machine that does not);
  * it must not require the network — asserted structurally, by importing the
    module with `socket` monkeypatched to explode on any connection attempt;
  * it must print the confidence number AND a non-empty diff — a demo that runs
    but shows nothing is worse than one that crashes;
  * the numbers on screen must be the MEASURED ones from results/master.csv, not
    something the demo script rounded, restated, or invented;
  * the exemplar must be a real dead zone: high confidence AND high WER. If a
    future re-run of the grid moves the dead zones, this test fails rather than
    letting the demo narrate a finding that is no longer true.
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

import csv
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos import demo_break

REPO = Path(_REPO_ROOT)
PY = str(Path(sys.executable))
CACHE = REPO / "results" / "demo" / "demo_cache.json"


def run_demo(*args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    """Run demo_break.py in a child process with the API key REMOVED."""
    env = dict(os.environ)
    env.pop("DEEPGRAM_API_KEY", None)
    env["TERM"] = "dumb"                      # force the plain-text branch
    return subprocess.run([PY, "demos/demo_break.py", *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=timeout)


def cache() -> dict:
    if not CACHE.is_file():
        r = run_demo("--prepare")
        if r.returncode != 0:
            raise unittest.SkipTest(f"could not bake the demo cache:\n{r.stderr[-2000:]}")
    return json.loads(CACHE.read_text())


class OfflinePath(unittest.TestCase):
    """The default invocation, with no key in the environment at all."""

    @classmethod
    def setUpClass(cls):
        cache()
        cls.r = run_demo("--offline", "--no-audio", "--pause", "0")

    def test_exits_zero_without_an_api_key(self):
        self.assertEqual(self.r.returncode, 0,
                         f"demo_break exited {self.r.returncode}\n"
                         f"STDERR:\n{self.r.stderr[-3000:]}")

    def test_never_mentions_a_key(self):
        blob = self.r.stdout + self.r.stderr
        self.assertNotIn("DEEPGRAM_API_KEY=", blob)
        for tok in ("Token ", "sk-", "Authorization"):
            self.assertNotIn(tok, blob, f"{tok!r} leaked into demo output")

    def test_prints_the_confidence_number(self):
        out = self.r.stdout
        self.assertIn("mean word confidence", out)
        conf = cache()["clips"][cache()["default_clip"]]["deadzone"]["mean_conf"]
        self.assertIn(f"{conf:.3f}", out,
                      "the dead-zone mean confidence is not on screen")

    def test_prints_a_non_empty_diff(self):
        out = self.r.stdout
        self.assertIn("ref", out)
        self.assertIn("hyp", out)
        entry = cache()["clips"][cache()["default_clip"]]
        ref_words = entry["ref"].split()
        hyp_words = entry["deadzone"]["transcript"].split()
        self.assertTrue(any(w in out for w in ref_words), "no reference words rendered")
        self.assertTrue(any(w in out for w in hyp_words), "no hypothesis words rendered")
        # the diff must actually SHOW a disagreement, not two identical lines
        changed = [w for w in hyp_words if w not in ref_words]
        self.assertTrue(changed, "the exemplar has no substituted words to show")
        self.assertTrue(any(w in out for w in changed),
                        "the words the model got wrong are not on screen")

    def test_shows_both_stages_and_the_delta(self):
        out = self.r.stdout
        self.assertIn("RAW RECORDING", out)
        self.assertIn("DEAD ZONE", out)
        self.assertIn("WHAT JUST HAPPENED", out)

    def test_plain_text_fallback_has_no_ansi(self):
        self.assertNotIn("\033[", self.r.stdout,
                         "ANSI escapes leaked into a non-tty run; piping to a "
                         "file would be unreadable")

    def test_marker_row_carries_the_edit_types_without_colour(self):
        """Colour is not the only channel: the marker row must encode the ops."""
        dz = cache()["clips"][cache()["default_clip"]]["deadzone"]
        ops = {e[0] for e in dz["edits"]}
        out = self.r.stdout
        if "sub" in ops:
            self.assertRegex(out, r"\^|\d\.\d\d",
                             "substitutions have no visible marker in plain text")

    def test_list_clips_and_check_both_exit_zero(self):
        for flag in ("--list-clips", "--check"):
            r = run_demo(flag, "--no-color")
            self.assertEqual(r.returncode, 0, f"{flag} exited {r.returncode}: {r.stderr[-1500:]}")


class NoNetwork(unittest.TestCase):
    """Structural proof, not a promise in a docstring."""

    def test_import_and_render_with_sockets_disabled(self):
        script = (
            "import socket, sys\n"
            "class Boom(Exception): pass\n"
            "def die(*a, **k): raise Boom('the demo tried to open a socket')\n"
            "socket.socket = die\n"
            "socket.create_connection = die\n"
            "from demos import demo_break\n"
            "c = demo_break.load_cache(auto_build=False, verbose=False)\n"
            "ink = demo_break.Ink(False)\n"
            "cid = c['default_clip']\n"
            "demo_break.run(c, cid, ink, 96, False, 0.0, False)\n"
            "print('NO_SOCKET_OK')\n"
        )
        env = dict(os.environ)
        env.pop("DEEPGRAM_API_KEY", None)
        r = subprocess.run([PY, "-c", script], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("NO_SOCKET_OK", r.stdout)


class CachedNumbersAreTheMeasuredOnes(unittest.TestCase):
    """
    The cache is a convenience, never a second source of truth. Every number the
    demo prints has to be traceable to a row in the master table.
    """

    @classmethod
    def setUpClass(cls):
        cls.cache = cache()

    def test_condition_is_a_measured_dead_zone(self):
        names = {z["name"] for z in self.cache["dead_zones"]}
        self.assertIn(self.cache["condition"]["name"], names)

    def test_dead_zone_rows_match_dead_zones_csv(self):
        rows = {r["condition_name"]: r
                for r in csv.DictReader(open(REPO / "results" / "dead_zones.csv"))}
        for z in self.cache["dead_zones"]:
            self.assertIn(z["name"], rows)
            self.assertAlmostEqual(z["wer"], float(rows[z["name"]]["wer"]), places=9)
            self.assertAlmostEqual(z["mean_conf"], float(rows[z["name"]]["mean_conf"]),
                                   places=9)

    def test_per_clip_facts_match_the_master_table(self):
        csv.field_size_limit(10**9)
        want = set(self.cache["clips"])
        cond = self.cache["condition"]["name"]
        found = {}
        with open(REPO / "results" / "master.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                if (row["model"] == self.cache["model"]
                        and row["condition_name"] == cond
                        and row["clip_id"] in want):
                    found[row["clip_id"]] = row
        self.assertEqual(set(found), want, "cached exemplars are not all in master.csv")
        for cid, row in found.items():
            dz = self.cache["clips"][cid]["deadzone"]
            self.assertEqual(dz["transcript"], row["transcript"])
            self.assertAlmostEqual(dz["wer"], float(row["wer"]), places=9)
            self.assertAlmostEqual(dz["mean_conf"], float(row["mean_conf"]), places=9)
            self.assertEqual(dz["edits"], [list(e) for e in json.loads(row["edits"])])

    def test_the_exemplar_is_actually_confidently_wrong(self):
        """If this fails, the demo is narrating a finding that no longer holds."""
        cid = self.cache["default_clip"]
        clean = self.cache["clips"][cid]["clean"]
        dz = self.cache["clips"][cid]["deadzone"]
        self.assertLessEqual(clean["wer"], 0.05, "the control clip is not clean")
        self.assertGreaterEqual(dz["wer"], 0.25, "the dead-zone clip barely fails")
        self.assertGreaterEqual(dz["mean_conf"], 0.75,
                                "the model was NOT confident — that is the opposite "
                                "of the point being demonstrated")

    def test_word_confidences_align_with_the_hypothesis_words(self):
        """
        SPEC B.5(3): edits come from normalized tokens, confidences from raw ones.
        Where they disagree the renderer must DROP the annotation, never zip it.
        """
        for cid, e in self.cache["clips"].items():
            dz = e["deadzone"]
            cols = demo_break.diff_columns(dz["edits"])
            hyp = [c for c in cols if c[0] in ("match", "sub", "ins")]
            if dz["word_confidences"]:
                lines = demo_break.render_diff(dz["edits"], 70, demo_break.Ink(False),
                                               dz["word_confidences"])
                self.assertTrue(lines)
                if len(hyp) != len(dz["word_confidences"]):
                    # mismatched: no per-word number may appear
                    joined = "\n".join(lines)
                    self.assertNotRegex(
                        joined, r"^\s+0\.\d\d\s",
                        f"{cid}: per-word confidence rendered despite a token mismatch")


class Artifacts(unittest.TestCase):

    def test_audio_files_exist_and_are_the_same_length(self):
        c = cache()
        import soundfile as sf
        for cid, e in c["clips"].items():
            a, b = Path(e["audio_clean"]), Path(e["audio_deadzone"])
            self.assertTrue(a.is_file(), f"{a} missing")
            self.assertTrue(b.is_file(), f"{b} missing")
            ia, ib = sf.info(str(a)), sf.info(str(b))
            self.assertEqual(ia.frames, ib.frames,
                             f"{cid}: degraded clip is not the same length as the "
                             f"original — a length change would be an onset artifact")
            self.assertEqual(ia.samplerate, c["fs"])

    def test_missing_player_does_not_raise(self):
        """A demo machine without afplay, or a missing wav, must still complete."""
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demo_break.play("results/demo/audio/does_not_exist.wav",
                            demo_break.Ink(False), enabled=True)
        self.assertIn("audio missing", buf.getvalue(),
                      "a missing wav must be reported, not swallowed")

    def test_makefile_declares_the_demo_targets(self):
        mk = (REPO / "Makefile").read_text()
        for target in ("help:", "test:", "demo-break:", "demo-al:", "dashboard:", "demo:"):
            self.assertIn(f"\n{target}", "\n" + mk, f"Makefile has no {target} target")

    def test_readme_has_the_three_command_section(self):
        rd = (REPO / "README.md").read_text()
        self.assertIn("Run the demo in three commands", rd)

    def test_requirements_lock_exists_and_pins_versions(self):
        p = REPO / "requirements.lock.txt"
        self.assertTrue(p.is_file(), "run `make lock`")
        lines = [l for l in p.read_text().splitlines() if l.strip()
                 and not l.startswith("#")]
        self.assertGreater(len(lines), 10)
        self.assertTrue(all("==" in l or " @ " in l for l in lines),
                        "requirements.lock.txt has unpinned entries")

    def test_dashboard_is_self_contained(self):
        """It has to open from file:// with wifi off — no CDN, no external assets."""
        html = (REPO / "dashboard" / "deadzone.html").read_text()
        import re
        for m in re.finditer(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", html):
            url = m.group(1)
            self.assertFalse(url.startswith(("http://", "https://", "//")),
                             f"dashboard loads an external asset: {url}")


class ActiveLearningDemo(unittest.TestCase):

    def test_demo_al_runs_offline_and_exits_zero(self):
        env = dict(os.environ)
        env.pop("DEEPGRAM_API_KEY", None)
        env["TERM"] = "dumb"
        r = subprocess.run([PY, "demos/demo_al.py", "--fast", "--no-anim", "--no-color",
                            "--hold", "0"],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("evaluations spent", r.stdout)
        self.assertIn("WHERE THE EVALUATIONS WENT", r.stdout)
        self.assertIn("no network", r.stdout)

    def test_it_labels_its_oracle_as_a_surrogate(self):
        """Provenance is not optional: this is a replication device, not a measurement."""
        src = (REPO / "demos" / "demo_al.py").read_text()
        self.assertIn("surrogate", src)
        self.assertIn("not a measurement", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
