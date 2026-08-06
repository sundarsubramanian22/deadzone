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
import filecmp
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from demos import demo_break
from scripts import make_demo_audio

REPO = Path(_REPO_ROOT)
PY = str(Path(sys.executable))
CACHE = REPO / "results" / "demo" / "demo_cache.json"
DEMO_AUDIO = REPO / "results" / "audio" / "demo" / "manifest.json"


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


def _num(v):
    """CSV cell -> float, or None when the column is blank (a NULL, not a zero)."""
    if v is None:
        return None
    v = v.strip()
    return float(v) if v else None


def demo_audio() -> dict:
    """
    The listening/demo manifest, self-healing like `cache()` above.

    Same convention on purpose: a test that fails with "run the prep step first"
    teaches whoever hits it to run prep by hand before demoing, and a demo that
    depends on someone remembering a setup step is a demo that breaks on stage.
    So the test runs prep itself, offline, with the API key stripped.
    """
    if not DEMO_AUDIO.is_file():
        env = dict(os.environ)
        env.pop("DEEPGRAM_API_KEY", None)
        r = subprocess.run([PY, "scripts/make_demo_audio.py"], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise unittest.SkipTest(
                "could not build the demo audio set (needs data/ + "
                f"results/master.csv):\n{r.stderr[-2000:]}")
    return json.loads(DEMO_AUDIO.read_text())


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
        """
        Every cached dead-zone number is traceable to a row of the CSV.

        `condition_name` is NOT a unique key: the table carries a `category`
        column (dead_zone / mute_zone / silence_driven) and a condition can be
        listed under two of them — a mute condition is reported at WER 1.00 as a
        mute row and at its measured WER as a dead-zone row. Keying a dict on the
        name alone silently keeps whichever row came last and then compares the
        cache against the wrong one. So match against ANY row bearing that name,
        which is what "traceable" actually means here.
        """
        rows: dict[str, list[dict]] = {}
        for r in csv.DictReader(open(REPO / "results" / "dead_zones.csv")):
            rows.setdefault(r["condition_name"], []).append(r)

        def close(a, b) -> bool:
            """NaN==NaN here on purpose: a mute condition's mean confidence is
            NULL (no words to be confident about), and NULL matching NULL is the
            correct outcome, not a comparison failure."""
            if a is None or b is None:
                return a is None and b is None
            if a != a or b != b:                                 # NaN
                return a != a and b != b
            return abs(float(a) - float(b)) < 1e-9

        for z in self.cache["dead_zones"]:
            self.assertIn(z["name"], rows)
            cands = rows[z["name"]]
            ok = [r for r in cands if close(z["wer"], _num(r.get("wer")))]
            self.assertTrue(ok, f"{z['name']}: cached WER {z['wer']} matches no row "
                                f"({[r.get('wer') for r in cands]})")
            self.assertTrue(
                any(close(z["mean_conf"], _num(r.get("mean_conf"))) for r in ok),
                f"{z['name']}: cached mean_conf {z['mean_conf']} matches no row with "
                f"that WER")

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

    def test_readme_tells_a_reader_how_to_run_the_demo(self):
        """The README must carry the demo entry points, and say they run offline.

        This used to assert one heading string ("Run the demo in three
        commands"). That pinned the WORDING, not the property: rewriting the
        section broke the test while the README still documented the demo
        perfectly, and the only available fix was to restore a sentence for the
        test's benefit. So the assertion is now the invariant a reader needs --
        the commands themselves, plus the offline claim that is the whole point
        of the kit -- and it stays true across any rewrite that keeps them.
        """
        rd = (REPO / "README.md").read_text()
        for cmd in ("make demo", "make demo-check", "make test"):
            self.assertIn(cmd, rd, f"README no longer documents `{cmd}`")
        self.assertRegex(
            rd, r"(?i)offline",
            "README no longer says the demo runs offline -- that claim is the "
            "reason the kit is built the way it is, and it is what a reader "
            "checks before turning wifi off")

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


class DemoAudioSet(unittest.TestCase):
    """
    `results/audio/demo/` — the listening exercise (scripts/make_demo_audio.py).

    This set makes CLAIMS out loud: that two conditions score the same, that
    three clips tie exactly, that ten of forty clips came back empty. Every one
    of those is re-derived here from `results/master.csv`, because a demo whose
    narration has drifted from its own study is the single worst thing that can
    happen in front of an audience — it is not a crash, so nothing tells you.
    """

    @classmethod
    def setUpClass(cls):
        cls.m = demo_audio()
        cls.dir = DEMO_AUDIO.parent
        csv.field_size_limit(10 ** 9)
        want = {cls.m["conditions"]["A"]["name"], cls.m["conditions"]["B"]["name"],
                cls.m["conditions"]["payoff"]["name"]}
        cls.rows = {}
        with open(REPO / "results" / "master.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["model"] == cls.m["model"] and row["condition_name"] in want:
                    cls.rows[(row["condition_name"], row["clip_id"])] = row

    # --- the files exist ---------------------------------------------------

    def test_every_manifest_file_exists(self):
        missing = [f for f in self.m["files"] if not (REPO / f).is_file()]
        self.assertEqual(missing, [], "run `python3 scripts/make_demo_audio.py --force`")

    def test_every_wav_named_in_a_document_exists(self):
        """A run-of-show that says 'play blind_04' and has no blind_04 is a
        stage failure, and markdown has no way to tell you."""
        docs = ["DEMO_SCRIPT.md", "KEY.md", "WHAT_TO_LISTEN_FOR.md",
                "PREREGISTERED_PREDICTION.md", "blind/BLIND_SHEET.md"]
        for rel in docs:
            p = self.dir / rel
            self.assertTrue(p.is_file(), f"{p} missing")
            text = p.read_text()
            for name in sorted(set(re.findall(r"[A-Za-z0-9_.\-]+\.wav", text))):
                if "*" in name:
                    continue
                # stimuli live in the demo dir; the docs also cite the RIR asset
                # files by name, and those live under data/.
                hits = list(self.dir.rglob(name)) or list((REPO / "data").rglob(name))
                self.assertTrue(hits, f"{rel} names {name}, which does not exist")

    def test_it_regenerates_with_no_key_and_no_outbound_socket(self):
        """
        Structural proof, not a promise in a docstring. `connect` is blocked
        rather than `socket.socket` replaced, because librosa's import chain
        pulls in `ssl`, which subclasses `socket.socket` at import time.
        """
        script = (
            "import sys, os\n"
            "os.environ.pop('DEEPGRAM_API_KEY', None)\n"
            "import librosa, soundfile\n"                 # warm the lazy imports
            "import socket\n"
            "class Boom(Exception): pass\n"
            "def die(*a, **k): raise Boom('the generator opened a socket')\n"
            "socket.socket.connect = die\n"
            "socket.socket.connect_ex = die\n"
            "socket.create_connection = die\n"
            "sys.argv = ['make_demo_audio.py', '--force']\n"
            "from scripts import make_demo_audio\n"
            "rc = make_demo_audio.main()\n"
            "print('NO_SOCKET_OK', rc)\n"
        )
        env = dict(os.environ)
        env.pop("DEEPGRAM_API_KEY", None)
        r = subprocess.run([PY, "-c", script], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("NO_SOCKET_OK 0", r.stdout)

    def test_check_mode_passes_and_reports_the_interval(self):
        r = subprocess.run([PY, "scripts/make_demo_audio.py", "--check"], cwd=REPO,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("spans zero", r.stdout)

    # --- the blind set is actually blind -----------------------------------

    def test_blind_copies_are_byte_identical_to_their_sources(self):
        for blind, work in self.m["blind_map"].items():
            b, w = self.dir / "blind" / blind, self.dir / work
            self.assertTrue(b.is_file() and w.is_file(), f"{blind} <- {work}")
            self.assertTrue(filecmp.cmp(b, w, shallow=False),
                            f"{blind} is not a copy of {work}")

    def test_blind_map_is_a_bijection_over_the_eight_stimuli(self):
        bm = self.m["blind_map"]
        self.assertEqual(len(bm), 8)
        self.assertEqual(len(set(bm.values())), 8, "two blind names share a source")
        self.assertEqual(sorted(bm), [f"blind_{i:02d}.wav" for i in range(1, 9)])

    def test_the_blind_directory_gives_nothing_away(self):
        """The whole exercise is void if the listener can read the answer."""
        # whole words only: "wer" is a substring of "answer", and a substring
        # match would make this test unmaintainable rather than strict.
        leaks = ("reverb", "reverberant", "babble", "rt60", "snr", "deadzone",
                 "dead zone", "codec", "opus", "rolloff", "wer", "nova",
                 "confidence", "transcript")
        for p in sorted((self.dir / "blind").iterdir()):
            self.assertNotRegex(p.name.lower(), r"reverb|babble|rt60|snr|payoff",
                                f"blind filename {p.name} names the condition")
            if p.suffix == ".md":
                low = p.read_text().lower()
                for tok in leaks:
                    self.assertNotRegex(low, rf"\b{re.escape(tok)}\b",
                                        f"{p.name} leaks {tok!r} to the listener")

    def test_no_pair_is_presented_with_the_same_arm_first_every_time(self):
        """
        Position would otherwise be perfectly confounded with condition: if the
        reverb arm always played first, "the second one was harder" could be an
        order effect and the exercise would not distinguish the two.
        """
        first_arm = []
        for p in self.m["pairs"]:
            a, b = p["A"]["blind"], p["B"]["blind"]
            first_arm.append("A" if min(a, b) == a else "B")
        self.assertEqual(len(set(first_arm)), 2,
                         f"every pair leads with the same arm: {first_arm}")

    # --- the numbers are the measured ones ---------------------------------

    def test_pair_clips_really_do_tie_and_really_are_wrong(self):
        A = self.m["conditions"]["A"]["name"]
        B = self.m["conditions"]["B"]["name"]
        for p in self.m["pairs"]:
            wa = float(self.rows[(A, p["clip_id"])]["wer"])
            wb = float(self.rows[(B, p["clip_id"])]["wer"])
            self.assertEqual(wa, wb, f"{p['clip_id']}: the arms no longer tie")
            self.assertGreater(wa, 0.0,
                               f"{p['clip_id']}: the model is equally RIGHT on both, "
                               f"which demonstrates nothing — that is the exact defect "
                               f"that retired the old u02 listening set")
            self.assertAlmostEqual(p["A"]["wer"], wa, places=9)
            self.assertAlmostEqual(p["B"]["wer"], wb, places=9)

    def test_per_clip_transcripts_match_the_master_table(self):
        A = self.m["conditions"]["A"]["name"]
        B = self.m["conditions"]["B"]["name"]
        for p in self.m["pairs"]:
            for arm, cond in (("A", A), ("B", B)):
                row = self.rows[(cond, p["clip_id"])]
                self.assertEqual(p[arm]["transcript"], row["transcript"])
                self.assertEqual(p[arm]["rir_key"], row["rir_key"])
                self.assertAlmostEqual(p[arm]["mean_conf"], float(row["mean_conf"]),
                                       places=9)

    def test_the_paired_result_recomputes_from_master_csv(self):
        got = make_demo_audio.paired_stats(self.rows)
        want = self.m["paired_result"]
        for k in ("n_clips", "mean_wer_A", "mean_wer_B", "paired_diff_A_minus_B",
                  "ci_lo", "ci_hi"):
            self.assertAlmostEqual(got[k], want[k], places=9,
                                   msg=f"{k} in the manifest is not what "
                                       f"master.csv now says")

    def test_the_interval_still_spans_zero(self):
        """
        The headline claim is 'statistically indistinguishable'. If a re-run of
        the grid separates the two conditions, this fails LOUDLY rather than
        letting the script keep saying it.
        """
        s = self.m["paired_result"]
        self.assertLess(s["ci_lo"], 0.0)
        self.assertGreater(s["ci_hi"], 0.0)
        self.assertTrue(s["spans_zero"])
        self.assertEqual(s["resample_unit"], "clip",
                         "resampling anything but clips throws the pairing away")

    def test_the_payoff_transcript_really_is_empty(self):
        pay = self.m["payoff"]
        row = self.rows[(self.m["conditions"]["payoff"]["name"], pay["clip_id"])]
        self.assertEqual(row["transcript"].strip(), "",
                         "the payoff clip is no longer an empty transcript")
        self.assertEqual(float(row["wer"]), 1.0)
        self.assertEqual(int(row["n_del"]), int(row["n_ref"]),
                         "an empty transcript must be all deletions")
        self.assertEqual(row["mean_conf"].strip(), "",
                         "mean confidence must be NULL, not 0.0 — 'no words to be "
                         "confident about' is the point being made")
        cond = self.m["conditions"]["payoff"]["name"]
        empty = sorted(c for (cn, c), r in self.rows.items()
                       if cn == cond and not r["transcript"].strip())
        self.assertEqual(empty, pay["empty_clips"])
        self.assertEqual(len(empty), pay["n_empty"])

    def test_both_isolating_conditions_carry_no_channel_degradation(self):
        """The pair claim is 'one degradation each'. Verify it, don't assert it."""
        for arm in ("A", "B"):
            c = self.m["conditions"][arm]
            self.assertEqual(c["codec"], "none", f"arm {arm} has a codec on it")
            self.assertEqual(c["mic_rolloff"], 0.0, f"arm {arm} has mic rolloff")
            self.assertEqual(c["noise_type"], self.m["conditions"]["B"]["noise_type"],
                             "the arms use different noise types, so noise TYPE is "
                             "confounded with the reverb/SNR contrast")

    # --- the audio itself ---------------------------------------------------

    def test_arms_are_the_same_length_and_rate(self):
        import soundfile as sf
        groups = [(p["A"]["file"], p["B"]["file"]) for p in self.m["pairs"]]
        groups.append((self.m["payoff"]["clean_file"],
                       self.m["payoff"]["deadzone_file"]))
        for a, b in groups:
            ia, ib = sf.info(str(REPO / a)), sf.info(str(REPO / b))
            self.assertEqual(ia.frames, ib.frames,
                             f"{a} and {b} differ in length — a length change "
                             f"between arms would be an onset artifact, not a "
                             f"condition effect")
            self.assertEqual(ia.samplerate, self.m["fs"])
            self.assertEqual(ib.samplerate, self.m["fs"])

    def test_the_degraded_audio_is_not_a_no_op(self):
        """A silent composer failure would leave a demo that plays the same clip
        twice and narrates a difference."""
        import soundfile as sf
        for p in self.m["pairs"]:
            a, _ = sf.read(str(REPO / p["A"]["file"]))
            b, _ = sf.read(str(REPO / p["B"]["file"]))
            self.assertGreater(float(np.abs(a - b).mean()), 1e-4,
                               f"pair {p['pair']}: the two arms are the same audio")
        c, _ = sf.read(str(REPO / self.m["payoff"]["clean_file"]))
        d, _ = sf.read(str(REPO / self.m["payoff"]["deadzone_file"]))
        self.assertGreater(float(np.abs(c - d).mean()), 1e-4,
                           "the payoff dead-zone clip is identical to the control")

    def test_the_isolation_ladder_is_complete(self):
        got = set(self.m["isolation"])
        from scripts.make_audio_sets import LADDER
        want = {"00_RAW_original"} | {label for label, _ in LADDER}
        self.assertEqual(got, want,
                         "the factor-isolation ladder drifted from LADDER in "
                         "scripts/make_audio_sets.py — there must be ONE definition")

    def test_the_old_listening_set_is_marked_superseded(self):
        old = REPO / "results" / "audio" / "listen"
        if not old.is_dir():
            self.skipTest("old listening set not present")
        note = old / "SUPERSEDED.md"
        self.assertTrue(note.is_file(),
                        "the retired DEADZONE_* files carry no supersession note, "
                        "so the next person will demo them again")
        self.assertIn("results/audio/demo", note.read_text())


# ---------------------------------------------------------------------------
# narration drift
# ---------------------------------------------------------------------------
#
# THE DEFECT THIS EXISTS FOR. demo_break.py printed the exemplar's factor line
# ("SNR 0 dB · engine · g726 · mic rolloff 0") and then, four lines later, a
# hardcoded sentence reading "Note the SNR: 20 dB is a QUIET room. This is reverb
# and channel, not loudness." The sentence was written for the exemplar that
# ranked #1 before SPEC Appendix G's estimand mismatch was corrected; when the
# corrected #1 turned out to be a 0 dB engine cell, the literal stayed. The demo
# ran, exited 0, and contradicted itself on the most-watched slide of the
# interview. No test could see it, because nothing crashed.
#
# So the checker below reads the narration block the demo actually prints and
# asks whether every acoustic quantity IN it agrees with the condition it is
# narrating. `test_the_checker_catches_the_sentence_that_shipped` is the negative
# control: it feeds the real historical sentence to the same checker and requires
# it to be flagged, so the test is pinned to the violation rather than to some
# incidental property of today's clean output.

# Values a narration is allowed to name, and the condition field each must equal.
_CODEC_TOKENS = ("g726", "opus-lowrate", "amr")
_NOISE_TOKENS = ("babble", "engine", "road")


def narration_conflicts(text: str, cond: dict) -> list[str]:
    """
    Every acoustic literal in `text` that disagrees with `cond`. Empty == clean.

    Deliberately checks VALUES, not phrasing: a narration is free to say whatever
    it likes about the cell as long as the numbers and the factor names it puts on
    screen are that cell's own.
    """
    bad = []
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*dB\b", text):
        got = float(m.group(1))
        if abs(got - float(cond["snr_db"])) > 1e-9:
            bad.append(f"names {got:g} dB, but the condition is "
                       f"SNR {float(cond['snr_db']):g} dB")
    for m in re.finditer(r"rt60[^0-9\-]{0,12}(\d+(?:\.\d+)?)", text, re.I):
        got = float(m.group(1))
        if not any(abs(got - float(cond[k])) < 5e-3
                   for k in ("rt60", "rt60_measured") if cond.get(k) is not None):
            bad.append(f"names rt60 {got:g}, but the condition is "
                       f"rt60 {float(cond['rt60']):g}")
    for tok in _CODEC_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", text) and tok != cond["codec"]:
            bad.append(f"names codec {tok!r}, but the condition uses "
                       f"{cond['codec']!r}")
    for tok in _NOISE_TOKENS:
        if re.search(rf"\b{tok}\b", text) and tok != cond["noise_type"]:
            bad.append(f"names noise {tok!r}, but the condition uses "
                       f"{cond['noise_type']!r}")
    return bad


def narration_block(stdout: str) -> str:
    """
    The stage-2 header + its narration, i.e. everything between the DEAD ZONE
    heading and the rule that closes it. Scoped on purpose: further down the page
    the demo prints the full dead-zone ranking, whose condition NAMES legitimately
    contain other cells' factor values.
    """
    lines = stdout.splitlines()
    start = next((i for i, l in enumerate(lines) if "DEAD ZONE #" in l), None)
    assert start is not None, "no stage-2 header in the demo output"
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].strip().startswith("─")), len(lines))
    return "\n".join(lines[start:end])


class NarrationTracksTheConditionOnScreen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cache()
        cls.r = run_demo("--offline", "--no-audio", "--pause", "0")
        cls.cond = cache()["condition"]

    def test_the_rendered_narration_agrees_with_the_demoed_condition(self):
        block = narration_block(self.r.stdout)
        self.assertGreater(len(block.splitlines()), 1,
                           "the stage-2 header has no narration under it")
        self.assertEqual(narration_conflicts(block, self.cond), [],
                         f"the narration contradicts the condition it is "
                         f"narrating:\n{block}")

    def test_the_checker_catches_the_sentence_that_shipped(self):
        """NEGATIVE CONTROL — the literal that actually regressed, verbatim."""
        shipped = ("Note the SNR: 20 dB is a QUIET room. This is reverb and "
                   "channel, not loudness.")
        conflicts = narration_conflicts(shipped, self.cond)
        self.assertTrue(conflicts,
                        "the drift checker does not flag the exact sentence that "
                        "shipped against the exact condition it shipped with — it "
                        "would not have caught the regression it exists for")
        self.assertTrue(any("dB" in c for c in conflicts), conflicts)

    def test_the_checker_passes_the_narration_for_the_condition_it_was_written_for(self):
        """
        The control's control: the same shipped sentence is CLEAN against the old
        exemplar. This pins the failure to the pairing (sentence x condition), not
        to the sentence — otherwise the checker could be flagging any prose at all
        and the test above would pass for the wrong reason.
        """
        old = {"rt60": 0.7, "rt60_measured": 0.68, "snr_db": 20.0,
               "noise_type": "babble", "codec": "opus-lowrate", "mic_rolloff": 1.0}
        shipped = ("Note the SNR: 20 dB is a QUIET room. This is reverb and "
                   "channel, not loudness.")
        self.assertEqual(narration_conflicts(shipped, old), [])

    def test_narrate_condition_follows_the_condition_it_is_given(self):
        """The generator is a pure function of the cell, on two opposite cells."""
        levels = {"rt60": [0.2, 0.45, 0.7, 1.0], "snr_db": [0.0, 5.0, 10.0, 20.0],
                  "mic_rolloff": [0.0, 0.5, 1.0], "codec": ["none", "g726"],
                  "noise_type": ["babble", "engine"]}
        loud = {"rt60": 0.45, "snr_db": 0.0, "noise_type": "engine",
                "codec": "g726", "mic_rolloff": 0.0}
        quiet = {"rt60": 1.0, "snr_db": 20.0, "noise_type": "babble",
                 "codec": "none", "mic_rolloff": 1.0}
        t_loud = demo_break.narrate_condition(loud, levels)
        t_quiet = demo_break.narrate_condition(quiet, levels)
        self.assertNotEqual(t_loud, t_quiet)
        self.assertEqual(narration_conflicts(t_loud, loud), [])
        self.assertEqual(narration_conflicts(t_quiet, quiet), [])
        # and the two are not interchangeable: swapping them is a contradiction
        self.assertTrue(narration_conflicts(t_loud, quiet)
                        or narration_conflicts(t_quiet, loud),
                        "the narration says the same thing about opposite cells, "
                        "so it is not actually derived from them")
        # the rhetorical point of the retired literal, now conditional on the data
        self.assertIn("not loudness", t_quiet)
        self.assertNotIn("not loudness", t_loud)

    def test_narrate_condition_survives_a_missing_level_set(self):
        """An older cache has no `grid_levels`; the demo must degrade, not crash."""
        cell = {"rt60": 0.45, "snr_db": 0.0, "noise_type": "engine",
                "codec": "g726", "mic_rolloff": 0.0}
        for levels in (None, {}, {"snr_db": []}):
            txt = demo_break.narrate_condition(cell, levels)
            self.assertTrue(txt.strip())
            self.assertEqual(narration_conflicts(txt, cell), [])

    def test_the_closing_correlation_is_cached_not_recited(self):
        """
        The demo's last claim used to be the literal `-0.980`. It is now read from
        `deadzone.analysis.confidence_gap`, on the paired estimand, and both the
        value and its n must be on screen so a presenter can say what it is over.
        """
        corr = cache().get("correlation") or {}
        self.assertEqual(corr.get("wer_key"), "wer_spoke",
                         "the correlation is not on the paired estimand")
        rho, n = float(corr["spearman"]), int(corr["n_conditions"])
        self.assertLess(rho, -0.3)
        out = self.r.stdout
        self.assertIn(f"{rho:.3f}", out, "the correlation is not on screen")
        self.assertIn(str(n), out, "the correlation's n is not on screen")
        self.assertNotIn("-0.980", (REPO / "demos" / "demo_break.py").read_text(),
                         "the correlation is a literal in the source again; it "
                         "happens to be right today, which is exactly how the SNR "
                         "sentence survived")

    def test_the_closing_correlation_follows_the_cache_and_not_a_constant(self):
        """
        NEGATIVE CONTROL for the line above: feed a doctored correlation and
        require the screen to change. Asserting only that `-0.980` is on screen
        would pass identically whether the number is derived or recited.
        """
        import contextlib
        import copy
        import io
        c = copy.deepcopy(cache())
        c["correlation"]["spearman"] = -0.5
        c["correlation"]["n_conditions"] = 7
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demo_break.run(c, c["default_clip"], demo_break.Ink(False), 96,
                           False, 0.0, False)
        got = buf.getvalue()
        self.assertIn("-0.500", got)
        self.assertIn("Across the 7 conditions", got)
        self.assertNotIn("-0.980", got)

    def test_a_cache_without_a_correlation_says_so_instead_of_reciting_one(self):
        import contextlib
        import copy
        import io
        c = copy.deepcopy(cache())
        c.pop("correlation", None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demo_break.run(c, c["default_clip"], demo_break.Ink(False), 96,
                           False, 0.0, False)
        got = buf.getvalue()
        self.assertIn("make demo-prep", got)
        self.assertNotRegex(got, r"spearman.*-0\.\d",
                            "a missing correlation must be reported, not "
                            "back-filled from memory")

    def test_the_aggregate_wer_is_paired_with_the_confidence(self):
        """
        SPEC Appendix G, in the demo's own summary: the WER printed beside a mean
        confidence must be `wer_spoke`. For today's exemplar the two are equal, so
        this cannot be caught by reading the screen — it is checked against the
        cache, and against the identity that defines the gap.
        """
        c = self.cond
        self.assertAlmostEqual(c["gap"],
                               c["mean_conf"] - (1.0 - c["wer_spoke"]), places=9,
                               msg="gap is not conf - (1 - wer_spoke); the demo "
                                   "is quoting a mismatched pairing")
        self.assertIn(f"{c['wer_spoke']:.3f}", self.r.stdout)
        for z in cache()["dead_zones"]:
            if z.get("n_silent"):
                self.assertIn(f"{z['wer_spoke']:.3f}", self.r.stdout,
                              f"{z['name']} is listed on the all-clips WER next "
                              f"to a spoke-only confidence")


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
