#!/usr/bin/env python3
"""
test_demo_live.py — the optional live beat is safe to put on a stage.

    ./.venv/bin/python tests/test_demo_live.py

THIS SUITE MAKES NO NETWORK CALL. `demos/demo_live.py` is the one file in the
repo that is *meant* to talk to a vendor, which makes it the one file whose
tests must be most careful not to. Every live path here is exercised against a
substituted adapter, and one test proves structurally that the offline path
opens no socket at all.

What is asserted is exactly what stage failure looks like:

  * the beat NEVER exits non-zero. No key, no network, a vendor error, a
    timeout, a hang — every one of them prints an explanation and exits 0,
    because `make demo` must not go red because the wifi did;
  * it never hangs. A call that never returns is abandoned, on a deadline, and
    the process still exits;
  * NO code path can print a credential. The violating input is constructed
    (a vendor error string with the key embedded in it, which is the realistic
    leak) and a negative control of the same shape is asserted to pass through,
    so the test is pinned to redaction and not to the demo simply printing
    nothing;
  * the `--offline` path renders the same beat, so the presenter can rehearse
    it with wifi off and reach for it as an instant fallback;
  * the clip and condition on screen are still the MEASURED ones. If a re-run
    of the grid moves the dead zones, this fails rather than letting the demo
    narrate a finding that is no longer true.
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
# Makes `deadzone`, `scripts` and `demos` importable when this file is run
# directly (`python tests/test_demo_live.py`) with no install step. Harmless
# when it is imported as a module instead.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# -------------------------------------------------------------------------

import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

from demos import demo_live

REPO = Path(_REPO_ROOT)
PY = str(Path(sys.executable))

# A value shaped like a real credential: long enough to clear Redactor.MIN_LEN,
# and embedded in the exact place a vendor SDK would put it.
FAKE_KEY = "fakekey-0123456789abcdef0123456789abcdef"

# A decoy of the SAME shape that is NOT a secret. It must survive to the screen
# untouched — otherwise a redactor that simply blanked everything would pass the
# leak tests and nobody would learn that the demo prints nothing useful.
DECOY = "notasecret-fedcba9876543210fedcba98765432"


def child(*args: str, key: str | None = None, timeout: int = 120,
          extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run demo_live.py in a child process with an explicitly-chosen key state."""
    env = dict(os.environ)
    env.pop("DEEPGRAM_API_KEY", None)
    if key is not None:
        env["DEEPGRAM_API_KEY"] = key
    env["TERM"] = "dumb"                             # force the plain-text branch
    env.update(extra_env or {})
    return subprocess.run([PY, "demos/demo_live.py", *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=timeout)


def in_process(argv: list[str], patch_transcribe=None, key: str | None = None,
               hide_env_file: bool = True) -> tuple[int, str]:
    """
    Run main() in this process with the adapter substituted, capturing stdout.

    The substitution targets `deadzone.audio_pipeline.transcribe_deepgram`, which
    `demo_live.transcribe_live` imports INSIDE the function body — so the patch
    is picked up at call time. That is the same seam the real code uses to stay
    import-light, exercised rather than worked around.

    `hide_env_file` neutralizes the `.env` fallback. Without it, `key=None` does
    NOT mean "no key": `load_credentials` correctly reads `.env`, so on a machine
    that has one the no-key test would silently become a live-call test. That is
    a test that passes for the wrong reason on the developer's machine and fails
    only in CI — worth the extra seam to rule out.
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
            rc = demo_live.main(argv)
        return rc, buf.getvalue()
    finally:
        ap.transcribe_deepgram = saved_fn
        rx.load_env = saved_env
        demo_live._REDACT = saved_redact
        if saved_key is None:
            os.environ.pop("DEEPGRAM_API_KEY", None)
        else:
            os.environ["DEEPGRAM_API_KEY"] = saved_key


def squeeze(s: str) -> str:
    """Strip ALL whitespace.

    A secret split across a line break by wrapping is still a leaked secret, and
    a substring test on the raw text would not see it. That is not hypothetical:
    it is the bug this file's redaction was rebuilt around (see demo_live.say).
    """
    return "".join(s.split())


def flat(s: str) -> str:
    """Collapse runs of whitespace to single spaces.

    Terminal output is WRAPPED, so `assertIn("some long sentence", out)` is a
    test that fails on line length rather than on behaviour. Every prose
    assertion in this file goes through here; only the exact numeric strings are
    matched raw.
    """
    return " ".join(s.split())


# ==========================================================================
# the offline path — the rehearsable one
# ==========================================================================

class OfflinePath(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = child("--offline", "--no-color")

    def test_exits_zero_with_no_key_in_the_environment(self):
        self.assertEqual(self.r.returncode, 0,
                         f"exited {self.r.returncode}\nSTDERR:\n{self.r.stderr[-3000:]}")

    def test_labels_itself_as_cached_not_live(self):
        """Provenance is not optional. A replay must never look like a live call."""
        self.assertIn("CACHED REPLAY", self.r.stdout)
        self.assertIn("no API call was made", self.r.stdout)
        self.assertNotIn("two real API calls", self.r.stdout)

    def test_renders_both_stages_and_the_comparison(self):
        for marker in ("RAW RECORDING", "DEAD ZONE", "SIDE BY SIDE",
                       "per-word confidence", "AGAINST THE MEASURED GRID"):
            self.assertIn(marker, self.r.stdout, f"{marker!r} missing from the beat")

    def test_shows_the_per_word_confidences_not_just_a_mean(self):
        """The 'I use your product' half of the beat is the payload itself."""
        clean, dz = self._facts()
        for c in dz["word_confidences"]:
            self.assertIn(f"{c:.4f}", self.r.stdout,
                          "a per-word confidence is not on screen")
        self.assertGreaterEqual(len(dz["word_confidences"]), 3)

    def test_shows_the_break_and_the_retained_confidence(self):
        clean, dz = self._facts()
        self.assertIn(f"{dz['wer']:.3f}", self.r.stdout)
        self.assertIn(f"{dz['mean_conf']:.3f}", self.r.stdout)
        # the point of the beat: WER moves a lot, confidence barely moves
        self.assertGreater(dz["wer"] - clean["wer"], 0.2)
        self.assertLess(clean["mean_conf"] - dz["mean_conf"], 0.2)

    def test_prints_no_cost_line_when_nothing_was_spent(self):
        """A cached run costs nothing; claiming a cost would be a false number."""
        self.assertNotIn("WHAT THAT COST", self.r.stdout)

    def test_plain_text_fallback_has_no_ansi(self):
        self.assertNotIn("\033[", self.r.stdout,
                         "ANSI escapes leaked into a non-tty run")

    def test_check_exits_zero_and_reports_the_fallback(self):
        r = child("--check", "--no-color")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("cached fallback", r.stdout)

    def _facts(self):
        refs = demo_live.load_manifest_refs()
        return demo_live.cached_facts(demo_live.DEFAULT_CLIP,
                                      demo_live.DEFAULT_CONDITION,
                                      refs[demo_live.DEFAULT_CLIP])


class NoNetwork(unittest.TestCase):
    """Structural proof, not a promise in a docstring."""

    def test_offline_path_opens_no_socket(self):
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
            "from demos import demo_live\n"
            "rc = demo_live.main(['--offline', '--no-color'])\n"
            "print('NO_SOCKET_OK', rc)\n"
        )
        env = dict(os.environ)
        env.pop("DEEPGRAM_API_KEY", None)
        env["TERM"] = "dumb"
        r = subprocess.run([PY, "-c", script], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("NO_SOCKET_OK 0", r.stdout)


# ==========================================================================
# graceful degradation — every way the live call can die
# ==========================================================================

class GracefulDegradation(unittest.TestCase):
    """
    Each case constructs the real failure and asserts the same three things:
    exit 0, a clear explanation naming the cause, and the beat still delivered
    from cache.
    """

    def _assert_survived(self, rc: int, out: str, cause: str) -> None:
        flat_out = flat(out)
        self.assertEqual(rc, 0, f"the beat exited non-zero on: {cause}\n{out[-2000:]}")
        self.assertIn("LIVE CALL SKIPPED", flat_out,
                      f"no explanation printed for: {cause}")
        self.assertIn("CACHED REPLAY", flat_out,
                      f"the beat was not delivered from cache after: {cause}")
        # it must say what would have been shown, not merely that it failed
        self.assertIn("What you would have seen live", flat_out)
        # and the finding itself still has to be on screen
        self.assertIn("SIDE BY SIDE", flat_out, f"the beat did not render after: {cause}")

    def test_no_api_key(self):
        rc, out = in_process(["--no-color", "--timeout", "2"], key=None,
                             patch_transcribe=_never_called(self))
        self._assert_survived(rc, out, "no API key")
        self.assertIn("not found in the environment or .env", flat(out))

    def test_network_error(self):
        def boom(*a, **k):
            raise OSError("[Errno 8] nodename nor servname provided, or not known")
        rc, out = in_process(["--no-color", "--timeout", "2"], key=FAKE_KEY,
                             patch_transcribe=boom)
        self._assert_survived(rc, out, "network error")
        self.assertIn("OSError", flat(out))

    def test_vendor_failure_sentinel(self):
        """The adapter swallows errors and returns a sentinel; that must be caught."""
        from deadzone.audio_pipeline import _empty_confidence_result
        def failed(*a, **k):
            return _empty_confidence_result(error="deepgram failed after 1 attempts")
        rc, out = in_process(["--no-color", "--timeout", "2"], key=FAKE_KEY,
                             patch_transcribe=failed)
        self._assert_survived(rc, out, "vendor failure sentinel")

    def test_empty_word_confidences_is_treated_as_a_failure(self):
        """
        SPEC 12's day-one gate, on stage: no per-word confidence means the
        headline signal is absent and a beat about confidence has nothing to
        show. Falling back beats narrating an empty list.
        """
        def confless(*a, **k):
            return {"transcript": "deliver it to sofia", "word_confidences": [],
                    "mean_conf": float("nan"), "utterance_conf": float("nan")}
        rc, out = in_process(["--no-color", "--timeout", "2"], key=FAKE_KEY,
                             patch_transcribe=confless)
        self._assert_survived(rc, out, "no per-word confidences")
        self.assertIn("no per-word confidences", flat(out))

    def test_timeout_is_enforced_and_the_process_does_not_hang(self):
        """
        A call that never returns must be abandoned on the deadline AND must not
        hold the process open on the way out — which is why the worker is a raw
        daemon thread and not a ThreadPoolExecutor (whose atexit hook joins).
        """
        def hang(*a, **k):
            time.sleep(600)
        t0 = time.monotonic()
        rc, out = in_process(["--no-color", "--timeout", "1"], key=FAKE_KEY,
                             patch_transcribe=hang)
        elapsed = time.monotonic() - t0
        self._assert_survived(rc, out, "a hung call")
        self.assertIn("no response within 1s", flat(out))
        self.assertLess(elapsed, 30, f"the deadline did not bound the wait ({elapsed:.1f}s)")

    def test_a_hung_call_does_not_block_interpreter_exit(self):
        """
        The in-process test above cannot see a shutdown hang, because the test
        runner keeps the interpreter alive. This one runs a whole child process
        and asserts it TERMINATES — the only way to prove the daemon-thread
        choice actually holds.
        """
        script = (
            "import sys, time\n"
            "sys.path.insert(0, '.')\n"
            "import os\n"
            "os.environ['DEEPGRAM_API_KEY'] = %r\n"
            "import deadzone.audio_pipeline as ap\n"
            "ap.transcribe_deepgram = lambda *a, **k: time.sleep(600)\n"
            "from demos import demo_live\n"
            "rc = demo_live.main(['--no-color', '--timeout', '1'])\n"
            "print('EXITED_CLEANLY', rc)\n"
        ) % FAKE_KEY
        env = dict(os.environ)
        env["TERM"] = "dumb"
        t0 = time.monotonic()
        # A generous timeout that is still far below sleep(600): if the atexit
        # join ever comes back, this raises TimeoutExpired instead of passing.
        r = subprocess.run([PY, "-c", script], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=90)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("EXITED_CLEANLY 0", r.stdout)
        self.assertLess(time.monotonic() - t0, 60)

    def test_an_unknown_condition_fails_LOUDLY_and_that_is_deliberate(self):
        """
        The one case that is NOT exit 0, pinned so nobody "fixes" it.

        The exit-0 contract covers ENVIRONMENTAL failure — no key, no network,
        a vendor error, a timeout. Those are things the world did to you two
        minutes before you present, and swallowing them is right. A `--condition`
        that does not exist is an operator asking for something that was never
        measured, and quietly falling back to a different condition than the one
        requested would be narrating the wrong cell — which is precisely the
        class of silent, plausible-looking wrongness this project exists to
        catch. `make demo-live` passes no arguments, so the stage path cannot
        reach this.
        """
        r = child("--no-color", "--condition", "not-a-real-condition", key=None)
        self.assertNotEqual(r.returncode, 0, "an unmeasured condition passed silently")
        blob = flat(r.stdout + r.stderr)
        self.assertIn("no nova-3 row", blob, "the error does not say what was wrong")
        self.assertIn("not-a-real-condition", blob,
                      "the error does not name the condition that was asked for")


def _never_called(test):
    def fn(*a, **k):
        test.fail("the adapter was called despite there being no API key")
    return fn


# ==========================================================================
# credentials — the property that cannot be allowed to regress
# ==========================================================================

class NeverPrintsACredential(unittest.TestCase):

    def test_the_realistic_leak_is_redacted(self):
        """
        VIOLATING INPUT: a vendor error carrying the key, which is exactly how
        this would happen — `transcribe_deepgram` folds the underlying
        exception's repr into its error string, and that string is printed.
        """
        def boom(*a, **k):
            raise RuntimeError(
                f"401 Unauthorized: Token {FAKE_KEY} rejected by api.deepgram.com")
        rc, out = in_process(["--no-color", "--timeout", "2"], key=FAKE_KEY,
                             patch_transcribe=boom)
        self.assertEqual(rc, 0)
        self.assertNotIn(FAKE_KEY, out, "the API key printed to stdout")
        self.assertNotIn(FAKE_KEY, squeeze(out),
                         "the API key printed, split across a line break by wrapping")
        self.assertIn("<redacted>", out, "nothing was redacted; the guard did not run")

    def test_negative_control_a_non_secret_of_the_same_shape_survives(self):
        """
        WITHOUT THIS the test above passes for the wrong reason: a redactor that
        blanked every long token, or a demo that printed nothing at all, would
        satisfy it. This pins the assertion to redaction of the SECRET
        specifically, and proves the error message still reaches the presenter.
        """
        def boom(*a, **k):
            raise RuntimeError(f"503 upstream unavailable, trace {DECOY}")
        rc, out = in_process(["--no-color", "--timeout", "2"], key=FAKE_KEY,
                             patch_transcribe=boom)
        self.assertEqual(rc, 0)
        self.assertIn(DECOY, squeeze(out),
                      "a non-secret diagnostic was scrubbed — the presenter now has "
                      "no way to see WHY the call failed")
        self.assertNotIn(FAKE_KEY, squeeze(out))

    def test_a_key_in_any_secret_named_variable_is_redacted(self):
        """The redactor keys off the variable NAME, so a new one is covered."""
        r = Redactor_with({"SOME_VENDOR_TOKEN": FAKE_KEY})
        self.assertNotIn(FAKE_KEY, r(f"boom {FAKE_KEY} boom"))

    def test_short_values_are_not_redacted(self):
        """
        A 1-2 character secret would match everywhere and turn the whole beat
        into placeholders — a denial of service on the presentation rather than
        a protection. Pinned so nobody 'hardens' MIN_LEN to 1.
        """
        r = Redactor_with({"TINY_KEY": "ab"})
        self.assertEqual(r("a cab in the lab"), "a cab in the lab")

    def test_not_even_a_prefix_of_the_key_is_printed(self):
        """
        Truncation is not redaction. `sk-abc...` is still enough to correlate a
        credential against a leak elsewhere, and it is the shape people reach
        for when they want to "show which key was used".
        """
        rc, out = in_process(["--no-color", "--timeout", "2"], key=FAKE_KEY,
                             patch_transcribe=lambda *a, **k: (_ for _ in ()).throw(
                                 OSError("no route to host")))
        self.assertEqual(rc, 0)
        for prefix_len in (8, 12, 16, 24):
            self.assertNotIn(FAKE_KEY[:prefix_len], squeeze(out),
                             f"the first {prefix_len} characters of the key printed")

    def test_no_environment_dump_reaches_the_screen(self):
        """The presenter may be screen-sharing (see the module docstring)."""
        r = child("--offline", "--no-color", key=FAKE_KEY,
                  extra_env={"A_SECRET_TOKEN_VALUE": DECOY})
        blob = r.stdout + r.stderr
        for tok in ("Authorization", "Bearer ", "Token ", "api_key=", "DEEPGRAM_API_KEY="):
            self.assertNotIn(tok, blob, f"{tok!r} leaked into demo output")
        self.assertNotIn(FAKE_KEY, squeeze(blob))

    def test_source_contains_no_real_credential(self):
        """
        A hardcoded key is the one failure no runtime guard can catch, because
        the guard is built FROM the environment and a literal in the source was
        never in the environment to be collected.

        Checked against the actual secrets on this machine rather than by
        pattern: an entropy heuristic would flag the condition names (which are
        long opaque-looking strings by design) and get muted, whereas this can
        only fire on a real leak. Neither the value nor its length is reported.
        """
        src = (REPO / "demos" / "demo_live.py").read_text()
        secrets = []
        env_file = REPO / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    name, val = line.split("=", 1)
                    val = val.strip().strip('"').strip("'")
                    if len(val) >= demo_live.Redactor.MIN_LEN:
                        secrets.append((name.strip(), val))
        for name, val in secrets:
            self.assertNotIn(val, src, f"demo_live.py hardcodes the value of {name}")
        if not secrets:
            self.skipTest("no .env secrets on this machine to check against")


def Redactor_with(env: dict) -> "demo_live.Redactor":
    """Build a Redactor over a temporary environment, without leaking it."""
    saved = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        return demo_live.Redactor()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ==========================================================================
# the live path, with the adapter substituted — no network
# ==========================================================================

class LivePathWithASubstitutedAdapter(unittest.TestCase):

    def setUp(self):
        refs = demo_live.load_manifest_refs()
        self.ref = refs[demo_live.DEFAULT_CLIP]
        _, self.dz = demo_live.cached_facts(demo_live.DEFAULT_CLIP,
                                            demo_live.DEFAULT_CONDITION, self.ref)

    def _replay(self):
        """Return the measured rows, as if the vendor had just returned them."""
        clean_rec = {}
        for line in open(REPO / "results" / "clean_transcripts.jsonl"):
            rec = json.loads(line)
            if rec["id"] == demo_live.DEFAULT_CLIP:
                clean_rec = rec
                break
        answers = {
            "clean": {"transcript": clean_rec["transcript"],
                      "word_confidences": clean_rec["word_confidences"]},
            "dz": {"transcript": self.dz["transcript"],
                   "word_confidences": self.dz["word_confidences"]},
        }

        def fn(path, *a, **k):
            key = "clean" if "clean" in str(path) else "dz"
            r = dict(answers[key])
            wc = r["word_confidences"]
            r["mean_conf"] = sum(wc) / len(wc)
            r["utterance_conf"] = r["mean_conf"]
            return r
        return fn

    def test_the_live_beat_renders_and_is_labelled_live(self):
        rc, out = in_process(["--no-color", "--timeout", "5"], key=FAKE_KEY,
                             patch_transcribe=self._replay())
        self.assertEqual(rc, 0, out[-2000:])
        self.assertIn("two real API calls", out)
        self.assertNotIn("LIVE CALL SKIPPED", out)
        self.assertIn("round trip", out, "the per-call latency is not shown")

    def test_it_names_the_variable_it_used_and_never_the_value(self):
        """
        'Which credential am I on?' is a real question in a room, and the safe
        answer is the variable NAME plus where it came from. Asserted on the
        SUCCESS path, because the failure path deliberately replaces this line
        with 'the live call did not happen'.
        """
        rc, out = in_process(["--no-color", "--timeout", "5"], key=FAKE_KEY,
                             patch_transcribe=self._replay())
        self.assertEqual(rc, 0)
        self.assertIn("DEEPGRAM_API_KEY", out, "the provenance line is missing")
        self.assertIn("from the environment", flat(out))
        self.assertNotIn(FAKE_KEY, squeeze(out))

    def test_the_live_beat_prints_a_cost_line(self):
        """This project prices everything; the habit is the point."""
        rc, out = in_process(["--no-color", "--timeout", "5"], key=FAKE_KEY,
                             patch_transcribe=self._replay())
        self.assertIn("WHAT THAT COST", out)
        self.assertIn("2 calls", out)
        self.assertIn(str(demo_live.USD_PER_MINUTE), out)
        self.assertIn(demo_live.RATE_AS_OF, out,
                      "the rate is quoted without the date it was checked")

    def test_live_and_offline_render_the_same_beat(self):
        """
        The `--offline` flag has to be a true stand-in, not a lesser mode: it is
        what the presenter reaches for when the wifi is gone, mid-beat.
        """
        _, live = in_process(["--no-color", "--timeout", "5"], key=FAKE_KEY,
                             patch_transcribe=self._replay())
        _, cached = in_process(["--offline", "--no-color"], key=FAKE_KEY)
        for marker in ("RAW RECORDING", "DEAD ZONE", "SIDE BY SIDE",
                       "per-word confidence", "AGAINST THE MEASURED GRID"):
            self.assertIn(marker, live)
            self.assertIn(marker, cached)
        self.assertIn(f"{self.dz['wer']:.3f}", live)
        self.assertIn(f"{self.dz['wer']:.3f}", cached)


# ==========================================================================
# the numbers on screen are the measured ones
# ==========================================================================

class TheExemplarIsStillTheMeasuredOne(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        csv.field_size_limit(10 ** 9)
        with open(REPO / "results" / "dead_zones.csv", newline="") as fh:
            cls.rows = list(csv.DictReader(fh))

    def test_the_condition_is_the_top_ranked_nova3_dead_zone(self):
        dz = [r for r in self.rows
              if r["model"] == demo_live.MODEL and r["category"] == "dead_zone"]
        self.assertTrue(dz, "no nova-3 dead zones in results/dead_zones.csv")
        self.assertEqual(dz[0]["condition_name"], demo_live.DEFAULT_CONDITION,
                         "the #1 dead zone moved; re-pick the demo condition")

    def test_the_condition_has_no_silent_clips(self):
        """
        The load-bearing property. A cell with silent clips has its confidence
        averaged over an easier subset than its WER, which is the estimand
        mismatch that demoted the previous #1 to `silence_driven`. Demoing one
        of those as a dead zone would be demoing the artifact.
        """
        row = next(r for r in self.rows
                   if r["model"] == demo_live.MODEL
                   and r["condition_name"] == demo_live.DEFAULT_CONDITION
                   and r["category"] == "dead_zone")
        self.assertEqual(int(row["n_silent"]), 0)
        # the identity that settles this file, since its columns must never be
        # read by counting (rt60_measured sits immediately before mean_conf and
        # reads exactly like a plausible confidence)
        gap = float(row["mean_conf"]) - (1.0 - float(row["wer_spoke"]))
        self.assertAlmostEqual(gap, float(row["gap_spoke"]), places=9,
                               msg="gap_spoke != mean_conf - (1 - wer_spoke): the row "
                                   "being quoted is not the row being read")

    def test_the_clip_is_confidently_wrong_on_its_own_row(self):
        """
        A condition average does not survive contact with an audience. The clip
        on screen has to fail on ITS OWN row, while the model is confident.
        """
        refs = demo_live.load_manifest_refs()
        clean, dz = demo_live.cached_facts(demo_live.DEFAULT_CLIP,
                                           demo_live.DEFAULT_CONDITION,
                                           refs[demo_live.DEFAULT_CLIP])
        self.assertEqual(clean["wer"], 0.0,
                         "the control is not clean — the contrast is dead before "
                         "stage 2 gets to make it")
        self.assertGreaterEqual(dz["wer"], 0.25, "the exemplar barely fails")
        self.assertGreaterEqual(dz["mean_conf"], 0.80,
                                "the model was NOT confident — that is the opposite "
                                "of the point being demonstrated")

    def test_the_model_is_confident_on_a_word_it_invented(self):
        """
        The sharpest claim the beat makes out loud: at least one SUBSTITUTED
        word carries a confidence above the utterance mean. If a re-run makes
        every error low-confidence, the narration is wrong and this fails.
        """
        refs = demo_live.load_manifest_refs()
        _, dz = demo_live.cached_facts(demo_live.DEFAULT_CLIP,
                                       demo_live.DEFAULT_CONDITION,
                                       refs[demo_live.DEFAULT_CLIP])
        cols = demo_live.diff_columns(dz["edits"])
        slots = [i for i, (op, _, _) in enumerate(cols) if op in ("match", "sub", "ins")]
        wc = dz["word_confidences"]
        self.assertEqual(len(slots), len(wc),
                         "confidences do not align with the hypothesis words, so the "
                         "per-word claim cannot be made for this clip")
        subs = [wc[k] for k, i in enumerate(slots) if cols[i][0] == "sub"]
        self.assertTrue(subs, "no substitutions to show")
        self.assertGreater(max(subs), dz["mean_conf"],
                           "no invented word beats the utterance mean confidence")

    def test_the_cached_facts_match_the_master_table(self):
        csv.field_size_limit(10 ** 9)
        row = None
        with open(REPO / "results" / "master.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                if (r["model"] == demo_live.MODEL
                        and r["condition_name"] == demo_live.DEFAULT_CONDITION
                        and r["clip_id"] == demo_live.DEFAULT_CLIP):
                    row = r
                    break
        self.assertIsNotNone(row)
        refs = demo_live.load_manifest_refs()
        _, dz = demo_live.cached_facts(demo_live.DEFAULT_CLIP,
                                       demo_live.DEFAULT_CONDITION,
                                       refs[demo_live.DEFAULT_CLIP])
        self.assertEqual(dz["transcript"], row["transcript"])
        self.assertAlmostEqual(dz["wer"], float(row["wer"]), places=9)
        self.assertAlmostEqual(dz["mean_conf"], float(row["mean_conf"]), places=9)


class PayloadPairing(unittest.TestCase):
    """
    SPEC B.5(3): edits come from normalized tokens, confidences from raw ones.
    Where the counts disagree, the payload must NOT be zipped.
    """

    def test_matched_counts_pair(self):
        got = demo_live.pair_words_with_confidences("a b c", [0.1, 0.2, 0.3])
        self.assertEqual(got, [("a", 0.1), ("b", 0.2), ("c", 0.3)])

    def test_mismatched_counts_refuse_to_pair(self):
        """VIOLATING INPUT: one more token than confidence."""
        self.assertIsNone(demo_live.pair_words_with_confidences("a b c", [0.1, 0.2]))

    def test_the_demo_prints_the_raw_list_instead_of_guessing(self):
        facts = {"transcript": "a b c", "word_confidences": [0.1, 0.2],
                 "wer": 0.5, "mean_conf": 0.15, "n_ref": 3,
                 "n_sub": 0, "n_del": 1, "n_ins": 0, "edits": []}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demo_live.show_payload(facts, demo_live.Ink(False), 96, set(), False)
        out = buf.getvalue()
        self.assertIn("not pairing", out,
                      "the demo did not say it was refusing to pair the payload")
        self.assertIn("0.1000", out, "the raw confidences were dropped entirely")
        self.assertIn("0.2000", out)
        # NEGATIVE CONTROL: with the counts matched, it DOES pair, so the test
        # above is pinned to the mismatch and not to show_payload being inert.
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            demo_live.show_payload({**facts, "word_confidences": [0.1, 0.2, 0.3]},
                                   demo_live.Ink(False), 96, set(), False)
        self.assertNotIn("not pairing", buf2.getvalue())
        self.assertIn("0.3000", buf2.getvalue())


class MakefileTarget(unittest.TestCase):

    def test_makefile_has_an_optional_demo_live_target(self):
        mk = (REPO / "Makefile").read_text()
        self.assertIn("\ndemo-live:", mk, "Makefile has no demo-live target")
        self.assertIn("make demo-live", mk, "demo-live is not listed in `make help`")

    def test_the_help_text_says_it_needs_network_and_a_key(self):
        """
        Every other target in this file is offline. A target that silently is
        not would be discovered on stage, so the help has to say so.
        """
        mk = (REPO / "Makefile").read_text()
        help_block = mk.split("\nhelp:", 1)[1].split("\n\n\n", 1)[0]
        self.assertRegex(help_block, r"(?i)needs.*wifi|network")
        self.assertIn("DEEPGRAM_API_KEY", help_block)

    def test_demo_live_is_not_chained_into_the_spine(self):
        """
        `make demo` is now `demos/demo_hero.py`, which DOES call the API — the
        one thing a cached beat cannot show is the payload arriving. The
        offline guarantee was not dropped, it moved: the hero falls back to the
        archived measurements and exits 0 on a missing key, a dead network, a
        vendor error or a timeout, and `make demo-replay` runs the whole beat
        from cache. `tests/test_demo_hero.py` asserts all of that.

        What must still hold, and is what this checks: THIS file's beat is not
        chained into it. Two independent live paths in one target means two
        chances to fail for one piece of evidence, and `demo-live` is the
        hand-held fallback precisely because it is not wired to anything.
        """
        mk = (REPO / "Makefile").read_text()
        line = next(l for l in mk.splitlines()
                    if l.startswith("demo:") or l.startswith("demo: "))
        self.assertNotIn("demo-live", line,
                         "demo-live is wired into the `demo` chain; it is the "
                         "hand-held fallback and must stay unchained")


if __name__ == "__main__":
    unittest.main(verbosity=2)
