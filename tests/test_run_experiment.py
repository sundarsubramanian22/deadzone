"""
Offline tests for the JOIN-2 grid runner (R4.2). FULLY offline: a counting fake
transcribe adapter, an in-memory AssetLibrary (same pattern as test_conditions.py),
synthetic audio. No API key, no recordings, no network, no ffmpeg.

The runner is the one module whose bugs are expensive in dollars and in hours, so
each test pins one property that a real run cannot afford to discover late:

  1. the master row is schema-COMPLETE and column-ORDERED (four downstream
     readers index it by name; drift is silent),
  2. the cache actually works — a repeat run makes ZERO transcribe calls, proved
     with a call counter, and survives a torn final line (the crash shape),
  3. failures are PRESERVED as rows with failed=True, never dropped and never
     scored as 0.0,
  4. make_grid_wer_fn is the exact callable design.screen_factors wants — proved
     by handing it to the real screen_factors, not by inspecting its signature,
  5. rir_rt60_measured records the DELIVERED reverb (closest measured asset), not
     the requested rt60 — the whole reason the column exists.

Run:  python3 tests/test_run_experiment.py
"""

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
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from deadzone.audio_pipeline import _empty_confidence_result
from deadzone.conditions import AssetLibrary, Condition, NoiseAsset, RirAsset
from deadzone.design import DEFAULT_FACTOR_SPACE, screen_factors
from scripts.run_experiment import (
    AL_CLIPS, MASTER_COLUMNS, SMOKE_CLIP, SOBOL_CLIPS, ResultCache, RunStats,
    _ordered, build_plan, format_plan, load_manifest, main, main_grid,
    make_grid_wer_fn, run_grid, run_one, write_degraded_wav, write_master,
)

FS = 16000


@contextlib.contextmanager
def quiet():
    """Swallow the runner's progress chatter so the test output stays readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


# ---------------------------------------------------------------------------
# in-memory fixtures (in-memory AssetLibrary pattern, per test_conditions.py)
# ---------------------------------------------------------------------------

def make_clean(seed: int = 0, secs: float = 1.0) -> np.ndarray:
    """Padded speech-like burst: silence | two harmonics | silence."""
    rng = np.random.default_rng(seed)
    n = int(secs * FS)
    t = np.arange(n) / FS
    burst = 0.4 * (np.sin(2 * np.pi * 190 * t) + 0.4 * np.sin(2 * np.pi * 620 * t))
    pad = 1e-4 * rng.standard_normal(int(0.3 * FS))
    return np.concatenate([pad, burst, pad])


def make_rir(rt60: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = max(8, int(rt60 * FS))
    h = rng.standard_normal(L) * np.exp(-6.9 * (np.arange(L) / FS) / rt60)
    h[0] = 1.0                                   # direct path
    return h


def make_library(with_road: bool = True) -> AssetLibrary:
    """Measured RT60s 0.2 / 0.5 / 0.9 — deliberately NOT the values the grid asks
    for, so the closest-match snap is observable (test 5)."""
    rirs = [RirAsset(key=f"rir_{rt:.1f}", rt60=rt, data=make_rir(rt, seed=i), fs=FS)
            for i, rt in enumerate((0.2, 0.5, 0.9))]
    rng = np.random.default_rng(7)
    noise = [NoiseAsset("babble_a", "babble", rng.standard_normal(FS * 2), FS),
             NoiseAsset("engine_a", "engine", rng.standard_normal(FS * 2), FS)]
    if with_road:
        noise.append(NoiseAsset("road_a", "road", rng.standard_normal(FS * 2), FS))
    return AssetLibrary(rirs=rirs, noise=noise)


REFS = {
    "u02": "call maria at four zero five nine one two seven seven",
    "u05": "the invoice total is ninety four dollars and twenty cents",
    "u17": "confirm booking reference alpha seven delta nine",
}
CLIPS = {cid: make_clean(seed=i) for i, cid in enumerate(REFS)}


def _norm(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


class FakeASR:
    """
    A counting stand-in for a transcribe adapter, honouring the audio_pipeline
    adapter contract exactly (transcript / word_confidences / mean_conf /
    utterance_conf, sentinel on failure).

    It reads the CONDITION out of the temp wav's filename (`<clip>__<name>.wav`,
    written by run_one) and plants a monotone degradation: more reverb and less
    SNR delete more trailing words while confidence stays high — i.e. a
    miniature silent failure, so the screen has real structure to recover.

    It also asserts the degraded wav really exists and is readable, which is the
    only way an offline test can prove run_one wrote a file at all.
    """

    def __init__(self, fail_clips=(), fail_conditions=()):
        self.calls = 0
        self.paths: list[str] = []
        self.fail_clips = set(fail_clips)
        self.fail_conditions = set(fail_conditions)

    def __call__(self, path: str) -> dict:
        self.calls += 1
        self.paths.append(path)
        x, fs = sf.read(path)                    # proves a real wav was written
        assert fs == FS and x.size > 0 and np.all(np.isfinite(x)), path

        clip_id, cond_name = Path(path).stem.split("__", 1)
        cond = Condition.from_name(cond_name)
        if clip_id in self.fail_clips or cond_name in self.fail_conditions:
            return _empty_confidence_result(error="simulated adapter failure")

        words = REFS[clip_id].split()
        severity = 0.5 * _norm(cond.rt60, 0.2, 1.0) + 0.5 * (1 - _norm(cond.snr_db, 0.0, 25.0))
        severity += 0.15 * cond.mic_rolloff + (0.0 if cond.codec == "none" else 0.10)
        n_drop = min(len(words) - 1, int(round(severity * len(words))))
        kept = words[:len(words) - n_drop]
        confs = [round(0.97 - 0.10 * severity, 4)] * len(kept)   # stays confident
        return {"transcript": " ".join(kept), "word_confidences": confs,
                "mean_conf": float(np.mean(confs)), "utterance_conf": float(np.mean(confs))}


def _passthrough_codec(x, fs, codec):
    """Stand-in for conditions.apply_codec. The real ffmpeg round-trip is covered
    by test_conditions.py (test 7) and smoke_codec.py; this suite must not shell
    out, so codec levels chosen by the DESIGN are neutralized here."""
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1: the row is schema-complete, ordered, and strict-JSON serializable
# ---------------------------------------------------------------------------

def test_row_schema_complete_and_ordered():
    asr = FakeASR()
    cond = Condition(0.5, 10.0, "babble", "none", 0.0)
    row = run_one("u02", CLIPS["u02"], REFS["u02"], cond, make_library(), FS,
                  asr, "fake-model", "run-test")

    assert list(row.keys()) == list(MASTER_COLUMNS), list(row.keys())
    assert row["failed"] is False and row["error"] is None
    assert isinstance(row["wer"], float) and row["n_ref"] == len(REFS["u02"].split())
    assert row["n_match"] + row["n_sub"] + row["n_del"] == row["n_ref"]
    assert row["condition_name"] == cond.name and row["model"] == "fake-model"
    # json columns are STRINGS (they have to survive a CSV cell and a parquet column)
    assert isinstance(row["word_confidences"], str) and isinstance(row["edits"], str)
    assert len(json.loads(row["word_confidences"])) == len(row["transcript"].split())
    assert all(len(e) == 3 for e in json.loads(row["edits"]))
    # the cache line must be STRICT json — no bare NaN tokens
    json.loads(json.dumps(row, allow_nan=False))

    # the csv mirror's header is the schema, in order
    with tempfile.TemporaryDirectory() as td, quiet():
        paths = write_master([row], td)
        header = Path(paths["csv"]).read_text().splitlines()[0]
    assert header.split(",") == list(MASTER_COLUMNS)

    # and drift is refused rather than written
    for bad in ({k: v for k, v in row.items() if k != "wer"}, {**row, "surprise": 1}):
        try:
            _ordered(bad)
            assert False, "schema drift should raise"
        except KeyError:
            pass
    print("OK 1: master row is schema-complete, column-ordered, strict-JSON; "
          "csv header matches; drift raises")


# ---------------------------------------------------------------------------
# 2: the cache makes a repeat run FREE (asserted with a call counter)
# ---------------------------------------------------------------------------

def test_cache_second_run_makes_zero_calls():
    conds = [Condition(0.2, 25.0, "babble", "none", 0.0),
             Condition(0.9, 0.0, "engine", "none", 0.5)]
    lib = make_library()
    asr = FakeASR()
    n_cells = len(CLIPS) * len(conds)

    with tempfile.TemporaryDirectory() as td:
        cache_path = Path(td) / "cache.jsonl"

        cache = ResultCache(cache_path)
        with quiet():
            res1 = run_grid(CLIPS, REFS, conds, {"fake": asr}, lib, fs=FS,
                            workers=4, cache=cache, progress_every=0)
        cache.close()
        assert asr.calls == n_cells, asr.calls
        assert res1["n_called"] == n_cells and res1["n_cached"] == 0
        assert len(res1["rows"]) == n_cells

        # reload from DISK (not the live object) — this is the crash-resume path
        cache2 = ResultCache(cache_path)
        assert len(cache2) == n_cells
        with quiet():
            res2 = run_grid(CLIPS, REFS, conds, {"fake": asr}, lib, fs=FS,
                            workers=4, cache=cache2, progress_every=0)
        cache2.close()

        assert asr.calls == n_cells, f"resume made {asr.calls - n_cells} extra calls"
        assert res2["n_called"] == 0 and res2["n_cached"] == n_cells
        # identical table, and provenance is NOT relabelled by the resuming run
        assert [r["wer"] for r in res2["rows"]] == [r["wer"] for r in res1["rows"]]
        assert [(r["clip_id"], r["condition_name"]) for r in res2["rows"]] == \
               [(r["clip_id"], r["condition_name"]) for r in res1["rows"]]
        assert {r["run_id"] for r in res2["rows"]} == {res1["run_id"]}

        # a torn final line (kill -9 mid-write) must not brick the cache
        with open(cache_path, "a") as f:
            f.write('{"clip_id": "u02", "condition_na')
        with quiet():
            cache3 = ResultCache(cache_path)
        assert len(cache3) == n_cells, "torn line broke the cache load"
    print(f"OK 2: {n_cells} cells cached; the repeat run made ZERO calls "
          f"(counter stayed at {asr.calls}); torn final line survived")


# ---------------------------------------------------------------------------
# 3: failures become rows, never silence, never 0.0
# ---------------------------------------------------------------------------

def test_failures_are_preserved_not_dropped():
    # two independent failure modes:
    #   (a) the composer raises  -> 'road' noise absent from the library
    #   (b) the adapter returns the skip-me sentinel -> clip u17
    lib = make_library(with_road=False)
    asr = FakeASR(fail_clips={"u17"})
    conds = [Condition(0.5, 10.0, "babble", "none", 0.0),      # fine
             Condition(0.5, 10.0, "road", "none", 0.0)]        # missing asset
    n_cells = len(CLIPS) * len(conds)

    with quiet():
        res = run_grid(CLIPS, REFS, conds, {"fake": asr}, lib, fs=FS,
                       workers=2, progress_every=0)
    rows = res["rows"]
    assert len(rows) == n_cells, "a failed cell was DROPPED"

    road = [r for r in rows if r["noise_type"] == "road"]
    assert len(road) == len(CLIPS) and all(r["failed"] is True for r in road)
    assert all("MissingAssetError" in r["error"] for r in road), road[0]["error"]

    sentinel = [r for r in rows if r["clip_id"] == "u17" and r["noise_type"] == "babble"]
    assert len(sentinel) == 1 and sentinel[0]["failed"] is True
    assert "simulated adapter failure" in sentinel[0]["error"]

    # failed rows carry NULL measurements — not 0.0 (a perfect transcription) and
    # not NaN (which would silently survive a mean)
    for r in [*road, *sentinel]:
        for col in ("wer", "transcript", "mean_conf", "n_ref", "edits"):
            assert r[col] is None, (col, r[col])
        assert list(r.keys()) == list(MASTER_COLUMNS)      # still schema-complete

    assert res["n_failed"] == len(road) + len(sentinel)
    assert res["fail_rate"] > 0.02
    ok = [r for r in rows if not r["failed"]]
    assert len(ok) == n_cells - res["n_failed"] and all(r["wer"] is not None for r in ok)
    print(f"OK 3: {res['n_failed']}/{n_cells} failures preserved as rows "
          f"(composer raise + adapter sentinel), measurements NULL, none dropped")


# ---------------------------------------------------------------------------
# 4: make_grid_wer_fn IS the callable design.py wants (proved by using it)
# ---------------------------------------------------------------------------

def test_make_grid_wer_fn_drives_screen_factors():
    lib = make_library()                       # has all three noise types
    asr = FakeASR()
    wer_fn = make_grid_wer_fn(CLIPS, REFS, lib, asr, FS, model="fake")

    v = wer_fn({"rt60": 0.5, "snr_db": 10.0, "noise_type": "babble",
                "codec": "none", "mic_rolloff": 0.0})
    assert isinstance(v, float) and 0.0 <= v <= 1.0, v
    assert asr.calls == len(CLIPS), "wer_fn must average over the WHOLE clip set"

    # a Stage-2 subset sample (survivors only) still works: missing factors are
    # held at their benign end rather than exploding with a KeyError
    v_sub = wer_fn({"rt60": 1.0, "snr_db": 0.0})
    assert isinstance(v_sub, float) and v_sub > v, (v_sub, v)

    # THE ACTUAL CONTRACT TEST: hand it to the real design.screen_factors.
    before = asr.calls
    with mock.patch("deadzone.conditions.apply_codec", _passthrough_codec), quiet():
        res = screen_factors(DEFAULT_FACTOR_SPACE, wer_fn)
    assert res["n_runs"] == 8                                   # PB design for 5 factors
    assert asr.calls - before == res["n_runs"] * len(CLIPS)
    assert len(res["survivors"]) >= 4 and set(res["survivors"]) <= set(DEFAULT_FACTOR_SPACE.names)
    # the planted signs must come back out: more reverb worse, more SNR better
    assert res["effects"]["rt60"] > 0 and res["effects"]["snr_db"] < 0, res["effects"]
    assert res["ranking"][0] in ("rt60", "snr_db")

    # all-clips-failed must RAISE, not fabricate a WER for the harshest corner
    dead = make_grid_wer_fn(CLIPS, REFS, lib, FakeASR(fail_clips=set(REFS)), FS, model="fake")
    try:
        dead({"rt60": 1.0, "snr_db": 0.0, "noise_type": "babble",
              "codec": "none", "mic_rolloff": 0.0})
        assert False, "a wer_fn with zero usable clips must raise, not invent a number"
    except RuntimeError as e:
        assert "refusing to invent" in str(e)
    print(f"OK 4: make_grid_wer_fn -> float, averages the clip set, fills held "
          f"factors, and drives design.screen_factors ({res['n_runs']} runs, "
          f"top factor {res['ranking'][0]})")


# ---------------------------------------------------------------------------
# 5: rir_rt60_measured is the DELIVERED reverb, not the request
# ---------------------------------------------------------------------------

def test_rir_rt60_measured_is_delivered_not_requested():
    lib = make_library()                       # measured assets: 0.2 / 0.5 / 0.9
    asr = FakeASR()
    # ask for 1.0 s: no such asset exists, pick_rir snaps to the 0.9 s one
    row = run_one("u02", CLIPS["u02"], REFS["u02"],
                  Condition(1.0, 10.0, "babble", "none", 0.0), lib, FS,
                  asr, "fake", "run-test")
    assert row["rt60"] == 1.0                                   # the REQUEST
    assert row["rir_key"] == "rir_0.9"
    assert row["rir_rt60_measured"] == 0.9                      # the DELIVERY
    assert row["rir_rt60_measured"] != row["rt60"], (
        "the snap is invisible — regressing on rt60 would attribute 0.9 s of "
        "reverb to a 1.0 s request")
    assert row["noise_key"] == "babble_a"

    # exact hits report exactly what was asked for
    hit = run_one("u02", CLIPS["u02"], REFS["u02"],
                  Condition(0.5, 10.0, "babble", "none", 0.0), lib, FS,
                  asr, "fake", "run-test")
    assert hit["rir_rt60_measured"] == 0.5 and hit["rir_key"] == "rir_0.5"

    # a failed cell still records what WOULD have been delivered
    with quiet():
        failed = run_one("u02", CLIPS["u02"], REFS["u02"],
                         Condition(1.0, 10.0, "babble", "none", 0.0), lib, FS,
                         FakeASR(fail_clips={"u02"}), "fake", "run-test")
    assert failed["failed"] is True and failed["rir_rt60_measured"] == 0.9
    print("OK 5: rir_rt60_measured = closest MEASURED asset (1.0 requested -> "
          "0.9 delivered), recorded even on failed rows")


# ---------------------------------------------------------------------------
# 6: --dry-run prices the plan and calls nothing
# ---------------------------------------------------------------------------

def test_dry_run_plans_and_prices_without_calling():
    grid = main_grid()
    clips = ["u02", "u05"]
    plan = build_plan(clips, grid, ["nova-3", "whisper-base"])
    assert plan["n_total"] == len(clips) * len(grid) * 2
    assert plan["n_remaining"] == plan["n_total"] and plan["n_cached"] == 0
    # only the cloud arm is billed; whisper is local and free
    assert plan["n_billed"] == len(clips) * len(grid)
    assert plan["per_model_remaining"]["whisper-base"] == len(clips) * len(grid)
    assert plan["est_cost_usd"] > 0 and plan["est_audio_minutes"] > 0

    txt = format_plan(plan)
    assert "est. cost" in txt and "$" in txt and grid[0].name in txt
    assert "full call list" not in txt
    assert "full call list" in format_plan(plan, verbose=True)

    # a cached cell drops out of the plan (and out of the cost)
    with tempfile.TemporaryDirectory() as td:
        cache = ResultCache(Path(td) / "cache.jsonl")
        cache.append(run_one("u02", CLIPS["u02"], REFS["u02"], grid[0],
                             make_library(), FS, FakeASR(), "nova-3", "r"))
        cache.close()
        plan2 = build_plan(clips, grid, ["nova-3"], cache=cache)
        assert plan2["n_cached"] == 1 and plan2["n_remaining"] == plan2["n_total"] - 1

        # the CLI path: --dry-run must not need data/rirs or the recordings, and
        # must not touch the network. It only reads the manifest.
        with quiet() as out:
            rc = main(["--dry-run", "--clips", "smoke", "--results", td,
                       "--models", "nova-3,whisper-base"])
        assert rc == 0 and "DRY RUN" in out.getvalue() and "est. cost" in out.getvalue()
        assert not (Path(td) / "master.csv").exists(), "dry run wrote output"
    print(f"OK 6: dry run plans {plan['n_total']} cells, "
          f"${plan['est_cost_usd']:.2f} est., zero calls, zero writes")


# ---------------------------------------------------------------------------
# 7: outputs — csv mirror always; parquet degrades instead of crashing
# ---------------------------------------------------------------------------

def test_write_master_degrades_to_csv_without_pyarrow():
    asr = FakeASR()
    rows = [run_one(cid, CLIPS[cid], REFS[cid],
                    Condition(0.5, 10.0, "babble", "none", 0.0),
                    make_library(), FS, asr, "fake", "run-test") for cid in CLIPS]

    with tempfile.TemporaryDirectory() as td, quiet():
        paths = write_master(rows, td)                 # whatever this host has
        assert Path(paths["csv"]).exists()
        # force the no-parquet host: the expensive part (the API calls) already
        # happened, so a missing optional dep must NOT lose the run
        with mock.patch.dict(sys.modules, {"pandas": None}):
            paths2 = write_master(rows, td, basename="master_csvonly")
        assert "parquet" not in paths2 and Path(paths2["csv"]).exists()
        lines = Path(paths2["csv"]).read_text().splitlines()
    assert len(lines) == len(rows) + 1                 # header + one row per cell
    print(f"OK 7: master.csv always written ({len(rows)} rows); parquet absent -> "
          f"csv only, no crash")


# ---------------------------------------------------------------------------
# 8: the grid + clip subsets are the fixed, documented ones
# ---------------------------------------------------------------------------

def test_grid_and_clip_subsets_are_fixed_constants():
    grid = main_grid()
    assert 50 <= len(grid) <= 70, len(grid)                   # R4.1: "~60 conditions"
    names = [c.name for c in grid]
    assert len(set(names)) == len(names), "duplicate conditions in the grid"
    assert main_grid() == grid, "grid must be deterministic across calls"
    for c in grid:                                            # names round-trip
        assert Condition.from_name(c.name) == c
    for f in DEFAULT_FACTOR_SPACE.factors:                    # values inside the space
        vals = [getattr(c, f.name) for c in grid]
        if f.kind == "continuous":
            assert min(vals) >= f.low and max(vals) <= f.high, f.name
        else:
            assert set(vals) <= set(f.levels), f.name
    assert {c.codec for c in grid} == {"none", "g726", "opus-lowrate"}
    assert {c.noise_type for c in grid} == {"babble", "engine", "road"}

    manifest = load_manifest("recording_manifest.csv")         # R1.1 constants
    assert SMOKE_CLIP == "u02" and SMOKE_CLIP in manifest
    assert len(AL_CLIPS) == 10 and len(set(AL_CLIPS)) == 10
    assert SOBOL_CLIPS == AL_CLIPS
    assert all(c in manifest for c in AL_CLIPS), "AL clip set not in the manifest"
    print(f"OK 8: {len(grid)} unique grid conditions, all in-space and name-"
          f"round-tripping; clip subsets pinned (smoke={SMOKE_CLIP}, AL={len(AL_CLIPS)})")


# ---------------------------------------------------------------------------
# 9: the PCM_16 peak guard (a writer-induced degradation we must not measure)
# ---------------------------------------------------------------------------

def test_peak_guard_prevents_silent_clipping():
    x = 1.8 * np.sin(2 * np.pi * 200 * np.arange(FS) / FS)     # way over full scale
    stats = RunStats()
    with tempfile.TemporaryDirectory() as td:
        hot = Path(td) / "hot.wav"
        assert write_degraded_wav(hot, x, FS) is True          # guard fired
        y, _ = sf.read(hot)
        assert np.max(np.abs(y)) < 1.0
        # UNIFORM GAIN, not clipping: the written signal is the input times one
        # scalar (to PCM_16 quantization). A clipped signal would flatten at the
        # rail and diverge from this by ~0.8 in the peaks.
        assert np.allclose(y, x * (0.99 / 1.8), atol=1e-4), "signal was clipped, not scaled"
        quiet_sig = 0.2 * x / 1.8
        assert write_degraded_wav(Path(td) / "ok.wav", quiet_sig, FS) is False

        # and the runner reports it rather than hiding it
        loud = {cid: 3.0 * CLIPS[cid] for cid in CLIPS}
        with quiet() as out:
            run_grid(loud, REFS, [Condition(0.2, 25.0, "babble", "none", 0.0)],
                     {"fake": FakeASR()}, make_library(), fs=FS, workers=2,
                     progress_every=0, stats=stats)
        assert stats.get("peak_guard") > 0 and "peak guard" in out.getvalue()
    print(f"OK 9: peak guard rescales instead of letting PCM_16 clip "
          f"({stats.get('peak_guard')} clip(s) rescaled and reported)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} RUN-EXPERIMENT TESTS PASSED — the grid runner writes "
          f"a frozen schema, resumes for free, and never loses a failure.")
