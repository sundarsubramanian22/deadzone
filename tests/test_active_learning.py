"""
Synthetic validation for active_learning.py (Task D3). No audio, no API — a
PLANTED oracle with a KNOWN failure boundary stands in for the real pipeline, and
we assert the active loop (a) reconstructs that boundary to a target accuracy
within a fixed budget, and (b) gets there in meaningfully fewer oracle calls than
random sampling. If it can map a boundary we drew on purpose, we trust it to map
the real one later.

Planted oracle (known boundary):
  WER is a SHARP SIGMOID across a line in the rt60 x snr_db plane:
      z = rt60_norm + (1 - snr_norm) - 1        (boundary at z = 0)
      WER = sigmoid(K_SHARP * z)                (clean below the line, failing above)
  plus small nuisance offsets from the categorical/mic factors, so the surrogate
  must find a mostly-2-D contour embedded in the full 5-D space.

Deterministic (fixed seeds), no GPU, runs in a few seconds.  Run:
    python3 tests/test_active_learning.py
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
import math

import numpy as np

from deadzone.design import DEFAULT_FACTOR_SPACE, Factor, FactorSpace
from deadzone.active_learning import (
    GPSurrogate, acquire, active_learn, random_baseline, compare_arms,
    learning_curve, evals_to_target, best_so_far, make_test_set, boundary_error,
    lhs_raw, sobol_pool_raw, DEFAULT_THRESHOLD,
)

SPACE = DEFAULT_FACTOR_SPACE
K_SHARP = 7.0                       # sigmoid steepness (sharp transition)
_NOISE = {"babble": 0.06, "engine": 0.03, "road": 0.0}
_CODEC = {"none": 0.0, "g726": 0.04, "opus-lowrate": 0.08}


def _norm(x, lo, hi):
    return (x - lo) / (hi - lo)


def planted_oracle(s: dict) -> float:
    """WER with a sharp, KNOWN failure contour in the rt60 x snr_db plane."""
    r = _norm(s["rt60"], 0.2, 1.0)          # 0..1, worse with reverb
    q = _norm(s["snr_db"], 0.0, 25.0)       # 0..1, cleaner with SNR
    z = r + (1.0 - q) - 1.0                  # boundary line: r == q
    z += 0.12 * (s["mic_rolloff"] - 0.5)     # small nuisance tilt
    z += _NOISE[s["noise_type"]] + _CODEC[s["codec"]] - 0.06
    return 1.0 / (1.0 + math.exp(-K_SHARP * z))


# ---- 0: the planted boundary really is sharp (sanity on the oracle) --------

def test_planted_boundary_is_sharp():
    # far below the line -> ~clean; far above -> ~failing
    clean = planted_oracle({"rt60": 0.2, "snr_db": 25.0, "noise_type": "road",
                            "codec": "none", "mic_rolloff": 0.0})
    fail = planted_oracle({"rt60": 1.0, "snr_db": 0.0, "noise_type": "babble",
                           "codec": "opus-lowrate", "mic_rolloff": 1.0})
    assert clean < 0.1 and fail > 0.9, (clean, fail)
    # on the line (r == q) WER sits near 0.5
    mid = planted_oracle({"rt60": 0.6, "snr_db": 12.5, "noise_type": "engine",
                          "codec": "g726", "mic_rolloff": 0.5})
    assert abs(mid - 0.5) < 0.2, mid
    print(f"OK 0: planted boundary sharp — clean={clean:.3f}, on-line={mid:.3f}, "
          f"fail={fail:.3f}")


# ---- 1: GP surrogate fits and predicts sensibly ----------------------------

def test_surrogate_fits():
    X = lhs_raw(SPACE, 40, seed=0)
    y = np.array([planted_oracle(SPACE.decode(r)) for r in X])
    gp = GPSurrogate(SPACE, seed=0).fit(X, y)
    mean, std = gp.predict(X, return_std=True)
    train_rmse = float(np.sqrt(np.mean((mean - y) ** 2)))
    assert train_rmse < 0.1, train_rmse
    assert np.all(std >= 0)
    # held-out generalization is at least reasonable
    Xt, yt = make_test_set(planted_oracle, SPACE, n=256, seed=5)
    homean, _ = gp.predict(Xt, return_std=True)
    ho_rmse = float(np.sqrt(np.mean((homean - yt) ** 2)))
    assert ho_rmse < 0.15, ho_rmse
    print(f"OK 1: GP fits (train RMSE={train_rmse:.3f}, held-out RMSE={ho_rmse:.3f})")


# ---- 2: both acquisition strategies pick valid, sensible points ------------

def test_acquisition_strategies():
    X = lhs_raw(SPACE, 15, seed=1)
    y = np.array([planted_oracle(SPACE.decode(r)) for r in X])
    gp = GPSurrogate(SPACE, seed=1).fit(X, y)
    pool = sobol_pool_raw(SPACE, 400, seed=2)

    ju = acquire("uncertainty", gp, pool, X)
    jb = acquire("boundary", gp, pool, X, threshold=DEFAULT_THRESHOLD)
    assert 0 <= ju < len(pool) and 0 <= jb < len(pool)

    # boundary pick should land near the 0.5 contour: |predicted - 0.5| is small
    mb, _ = (float(v[0]) for v in gp.predict([pool[jb]]))
    mu, su = (float(v[0]) for v in gp.predict([pool[ju]]))
    # uncertainty pick is a high-std point; boundary pick sits near the threshold
    assert abs(mb - 0.5) < 0.25, mb
    print(f"OK 2: strategies valid — boundary pick pred={mb:.3f} (near 0.5), "
          f"uncertainty pick std={su:.3f}")


# ---- 3 & 4: THE FINDING — active reconstructs the boundary in fewer calls ---

QUALITY_BAR = 0.045          # active must map the boundary at least this well
METRIC = "boundary_rmse"     # headline: surrogate RMSE near the failure contour

def test_active_beats_random():
    res = compare_arms(planted_oracle, SPACE, n_seed=15, budget=30,
                       pool_size=600, seed=0, test_n=2048)
    n_total = res["n_total"]
    # best-so-far (monotone) curves make "oracle calls to reach fidelity X" robust
    cb = best_so_far(res["curves"]["active_boundary"], METRIC)
    cr = best_so_far(res["curves"]["random"], METRIC)
    cu = best_so_far(res["curves"]["active_uncertainty"], METRIC)

    final_active = cb[-1][METRIC]
    final_random = cr[-1][METRIC]

    # (a) active reconstructs the KNOWN boundary to a target accuracy in-budget
    assert final_active <= QUALITY_BAR, (final_active, cb)
    # (b) at equal (full) budget, active maps the boundary better than random
    assert final_active <= 0.9 * final_random, (final_active, final_random)

    # (c) EQUAL-ACCURACY, FEWER CALLS: how many oracle calls each arm needs to
    #     reach the boundary fidelity random achieves with its WHOLE budget.
    target = final_random
    ea = evals_to_target(cb, target, METRIC)
    er = evals_to_target(cr, target, METRIC)
    assert math.isfinite(ea), ("active never reached random's fidelity", cb)
    assert ea < er, (f"active {ea} vs random {er} — no speedup", cb, cr)
    assert er - ea >= 3, (f"margin {er - ea} calls too small", cb, cr)

    print("\n--- boundary-fidelity (near-contour RMSE, best-so-far) vs oracle calls ---")
    print(f"  {'n_evals':>7} | {'active(bdy)':>11} | {'active(unc)':>11} | {'random':>8}")
    cbm = {p["n_evals"]: p[METRIC] for p in cb}
    cum = {p["n_evals"]: p[METRIC] for p in cu}
    crm = {p["n_evals"]: p[METRIC] for p in cr}
    for n in sorted(set(cbm) | set(cum) | set(crm)):
        print(f"  {n:>7} | {cbm.get(n, float('nan')):>11.3f} | "
              f"{cum.get(n, float('nan')):>11.3f} | {crm.get(n, float('nan')):>8.3f}")
    print("--- end ---")
    print(f"OK 3: active(boundary) near-contour RMSE = {final_active:.3f} "
          f"(<= {QUALITY_BAR}); random = {final_random:.3f} "
          f"({final_random / final_active:.2f}x worse at equal budget)")
    print(f"OK 4: to reach random's full-{n_total}-call fidelity ({target:.3f}), "
          f"active needed {ea:.0f} calls vs random's {er:.0f} — "
          f"active saved {er - ea:.0f} oracle calls")
    return res


# ===========================================================================
# 5-7: THE DRR REPARAMETERISATION RE-RUN  (scripts/run_al_drr.py)
# ===========================================================================
#
# The real-grid D3b result is a NULL, and `results/interaction_report.txt` names
# the obvious suspect: the GP is fitted with `rt60` as a CONTINUOUS coordinate,
# but `AssetLibrary.resolve()` snaps each rt60 request to the nearest measured
# RIR, so every level of that axis is a different real ROOM. The re-run swaps the
# coordinate to DRR and asks whether the null survives.
#
# The comparison is only meaningful if the coordinate is the ONLY thing that
# changed, so the two properties below are pinned here rather than asserted in
# prose: (5) every condition takes the coordinate of the RIR it actually resolved
# to, and (6) both arms race on one shared oracle and one shared test set. Test 7
# checks the identity guard can actually DETECT a difference — a check that
# always passes would license exactly the confound it exists to rule out.

_ARTIFACTS = ("results/master.csv",)

# The published DRR/C50 table from results/interaction_report.txt, produced by
# scripts/run_d3a.py. `rir_acoustics` must reproduce it or the two have drifted.
_PUBLISHED_ACOUSTICS = {
    "mit_rt60-0.20_h114_Restaurant_txts.wav":            (16.90, 28.10),
    "mit_rt60-0.47_h174_Bar_1txts.wav":                  (-2.05, 10.22),
    "mit_rt60-0.68_h058_Campground_Dininghall_3txts.wav": (4.26, 10.03),
    "mit_rt60-0.99_h081_Shower_2txts.wav":              (-10.02, 2.12),
}


def _have(*paths) -> bool:
    return all(_os.path.exists(_os.path.join(_REPO_ROOT, p)) for p in paths)


def _fake_rows(coord_col="drr_db"):
    """Two rooms x two other-factor settings, in master-table shape."""
    base = {"snr_db": "10.0", "noise_type": "babble", "codec": "none",
            "mic_rolloff": "0.0", "wer": "0.4", "failed": "False",
            "model": "nova-3", "condition_name": "c"}
    rows = []
    for key, rt in (("/rirs/roomA.wav", "0.2"), ("/rirs/roomB.wav", "1.0")):
        for snr in ("10.0", "0.0"):
            r = dict(base)
            r.update({"rir_key": key, "rt60": rt, "rir_rt60_measured": rt,
                      "snr_db": snr})
            rows.append(r)
    return rows


def test_coordinate_maps_to_the_resolved_rir():
    """
    5: each condition takes the coordinate of the RIR IT ACTUALLY RESOLVED TO.

    The join key is the master table's `rir_key` — the field that records which
    RIR the composer really used — never the requested `rt60`, because
    `resolve()` snaps requests to the nearest measured RIR and the two are
    different objects. An unmapped key must RAISE: silently dropping a room would
    delete a quarter of the reverb axis and still produce a plausible number.
    """
    from scripts.run_al_drr import remap_rows, Coordinate

    values = {"/rirs/roomA.wav": 16.9, "/rirs/roomB.wav": -10.02}
    coord = Coordinate("drr_db", "test", "primary", values, (-10.02, 16.9))
    out = remap_rows(_fake_rows(), coord)
    assert len(out) == 4, len(out)
    for r in out:
        assert r["drr_db"] == values[r["rir_key"]], r
        # nothing else may move
        assert r["snr_db"] in ("10.0", "0.0") and r["noise_type"] == "babble"
    # rows sharing a room share a coordinate; different rooms do not
    a = {r["drr_db"] for r in out if r["rir_key"].endswith("roomA.wav")}
    b = {r["drr_db"] for r in out if r["rir_key"].endswith("roomB.wav")}
    assert a == {16.9} and b == {-10.02}, (a, b)

    # an unmapped room must raise, not vanish
    partial = Coordinate("drr_db", "t", "primary",
                         {"/rirs/roomA.wav": 1.0}, (0.0, 1.0))
    try:
        remap_rows(_fake_rows(), partial)
    except KeyError as e:
        assert "roomB" in str(e), e
    else:
        raise AssertionError("remap_rows silently dropped an unmapped rir_key")

    print("OK 5: coordinate joins on rir_key (the RIR actually resolved to); "
          "an unmapped room raises instead of shrinking the axis")


def test_coordinate_mapping_against_the_real_grid():
    """
    5b: the same mapping on the REAL master table, cross-checked against the
    published DRR/C50 table, and confirming a coordinate swap changes ONLY the
    reverb column — same conditions, same WERs, same ordering.
    """
    if not _have(*_ARTIFACTS):
        print("SKIP 5b: results/master.csv absent (gitignored) — pure-function "
              "mapping is still covered by test 5")
        return
    import numpy as _np
    from scripts.run_al_drr import (
        measure_delivered_rooms, primary_coordinates, remap_rows,
    )
    from deadzone.analysis.interactions import condition_matrix, load_master_rows

    rows = load_master_rows("results/master.csv")
    try:
        rooms = measure_delivered_rooms(rows)
    except (FileNotFoundError, RuntimeError):
        print("SKIP 5b: RIR files absent (data/ is gitignored)")
        return

    for r in rooms:
        want = _PUBLISHED_ACOUSTICS.get(r["room"])
        if want is None:
            continue
        assert abs(r["drr_db"] - want[0]) < 0.02, (r["room"], r["drr_db"], want)
        assert abs(r["c50_db"] - want[1]) < 0.02, (r["room"], r["c50_db"], want)

    coords = primary_coordinates(rooms)
    nova = [r for r in rows if r.get("model") == "nova-3"]
    ref = condition_matrix(nova, use_measured_rt60=True)
    for c in coords:
        remapped = remap_rows(rows, c)
        # every row carries its OWN room's coordinate
        for r in remapped[:200]:
            assert r[c.name] == c.values[r["rir_key"]]
        mat = condition_matrix(remapped, c.space(), model="nova-3",
                               use_measured_rt60=False)
        assert mat["n_conditions"] == ref["n_conditions"], (c.name, mat["n_conditions"])
        assert _np.allclose(mat["y"], ref["y"]), c.name
        assert ([x["condition_name"] for x in mat["conditions"]]
                == [x["condition_name"] for x in ref["conditions"]]), c.name
    print(f"OK 5b: {len(rooms)} delivered rooms match the published DRR/C50 table; "
          f"all {len(coords)} coordinates keep the same {ref['n_conditions']} "
          f"conditions, WERs and ordering — only the reverb axis moves")


def test_both_arms_share_one_oracle_and_one_test_set():
    """
    6: within a coordinate system, active and random race on ONE oracle and ONE
    held-out set.

    If the arms saw different oracles or different yardsticks, any difference
    between them would be unattributable — and so would any difference between
    coordinate systems built on top of them. `run_seed` is the single place that
    contract lives, so it is checked directly: one oracle object serves every
    call, all three arms spend the same budget, and one test set scores them all.
    """
    from deadzone.analysis.al_savings import run_seed

    calls = {"n": 0, "owners": set()}

    class RecordingOracle:
        def __call__(self, sample):
            calls["n"] += 1
            calls["owners"].add(id(self))
            return planted_oracle(sample)

    oracle = RecordingOracle()
    Xt, yt = make_test_set(planted_oracle, SPACE, n=128, seed=11)
    n_seed, budget = 6, 4
    res = run_seed(oracle, SPACE, seed=0, X_test=Xt, y_test=yt,
                   n_seed=n_seed, budget=budget)

    assert set(res["curves"]) == {"active_boundary", "active_uncertainty", "random"}
    # ONE oracle instance served every call, and every arm spent the same budget
    assert calls["owners"] == {id(oracle)}, calls["owners"]
    assert calls["n"] == 3 * (n_seed + budget), calls["n"]
    for arm, t in res["trajectories"].items():
        assert t["n_evals"] == n_seed + budget, (arm, t["n_evals"])
    # ONE test set scored all three arms, and it cost zero oracle calls to build
    assert res["n_test_points"] == len(yt), (res["n_test_points"], len(yt))
    assert res["test_oracle_calls"] == 0
    xs = {tuple(p["n_evals"] for p in c) for c in res["curves"].values()}
    assert len(xs) == 1, ("arms scored on different checkpoint schedules", xs)

    print(f"OK 6: one oracle instance served all {calls['n']} calls, all three "
          f"arms spent {n_seed + budget} evals on one shared "
          f"{res['n_test_points']}-point held-out set")


def test_split_robustness_fits_the_oracle_in_the_given_space():
    """
    6b: `split_robustness` must build its surrogate oracle in the space it was
    HANDED, not in DEFAULT_FACTOR_SPACE.

    `surrogate_oracle_from_master` encodes every sample with
    `encode_sample(space, ...)`, and it defaults `space` to DEFAULT_FACTOR_SPACE.
    `split_robustness` used to omit the argument, so the arms sampled and were
    scored in the caller's space while the oracle scored them in the default one.
    With the default space that is a no-op — which is exactly why it survived —
    but any renamed axis (the DRR re-run) breaks, and any custom space that
    merely REORDERS or re-bounds the same factor names breaks SILENTLY.
    """
    import inspect
    from deadzone.analysis import al_savings as ALS

    src = inspect.getsource(ALS.split_robustness)
    call = src[src.index("surrogate_oracle_from_master"):]
    call = call[:call.index(")") + 1]
    assert "space=space" in call, (
        "split_robustness must forward `space` to surrogate_oracle_from_master; "
        f"found: {call!r}")

    # and behaviourally: a renamed axis must not blow up or fall back
    renamed = FactorSpace(
        [Factor("reverb_x", "continuous", low=0.0, high=1.0, degradation="up")]
        + list(SPACE.factors[1:]))
    split = {"X_train": lhs_raw(renamed, 24, seed=0),
             "y_train": None, "n_train": 24, "n_test": 0}
    split["y_train"] = np.array(
        [planted_oracle({**renamed.decode(r), "rt60": 0.2 + 0.8 * r[0]})
         for r in split["X_train"]])
    oracle, _ = ALS.surrogate_oracle_from_master(split=split, space=renamed, seed=0)
    v = oracle(renamed.decode(lhs_raw(renamed, 1, seed=3)[0]))
    assert math.isfinite(v), v
    print("OK 6b: split_robustness forwards `space` to the oracle factory; a "
          "renamed reverb axis is fitted in its own coordinates")


def test_identity_guard_detects_a_different_test_set():
    """
    7: the cross-coordinate identity guard can actually FAIL.

    `identity_check` is what licenses comparing coordinate systems at all — it
    asserts each one held out the same conditions with the same WERs. A guard
    that cannot detect a difference would silently license the confound it exists
    to rule out, so feed it a corrupted fingerprint and require a NO.
    """
    from scripts.run_al_drr import identity_check

    def fp(y_test, names, near=3):
        return {"name": "x", "split_fingerprint": {
            "n_conditions": 10, "n_train": 6, "n_test": len(y_test),
            "n_test_near_boundary": near, "y_train_sum": 1.0,
            "y_test_sum": float(sum(y_test)), "y_test": list(y_test),
            "test_condition_names": list(names), "train_test_shared_rows": 0}}

    good = [dict(fp([0.1, 0.5, 0.9], ["a", "b", "c"]), name=n) for n in ("p", "q")]
    assert identity_check(good)["identical_test_set_across_coordinates"] is True

    for label, bad in (
        ("different WERs", fp([0.1, 0.5, 0.8], ["a", "b", "c"])),
        ("different conditions", fp([0.1, 0.5, 0.9], ["a", "b", "z"])),
        ("different near-boundary count", fp([0.1, 0.5, 0.9], ["a", "b", "c"], near=2)),
    ):
        bad = dict(bad, name="bad")
        res = identity_check([good[0], bad])
        assert res["identical_test_set_across_coordinates"] is False, label
        assert "bad" in res["coordinates_differing"], label

    print("OK 7: identity guard flags a changed test set (WERs, conditions or "
          "near-boundary count) instead of waving it through")


if __name__ == "__main__":
    print("factor space:", SPACE.names)
    print(f"planted oracle: sharp sigmoid (K={K_SHARP}) across the rt60==snr_db "
          f"line; threshold={DEFAULT_THRESHOLD}\n")
    test_planted_boundary_is_sharp()
    test_surrogate_fits()
    test_acquisition_strategies()
    test_active_beats_random()
    print()
    test_coordinate_maps_to_the_resolved_rir()
    test_coordinate_mapping_against_the_real_grid()
    test_both_arms_share_one_oracle_and_one_test_set()
    test_split_robustness_fits_the_oracle_in_the_given_space()
    test_identity_guard_detects_a_different_test_set()
    print("\nAll active-learning tests passed — the surrogate maps the planted "
          "failure boundary, and active sampling reaches target fidelity in far "
          "fewer oracle calls than random. The DRR re-run's coordinate mapping "
          "and its shared-oracle/shared-test-set contract are pinned.")
