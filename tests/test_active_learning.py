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

from deadzone.design import DEFAULT_FACTOR_SPACE
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


if __name__ == "__main__":
    print("factor space:", SPACE.names)
    print(f"planted oracle: sharp sigmoid (K={K_SHARP}) across the rt60==snr_db "
          f"line; threshold={DEFAULT_THRESHOLD}\n")
    test_planted_boundary_is_sharp()
    test_surrogate_fits()
    test_acquisition_strategies()
    test_active_beats_random()
    print("\nAll active-learning tests passed — the surrogate maps the planted "
          "failure boundary, and active sampling reaches target fidelity in far "
          "fewer oracle calls than random.")
