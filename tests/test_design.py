"""
Synthetic validation for design.py (Task C). No audio, no API — a wer_fn with
KNOWN planted structure stands in for the real pipeline, and we assert the
machinery recovers what we planted. If it can find structure we buried on
purpose, we trust it to find real structure later.

Planted into the synthetic WER surface:
  * additive main effects on all five factors,
  * ONE strong pairwise interaction: rt60 x snr_db (degradations compound),
  * ONE non-monotonic bump: WER dips in a mid-SNR sweet spot (the "a little
    noise nudges toward the training distribution" effect).

Asserted:
  * Stage-1 screen keeps the interacting pair (never silently drops it),
  * Sobol S2 ranks rt60 x snr_db as the top interacting pair,
  * those two factors carry the largest ST - S1 interaction gap,
  * the counterintuitive detector flags the planted non-monotonic dip on snr_db.

Deterministic: fixed seeds throughout.  Run:  python3 tests/test_design.py
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

from deadzone.design import (
    DEFAULT_FACTOR_SPACE, screen_factors, sobol_indices,
    format_sobol_tables, find_counterintuitive_cells,
)

# ---- planted synthetic WER surface -----------------------------------------

SNR0, SNR_W = 8.0, 3.0     # non-monotonic dip location / width (dB)
K_INT = 0.40               # strength of the planted rt60 x snr interaction
K_BUMP = 0.20              # depth of the planted non-monotonic dip

_NOISE = {"babble": 0.10, "engine": 0.05, "road": 0.02}
_CODEC = {"none": 0.00, "g726": 0.06, "opus-lowrate": 0.12}


def _norm(x, lo, hi):
    return (x - lo) / (hi - lo)


def planted_wer_fn(s: dict) -> float:
    r = _norm(s["rt60"], 0.2, 1.0)          # 0..1, worse with reverb
    q = _norm(s["snr_db"], 0.0, 25.0)       # 0..1, cleaner with SNR
    main = (0.20 * r + 0.25 * (1 - q)
            + _NOISE[s["noise_type"]] + _CODEC[s["codec"]]
            + 0.10 * s["mic_rolloff"])
    interaction = K_INT * r * (1 - q)                              # compounding
    bump = K_BUMP * math.exp(-((s["snr_db"] - SNR0) ** 2) / (2 * SNR_W ** 2))
    return float(min(max(0.05 + main + interaction - bump, 0.0), 1.0))


# ---- 1: Stage-1 screen keeps the interacting pair --------------------------

def test_screen_keeps_interactors():
    res = screen_factors(DEFAULT_FACTOR_SPACE, planted_wer_fn)
    assert len(res["survivors"]) >= 4, res["survivors"]
    assert "rt60" in res["survivors"] and "snr_db" in res["survivors"], res["survivors"]
    # every factor has a real planted main effect, so none should be dropped here
    assert res["dropped"] == [], res["dropped"]
    print(f"OK 1: screen ran {res['n_runs']} PB runs, ranking={res['ranking']}, "
          f"survivors={res['survivors']}")


# ---- 2 & 3: Sobol recovers the interacting pair (S2 + ST-S1 gap) -----------

def test_sobol_recovers_planted_interaction():
    space = DEFAULT_FACTOR_SPACE
    res = sobol_indices(space, planted_wer_fn, N=1024, seed=0)

    top_pair = set(res["s2_ranked"][0]["pair"])
    assert top_pair == {"rt60", "snr_db"}, res["s2_ranked"][:3]
    # NOT a magnitude threshold. SPEC §5: S2 says WHICH pair interacts, never HOW
    # MUCH -- its bootstrap CI crossed zero even with this planted interaction at
    # N=1024, so asserting a floor on its value would be testing noise. The
    # ranking above is the only claim S2 is allowed to support; we check it is
    # positive and that its own CI is reported as wide enough to bar a magnitude
    # reading.
    assert res["s2_ranked"][0]["S2"] > 0.0, res["s2_ranked"][0]
    assert res["s2_ranked"][0]["S2_conf"] > res["s2_ranked"][0]["S2"], (
        "if S2's CI ever tightens below its estimate, revisit SPEC §5's guidance "
        "that S2 magnitude is unusable")

    # the two interactors carry the largest ST - S1 gaps; the pure main-effect
    # factors (noise/codec/mic) sit well below them.
    gap = {d["factor"]: d["gap"] for d in res["interaction_gap"]}
    top_two_gap = {res["interaction_gap"][0]["factor"], res["interaction_gap"][1]["factor"]}
    assert top_two_gap == {"rt60", "snr_db"}, res["interaction_gap"]
    for other in ("noise_type", "codec", "mic_rolloff"):
        assert gap["rt60"] > gap[other] and gap["snr_db"] > gap[other], (gap, other)

    print(f"OK 2/3: n_eval={res['n_eval']}; top S2 pair = rt60 x snr_db "
          f"(S2={res['s2_ranked'][0]['S2']:.3f}); "
          f"ST-S1 gap rt60={gap['rt60']:.3f}, snr_db={gap['snr_db']:.3f}")
    print("\n--- Sobol tables on the planted function ---")
    print(format_sobol_tables(res))
    print("--- end tables ---\n")
    return res


# ---- 4: counterintuitive detector flags the planted non-monotonic dip ------

def test_detector_flags_nonmonotonic_bump():
    space = DEFAULT_FACTOR_SPACE
    cells = find_counterintuitive_cells(space, planted_wer_fn, grid=11, top_k=8)
    nm = cells["non_monotonic"]
    assert nm, "detector found no non-monotonic cells"

    # the strongest dip must be on snr_db, near the planted sweet spot SNR0=8dB
    strongest = nm[0]
    assert strongest["factor"] == "snr_db", strongest
    assert abs(strongest["at_value"] - SNR0) <= 4.0, strongest

    # and every non-monotonic cell we flagged is on snr_db (the only planted one);
    # the monotone factors (rt60/mic) must NOT produce spurious dips.
    assert all(c["factor"] == "snr_db" for c in nm), [c["factor"] for c in nm]

    print(f"OK 4: flagged {len(nm)} non-monotonic cells, all on snr_db; "
          f"strongest dip at snr_db={strongest['at_value']:.2f} dB "
          f"(planted {SNR0} dB), depth={strongest['depth']:.3f}")
    print("    top counterintuitive cell coords:", strongest["coords"])
    if cells["reversals"]:
        print("    (effect reversals also flagged:",
              [(r["factor"], r["moderator"]) for r in cells["reversals"]], ")")
    else:
        print("    (no effect reversals — none were planted; detector returned clean)")


if __name__ == "__main__":
    print("factor space:")
    for f in DEFAULT_FACTOR_SPACE.factors:
        rng = f"[{f.low}, {f.high}]" if f.kind == "continuous" else f"{f.levels}"
        print(f"  {f.name:<12} {f.kind:<12} {rng}"
              + (f"  (worse as value goes {f.degradation})" if f.degradation else ""))
    print()
    test_screen_keeps_interactors()
    test_sobol_recovers_planted_interaction()
    test_detector_flags_nonmonotonic_bump()
    print("\nAll design tests passed — screen + Sobol + counterintuitive detector "
          "recover the planted interaction AND the planted non-monotonic bump.")
