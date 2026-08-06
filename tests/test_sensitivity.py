"""
test_sensitivity.py — offline tests for analysis/sensitivity.py (R4.3 / R4.5).

Fully synthetic: plant a response surface whose exact Sobol decomposition is known
in closed form, then assert the code recovers it. No audio, no API, no master
table.

WHY THESE PARTICULAR TESTS. The failure mode of a variance decomposition is not a
crash — a transposed axis or a botched Mobius subtraction yields indices that are
in [0, 1], plausibly ranked, and wrong, and nothing downstream can tell. Three
independent invariants catch it:

  1. PARTITION. sum over all subsets of S_u == 1 exactly. Catches most indexing
     bugs immediately because a mis-indexed term double-counts or drops variance.
  2. ADDITIVE => NO INTERACTION. For a purely additive surface every S2 must be 0
     and ST must equal S1 for every factor. A decomposition that leaks main-effect
     variance into interaction terms fails here even though it still sums to 1.
  3. PLANTED INTERACTION => GAP IN EXACTLY THE RIGHT PLACE. Catches the opposite
     error: a decomposition that attributes an interaction to the wrong pair.

Run: python3 tests/test_sensitivity.py
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

import itertools
import json
import math
import os
import sys
import tempfile

import numpy as np


from deadzone.analysis.sensitivity import (            # noqa: E402
    anova_variance_terms, _check_partition, sobol_from_terms, _subsets,
    decompose, load_factorial, measured_counterintuitive,
    screen_from_factorial, to_json,
    PRIMARY_FACTORS, PRIMARY_FIXED,
)
from deadzone.analysis.interactions import normalize_sobol_result, load_sobol_json  # noqa: E402


_FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        _FAILS.append(name)


def close(a, b, tol=1e-9) -> bool:
    return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) <= tol))


# ---------------------------------------------------------------------------
# helpers: build a (n_clips, *shape) block with a planted cell-mean surface
# ---------------------------------------------------------------------------

def make_block(Ycells: np.ndarray, factors, levels, n_clips=40, clip_sd=0.0, seed=0):
    """Wrap a planted cell-mean surface as a factorial block.

    `clip_sd` adds a per-clip additive offset (the same offset in every cell), which
    is exactly the correlation structure the clip bootstrap exists to handle: it
    moves the grand mean but leaves every ANOVA effect untouched, so the point
    estimates must be unchanged while the CIs stay finite.
    """
    rng = np.random.default_rng(seed)
    W = np.broadcast_to(Ycells, (n_clips,) + Ycells.shape).copy()
    if clip_sd:
        W += rng.normal(0.0, clip_sd, size=(n_clips,) + (1,) * Ycells.ndim)
    return {"W": W, "factors": list(factors),
            "levels": {f: list(levels[f]) for f in factors},
            "shape": tuple(Ycells.shape),
            "clip_ids": [f"c{i:02d}" for i in range(n_clips)],
            "n_clips": n_clips, "n_cells": int(Ycells.size),
            "n_rows": int(n_clips * Ycells.size), "model": "synthetic",
            "fixed": {}, "response": "wer"}


# ===========================================================================
print("\n[1] subset enumeration + partition identity")
# ===========================================================================

subs = _subsets(4)
check("2^4-1 = 15 non-empty subsets", len(subs) == 15, str(len(subs)))
check("subsets are size-ascending (Mobius needs proper subsets first)",
      [len(u) for u in subs] == sorted(len(u) for u in subs))
check("every subset unique", len(set(subs)) == len(subs))

rng = np.random.default_rng(7)
Yrand = rng.normal(size=(4, 4, 3, 3))
Vt, Vtot, grand = anova_variance_terms(Yrand, 4)
err = _check_partition(Vt, Vtot)
check("random surface: sum(V_u) == V_total exactly", err < 1e-12, f"err={err:.2e}")
check("grand mean matches numpy mean", close(grand, Yrand.mean()))
check("V_total matches numpy var", close(Vtot, Yrand.var()))
check("all 15 terms present", len(Vt) == 15)


# ===========================================================================
print("\n[2] PURELY ADDITIVE surface -> all S2 == 0 and ST == S1")
# ===========================================================================

a = np.array([0.0, 1.0, 2.0, 5.0])
b = np.array([0.0, 3.0, -1.0, 2.0])
c = np.array([0.0, 4.0, 1.0])
d = np.array([0.0, -2.0, 6.0])
Yadd = (a[:, None, None, None] + b[None, :, None, None]
        + c[None, None, :, None] + d[None, None, None, :])

Vt, Vtot, _ = anova_variance_terms(Yadd, 4)
_check_partition(Vt, Vtot)
s = sobol_from_terms(Vt, Vtot, 4)
S1, ST, S2 = s["S1"], s["ST"], s["S2"]

check("additive: sum(S1) == 1", close(S1.sum(), 1.0, 1e-12), f"{S1.sum():.12f}")
check("additive: every S2 (upper triangle) == 0",
      close(S2[~np.isnan(S2)], 0.0, 1e-12), f"max={np.nanmax(np.abs(S2)):.2e}")
check("additive: ST == S1 for every factor", close(ST, S1, 1e-12),
      f"max gap={np.max(np.abs(ST - S1)):.2e}")
check("additive: all higher-order terms == 0",
      close([Vt[u] for u in Vt if len(u) > 1], 0.0, 1e-12))

# analytic check: S1_i == var(effect_i) / sum(var(effect_j))
var_expect = np.array([a.var(), b.var(), c.var(), d.var()])
check("additive: S1 equals each factor's variance share",
      close(S1, var_expect / var_expect.sum(), 1e-12))


# ===========================================================================
print("\n[3] PLANTED 2-WAY INTERACTION -> gap appears for exactly that pair")
# ===========================================================================

# y = a(i) + b(j) + c(k) + d(l) + LAMBDA * g(i) * h(j)   -- interaction on (0, 1)
g = np.array([-1.5, -0.5, 0.5, 1.5])
h = np.array([-1.5, -0.5, 0.5, 1.5])
LAM = 2.0
Yint = Yadd + LAM * g[:, None, None, None] * h[None, :, None, None]

Vt, Vtot, _ = anova_variance_terms(Yint, 4)
_check_partition(Vt, Vtot)
s = sobol_from_terms(Vt, Vtot, 4)
S1i, STi, S2i = s["S1"], s["ST"], s["S2"]

gap = STi - S1i
check("interaction: partition still sums to 1", close(S1i.sum() + np.nansum(S2i), 1.0, 1e-12),
      f"{S1i.sum() + np.nansum(S2i):.12f}")
check("interaction: gap > 0 for factor 0", gap[0] > 1e-6, f"{gap[0]:.6f}")
check("interaction: gap > 0 for factor 1", gap[1] > 1e-6, f"{gap[1]:.6f}")
check("interaction: gap == 0 for factor 2", abs(gap[2]) < 1e-12, f"{gap[2]:.2e}")
check("interaction: gap == 0 for factor 3", abs(gap[3]) < 1e-12, f"{gap[3]:.2e}")
check("interaction: S2[0,1] is the ONLY non-zero pair",
      S2i[0, 1] > 1e-6 and close([S2i[i, j] for i in range(4) for j in range(i + 1, 4)
                                  if (i, j) != (0, 1)], 0.0, 1e-12),
      f"S2[0,1]={S2i[0,1]:.6f}")
check("interaction: gap[0] == gap[1] == S2[0,1] (a pure 2-way term)",
      close(gap[0], S2i[0, 1], 1e-12) and close(gap[1], S2i[0, 1], 1e-12))

# analytic magnitude: V_01 = LAM^2 * var(g) * var(h)
V01_expect = (LAM ** 2) * g.var() * h.var()
check("interaction: V_{0,1} matches the closed form",
      close(Vt[(0, 1)], V01_expect, 1e-10),
      f"got {float(Vt[(0,1)]):.6f} want {V01_expect:.6f}")


# ===========================================================================
print("\n[4] PLANTED 3-WAY interaction -> gap with ZERO 2-way term")
# ===========================================================================
# The nastiest confuser: ST > S1 but every S2 is zero. A decomposition that
# lumps all higher-order variance into the 2-way terms passes tests 2 and 3 and
# fails here.

# e3 MUST be mean-zero. With a non-centred multiplier, g*h*e3 factorises as
# mean(e3)*g*h + g*h*centred(e3): the first piece is a genuine TWO-way (0,1)
# term, so the surface is not purely third-order and asserting S2 == 0 would be
# testing a false premise rather than the code.
e3 = np.array([-1.0, 0.0, 1.0])
assert abs(e3.mean()) < 1e-15 and abs(g.mean()) < 1e-15 and abs(h.mean()) < 1e-15
Y3 = Yadd + 3.0 * (g[:, None, None, None] * h[None, :, None, None]
                   * e3[None, None, :, None])
Vt3, Vtot3, _ = anova_variance_terms(Y3, 4)
_check_partition(Vt3, Vtot3)
s3 = sobol_from_terms(Vt3, Vtot3, 4)
gap3 = s3["ST"] - s3["S1"]
check("3-way: gap > 0 for factors 0,1,2",
      all(gap3[i] > 1e-6 for i in (0, 1, 2)), str(gap3[:3]))
check("3-way: gap == 0 for factor 3", abs(gap3[3]) < 1e-12, f"{gap3[3]:.2e}")
check("3-way: EVERY S2 == 0 (variance is purely third-order)",
      close(s3["S2"][~np.isnan(s3["S2"])], 0.0, 1e-12),
      f"max={np.nanmax(np.abs(s3['S2'])):.2e}")
check("3-way: order-3 term carries it", float(Vt3[(0, 1, 2)]) > 1e-6)


# ===========================================================================
print("\n[5] partition guard actually fires")
# ===========================================================================

bad = {k: v * 1.0 for k, v in Vt.items()}
bad[(0,)] = bad[(0,)] * 1.5                       # corrupt one term
raised = False
try:
    _check_partition(bad, Vtot)
except AssertionError:
    raised = True
check("_check_partition raises on a corrupted decomposition", raised)


# ===========================================================================
print("\n[6] decompose(): bootstrap, CI behaviour, output shape")
# ===========================================================================

levels = {"f0": [0.2, 0.45, 0.7, 1.0], "f1": [0.0, 5.0, 10.0, 20.0],
          "f2": ["none", "g726", "opus-lowrate"], "f3": [0.0, 0.5, 1.0]}
names = ["f0", "f1", "f2", "f3"]

blk0 = make_block(Yint, names, levels, n_clips=40, clip_sd=0.0)
res0 = decompose(blk0, bootstrap=200, seed=1)

check("decompose: variance_explained_check == 1", abs(res0["variance_explained_check"] - 1.0) < 1e-12,
      f"{res0['variance_explained_check']:.12f}")
check("decompose: names preserved", res0["names"] == names)
check("decompose: S1 shape", np.asarray(res0["S1"]).shape == (4,))
check("decompose: S2 shape", np.asarray(res0["S2"]).shape == (4, 4))
check("decompose: S2 lower triangle is NaN",
      bool(np.all(np.isnan(np.asarray(res0["S2"])[np.tril_indices(4)]))))
check("decompose: recovers the planted pair as rank-1 S2",
      tuple(res0["s2_ranked"][0]["pair"]) == ("f0", "f1"),
      str(res0["s2_ranked"][0]["pair"]))
check("decompose: interaction_gap sorted descending",
      [d["gap"] for d in res0["interaction_gap"]] ==
      sorted((d["gap"] for d in res0["interaction_gap"]), reverse=True))

# clip_sd=0 => every bootstrap replicate is the identical experiment => CI == 0
check("decompose: identical clips give zero-width CIs",
      close(np.asarray(res0["S1_conf"]), 0.0, 1e-9) and
      close(np.asarray(res0["ST_conf"]), 0.0, 1e-9),
      f"max S1_conf={np.max(np.asarray(res0['S1_conf'])):.2e}")

# a clip offset shifts the grand mean only: every ANOVA effect is unchanged, so the
# indices must be identical and the CI must STILL be ~0 (the offset cancels).
blk1 = make_block(Yint, names, levels, n_clips=40, clip_sd=0.5, seed=3)
res1 = decompose(blk1, bootstrap=200, seed=1)
check("decompose: a per-clip additive offset does not move the indices",
      close(np.asarray(res0["S1"]), np.asarray(res1["S1"]), 1e-12))

# a clip x factor interaction DOES create real sampling variability => CI > 0
rng = np.random.default_rng(11)
Wn = np.broadcast_to(Yint, (40,) + Yint.shape) + rng.normal(0, 1.0, (40,) + Yint.shape)
blk2 = dict(blk0, W=Wn)
res2 = decompose(blk2, bootstrap=300, seed=2)
check("decompose: clip-level noise produces non-zero CIs",
      float(np.max(np.asarray(res2["S1_conf"]))) > 1e-4,
      f"max={float(np.max(np.asarray(res2['S1_conf']))):.4f}")
check("decompose: noisy block still partitions exactly",
      abs(res2["variance_explained_check"] - 1.0) < 1e-12)
check("decompose: variance_share_by_order sums to 1",
      abs(sum(float(v) for v in res2["variance_share_by_order"].values()) - 1.0) < 1e-12)


# ===========================================================================
print("\n[7] additive block end-to-end: ST == S1, S2 == 0 through decompose()")
# ===========================================================================

blk_add = make_block(Yadd, names, levels, n_clips=40)
res_add = decompose(blk_add, bootstrap=100, seed=5)
check("additive e2e: every ST == S1",
      close(np.asarray(res_add["ST"]), np.asarray(res_add["S1"]), 1e-12))
check("additive e2e: every gap == 0",
      all(abs(d["gap"]) < 1e-12 for d in res_add["interaction_gap"]))
check("additive e2e: every ranked S2 == 0",
      all(abs(d["S2"]) < 1e-12 for d in res_add["s2_ranked"]))
check("additive e2e: order-1 share == 1",
      abs(float(res_add["variance_share_by_order"][1]) - 1.0) < 1e-12)


# ===========================================================================
print("\n[8] screen_from_factorial: recovers planted marginal contrasts")
# ===========================================================================

scr = screen_from_factorial(blk_add, bootstrap=100, seed=0)
want = {"f0": a[-1] - a[0], "f1": b[-1] - b[0], "f2": c[-1] - c[0], "f3": d[-1] - d[0]}
check("screen: exact marginal effects match the planted contrasts",
      all(abs(scr["effects"][f] - want[f]) < 1e-12 for f in names),
      json.dumps(scr["effects"]))
check("screen: ranking is by |effect|",
      scr["ranking"] == sorted(names, key=lambda f: abs(want[f]), reverse=True))
check("screen: spends zero API calls", scr["n_api_calls_spent"] == 0)
check("screen: PB cross-check ran 8 runs", scr["pb_cross_check"]["n_runs"] == 8)
# on a purely additive surface PB has nothing to alias, so it must be EXACT
check("screen: on an additive surface PB == exact (nothing to alias)",
      all(abs(r["pb_minus_exact"]) < 1e-12 for r in scr["rows"]),
      str([r["pb_minus_exact"] for r in scr["rows"]]))

check("screen: caveat is design.SCREEN_CAVEAT verbatim",
      scr["caveat"] == __import__("deadzone.design", fromlist=["SCREEN_CAVEAT"]).SCREEN_CAVEAT)


# --- the alias structure of THIS design, and what it does to the PB estimate ---
# With 4 factors in an 8-run PB, only SOME pair-products land on a used main-effect
# column. Which ones is a property of the design matrix, so it is computed, and the
# two tests below pin down both branches of the consequence.

from deadzone.analysis.sensitivity import pb_alias_structure      # noqa: E402
from deadzone.design import plackett_burman                       # noqa: E402

alias = pb_alias_structure(plackett_burman(4), names)
unaliased = {tuple(p) for p in alias["unaliased_pairs"]}
conf_onto = {tuple(d["pair"]): d["aliased_onto"] for d in alias["confounded_pairs"]}

check("alias: f0 x f1 lands on an UNUSED column (contaminates no main effect)",
      ("f0", "f1") in unaliased, str(alias["unaliased_pairs"]))
check("alias: f0 x f2 is confounded with f3's main effect",
      conf_onto.get(("f0", "f2")) == "f3", str(alias["confounded_pairs"]))
check("alias: exactly 3 of the 6 pairs are confounded",
      len(conf_onto) == 3 and len(unaliased) == 3,
      f"{len(conf_onto)} confounded, {len(unaliased)} clean")

# (a) interaction on the UNALIASED pair f0 x f1 -> PB is exact for every factor.
# This is the case that matters for SPEC §5: the pre-registered pair is (rt60,
# snr_db) = columns 0 x 1, so even a fresh PB screen would have estimated both
# registered factors' main effects without contamination.
scr_01 = screen_from_factorial(blk0, bootstrap=50, seed=0)
check("screen: interaction on the UNALIASED pair leaves PB exact",
      all(abs(r["pb_minus_exact"]) < 1e-12 for r in scr_01["rows"]),
      str([round(r["pb_minus_exact"], 9) for r in scr_01["rows"]]))

# (b) interaction on the CONFOUNDED pair f0 x f2 -> PB's estimate of f3 is biased.
# q's endpoints must differ (q[0] != q[-1]); if they were symmetric the aliased
# contribution would cancel in the contrast and the bias would be invisible.
q = np.array([0.0, 4.0, 2.0])
Yali = Yadd + 2.0 * g[:, None, None, None] * q[None, None, :, None]
blk_ali = make_block(Yali, names, levels, n_clips=40)
scr_ali = screen_from_factorial(blk_ali, bootstrap=50, seed=0)
dev = {r["factor"]: r["pb_minus_exact"] for r in scr_ali["rows"]}
check("screen: interaction on the CONFOUNDED pair biases PB's f3 estimate",
      abs(dev["f3"]) > 1e-6, f"pb-exact for f3 = {dev['f3']:.6f}")
check("screen: f1 (orthogonal to that interaction) stays exact",
      abs(dev["f1"]) < 1e-12, f"{dev['f1']:.2e}")
check("screen: the exact marginal effect is unaffected by aliasing (it has none)",
      abs(scr_ali["effects"]["f1"] - (b[-1] - b[0])) < 1e-12)


# ===========================================================================
print("\n[9] load_factorial: refuses ragged / incomplete blocks")
# ===========================================================================

def mkrow(clip, rt, snr, codec, roll, wer, noise="babble", model="nova-3"):
    return {"clip_id": clip, "condition_name": f"{rt}_{snr}_{codec}_{roll}",
            "rt60": str(rt), "snr_db": str(snr), "noise_type": noise,
            "codec": codec, "mic_rolloff": str(roll), "model": model,
            "wer": str(wer), "failed": "False"}


rows_ok = []
for ci in range(3):
    for rt in (0.2, 1.0):
        for snr in (0.0, 20.0):
            for cd in ("none", "g726"):
                for ro in (0.0, 1.0):
                    rows_ok.append(mkrow(f"c{ci}", rt, snr, cd, ro, 0.1 * ci + rt))
blk = load_factorial(rows_ok, PRIMARY_FACTORS, model="nova-3", fixed=PRIMARY_FIXED)
check("load_factorial: complete block loads", blk["n_cells"] == 16 and blk["n_clips"] == 3,
      f"{blk['n_cells']} cells, {blk['n_clips']} clips")
check("load_factorial: categorical level order follows DEFAULT_FACTOR_SPACE",
      blk["levels"]["codec"] == ["none", "g726"], str(blk["levels"]["codec"]))
check("load_factorial: W has no NaN", not bool(np.isnan(blk["W"]).any()))

raised = False
try:
    load_factorial(rows_ok[:-1], PRIMARY_FACTORS, model="nova-3", fixed=PRIMARY_FIXED)
except ValueError as e:
    raised = "INCOMPLETE" in str(e)
check("load_factorial: raises on a hole in the block", raised)

# a failed=True row must be treated as MISSING (not averaged in as a sentinel)
rows_failed = [dict(r) for r in rows_ok]
rows_failed[0] = dict(rows_failed[0], failed="True", wer="1.0")
raised = False
try:
    load_factorial(rows_failed, PRIMARY_FACTORS, model="nova-3", fixed=PRIMARY_FIXED)
except ValueError:
    raised = True
check("load_factorial: a failed row leaves a hole rather than a sentinel", raised)

# the string "False" is truthy — the classic CSV trap
check("load_factorial: 'False' string is correctly falsy",
      load_factorial(rows_ok, PRIMARY_FACTORS, model="nova-3",
                     fixed=PRIMARY_FIXED)["n_rows"] == len(rows_ok))

# wrong model is excluded
rows_mixed = rows_ok + [mkrow("c0", 0.2, 0.0, "none", 0.0, 0.9, model="whisper-base")]
check("load_factorial: filters by model",
      load_factorial(rows_mixed, PRIMARY_FACTORS, model="nova-3",
                     fixed=PRIMARY_FIXED)["n_rows"] == len(rows_ok))

# a DUPLICATE (clip, cell) row is a different hole than an incomplete block, and it
# needs its own test: the fill loop assigns W[clip, cell] = y, so a second row for a
# cell already written silently OVERWRITES the first and leaves no NaN behind. The
# "INCOMPLETE" check above never fires -- the block still looks complete, sums to
# W.size, and the exact ANOVA would happily decompose whichever row happened to land
# last. That is this project's signature failure mode: a clean-looking number with
# no error message. The realistic cause is mundane (a partial re-run appending a
# fresh transcript for a cell that was already measured), which is exactly why
# `n_used != W.size` is asserted rather than trusted.
rows_dup = rows_ok + [mkrow("c0", 0.2, 0.0, "none", 0.0, 9.9)]     # same cell as an
                                                                    # existing row,
                                                                    # different wer
raised = False
try:
    load_factorial(rows_dup, PRIMARY_FACTORS, model="nova-3", fixed=PRIMARY_FIXED)
except ValueError as e:
    raised = "DUPLICATE" in str(e)
check("load_factorial: raises on a duplicate (clip, cell) row", raised)

# negative case, same table minus the duplicate: must load cleanly, so the guard is
# pinned to the duplicate specifically and not to some incidental property of rows_ok.
check("load_factorial: the same table WITHOUT the duplicate still loads fine",
      load_factorial(rows_ok, PRIMARY_FACTORS, model="nova-3",
                     fixed=PRIMARY_FIXED)["n_rows"] == len(rows_ok))


# ===========================================================================
print("\n[9b] block integrity: the decomposition refuses input it cannot see through")
# ===========================================================================
#
# `load_factorial` establishes all of this for a block it builds, but decompose /
# screen_from_factorial / measured_counterintuitive are public and take
# hand-assembled blocks (this file builds them with make_block; so would any
# caller carving a sub-block). Each failure below is SILENT if unchecked.

# (a) THE ONE THAT MATTERS MOST: a NaN in W. It propagates into every ANOVA term,
# and `_check_partition`'s `nan > tol` is FALSE -- so without an explicit
# finiteness check the partition assertion, the single guard this whole
# decomposition leans on, would PASS and hand back an all-nan S1/ST table that
# prints as `nan` rather than as an error.
Vt_nan, Vtot_nan, _ = anova_variance_terms(
    np.where(np.arange(Yint.size).reshape(Yint.shape) == 0, np.nan, Yint), 4)
sum_nan = sum(Vt_nan.values())
check("partition check: a NaN response really does make sum(V_u) non-finite",
      not np.isfinite(np.asarray(sum_nan)).all())
check("partition check: `nan > tol` is False, so a bare threshold would PASS",
      not (float(np.max(np.abs(np.asarray(sum_nan) / float(Vtot_nan) - 1.0))) > 1e-9))
raised = ""
try:
    _check_partition(Vt_nan, Vtot_nan)
except AssertionError as e:
    raised = str(e)
check("partition check: raises NON-FINITE instead of passing silently",
      "NON-FINITE" in raised, raised[:80])

W_nan = np.array(blk0["W"], dtype=float)
W_nan[3, 1, 2, 0, 1] = np.nan
raised = ""
try:
    decompose(dict(blk0, W=W_nan), bootstrap=20, seed=0)
except ValueError as e:
    raised = str(e)
check("decompose: raises on a non-finite response",
      "NOT\nFINITE" in raised or "NOT FINITE" in raised, raised[:90])
check("decompose: the message names how many and where",
      "1 of" in raised and "[3, 1, 2, 0, 1]" in raised, raised[:120])
for name, fn in (("screen_from_factorial",
                  lambda b: screen_from_factorial(b, bootstrap=20)),
                 ("measured_counterintuitive",
                  lambda b: measured_counterintuitive(b, bootstrap=20))):
    ok = False
    try:
        fn(dict(blk0, W=W_nan))
    except ValueError as e:
        ok = "FINITE" in str(e)
    check(f"{name}: refuses a non-finite response too", ok)

# (b) the row-count identity. `n_rows` is what every report prints as "real
# transcriptions decomposed"; if it disagrees with W.size the provenance line
# describes a different dataset than the numbers came from.
raised = ""
try:
    decompose(dict(blk0, n_rows=blk0["n_rows"] + 40), bootstrap=20)
except ValueError as e:
    raised = str(e)
check("decompose: raises when declared n_rows != W.size",
      "row-count identity" in raised, raised[:90])

# (c) a shape that does not match what the block declares would decompose over
# the wrong axes and still return in-range indices.
raised = ""
try:
    decompose(dict(blk0, shape=(4, 4, 3, 3, 1)), bootstrap=20)
except ValueError as e:
    raised = str(e)
check("decompose: raises on a shape/n_clips identity mismatch",
      "shape identity" in raised, raised[:90])

# (d) fewer than two clips makes every bootstrap std(ddof=1) nan, i.e. every CI
# silently prints as nan next to a perfectly good point estimate.
blk_one = make_block(Yint, names, levels, n_clips=1)
raised = ""
try:
    decompose(blk_one, bootstrap=20)
except ValueError as e:
    raised = str(e)
check("decompose: refuses a single-clip block (every CI would be nan)",
      "bootstrap" in raised and "1 clip" in raised, raised[:90])

# (e) same for the replicate count itself.
raised = ""
try:
    decompose(blk0, bootstrap=1)
except ValueError as e:
    raised = str(e)
check("decompose: refuses bootstrap<2 (no CI is computable)",
      "bootstrap=1" in raised, raised[:90])

# negative control: the untouched block still decomposes, so every guard above is
# pinned to its violation and not to some incidental property of blk0.
check("block guards do not fire on the clean block",
      abs(decompose(blk0, bootstrap=20)["variance_explained_check"] - 1.0) < 1e-12)


# ===========================================================================
print("\n[10] JSON round-trip through analysis.interactions")
# ===========================================================================

payload = to_json(res0)
for key in ("names", "S1", "ST", "S1_conf", "ST_conf", "S2", "S2_conf", "N", "n_eval"):
    check(f"json: has `{key}`", key in payload)
for key in ("method", "source", "n_cells", "n_clips", "bootstrap_reps",
            "variance_explained_check"):
    check(f"json: provenance key `{key}`", key in payload)
check("json: method is exact-factorial-anova", payload["method"] == "exact-factorial-anova")

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "sobol.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    back = load_sobol_json(p)

check("round-trip: normalize_sobol_result accepts it", back["names"] == names)
check("round-trip: S1 survives as an array",
      close(np.asarray(back["S1"]), np.asarray(res0["S1"]), 1e-12))
check("round-trip: S2 lower triangle still NaN",
      bool(np.all(np.isnan(np.asarray(back["S2"])[np.tril_indices(4)]))))
check("round-trip: s2_ranked pairs are tuples",
      all(isinstance(d["pair"], tuple) for d in back["s2_ranked"]))
check("round-trip: interaction_gap preserved",
      close([d["gap"] for d in back["interaction_gap"]],
            [d["gap"] for d in res0["interaction_gap"]], 1e-12))
check("round-trip: NaN off-triangle is skipped by s2_ranked (6 pairs, not 16)",
      len(back["s2_ranked"]) == 6, str(len(back["s2_ranked"])))

# in-memory (no disk) normalization must work too
check("normalize_sobol_result works on the live dict",
      normalize_sobol_result(res0)["names"] == names)


# ===========================================================================
print("\n[10b] every written payload states what file it is and what it is FOR")
# ===========================================================================
#
# `results/sobol_5factor.json` was flagged by an artifact audit as an ORPHAN:
# regenerated on every run, internally sound, referenced by no analysis module,
# no dashboard panel, no test and no write-up claim. It is not dead output -- it
# is the only block in which `noise_type` varies -- but a numerically-plausible
# JSON with no stated scope is exactly what gets quoted beside the headline
# indices by someone who does not know the two files decompose DIFFERENT grids.

from deadzone.analysis.sensitivity import (                       # noqa: E402
    PRIMARY_BLOCK_DOC, SECONDARY_BLOCK_DOC, tag_artifact,
)

tagged_p = tag_artifact(res0, PRIMARY_BLOCK_DOC, "results/sobol.json")
tagged_s = tag_artifact(res0, SECONDARY_BLOCK_DOC, "results/sobol_5factor.json")

check("tag_artifact: names the file it is destined for",
      tagged_p["artifact"] == "results/sobol.json" and
      tagged_s["artifact"] == "results/sobol_5factor.json")
check("tag_artifact: the provenance header comes FIRST in the payload",
      list(tagged_s)[0] == "artifact" and "block_role" in list(tagged_s)[:3],
      str(list(tagged_s)[:4]))
check("tag_artifact: preserves every numeric field untouched",
      all(k in tagged_p and tagged_p[k] is res0[k] for k in res0),
      str(sorted(set(res0) - set(tagged_p))))
check("primary block is marked quotable, secondary is NOT",
      tagged_p["quotable"] is True and tagged_s["quotable"] is False)
check("primary block names its real consumers",
      any("interactions" in c for c in tagged_p["consumed_by"]) and
      any("writeup" in c for c in tagged_p["consumed_by"]))
check("secondary block declares itself unconsumed, and says why it is kept",
      tagged_s["consumed_by"] == [] and "REFERENCE ONLY" in tagged_s["status"] and
      "noise_type" in tagged_s["purpose"])
check("secondary block warns against tabling its indices beside the primary's",
      "do_not" in tagged_s and "sobol.json" in tagged_s["do_not"])
# NEGATIVE CONTROL: the tag is a header, not a rewrite. A payload that came back
# with a moved index would mean the provenance step had touched the numbers.
check("tag_artifact does not mutate the source payload",
      "artifact" not in res0 and "block_role" not in res0)
check("tag_artifact: S1/ST/partition survive the tagging bit-for-bit",
      close(np.asarray(tagged_p["S1"]), np.asarray(res0["S1"]), 0.0) and
      close(np.asarray(tagged_p["ST"]), np.asarray(res0["ST"]), 0.0) and
      tagged_p["variance_explained_check"] == res0["variance_explained_check"])
# and a tagged payload must still round-trip through the consumer that reads it
check("a tagged payload still normalizes for analysis.interactions",
      normalize_sobol_result(tagged_p)["names"] == names)


# ===========================================================================
print("\n[11] the *_conf convention is a HALF-WIDTH (interactions.py depends on it)")
# ===========================================================================
# analysis.interactions builds gap_lo = gap - conf. If *_conf were a full width or
# a percentile bound, every CI in the verdict would be silently wrong by 2x.

S1_conf = np.asarray(res2["S1_conf"], dtype=float)
S1_lo = np.asarray(res2["S1_ci_lo"], dtype=float)
S1_hi = np.asarray(res2["S1_ci_hi"], dtype=float)
half_from_pct = (S1_hi - S1_lo) / 2.0
check("half-width is ~ (percentile hi - lo)/2 (normal-approx vs percentile)",
      bool(np.all(np.abs(S1_conf - half_from_pct) < 0.35 * np.maximum(half_from_pct, 1e-6) + 1e-6)),
      f"conf={np.round(S1_conf,4)} pct-half={np.round(half_from_pct,4)}")
check("all *_conf are non-negative",
      bool(np.all(S1_conf >= 0) and np.all(np.asarray(res2['ST_conf']) >= 0)))


# ===========================================================================
print("\n[11b] the ST-S1 gap publishes BOTH intervals, under distinguishable names")
# ===========================================================================
#
# THE DEFECT THIS SECTION PINS (artifact audit, 2026-08-05). report/writeup.md
# published the QUADRATURE gap interval; results/sensitivity_report.txt printed
# the DIRECT bootstrap interval under the header `gap 95% CI`; results/sobol.json
# stored only the direct one. Two different intervals, one label, sitting
# directly beneath the pre-registration verdict. No verdict moved (both forms
# clear 0.020 by >=4.5x) but a reader diffing the artifacts had no way to tell
# them apart, and the quadrature form -- the one the registered test is actually
# read off -- was stored nowhere at all.

from deadzone.analysis.sensitivity import (                       # noqa: E402
    GAP_CI_NOTE, PREREGISTRATION_CI_FORM, _pearson,
    format_sensitivity_report,
)
from deadzone.analysis.interactions import _gap_ci, interaction_gap_table  # noqa: E402

gaps2 = res2["interaction_gap"]

# (a) THE VIOLATION: no gap interval may be published under an ambiguous name.
# Every gap CI field must say WHICH of the two it is.
amb = sorted({k for d in gaps2 for k in d
              if k.startswith("gap_c") and not
              (k.endswith("_direct") or k.endswith("_quadrature")
               or k.endswith("_quadrature_over_direct"))})
check("gap CI fields all carry a _direct / _quadrature suffix", not amb, str(amb))
for key in ("gap_conf_direct", "gap_ci_lo_direct", "gap_ci_hi_direct",
            "gap_conf_quadrature", "gap_ci_lo_quadrature", "gap_ci_hi_quadrature",
            "s1_st_bootstrap_corr"):
    check(f"interaction_gap publishes `{key}`", all(key in d for d in gaps2))
check("payload names which form the pre-registered verdict uses",
      res2["preregistration_ci_form"] == PREREGISTRATION_CI_FORM and
      all(d["preregistration_ci_form"] == PREREGISTRATION_CI_FORM for d in gaps2))
check("payload carries the note explaining why both are kept",
      res2["gap_ci_note"] == GAP_CI_NOTE and "not interchangeable" in GAP_CI_NOTE.lower())

# (b) CROSS-MODULE AGREEMENT — the actual bug was that the write-up's number came
# from analysis.interactions and the artifact's from here. Pin them to be the
# same float, so the two producers can never drift apart again.
itx = {r["factor"]: r for r in interaction_gap_table(res2)}
check("quadrature half-width == interactions._gap_ci(S1_conf, ST_conf) EXACTLY",
      all(d["gap_conf_quadrature"] == _gap_ci(d["S1_conf"], d["ST_conf"])
          for d in gaps2))
check("quadrature bounds == interactions' gap_lo / gap_hi EXACTLY",
      all(d["gap_ci_lo_quadrature"] == itx[d["factor"]]["gap_lo"] and
          d["gap_ci_hi_quadrature"] == itx[d["factor"]]["gap_hi"] for d in gaps2))

# (c) WHY THEY DIFFER, as an exact identity rather than a claim. From
# Var(ST-S1) = Var(ST) + Var(S1) - 2*Cov(S1,ST) and conf = Z*sd:
#     gap_conf_direct^2 == S1_conf^2 + ST_conf^2 - 2*corr*S1_conf*ST_conf
# The quadrature form is that expression with the covariance term DELETED, which
# is why it is wider whenever corr > 0 -- and this identity is what makes the
# over-statement auditable from the published fields alone.
for gd in gaps2:                     # NB: `d` is a planted effect array above
    lhs = gd["gap_conf_direct"] ** 2
    rhs = (gd["S1_conf"] ** 2 + gd["ST_conf"] ** 2
           - 2.0 * gd["s1_st_bootstrap_corr"] * gd["S1_conf"] * gd["ST_conf"])
    check(f"variance identity holds for {gd['factor']}",
          abs(lhs - rhs) <= 1e-9 * max(lhs, 1e-12), f"{lhs:.6e} vs {rhs:.6e}")
check("S1 and ST are POSITIVELY correlated across replicates (the stated cause)",
      all(d["s1_st_bootstrap_corr"] > 0 for d in gaps2),
      str([round(d["s1_st_bootstrap_corr"], 3) for d in gaps2]))
check("so quadrature is WIDER than direct on every factor",
      all(d["gap_conf_quadrature"] > d["gap_conf_direct"] for d in gaps2),
      str([round(d["gap_conf_ratio_quadrature_over_direct"], 3) for d in gaps2]))
check("the width ratio is reported and matches the two half-widths",
      all(abs(d["gap_conf_ratio_quadrature_over_direct"]
              - d["gap_conf_quadrature"] / d["gap_conf_direct"]) < 1e-12
          for d in gaps2))

# (d) NEGATIVE CONTROL, and it is the sharpest evidence the two are genuinely
# different quantities rather than one rescaled. Build a block that is ADDITIVE
# in every clip -- per-clip independent scaling of each additive component -- so
# every bootstrap replicate is still additive and therefore has ST == S1 exactly.
# The gap is provably, exactly zero with zero uncertainty, yet quadrature returns
# a WIDE non-zero interval around it. corr must come out at exactly +1.
rngA = np.random.default_rng(21)
comp = [a[:, None, None, None], b[None, :, None, None],
        c[None, None, :, None], d[None, None, None, :]]
W_add = np.zeros((40,) + Yadd.shape, dtype=float)
for ci in range(40):
    for comp_arr in comp:
        W_add[ci] = W_add[ci] + rngA.uniform(0.5, 1.5) * comp_arr
blk_pure_add = dict(make_block(Yadd, names, levels, n_clips=40), W=W_add)
res_pa = decompose(blk_pure_add, bootstrap=300, seed=4)
gpa = res_pa["interaction_gap"]
check("neg control: a per-clip-additive block has ST == S1 exactly",
      close(np.asarray(res_pa["ST"]), np.asarray(res_pa["S1"]), 1e-12))
check("neg control: corr(S1, ST) is exactly +1 there",
      all(abs(g["s1_st_bootstrap_corr"] - 1.0) < 1e-12 for g in gpa),
      str([g["s1_st_bootstrap_corr"] for g in gpa]))
check("neg control: the DIRECT interval is exactly zero-width (no interaction)",
      all(g["gap_conf_direct"] < 1e-12 for g in gpa),
      str([g["gap_conf_direct"] for g in gpa]))
check("neg control: the QUADRATURE interval is NOT zero-width there",
      any(g["gap_conf_quadrature"] > 1e-6 for g in gpa),
      str([round(g["gap_conf_quadrature"], 6) for g in gpa]))
check("neg control: the identity still holds at corr == 1",
      all(abs(g["gap_conf_direct"] ** 2
              - (g["S1_conf"] ** 2 + g["ST_conf"] ** 2
                 - 2.0 * g["s1_st_bootstrap_corr"] * g["S1_conf"] * g["ST_conf"]))
          < 1e-12 for g in gpa))
# ...and the bootstrap really did have something to vary, so "zero-width direct"
# is a fact about the gap and not about a degenerate resample.
check("neg control: S1 itself DID vary across replicates (not a dead bootstrap)",
      float(np.max(np.asarray(res_pa["S1_conf"]))) > 1e-6,
      f"max S1_conf={float(np.max(np.asarray(res_pa['S1_conf']))):.2e}")

# (e) _pearson returns NaN, never 0.0, for a constant replicate vector: 0.0 would
# read as "S1 and ST are independent", the one claim this number exists to refute.
check("_pearson: a constant vector gives NaN, not 0.0",
      math.isnan(_pearson(np.ones(50), np.arange(50.0))))
check("_pearson: matches numpy on a normal case",
      abs(_pearson(np.arange(50.0), np.arange(50.0) ** 1.3)
          - float(np.corrcoef(np.arange(50.0), np.arange(50.0) ** 1.3)[0, 1])) < 1e-12)
# res0 has IDENTICAL clips, so the bootstrap is degenerate: every replicate is
# the same experiment and both half-widths collapse to ~0. The requirement is not
# a particular corr -- it is that a degenerate bootstrap never fabricates a WIDE
# interval, and never crashes on the 0/0 ratio.
check("degenerate bootstrap: both gap intervals collapse to ~0 width",
      all(gd["gap_conf_direct"] < 1e-12 and gd["gap_conf_quadrature"] < 1e-12
          for gd in res0["interaction_gap"]),
      str([(gd["gap_conf_direct"], gd["gap_conf_quadrature"])
           for gd in res0["interaction_gap"]]))
check("degenerate bootstrap: the width ratio is inf/NaN, never a plausible number",
      all(not math.isfinite(gd["gap_conf_ratio_quadrature_over_direct"])
          or gd["gap_conf_quadrature"] < 1e-12
          for gd in res0["interaction_gap"]))

# (f) the printed report must show BOTH, labelled. A number printed under a bare
# `gap 95% CI` header is exactly what shipped before.
rep_txt = format_sensitivity_report(res2)
check("report labels the first interval as (direct)", "gap 95% CI (direct)" in rep_txt)
check("report prints the side-by-side block with both labels",
      "direct (correct)" in rep_txt and "quadrature (CONSERVATIVE)" in rep_txt)
check("report names which interval the pre-registration uses",
      "PRE-REGISTERED VERDICT IS READ OFF THE QUADRATURE INTERVAL" in rep_txt)
check("report no longer prints a bare, unqualified `gap 95% CI` header",
      "gap 95% CI\n" not in rep_txt and not any(
          line.rstrip().endswith("gap 95% CI") for line in rep_txt.splitlines()))


# ===========================================================================
print("\n[12] measured_counterintuitive: dips found IN the grid, not in a surrogate")
# ===========================================================================

from deadzone.analysis.sensitivity import measured_counterintuitive        # noqa: E402

# (a) a strictly monotone surface must yield NO dips at all.
mono = (np.arange(4)[:, None, None, None] * 1.0
        + np.arange(4)[None, :, None, None] * 0.5
        + np.zeros((1, 1, 3, 1)) + np.arange(3)[None, None, None, :] * 0.25)
blk_mono = make_block(mono, names, levels, n_clips=40, clip_sd=0.05, seed=1)
mc_mono = measured_counterintuitive(blk_mono, bootstrap=200, seed=0)
check("measured: monotone surface yields no marginal dips",
      mc_mono["n_marginal"] == 0, str(mc_mono["n_marginal"]))
check("measured: monotone surface yields no significant cellwise dips",
      mc_mono["n_cellwise_significant"] == 0, str(mc_mono["n_cellwise_significant"]))
check("measured: spends zero API calls", mc_mono["api_calls_spent"] == 0)
check("measured: is presentable as measured by construction",
      mc_mono["presentable_as_measured"] is True)

# (b) plant an explicit dip on factor f0 at level index 1 and recover it.
# f0 marginal goes 0.6 -> 0.2 -> 0.7 -> 0.9: an interior minimum of depth 0.4.
dip_curve = np.array([0.6, 0.2, 0.7, 0.9])
Ydip = np.broadcast_to(dip_curve[:, None, None, None], (4, 4, 3, 3)).copy()
blk_dip = make_block(Ydip, names, levels, n_clips=40, clip_sd=0.0)
mc = measured_counterintuitive(blk_dip, bootstrap=200, seed=0)
check("measured: recovers exactly one marginal dip", mc["n_marginal"] == 1,
      str([(d["factor"], d["level"]) for d in mc["marginal"]]))
m0 = mc["marginal"][0]
check("measured: dip is on the planted factor", m0["factor"] == "f0", m0["factor"])
check("measured: dip is at the planted level", m0["level"] == levels["f0"][1],
      str(m0["level"]))
check("measured: dip depth == planted 0.4", abs(m0["depth"] - 0.4) < 1e-12,
      f"{m0['depth']:.6f}")
check("measured: triplet WERs are the planted curve",
      close([m0["wer_before"], m0["wer_at_dip"], m0["wer_after"]],
            [0.6, 0.2, 0.7], 1e-12))
check("measured: identical clips -> zero-width depth CI",
      abs(m0["depth_conf"]) < 1e-9, f"{m0['depth_conf']:.2e}")
check("measured: dip flagged significant", m0["significant"] is True)

# (c) categorical factors are never scanned — level ORDER is a naming convention.
check("measured: codec-like categorical is skipped",
      "f2" in mc["categorical_skipped"] and "f2" not in mc["ordered_factors_scanned"],
      f"scanned={mc['ordered_factors_scanned']} skipped={mc['categorical_skipped']}")
check("measured: no dip is ever reported on a categorical factor",
      all(d["factor"] != "f2" for d in mc["marginal"] + mc["cellwise"]))

# (d) noise must widen the CI enough to suppress a dip that is not real.
rngd = np.random.default_rng(4)
Wnoisy = np.broadcast_to(mono, (40,) + mono.shape) + rngd.normal(0, 2.0, (40,) + mono.shape)
mc_noisy = measured_counterintuitive(dict(blk_mono, W=Wnoisy), bootstrap=300, seed=0)
check("measured: cellwise dips whose CI crosses zero are excluded",
      all(d["depth_ci_lo"] > 0.0 for d in mc_noisy["cellwise"]),
      f"{mc_noisy['n_cellwise_significant']} kept")


# ===========================================================================
print("\n[13] real master table, if present (smoke only — no assertions on values)")
# ===========================================================================

if os.path.exists("results/master.csv"):
    from deadzone.analysis.interactions import load_master_rows
    rows = load_master_rows("results/master.csv")
    blkR = load_factorial(rows, PRIMARY_FACTORS, model="nova-3", fixed=PRIMARY_FIXED)
    check("real grid: 144-cell factorial is complete and clip-balanced",
          blkR["n_cells"] == 144 and blkR["n_clips"] == 40 and blkR["n_rows"] == 5760,
          f"{blkR['n_cells']} cells x {blkR['n_clips']} clips = {blkR['n_rows']} rows")
    resR = decompose(blkR, bootstrap=50, seed=0)
    check("real grid: decomposition partitions exactly",
          abs(resR["variance_explained_check"] - 1.0) < 1e-12,
          f"{resR['variance_explained_check']:.12f}")
    check("real grid: all S1 in [0,1]",
          bool(np.all(np.asarray(resR["S1"]) >= -1e-12) and
               np.all(np.asarray(resR["S1"]) <= 1.0 + 1e-12)))
    check("real grid: ST >= S1 for every factor",
          bool(np.all(np.asarray(resR["ST"]) >= np.asarray(resR["S1"]) - 1e-12)))
else:
    print("  skip results/master.csv not present")


# ===========================================================================
print()
if _FAILS:
    print(f"FAILED {len(_FAILS)} check(s): {_FAILS}")
    raise SystemExit(1)
print("test_sensitivity.py — ALL CHECKS PASSED")
