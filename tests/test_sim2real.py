"""
Offline validation for D4 (analysis/sim2real.py). No API, no audio, no master
table on disk — synthetic REAL-run and SIM-run tables with KNOWN planted
structure stand in for the two grid runs, and we assert D4 recovers it.

The five things that can silently break this layer, one test each:

  1. PAIRING ON THE WRONG COLUMN. `rt60` is what the design ASKED for;
     `rir_rt60_measured` is what the RIR actually DELIVERED (pick_rir snaps to the
     closest asset; pyroomacoustics' realized RT60 diverges from the Sabine target
     by up to ~30%). We feed a table where the two DISAGREE — scrambled, so a
     requested-based join produces a different assignment — and assert every pair
     joins on the measured value. Also: a table with no measured column must RAISE,
     never fall back to the requested one.
  2. THE LEVEL CLAIM. A known constant WER offset between the arms must come back
     as the mean signed gap, with the planted value inside the bootstrap CI.
  3. THE HEADLINE DISTINCTION. Absolute WERs can differ enormously while the
     ORDERING survives: a sim-only testbed is then useless for quoting numbers and
     still fine for ranking conditions. Planted as wer_sim = a + b*wer_real
     (monotone): assert Spearman ~ 1 AND a large mean gap AND the combined verdict
     "ORDER PRESERVED, LEVEL OFFSET".
  4. THE OPPOSITE CASE. Ordering genuinely reversed -> rank correlation must drop
     (and the verdict must flip), or the layer would bless a sim that reorders the
     failure map.
  5. FAILED ROWS. `failed=True` is a MISSING measurement carrying WER 1.0 and NaN
     confidence. Averaging it in manufactures a fake dead zone. Assert it is
     excluded from the per-condition mean, counted in the report, and counted only
     for the model under analysis.
  6. UNMATCHED CLIP SETS — the other half of the module's own premise ("same
     clips, same condition list, only the reverb ingredient swapped"). The real
     arm ran 40 utterances, the sim arm the 10-clip AL subset. Average one
     condition over 40 clips and its partner over a DIFFERENT 10 and the gap is
     (RIR provenance) + (clip difficulty), inseparably. On the real grid that was
     worth 7.8 points of a 19.9-point headline. Planted here as clips the sim arm
     never ran that are deliberately EASIER, so the unmatched arithmetic and the
     matched arithmetic differ by a large, known amount: assert the module returns
     the MATCHED value and REPORTS the restriction.

  7. THE SAME THING AS AN ENFORCED INVARIANT, not a description of today's code.
     Test 6 pins the OUTPUT of one fixture; a future edit could delete the
     restriction and be caught only if that fixture still happens to expose it.
     Test 9 instead asserts the structural property directly on the public path:
     every per-condition aggregate on both arms is computed over the SAME clip
     set and the SAME count, the returned gap equals the clip-matched arithmetic
     recomputed inline (and materially differs from the unmatched arithmetic,
     also recomputed inline), and the census is truthful against an independent
     recount of the raw rows. Test 9b closes the last door: it spies on
     `aggregate_conditions` while the real entry point runs, so a consumer that
     re-introduces an UNRESTRICTED aggregation call fails immediately, even if
     the resulting numbers would have looked plausible.

Plus: the dead-zone set comparison must be the SAME flags D1/L1 use
(`model_compare.dead_zone_flags`), not a private reimplementation.

Deterministic (closed-form tables, fixed bootstrap seed). Run:
    python3 tests/test_sim2real.py
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

from deadzone.analysis.sim2real import (
    MEASURED_RT60, REQUESTED_RT60, ClipSetMismatchError, aggregate_conditions,
    clip_intersection, dead_zone_agreement, gap_summary, pair_conditions,
    rank_agreement, sim2real_report,
)
from deadzone.model_compare import dead_zone_flags
from scripts.run_experiment import MASTER_COLUMNS

# The two arms deliver nearly-but-not-exactly the same reverb (make_sim_rirs
# guarantees only |delta| <= 0.05 s), so the tolerance is exercised, not bypassed.
RT60_MEASURED = (0.20, 0.45, 0.70, 0.95)
RT60_SIM_DELTA = (+0.010, -0.015, +0.008, -0.012)
# Requested values, deliberately SCRAMBLED relative to the delivered ones. Pairing
# on `rt60` would therefore produce a different assignment than pairing on
# `rir_rt60_measured` — which is exactly what test 1 detects.
RT60_REQUESTED_REAL = (0.90, 0.30, 1.00, 0.50)
SNR = (0.0, 12.0, 25.0)
N_CLIPS = 3
MODEL = "nova-3"


# ---------------------------------------------------------------------------
# Table builders — full frozen schema, so these rows are what the runner writes
# ---------------------------------------------------------------------------

def _row(clip_id, name, rt60_req, rt60_meas, snr, wer, conf,
         model=MODEL, failed=False):
    row = {
        "clip_id": clip_id, "condition_name": name,
        "rt60": float(rt60_req), "snr_db": float(snr), "noise_type": "babble",
        "codec": "none", "mic_rolloff": 0.0, "model": model,
        "transcript": None if failed else "call maria at four zero five",
        "wer": float(wer), "n_ref": 6, "n_sub": 1, "n_del": 0, "n_ins": 0,
        "n_match": 5, "mean_conf": float(conf), "utterance_conf": float(conf),
        "word_confidences": "[]", "edits": "[]",
        "rir_key": f"rir/{name}", "rir_rt60_measured": float(rt60_meas),
        "noise_key": "noise/babble/b1.wav", "failed": failed,
        "error": "timeout" if failed else "", "run_id": "test", "ts": "2026-01-01",
    }
    assert set(row) == set(MASTER_COLUMNS), "test rows drifted from the frozen schema"
    return row


def _wer_real(rt60_meas, snr):
    """Planted truth: WER rises with DELIVERED reverb and falls with SNR."""
    return 0.06 + 0.45 * (rt60_meas - 0.20) + 0.30 * (1.0 - snr / 25.0)


def build_tables(sim_wer, sim_conf=None, real_conf=None):
    """
    One real-run table and one sim-run table over the same condition list.

    `sim_wer(wer_real, i)` maps the real WER of the i-th condition to the sim WER —
    that mapping IS the planted structure each test asserts on.
    """
    real, sim = [], []
    i = 0
    for k, rt_m in enumerate(RT60_MEASURED):
        rt_sim = rt_m + RT60_SIM_DELTA[k]
        for snr in SNR:
            wr = _wer_real(rt_m, snr)
            ws = sim_wer(wr, i)
            cr = real_conf(wr, rt_m) if real_conf else 0.95 - 0.2 * wr
            cs = sim_conf(ws, rt_sim) if sim_conf else 0.95 - 0.2 * ws
            name_r = f"rt60{RT60_REQUESTED_REAL[k]:g}_snr{snr:g}"
            name_s = f"rt60{rt_m:g}_snr{snr:g}"          # sim run requested != real
            for c in range(N_CLIPS):
                real.append(_row(f"u{c:02d}", name_r, RT60_REQUESTED_REAL[k],
                                 rt_m, snr, wr, cr))
                sim.append(_row(f"u{c:02d}", name_s, rt_m, rt_sim, snr, ws, cs))
            i += 1
    return real, sim


# ---- 1: the join key is the DELIVERED RT60, never the requested one ----------

def test_pairs_on_measured_rt60():
    real, sim = build_tables(lambda w, i: w + 0.10)
    paired = pair_conditions(real, sim, model=MODEL)
    pairs = paired["pairs"]

    assert paired["matched_on"] == MEASURED_RT60 == "rir_rt60_measured"
    assert len(pairs) == len(RT60_MEASURED) * len(SNR) == 12, len(pairs)
    assert not paired["unmatched_real"] and not paired["unmatched_sim"]

    for p in pairs:
        # joined on delivered reverb, inside the make_sim_rirs tolerance...
        assert abs(p["rt60_delta"]) <= 0.02, p
        assert abs(p["rt60_measured_real"] - p["rt60_measured_sim"]) <= 0.02, p
        # ...while the REQUESTED values disagree, so a join on `rt60` was impossible
        assert p["rt60_requested_real"] != p["rt60_requested_sim"], p
        # and only the measured join reproduces the planted constant offset
        assert abs(p["gap"] - 0.10) < 1e-9, p

    # a requested-based join would have mismatched the reverb: prove the scramble
    # really is a different assignment, not an accidental identity
    reqs = {(p["rt60_requested_real"], p["rt60_requested_sim"]) for p in pairs}
    assert all(a != b for a, b in reqs), reqs
    print(f"OK 1: {len(pairs)} pairs joined on {MEASURED_RT60} "
          f"(max |delta| {paired['max_abs_rt60_delta']:.3f} s) while the requested "
          f"{REQUESTED_RT60} values disagree on every pair")


def test_missing_measured_column_raises():
    """No delivered-RT60 column => refuse, never silently fall back to `rt60`."""
    real, sim = build_tables(lambda w, i: w + 0.10)
    for r in sim:
        r[MEASURED_RT60] = ""            # what a runner that skipped it would write
    try:
        pair_conditions(real, sim, model=MODEL)
    except ValueError as e:
        assert MEASURED_RT60 in str(e), str(e)
        print(f"OK 1b: missing {MEASURED_RT60} raises instead of falling back to "
              f"{REQUESTED_RT60!r}")
        return
    raise AssertionError("pairing accepted a table with no delivered-RT60 column")


# ---- 2: a planted constant level offset comes back, inside its CI ------------

def test_constant_offset_recovered():
    offset = 0.12
    # deterministic wiggle so the bootstrap CI has real width (a perfectly
    # constant gap gives a degenerate zero-width interval and proves nothing)
    real, sim = build_tables(lambda w, i: w + offset + 0.02 * math.sin(i * 1.7))
    paired = pair_conditions(real, sim, model=MODEL)
    lv = gap_summary(paired["pairs"], seed=0)

    assert lv["n"] == 12, lv
    assert abs(lv["mean_gap"] - offset) < 0.01, lv["mean_gap"]
    assert lv["ci_lo"] <= offset <= lv["ci_hi"], lv
    assert lv["ci_hi"] > lv["ci_lo"], "degenerate CI"
    assert lv["direction"] == "overestimates" and lv["significant"], lv
    print(f"OK 2: planted +{offset:.2f} WER offset recovered as "
          f"{lv['mean_gap']:+.4f} [95% CI {lv['ci_lo']:+.4f}, {lv['ci_hi']:+.4f}] "
          f"— planted value inside the CI")


# ---- 3: THE HEADLINE — level can be badly wrong while ORDER survives ---------

def test_order_survives_a_large_level_offset():
    # monotone in wer_real => ranking identical; intercept+slope => level far off
    real, sim = build_tables(lambda w, i: 0.35 + 0.5 * w)
    res = sim2real_report(real, sim, model=MODEL, n_boot=1000, seed=0)
    lv, od = res["level"], res["order"]

    assert od["n"] == 12
    assert od["spearman"] > 0.99, od          # ordering preserved
    assert od["kendall"] > 0.99, od
    assert od["verdict"] == "ordering preserved", od
    assert lv["mean_abs_gap"] > 0.15, lv      # ...while the level is far off
    assert lv["significant"], lv
    assert res["headline"]["verdict"] == "ORDER PRESERVED, LEVEL OFFSET", res["headline"]

    # the two arms really are far apart in absolute terms, not just noisy
    worst = max(abs(p["gap"]) for p in res["pairs"])
    assert worst > 0.15, worst
    print(f"OK 3: rho = {od['spearman']:.3f} (order preserved) with a mean |gap| of "
          f"{lv['mean_abs_gap'] * 100:.1f} pts -> '{res['headline']['verdict']}' — "
          f"rank the conditions with sim, don't quote its numbers")


# ---- 4: ...and when the ordering genuinely differs, rho must drop ------------

def test_order_broken_is_detected():
    # sim ranks the conditions in the OPPOSITE order (max real WER ~ 0.7)
    real, sim = build_tables(lambda w, i: 0.75 - w)
    res = sim2real_report(real, sim, model=MODEL, n_boot=1000, seed=0)
    od = res["order"]

    assert od["spearman"] < -0.9, od
    assert od["verdict"] == "ordering NOT preserved", od
    assert res["headline"]["verdict"] == "ORDER NOT PRESERVED", res["headline"]

    # and the preserved-order case really does score higher on the same tables
    _, sim_ok = build_tables(lambda w, i: 0.35 + 0.5 * w)
    rho_ok = rank_agreement(pair_conditions(real, sim_ok, model=MODEL)["pairs"])
    assert rho_ok["spearman"] - od["spearman"] > 1.5, (rho_ok, od)
    print(f"OK 4: reordered sim -> rho = {od['spearman']:.3f} "
          f"('{od['verdict']}') vs rho = {rho_ok['spearman']:.3f} when the same "
          f"real table is paired with an order-preserving sim")


# ---- 5: failed rows are excluded and counted, per model ---------------------

def test_failed_rows_excluded_not_averaged():
    real, sim = build_tables(lambda w, i: w + 0.10)
    target = real[0]["condition_name"]
    clean_wer = real[0]["wer"]

    # one extra FAILED measurement in one condition: the runner's sentinel is
    # WER 1.0 with a null transcript and NaN confidence
    real.append(_row("u99", target, real[0]["rt60"], real[0][MEASURED_RT60],
                     real[0]["snr_db"], 1.0, float("nan"), failed=True))
    # a second model in the same table, failing everywhere — must not be counted
    for r in list(real):
        real.append(_row(r["clip_id"], r["condition_name"], r["rt60"],
                         r[MEASURED_RT60], r["snr_db"], 1.0, float("nan"),
                         model="whisper-base", failed=True))

    agg = {c["condition_name"]: c for c in aggregate_conditions(real, model=MODEL)}
    rec = agg[target]
    assert rec["n"] == N_CLIPS, rec["n"]                        # the failure is gone
    assert abs(rec["wer"] - clean_wer) < 1e-12, rec["wer"]      # not dragged toward 1.0
    naive = (N_CLIPS * clean_wer + 1.0) / (N_CLIPS + 1)
    assert abs(rec["wer"] - naive) > 0.15, "failed row was averaged in"

    res = sim2real_report(real, sim, model=MODEL, n_boot=500, seed=0)
    assert all(abs(p["gap"] - 0.10) < 1e-9 for p in res["pairs"]), "failure leaked"
    fr = res["failures_real"]
    assert fr["n_failed"] == 1, fr        # only this model's failure, not whisper's
    assert fr["failed_by_condition"] == {target: 1}, fr
    print(f"OK 5: failed row excluded (mean WER {rec['wer']:.3f}, not the "
          f"averaged-in {naive:.3f}), reported as {fr['n_failed']}/{fr['n_rows']} "
          f"for {MODEL} only")


# ---- 6: the dead-zone set is model_compare's flags, not a private copy ------

def test_dead_zone_set_reuses_model_compare():
    # real: the two highest-reverb rows stay CONFIDENT while WER is high
    # (a dead zone). sim: same WERs, but its confidence collapses there, so a
    # sim-only testbed would call it a loud failure and MISS the dead zone.
    real, sim = build_tables(
        lambda w, i: w,
        real_conf=lambda w, rt: 0.95 if rt >= 0.70 else 0.55,
        sim_conf=lambda w, rt: 0.40 if rt >= 0.70 else 0.90,
    )
    res = sim2real_report(real, sim, model=MODEL, n_boot=500, seed=0)
    dz = res["dead_zones"]

    assert dz["n_real"] > 0 and dz["n_real"] > dz["n_sim"], dz
    assert dz["real_only"], dz            # sim would miss these dead zones
    assert not dz["same_set"], dz
    assert 0.0 <= dz["jaccard"] < 1.0, dz

    # identical to calling model_compare.dead_zone_flags directly on the same
    # aggregated table => D4 flags exactly the cells D1/L1 flag
    tbl = aggregate_conditions(real, model=MODEL)
    direct = {c["condition_name"] for c, f
              in zip(tbl, dead_zone_flags(tbl, dz["wer_hi"], dz["conf_pct_hi"]))
              if f}
    assert direct == set(dz["agree"]) | set(dz["real_only"]), (direct, dz)
    print(f"OK 6: dead zones real={dz['n_real']} sim={dz['n_sim']} "
          f"(Jaccard {dz['jaccard']:.2f}); sim MISSES {len(dz['real_only'])} — set "
          f"matches model_compare.dead_zone_flags exactly")


# ---- 7: the CLIP half of the premise — arms are intersected, and it is said --

# The sim arm ran a strict SUBSET of the real arm's clips: exactly the shape of
# the real tables, where the sim arm was run on the 10-clip AL set to save API
# spend while the real arm ran all 40.
SIM_CLIPS = ("u00", "u01", "u02")
REAL_ONLY_CLIPS = ("u03", "u04", "u05", "u06")
CLIP_DIFFICULTY = 0.30      # how much HARDER the clips only the real arm ran are


def build_unmatched_clip_tables(offset=0.10):
    """
    Same conditions on both arms, DIFFERENT clip sets — the planted confound.

    Every condition's sim WER is its real WER plus `offset`, on the clips both
    arms actually ran: `offset` IS the sim-vs-real effect and nothing else. The
    clips only the real arm ran are planted `CLIP_DIFFICULTY` points harder, so:

      matched   (common clips only) : gap == +offset            (what is true)
      unmatched (each arm's own set): gap == offset - 0.30*4/7  (what a silent
                                      join reports) == -0.071

    The confound is deliberately large enough to flip the SIGN of the finding,
    so a test that merely checked "gap is finite" could not pass by accident.
    """
    real, sim = [], []
    for k, rt_m in enumerate(RT60_MEASURED):
        rt_sim = rt_m + RT60_SIM_DELTA[k]
        for snr in SNR:
            wr = _wer_real(rt_m, snr)
            ws = wr + offset
            name_r = f"rt60{RT60_REQUESTED_REAL[k]:g}_snr{snr:g}"
            name_s = f"rt60{rt_m:g}_snr{snr:g}"
            for c in SIM_CLIPS:
                real.append(_row(c, name_r, RT60_REQUESTED_REAL[k], rt_m, snr,
                                 wr, 0.95 - 0.2 * wr))
                sim.append(_row(c, name_s, rt_m, rt_sim, snr, ws, 0.95 - 0.2 * ws))
            for c in REAL_ONLY_CLIPS:
                hard = min(1.0, wr + CLIP_DIFFICULTY)
                real.append(_row(c, name_r, RT60_REQUESTED_REAL[k], rt_m, snr,
                                 hard, 0.95 - 0.2 * hard))
    return real, sim


def test_clip_sets_are_intersected_before_aggregating():
    """
    The defect this test pins: each condition's real mean was taken over the
    real arm's 40 clips and its sim mean over the sim arm's 10 DIFFERENT ones,
    so the reported gap silently absorbed clip difficulty on top of the RIR
    provenance it claims to isolate.
    """
    from deadzone.analysis.sim2real import format_sim2real

    offset = 0.10
    real, sim = build_unmatched_clip_tables(offset)

    # Compute BOTH arithmetics here rather than assume them, so the test states
    # the size of the error it is guarding against.
    sim_wer = {c["condition_name"]: c["wer"]
               for c in aggregate_conditions(sim, model=MODEL)}
    unmatched = {c["condition_name"]: c["wer"]
                 for c in aggregate_conditions(real, model=MODEL)}
    matched = {c["condition_name"]: c["wer"]
               for c in aggregate_conditions(real, model=MODEL, clips=SIM_CLIPS)}

    # sim/real condition names differ (the requested rt60 is scrambled), so pair
    # them by position in the sorted order the builder generates.
    gap_unmatched = float(np.mean([s - u for s, u in
                                   zip(sorted(sim_wer.values()),
                                       sorted(unmatched.values()))]))
    gap_matched = float(np.mean([s - m for s, m in
                                 zip(sorted(sim_wer.values()),
                                     sorted(matched.values()))]))
    assert abs(gap_matched - offset) < 1e-9, gap_matched
    assert gap_unmatched < 0 < gap_matched, (gap_unmatched, gap_matched)
    assert abs(gap_unmatched - gap_matched) > 0.15, (gap_unmatched, gap_matched)

    res = sim2real_report(real, sim, model=MODEL, n_boot=500, seed=0)
    lv, cm = res["level"], res["clip_match"]

    # 1. the module returns the MATCHED value, not the confounded one
    assert abs(lv["mean_gap"] - offset) < 1e-9, lv["mean_gap"]
    assert abs(lv["mean_gap"] - gap_unmatched) > 0.15, lv["mean_gap"]
    assert lv["direction"] == "overestimates", lv
    assert all(abs(p["gap"] - offset) < 1e-9 for p in res["pairs"]), "clips leaked"
    assert all(p["n_real"] == p["n_sim"] == len(SIM_CLIPS) for p in res["pairs"])

    # 2. and it is REPORTED, not silent — census, drop counts, and the clip set
    assert cm["matched"] is False, cm
    assert cm["n_clips_real"] == len(SIM_CLIPS) + len(REAL_ONLY_CLIPS) == 7, cm
    assert cm["n_clips_sim"] == cm["n_common"] == len(SIM_CLIPS) == 3, cm
    assert cm["common"] == sorted(SIM_CLIPS), cm
    assert cm["real_only"] == sorted(REAL_ONLY_CLIPS) and not cm["sim_only"], cm
    n_cond = len(RT60_MEASURED) * len(SNR)
    assert cm["n_rows_real_dropped"] == len(REAL_ONLY_CLIPS) * n_cond, cm
    assert cm["n_rows_sim_dropped"] == 0, cm

    # 3. and it survives into the formatted block and the dashboard payload
    text = format_sim2real(res)
    assert "CLIP SET" in text and "RESTRICTED" in text, text
    assert "real arm 7 clips, sim arm 3, common 3" in text, text
    hl = res["plot"]["headline"]
    assert hl["n_clips"] == 3 and hl["clips_matched"] is False, hl
    assert res["plot"]["clip_match"] == cm

    print(f"OK 7: sim arm ran {cm['n_clips_sim']}/{cm['n_clips_real']} clips -> both "
          f"arms restricted to the {cm['n_common']} common ones; gap "
          f"{lv['mean_gap']:+.4f} (the planted RIR effect) not {gap_unmatched:+.4f} "
          f"(RIR + clip difficulty, opposite sign), restriction reported")


def test_matched_clip_sets_report_no_restriction():
    """The clean case must say so too, so 'MATCHED' is a positive statement."""
    real, sim = build_tables(lambda w, i: w + 0.10)
    cm = clip_intersection(real, sim, model=MODEL)
    assert cm["matched"] is True, cm
    assert cm["n_common"] == cm["n_clips_real"] == cm["n_clips_sim"] == N_CLIPS, cm
    assert not cm["real_only"] and not cm["sim_only"], cm
    assert cm["n_rows_real_dropped"] == cm["n_rows_sim_dropped"] == 0, cm
    assert "no restriction" in cm["note"], cm["note"]
    print(f"OK 7b: identical clip sets -> matched={cm['matched']}, "
          f"{cm['n_common']} clips, 0 rows dropped")


def test_too_few_common_clips_raises():
    """
    The floor under report-and-proceed. Restriction is right for 10-vs-40; it is
    not right when the intersection is an anecdote about two utterances.
    """
    real, sim = build_unmatched_clip_tables()
    sim = [r for r in sim if r["clip_id"] in ("u00", "u01")]     # 2 common clips
    try:
        sim2real_report(real, sim, model=MODEL, n_boot=100, seed=0)
    except ClipSetMismatchError as e:
        assert "share only 2 clip(s)" in str(e), str(e)
        print(f"OK 7c: 2 common clips (< the {3}-clip floor) raises "
              f"ClipSetMismatchError instead of publishing a two-utterance gap")
        return
    raise AssertionError("an implausibly small clip intersection was accepted")


# ---- 8: the plot payload is JSON-serializable and self-consistent -----------

def test_plot_payload():
    import json
    real, sim = build_tables(lambda w, i: 0.35 + 0.5 * w)
    res = sim2real_report(real, sim, model=MODEL, n_boot=500, seed=0)
    payload = res["plot"]
    json.dumps(payload)                                  # must round-trip for E2
    assert len(payload["scatter"]) == len(res["pairs"])
    assert len(payload["gap_vs_rt60"]) == len(res["pairs"])
    rts = [d["rt60_measured_real"] for d in payload["gap_vs_rt60"]]
    assert rts == sorted(rts), rts
    assert payload["headline"]["spearman"] == res["order"]["spearman"]
    print(f"OK 8: plot payload serializes — {len(payload['scatter'])} scatter "
          f"points, gap-vs-RT60 sorted, headline carries both numbers")


# ---- 9: clip matching as an ENFORCED INVARIANT of the public path -----------

# Test 7 plants a one-sided subset (sim ⊂ real), which is the shape the real
# tables happen to have. An invariant must not depend on that shape, so this
# fixture disagrees in BOTH directions and adds a clip that exists on the sim
# arm only as a FAILED row — a clip that arm never actually measured, and which
# the census must therefore not count as one it ran.
INV_COMMON_CLIPS = ("u02", "u05", "u06", "u11")
INV_REAL_ONLY_CLIPS = ("u20", "u21", "u22", "u23", "u24", "u25")   # HARDER
INV_SIM_ONLY_CLIPS = ("u30", "u31")                                # EASIER
INV_SIM_FAILED_CLIP = "u77"          # sim-arm rows exist but every one failed
INV_REAL_ONLY_PENALTY = 0.30
INV_SIM_ONLY_BONUS = 0.15
# Below this the fixture has stopped discriminating the two arithmetics and the
# test can no longer tell a fixed module from a broken one — so it says so
# rather than passing vacuously.
MIN_ARITHMETIC_SEPARATION = 0.15


def build_two_sided_clip_tables(offset=0.10):
    """
    Same condition list on both arms; clip sets that disagree in both directions.

    On the clips BOTH arms ran, sim WER is exactly real WER + `offset`, so
    `offset` is the whole sim-vs-real effect and nothing else. The clips only one
    arm ran are deliberately skewed — real-only harder, sim-only easier — so the
    unmatched arithmetic lands a long way from `offset` (here: the other side of
    zero). The test recomputes BOTH arithmetics from these rows rather than
    hardcoding either, so it keeps its meaning if this fixture is edited.
    """
    real, sim = [], []
    for k, rt_m in enumerate(RT60_MEASURED):
        rt_sim = rt_m + RT60_SIM_DELTA[k]
        for snr in SNR:
            wr = _wer_real(rt_m, snr)
            ws = wr + offset
            name_r = f"rt60{RT60_REQUESTED_REAL[k]:g}_snr{snr:g}"
            name_s = f"rt60{rt_m:g}_snr{snr:g}"
            for c in INV_COMMON_CLIPS:
                real.append(_row(c, name_r, RT60_REQUESTED_REAL[k], rt_m, snr,
                                 wr, 0.95 - 0.2 * wr))
                sim.append(_row(c, name_s, rt_m, rt_sim, snr, ws, 0.95 - 0.2 * ws))
            for c in INV_REAL_ONLY_CLIPS:
                hard = wr + INV_REAL_ONLY_PENALTY
                real.append(_row(c, name_r, RT60_REQUESTED_REAL[k], rt_m, snr,
                                 hard, 0.95 - 0.2 * hard))
            for c in INV_SIM_ONLY_CLIPS:
                easy = ws - INV_SIM_ONLY_BONUS
                sim.append(_row(c, name_s, rt_m, rt_sim, snr, easy,
                                0.95 - 0.2 * easy))
            # measured by neither arm: present, but every row is a failure
            sim.append(_row(INV_SIM_FAILED_CLIP, name_s, rt_m, rt_sim, snr,
                            1.0, float("nan"), failed=True))
    return real, sim


def _clip_ids(rows, model=MODEL):
    """Independent recount of which clips an arm actually MEASURED.

    Deliberately does not call the module's own `usable_rows`: a census is only
    a check if something outside the code under test computes the expected value.
    """
    return {r["clip_id"] for r in rows
            if r["model"] == model and not r["failed"]}


def test_clip_matching_is_an_enforced_invariant():
    """
    The invariant, asserted on the public path rather than on one fixture's
    output: both arms' per-condition aggregates are computed over the SAME clips
    and the SAME number of them, the reported gap IS the clip-matched arithmetic
    and is NOT the unmatched one, and the census tells the truth about which
    clips that was and how many rows it cost.

    Designed to fail loudly, not to document: if a future edit drops the
    restriction, `n_real != n_sim` on the pairs, the aggregate tables carry a
    clip count that is not the intersection, and the gap swings to the unmatched
    value — three independent assertions, any one of which fires.
    """
    offset = 0.10
    real, sim = build_two_sided_clip_tables(offset)

    # --- what the two arithmetics ARE, recomputed here from the raw rows ------
    common = sorted(_clip_ids(real) & _clip_ids(sim))
    res = sim2real_report(real, sim, model=MODEL, n_boot=500, seed=0)
    paired = pair_conditions(real, sim, model=MODEL)

    def _wer_by_condition(rows, clips):
        return {c["condition_name"]: c["wer"]
                for c in aggregate_conditions(rows, model=MODEL, clips=clips)}

    real_matched = _wer_by_condition(real, common)
    sim_matched = _wer_by_condition(sim, common)
    real_unmatched = _wer_by_condition(real, None)      # each arm's OWN clip set
    sim_unmatched = _wer_by_condition(sim, None)

    # Pair the two arms exactly as the module does (on delivered RT60), so the
    # only thing that differs between the arithmetics is the CLIP SET.
    names = [(p["condition_real"], p["condition_sim"]) for p in res["pairs"]]
    assert names, "no pairs — fixture broken before the invariant is even tested"
    gap_matched = float(np.mean([sim_matched[s] - real_matched[r]
                                 for r, s in names]))
    gap_unmatched = float(np.mean([sim_unmatched[s] - real_unmatched[r]
                                   for r, s in names]))

    # Precondition on the FIXTURE, not on the module: if these two ever converge
    # this test can no longer distinguish a fixed module from a broken one, and
    # it must say so instead of passing for free.
    assert abs(gap_matched - gap_unmatched) > MIN_ARITHMETIC_SEPARATION, (
        f"fixture no longer separates the matched ({gap_matched:+.4f}) and "
        f"unmatched ({gap_unmatched:+.4f}) arithmetics — this test cannot "
        f"detect the regression it exists for; re-skew the one-arm-only clips")

    # --- 1. the aggregates are over the SAME clip set, and the SAME count -----
    n_common = len(common)
    for p in res["pairs"]:
        assert p["n_real"] == p["n_sim"] == n_common, (
            f"per-condition aggregates over different clip counts: "
            f"real n={p['n_real']} vs sim n={p['n_sim']} (common {n_common}) "
            f"on {p['condition_real']}")
    for side in ("real_table", "sim_table"):
        for c in paired[side]:
            assert c["n_clips"] == n_common, (
                f"{side} row {c['condition_name']} aggregated over "
                f"{c['n_clips']} clips, not the {n_common}-clip intersection")
            assert c["n"] == n_common, (side, c["condition_name"], c["n"])

    # --- 2. the gap IS the matched arithmetic and is NOT the unmatched one ----
    assert abs(res["level"]["mean_gap"] - gap_matched) < 1e-12, (
        res["level"]["mean_gap"], gap_matched)
    assert abs(res["level"]["mean_gap"] - gap_unmatched) > MIN_ARITHMETIC_SEPARATION, (
        f"reported gap {res['level']['mean_gap']:+.4f} is the UNMATCHED "
        f"arithmetic {gap_unmatched:+.4f}, not the clip-matched "
        f"{gap_matched:+.4f} — clip difficulty is back in the finding")
    # the planted RIR effect, uncontaminated, on every pair
    assert abs(res["level"]["mean_gap"] - offset) < 1e-12, res["level"]["mean_gap"]
    assert all(abs(p["gap"] - offset) < 1e-9 for p in res["pairs"]), "clips leaked"

    # --- 3. the census is present AND truthful ------------------------------
    cm = res["clip_match"]
    assert cm["matched"] is False, cm
    assert cm["common"] == common and cm["n_common"] == n_common, cm
    assert cm["n_clips_real"] == len(_clip_ids(real)), cm
    assert cm["n_clips_sim"] == len(_clip_ids(sim)), cm
    assert cm["real_only"] == sorted(INV_REAL_ONLY_CLIPS), cm
    assert cm["sim_only"] == sorted(INV_SIM_ONLY_CLIPS), cm
    # a clip the sim arm only ever FAILED on was never measured there: it must
    # not be counted as a clip that arm ran, nor leak into either set
    assert INV_SIM_FAILED_CLIP not in cm["common"], cm
    assert INV_SIM_FAILED_CLIP not in cm["sim_only"], cm
    # dropped-row counts, recounted independently
    exp_real_drop = sum(1 for r in real if r["model"] == MODEL and not r["failed"]
                        and r["clip_id"] not in set(common))
    exp_sim_drop = sum(1 for r in sim if r["model"] == MODEL and not r["failed"]
                       and r["clip_id"] not in set(common))
    assert cm["n_rows_real_dropped"] == exp_real_drop > 0, (cm, exp_real_drop)
    assert cm["n_rows_sim_dropped"] == exp_sim_drop > 0, (cm, exp_sim_drop)
    assert cm["n_rows_real_kept"] + cm["n_rows_real_dropped"] == cm["n_rows_real"]
    assert cm["n_rows_sim_kept"] + cm["n_rows_sim_dropped"] == cm["n_rows_sim"]
    # and it reaches the consumer surfaces, not just the internal dict
    hl = res["plot"]["headline"]
    assert hl["n_clips"] == n_common and hl["clips_matched"] is False, hl
    assert res["plot"]["clip_match"] == cm

    print(f"OK 9: invariant holds on the public path — both arms aggregated over "
          f"the same {n_common} clips ({', '.join(common)}); gap "
          f"{res['level']['mean_gap']:+.4f} == matched {gap_matched:+.4f}, not "
          f"unmatched {gap_unmatched:+.4f}; census truthful "
          f"({cm['n_clips_real']}/{cm['n_clips_sim']} clips, "
          f"{cm['n_rows_real_dropped']}/{cm['n_rows_sim_dropped']} rows dropped)")


def test_no_unrestricted_aggregation_path_survives():
    """
    The consumer guard. Every route into a per-condition mean must be restricted.

    The previous test checks the NUMBERS; this one checks the CALLS, because a
    consumer could re-introduce an unrestricted aggregation whose output still
    looks plausible on a given fixture. `aggregate_conditions(rows, model)` with
    `clips=None` averages over whichever clips that arm happens to hold — that
    single defaulted argument IS the defect — so we spy on it while the only
    public entry point runs and assert no call omitted the restriction.
    """
    import deadzone.analysis.sim2real as s2r

    real, sim = build_two_sided_clip_tables(0.10)
    expected = set(_clip_ids(real) & _clip_ids(sim))

    seen: list[object] = []
    original = s2r.aggregate_conditions

    def _spy(rows, model=None, clips=None):
        seen.append(clips)
        return original(rows, model, clips=clips)

    s2r.aggregate_conditions = _spy
    try:
        res = s2r.sim2real_report(real, sim, model=MODEL, n_boot=200, seed=0)
    finally:
        s2r.aggregate_conditions = original      # never leak the patch

    assert seen, "sim2real_report aggregated nothing — the spy never fired"
    for i, clips in enumerate(seen):
        assert clips is not None, (
            f"aggregation call {i} ran UNRESTRICTED (clips=None): it averaged "
            f"over whichever clips that arm holds, which is exactly the "
            f"confound this layer exists to exclude")
        assert set(clips) == expected, (
            f"aggregation call {i} restricted to {sorted(set(clips))}, not the "
            f"{sorted(expected)} clips both arms measured")
    # both arms, and only both arms
    assert len(seen) == 2, (f"expected one restricted aggregation per arm, saw "
                            f"{len(seen)} — a new aggregation path appeared")
    assert res["clip_match"]["n_common"] == len(expected), res["clip_match"]

    print(f"OK 9b: all {len(seen)} aggregation call(s) on the public path were "
          f"restricted to the {len(expected)}-clip intersection — no unrestricted "
          f"path survives")


if __name__ == "__main__":
    test_pairs_on_measured_rt60()
    test_missing_measured_column_raises()
    test_constant_offset_recovered()
    test_order_survives_a_large_level_offset()
    test_order_broken_is_detected()
    test_failed_rows_excluded_not_averaged()
    test_dead_zone_set_reuses_model_compare()
    test_clip_sets_are_intersected_before_aggregating()
    test_matched_clip_sets_report_no_restriction()
    test_too_few_common_clips_raises()
    test_plot_payload()
    test_clip_matching_is_an_enforced_invariant()
    test_no_unrestricted_aggregation_path_survives()
    print("\nAll sim2real tests passed — D4 pairs on DELIVERED RT60 over the two "
          "arms' COMMON CLIP SET (and says which), recovers a planted level "
          "offset inside its CI, separates 'wrong level' from 'wrong order', and "
          "never averages a failed measurement in. Clip matching is enforced as "
          "an invariant of the public path, not merely exhibited: same clip set "
          "and same count on both arms, the gap checked against BOTH arithmetics "
          "recomputed inline, a truthful census, and no unrestricted aggregation "
          "call left anywhere.")
