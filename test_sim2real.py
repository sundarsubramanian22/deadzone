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

Plus: the dead-zone set comparison must be the SAME flags D1/L1 use
(`model_compare.dead_zone_flags`), not a private reimplementation.

Deterministic (closed-form tables, fixed bootstrap seed). Run:
    python3 test_sim2real.py
"""
import math

import numpy as np

from analysis.sim2real import (
    MEASURED_RT60, REQUESTED_RT60, aggregate_conditions, dead_zone_agreement,
    gap_summary, pair_conditions, rank_agreement, sim2real_report,
)
from model_compare import dead_zone_flags
from run_experiment import MASTER_COLUMNS

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


# ---- 7: the plot payload is JSON-serializable and self-consistent -----------

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
    print(f"OK 7: plot payload serializes — {len(payload['scatter'])} scatter "
          f"points, gap-vs-RT60 sorted, headline carries both numbers")


if __name__ == "__main__":
    test_pairs_on_measured_rt60()
    test_missing_measured_column_raises()
    test_constant_offset_recovered()
    test_order_survives_a_large_level_offset()
    test_order_broken_is_detected()
    test_failed_rows_excluded_not_averaged()
    test_dead_zone_set_reuses_model_compare()
    test_plot_payload()
    print("\nAll sim2real tests passed — D4 pairs on DELIVERED RT60, recovers a "
          "planted level offset inside its CI, separates 'wrong level' from "
          "'wrong order', and never averages a failed measurement in.")
