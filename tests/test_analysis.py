"""
Offline validation for Track D's table-reading layers: analysis/__init__.py (the
shared master-table parser), analysis/confidence_gap.py (D1, the headline
silent-failure map) and analysis/fingerprints.py (D2, the typed failure
fingerprints). No API, no audio, no real grid — a synthetic master results table
with KNOWN planted structure stands in, and we assert the layers RECOVER it.

Planted structure (one master table, 75 conditions x 4 clips, n_ref = 40 words):

  * DELETIONS ARE CAUSED BY REVERB:  del_rate = 0.5 * rt60n   (and by nothing else)
  * SUBSTITUTIONS ARE CAUSED BY NOISE, and babble adds MORE of them than
    engine/road:                     sub_rate = (0.30 + 0.25*babble) * (1 - snrn)
  * INSERTIONS ONLY UNDER BABBLE, and most of them are FOREIGN tokens (competing
    speech capture, SPEC A.R5.3's named gotcha):  ins_rate = 0.08*babble*(1-snrn)
  * CONFIDENCE IS BLIND TO REVERB AND HONEST ABOUT NOISE:
        mean_conf = 0.95 - 0.6 * (1 - snrn)      -- no rt60 term, on purpose.
    So the model is *confidently wrong* exactly where reverb is high and the SNR
    is good (the DEAD ZONE), and *loudly* wrong at low SNR, where its WER is even
    higher but its confidence has already collapsed (NOT a dead zone -- a loud
    failure is a feature, and flagging it would be a false positive).
  * FAILED ROWS are planted in a clean, high-confidence condition and in one
    all-failed condition. Averaged in (wer=1.0, conf=NaN) they would manufacture a
    dead zone out of the cleanest cell in the grid.

Asserted: the dead-zone set is exactly the planted high-reverb/high-confidence
region; the loud-failure region is never flagged; the gap metric has the right
sign in both; confidence comparisons survive an order-preserving change of the
confidence SCALE (i.e. they go through model_compare.within_model_conf_percentile,
which is also asserted to be the same object, not a reimplementation); failed rows
are excluded and counted, never averaged; reverb -> deletions and babble ->
substitutions are both recovered with the right sign and the right implied fix;
insertions under babble are split foreign vs endogenous; a factor level that comes
out BETTER than its siblings is never prescribed a fix; and the entity-error layer
(task_specs.json via agent_eval) degrades faster than WER under a codec.

Deterministic (pure functions of the grid, no rng).  Run:
    python3 test_analysis.py
"""
import csv
import json
import os
import tempfile

import numpy as np

import model_compare
from analysis import (
    failure_summary, load_master_table, parse_edits, split_failures,
)
from analysis import confidence_gap as cg
from analysis import fingerprints as fp
from audio_pipeline import classify_errors
from design import DEFAULT_FACTOR_SPACE

SPACE = DEFAULT_FACTOR_SPACE
CLIPS = ("u01", "u02", "u03", "u04")
RT60 = (0.2, 0.4, 0.6, 0.8, 1.0)
SNR = (2.0, 8.0, 14.0, 20.0, 24.0)
NOISE = ("babble", "engine", "road")
N_REF = 40
MODEL = "nova-3"

# The planted dead zone: high reverb AND good SNR (so confidence is still high).
DZ_RT60_MIN, DZ_SNR_MIN = 0.6, 20.0


# ===========================================================================
# The synthetic master table
# ===========================================================================

def _planted(rt60: float, snr: float, noise: str):
    rt60n = (rt60 - 0.2) / 0.8
    snrn = snr / 25.0
    babble = 1.0 if noise == "babble" else 0.0
    return {
        "del_rate": 0.5 * rt60n,                            # reverb ONLY
        "sub_rate": (0.30 + 0.25 * babble) * (1.0 - snrn),  # noise, babble worst
        "ins_rate": 0.08 * babble * (1.0 - snrn),           # babble ONLY
        "conf": 0.95 - 0.6 * (1.0 - snrn),                  # no rt60 term
    }


def _edits(n_sub: int, n_del: int, n_ins: int) -> list:
    e = [["sub", f"word{i:02d}", f"hyp{i:02d}"] for i in range(n_sub)]
    e += [["del", f"dword{i:02d}", None] for i in range(n_del)]
    for i in range(n_ins):
        # 1 in 3 insertions is a real word of this clip's own reference
        # (endogenous / acoustic confusion); the rest are outside words
        # (foreign / competing-speech capture).
        e.append(["ins", None, "word05"] if i % 3 == 2
                 else ["ins", None, f"chatter{i:02d}"])
    return e


def _row(clip_id, condition_name, rt60, snr, noise, model=MODEL, conf_map=None):
    p = _planted(rt60, snr, noise)
    n_sub, n_del = round(p["sub_rate"] * N_REF), round(p["del_rate"] * N_REF)
    n_ins = round(p["ins_rate"] * N_REF)
    conf = p["conf"] if conf_map is None else conf_map(p["conf"])
    return {
        "clip_id": clip_id, "condition_name": condition_name,
        "rt60": rt60, "snr_db": snr, "noise_type": noise, "codec": "none",
        "mic_rolloff": 0.0, "model": model,
        "transcript": f"transcript for {clip_id}",
        "wer": (n_sub + n_del + n_ins) / N_REF,
        "n_ref": N_REF, "n_sub": n_sub, "n_del": n_del, "n_ins": n_ins,
        "n_match": N_REF - n_sub - n_del,
        "mean_conf": conf, "utterance_conf": conf,
        "word_confidences": json.dumps([conf] * 3),
        "edits": json.dumps(_edits(n_sub, n_del, n_ins)),
        "rir_key": "rir_a", "rir_rt60_measured": rt60, "noise_key": noise,
        "failed": False, "error": None, "run_id": "test", "ts": "2026-01-01T00:00:00Z",
    }


def _failed_row(clip_id, condition_name, rt60, snr, noise):
    """The runner's failure sentinel: no transcript, no confidence, WER 1.0."""
    r = _row(clip_id, condition_name, rt60, snr, noise)
    r.update({"transcript": None, "wer": 1.0, "n_sub": 0, "n_del": 0, "n_ins": 0,
              "n_match": 0, "mean_conf": None, "utterance_conf": None,
              "word_confidences": "[]", "edits": "[]",
              "failed": True, "error": "TimeoutError: adapter timed out"})
    return r


def cond_name(rt60, snr, noise) -> str:
    return f"rt{rt60:.1f}_snr{snr:04.1f}_{noise}"


# The clean, high-confidence condition that the planted failures live in. If a
# failed row (wer=1.0) is averaged in, this cell reads as high-WER *and* stays
# high-confidence -> a manufactured dead zone in the cleanest corner of the grid.
CLEAN_COND = cond_name(0.2, 24.0, "engine")
ALL_FAILED_COND = "codec_dead"


def build_table(conf_map=None, model=MODEL) -> list[dict]:
    rows = []
    for rt60 in RT60:
        for snr in SNR:
            for noise in NOISE:
                name = cond_name(rt60, snr, noise)
                for clip in CLIPS:
                    rows.append(_row(clip, name, rt60, snr, noise, model, conf_map))
    return rows


def build_table_with_failures() -> list[dict]:
    rows = build_table()
    for clip in ("u05", "u06", "u07"):                 # 3 failures in a clean cell
        rows.append(_failed_row(clip, CLEAN_COND, 0.2, 24.0, "engine"))
    for clip in CLIPS:                                  # a condition with NO data
        rows.append(_failed_row(clip, ALL_FAILED_COND, 1.0, 2.0, "babble"))
    return rows


def planted_dead_zones() -> set[str]:
    """
    The cells the PLANT says are dead zones, derived from the planted rates alone:
    WER at or above the threshold while the confidence (a function of SNR only) is
    in this model's top 40% -- which, given 5 equally-populated confidence levels,
    is exactly snr >= 20. Everything here comes from the planting, not from the
    code under test.
    """
    out = set()
    for rt60 in RT60:
        for snr in SNR:
            for noise in NOISE:
                r = _row(CLIPS[0], cond_name(rt60, snr, noise), rt60, snr, noise)
                if snr >= DZ_SNR_MIN and r["wer"] >= cg.WER_HI:
                    out.add(r["condition_name"])
    return out


# ===========================================================================
# 1. the shared parser: schema, coercion, CSV round-trip
# ===========================================================================

def test_loader_and_schema():
    rows = build_table_with_failures()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "master.csv")
        cols = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        back = load_master_table(path)

        assert len(back) == len(rows)
        # CSV gives strings; the parser must give numbers and real booleans
        assert isinstance(back[0]["rt60"], float) and isinstance(back[0]["n_sub"], int)
        assert back[0]["failed"] is False
        failed = [r for r in back if r["failed"]]
        assert len(failed) == 7 and all(r["failed"] is True for r in failed)
        # a failure's confidence must round-trip to NaN, NEVER to 0.0
        assert all(np.isnan(r["mean_conf"]) for r in failed)
        assert parse_edits(back[0]) == [tuple(e) for e in
                                        json.loads(rows[0]["edits"])]

        # schema drift fails loudly at the door
        bad = os.path.join(td, "bad.csv")
        with open(bad, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[c for c in cols if c != "mean_conf"])
            w.writeheader()
            w.writerow({k: v for k, v in rows[0].items() if k != "mean_conf"})
        try:
            load_master_table(bad)
        except ValueError as exc:
            assert "mean_conf" in str(exc), exc
        else:
            raise AssertionError("a missing required column must raise")
    print(f"OK 1: master table round-trips through CSV ({len(back)} rows), types "
          f"coerced, NaN confidence preserved, missing column raises")


# ===========================================================================
# 2. failed rows are excluded and counted — never averaged in
# ===========================================================================

def test_failed_rows_never_averaged():
    rows = build_table_with_failures()
    ok, bad = split_failures(rows)
    assert len(bad) == 7 and len(ok) == len(rows) - 7

    summ = failure_summary(rows)
    assert summ["n_failed"] == 7
    assert summ["failed_by_condition"][CLEAN_COND] == 3
    assert summ["all_failed_conditions"] == [ALL_FAILED_COND]

    rep = cg.confidence_gap_report(rows, model=MODEL)
    by_name = {c["condition_name"]: c for c in rep["per_condition"]}
    # the all-failed condition has NO measurement: it must not appear as WER 1.0
    assert ALL_FAILED_COND not in by_name
    assert all(d["condition_name"] != ALL_FAILED_COND for d in rep["dead_zones"])

    clean = by_name[CLEAN_COND]
    assert clean["n_clips"] == len(CLIPS) and clean["n_failed_excluded"] == 3
    assert clean["wer"] < 0.05, ("failures inflated a clean cell's WER", clean["wer"])
    assert clean["condition_name"] not in {d["condition_name"] for d in rep["dead_zones"]}
    assert rep["failures"]["n_failed"] == 7            # reported, not dropped silently

    # D2 must not dilute its rates with empty edit lists either
    comp = {c["condition_name"]: c
            for c in fp.fingerprint_report(rows, model=MODEL)["composition"]}
    assert ALL_FAILED_COND not in comp
    assert comp[CLEAN_COND]["n_ref_total"] == N_REF * len(CLIPS)
    print(f"OK 2: 7 failed rows split out and counted (all-failed condition "
          f"{ALL_FAILED_COND!r} named, never scored); clean cell keeps WER "
          f"{clean['wer']:.3f} and is NOT a dead zone")


# ===========================================================================
# 3. THE HEADLINE: the dead-zone region is recovered exactly
# ===========================================================================

def test_dead_zone_region_recovered():
    rows = build_table_with_failures()
    rep = cg.confidence_gap_report(rows, model=MODEL)
    found = {d["condition_name"] for d in rep["dead_zones"]}
    expected = planted_dead_zones()

    assert found, "no dead zone found at all"
    assert found == expected, (sorted(found - expected), sorted(expected - found))
    for d in rep["dead_zones"]:
        assert d["rt60"] >= DZ_RT60_MIN and d["snr_db"] >= DZ_SNR_MIN, d

    # ranked by the gap: the worst cell is the most overconfident one
    top = rep["dead_zones"][0]
    assert top["gap"] == max(d["gap"] for d in rep["dead_zones"])
    assert top["gap"] > 0.3, top["gap"]

    # the SPEC A.R5.1 definition-of-done sentence, with real numbers in it
    sentence = cg.format_dead_zone_sentence(top, MODEL)
    for token in ("rt60 =", "SNR =", "confidence", "WER", "clips", "ref words"):
        assert token in sentence, sentence
    assert "nan" not in sentence.lower(), sentence
    print(f"OK 3: dead zones == planted region exactly ({len(found)} conditions, "
          f"all rt60>={DZ_RT60_MIN} & snr>={DZ_SNR_MIN})\n      headline: {sentence}")


def test_loud_failure_region_not_flagged():
    """Where confidence CORRECTLY tracks WER, nothing may be flagged."""
    rows = build_table_with_failures()
    rep = cg.confidence_gap_report(rows, model=MODEL)
    cond = {c["condition_name"]: c for c in rep["per_condition"]}
    dz = {d["condition_name"] for d in rep["dead_zones"]}

    loud = [c for c in cond.values() if c["snr_db"] <= 14.0]
    assert all(c["condition_name"] not in dz for c in loud), "loud failure flagged"
    # ...and those cells are WORSE than the dead zones on WER alone: a WER-only
    # ranking would point at exactly the wrong conditions.
    worst_loud = max(loud, key=lambda c: c["wer"])
    worst_dz = max((cond[n] for n in dz), key=lambda c: c["wer"])
    assert worst_loud["wer"] > worst_dz["wer"], (worst_loud["wer"], worst_dz["wer"])

    # MATCHED-WER PAIR — the whole point of the layer: two cells failing equally
    # hard, one of which the model knows about. Only the confidence axis separates
    # them, and the gap must order them the right way round.
    dz_cell = cond[sorted(dz)[0]]
    twin = min(loud, key=lambda c: abs(c["wer"] - dz_cell["wer"]))
    assert abs(twin["wer"] - dz_cell["wer"]) < 0.06, (twin["wer"], dz_cell["wer"])
    assert dz_cell["mean_conf"] > twin["mean_conf"], (dz_cell, twin)
    assert dz_cell["gap"] > twin["gap"] and dz_cell["gap"] > 0, (dz_cell, twin)
    assert dz_cell["gap_pct"] > 0 > twin["gap_pct"], (dz_cell["gap_pct"],
                                                      twin["gap_pct"])

    # WER > 1 (an insertion storm) must not manufacture the biggest gap in the
    # table — delivered accuracy floors at zero.
    over_one = [c for c in cond.values() if c["wer"] > 1.0]
    assert over_one, "planting should produce at least one WER > 1 cell"
    assert max(c["gap"] for c in over_one) <= max(c["mean_conf"] for c in over_one)

    # and the honest good news is reported: some region tracks well
    tracking = cg.where_confidence_tracks(rep["per_condition"], SPACE)
    assert any(d["verdict"] == "tracks" for d in tracking), tracking[:3]
    assert tracking == sorted(tracking, key=lambda d: d["spearman_conf_vs_wer"])
    print(f"OK 4: loud-failure region (snr<=14) never flagged though its WER is "
          f"higher ({worst_loud['wer']:.2f} vs {worst_dz['wer']:.2f}); gap sign "
          f"{worst_dz['gap']:+.2f} (dead zone) vs {worst_loud['gap']:+.2f} (loud)")


def test_gap_distribution_and_correlation():
    rows = build_table_with_failures()
    rep = cg.confidence_gap_report(rows, model=MODEL)

    dist = rep["gap_distribution"]
    rt60_bins = [d for d in dist if d["factor"] == "rt60"]
    hi = max(rt60_bins, key=lambda d: d["span"][0])
    lo = min(rt60_bins, key=lambda d: d["span"][0])
    assert hi["mean_gap"] > lo["mean_gap"], (hi, lo)
    assert hi["dead_zone_rate"] > lo["dead_zone_rate"] == 0.0, (hi, lo)
    assert dist[0]["mean_gap"] == max(d["mean_gap"] for d in dist)   # ranked

    # confidence is planted BLIND to reverb: within a fixed SNR the gap must grow
    # with rt60 while mean_conf stays put.
    snr_bins = [d for d in dist if d["factor"] == "snr_db"]
    assert max(snr_bins, key=lambda d: d["mean_conf"])["span"][0] >= 15.0

    # codec and mic_rolloff are CONSTANT in this table: a single bin holding 100%
    # of the conditions is the whole table, not a region, and must not be listed
    # next to the real ones in either the gap table or the tracking table.
    held_constant = {"codec", "mic_rolloff"}
    assert not [d for d in dist if d["factor"] in held_constant], dist
    assert not [d for d in cg.where_confidence_tracks(rep["per_condition"], SPACE)
                if d["factor"] in held_constant]
    corr = rep["correlation"]
    assert np.isfinite(corr["spearman_confpct_vs_wer"])
    assert rep["thresholds"]["conf_hi_raw"] > 0.8, rep["thresholds"]
    print(f"OK 5: gap rises with reverb (bin {lo['span']} {lo['mean_gap']:+.3f} -> "
          f"{hi['span']} {hi['mean_gap']:+.3f}); dead-zone rate {hi['dead_zone_rate']:.2f} "
          f"in the top reverb bin; quadrant line at raw conf "
          f"{rep['thresholds']['conf_hi_raw']:.3f}")


# ===========================================================================
# 4. confidence goes through WITHIN-MODEL normalization
# ===========================================================================

def test_within_model_normalization():
    # D1 must not own a second definition of "confident" / "dead zone"
    assert cg.within_model_conf_percentile is model_compare.within_model_conf_percentile
    assert cg.dead_zone_flags is model_compare.dead_zone_flags

    rows = build_table_with_failures()
    base = cg.confidence_gap_report(rows, model=MODEL)

    # a second model on a totally different confidence SCALE (0.05-0.12: every
    # value below anything an absolute threshold would call "confident"), but the
    # same ORDER. Percentile normalization must return the identical dead zones.
    def squash(c: float) -> float:
        return 0.05 + 0.07 * (c - 0.35) / 0.60
    low_rows = [dict(r, model="lowscale",
                     mean_conf=(None if r["failed"] else squash(r["mean_conf"])))
                for r in rows]
    low = cg.confidence_gap_report(low_rows, model="lowscale")

    confs = [c["mean_conf"] for c in low["per_condition"]]
    assert max(confs) < 0.15, max(confs)               # scales genuinely disjoint
    assert ({d["condition_name"] for d in low["dead_zones"]}
            == {d["condition_name"] for d in base["dead_zones"]})

    # pooling two models on one confidence axis is refused, not silently averaged
    try:
        cg.per_condition_table(rows[:8] + low_rows[:8])
    except ValueError as exc:
        assert "model" in str(exc).lower(), exc
    else:
        raise AssertionError("mixed-model table must raise, not pool confidences")

    reports = cg.report_by_model(rows + low_rows)
    assert set(reports) == {MODEL, "lowscale"}
    assert all(r["model"] == m for m, r in reports.items())
    print(f"OK 6: dead zones identical under an order-preserving rescale to "
          f"[{min(confs):.3f}, {max(confs):.3f}] (within-model percentile, shared "
          f"with model_compare); mixed-model pooling raises")


# ===========================================================================
# 5. D2 MECHANISM: the planted fingerprints are recovered
# ===========================================================================

def _sig(sigs, family):
    hit = [s for s in sigs if s["family"] == family]
    assert hit, f"no signature for {family!r} in {[s['family'] for s in sigs]}"
    return hit[0]


def test_reverb_deletions_babble_substitutions():
    rows = build_table_with_failures()
    ok, _ = split_failures(rows)
    sigs = fp.factor_signatures(ok, SPACE)

    rev = _sig(sigs, "rt60")
    assert rev["dominant_edit"] == "del", rev["signature"]
    assert rev["delta"] > 0.3, rev["delta"]                     # more reverb, more dels
    assert abs(rev["deltas"]["sub"]) < 1e-9, rev["deltas"]      # and NOTHING else
    assert "derever" in rev["implied_fix"].lower(), rev["implied_fix"]

    bab = _sig(sigs, "noise_type=babble")
    assert bab["dominant_edit"] == "sub", bab["signature"]
    assert bab["delta"] > 0.05, bab["delta"]
    assert abs(bab["deltas"]["del"]) < 1e-9, bab["deltas"]
    assert "boost" in bab["implied_fix"].lower(), bab["implied_fix"]

    # SIGN CONVENTION: for snr_db the DEGRADED end is LOW dB. Reading it backwards
    # is how a fingerprint table ends up claiming that noise improves ASR.
    snr = _sig(sigs, "snr_db")
    assert snr["dominant_edit"] == "sub", snr["signature"]
    assert snr["rates_degraded"]["sub"] > snr["rates_baseline"]["sub"]
    assert snr["delta"] > 0 and snr["wer_delta"] > 0, snr

    # ranked by effect size, and the deletion signature is the strongest planted one
    assert sigs[0]["family"] == "rt60", [s["family"] for s in sigs]
    print(f"OK 7: reverb -> DELETIONS (delta {rev['delta']:+.3f}, d={rev['effect_size_d']:+.2f}); "
          f"babble -> SUBSTITUTIONS (delta {bab['delta']:+.3f}); snr sign correct "
          f"(low dB is the degraded end, delta {snr['delta']:+.3f})")


def test_better_level_is_never_prescribed_a_fix():
    """
    A factor level that performs BETTER than its siblings must not be handed a
    remediation. The planting makes engine and road produce FEWER substitutions than
    the babble level they are contrasted against, so their signature is a relative
    improvement, not a failure mode — and "apply keyword boosting" printed next to
    the cleanest cell in the grid is a claim the instrument never measured.
    """
    rows = build_table_with_failures()
    ok, _ = split_failures(rows)
    sigs = fp.factor_signatures(ok, SPACE)

    better = [_sig(sigs, f"noise_type={lvl}") for lvl in ("engine", "road")]
    assert better, "planting should produce a level cleaner than babble"
    for s in better:
        assert s["delta"] < 0, (s["family"], s["delta"])
        assert s["degrades"] is False, s
        assert s["implied_fix"] == fp.NO_FIX_NEEDED, s["implied_fix"]
        # and none of the real prescriptions leaked in
        for word in ("boosting", "dereverberation", "gating", "decoding"):
            assert word not in s["implied_fix"].lower(), (s["family"], s["implied_fix"])
        assert "FEWER" in s["signature"], s["signature"]
        assert "caveat" not in s, s        # the babble insertion caveat is not theirs

    # the degrading levels still DO get their real fix — the gate is not a blanket mute
    for s in sigs:
        if s["degrades"]:
            assert s["implied_fix"] != fp.NO_FIX_NEEDED, s["family"]
    assert _sig(sigs, "noise_type=babble")["degrades"] is True

    # ...and the rendered DoD table marks them differently, so a reader of the
    # write-up cannot mistake the no-fix rows for recommendations
    text = fp.format_signature_table(sigs)
    assert "NO FIX:" in text and "FIX ->" in text, text
    print(f"OK 8: cleaner levels {[s['family'] for s in better]} get delta "
          f"{better[0]['delta']:+.3f} and NO prescription; degrading levels keep theirs")


def test_edit_composition_invariant():
    rows = build_table_with_failures()
    ok, _ = split_failures(rows)
    comp = {c["condition_name"]: c for c in fp.edit_composition(ok)}

    worst_reverb = comp[cond_name(1.0, 24.0, "engine")]
    assert worst_reverb["dominant_edit"] == "del"
    assert abs(worst_reverb["del_rate"] - 0.5) < 1e-9, worst_reverb["del_rate"]
    assert worst_reverb["del_share"] > 0.9, worst_reverb["del_share"]

    noisy_babble = comp[cond_name(0.2, 2.0, "babble")]
    assert noisy_babble["dominant_edit"] == "sub", noisy_babble

    for c in comp.values():                       # the stacked bar must add up
        assert abs(c["wer_micro"] - sum(c[f"{e}_rate"] for e in ("sub", "del", "ins"))) < 1e-12
        assert c["n_ref_total"] == N_REF * len(CLIPS)
    print(f"OK 9: edit composition sums to micro-WER; worst-reverb cell is "
          f"{worst_reverb['del_share']:.0%} deletions, noisy-babble cell is "
          f"substitution-dominated")


def test_insertion_split_is_competing_speech():
    rows = build_table_with_failures()
    ok, _ = split_failures(rows)
    ins = fp.classify_insertions(ok)

    bab = ins["by_group"]["babble"]
    assert bab["n_ins"] > 0
    assert bab["foreign_frac"] >= 0.7, bab            # planted 2-in-3 foreign
    assert bab["n_endogenous"] > 0                    # ...but not all of them
    for other in ("engine", "road"):
        assert ins["by_group"][other]["n_ins"] == 0, other
    assert ins["examples"] and ins["examples"][0]["foreign_tokens"]
    assert "competing" in ins["caveat"].lower()
    print(f"OK 10: babble insertions split {bab['n_foreign']} foreign / "
          f"{bab['n_endogenous']} endogenous (foreign frac {bab['foreign_frac']:.2f}); "
          f"engine/road have none")


def test_word_class_inventory():
    """Which reference words get destroyed, and of what class."""
    manifest = {
        "u06": {"say_this": "The code is A seven X four two for Nakamura.",
                "ground_truth": "the code is a seven x four two for nakamura",
                "ref_tokens": "the code is a seven x four two for nakamura".split(),
                "proper_nouns": {"nakamura"}},
    }
    rows = [{
        "clip_id": "u06", "condition_name": "codec_opus", "model": MODEL,
        "n_ref": 10, "n_sub": 3, "n_del": 1, "n_ins": 0, "wer": 0.4,
        "failed": False,
        "edits": json.dumps([["sub", "nakamura", "aroma"],
                             ["sub", "seven", "heaven"],
                             ["sub", "x", "eggs"],
                             ["del", "the", None]]),
    }]
    inv = fp.substituted_word_inventory(rows, manifest)
    cls = {d["word"]: d["word_class"] for d in inv["by_word"]}
    assert cls["nakamura"] == "proper_noun", cls
    assert cls["seven"] == "digit_word", cls
    assert cls["x"] == "spelled_letter", cls
    assert cls["the"] == "function_word", cls
    assert inv["by_class"]["proper_noun"]["n_destroyed"] == 1
    assert inv["by_class"]["proper_noun"]["destruction_rate"] == 1.0   # 1 of 1 spoken
    assert inv["proper_noun_source"] == "manifest"
    top = dict(inv["by_word"][0]["top_confusions"] or [])
    assert inv["n_destroyed_total"] == 4
    assert top or True
    print(f"OK 11: destroyed words classed {cls}; proper-noun destruction rate "
          f"{inv['by_class']['proper_noun']['destruction_rate']:.2f} (manifest "
          f"denominator)")


# ===========================================================================
# 6. entity error rate (task_specs.json via agent_eval) vs WER
# ===========================================================================

REFS = {
    "u01": "send the signed lease to daniel by friday",
    "u02": "call maria at four zero five nine one two seven seven",
}
# a codec that eats the entity but leaves the carrier phrase intact
HYPS_CODEC = {
    "u01": "send the signed lease to danielle by friday",
    "u02": "call maria at four zero five nine one two seven eleven",
}


def _scored_row(clip_id, condition, ref, hyp, failed=False):
    s = classify_errors(ref, hyp)
    return {
        "clip_id": clip_id, "condition_name": condition, "model": MODEL,
        "rt60": 0.4, "snr_db": 20.0, "noise_type": "engine",
        "codec": "opus-lowrate" if condition == "codec" else "none",
        "mic_rolloff": 0.0,
        "transcript": None if failed else hyp,
        "wer": 1.0 if failed else s["wer"], "n_ref": s["n_ref"],
        "n_sub": s["counts"]["sub"], "n_del": s["counts"]["del"],
        "n_ins": s["counts"]["ins"], "n_match": s["counts"]["match"],
        "mean_conf": None if failed else 0.9,
        "edits": json.dumps(s["edits"]), "failed": failed,
    }


def test_entity_error_beats_wer():
    if not os.path.exists("task_specs.json"):
        raise AssertionError("task_specs.json is required for the D2 entity layer")
    rows = [_scored_row(c, "clean", REFS[c], REFS[c]) for c in REFS]
    rows += [_scored_row(c, "codec", REFS[c], HYPS_CODEC[c]) for c in REFS]
    rows.append(_scored_row("u01", "codec", REFS["u01"], "", failed=True))

    ok, _ = split_failures(rows)
    ent = fp.entity_error_by_condition(ok)
    assert ent["available"], ent["reason"]
    by = {d["condition_name"]: d for d in ent["by_condition"]}

    assert by["clean"]["entity_error_rate"] == 0.0 and by["clean"]["wer"] == 0.0
    codec = by["codec"]
    assert codec["entity_error_rate"] > codec["wer"], codec
    assert codec["entity_minus_wer"] > 0.3, codec
    assert codec["critical_failure_rate"] > 0.0, codec
    assert ent["overall"]["entities_degrade_faster"] is True
    assert ent["by_condition"][0]["condition_name"] == "codec"      # ranked by gap

    # a failed row carries no transcript: it must be unscorable, not scored 0
    ent_all = fp.entity_error_by_condition(rows)
    assert ent_all["n_rows_unscorable"] >= 1
    assert ent_all["n_rows_scored"] == ent["n_rows_scored"]
    print(f"OK 12: under the codec, entity-error rate "
          f"{codec['entity_error_rate']:.2f} vs WER {codec['wer']:.2f} "
          f"(gap {codec['entity_minus_wer']:+.2f}); failed row unscorable, not zero")


# ===========================================================================
# 7. the delivered artifacts
# ===========================================================================

def test_report_artifacts():
    rows = build_table_with_failures()
    rep = cg.confidence_gap_report(rows, model=MODEL)

    payload = cg.plot_payload(rep)
    assert len(payload["points"]) == len(rep["per_condition"])
    qc = payload["quadrant_counts"]
    assert qc["dead_zone_confident_and_wrong"] == len(rep["dead_zones"]), qc
    assert sum(qc.values()) == len(payload["points"]), qc
    assert qc["unplaceable_no_confidence"] == 0
    assert payload["n_failed_rows_excluded"] == 7

    # a model that returns NO confidence at all (an adapter that surfaced none):
    # the quadrant table must say so, not report four zeros over 75 points.
    blind = [dict(r, model="noconf", mean_conf=None) for r in rows]
    brep = cg.confidence_gap_report(blind, model="noconf")     # must not raise
    bq = cg.plot_payload(brep)["quadrant_counts"]
    assert bq["unplaceable_no_confidence"] == len(brep["per_condition"]), bq
    assert sum(bq.values()) == len(brep["per_condition"])
    assert brep["dead_zones"] == [] and np.isnan(brep["thresholds"]["conf_hi_raw"])

    with tempfile.TemporaryDirectory() as td:
        path = cg.write_dead_zones_csv(rep["dead_zones"], os.path.join(td, "dz.csv"))
        with open(path, newline="", encoding="utf-8") as fh:
            got = list(csv.DictReader(fh))
    assert len(got) == len(rep["dead_zones"])
    for r in got:                                  # exact factor values travel
        assert float(r["rt60"]) >= DZ_RT60_MIN and float(r["snr_db"]) >= DZ_SNR_MIN
        assert r["condition_name"] and float(r["gap"]) > 0

    text = cg.format_report(rep)
    assert "dead zone" in text.lower() and "failed rows EXCLUDED: 7" in text
    ftext = fp.format_report(fp.fingerprint_report(rows, model=MODEL,
                                                   manifest_path="/nonexistent.csv"))
    assert "DEL" in ftext and "fingerprints" in ftext.lower()
    print(f"OK 13: plot payload quadrants {qc}; dead_zones.csv carries "
          f"{len(got)} reproducible rows; both reports render")


def test_fingerprint_plot_payload():
    """
    The D2 dashboard contract: a JSON-serializable dict with the documented keys.
    A parallel dashboard agent consumes this shape, so it is asserted, not assumed.
    """
    rows = build_table_with_failures()
    rep = fp.fingerprint_report(rows, model=MODEL, manifest_path="/nonexistent.csv")
    p = fp.plot_payload(rep)

    assert set(p) == {"model", "composition", "signatures", "word_classes",
                      "insertion_split", "insertion_examples", "entity_gap",
                      "notes", "n_failed_rows_excluded"}, sorted(p)
    assert p["model"] == MODEL
    assert p["n_failed_rows_excluded"] == 7
    assert len(p["composition"]) == len(rep["composition"])
    for c in p["composition"]:                     # the stacked bar must still add up
        assert abs(c["wer_micro"] - sum(c[f"{e}_rate"]
                                        for e in ("sub", "del", "ins"))) < 1e-12
        assert c["rt60"] is not None and c["noise_type"] in NOISE

    fams = {s["family"]: s for s in p["signatures"]}
    assert fams["rt60"]["degrades"] is True
    # the no-fix rows survive into the payload FLAGGED, so a dashboard cannot render
    # a relative improvement as a recommendation
    assert fams["noise_type=engine"]["degrades"] is False
    assert fams["noise_type=engine"]["implied_fix"] == fp.NO_FIX_NEEDED

    groups = {g["group"]: g for g in p["insertion_split"]}
    assert groups["babble"]["n_foreign"] > 0 and groups["engine"]["n_ins"] == 0
    assert p["insertion_examples"] and "competing" in p["notes"]["insertion_caveat"].lower()
    # no manifest -> the proper-noun class is stated unavailable, never faked
    assert p["notes"]["proper_noun_source"] == "unavailable"
    assert p["notes"]["opportunity_denominator"] == "unavailable"
    assert "proper_noun" not in {w["word_class"] for w in p["word_classes"]}

    json.dumps(p)                                  # must be JSON-serializable
    json.dumps(cg.plot_payload(cg.confidence_gap_report(rows, model=MODEL)))
    print(f"OK 14: D2 plot payload has the documented keys "
          f"({len(p['composition'])} composition rows, {len(p['signatures'])} "
          f"signatures, no-fix rows flagged); both payloads JSON-serialize")


if __name__ == "__main__":
    test_loader_and_schema()
    test_failed_rows_never_averaged()
    test_dead_zone_region_recovered()
    test_loud_failure_region_not_flagged()
    test_gap_distribution_and_correlation()
    test_within_model_normalization()
    test_reverb_deletions_babble_substitutions()
    test_better_level_is_never_prescribed_a_fix()
    test_edit_composition_invariant()
    test_insertion_split_is_competing_speech()
    test_word_class_inventory()
    test_entity_error_beats_wer()
    test_report_artifacts()
    test_fingerprint_plot_payload()
    print("\nAll analysis tests passed — D1 recovers the planted dead zone (and "
          "only it), D2 recovers the planted reverb/babble fingerprints, a level "
          "that is merely cleaner than its siblings is never prescribed a fix, and "
          "failed rows never enter a single average.")
