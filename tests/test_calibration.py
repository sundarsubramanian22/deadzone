"""
Offline validation for Layer 2 (calibration.py). No audio, no API — a synthetic
population of word outcomes with a KNOWN planted miscalibration stands in for real
model confidences, and we assert the calibrators recover the correction.

Planted structure:
  * true correctness prob per word: true_p = clip(0.97 - 0.55*rt60n - 0.30*(1-snrn))
    (worse with reverb and noise),
  * OVERCONFIDENCE that GROWS with reverb: over = 0.30 * rt60n, so the reported
    confidence raw = clip(true_p + over) — the model is increasingly, and
    condition-dependently, overconfident,
  * binary outcomes y ~ Bernoulli(true_p) with a fixed seed.

Asserted:
  * the raw confidence is meaningfully miscalibrated (ECE well above 0),
  * temperature scaling (global) reduces ECE,
  * the feature-conditioned calibrator reduces ECE substantially MORE (it can see
    rt60), cutting it to a small fraction of the raw ECE,
  * it recovers the KNOWN correction: its per-word correction correlates with the
    planted overconfidence, and its calibrated confidence tracks the true prob far
    better than raw.
All fit on a train split, evaluated on a held-out test split. Deterministic.
Run:  python3 tests/test_calibration.py
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
import numpy as np
from scipy.stats import spearmanr

from deadzone.design import DEFAULT_FACTOR_SPACE
from deadzone.calibration import (
    TemperatureScaler, FeatureCalibrator, expected_calibration_error,
    reliability_curve, calibration_report,
)

SPACE = DEFAULT_FACTOR_SPACE
OVER_MAX = 0.30            # planted overconfidence at max reverb
N = 8000
NOISE = ("babble", "engine", "road")
CODEC = ("none", "g726", "opus-lowrate")


def make_population(seed=0):
    rng = np.random.default_rng(seed)
    rt60 = rng.uniform(0.2, 1.0, N)
    snr = rng.uniform(0.0, 25.0, N)
    rt60n = (rt60 - 0.2) / 0.8
    snrn = snr / 25.0

    true_p = np.clip(0.97 - 0.55 * rt60n - 0.30 * (1 - snrn), 0.02, 0.98)
    over = OVER_MAX * rt60n                                  # grows with reverb
    raw_conf = np.clip(true_p + over, 0.01, 0.999)
    y = (rng.random(N) < true_p).astype(int)                # observed correctness

    rows = [{"rt60": float(rt60[i]), "snr_db": float(snr[i]),
             "noise_type": NOISE[i % 3], "codec": CODEC[i % 3],
             "mic_rolloff": float(rng_val)}
            for i, rng_val in enumerate(rng.uniform(0, 1, N))]
    return rows, raw_conf, y, true_p, over


def split(n, frac=0.5, seed=1):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]


def _take(seq, idx):
    return [seq[i] for i in idx]


# ---- 1: ECE + reliability curve behave sanely ------------------------------

def test_metrics_sane():
    # a perfectly calibrated toy: conf == P(correct) exactly -> ECE ~ 0
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    ece = expected_calibration_error(p, y, n_bins=15)
    assert ece < 0.02, ece
    curve = reliability_curve(p, y, n_bins=15)
    assert len(curve) >= 10 and all(c["count"] > 0 for c in curve)
    # a constant-overconfident toy: conf=0.9 but only 60% correct -> ECE ~ 0.3
    ece_bad = expected_calibration_error(np.full(5000, 0.9),
                                         (rng.random(5000) < 0.6).astype(int))
    assert abs(ece_bad - 0.3) < 0.03, ece_bad
    print(f"OK 1: ECE ~0 when calibrated ({ece:.3f}); ~0.30 when 0.9-conf/0.6-acc "
          f"({ece_bad:.3f}); reliability curve well-formed")


# ---- 2: temperature scaling reduces ECE (global baseline) ------------------

def test_temperature_reduces_ece():
    rows, raw, y, true_p, over = make_population(seed=0)
    tr, te = split(N)
    ece_raw = expected_calibration_error(raw[te], y[te])
    assert ece_raw > 0.05, ("raw should be clearly miscalibrated", ece_raw)

    temp = TemperatureScaler().fit(raw[tr], y[tr])
    ece_temp = expected_calibration_error(temp.transform(raw[te]), y[te])
    assert temp.T > 1.0, ("overconfident model needs T>1 to soften", temp.T)
    assert ece_temp < ece_raw, (ece_temp, ece_raw)
    print(f"OK 2: temp scaling T={temp.T:.2f} cuts ECE {ece_raw:.3f} -> {ece_temp:.3f}")
    return ece_raw, ece_temp


# ---- 3: feature-conditioned calibrator does substantially better -----------

def test_feature_calibrator_beats_temp():
    rows, raw, y, true_p, over = make_population(seed=0)
    tr, te = split(N)
    ece_raw = expected_calibration_error(raw[te], y[te])
    ece_temp = expected_calibration_error(
        TemperatureScaler().fit(raw[tr], y[tr]).transform(raw[te]), y[te])

    fc = FeatureCalibrator(SPACE).fit(_take(rows, tr), raw[tr], y[tr])
    cal_te = fc.transform(_take(rows, te), raw[te])
    ece_feat = expected_calibration_error(cal_te, y[te])

    assert ece_feat < ece_temp, (ece_feat, ece_temp)
    assert ece_feat < 0.5 * ece_raw, ("should cut raw ECE by >half", ece_feat, ece_raw)
    print(f"OK 3: feature calibrator ECE {ece_feat:.3f} < temp {ece_temp:.3f} "
          f"< raw {ece_raw:.3f} (cut to {ece_feat/ece_raw:.0%} of raw)")


# ---- 4: recovers the KNOWN correction --------------------------------------

def test_recovers_planted_correction():
    rows, raw, y, true_p, over = make_population(seed=0)
    tr, te = split(N)
    fc = FeatureCalibrator(SPACE).fit(_take(rows, tr), raw[tr], y[tr])
    cal_te = fc.transform(_take(rows, te), raw[te])

    # the correction it applied should track the planted overconfidence over(rt60)
    correction = raw[te] - cal_te
    rho, _ = spearmanr(correction, over[te])
    assert rho > 0.8, ("correction should track planted overconfidence", rho)

    # and calibrated confidence should track the TRUE prob far better than raw
    rmse_raw = float(np.sqrt(np.mean((raw[te] - true_p[te]) ** 2)))
    rmse_cal = float(np.sqrt(np.mean((cal_te - true_p[te]) ** 2)))
    assert rmse_cal < 0.5 * rmse_raw, (rmse_cal, rmse_raw)
    print(f"OK 4: correction vs planted overconfidence rho={rho:.2f}; "
          f"RMSE-to-true_p {rmse_raw:.3f} (raw) -> {rmse_cal:.3f} (calibrated)")


# ---- 5: report payload has before/after ECE + reliability data -------------

def test_report_payload():
    rows, raw, y, true_p, over = make_population(seed=0)
    tr, te = split(N)
    fc = FeatureCalibrator(SPACE).fit(_take(rows, tr), raw[tr], y[tr])
    rep = calibration_report(raw[te], fc.transform(_take(rows, te), raw[te]), y[te])
    assert rep["ece_after"] < rep["ece_before"]
    assert rep["reliability_before"] and rep["reliability_after"]
    for c in rep["reliability_after"]:
        assert {"bin_lo", "bin_hi", "conf_mean", "accuracy", "count"} <= set(c)
    print(f"OK 5: report ECE {rep['ece_before']:.3f} -> {rep['ece_after']:.3f} "
          f"with reliability-diagram data for both")


# ===========================================================================
# analysis/calibration_report.py — the REAL-DATA L2 runner (A.R5.8)
# ===========================================================================
# Same discipline as everything else here: a synthetic master table with PLANTED
# structure stands in for the real grid, and we assert the runner recovers it —
# the alignment accounting, the deletion blind spot, the grouped split, and the
# plain-language statement. No CSV, no API.

import json as _json

from deadzone.analysis import calibration_report as CR

_L2_RT60 = (0.2, 0.45, 0.7, 1.0)
_L2_SNR = (0.0, 10.0, 20.0)
_L2_CLIPS = ("u01", "u02", "u03", "u04")
_N_REF = 10


def _row(clip, cond, rt60, snr, edits, confs, transcript=""):
    counts = {"match": 0, "sub": 0, "del": 0, "ins": 0}
    for e in edits:
        counts[e[0]] += 1
    return {"clip_id": clip, "condition_name": cond, "model": "nova-3",
            "rt60": rt60, "snr_db": snr, "noise_type": "babble", "codec": "none",
            "mic_rolloff": 0.5, "failed": False, "transcript": transcript,
            "wer": (counts["sub"] + counts["del"] + counts["ins"]) / _N_REF,
            "n_ref": _N_REF, "n_match": counts["match"], "n_sub": counts["sub"],
            "n_del": counts["del"], "n_ins": counts["ins"],
            "edits": _json.dumps(edits),
            "word_confidences": _json.dumps(confs)}


def make_master_rows(seed=0):
    """Planted: overconfidence grows with rt60; deletions grow with rt60; one
    condition is totally SILENT (empty transcript everywhere)."""
    rng = np.random.default_rng(seed)
    rows = []
    for rt60 in _L2_RT60:
        for snr in _L2_SNR:
            cond = f"rt60-{rt60}_snr-{snr:g}"
            rt60n = (rt60 - 0.2) / 0.8
            true_p = float(np.clip(0.95 - 0.45 * rt60n - 0.25 * (1 - snr / 25.0),
                                   0.05, 0.98))
            conf_val = float(np.clip(true_p + 0.30 * rt60n, 0.02, 0.999))
            n_del = int(round(3 * rt60n))
            for clip in _L2_CLIPS:
                edits = [["del", f"w{i}", None] for i in range(n_del)]
                confs = []
                for i in range(_N_REF - n_del):
                    ok = bool(rng.random() < true_p)
                    edits.append(["match" if ok else "sub", f"w{i}",
                                  f"w{i}" if ok else f"x{i}"])
                    confs.append(conf_val)
                rows.append(_row(clip, cond, rt60, snr, edits, confs,
                                 transcript=" ".join(e[2] for e in edits
                                                     if e[2] is not None)))
    # a SILENT condition: empty transcript on every clip -> 0 words, all deletions
    for clip in _L2_CLIPS:
        edits = [["del", f"w{i}", None] for i in range(_N_REF)]
        rows.append(_row(clip, "rt60-1_snr-0_dead", 1.0, 0.0, edits, [],
                         transcript=""))
    return rows


def make_misaligned_rows():
    """The two real normalization events that broke the 1:1 assumption on the
    grid: a MERGE (wi fi -> wifi, one confidence too many) and a SPLIT
    (follow-up -> follow up, one too few)."""
    merge_edits = [["match", "call", "call"], ["match", "the", "the"],
                   ["match", "wifi", "wifi"], ["sub", "desk", "disk"]]
    merge = _row("u09", "cond-merge", 0.7, 10.0, merge_edits,
                 [0.9, 0.8, 0.7, 0.6, 0.5], transcript="call the wi fi desk")
    split_edits = [["match", "send", "send"], ["match", "the", "the"],
                   ["match", "follow", "follow"], ["match", "up", "up"],
                   ["match", "note", "note"]]
    split = _row("u10", "cond-split", 0.7, 10.0, split_edits,
                 [0.9, 0.8, 0.7, 0.6], transcript="send the follow-up note")
    return [merge, split]


# ---- 6: the alignment fix is accounted for, cause by cause -----------------

def test_runner_alignment_recovery():
    base = make_master_rows()
    rows = base + make_misaligned_rows()

    kept, dropped = CR.legacy_split(rows)
    assert len(dropped) == 2, len(dropped)          # exactly the two planted rows
    assert {r["condition_name"] for r in dropped} == {"cond-merge", "cond-split"}

    rec = CR.alignment_recovery(rows)
    assert rec["n_rows_realigned"] == 2, rec
    assert rec["n_rows_still_misaligned"] == 0
    # merge row contributes 4 hyp words, split row 5 -> 9 recovered
    assert rec["n_words_recovered"] == 9, rec
    assert (rec["n_words_after_fix"] - rec["n_words_before_fix"]
            == rec["n_words_recovered"])
    causes = rec["causes"]
    assert any("wifi" in k for k in causes["merge_events"]), causes
    assert any("follow-up" in k for k in causes["split_events"]), causes
    print(f"OK 6: alignment recovery attributes both events — "
          f"{rec['n_rows_realigned']} rows / {rec['n_words_recovered']} words "
          f"recovered, {rec['n_rows_still_misaligned']} still misaligned")


# ---- 7: the deletion blind spot is measured, including silent cells --------

def test_runner_deletion_blindness():
    rows = make_master_rows()
    db = CR.deletion_blindness(rows)

    n_del = sum(r["n_del"] for r in rows)
    n_ref = sum(r["n_ref"] for r in rows)
    assert db["n_deletions"] == n_del and db["n_ref_words"] == n_ref
    assert abs(db["deleted_fraction_of_reference"] - n_del / n_ref) < 1e-12
    assert 0 < db["deleted_fraction_of_errors"] <= 1
    # emitted-word accuracy must OVERSTATE reference recovery whenever anything
    # was deleted — that gap is the whole point of the blind spot
    assert db["emitted_word_accuracy"] > db["reference_word_recovery"]
    assert abs(db["accuracy_overstatement"]
               - (db["emitted_word_accuracy"] - db["reference_word_recovery"])) < 1e-12
    # the planted all-deletion condition contributes zero words and drops out
    assert db["n_silent_conditions"] == 1, db["silent_conditions"]
    assert db["silent_conditions"][0]["condition_name"] == "rt60-1_snr-0_dead"
    assert db["silent_conditions"][0]["mean_wer"] == 1.0
    assert (db["n_conditions_in_calibration_set"]
            == db["n_conditions"] - db["n_silent_conditions"])
    print(f"OK 7: deletion blindness measured — {db['deleted_fraction_of_reference']:.1%} "
          f"of reference words / {db['deleted_fraction_of_errors']:.1%} of errors "
          f"invisible; emitted-word accuracy overstates recovery by "
          f"{db['accuracy_overstatement']:.3f}; {db['n_silent_conditions']} silent "
          "condition drops out of the split entirely")


# ---- 8: the end-to-end report — grouped split, seed band, statement --------

def test_runner_build_report():
    rows = make_master_rows() + make_misaligned_rows()
    rep = CR.build_report(model="nova-3", seeds=(0, 1, 2), rows=rows)

    # the split protocol is grouped and says so; build_report re-verifies it
    assert rep["protocol"]["split"] == CR.PRIMARY_SPLIT == "condition"
    assert "word" not in rep["protocol"]["split_modes_offered"]
    assert "random" not in rep["protocol"]["split_modes_offered"]

    p = rep["primary"]
    assert p["ece_raw"]["median"] > p["ece_temperature"]["median"]
    assert p["ece_feature"]["median"] < p["ece_temperature"]["median"]
    assert p["temperature_T"]["median"] > 1.0            # overconfident -> T>1
    for band in (p["ece_raw"], p["ece_feature"]):
        assert band["min"] <= band["median"] <= band["max"]
        assert len(band["per_seed"]) == 3
    # both grouped protocols run
    assert rep["secondary"]["split_by"] == "clip"

    # the alignment delta is reported with a verdict either way
    mv = rep["alignment_effect_on_ece"]
    assert mv["verdict"] in ("material", "negligible")
    assert mv["legacy"]["n_words"] < p["n_words"]

    # reliability-diagram payload survives into the JSON for both calibrators
    fit = p["headline_fit"]
    for arm in ("temperature", "feature"):
        r = fit[arm]["report"]
        assert r["reliability_before"] and r["reliability_after"]

    # the plain-language statement, in the A.R5.8 form, with real numbers
    st = rep["statement"]
    assert "held-out conditions" in st and "discounted by" in st
    assert "rt60 = 0.7" in st
    assert "no hypothesis token" in st and "empty transcript" in st
    assert "premise" in rep["premise"].lower()

    # and the whole thing is serializable — it is written to results/*.json
    txt = CR.format_report(rep)
    assert "PROTOCOL" in txt and "DELETION BLINDNESS" in txt
    _json.loads(_json.dumps(rep, default=float))
    print(f"OK 8: end-to-end report — ECE {p['ece_raw']['median']:.3f} -> "
          f"{p['ece_temperature']['median']:.3f} (T="
          f"{p['temperature_T']['median']:.2f}) -> "
          f"{p['ece_feature']['median']:.3f}; grouped split verified, "
          "statement + reliability payload + JSON round-trip all present")


# ---- 9: a leaked (non-grouped) split is caught by the report-time check ----

def test_runner_rejects_ungrouped_split():
    rows = make_master_rows()
    ok = [r for r in rows if r["condition_name"] != "rt60-1_snr-0_dead"]
    rep = CR.build_report(model="nova-3", seeds=(0,), rows=ok)
    fit = dict(rep["primary"]["headline_fit"])
    # pretend the split had produced different sizes than the grouped one does:
    # the re-derivation must notice rather than trust the reported numbers
    fit["n_test_words"] = fit["n_test_words"] + 1
    try:
        CR._assert_grouped(ok, fit, "condition")
    except AssertionError as e:
        assert "does not match" in str(e), str(e)
    else:
        raise AssertionError("a mismatched split must be caught, not trusted")
    print("OK 9: the grouped-split claim is re-derived at report time, not "
          "inherited on trust (leakage would show up as a BETTER ECE)")


def test_parallel_arrays_are_asserted_not_assumed():
    """
    Every function here takes (confidence, correctness) as two PARALLEL arrays
    the CALLER assembled by walking an alignment. Two ways that pairing breaks,
    both of which still produce a temperature and an ECE:

      * a LENGTH MISMATCH (an upstream filter applied to one list only, or a
        `zip` that truncated) — numpy broadcasts a length-1 array happily;
      * a NON-FINITE CONFIDENCE — `np.clip(nan, eps, 1-eps)` is still nan, the
        NLL is nan at every T, and `minimize_scalar` returns a bounded-search
        value WITHOUT raising, so the reported T is a property of the optimizer.
    """
    all_rows, raw_conf, y_all, _true_p, _over = make_population(seed=0)
    conf = np.asarray(raw_conf, dtype=float)[:200]
    y = np.asarray(y_all, dtype=float)[:200]
    rows = [dict(r) for r in all_rows[:200]]

    # the un-guarded arithmetic really does return a T without complaining
    nan_conf = conf.copy()
    nan_conf[7] = np.nan
    for label, call in (
            ("TemperatureScaler.fit length",
             lambda: TemperatureScaler().fit(conf[:100], y)),
            ("TemperatureScaler.fit NaN",
             lambda: TemperatureScaler().fit(nan_conf, y)),
            ("reliability_curve length",
             lambda: reliability_curve(conf, y[:100])),
            ("expected_calibration_error NaN",
             lambda: expected_calibration_error(nan_conf, y)),
            ("FeatureCalibrator.fit rows/conf",
             lambda: FeatureCalibrator(SPACE).fit(rows[:100], conf, y)),
            ("calibration_report before/after",
             lambda: calibration_report(conf, conf[:100], y)),
            ("empty input",
             lambda: TemperatureScaler().fit([], []))):
        try:
            call()
            raise AssertionError(f"{label}: must raise, not fit silently")
        except ValueError as exc:
            assert "confidence" in str(exc).lower() or "words" in str(exc).lower(), exc

    # negative control: the well-formed triple still fits and scores
    t = TemperatureScaler().fit(conf, y)
    assert np.isfinite(t.T) and t.T > 0
    FeatureCalibrator(SPACE).fit(rows, conf, y)
    assert np.isfinite(expected_calibration_error(conf, y))
    print("OK 10: mismatched-length and non-finite (confidence, label) inputs "
          f"raise instead of silently fitting; the clean triple still gives "
          f"T={t.T:.2f}")


def test_seed_band_refuses_a_fake_band():
    """
    The BAND is the claim. A repeated seed re-runs the IDENTICAL grouped split,
    so `seeds=(0, 0, 0)` yields three identical fits and a zero-width band still
    labelled "median [min, max] over 3 grouped splits" — one draw wearing the
    strongest possible form of the number.
    """
    rows, _failed = CR.usable_rows(
        [r for r in make_master_rows() if str(r.get("model")) == "nova-3"])
    for label, seeds in (("duplicate", (0, 0, 0)), ("empty", ())):
        try:
            CR.seed_band(rows, SPACE, CR.PRIMARY_SPLIT, 0.5, seeds)
            raise AssertionError(f"{label} seeds must raise")
        except ValueError as exc:
            assert "seed" in str(exc).lower(), exc
    band = CR.seed_band(rows, SPACE, CR.PRIMARY_SPLIT, 0.5, (0, 1, 2))
    assert len(band["ece_raw"]["per_seed"]) == 3
    print("OK 11: duplicate/empty grouped-split seeds raise; three distinct "
          "seeds still produce a band")


def test_the_statement_names_the_arm_it_was_fit_on():
    """
    L2 is single-model BY DESIGN — confidences are never pooled across arms — but
    `--model` is a parameter, and the plain-language statement hardcoded the
    spine's name. Run it on a third arm and it produced a fluent, numerate
    sentence attributing that arm's ECE to nova-3, with nothing in the output to
    contradict it. Purely a label, and purely the dangerous class: plausible.
    """
    third = "elevenlabs-scribe"
    rows = make_master_rows()
    relabelled = [dict(r, model=third) for r in rows]

    rep = CR.build_report(model=third, seeds=(0, 1), rows=relabelled)
    assert rep["model"] == third, rep["model"]
    assert f"{third}'s raw word confidence" in rep["statement"], rep["statement"][:200]
    assert "nova-3" not in rep["statement"], rep["statement"][:200]
    assert "nova-3" not in CR.format_report(rep)

    # NEGATIVE CONTROL: the same call on the spine still says nova-3, so the
    # assertion above is about the PARAMETER and not about a string that was
    # simply deleted.
    spine = CR.build_report(model="nova-3", seeds=(0, 1), rows=rows)
    assert "nova-3's raw word confidence" in spine["statement"], spine["statement"][:200]
    assert third not in spine["statement"]
    print(f"ok: the L2 statement names the arm it was fit on ({third} vs nova-3), "
          "never a hardcoded one")


if __name__ == "__main__":
    test_the_statement_names_the_arm_it_was_fit_on()
    test_metrics_sane()
    test_temperature_reduces_ece()
    test_feature_calibrator_beats_temp()
    test_recovers_planted_correction()
    test_report_payload()
    test_runner_alignment_recovery()
    test_runner_deletion_blindness()
    test_runner_build_report()
    test_runner_rejects_ungrouped_split()
    test_parallel_arrays_are_asserted_not_assumed()
    test_seed_band_refuses_a_fake_band()
    print("\nAll calibration tests passed — feature-conditioned calibration "
          "recovers the planted condition-dependent overconfidence, ECE and all; "
          "the real-data runner accounts for the alignment fix, measures the "
          "deletion blind spot, and refuses an ungrouped split.")
