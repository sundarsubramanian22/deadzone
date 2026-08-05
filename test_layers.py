"""
Offline validation for analysis/layers.py — the real-data runners for L1/L2/L3
(SPEC A.R5.7 / A.R5.8 / A.R5.9). No API, no real recordings, no network: every
input is a synthetic master-table row or a tiny wav synthesized into a temp dir,
with KNOWN structure planted in it, and each test asserts the runner recovers
that structure *and* refuses the trap that would have hidden it.

WHAT IS PLANTED, LAYER BY LAYER

L1 (multi-model comparison)
  * two arms over the same rt60 x snr grid; "whisper-base" gets a large WER bump
    for rt60 > 0.7 while staying at the top of ITS OWN confidence range (a dead
    zone), "nova-3" has no bump and its confidence tracks its errors;
  * the two arms live on DISJOINT raw confidence scales (whisper 0.30-0.55,
    nova 0.85-0.99) — a raw comparison would call whisper permanently
    unconfident and MISS its dead zone entirely;
  * a handful of whisper rows carry a fluent hallucination (n_ins >> n_ref);
  * asserted: the rt60 > 0.7 divergence region is named with whisper as the
    worse arm; the verdict is INVARIANT under a strictly-increasing rescaling
    of whisper's confidences (the operational proof that raw cross-family
    confidences are never compared); hallucinations are surfaced separately;
    failed rows are excluded and counted; a CSV round trip changes nothing.

L2 (learned confidence calibration)
  * per-word correctness drawn from a true probability that falls with reverb
    and noise, with OVERCONFIDENCE THAT GROWS WITH REVERB (over = 0.30*rt60n) —
    exactly the miscalibration a single temperature cannot fix;
  * deletions planted at a rate that grows with reverb (they carry no
    confidence and must stay invisible), plus confidences of exactly 1.0;
  * asserted: the word-label recipe is followed to the letter; the
    feature-conditioned calibrator cuts ECE substantially more than temperature
    scaling alone (the entire point of conditioning on acoustics); a random
    word-level split is not offered at all; misaligned confidences RAISE
    instead of being silently truncated.

L3 (paralinguistic decoupling)
  * a clean tone plus an 8-rung noise sweep on disk; spectral flatness drifts on
    a known monotone schedule while the lexical (WER) curve is planted on a
    DIFFERENT schedule — late-breaking (feature leads), early-breaking (lexical
    leads), and identical (coupled);
  * asserted: decoupling is detected, the correct curve is named as leading,
    the coupled case is reported as coupled, and using a sweep clip as the
    "clean" baseline raises instead of quietly shrinking the drift curve.

Deterministic (fixed seeds). Run:  python3 test_layers.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.layers import (
    SPLIT_MODES, AlignmentError, SweepPoint, aggregate_by_condition,
    find_hallucinations, grouped_split, load_master_csv, run_l1_model_comparison,
    run_l2_calibration, run_l3_decoupling, summarize_l3, sweep_from_dir,
    usable_rows, word_records,
)
from design import DEFAULT_FACTOR_SPACE

SPACE = DEFAULT_FACTOR_SPACE
FS = 16000


# ===========================================================================
# L1 fixtures — two arms, one planted weak region, disjoint confidence scales
# ===========================================================================

RT60 = np.linspace(0.2, 1.0, 9)
SNR = np.linspace(0.0, 25.0, 5)
N_CLIPS = 2


def _base_wer(rt60: float, snr: float) -> float:
    rn, qn = (rt60 - 0.2) / 0.8, snr / 25.0
    return 0.05 + 0.20 * rn + 0.30 * (1 - qn)


def _l1_row(clip_id, cond, rt60, snr, model, wer, conf, n_ins=1, n_ref=8):
    wer = float(min(max(wer, 0.0), 1.0))
    n_sub = int(round(wer * n_ref))
    return {
        "clip_id": clip_id, "condition_name": cond,
        "rt60": float(rt60), "snr_db": float(snr), "noise_type": "babble",
        "codec": "none", "mic_rolloff": 0.0, "model": model,
        "transcript": "call maria at four zero five",
        "wer": wer, "n_ref": n_ref, "n_sub": n_sub, "n_del": 0,
        "n_ins": int(n_ins), "n_match": max(n_ref - n_sub, 0),
        "mean_conf": float(conf), "utterance_conf": float(conf),
        "word_confidences": json.dumps([float(conf)] * n_ref),
        "edits": json.dumps([]),
        "failed": False, "error": None,
    }


def build_l1_table(conf_map=None, hallucinate=True):
    """One master table holding BOTH arms (as run_experiment writes it)."""
    conf_map = conf_map or (lambda m, c: c)
    rows = []
    for rt60 in RT60:
        for snr in SNR:
            cond = f"rt{rt60:.2f}_snr{snr:.0f}"
            base = _base_wer(rt60, snr)
            qn = snr / 25.0
            rn = (rt60 - 0.2) / 0.8
            # nova-3: no bump; high-scale confidence that tracks cleanliness
            nova_conf = min(max(0.90 + 0.09 * qn - 0.05 * rn, 0.0), 1.0)
            # whisper: big WER bump past rt60=0.7 but sits at the TOP of its own
            # (low) confidence range there -> confidently wrong.
            bump = 0.5 if rt60 > 0.7 else 0.0
            ws_conf = 0.55 if rt60 > 0.7 else (0.30 + 0.10 * qn)
            for k in range(N_CLIPS):
                clip = f"u{k+1:02d}"
                rows.append(_l1_row(clip, cond, rt60, snr, "nova-3",
                                    base, conf_map("nova-3", nova_conf)))
                # the hallucination: fluent invented text at the harsh corner
                # (>0.95, not >0.9: linspace lands on 0.9000000000000001)
                halluc = hallucinate and rt60 > 0.95 and snr < 1.0
                rows.append(_l1_row(
                    clip, cond, rt60, snr, "whisper-base", base + bump,
                    conf_map("whisper-base", ws_conf),
                    n_ins=40 if halluc else 1))
                if halluc:
                    rows[-1]["transcript"] = (
                        "thank you for watching and please subscribe to the "
                        "channel for more videos like this one thank you")
    return rows


# ---- 1: L1 names the planted divergence region -----------------------------

def test_l1_names_divergence_region():
    rows = build_l1_table()
    rep = run_l1_model_comparison(rows, SPACE, n_bins=4, wer_hi=0.3,
                                  conf_pct_hi=0.6)

    # aggregation collapsed the clip set: 45 conditions per arm, 2 clips each
    assert rep["aggregate"] == "condition"
    for m in ("nova-3", "whisper-base"):
        assert rep["per_model"][m]["n"] == len(RT60) * len(SNR), rep["per_model"][m]

    top = rep["divergence_regions"][0]
    assert top["factor"] == "rt60", top
    assert top["worse_model"] == "whisper-base", top
    assert top["better_model"] == "nova-3", top
    lo, hi = top["span"]
    assert lo >= 0.7, ("divergence should sit at the high-reverb end", top["span"])
    assert top["wer_gap"] > 0.3, top["wer_gap"]

    dz = top["dead_zone_rate_by_model"]
    assert dz["whisper-base"] > 0.8 and dz["nova-3"] == 0.0, dz

    # the named dead zones carry their exact factor values (the write-up table)
    zones = rep["per_model"]["whisper-base"]["dead_zone_conditions"]
    assert zones and all(z["rt60"] > 0.7 for z in zones), zones[:3]
    assert all(z["conf_percentile"] >= 0.6 for z in zones), zones[:3]
    assert rep["per_model"]["nova-3"]["dead_zone_conditions"] == []

    # and the confidence-vs-WER shape separates a self-aware arm from a blind one
    s_nova = rep["per_model"]["nova-3"]["shape"]["spearman"]
    s_ws = rep["per_model"]["whisper-base"]["shape"]["spearman"]
    assert s_nova < 0 < s_ws, (s_nova, s_ws)

    print(f"OK 1: L1 names {top['factor']} {top['span']} — worse=whisper-base "
          f"(WER gap {top['wer_gap']:.2f}), dead-zone rate "
          f"{dz['whisper-base']:.2f} vs {dz['nova-3']:.2f}; shape rho "
          f"nova={s_nova:.2f} whisper={s_ws:.2f}")
    return rep


# ---- 2: raw cross-family confidences are NEVER compared --------------------

def test_l1_raw_confidences_never_compared():
    rows = build_l1_table()
    rep = run_l1_model_comparison(rows, SPACE)

    scales = rep["raw_confidence_scales"]["per_model"]
    # the planted scales genuinely do not overlap: a raw comparison is invalid
    assert scales["whisper-base"]["max"] < scales["nova-3"]["min"], scales
    assert rep["raw_confidence_scales"]["scales_overlap"] is False
    assert rep["confidence_comparable_across_models"] is False
    # ...yet whisper's dead zone IS found, at its own top percentile, despite
    # every one of its confidences being below every one of nova's.
    assert rep["per_model"]["whisper-base"]["dead_zone_rate"] > 0.0

    # THE OPERATIONAL PROOF: rescale one arm's confidence by a strictly
    # increasing map. Any raw comparison anywhere would move; a within-model
    # percentile cannot. Every cross-model number must be bit-identical.
    rescale = lambda m, c: (0.001 + 0.4 * c ** 3) if m == "whisper-base" else c  # noqa: E731
    rep2 = run_l1_model_comparison(build_l1_table(conf_map=rescale), SPACE)

    assert (rep2["raw_confidence_scales"]["per_model"]["whisper-base"]["max"]
            < scales["whisper-base"]["max"]), "the rescale must actually move raw conf"
    for m in ("nova-3", "whisper-base"):
        assert (rep2["per_model"][m]["dead_zone_rate"]
                == rep["per_model"][m]["dead_zone_rate"]), m
        assert (rep2["per_model"][m]["shape"]["spearman"]
                == rep["per_model"][m]["shape"]["spearman"]), m
    a, b = rep["divergence_regions"][0], rep2["divergence_regions"][0]
    assert (a["factor"], a["span"], a["worse_model"]) == \
           (b["factor"], b["span"], b["worse_model"]), (a, b)
    assert a["dead_zone_rate_by_model"] == b["dead_zone_rate_by_model"]

    print("OK 2: whisper raw conf "
          f"[{scales['whisper-base']['min']:.2f},{scales['whisper-base']['max']:.2f}] "
          f"is entirely below nova's "
          f"[{scales['nova-3']['min']:.2f},{scales['nova-3']['max']:.2f}] "
          "(no overlap), yet the dead zone is still found — and every "
          "cross-model number is invariant under a monotone rescale")


# ---- 3: Whisper's fluent hallucination is surfaced on its own ---------------

def test_l1_hallucination_surfaced():
    rows = build_l1_table()
    rep = run_l1_model_comparison(rows, SPACE, ins_ratio=1.5, min_ins=5)

    ws = rep["per_model"]["whisper-base"]["hallucination"]
    nova = rep["per_model"]["nova-3"]["hallucination"]
    assert ws["n_flagged"] == N_CLIPS, ws          # rt60=1.0, snr=0 only
    assert nova["n_flagged"] == 0, nova
    ex = ws["examples"][0]
    assert ex["n_ins"] == 40 and ex["n_ref"] == 8, ex
    assert ex["ins_over_ref"] >= 1.5 and "subscribe" in ex["transcript"]
    assert ex["rt60"] == 1.0 and ex["snr_db"] == 0.0, ex

    # it is reported SEPARATELY from the dead-zone listing, not merged into it
    zone_keys = set(rep["per_model"]["whisper-base"]["dead_zone_conditions"][0])
    assert "n_ins" not in zone_keys and "transcript" not in zone_keys
    assert "hallucinat" in rep["statement"].lower(), rep["statement"]

    # the threshold is a real discriminator, not a tautology
    ordinary = find_hallucinations(
        [{"n_ins": 3, "n_ref": 8, "clip_id": "u01"}], ins_ratio=1.5, min_ins=5)
    assert ordinary["n_flagged"] == 0, ordinary

    print(f"OK 3: {ws['n_flagged']} fluent-hallucination rows flagged for "
          f"whisper-base (n_ins={ex['n_ins']} vs n_ref={ex['n_ref']}, "
          f"{ex['ins_over_ref']:.1f}x), 0 for nova-3, reported separately from "
          "the dead zones")


# ---- 4: failed rows excluded + a CSV round trip changes nothing -------------

def test_l1_failures_and_csv_round_trip():
    rows = build_l1_table()
    failed = dict(rows[1])
    failed.update({"failed": True, "error": "APIError: 503", "transcript": None,
                   "wer": None, "mean_conf": None, "n_ref": None, "n_ins": None})
    rows_with_failure = rows + [failed]

    ok, bad = usable_rows(rows_with_failure)
    assert len(bad) == 1 and len(ok) == len(rows)

    rep = run_l1_model_comparison(rows_with_failure, SPACE)
    assert rep["per_model"]["whisper-base"]["n_failed_rows"] == 1
    assert rep["per_model"]["whisper-base"]["failure_rate"] > 0
    assert rep["per_model"]["nova-3"]["n_failed_rows"] == 0
    # the failure must not have manufactured a dead zone or moved the region
    base = run_l1_model_comparison(rows, SPACE)
    assert (rep["divergence_regions"][0]["wer_gap"]
            == base["divergence_regions"][0]["wer_gap"])

    # everything survives the string-typed CSV round trip the real table takes
    with tempfile.TemporaryDirectory() as td:
        import csv
        path = os.path.join(td, "master.csv")
        cols = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows_with_failure:
                w.writerow({c: r.get(c) for c in cols})
        rep_csv = run_l1_model_comparison(load_master_csv(path), SPACE)

    a, b = base["divergence_regions"][0], rep_csv["divergence_regions"][0]
    assert (a["factor"], a["worse_model"]) == (b["factor"], b["worse_model"])
    assert abs(a["wer_gap"] - b["wer_gap"]) < 1e-9, (a["wer_gap"], b["wer_gap"])
    print(f"OK 4: failed row split out and counted (failure_rate "
          f"{rep['per_model']['whisper-base']['failure_rate']:.3f}), verdict "
          "unchanged; identical verdict after a CSV round trip")


# ===========================================================================
# L2 fixtures — overconfidence that GROWS with reverb, plus deletions
# ===========================================================================

L2_RT60 = np.linspace(0.2, 1.0, 8)
L2_SNR = np.linspace(0.0, 25.0, 5)
L2_CLIPS = [f"u{i:02d}" for i in range(1, 11)]
WORDS_PER_ROW = 12
OVER_MAX = 0.30


def build_l2_table(seed: int = 0):
    """Master-table rows carrying real `edits` + `word_confidences` payloads."""
    rng = np.random.default_rng(seed)
    rows = []
    for rt60 in L2_RT60:
        rn = (rt60 - 0.2) / 0.8
        for snr in L2_SNR:
            qn = snr / 25.0
            cond = f"rt{rt60:.2f}_snr{snr:.0f}"
            true_p = float(np.clip(0.97 - 0.55 * rn - 0.30 * (1 - qn), 0.02, 0.98))
            over = OVER_MAX * rn                       # grows with reverb
            for clip in L2_CLIPS:
                y = (rng.random(WORDS_PER_ROW) < true_p).astype(int)
                conf = np.clip(true_p + over, 0.01, 1.0) + rng.normal(0, 0.01,
                                                                     WORDS_PER_ROW)
                conf = np.clip(conf, 0.01, 1.0)
                if true_p > 0.9:
                    conf[::4] = 1.0        # vendors DO return exactly 1.0 on the
                    #                        easy words — the _logit edge case
                # hypothesis words, in order: match or sub
                edits = [("match" if y[i] else "sub", f"r{i}",
                          f"r{i}" if y[i] else f"x{i}")
                         for i in range(WORDS_PER_ROW)]
                # deletions grow with reverb; they carry NO hypothesis word and
                # NO confidence, and are interleaved to exercise the filter
                n_del = int(round(3 * rn))
                for j in range(n_del):
                    edits.insert(min(2 + 3 * j, len(edits)), ("del", f"d{j}", None))
                n_sub = int(WORDS_PER_ROW - y.sum())
                n_ref = WORDS_PER_ROW + n_del
                rows.append({
                    "clip_id": clip, "condition_name": cond,
                    "rt60": float(rt60), "snr_db": float(snr),
                    "noise_type": "babble", "codec": "none",
                    "mic_rolloff": float(rng.uniform(0, 1)), "model": "nova-3",
                    "transcript": " ".join(e[2] for e in edits if e[2] is not None),
                    "wer": (n_sub + n_del) / n_ref, "n_ref": n_ref,
                    "n_sub": n_sub, "n_del": n_del, "n_ins": 0,
                    "n_match": int(y.sum()),
                    "mean_conf": float(conf.mean()),
                    "utterance_conf": float(conf.mean()),
                    "word_confidences": json.dumps([float(c) for c in conf]),
                    "edits": json.dumps([list(e) for e in edits]),
                    "failed": False, "error": None,
                })
    return rows


# ---- 5: the word-label recipe, to the letter --------------------------------

def test_l2_word_label_recipe():
    edits = [["match", "call", "call"],
             ["del", "maria", None],                 # no hyp word -> no confidence
             ["sub", "at", "add"],
             ["ins", None, "please"],
             ["match", "four", "four"]]
    row = {"clip_id": "u02", "condition_name": "c1", "model": "nova-3",
           "rt60": 0.6, "snr_db": 10.0, "noise_type": "babble", "codec": "none",
           "mic_rolloff": 0.2, "n_ref": 4, "n_del": 1, "failed": False,
           "edits": json.dumps(edits),
           "word_confidences": json.dumps([0.99, 0.61, 0.55, 1.0])}

    w = word_records([row])
    # the hyp_word-not-None filter IS the hypothesis, in order, 1:1 with conf
    assert w["n_words"] == 4, w["n_words"]
    assert [r["hyp_word"] for r in w["records"]] == ["call", "add", "please", "four"]
    assert [r["op"] for r in w["records"]] == ["match", "sub", "ins", "match"]
    assert list(w["correct"]) == [1, 0, 0, 1], w["correct"]
    assert "del" not in {r["op"] for r in w["records"]}
    # deletions counted but unrepresented; conf exactly 1.0 clipped off the edge
    assert w["n_deletions"] == 1 and w["n_conf_clipped"] == 1
    assert w["conf"].max() < 1.0 and np.all(np.isfinite(w["conf"]))
    # the factor coordinates ride along for the feature-conditioned calibrator
    assert w["records"][0]["rt60"] == 0.6 and w["records"][0]["snr_db"] == 10.0
    print("OK 5: word labels follow the recipe — hyp_word-not-None filter aligns "
          "1:1 with word_confidences, match=1 / sub=ins=0, deletions counted but "
          "invisible, conf 1.0 clipped into (0,1)")


# ---- 6: misaligned confidences RAISE, never truncate ------------------------

def test_l2_misalignment_raises():
    edits = [["match", "a", "a"], ["match", "b", "b"], ["match", "c", "c"]]
    bad = {"clip_id": "u07", "condition_name": "c9", "model": "nova-3",
           "rt60": 0.4, "snr_db": 5.0, "noise_type": "babble", "codec": "none",
           "mic_rolloff": 0.0, "n_ref": 3, "failed": False,
           "edits": json.dumps(edits),
           # one confidence too few — a zip() would silently drop "c" and bind
           # the remaining confidences to the wrong words
           "word_confidences": json.dumps([0.9, 0.8])}

    try:
        word_records([bad])
    except AlignmentError as e:
        assert "3 hypothesis words but 2 word confidences" in str(e), str(e)
    else:
        raise AssertionError("misaligned confidences must raise, not truncate")

    # the escape hatch drops WHOLE rows and counts them — still never truncates
    good = dict(bad, word_confidences=json.dumps([0.9, 0.8, 0.7]),
                clip_id="u08", condition_name="c10")
    w = word_records([bad, good], on_misalign="skip")
    assert w["n_words"] == 3 and w["n_rows_used"] == 1
    assert w["n_misaligned_rows"] == 1
    assert w["misaligned"][0]["n_hyp_words"] == 3
    assert w["misaligned"][0]["n_confidences"] == 2
    print("OK 6: misaligned row raises AlignmentError; on_misalign='skip' drops "
          "the whole row and reports it (never a silent zip() truncation)")


# ---- 7: a random word-level split is not on the menu ------------------------

def test_l2_no_word_level_split():
    assert set(SPLIT_MODES) == {"condition", "clip"}, SPLIT_MODES
    assert "word" not in SPLIT_MODES and "random" not in SPLIT_MODES

    rows = build_l2_table()
    for mode in ("word", "random", "none"):
        try:
            run_l2_calibration(rows, SPACE, split_by=mode)
        except ValueError as e:
            assert "random/word-level split is NOT offered" in str(e), str(e)
        else:
            raise AssertionError(f"split_by={mode!r} must raise")

    # and the grouped split really does keep a group whole
    w = word_records(rows)
    tr, te = grouped_split(w["records"], split_by="condition", frac=0.5, seed=0)
    g = lambda idx: {w["records"][i]["condition_name"] for i in idx}   # noqa: E731
    assert not (g(tr) & g(te)), "a condition leaked across the split"
    assert len(g(tr)) + len(g(te)) == len(L2_RT60) * len(L2_SNR)
    # clip-level grouping is the other offered mode, and it is also disjoint
    tr2, te2 = grouped_split(w["records"], split_by="clip", seed=0)
    gc = lambda idx: {w["records"][i]["clip_id"] for i in idx}          # noqa: E731
    assert not (gc(tr2) & gc(te2))
    print(f"OK 7: SPLIT_MODES = {SPLIT_MODES}; word/random splits raise; "
          f"grouped split keeps all {len(g(tr))}+{len(g(te))} conditions whole")


# ---- 8: conditioning on acoustics is what fixes the miscalibration ---------

def test_l2_feature_calibrator_beats_temperature():
    rows = build_l2_table()
    rep = run_l2_calibration(rows, SPACE, split_by="condition", frac=0.5, seed=0)

    ece_raw = rep["ece_raw"]
    ece_temp = rep["temperature"]["ece"]
    ece_feat = rep["feature"]["ece"]

    assert ece_raw > 0.05, ("raw confidence must be clearly miscalibrated", ece_raw)
    assert rep["temperature"]["T"] > 1.0, ("overconfident -> T>1",
                                           rep["temperature"]["T"])
    assert ece_temp < ece_raw, (ece_temp, ece_raw)
    assert rep["n_conf_clipped"] > 0, "the exactly-1.0 confidences must be clipped"
    # THE POINT OF L2: a single global temperature cannot undo miscalibration
    # that VARIES with the condition; conditioning on rt60/snr can.
    assert ece_feat < 0.5 * ece_temp, ("feature calibrator must beat temperature "
                                       "substantially", ece_feat, ece_temp)
    assert ece_feat < 0.25 * ece_raw, (ece_feat, ece_raw)

    # the reliability-diagram payload is there for both calibrators
    for arm in ("temperature", "feature"):
        r = rep[arm]["report"]
        assert r["reliability_before"] and r["reliability_after"]
        for c in r["reliability_after"]:
            assert {"bin_lo", "bin_hi", "conf_mean", "accuracy", "count"} <= set(c)

    # what it learned, in plain language: a real discount above rt60 = 0.7
    d = rep["discount"]["rt60"]
    assert d["mean_discount"] > 0.05, d
    assert d["mean_raw_conf"] > d["empirical_accuracy"], d
    assert "discounted by" in rep["statement"], rep["statement"]

    # the blind spot is measured and stated, not implied away
    db = rep["deletion_blindness"]
    assert db["n_deletions"] > 0 and 0 < db["deleted_fraction_of_reference"] < 1
    assert "no confidence" in db["note"]

    print(f"OK 8: ECE raw {ece_raw:.3f} -> temp {ece_temp:.3f} (T="
          f"{rep['temperature']['T']:.2f}) -> feature {ece_feat:.3f} "
          f"({ece_feat/ece_raw:.0%} of raw, {ece_feat/ece_temp:.0%} of temp); "
          f"discount above rt60=0.7 is {d['mean_discount']:.2f}; "
          f"{db['deleted_fraction_of_reference']:.1%} of reference words are "
          "deletion-blind")
    return rep


# ---- 8b: cross-model calibration is refused ---------------------------------

def test_l2_refuses_mixed_models():
    rows = build_l2_table()[:40]
    mixed = rows + [dict(rows[0], model="whisper-base")]
    try:
        run_l2_calibration(mixed, SPACE)
    except ValueError as e:
        assert "ONE model's rows" in str(e), str(e)
    else:
        raise AssertionError("a calibrator fitted across model families must raise")
    print("OK 8b: fitting one calibrator across two model families is refused "
          "(it would learn the scale difference, not the miscalibration)")


# ===========================================================================
# L3 fixtures — a real sweep of tiny wavs on disk
# ===========================================================================

def _tone(freq=200.0, dur=0.6, amp=0.6):
    t = np.arange(int(dur * FS)) / FS
    return amp * np.sin(2 * np.pi * freq * t)


def _write_sweep(root: str, factor: str = "rt60"):
    """Clean RAW recording in one dir; the 8-rung degraded sweep in another.

    The two directories are separate ON PURPOSE — that is the physical layout
    the layer requires (raw capture vs composed conditions), and the guard
    enforces it.
    """
    raw_dir = os.path.join(root, "recordings")
    sweep_dir = os.path.join(root, "sweep")
    os.makedirs(raw_dir), os.makedirs(sweep_dir)

    clean = _tone()
    clean_path = os.path.join(raw_dir, "u02.wav")
    sf.write(clean_path, clean, FS)

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(clean))
    levels = np.round(np.linspace(0.2, 0.9, 8), 2)
    amps = np.linspace(0.0, 0.5, 8)            # the KNOWN degradation schedule
    paths = []
    for lv, a in zip(levels, amps):
        p = os.path.join(sweep_dir, f"{factor}_{lv:.2f}.wav")
        sf.write(p, clean + a * noise, FS)
        paths.append(p)
    return clean_path, sweep_dir, list(levels), paths


# ---- 9: decoupling detected and the leading curve named --------------------

def test_l3_decoupling_and_leader():
    with tempfile.TemporaryDirectory() as td:
        clean_path, sweep_dir, levels, _ = _write_sweep(td)

        # (a) LEXICAL LATE: WER only breaks at the top of the sweep, while
        #     flatness has been drifting since the first rung -> feature leads.
        ln = (np.asarray(levels) - levels[0]) / (levels[-1] - levels[0])
        late = ln ** 4
        sweep = sweep_from_dir(sweep_dir, "rt60", late)
        a = run_l3_decoupling(clean_path, sweep, "rt60",
                              feature_keys=("flatness",))
        assert a["verdict"] == "DECOUPLED", a["headline"]
        assert a["headline"]["leads"] == "feature", a["headline"]
        assert (a["headline"]["feature_half_level"]
                < a["headline"]["lexical_half_level"]), a["headline"]
        assert "monitor would alarm before" in a["statement"], a["statement"]

        # (b) LEXICAL EARLY: WER halves almost immediately while the feature
        #     holds -> lexical leads, i.e. the agent is flying blind. This is
        #     the A.R5.9 DoD sentence.
        early = [0.0, 0.45, 0.55, 0.60, 0.62, 0.63, 0.64, 0.65]
        sweep_b = sweep_from_dir(sweep_dir, "rt60", early)
        b = run_l3_decoupling(clean_path, sweep_b, "rt60",
                              feature_keys=("flatness",))
        assert b["verdict"] == "DECOUPLED", b["headline"]
        assert b["headline"]["leads"] == "lexical", b["headline"]
        assert (b["headline"]["lexical_half_level"]
                < b["headline"]["feature_half_level"]), b["headline"]
        assert "would not notice its ASR had already failed" in b["statement"]

        # the drift curve really is the planted monotone schedule
        drift = np.asarray(b["headline"]["drift_curve"])
        assert drift[0] == 0.0 and np.all(np.diff(drift) > 0), drift

        # a full-feature run reports every requested feature with a verdict
        full = run_l3_decoupling(clean_path, sweep_b, "rt60")
        assert set(full["per_feature"]) >= {"f0", "rms", "flatness", "centroid"}
        assert full["n_features"] == len(full["per_feature"])
        assert full["primary_feature"] in full["per_feature"]
        assert "decoupled from WER" in summarize_l3(full)

    print(f"OK 9: late-breaking WER -> feature leads (half "
          f"{a['headline']['feature_half_level']:.2f} vs "
          f"{a['headline']['lexical_half_level']:.2f}); early-breaking WER -> "
          f"lexical leads (half {b['headline']['lexical_half_level']:.2f} vs "
          f"{b['headline']['feature_half_level']:.2f}) — "
          f"\"{b['statement']}\"")


# ---- 10: the coupled case is reported as coupled ---------------------------

def test_l3_coupled_case():
    with tempfile.TemporaryDirectory() as td:
        clean_path, sweep_dir, levels, _ = _write_sweep(td)
        # first measure the feature's own drift (the WER curve here is a
        # placeholder), then plant a lexical curve on the SAME schedule: the
        # honest "they degrade together" result.
        probe = run_l3_decoupling(clean_path,
                                  sweep_from_dir(sweep_dir, "rt60",
                                                 list(np.linspace(0, 1, 8))),
                                  "rt60", feature_keys=("flatness",))
        drift = probe["per_feature"]["flatness"]["drift_curve"]

        coupled = run_l3_decoupling(clean_path,
                                    sweep_from_dir(sweep_dir, "rt60", drift),
                                    "rt60", feature_keys=("flatness",))
    h = coupled["per_feature"]["flatness"]
    assert h["coupled"] is True, h
    assert coupled["verdict"] == "COUPLED", coupled["verdict"]
    assert h["max_abs_gap"] < 1e-9, h["max_abs_gap"]
    assert coupled["n_features_decoupled"] == 0
    assert "usable early warning" in coupled["statement"], coupled["statement"]
    print(f"OK 10: identical schedules -> COUPLED (max gap {h['max_abs_gap']:.1e}, "
          f"spearman {h['spearman']:.2f}) — reported as a valid result")


# ---- 11: the clean baseline must be the RAW recording ----------------------

def test_l3_clean_reference_guard():
    with tempfile.TemporaryDirectory() as td:
        clean_path, sweep_dir, levels, paths = _write_sweep(td)
        sweep = sweep_from_dir(sweep_dir, "rt60", list(np.linspace(0.05, 0.8, 8)))

        # (a) using the mildest sweep rung as the baseline: refused
        try:
            run_l3_decoupling(paths[0], sweep, "rt60", feature_keys=("flatness",))
        except ValueError as e:
            assert "one of the sweep clips" in str(e), str(e)
        else:
            raise AssertionError("a sweep clip must not be accepted as the baseline")

        # (b) any other file staged in the sweep dir: refused by default
        stray = os.path.join(sweep_dir, "clean_copy.wav")
        sf.write(stray, _tone(), FS)
        try:
            run_l3_decoupling(stray, sweep, "rt60", feature_keys=("flatness",))
        except ValueError as e:
            assert "lives in the sweep directory" in str(e), str(e)
        else:
            raise AssertionError("same-directory baseline must be refused by default")
        # ...and can be opted into explicitly, never silently
        ok = run_l3_decoupling(stray, sweep, "rt60", feature_keys=("flatness",),
                               allow_same_dir=True)
        assert ok["clean_reference"] == stray

        # sweep parsing: levels come off the filenames, ascending, 1:1 with WER
        wer_map = {lv: 0.1 * i for i, lv in enumerate(levels)}
        pts = sweep_from_dir(sweep_dir, "rt60", wer_map)
        assert [p.level for p in pts] == levels
        assert [round(p.wer, 6) for p in pts] == [round(wer_map[l], 6) for l in levels]
        assert all(isinstance(p, SweepPoint) for p in pts)
        try:
            sweep_from_dir(sweep_dir, "rt60", [0.1, 0.2])       # wrong length
        except ValueError as e:
            assert "must line up one-to-one" in str(e), str(e)
        else:
            raise AssertionError("mismatched WER count must raise")

    print("OK 11: baseline guard refuses a sweep clip and a same-dir file "
          "(explicit opt-in only); sweep_from_dir parses 8 levels off the "
          "filenames and pairs them 1:1 with WER")


# ---- 12: aggregation is per condition, not per clip ------------------------

def test_aggregate_by_condition():
    rows = [r for r in build_l1_table() if r["model"] == "nova-3"]
    agg = aggregate_by_condition(rows)
    assert len(agg) == len(RT60) * len(SNR)
    assert all(a["n_clips"] == N_CLIPS for a in agg)
    one = agg[0]
    src = [r for r in rows if r["condition_name"] == one["condition_name"]]
    assert abs(one["wer"] - float(np.mean([r["wer"] for r in src]))) < 1e-12
    assert one["n_ref"] == sum(r["n_ref"] for r in src)      # counts SUM
    assert one["noise_type"] == "babble" and isinstance(one["rt60"], float)
    print(f"OK 12: {len(rows)} clip-rows -> {len(agg)} condition-rows "
          f"({N_CLIPS} clips each); WER averaged, edit counts summed, factor "
          "coordinates preserved")


if __name__ == "__main__":
    test_l1_names_divergence_region()
    test_l1_raw_confidences_never_compared()
    test_l1_hallucination_surfaced()
    test_l1_failures_and_csv_round_trip()
    test_l2_word_label_recipe()
    test_l2_misalignment_raises()
    test_l2_no_word_level_split()
    test_l2_feature_calibrator_beats_temperature()
    test_l2_refuses_mixed_models()
    test_l3_decoupling_and_leader()
    test_l3_coupled_case()
    test_l3_clean_reference_guard()
    test_aggregate_by_condition()
    print("\nAll layers tests passed — L1 names the planted divergence region "
          "without ever comparing raw cross-family confidence, L2 recovers the "
          "reverb-growing overconfidence on a grouped split (and refuses the "
          "leaky one), L3 detects decoupling and names the leading curve.")
