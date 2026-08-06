"""
Synthetic validation for the D3 reporting layers — analysis/interactions.py
(D3a) and analysis/al_savings.py (D3b). FULLY OFFLINE: no audio, no API, no
network. Planted structure stands in for the real grid and we assert the
reporters recover it *and* refuse to overclaim when it isn't there.

What is planted, and what each plant proves:

  1. A STRONG, CENTRED rt60 x snr_db interaction -> the ST-S1 gap must single out
     exactly those two factors and the pre-registration verdict must read
     CONFIRMED. (Centred, i.e. K*(r-.5)*((1-q)-.5), so the interaction contributes
     essentially no main effect and the gap is unambiguous. SPEC §5 already warns
     that a *moderate* second-order effect is not resolvable at N=1024 — this test
     validates the decision machinery, not the achievable power.)

  2. NO interaction at all (a purely additive surface) -> the verdict must come
     back NOT CONFIRMED. This is the more important half. A verdict function that
     always confirms is worse than no verdict, and a pre-registration is only
     worth something if it can fail.

  3. A non-monotonic WER dip inside a synthetic MASTER TABLE -> the surrogate must
     propose it as a candidate counterintuitive cell, and every proposal must come
     back marked unconfirmed and never presentable as measured. A monotone oracle
     must then REFUTE it (the surrogate imagined a detail the "measurement"
     doesn't have) — which is the exact failure mode the two-stage protocol exists
     to catch.

  4. A sharp failure boundary inside a synthetic master table -> the AL savings
     reporter must produce a multi-seed BAND, must refuse a single-seed claim, and
     must build its held-out test set from the table with ZERO oracle calls
     (asserted with a call counter, not by inspection).

Deterministic: fixed seeds throughout.  Run:  python3 tests/test_d3.py
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
import csv
import json
import math
import os
import tempfile
from unittest import mock

import numpy as np

from deadzone.design import DEFAULT_FACTOR_SPACE, sobol_indices
from deadzone.active_learning import DEFAULT_THRESHOLD

from deadzone.analysis.interactions import (
    CONFIRMED, PROPOSED, REFUTED, PreRegistration, S2_CAVEAT,
    condition_matrix, confirm_cells, encode_sample, format_interaction_report,
    format_verdict, interaction_evidence, interaction_gap_table,
    interaction_report, load_master_rows, load_sobol_json, normalize_sobol_result,
    plot_payload as interactions_plot_payload,
    preregistration_verdict, propose_counterintuitive_cells, sobol_to_json,
    s2_direction_only,
)
from deadzone.analysis.al_savings import (
    INSUFFICIENT, MIN_SEEDS, SUFFICIENT, InsufficientSeedsError,
    al_savings_report, format_al_savings, master_split_for_al, multi_seed_curves,
    plot_payload as al_plot_payload, run_seed, savings_headline, seed_band,
    surrogate_oracle_from_master, test_set_from_master,
)

SPACE = DEFAULT_FACTOR_SPACE

# ---- planted surfaces -------------------------------------------------------

_NOISE = {"babble": 0.06, "engine": 0.03, "road": 0.0}
_CODEC = {"none": 0.0, "g726": 0.03, "opus-lowrate": 0.06}

K_INT = 0.90          # planted interaction strength (centred -> pure 2nd order)
SNR0, SNR_W = 8.0, 3.5    # planted non-monotonic dip location / width (dB)
K_DIP = 0.22              # planted dip depth
K_SHARP = 7.0             # planted failure-boundary steepness


def _norm(x, lo, hi):
    return (x - lo) / (hi - lo)


def planted_interaction_wer(s: dict) -> float:
    """Small main effects + ONE strong, CENTRED rt60 x snr_db interaction."""
    r = _norm(s["rt60"], 0.2, 1.0)
    q = _norm(s["snr_db"], 0.0, 25.0)
    main = (0.06 * r + 0.08 * (1 - q) + _NOISE[s["noise_type"]]
            + _CODEC[s["codec"]] + 0.04 * s["mic_rolloff"])
    inter = K_INT * (r - 0.5) * ((1 - q) - 0.5)
    return float(min(max(0.30 + main + inter, 0.0), 1.0))


def additive_wer(s: dict) -> float:
    """Every factor acts alone. NOTHING to find — the negative control."""
    r = _norm(s["rt60"], 0.2, 1.0)
    q = _norm(s["snr_db"], 0.0, 25.0)
    return float(min(max(0.05 + 0.30 * r + 0.35 * (1 - q) + _NOISE[s["noise_type"]]
                         + _CODEC[s["codec"]] + 0.10 * s["mic_rolloff"], 0.0), 1.0))


def dip_surface(s: dict) -> float:
    """Monotone in reverb/noise EXCEPT for a sweet spot dip at snr ~ SNR0."""
    r = _norm(s["rt60"], 0.2, 1.0)
    q = _norm(s["snr_db"], 0.0, 25.0)
    base = 0.10 + 0.30 * r + 0.45 * (1 - q) + _NOISE[s["noise_type"]]
    dip = K_DIP * math.exp(-((s["snr_db"] - SNR0) ** 2) / (2 * SNR_W ** 2))
    return float(min(max(base - dip, 0.0), 1.0))


def monotone_surface(s: dict) -> float:
    """dip_surface with the dip REMOVED — the 'oracle disagrees' control."""
    r = _norm(s["rt60"], 0.2, 1.0)
    q = _norm(s["snr_db"], 0.0, 25.0)
    return float(min(max(0.10 + 0.30 * r + 0.45 * (1 - q) + _NOISE[s["noise_type"]],
                         0.0), 1.0))


def boundary_surface(s: dict) -> float:
    """A sharp, KNOWN failure contour in the rt60 x snr_db plane (as D3's oracle)."""
    r = _norm(s["rt60"], 0.2, 1.0)
    q = _norm(s["snr_db"], 0.0, 25.0)
    z = r + (1.0 - q) - 1.0
    z += 0.12 * (s["mic_rolloff"] - 0.5) + _NOISE[s["noise_type"]] + _CODEC[s["codec"]] - 0.06
    return 1.0 / (1.0 + math.exp(-K_SHARP * z))


# ---- a synthetic MASTER TABLE, written and read back as real CSV ------------

MASTER_HEADER = (
    "clip_id", "condition_name", "rt60", "snr_db", "noise_type", "codec",
    "mic_rolloff", "model", "transcript", "wer", "n_ref", "n_sub", "n_del",
    "n_ins", "n_match", "mean_conf", "utterance_conf", "word_confidences",
    "edits", "rir_key", "rir_rt60_measured", "noise_key", "failed", "error",
    "run_id", "ts",
)

RT60_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
SNR_LEVELS = (0.0, 5.0, 8.0, 12.0, 18.0, 25.0)
NOISE_LEVELS = ("babble", "road")
CLIPS = ("u02", "u05", "u06")


def build_master_rows(surface, seed: int = 0, n_failed: int = 4,
                      clip_noise: float = 0.02, model: str = "nova-3") -> list[dict]:
    """
    A master table with the frozen schema, per-clip measurement noise, a few
    `failed=True` sentinel rows, and `rir_rt60_measured` deliberately OFFSET from
    the requested `rt60` (the real library snaps a request to the nearest measured
    RIR). Both details exist so the loader is tested against the table it will
    actually meet, not a tidied-up version of it.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for rt in RT60_LEVELS:
        for snr in SNR_LEVELS:
            for nz in NOISE_LEVELS:
                sample = {"rt60": rt, "snr_db": snr, "noise_type": nz,
                          "codec": "none", "mic_rolloff": 0.0}
                truth = surface(sample)
                # delivered reverb != requested reverb (asset snapping)
                measured_rt60 = round(min(max(rt + 0.03, 0.2), 1.0), 3)
                name = f"rt{rt}_snr{snr}_{nz}"
                for clip in CLIPS:
                    w = float(np.clip(truth + rng.normal(0, clip_noise), 0.0, 1.0))
                    rows.append({
                        "clip_id": clip, "condition_name": name,
                        "rt60": rt, "snr_db": snr, "noise_type": nz,
                        "codec": "none", "mic_rolloff": 0.0, "model": model,
                        "transcript": "synthetic", "wer": round(w, 6),
                        "n_ref": 9, "n_sub": 1, "n_del": 0, "n_ins": 0, "n_match": 8,
                        "mean_conf": 0.9, "utterance_conf": 0.9,
                        "word_confidences": "[0.9]", "edits": "[]",
                        "rir_key": f"rir_{rt}", "rir_rt60_measured": measured_rt60,
                        "noise_key": f"{nz}_1", "failed": False, "error": "",
                        "run_id": "test", "ts": "2026-08-04T00:00:00Z",
                    })
    # failure sentinels: WER 1.0 with failed=True. If the loader averages these
    # in, it manufactures a failure region in the harshest corner of the space.
    for i in range(n_failed):
        r = dict(rows[i])
        r.update({"clip_id": "u99", "wer": 1.0, "failed": True,
                  "error": "synthetic failure", "transcript": ""})
        rows.append(r)
    return rows


def write_and_reload(rows, tmpdir: str) -> list[dict]:
    """Round-trip through real CSV so every value arrives as a STRING — including
    `failed`, whose "False" is truthy and is the classic silent killer."""
    path = os.path.join(tmpdir, "master.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(MASTER_HEADER))
        w.writeheader()
        w.writerows(rows)
    return load_master_rows(path)


class CountingOracle:
    """An oracle wrapper that counts calls — the only honest way to assert that a
    code path is free."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, sample: dict) -> float:
        self.calls += 1
        return float(self.fn(sample))


# ============================================================================
# 1 — PLANTED INTERACTION: the ST-S1 gap finds it, the verdict says CONFIRMED
# ============================================================================

def test_verdict_confirms_planted_interaction():
    res = sobol_indices(SPACE, planted_interaction_wer, N=1024, seed=0)
    ev = interaction_evidence(res)

    # (a) PRIMARY evidence: the two interacting factors carry the largest gaps,
    #     and only they clear the pre-set threshold with their CIs.
    gaps = {g["factor"]: g for g in ev["primary_gap"]}
    top_two = {ev["primary_gap"][0]["factor"], ev["primary_gap"][1]["factor"]}
    assert top_two == {"rt60", "snr_db"}, ev["primary_gap"]
    assert set(ev["top_gap_factors"]) == {"rt60", "snr_db"}, ev["top_gap_factors"]
    for other in ("noise_type", "codec", "mic_rolloff"):
        assert not gaps[other]["significant"], gaps[other]
        assert gaps["rt60"]["gap"] > gaps[other]["gap"], (gaps, other)

    # (b) S2 is consulted for DIRECTION ONLY, and says so in its own payload
    s2 = s2_direction_only(res, ("rt60", "snr_db"))
    assert s2["is_top_pair"], s2
    assert s2["magnitude_usable"] is False, s2
    assert "never how much" in s2["usable_for"]

    # (c) the verdict
    v = preregistration_verdict(res, commit="abc1234", date="2026-08-04")
    assert v["confirmed"] is True and v["status"] == "CONFIRMED", v["sentence"]
    assert v["gap_criterion_met"] and v["pair_criterion_met"]
    # the sentence must carry REAL numbers and the CI, not adjectives
    for token in ("pre-registered", "ST-S1 gap", "CONFIRMED", "abc1234", "2026-08-04"):
        assert token in v["sentence"], (token, v["sentence"])
    assert f"{gaps['rt60']['gap']:.3f}" in v["sentence"]

    # (d) the renderer leads with the gap and never sells S2 as a magnitude
    text = format_interaction_report(interaction_report(res))
    i_primary = text.index("PRIMARY INTERACTION EVIDENCE")
    i_secondary = text.index("SECONDARY — S2, WHICH PAIR ONLY")
    assert i_primary < i_secondary, "S2 must not be presented before the ST-S1 gap"
    assert S2_CAVEAT in text
    assert "PRE-REGISTRATION: CONFIRMED" in text
    assert text.index("PRE-REGISTRATION") < i_primary, "the verdict must come first"

    print(f"OK 1: planted interaction CONFIRMED — ST-S1 gap "
          f"rt60={gaps['rt60']['gap']:.3f} [{gaps['rt60']['gap_lo']:.3f}, "
          f"{gaps['rt60']['gap_hi']:.3f}], snr_db={gaps['snr_db']['gap']:.3f} "
          f"[{gaps['snr_db']['gap_lo']:.3f}, {gaps['snr_db']['gap_hi']:.3f}]; "
          f"S2 rank {s2['rank']}/{s2['n_pairs']} (direction only)")
    print("    verdict:", v["sentence"][:150] + " ...")
    return res


# ============================================================================
# 2 — THE NEGATIVE CASE (matters more): additive surface -> NOT CONFIRMED
# ============================================================================

def test_verdict_rejects_when_nothing_is_planted():
    res = sobol_indices(SPACE, additive_wer, N=1024, seed=0)
    v = preregistration_verdict(res)

    assert v["confirmed"] is False, v["sentence"]
    assert v["status"] == "NOT CONFIRMED", v
    assert v["gap_criterion_met"] is False, v["reasons"]

    # every factor's gap must be indistinguishable from zero
    for g in interaction_gap_table(res, min_gap=PreRegistration().min_gap):
        assert not g["significant"], g
        assert abs(g["gap"]) < 0.05, g

    # ... and the write-up rendering must present that as a RESULT, not a hedge
    assert "NOT CONFIRMED" in v["sentence"]
    assert ("ADDITIVE" in v["sentence"]) or ("UNDERPOWERED" in v["sentence"]), v["sentence"]
    text = format_verdict(v)
    assert "this is a RESULT, not a failure" in text
    # the verdict is never omitted, and never softened into a maybe
    assert "sentence" in v and len(v["sentence"]) > 120

    # a verdict function that can only confirm is worthless: prove the two
    # branches differ on the same machinery.
    assert preregistration_verdict(
        sobol_indices(SPACE, planted_interaction_wer, N=512, seed=1))["confirmed"]

    print(f"OK 2: additive surface -> {v['status']} (largest gap "
          f"{max(g['gap'] for g in v['primary'].values()):+.4f}, CI includes 0); "
          f"rendered as a result, not a hedge")
    return v


# ============================================================================
# 3 — results/sobol.json round trip (R4.5 writes it, R5.4 reads it)
# ============================================================================

def test_sobol_json_roundtrip():
    res = sobol_indices(SPACE, planted_interaction_wer, N=256, seed=3)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sobol.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sobol_to_json(res), f)
        back = load_sobol_json(path)

    # tuples survived the JSON list round trip, arrays came back as arrays
    assert isinstance(back["s2_ranked"][0]["pair"], tuple), back["s2_ranked"][0]
    assert isinstance(back["S1"], np.ndarray)
    v_live = preregistration_verdict(res)
    v_disk = preregistration_verdict(back)
    assert v_live["status"] == v_disk["status"]
    assert v_live["sentence"] == v_disk["sentence"]

    # and a result whose derived views were dropped by the serializer rebuilds
    stripped = {k: val for k, val in sobol_to_json(res).items()
                if k not in ("interaction_gap", "s2_ranked")}
    rebuilt = normalize_sobol_result(stripped)
    assert preregistration_verdict(rebuilt)["sentence"] == v_live["sentence"]

    print("OK 3: sobol.json round-trips — same verdict from memory, from disk, "
          "and from a stripped payload")


# ============================================================================
# 4 — SURROGATE-PROPOSED CELLS ARE UNCONFIRMED, ALWAYS
# ============================================================================

def test_counterintuitive_cells_are_proposed_not_measured():
    with tempfile.TemporaryDirectory() as td:
        rows = write_and_reload(build_master_rows(dip_surface, seed=1), td)

    # the loader must exclude the failure sentinels, not average them in
    mat = condition_matrix(rows, SPACE, model="nova-3")
    assert mat["n_failed_rows"] == 4, mat["n_failed_rows"]
    assert mat["n_conditions"] == len(RT60_LEVELS) * len(SNR_LEVELS) * len(NOISE_LEVELS)
    assert mat["y"].max() < 1.0, "a failure sentinel (WER=1.0) leaked into the surface"
    # rt60 coordinate is the DELIVERED (measured) value, not the request
    def _delivered(rt):
        return round(min(max(rt + 0.03, 0.2), 1.0), 3)
    assert all(abs(c["sample"]["rt60"] - _delivered(c["rt60_requested"])) < 1e-6
               for c in mat["conditions"]), "regressed against requested rt60"
    assert any(c["sample"]["rt60"] != c["rt60_requested"] for c in mat["conditions"])

    prop = propose_counterintuitive_cells(rows, SPACE, model="nova-3", grid=9,
                                          top_k=6, seed=0)

    assert prop["status"] == PROPOSED
    assert prop["presentable_as_measured"] is False
    assert prop["n_candidates"] > 0, "surrogate proposed nothing on a planted dip"
    assert prop["n_oracle_calls_required"] == len(prop["conditions_to_evaluate"]) > 0
    for c in prop["candidates"]:
        assert c["confirmed"] is False, c
        assert c["status"] == PROPOSED, c
        assert "NOT measured" in c["evidence"], c
        assert c["probe_conditions"], c
    # the planted dip is on snr_db and the surrogate found it there
    dips = [c for c in prop["candidates"] if c["kind"] == "non_monotonic"]
    assert dips, prop["candidates"]
    assert any(c["factor"] == "snr_db" for c in dips), [c["factor"] for c in dips]

    # the printed report must SAY unconfirmed, in the report itself
    text = format_interaction_report({
        "sobol_tables": "", "evidence": interaction_evidence(
            sobol_indices(SPACE, additive_wer, N=64, seed=0)),
        "verdict": preregistration_verdict(sobol_indices(SPACE, additive_wer, N=64, seed=0)),
        "counterintuitive": prop,
    })
    assert PROPOSED in text
    assert "[unconfirmed]" in text
    assert "[CONFIRMED]" not in text
    assert "may be presented as a measured finding until then" in text

    # the dashboard payload carries the same flag — no laundering via JSON
    payload = interactions_plot_payload({
        "evidence": interaction_evidence(sobol_indices(SPACE, additive_wer, N=64, seed=0)),
        "verdict": preregistration_verdict(sobol_indices(SPACE, additive_wer, N=64, seed=0)),
        "counterintuitive": prop,
    })
    assert payload["counterintuitive"]["presentable_as_measured"] is False
    assert all(c["confirmed"] is False for c in payload["counterintuitive"]["cells"])
    assert all(p["magnitude_usable"] is False for p in payload["s2_pairs"])

    print(f"OK 4: {prop['n_candidates']} counterintuitive cells PROPOSED from the "
          f"surrogate ({prop['surrogate_train_conditions']} measured conditions), "
          f"all marked unconfirmed; {prop['n_oracle_calls_required']} oracle calls "
          f"needed to confirm them")
    return rows, prop


# ============================================================================
# 5 — STAGE 2: only real oracle calls can flip `confirmed`
# ============================================================================

def test_confirmation_needs_the_real_oracle(rows, prop):
    # (a) the true surface HAS the dip -> the candidate confirms
    truth = CountingOracle(dip_surface)
    conf = confirm_cells(prop, truth)
    assert conf["n_oracle_calls_spent"] == prop["n_oracle_calls_required"], (
        conf["n_oracle_calls_spent"], prop["n_oracle_calls_required"])
    assert truth.calls == conf["n_oracle_calls_spent"], "probe de-duplication broke"
    assert conf["presentable_as_measured"] is True
    dips = [c for c in conf["candidates"]
            if c["kind"] == "non_monotonic" and c["factor"] == "snr_db"]
    assert any(c["confirmed"] and c["status"] == CONFIRMED for c in dips), dips

    # (b) a surface WITHOUT the dip -> the same proposals are refuted, and kept
    flat = CountingOracle(monotone_surface)
    ref = confirm_cells(prop, flat)
    assert flat.calls > 0
    snr_dips = [c for c in ref["candidates"]
                if c["kind"] == "non_monotonic" and c["factor"] == "snr_db"]
    assert snr_dips and not any(c["confirmed"] for c in snr_dips), snr_dips
    assert all(c["status"] == REFUTED for c in snr_dips)
    assert ref["n_refuted"] >= len(snr_dips)
    assert len(ref["candidates"]) == len(prop["candidates"]), "refuted cells dropped"

    # (c) the proposal itself is never mutated into a confirmation in place
    assert all(c["confirmed"] is False for c in prop["candidates"])

    print(f"OK 5: confirmation is oracle-gated — {conf['n_confirmed']} confirmed "
          f"against the true surface, {ref['n_refuted']} refuted against a surface "
          f"without the dip ({conf['n_oracle_calls_spent']} calls each, deduped)")


# ============================================================================
# 6 — THE AL TEST SET COMES FROM THE MASTER TABLE, AT ZERO ORACLE COST
# ============================================================================

def test_al_test_set_from_master_costs_nothing():
    with tempfile.TemporaryDirectory() as td:
        rows = write_and_reload(build_master_rows(boundary_surface, seed=2), td)

    counter = CountingOracle(boundary_surface)

    split = test_set_from_master(rows, SPACE, model="nova-3", holdout_frac=0.4, seed=0)
    assert counter.calls == 0, "building the test set called the oracle"
    assert split["oracle_calls"] == 0
    assert split["n_test"] >= 8 and split["n_train"] >= 8
    assert split["n_train"] + split["n_test"] == split["n_conditions"]
    assert split["n_test_near_boundary"] > 0

    # train/test really are disjoint (score a surrogate oracle on its own training
    # points and every arm looks perfect)
    train = {tuple(np.round(r, 9)) for r in split["X_train"]}
    test = {tuple(np.round(r, 9)) for r in split["X_test"]}
    assert not (train & test), "surrogate-oracle training set overlaps the test set"

    # the surrogate oracle is also free to build, and tracks the planted surface
    oracle, meta = surrogate_oracle_from_master(split=split, seed=0)
    assert meta["oracle_calls_to_build"] == 0 and meta["provenance"] == "surrogate"
    probe = {"rt60": 0.9, "snr_db": 2.0, "noise_type": "babble", "codec": "none",
             "mic_rolloff": 0.0}
    assert abs(oracle(probe) - boundary_surface(probe)) < 0.2, (
        oracle(probe), boundary_surface(probe))
    assert counter.calls == 0, "the free path spent oracle calls after all"

    # there is NO "build me a fresh test set" path to fall back into: the held-out
    # set is a required argument, so a run cannot silently start buying hundreds of
    # extra oracle calls per seed via make_test_set.
    for fn, kw in ((run_seed, {}), (multi_seed_curves, {"seeds": (0, 1, 2)})):
        try:
            fn(counter, SPACE, **({"seed": 0} if fn is run_seed else {}), **kw)
            raise AssertionError(f"{fn.__name__} ran without a held-out test set")
        except TypeError:
            pass
    assert counter.calls == 0, "a rejected call still burned oracle calls"

    # encode/decode round trip, since every one of the above depends on it
    assert SPACE.decode(encode_sample(SPACE, probe)) == probe

    print(f"OK 6: AL test set built from {split['n_conditions']} measured "
          f"conditions ({split['n_test']} held out, "
          f"{split['n_test_near_boundary']} near the contour) — 0 oracle calls")
    return rows, split


# ============================================================================
# 7 — ONE SEED IS NOT EVIDENCE
# ============================================================================

def test_repeated_seeds_cannot_fake_a_band(split):
    """
    TRAP 1's blind spot. The seed COUNT was checked; the seeds being DISTINCT
    was not. `active_learn`/`random_baseline` are deterministic in the seed, so
    `seeds=(0, 0, 0)` runs three identical trajectories, passes the MIN_SEEDS
    gate, and produces a zero-width band reported as "median over 3 seeds" —
    the single-seed anecdote this module exists to prevent, presented in the
    STRONGEST available form because the spread vanished.
    """
    oracle = CountingOracle(boundary_surface)
    try:
        multi_seed_curves(oracle, SPACE, X_test=split["X_test"],
                          y_test=split["y_test"], seeds=[0, 0, 0], n_seed=6,
                          budget=6)
        raise AssertionError("three copies of one seed were accepted as a band")
    except InsufficientSeedsError as e:
        assert "duplicate seed" in str(e) and "zero width" in str(e), e
    assert oracle.calls == 0, "the refused run still burned oracle calls"

    try:
        multi_seed_curves(oracle, SPACE, X_test=split["X_test"],
                          y_test=split["y_test"], seeds=[], n_seed=6, budget=6)
        raise AssertionError("an empty seed list was accepted")
    except InsufficientSeedsError as e:
        assert "no seeds" in str(e), e

    # negative control: three DISTINCT seeds are still accepted and the band has
    # real width, i.e. the guard is pinned to the repetition and nothing else.
    multi = multi_seed_curves(oracle, SPACE, X_test=split["X_test"],
                              y_test=split["y_test"], seeds=[0, 1, 2], n_seed=6,
                              budget=6)
    finals = [ps["final"]["active_boundary"]["boundary_rmse"]
              for ps in multi["per_seed"]]
    assert len(set(finals)) > 1, ("distinct seeds produced identical runs", finals)
    # The robustness verdict interpolates `np.median(all_diffs)` directly into
    # its SENTENCE. `paired_arm_difference` keeps only the seeds where both arms
    # ended finite, so that list can come back empty — and `np.median([])` is
    # nan, printing "median paired difference nan", which reads as a MEASURED
    # TIE rather than as no measurement at all. (The dict field was already
    # guarded; the quoted sentence was not.)
    from deadzone.analysis import al_savings as ALS
    fake_multi = {"per_seed": [{"seed": 0, "curves": {}}], "threshold": 0.5,
                  "band": 0.15}
    with mock.patch.object(
            ALS, "paired_arm_difference",
            lambda *a, **k: {"per_seed": [], "median_diff": float("nan")}):
        with mock.patch.object(ALS, "test_set_from_master",
                               lambda *a, **k: split), \
             mock.patch.object(ALS, "surrogate_oracle_from_master",
                               lambda **k: (oracle, {})), \
             mock.patch.object(ALS, "multi_seed_curves",
                               lambda *a, **k: fake_multi), \
             mock.patch.object(ALS, "oracle_fidelity_floor",
                               lambda *a, **k: {"boundary_rmse": 0.1}):
            try:
                ALS.split_robustness([], SPACE, split_seeds=(0,), seeds=(0, 1, 2))
                raise AssertionError("an all-nan robustness verdict was returned")
            except ValueError as exc:
                assert "no paired runs survived" in str(exc), exc
                assert "measured tie" in str(exc), exc

    print("OK 6b: repeated (and empty) seed lists are refused — a zero-width "
          "band cannot be printed as 'median over N seeds' — and a robustness "
          "verdict with no comparable runs refuses rather than printing "
          "'median paired difference nan'")


def test_single_seed_is_refused_or_flagged(split):
    oracle = CountingOracle(boundary_surface)

    # (a) refused outright by default
    try:
        multi_seed_curves(oracle, SPACE, X_test=split["X_test"],
                          y_test=split["y_test"], seeds=[0], n_seed=6, budget=6)
        raise AssertionError("a single-seed savings run was allowed silently")
    except InsufficientSeedsError as e:
        assert str(MIN_SEEDS) in str(e)
    assert oracle.calls == 0, "the refused run still burned oracle calls"

    # (b) the escape hatch does not launder it into a claim
    multi = multi_seed_curves(oracle, SPACE, X_test=split["X_test"],
                              y_test=split["y_test"], seeds=[0], n_seed=6,
                              budget=6, allow_single_seed=True)
    assert multi["evidence_level"] == INSUFFICIENT
    res = al_savings_report(multi)
    h = res["headline"]
    assert h["presentable"] is False
    assert "NOT EVIDENCE" in h["sentence"], h["sentence"]
    assert "reduction in oracle calls" not in h["sentence"]

    text = format_al_savings(res)
    assert "INSUFFICIENT EVIDENCE" in text
    assert "% reduction" not in text and "reduction in oracle calls" not in text
    assert al_plot_payload(res)["presentable"] is False

    print("OK 7: one seed is refused by default and, when forced, is stamped "
          "INSUFFICIENT with no savings number anywhere in the rendering")


# ============================================================================
# 8 — THE HEADLINE IS A MULTI-SEED BAND
# ============================================================================

N_SEED, BUDGET = 8, 12
SEEDS = (0, 1, 2)


def test_multi_seed_band_and_headline(split):
    oracle = CountingOracle(boundary_surface)
    multi = multi_seed_curves(oracle, SPACE, X_test=split["X_test"],
                              y_test=split["y_test"], seeds=SEEDS,
                              n_seed=N_SEED, budget=BUDGET,
                              api_confirmed_seeds=(0,))
    assert multi["evidence_level"] == SUFFICIENT
    assert multi["test_oracle_calls"] == 0, "the test set cost oracle calls"

    # EXACT accounting: 3 arms x (n_seed + budget) per seed, and NOT ONE MORE.
    # The whole point of the master-table test set is that it adds zero.
    expected = len(SEEDS) * 3 * (N_SEED + BUDGET)
    assert oracle.calls == expected, (oracle.calls, expected)

    res = al_savings_report(multi)
    band = res["bands"]["active_boundary"]
    assert band and all(p["n_seeds"] == len(SEEDS) for p in band), band
    for p in band:
        assert p["lo"] <= p["median"] <= p["hi"], p
        assert len(p["values"]) == len(SEEDS)
    # the band is reported for the passive arm too, or there is nothing to compare
    assert res["bands"]["random"]

    h = res["headline"]
    assert h["presentable"] is True
    assert h["n_seeds"] == len(SEEDS)
    assert len(h["per_seed_evals"]["active_boundary"]) == len(SEEDS)
    assert "median over 3 seeds" in h["sentence"], h["sentence"]
    assert "range" in h["sentence"], h["sentence"]
    # the headline number itself: evals_to_target -> "P% fewer oracle calls"
    assert math.isfinite(h["pct_reduction"]) and h["pct_reduction"] > 0, h
    assert "% reduction in oracle calls" in h["sentence"], h["sentence"]
    # provenance is not optional: the reader must not assume 3 API-backed runs
    assert "confirmed" in h["provenance"]["statement"]
    assert h["provenance"]["surrogate_only_seeds"] == [1, 2]

    # the finding itself: active reaches the passive arm's own fidelity sooner
    med_a = h["median_evals"]["active_boundary"]
    med_p = h["median_evals"]["random"]
    assert math.isfinite(med_p), h["per_seed_evals"]
    assert med_a <= med_p, (med_a, med_p, h["per_seed_evals"])

    # no "best seed" anywhere in the API or the rendering
    text = format_al_savings(res)
    assert "best seed" not in text.lower()
    assert "MEDIAN [min-max]" in text
    assert "median over 3 seeds" in text

    payload = al_plot_payload(res)
    assert {s["arm"] for s in payload["series"]} >= {"active_boundary", "random"}
    for s in payload["series"]:
        assert len(s["n_evals"]) == len(s["median"]) == len(s["lo"]) == len(s["hi"])
    json.dumps(payload)          # must be dashboard-serializable

    print("\n--- boundary_rmse vs oracle calls, median [min-max] over 3 seeds ---")
    print(text.split("-" * 78)[-1].strip()[:800])
    print(f"OK 8: {h['sentence']}")
    return res


# ============================================================================
# 9 — schema drift guard against the frozen master table
# ============================================================================

def test_schema_matches_the_runner():
    try:
        from scripts.run_experiment import MASTER_COLUMNS
    except Exception as e:                       # noqa: BLE001 — optional dep path
        print(f"SKIP 9: run_experiment not importable ({type(e).__name__}: {e})")
        return
    assert tuple(MASTER_HEADER) == tuple(MASTER_COLUMNS), (
        "this test's synthetic table drifted from the frozen schema")
    # the factor columns ARE the factor space — D3 reads them off design.py, so a
    # rename in either place must fail here rather than silently produce NaNs.
    assert MASTER_COLUMNS[2:7] == tuple(SPACE.names)
    print(f"OK 9: synthetic master table matches the frozen schema "
          f"({len(MASTER_COLUMNS)} columns), factor columns == DEFAULT_FACTOR_SPACE")


if __name__ == "__main__":
    print("factor space:", SPACE.names)
    print(f"pre-registration: {PreRegistration().factor_a} x "
          f"{PreRegistration().factor_b}; rule: {PreRegistration().rule}\n")

    test_verdict_confirms_planted_interaction()
    test_verdict_rejects_when_nothing_is_planted()
    test_sobol_json_roundtrip()
    rows, prop = test_counterintuitive_cells_are_proposed_not_measured()
    test_confirmation_needs_the_real_oracle(rows, prop)
    _, split = test_al_test_set_from_master_costs_nothing()
    test_repeated_seeds_cannot_fake_a_band(split)
    test_single_seed_is_refused_or_flagged(split)
    test_multi_seed_band_and_headline(split)
    test_schema_matches_the_runner()

    print("\nAll D3 tests passed — the pre-registration verdict confirms a planted "
          "interaction and REJECTS an additive surface, surrogate-proposed cells "
          "stay unconfirmed until real oracle calls confirm them, and the AL "
          "savings claim is a multi-seed band built from the master table at zero "
          "oracle cost.")
