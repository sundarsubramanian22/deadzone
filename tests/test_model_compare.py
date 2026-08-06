"""
Offline validation for Layer 1 (model_compare.py). No API, no audio — synthetic
per-model results tables with a KNOWN planted difference stand in for real grid
runs, and we assert the comparison surfaces it.

Planted structure (two models over the same rt60 x snr grid):
  * shared base WER rising with reverb and falling with SNR,
  * model "weak" gets a big WER BUMP for rt60 > 0.7 while staying CONFIDENT there
    (a dead zone: confidently wrong); "strong" has no bump and its confidence
    correctly DROPS as conditions worsen,
  * the two models live on DIFFERENT confidence scales — every "weak" confidence
    (<=0.70) is below every "strong" confidence (>=0.85). A naive raw-confidence
    comparison would call weak "never confident" and MISS its dead zone; only the
    within-model normalization catches it. That is the whole point of Layer 1.

Asserted: registry has three one-line arms; the comparison flags the rt60>0.7
region with weak as the worse model; weak has a dead zone there and strong does
not (despite weak's lower absolute confidence); strong's confidence tracks its
errors (negative shape) and weak's does not.

Deterministic (pure functions of the grid, no rng).  Run:
    python3 tests/test_model_compare.py
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

from deadzone.design import DEFAULT_FACTOR_SPACE
from deadzone.model_compare import (
    MODEL_REGISTRY, WER_INCOMPARABLE_ARMS, WerIncomparableArmError, get_model,
    compare_models, dead_zone_flags, find_divergence_regions, is_wer_comparable,
    within_model_conf_percentile, confidence_wer_shape, wer_comparability,
    wer_comparable_tables,
)
from deadzone.audio_pipeline import _parse_vosk_result

SPACE = DEFAULT_FACTOR_SPACE
RT60 = np.linspace(0.2, 1.0, 9)
SNR = np.linspace(0.0, 25.0, 5)


def _base_wer(rt60, snr):
    rn, qn = (rt60 - 0.2) / 0.8, snr / 25.0
    return 0.05 + 0.2 * rn + 0.3 * (1 - qn)


def _row(rt60, snr, wer, conf):
    return {"rt60": float(rt60), "snr_db": float(snr), "noise_type": "babble",
            "codec": "none", "mic_rolloff": 0.0,
            "wer": float(min(max(wer, 0.0), 1.0)), "mean_conf": float(conf)}


def build_tables():
    strong, weak = [], []
    for rt60 in RT60:
        rn = (rt60 - 0.2) / 0.8
        for snr in SNR:
            qn = snr / 25.0
            base = _base_wer(rt60, snr)
            # strong: no bump; confidence high-scale and tracks cleanliness
            strong.append(_row(rt60, snr, base,
                               conf=min(max(0.90 + 0.09 * qn - 0.05 * rn, 0), 1)))
            # weak: big WER bump past rt60=0.7 but STAYS confident there (its own
            # top). Elsewhere its confidence is low-scale (~0.45-0.55).
            bump = 0.5 if rt60 > 0.7 else 0.0
            wconf = 0.70 if rt60 > 0.7 else (0.45 + 0.10 * qn)
            weak.append(_row(rt60, snr, base + bump, conf=wconf))
    return {"strong": strong, "weak": weak}


# ---- 1: registry is N one-line arms, resolvable but not called -------------

def test_registry():
    """
    The registry is OPEN. It is asserted as a contract — every listed arm
    resolves to a callable, no duplicates, the spine and the open baseline are
    both there — and NOT as a frozen set of names.

    Pinning the exact membership was itself a two-arm-era assumption: it turns
    "a new arm was added" into a test failure that reads like a regression, which
    trains the next person to edit the assertion rather than to check whether the
    arm was wired in correctly. The properties below fail for a BROKEN arm and
    pass for a new working one, which is the distinction that matters.
    """
    required = {"nova-3", "whisper-base"}             # the spine + the open baseline
    assert required <= set(MODEL_REGISTRY), (required - set(MODEL_REGISTRY))
    assert len(set(MODEL_REGISTRY)) == len(MODEL_REGISTRY), MODEL_REGISTRY
    assert len(MODEL_REGISTRY) >= 2, MODEL_REGISTRY
    for name in MODEL_REGISTRY:
        assert isinstance(name, str) and name, name
        assert callable(get_model(name)), name        # resolvable, NOT invoked
    # negative control: the contract is a real filter, not a tautology.
    try:
        get_model("no-such-arm")
        assert False, "an unknown arm must raise, not resolve to something"
    except KeyError as exc:
        assert "no-such-arm" in str(exc), exc
    print(f"OK 1: MODEL_REGISTRY = {tuple(MODEL_REGISTRY)} "
          f"({len(MODEL_REGISTRY)} one-line arms, each resolvable)")


# ---- 2: cross-scale confidence handled WITHIN each model --------------------

def test_within_model_normalization():
    tables = build_tables()
    weak_max = max(r["mean_conf"] for r in tables["weak"])
    strong_min = min(r["mean_conf"] for r in tables["strong"])
    # the scales genuinely don't overlap -> raw cross-model comparison is invalid
    assert weak_max < strong_min, (weak_max, strong_min)
    # yet within-model percentile still puts weak's confident rows at the top
    # (region rows tie at percentile ~0.84; non-region tops out ~0.60)
    pct = within_model_conf_percentile(tables["weak"])
    top_rows = [tables["weak"][i] for i in np.where(pct >= 0.7)[0]]
    assert top_rows and all(r["rt60"] > 0.7 for r in top_rows), "weak's top-confidence rows"
    print(f"OK 2: weak conf<= {weak_max:.2f} < strong conf>= {strong_min:.2f}; "
          f"within-model percentile still isolates weak's confident region")


# ---- 3: the comparison surfaces the planted weak region --------------------

def test_comparison_surfaces_weak_region():
    tables = build_tables()
    res = compare_models(tables, SPACE, n_bins=4, wer_hi=0.3, conf_pct_hi=0.6)

    top = res["divergence_regions"][0]
    assert top["factor"] == "rt60", top
    assert top["worse_model"] == "weak", top
    assert top["better_model"] == "strong", top
    lo, hi = top["span"]
    assert lo >= 0.7, ("divergence should be at the high-reverb end", top["span"])
    assert top["wer_gap"] > 0.3, top["wer_gap"]

    dz = top["dead_zone_rate_by_model"]
    assert dz["weak"] > 0.8 and dz["strong"] == 0.0, dz    # weak silent-fails here
    print(f"OK 3: top divergence = {top['factor']} {top['span']}, worse=weak "
          f"(WER gap {top['wer_gap']:.2f}); dead-zone rate weak={dz['weak']:.2f} "
          f"strong={dz['strong']:.2f}")
    return res


# ---- 4: dead-zone rate + confidence-shape distinguish the models -----------

def test_dead_zone_and_shape():
    tables = build_tables()
    res = compare_models(tables, SPACE, n_bins=4)
    pm = res["per_model"]

    # weak silently fails overall; strong (confidence tracks error) does not
    assert pm["weak"]["dead_zone_rate"] > 0.0, pm["weak"]
    assert pm["strong"]["dead_zone_rate"] == 0.0, pm["strong"]

    # shape: strong's confidence tracks its WER (negative); weak's fails to
    s_strong = pm["strong"]["shape"]["spearman"]
    s_weak = pm["weak"]["shape"]["spearman"]
    assert s_strong < 0, s_strong
    assert s_weak > s_strong, (s_weak, s_strong)      # weak is less self-aware
    print(f"OK 4: dead-zone rate weak={pm['weak']['dead_zone_rate']:.2f} vs "
          f"strong={pm['strong']['dead_zone_rate']:.2f}; conf-vs-WER shape "
          f"strong rho={s_strong:.2f} < weak rho={s_weak:.2f}")


# ---- 4b: the two ways a comparison can be silently wrong -------------------

def test_nan_wer_never_dilutes_the_dead_zone_rate():
    """
    A NaN confidence is legitimate and handled (it sinks to percentile 0). A NaN
    WER is not: `nan >= wer_hi` is False, so the row is classified NOT a dead
    zone AND still counted in the denominator of `mean(flags)` — an unmeasured
    condition quietly dilutes the dead-zone rate, which is D1's headline number,
    with nothing anywhere reading as missing.
    """
    tables = build_tables()
    clean_flags = dead_zone_flags(tables["weak"])
    victim = int(np.flatnonzero(clean_flags)[0])       # a genuine dead-zone cell
    poisoned = [dict(r) for r in tables["weak"]]
    poisoned[victim]["wer"] = float("nan")             # e.g. an unfiltered failure

    clean_rate = float(np.mean(clean_flags))
    # what the un-guarded arithmetic WOULD have produced: a silently lower rate
    wer = np.array([r["wer"] for r in poisoned], dtype=float)
    pct = within_model_conf_percentile(poisoned)
    silent_rate = float(np.mean((wer >= 0.3) & (pct >= 0.6)))
    assert silent_rate < clean_rate, (silent_rate, clean_rate)

    try:
        dead_zone_flags(poisoned)
        assert False, "expected a non-finite-WER raise"
    except ValueError as exc:
        assert "non-finite WER" in str(exc) and "dilute" in str(exc), exc
    print(f"ok 4b: a NaN WER raises instead of silently dropping the dead-zone "
          f"rate from {clean_rate:.3f} to {silent_rate:.3f}")


def test_wer_key_selects_the_estimand():
    """
    `dead_zone_flags` must flag against the WER the caller names, because
    "confidently wrong" is a claim about the clips the confidence was averaged
    over. A per-condition row whose `mean_conf` comes only from the clips that
    produced words has to be judged on `wer_spoke`, not on the all-clips `wer` —
    that mismatch is what inflated D1's published gaps.
    """
    tables = build_tables()
    rows = [dict(r) for r in tables["strong"]]
    # `strong` is planted to have NO dead zones on its own WER. Give every row a
    # second, worse WER column: the same confidences must now flag under the new
    # key and not under the old one. Only the estimand changed.
    for r in rows:
        r["wer_spoke"] = min(r["wer"] + 0.6, 1.0)

    on_wer = dead_zone_flags(rows, wer_key="wer")
    on_spoke = dead_zone_flags(rows, wer_key="wer_spoke")
    assert on_wer.sum() == 0, "the control table must have no dead zones on `wer`"
    assert on_spoke.sum() > 0, "the same rows must flag under the worse estimand"
    # ...and the confidence half is untouched: the flip is entirely the WER key
    pct = within_model_conf_percentile(rows)
    assert np.array_equal(on_spoke, (np.array([r["wer_spoke"] for r in rows]) >= 0.3)
                          & (pct >= 0.6))

    # NEGATIVE CONTROL: when the two columns agree (no silent clips), naming
    # either key must give bit-identical flags.
    same = [dict(r, wer_spoke=r["wer"]) for r in tables["weak"]]
    assert np.array_equal(dead_zone_flags(same, wer_key="wer"),
                          dead_zone_flags(same, wer_key="wer_spoke"))

    # a missing/NaN value under the NAMED key raises, naming that key — a mute
    # condition must be held out explicitly, never read as "not a dead zone"
    holed = [dict(r) for r in rows]
    holed[3]["wer_spoke"] = float("nan")
    try:
        dead_zone_flags(holed, wer_key="wer_spoke")
        assert False, "expected a non-finite raise under the named key"
    except ValueError as exc:
        assert "wer_spoke" in str(exc), exc

    shape_a = confidence_wer_shape(rows, wer_key="wer")["spearman"]
    shape_b = confidence_wer_shape(rows, wer_key="wer_spoke")["spearman"]
    assert np.isfinite(shape_a) and np.isfinite(shape_b)
    print(f"ok 4c: wer_key selects the estimand — 0 dead zones on `wer`, "
          f"{int(on_spoke.sum())} on `wer_spoke`, identical when the two agree; "
          f"a NaN under the named key raises and names it")


def test_ragged_region_coverage_raises():
    """
    `wer_gap` subtracts one model's mean over ITS rows in a region from
    another's mean over ITS rows. If the arms did not run the same cells there,
    the gap is a model effect PLUS a coverage effect and the two cannot be
    separated afterwards — while the ranked region table still looks perfectly
    meaningful. This is the same defect the sim2real arms had.
    """
    tables = build_tables()
    find_divergence_regions(tables, SPACE, n_bins=4)      # matched: no false positive

    ragged = {"strong": tables["strong"],
              "weak": [r for r in tables["weak"] if r["rt60"] > 0.2]}
    try:
        find_divergence_regions(ragged, SPACE, n_bins=4)
        assert False, "expected a ragged-coverage raise"
    except ValueError as exc:
        assert "ragged coverage" in str(exc) and "matched_arms" in str(exc), exc

    try:
        find_divergence_regions({"only-one": tables["strong"]}, SPACE)
        assert False, "expected a single-arm raise"
    except ValueError as exc:
        assert ">= 2 model tables" in str(exc), exc
    print("ok 4b: a region where the arms cover different numbers of cells "
          "raises rather than reporting a confounded WER gap")


# ---- 4c: three arms — the gap is over ALL of them, and coverage is named ----

def test_divergence_over_three_arms_uses_every_arm():
    """
    `wer_gap` is max - min over the arms in a region, and with three arms the
    MIDDLE arm is in neither `worse_model` nor `better_model`. Two things must
    hold: the third arm actually participates in the gap, and an arm with no
    rows in a region is NAMED rather than dropped — "these arms diverge here"
    means something different when a third arm was never measured in that slice.
    """
    tables = build_tables()
    # a middling arm: worse than `strong`, better than `weak` everywhere.
    middle = [dict(r, wer=min(1.0, r["wer"] + 0.12), mean_conf=0.62)
              for r in tables["strong"]]
    three = {"strong": tables["strong"], "middle": middle, "weak": tables["weak"]}

    regions = find_divergence_regions(three, SPACE, n_bins=4)
    assert regions, "no regions produced"
    for r in regions:
        assert set(r["wer_by_model"]) == {"strong", "middle", "weak"}, r
        assert set(r["n_by_model"]) == {"strong", "middle", "weak"}, r
        assert r["models_absent"] == [], r
        vals = [v for v in r["wer_by_model"].values() if np.isfinite(v)]
        assert abs(r["wer_gap"] - (max(vals) - min(vals))) < 1e-12, r
    # `worse_model`/`better_model` name only the extremes, so with three arms
    # there are regions whose middle arm appears in NEITHER field. That is why a
    # consumer must iterate `wer_by_model` rather than read the two names.
    hidden = [r for r in regions
              if "middle" not in (r["worse_model"], r["better_model"])]
    assert hidden, "no region exercised the third-arm-not-an-extreme case"

    # NEGATIVE CONTROL — the third arm must MATTER. Drop the extreme arm and the
    # top region's gap has to shrink; if it did not, this test would pass on an
    # implementation that quietly compared only two arms.
    top3 = regions[0]
    top2 = find_divergence_regions({"strong": tables["strong"], "middle": middle},
                                   SPACE, n_bins=4)[0]
    assert top2["wer_gap"] < top3["wer_gap"] - 1e-9, (top2["wer_gap"], top3["wer_gap"])

    # An arm with NO rows in a whole region is named, not silently omitted. The
    # fixture keeps every arm's per-bin count equal on the continuous factors
    # (so the ragged-coverage guard is not what fires) and differs only in which
    # noise_type levels it covers — arm C never ran `engine`.
    def _cover(model_wer, levels):
        return [{"rt60": rt, "snr_db": snr, "noise_type": lv, "codec": "none",
                 "mic_rolloff": 0.0, "wer": model_wer(rt, lv),
                 "mean_conf": 0.9 - 0.5 * model_wer(rt, lv)}
                for rt in (0.2, 0.6, 1.0) for snr in (0.0, 25.0) for lv in levels]

    absent = {
        "A": _cover(lambda rt, lv: 0.10 + 0.1 * rt, ("babble", "engine")),
        "B": _cover(lambda rt, lv: 0.40 + 0.1 * rt, ("babble", "engine")),
        "C": _cover(lambda rt, lv: 0.25 + 0.1 * rt, ("babble", "road")),
    }
    regions_a = find_divergence_regions(absent, SPACE, n_bins=4)
    by_span = {(r["factor"], str(r["span"])): r for r in regions_a}
    eng = by_span[("noise_type", "engine")]
    assert eng["models_absent"] == ["C"], eng
    assert eng["models_compared"] == ["A", "B"], eng
    assert eng["n_by_model"]["C"] == 0, eng
    # NEGATIVE CONTROL: the level every arm DID run reports nobody absent, so
    # the flag is pinned to the missing coverage rather than to the fixture.
    bab = by_span[("noise_type", "babble")]
    assert bab["models_absent"] == [] and set(bab["models_compared"]) == {"A", "B", "C"}, bab
    print(f"ok 4c: three-arm gap spans all arms (top {top3['wer_gap']:.3f} vs "
          f"{top2['wer_gap']:.3f} for two); an arm absent from a whole region is "
          f"named instead of dropped")


# ---- 4d: WER comparability — an arm property, enforced, with no bypass -----

def _scribe_arm(tables):
    """A third arm named `elevenlabs-scribe`, over the SAME grid as the others.

    Planted so it is a real arm rather than a placeholder: its own confident-
    but-wrong region above rt60 0.7 (so its within-model dead-zone rate is
    NON-zero and its confidence shape is finite), and a WER that makes it the
    worst arm in the low-rt60 bins (so dropping it demonstrably moves the gaps).
    """
    out = []
    for r in tables["strong"]:
        hot = r["rt60"] > 0.7
        wer = 0.55 if hot else min(1.0, r["wer"] + 0.2)
        conf = 0.93 if hot else 0.60 + 0.006 * r["snr_db"]
        out.append(dict(r, wer=wer, mean_conf=conf))
    return out


def test_wer_incomparable_arm_cannot_enter_a_cross_model_wer_comparison():
    """
    Some arms may be measured but NOT ranked by WER against other arms.

    `elevenlabs-scribe`'s orthography is non-deterministic: four repeat calls on
    byte-identical audio returned different transcripts on 5 of 6 probe clips
    (`A7X42` vs `A seven X four two`), worth up to 0.727 strict WER on the SAME
    input. Whisper's formatting offset is a constant (+0.090) and can be
    measured once and subtracted; a per-call draw cannot. A WER gap that is a
    coin flip on identical input is not a measurement of the model.

    The failure this test exists to prevent is not a crash. It is a divergence
    table that comes out complete, ranked and well-formed, whose top region is a
    formatting draw wearing an acoustic result's name. So the cross-model WER
    path RAISES on such an arm, and the only way past is a keyword the caller
    has to type — there is no argument that includes the arm.
    """
    tables = build_tables()
    scribe = _scribe_arm(tables)
    three = {"strong": tables["strong"], "weak": tables["weak"],
             "elevenlabs-scribe": scribe}

    # 1. it RAISES, and the message carries the evidence rather than a rule name
    try:
        find_divergence_regions(three, SPACE, n_bins=4)
        assert False, "a WER-incomparable arm must not enter a WER comparison"
    except WerIncomparableArmError as exc:
        assert "elevenlabs-scribe" in str(exc), exc
        assert "exclude_incomparable=True" in str(exc), exc
        assert "identical" in str(exc).lower(), "the reason must travel with the raise"

    # 2. NEGATIVE CONTROL — pinned to the ARM, not to "three arms" or to the
    #    fixture. Rename the same rows to a comparable arm and it must NOT raise.
    #    Without this the test would pass on an implementation that refused any
    #    third arm, or refused everything.
    renamed = {"strong": tables["strong"], "weak": tables["weak"],
               "some-other-arm": scribe}
    ok_regions = find_divergence_regions(renamed, SPACE, n_bins=4)
    assert ok_regions, "the comparable-arm control produced no regions"
    assert all("some-other-arm" in r["wer_by_model"] for r in ok_regions)
    assert all(not r["models_excluded_wer_incomparable"] for r in ok_regions)

    # 3. with the exclusion typed at the call site it proceeds, WITHOUT the arm,
    #    and every region says so
    kept = find_divergence_regions(three, SPACE, n_bins=4,
                                   exclude_incomparable=True)
    assert kept, "no regions after exclusion"
    for r in kept:
        assert "elevenlabs-scribe" not in r["wer_by_model"], r
        assert set(r["wer_by_model"]) == {"strong", "weak"}, r
        assert r["models_excluded_wer_incomparable"] == ["elevenlabs-scribe"], r
        assert r["wer_comparability"]["comparable"] == ["strong", "weak"], r

    # 4. and the exclusion is REAL: it must reproduce the two-arm scan exactly,
    #    AND differ from the scan that keeps the arm. Without both halves the
    #    mechanism could be a no-op that only edits the labels.
    two_only = find_divergence_regions(
        {"strong": tables["strong"], "weak": tables["weak"]}, SPACE, n_bins=4)
    assert ([round(r["wer_gap"], 12) for r in kept]
            == [round(r["wer_gap"], 12) for r in two_only]), \
        "excluding the arm must give exactly the two-arm comparison"
    with_scribe = find_divergence_regions(renamed, SPACE, n_bins=4)
    assert ([round(r["wer_gap"], 12) for r in with_scribe]
            != [round(r["wer_gap"], 12) for r in kept]), \
        "the excluded arm never affected any gap — the fixture cannot detect a no-op"

    print("ok 4d: a WER-incomparable arm raises in the cross-model WER path, is "
          "dropped only when the caller types exclude_incomparable=True, and an "
          "identically-shaped comparable arm is untouched")


def test_the_within_model_half_still_includes_the_excluded_arm():
    """
    THE OTHER HALF OF THE SCOPE DECISION, and the one that is easy to get wrong
    by over-correcting: excluding an arm from the WER comparison must NOT
    exclude it from the within-model analyses. Its dead-zone rate and its
    confidence-vs-WER shape are computed against its OWN confidence distribution
    and its OWN WER, so nothing is subtracted across arms and the arm is a
    first-class member there.
    """
    tables = build_tables()
    scribe = _scribe_arm(tables)
    three = {"strong": tables["strong"], "weak": tables["weak"],
             "elevenlabs-scribe": scribe}

    # compare_models refuses by default for exactly the same reason...
    try:
        compare_models(three, SPACE, n_bins=4)
        assert False, "compare_models must refuse an incomparable arm by default"
    except WerIncomparableArmError:
        pass

    res = compare_models(three, SPACE, n_bins=4, exclude_incomparable=True)
    # ... and with the exclusion typed, the two halves carry DIFFERENT arm sets
    assert set(res["per_model"]) == {"strong", "weak", "elevenlabs-scribe"}, res["per_model"]
    assert res["models_within"] == list(three), res
    assert res["models_wer_compared"] == ["strong", "weak"], res
    for r in res["divergence_regions"]:
        assert "elevenlabs-scribe" not in r["wer_by_model"], r
    # the within-model numbers are REAL, not placeholders
    dz = res["per_model"]["elevenlabs-scribe"]
    assert np.isfinite(dz["dead_zone_rate"]), dz
    assert np.isfinite(dz["shape"]["spearman"]), dz
    assert dz["n"] == len(scribe), dz

    # NEGATIVE CONTROL: the within-model numbers must be BIT-IDENTICAL to what
    # the arm gets computed entirely on its own rows, with no other arm in the
    # room. That is the claim "the exclusion changed nothing within the arm"
    # made executable, and it would fail if the exclusion had leaked into the
    # within-model half (e.g. by dropping the arm's rows before flagging).
    solo_rate = float(np.mean(dead_zone_flags(scribe, 0.3, 0.6)))
    solo_shape = confidence_wer_shape(scribe)["spearman"]
    assert solo_rate == dz["dead_zone_rate"], (solo_rate, dz["dead_zone_rate"])
    assert solo_shape == dz["shape"]["spearman"], (solo_shape, dz["shape"])
    assert dz["dead_zone_rate"] > 0.0, ("the fixture must plant a real dead zone, "
                                        "or 'the numbers are real' is untested")

    print(f"ok 4e: the excluded arm keeps a real within-model dead-zone rate "
          f"({dz['dead_zone_rate']:.3f}) and shape "
          f"({dz['shape']['spearman']:+.3f}), bit-identical to analysing it "
          f"alone, while being absent from every WER gap")


def test_comparability_is_a_registry_property_not_a_hardcoded_name():
    """
    GENERALITY. The next arm with unstable formatting must be handled by adding
    one line to `WER_INCOMPARABLE_ARMS`, not by editing a conditional. Register a
    fake arm, prove the same refusal fires, then unregister and prove it stops.
    """
    tables = build_tables()
    fake = {"strong": tables["strong"], "weak": tables["weak"],
            "brand-new-arm": [dict(r) for r in tables["strong"]]}

    assert is_wer_comparable("brand-new-arm")
    find_divergence_regions(fake, SPACE, n_bins=4)      # comparable -> fine

    WER_INCOMPARABLE_ARMS["brand-new-arm"] = "planted for the test"
    try:
        assert not is_wer_comparable("brand-new-arm")
        try:
            find_divergence_regions(fake, SPACE, n_bins=4)
            assert False, "registering an arm must make the WER path refuse it"
        except WerIncomparableArmError as exc:
            assert "brand-new-arm" in str(exc) and "planted for the test" in str(exc)
        cen = wer_comparability(list(fake))
        assert cen["excluded"] == ["brand-new-arm"], cen
        assert "brand-new-arm" in cen["statement"], cen["statement"]
    finally:
        WER_INCOMPARABLE_ARMS.pop("brand-new-arm", None)
    # restored: the refusal is a property of the registry, not of the code path
    assert is_wer_comparable("brand-new-arm")
    find_divergence_regions(fake, SPACE, n_bins=4)

    # every registered exclusion must carry a substantive reason — an exclusion
    # whose justification lives in a commit message gets "fixed" by the next
    # person who reads the code.
    for arm, reason in WER_INCOMPARABLE_ARMS.items():
        assert isinstance(reason, str) and len(reason) > 120, (arm, reason)

    # The chokepoint refuses to report on fewer than two SURVIVING arms. This is
    # the case that defeats every other check: after the exclusion the remaining
    # tables are internally consistent, equal-sized and perfectly well-formed —
    # there is just nothing left to compare them to.
    try:
        wer_comparable_tables({"elevenlabs-scribe": [], "nova-3": []},
                              exclude_incomparable=True)
        assert False, "one surviving arm is not a comparison"
    except WerIncomparableArmError as exc:
        assert ">= 2" in str(exc) and "elevenlabs-scribe" in str(exc), exc
    # NEGATIVE CONTROL: add a second comparable arm and the same call succeeds,
    # so the raise is pinned to the survivor COUNT, not to the excluded arm
    # merely being present.
    kept, cen = wer_comparable_tables(
        {"elevenlabs-scribe": [], "nova-3": [], "whisper-base": []},
        exclude_incomparable=True)
    assert set(kept) == {"nova-3", "whisper-base"}, sorted(kept)
    assert cen["excluded"] == ["elevenlabs-scribe"], cen

    print(f"ok 4f: WER comparability is a registry property — "
          f"{sorted(WER_INCOMPARABLE_ARMS)} — and a planted arm is refused and "
          f"un-refused by editing the registry alone")


# ---- 5: the third adapter arm parses to the shared contract (offline) ------

def test_vosk_parses_to_contract():
    canned = {"text": "call maria at four zero five",
              "result": [{"word": "call",  "conf": 0.98, "start": 0.1, "end": 0.3},
                         {"word": "maria", "conf": 0.91, "start": 0.3, "end": 0.9},
                         {"word": "at",    "conf": 0.88, "start": 0.9, "end": 1.0},
                         {"word": "four",  "conf": 0.95, "start": 1.0, "end": 1.2},
                         {"word": "zero",  "conf": 0.93, "start": 1.2, "end": 1.4},
                         {"word": "five",  "conf": 0.90, "start": 1.4, "end": 1.7}]}
    out = _parse_vosk_result(canned)
    assert set(out) == {"transcript", "word_confidences", "mean_conf", "utterance_conf"}
    assert out["transcript"] == "call maria at four zero five"
    assert len(out["word_confidences"]) == 6
    assert abs(out["mean_conf"] - float(np.mean([0.98, 0.91, 0.88, 0.95, 0.93, 0.90]))) < 1e-9
    assert out["utterance_conf"] == out["mean_conf"]     # Vosk has no separate one
    # honest empty case: no words -> no fabricated confidences
    empty = _parse_vosk_result({"text": ""})
    assert empty["word_confidences"] == [] and np.isnan(empty["mean_conf"])
    print("OK 5: Vosk arm parses to the shared contract (per-word conf surfaced, "
          "no fabrication on empty)")


if __name__ == "__main__":
    test_registry()
    test_within_model_normalization()
    test_comparison_surfaces_weak_region()
    test_dead_zone_and_shape()
    test_nan_wer_never_dilutes_the_dead_zone_rate()
    test_wer_key_selects_the_estimand()
    test_ragged_region_coverage_raises()
    test_divergence_over_three_arms_uses_every_arm()
    test_wer_incomparable_arm_cannot_enter_a_cross_model_wer_comparison()
    test_the_within_model_half_still_includes_the_excluded_arm()
    test_comparability_is_a_registry_property_not_a_hardcoded_name()
    test_vosk_parses_to_contract()
    print("\nAll model-compare tests passed — cross-model dead-zone comparison "
          "works within-model (scale-free) and surfaces the planted weak region.")
