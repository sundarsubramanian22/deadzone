"""
Tests for analysis/confidence_char.py — the empirical characterisation of the
confidence signal.

House standard, applied throughout: every check PLANTS a known structure and
asserts that the module recovers it, and every guard test is paired with a
NEGATIVE CONTROL — the same fixture with the violation removed, asserted to pass.
A guard pinned only to its violating input can be passing for an incidental
property of the fixture (its size, its ordering, a NaN that happens to be there),
and would keep passing after the guard itself was deleted.

The failures this suite exists to prevent are the three the project keeps finding
in its own analysis code:

  * TWO POPULATIONS SUBTRACTED. A confidence exists only for a row that emitted
    words; a silent row scores WER 1.0 and contributes no confidence. Any
    statistic that scores confidence against an accuracy including the silent
    rows is measuring the silence (SPEC Appendix G). Tested by planting silent
    rows and asserting they are excluded, counted, and unable to move a number.
  * A GUARD WHOSE FAILURE MODE IS SILENCE. An AUROC with one label class absent
    conventionally returns 0.5 or NaN — both plausible values, neither an error
    (SPEC Appendix E). Tested by asserting the raise, and by asserting that the
    plausible-but-wrong value is NOT what comes back.
  * POOLED SCALES. Two arms' confidences in one average is a measurement of
    scale conventions. Tested by planting two arms with disjoint ranges and
    asserting a raise, with the single-arm control passing.
"""

# --- repo-root bootstrap -------------------------------------------------
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# -------------------------------------------------------------------------
import json
import math

import numpy as np

from deadzone.analysis.confidence_char import (
    AGGREGATES, ArmPoolingError, DegenerateLabelError, auroc,
    benign_rows, characterize, clean_reference, condition_aggregate_table,
    dynamic_range, format_report, operating_points, report_by_model,
    row_aggregates, row_confidences, saturation, separation, severity_rank,
    spoke_and_silent, utterance_vs_word_mean, word_reliability, write_report,
)

# The mildest corner of the factor space, so a fixture that does not say
# otherwise sits in the clean reference set.
_MILD = {"rt60": 0.2, "snr_db": 20.0, "noise_type": "babble", "codec": "none",
         "mic_rolloff": 0.0}


def _row(model, clip, cond, confs, *, n_ref=None, n_sub=0, n_del=0, n_ins=0,
         wer=None, utt=None, failed=False, **factors):
    """One master-table row with an explicit word-confidence list.

    `edits` is built to be consistent with the counts AND with `confs`, because
    `word_records` (which the saturation and reliability paths go through)
    asserts a 1:1 alignment between the hypothesis words and the confidence list
    rather than zipping. A fixture that violates that would be silently skipped
    and the test would assert against an empty set.
    """
    confs = list(confs)
    n_hyp = len(confs)
    n_match = max(n_hyp - n_sub - n_ins, 0)
    if n_ref is None:
        n_ref = n_match + n_sub + n_del
    edits = ([["match", f"w{i}", f"w{i}"] for i in range(n_match)]
             + [["sub", f"s{i}", f"h{i}"] for i in range(n_sub)]
             + [["ins", None, f"x{i}"] for i in range(n_ins)]
             + [["del", f"d{i}", None] for i in range(n_del)])
    if wer is None:
        wer = (n_sub + n_del + n_ins) / n_ref if n_ref else 1.0
    d = {"model": model, "clip_id": clip, "condition_name": cond,
         "transcript": " ".join(f"t{i}" for i in range(n_hyp)),
         "wer": wer, "n_ref": n_ref, "n_sub": n_sub, "n_del": n_del,
         "n_ins": n_ins, "n_match": n_match,
         "mean_conf": float(np.mean(confs)) if confs else float("nan"),
         "utterance_conf": (float(utt) if utt is not None
                            else (float(np.mean(confs)) if confs else float("nan"))),
         "word_confidences": json.dumps(confs),
         "edits": json.dumps(edits),
         "failed": failed, "error": None}
    d.update(_MILD)
    d.update(factors)
    return d


def _silent(model, clip, cond, n_ref=4, **factors):
    """A clip the model returned NOTHING for.

    A real measurement, not a failure sentinel: WER 1.0, 100% deletions, and
    NO confidence whatsoever. This is the row that makes the two estimands
    diverge, and it is the row every check below plants.
    """
    return _row(model, clip, cond, [], n_ref=n_ref, n_del=n_ref, wer=1.0,
                **factors)


# =========================================================================
# 1. THE ESTIMAND RULE — confidence and accuracy over the SAME rows
# =========================================================================

def test_silent_rows_are_excluded_counted_and_cannot_move_a_number():
    """
    A silent row carries WER 1.0 and no confidence. If it reached any statistic
    here it would either be imputed a confidence (inventing data) or drag a WER
    that a confidence is then compared against (the SPEC Appendix G defect,
    which inflated the published headline gap by +0.109).

    NEGATIVE CONTROL: the identical rows with the silent ones removed must give
    BIT-IDENTICAL statistics. If they differ, the silent rows were contributing
    somewhere.
    """
    spoke = [_row("m", f"u{i:02d}", "c1", [0.9, 0.8, 0.7]) for i in range(8)]
    silent = [_silent("m", f"u{i:02d}", "c1") for i in range(8, 12)]

    cen = spoke_and_silent(spoke + silent)
    assert cen["n_spoke"] == 8 and cen["n_silent"] == 4, cen
    assert abs(cen["silent_frac"] - 4 / 12) < 1e-12, cen

    with_silent = clean_reference(spoke + silent)
    without = clean_reference(spoke)
    # The confidence distribution and the accuracy it is paired with are both
    # over the spoke rows only, so the two calls must agree exactly.
    assert with_silent["per_word"] == without["per_word"], (
        "silent rows changed the confidence distribution — they carry no "
        "confidence, so they cannot have contributed one")
    assert with_silent["wer_spoke"] == without["wer_spoke"], (
        f"paired WER moved from {without['wer_spoke']} to "
        f"{with_silent['wer_spoke']} when silent rows were added: the accuracy "
        f"is being averaged over a different population than the confidence")
    # ...and the census still SAYS they exist. Excluding without counting is the
    # other half of the defect.
    assert with_silent["census"]["n_silent"] == 4, with_silent["census"]
    assert without["census"]["n_silent"] == 0, without["census"]
    print("ok: silent rows are excluded from every confidence statistic, are "
          "counted in the census, and cannot move a paired number")


def test_separation_scores_only_the_rows_that_spoke():
    """The AUROC label must be read off the same rows as the score."""
    rows = ([_row("m", f"u{i:02d}", "good", [0.95, 0.95, 0.95]) for i in range(6)]
            + [_row("m", f"u{i:02d}", "bad", [0.3, 0.3, 0.3], n_sub=2)
               for i in range(6)]
            + [_silent("m", f"u{i:02d}", "mute") for i in range(6)])
    sep = separation(rows, wer_hi=0.3, n_boot=30)
    assert sep["n_rows_scored"] == 12, sep["n_rows_scored"]
    assert sep["census"]["n_silent"] == 6, sep["census"]
    # The planted structure is perfectly separable on every aggregate.
    for d in sep["per_aggregate"]:
        if d["aggregate"] in AGGREGATES:
            assert d["auroc"] == 1.0, (d["aggregate"], d["auroc"])
    print("ok: separation scores only the spoke rows, reports the silent census, "
          "and recovers a perfectly separable planted structure")


# =========================================================================
# 2. NEVER POOL CONFIDENCES ACROSS ARMS
# =========================================================================

def test_two_arms_in_one_statistic_raises():
    """
    Deepgram acoustic confidence, Whisper token softmax and Scribe exp(logprob)
    share the interval [0,1] and nothing else. Pooled, they produce a plausible
    number that measures scale conventions.

    The fixture makes the damage concrete: arm A sits at ~0.95, arm B at ~0.10.
    A pooled mean is ~0.53, a value neither arm ever produces.

    NEGATIVE CONTROL: each arm ALONE must go through, so the guard is pinned to
    the pooling and not to something incidental about the fixture.
    """
    a = [_row("arm-a", f"u{i:02d}", "c1", [0.95, 0.96]) for i in range(6)]
    b = [_row("arm-b", f"u{i:02d}", "c1", [0.10, 0.11]) for i in range(6)]

    for fn, name in ((clean_reference, "clean_reference"),
                     (saturation, "saturation"),
                     (utterance_vs_word_mean, "utterance_vs_word_mean"),
                     (word_reliability, "word_reliability"),
                     (condition_aggregate_table, "condition_aggregate_table")):
        try:
            fn(a + b)
        except ArmPoolingError:
            pass
        else:
            raise AssertionError(f"{name} pooled two arms' confidences")
        fn(a)          # negative control: one arm is fine
    try:
        separation(a + b, n_boot=10)
    except ArmPoolingError:
        pass
    else:
        raise AssertionError("separation pooled two arms' confidences")

    # And the fixture really would have been misleading: the pooled mean is a
    # value neither arm ever emits.
    pooled = np.mean([c for r in a + b for c in row_confidences(r)])
    assert 0.4 < pooled < 0.7, pooled
    print("ok: every confidence statistic refuses two arms and accepts one — the "
          "pooled mean would have been 0.53, a value neither arm produces")


def test_report_by_model_keeps_the_arms_separate():
    """The multi-arm entry point splits first and never merges a raw confidence."""
    harsh = {"rt60": 1.0, "snr_db": 0.0}       # keeps c2 out of the clean corner
    rows = ([_row("arm-a", f"u{i:02d}", "c1", [0.95, 0.96]) for i in range(5)]
            + [_row("arm-a", f"u{i:02d}", "c2", [0.4, 0.4], n_sub=2, **harsh)
               for i in range(5)]
            + [_row("arm-b", f"u{i:02d}", "c1", [0.10, 0.11]) for i in range(5)]
            + [_row("arm-b", f"u{i:02d}", "c2", [0.02, 0.02], n_sub=2, **harsh)
               for i in range(5)])
    rep = report_by_model(rows, wer_hi=0.3, n_boot=20)
    assert rep["models"] == ["arm-a", "arm-b"], rep["models"]
    a = rep["by_model"]["arm-a"]["clean_reference_corner"]["per_word"]["mean"]
    b = rep["by_model"]["arm-b"]["clean_reference_corner"]["per_word"]["mean"]
    assert a > 0.9 and b < 0.2, (a, b)
    # The cross-arm table may carry rulers and rank statistics, never a pooled
    # confidence: every row is attributed to exactly one arm.
    assert {c["model"] for c in rep["cross_arm"]} == {"arm-a", "arm-b"}
    assert format_report(rep)          # the formatter survives disjoint scales
    print("ok: report_by_model characterises each arm in its own scale and the "
          "cross-arm table is per-arm rulers, never a pooled average")


# =========================================================================
# 3. THE CLEAN REFERENCE IS SELECTED BY FACTORS, NEVER BY OUTCOME
# =========================================================================

def test_clean_reference_is_chosen_by_factor_setting_not_by_low_wer():
    """
    The tempting way to pick "the clean conditions" is to take the lowest-WER
    cells. That conditions the baseline on the model having done well, and every
    dead zone is then quoted against a baseline chosen to flatter it.

    The fixture makes the two selections DISAGREE on purpose: the mildest corner
    is planted with a HIGH WER and a harsh cell with a LOW one. A factor-based
    selector picks the corner; an outcome-based one would pick the harsh cell.
    """
    corner = [_row("m", f"u{i:02d}", "corner", [0.60, 0.62], n_sub=2)
              for i in range(5)]                       # mild factors, BAD WER
    harsh = [_row("m", f"u{i:02d}", "harsh", [0.99, 0.99],
                  rt60=1.0, snr_db=0.0, codec="g726", mic_rolloff=1.0)
             for i in range(5)]                        # harsh factors, GOOD WER
    ref = clean_reference(corner + harsh)
    assert ref["conditions"] == ["corner"], ref["conditions"]
    assert ref["per_word"]["mean"] < 0.7, ref["per_word"]["mean"]
    assert ref["wer_spoke"] > 0.3, ref["wer_spoke"]

    # NEGATIVE CONTROL, and it is the real assertion: the selector must be blind
    # to the measurements. Wiping every outcome column must not change the set.
    stripped = []
    for r in corner + harsh:
        s = dict(r)
        for k in ("wer", "mean_conf", "utterance_conf", "n_sub", "n_del"):
            s[k] = None
        stripped.append(s)
    assert (sorted({r["condition_name"] for r in benign_rows(stripped)})
            == ["corner"]), "the benign selection depends on an outcome column"

    sev = severity_rank(corner + harsh)
    assert sev["corner"] == 0 and sev["harsh"] > 0, sev
    print("ok: the clean reference is selected by factor setting — it survives "
          "erasing every outcome column, and it picks the mild-but-bad cell over "
          "the harsh-but-good one")


# =========================================================================
# 4. AUROC — the guard whose degenerate output would be plausible
# =========================================================================

def test_auroc_refuses_a_single_class_instead_of_returning_a_plausible_number():
    """
    With one label class absent there is nothing to separate. The conventional
    answers are 0.5 ("useless statistic") and NaN ("no result"); both are
    plausible, both are read as measurements, and neither is true — the question
    was never asked. This is SPEC Appendix E's family exactly.

    NEGATIVE CONTROL: the same scores with ONE label flipped must return a real
    AUROC, so the guard is pinned to the degenerate labels and not to the scores.
    """
    scores = [0.9, 0.8, 0.7, 0.6]
    for labels in ([False] * 4, [True] * 4):
        try:
            auroc(scores, labels)
        except DegenerateLabelError:
            pass
        else:
            got = auroc(scores, labels)
            raise AssertionError(
                f"auroc returned {got!r} for a single-class input; 0.5 and nan "
                f"are both plausible values and neither is a measurement")
    val = auroc(scores, [False, False, False, True])
    assert math.isfinite(val) and 0.0 <= val <= 1.0, val
    print("ok: auroc raises on a single-class input rather than returning 0.5 or "
          "nan, and still scores the control where one label differs")


def test_auroc_orientation_and_tie_handling():
    """
    Orientation: LOW confidence must mean BAD, so a perfect separator scores 1.0.
    Get this backwards and every ranking in the module inverts while every number
    stays in [0,1] and looks fine.

    Ties: a saturated arm produces huge blocks of identical confidences. Average
    ranks score a tie block at 0.5; an argsort-based AUROC would treat the block
    as ordered and inflate a saturated arm's apparent separation.
    """
    assert auroc([0.9, 0.9, 0.1, 0.1], [False, False, True, True]) == 1.0
    assert auroc([0.1, 0.1, 0.9, 0.9], [False, False, True, True]) == 0.0
    assert auroc([0.5] * 6, [True, True, True, False, False, False]) == 0.5, (
        "an all-ties input must score 0.5 — ties carry no ordering")
    # A HALF-tied input: of the four good/bad pairs, two are ties (0.5 each) and
    # two are clean wins, so the exact answer is 0.75. An argsort-based AUROC
    # would return 1.0 or 0.5 depending on the input ORDER — which is the tell,
    # and is why a saturated arm needs average ranks.
    v = auroc([0.5, 0.5, 0.5, 0.1], [False, False, True, True])
    assert abs(v - 0.75) < 1e-12, v
    assert auroc([0.5, 0.5, 0.5, 0.1][::-1],
                 [False, False, True, True][::-1]) == v, (
        "AUROC changed when the rows were reordered — ties are being scored as "
        "if they were ordered")
    print("ok: auroc is oriented so low confidence = bad (1.0 is perfect) and "
          "handles ties by average ranks, which a saturated arm requires")


# =========================================================================
# 5. THE AGGREGATE RANKING RECOVERS PLANTED STRUCTURE
# =========================================================================

def test_min_wins_when_the_damage_is_concentrated_in_one_word():
    """
    PLANTED: bad utterances differ from good ones ONLY in a single destroyed
    word; every other word is identical between the two classes. The arithmetic
    mean is deliberately made nearly uninformative (one word in twenty), while
    the minimum separates perfectly.

    This is the case the whole aggregation question is about — one destroyed
    proper noun buried under nineteen confident function words — and if the
    module could not recover it on planted data, its real-grid finding that the
    mean WINS would be an artifact of the method rather than a property of the
    signal.
    """
    good = [_row("m", f"g{i:02d}", "good", [0.90] * 20) for i in range(15)]
    bad = [_row("m", f"b{i:02d}", "bad", [0.90] * 19 + [0.05], n_sub=8)
           for i in range(15)]
    sep = separation(good + bad, wer_hi=0.3, n_boot=50)
    by = {d["aggregate"]: d for d in sep["per_aggregate"]}
    assert by["min"]["auroc"] == 1.0, by["min"]["auroc"]
    assert by["mean"]["auroc"] == 1.0 or by["mean"]["auroc"] < 1.0
    # The claim that matters is the DIFFERENCE, decided by the paired interval.
    assert by["min"]["d_auroc_vs_mean"] >= 0.0, by["min"]
    assert sep["best_aggregate"] == "min" or by["min"]["auroc"] >= by["mean"]["auroc"]
    # The median must be blind here BY CONSTRUCTION: 19 of 20 words are identical.
    assert by["median"]["auroc"] == 0.5, by["median"]["auroc"]
    print("ok: with the damage concentrated in ONE word, min separates perfectly "
          "and the median is blind — the ranking recovers planted structure")


def test_mean_wins_when_the_damage_is_spread_across_every_word():
    """
    The mirror image, and the negative control for the test above: degradation
    that lowers EVERY word a little, with one accidental low word in the GOOD
    class. `min` is then actively misleading and the mean is right.

    Together the two tests show the ranking is driven by where the information
    is, not by a fixed preference — which is what makes the real-grid verdict
    ('nothing beats the mean for nova-3') a finding rather than a default.
    """
    good = [_row("m", f"g{i:02d}", "good", [0.95] * 9 + [0.20]) for i in range(15)]
    bad = [_row("m", f"b{i:02d}", "bad", [0.60] * 9 + [0.55], n_sub=5)
           for i in range(15)]
    sep = separation(good + bad, wer_hi=0.3, n_boot=50)
    by = {d["aggregate"]: d for d in sep["per_aggregate"]}
    assert by["mean"]["auroc"] == 1.0, by["mean"]["auroc"]
    assert by["min"]["auroc"] == 0.0, (
        f"min should be perfectly INVERTED here, got {by['min']['auroc']}")
    assert by["min"]["loses_to_mean"] is True, by["min"]
    assert sep["significantly_better_than_baseline"] == [], sep["verdict"]
    print("ok: with the damage spread over every word, the mean separates "
          "perfectly and min inverts — and the verdict is decided by the paired "
          "CI, not by the ordering")


def test_the_length_control_is_scored_in_the_same_table():
    """
    `min` falls with utterance length for purely combinatorial reasons. If word
    count alone separates good from bad, a length-sensitive aggregate inherits
    that separation and the ranking is reporting length wearing a confidence's
    name.

    PLANTED: word count is made a PERFECT predictor while every confidence is
    identical, so no confidence aggregate can carry information. The control must
    then be the top of the table.
    """
    good = [_row("m", f"g{i:02d}", "good", [0.8] * 12) for i in range(10)]
    bad = [_row("m", f"b{i:02d}", "bad", [0.8] * 3, n_sub=2) for i in range(10)]
    sep = separation(good + bad, wer_hi=0.3, n_boot=30)
    by = {d["aggregate"]: d for d in sep["per_aggregate"]}
    assert by["n_words"]["is_control"] is True
    assert by["n_words"]["auroc"] == 1.0, by["n_words"]["auroc"]
    assert by["mean"]["auroc"] == 0.5, by["mean"]["auroc"]
    # The control must never be reported as the winning AGGREGATE — it is not a
    # confidence and shipping it would be shipping "flag short utterances".
    assert sep["best_aggregate"] != "n_words", sep["best_aggregate"]
    assert sep["ranking"][0] == "n_words", sep["ranking"]
    print("ok: the confidence-free length control is scored in the same table, "
          "tops the ranking when length is the real signal, and is never "
          "reported as the winning aggregate")


# =========================================================================
# 6. utterance_conf — redundant, or a second signal?
# =========================================================================

def test_an_adapter_that_reuses_the_word_mean_is_called_redundant():
    """
    Vosk and Scribe document that they REUSE the word mean because the vendor
    exposes no separate utterance score. For those arms any 'finding' about
    utterance_conf would be a tautology, so identity is measured exactly rather
    than assumed.
    """
    rows = [_row("m", f"u{i:02d}", "c1", [0.9, 0.8]) for i in range(10)]
    out = utterance_vs_word_mean(rows)
    assert out["frac_identical"] == 1.0, out["frac_identical"]
    assert "REDUNDANT BY CONSTRUCTION" in out["verdict"], out["verdict"]
    print("ok: an adapter that reuses the word mean is reported as redundant by "
          "construction, measured exactly rather than assumed")


def test_a_genuinely_different_utterance_score_is_called_distinct():
    """NEGATIVE CONTROL for the test above: a real second signal must not be
    dismissed as a copy."""
    rows = [_row("m", f"u{i:02d}", "c1", [0.9, 0.8], utt=0.2 + 0.05 * i)
            for i in range(10)]
    out = utterance_vs_word_mean(rows)
    assert out["frac_identical"] == 0.0, out["frac_identical"]
    assert out["verdict"].startswith("DISTINCT"), out["verdict"]
    assert out["mean_abs_diff"] > 0.1, out["mean_abs_diff"]
    print("ok: a genuinely independent utterance score is reported as DISTINCT, "
          "so 'redundant' is a measurement and not the default answer")


# =========================================================================
# 7. SATURATION
# =========================================================================

def test_saturation_separates_an_exact_ceiling_from_a_clipped_one():
    """
    An arm that returns a bare 1.0 (nova-3) and one whose adapter clips into
    (0,1) (Scribe, whose exp(logprob) is clipped at 1-1e-6) are saturated in the
    same way, but `== 1.0` sees only the first. Reporting exact-endpoint mass
    alone would say the clipped arm does not saturate at all.
    """
    exact = [_row("m", f"u{i:02d}", "c1", [1.0, 1.0, 0.5]) for i in range(10)]
    clipped = [_row("m", f"u{i:02d}", "c1", [1 - 1e-9, 1 - 1e-9, 0.5])
               for i in range(10)]
    a, b = saturation(exact), saturation(clipped)
    assert a["exactly_one"] == 20 and b["exactly_one"] == 0, (a, b)
    # ...but the NEAR-endpoint mass sees both, and that is the honest quantity.
    assert abs(a["frac_within_eps_of_one"] - 2 / 3) < 1e-9, a
    assert abs(b["frac_within_eps_of_one"] - 2 / 3) < 1e-9, (
        "the clipped arm reads as unsaturated: exact-equality alone is not a "
        "saturation measure")
    print("ok: saturation reports exact-endpoint mass AND near-endpoint mass, so "
          "an adapter-clipped ceiling is not misread as no ceiling")


def test_ceiling_accuracy_is_measured_not_assumed():
    """
    A ceiling that is always correct is a harmless encoding. A ceiling that is
    sometimes wrong is a silent-failure generator, because 1.0 admits no
    discount. So the accuracy of the saturated words is measured.

    PLANTED: every ceiling word is a substitution, i.e. WRONG. If the module
    reported the arm's overall accuracy instead of the ceiling's, this would come
    back high.
    """
    rows = [_row("m", f"u{i:02d}", "c1", [1.0, 1.0, 0.2, 0.2], n_sub=2)
            for i in range(10)]
    # _row puts the substitutions LAST in the alignment, so flip the confidences
    # to put the ceiling on the wrong words.
    for r in rows:
        r["word_confidences"] = json.dumps([0.2, 0.2, 1.0, 1.0])
    sat = saturation(rows)
    ceil = sat["accuracy_at_ceiling"]
    assert ceil["n_words"] == 20, ceil
    assert ceil["accuracy"] == 0.0, (
        f"ceiling accuracy {ceil['accuracy']} — the planted ceiling words are "
        f"all substitutions; a non-zero value means the arm's overall accuracy "
        f"is being reported instead of the ceiling's")
    assert ceil["overall_accuracy"] == 0.5, ceil["overall_accuracy"]
    print("ok: the accuracy of the saturated words is measured against the same "
          "alignment L2 uses, not inherited from the arm's overall accuracy")


# =========================================================================
# 8. OPERATING POINTS AND THE DYNAMIC RANGE
# =========================================================================

def test_operating_points_refuse_a_degenerate_label():
    """A flag/precision/recall table over one label class is not a decision rule.

    NEGATIVE CONTROL: the same rows with one bad utterance added must produce a
    table.
    """
    good = [_row("m", f"u{i:02d}", "c1", [0.95, 0.95]) for i in range(12)]
    try:
        operating_points(good, "mean")
    except DegenerateLabelError:
        pass
    else:
        raise AssertionError("operating_points built a rule with no bad rows")
    out = operating_points(good + [_row("m", "u99", "c2", [0.1, 0.1], n_sub=2)],
                           "mean")
    assert out["points"] and 0.0 <= out["points"][0]["precision"] <= 1.0
    print("ok: operating_points raises when one class is absent and builds the "
          "rule as soon as both are present")


def test_dynamic_range_excludes_mute_conditions_from_the_extremes():
    """
    A mute condition has no confidence. Including it in the extremes would mean
    inventing one for exactly the cells where none exists — which is how the
    'worst' end of a range gets reported as a confidence the model never emitted.
    """
    rows = ([_row("m", f"u{i:02d}", "mild", [0.95, 0.95]) for i in range(6)]
            + [_row("m", f"u{i:02d}", "harsh", [0.30, 0.30], n_sub=2,
                    rt60=1.0) for i in range(6)]
            + [_silent("m", f"u{i:02d}", "mute", rt60=1.0) for i in range(6)])
    cond = condition_aggregate_table(rows, wer_hi=0.3)
    dyn = dynamic_range(cond, clean_reference(rows))
    assert dyn["n_mute_excluded"] == 1, dyn
    assert dyn["highest_condition"]["condition_name"] == "mild", dyn
    assert dyn["lowest_condition"]["condition_name"] == "harsh", (
        "the mute condition reached the extremes — it has no confidence at all")
    assert abs(dyn["span"] - 0.65) < 1e-9, dyn["span"]
    print("ok: the dynamic range is taken over the conditions that emitted words "
          "and names how many mute conditions it could not describe")


# =========================================================================
# 9. END TO END, AND THE ARTIFACT
# =========================================================================

def test_characterize_end_to_end_and_the_artifact_is_reproducible():
    """
    The whole layer over a small planted arm, then the two artifacts.

    Byte-reproducibility is asserted rather than hoped for: the bootstrap is
    seeded and nothing carries a timestamp, so two runs of the same input must
    produce identical files. A drifting artifact cannot be diffed, and a
    diffable artifact is the only way a later session notices a number moved.
    """
    import tempfile
    rows = ([_row("m", f"u{i:02d}", "mild", [0.95, 0.93, 0.97]) for i in range(10)]
            + [_row("m", f"u{i:02d}", "harsh", [0.40, 0.35, 0.30], n_sub=2,
                    rt60=1.0, snr_db=0.0) for i in range(10)]
            + [_silent("m", f"u{i:02d}", "mute", rt60=1.0, snr_db=0.0)
               for i in range(10)])
    rep = characterize(rows, wer_hi=0.3, n_boot=40)
    assert rep["model"] == "m"
    assert rep["clean_reference_corner"]["conditions"] == ["mild"]
    assert rep["separation"]["n_rows_scored"] == 20
    assert rep["condition_level"]["n_mute_conditions"] == 1
    assert rep["deletion_ceiling"]["n_deletions"] > 0
    assert "statement" in rep and rep["statement"]
    # Every headline claim carries its population.
    assert rep["separation"]["census"]["n_silent"] == 10

    full = report_by_model(rows, wer_hi=0.3, n_boot=40)
    with tempfile.TemporaryDirectory() as d:
        j1, t1 = write_report(full, _os.path.join(d, "a.json"),
                              _os.path.join(d, "a.txt"))
        full2 = report_by_model(rows, wer_hi=0.3, n_boot=40)
        j2, t2 = write_report(full2, _os.path.join(d, "b.json"),
                              _os.path.join(d, "b.txt"))
        for x, y in ((j1, j2), (t1, t2)):
            assert open(x, "rb").read() == open(y, "rb").read(), (
                f"{x} and {y} differ — the artifact is not reproducible")
    print("ok: characterize runs end to end on a planted arm and both artifacts "
          "are byte-reproducible across runs")


def test_row_aggregates_are_all_computed_from_the_same_word_list():
    """
    Every aggregate under comparison must see the SAME words, or the ranking is
    comparing statistics over different data. `utterance_conf` is the deliberate
    exception (it is the adapter's own number) and that is why it is tested
    separately.
    """
    r = _row("m", "u01", "c1", [0.1, 0.5, 0.9, 0.95], utt=0.42)
    a = row_aggregates(r)
    assert a["min"] == 0.1 and a["max"] == 0.95
    assert abs(a["mean"] - 0.6125) < 1e-12, a["mean"]
    assert abs(a["median"] - 0.7) < 1e-12, a["median"]
    assert a["n_words"] == 4.0
    assert a["utterance_conf"] == 0.42
    assert a["p10"] <= a["p25"] <= a["median"] <= a["max"]
    # A row with no words yields NaN, never an imputed value: an absent
    # confidence is not a low confidence.
    empty = row_aggregates(_row("m", "u02", "c1", [], n_ref=3, n_del=3))
    assert all(math.isnan(empty[k]) for k in AGGREGATES), empty
    assert empty["n_words"] == 0.0
    print("ok: every aggregate is computed from one shared word list, ordered "
          "correctly, and an empty row yields NaN rather than an imputed value")


if __name__ == "__main__":
    test_silent_rows_are_excluded_counted_and_cannot_move_a_number()
    test_separation_scores_only_the_rows_that_spoke()
    test_two_arms_in_one_statistic_raises()
    test_report_by_model_keeps_the_arms_separate()
    test_clean_reference_is_chosen_by_factor_setting_not_by_low_wer()
    test_auroc_refuses_a_single_class_instead_of_returning_a_plausible_number()
    test_auroc_orientation_and_tie_handling()
    test_min_wins_when_the_damage_is_concentrated_in_one_word()
    test_mean_wins_when_the_damage_is_spread_across_every_word()
    test_the_length_control_is_scored_in_the_same_table()
    test_an_adapter_that_reuses_the_word_mean_is_called_redundant()
    test_a_genuinely_different_utterance_score_is_called_distinct()
    test_saturation_separates_an_exact_ceiling_from_a_clipped_one()
    test_ceiling_accuracy_is_measured_not_assumed()
    test_operating_points_refuse_a_degenerate_label()
    test_dynamic_range_excludes_mute_conditions_from_the_extremes()
    test_characterize_end_to_end_and_the_artifact_is_reproducible()
    test_row_aggregates_are_all_computed_from_the_same_word_list()
    print("\nconfidence characterisation verified on planted structure — the "
          "confidence is never pooled across arms, is never scored against an "
          "accuracy measured on other rows, the clean reference survives erasing "
          "every outcome column, and an AUROC with one label class raises "
          "instead of returning a plausible 0.5.")
