"""
Tests for the L1 multi-model comparison layer.

Offline and synthetic: every row here is fabricated with a KNOWN structure, so
the test asserts that the analysis recovers what was planted rather than that it
runs without crashing. The failure this suite exists to prevent is a comparison
that silently mixes a model effect with a coverage effect, or that launders one
arm's formatting into an apparent accuracy difference.
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

from deadzone.analysis.model_arms import (
    RaggedArmsError, condition_table, edit_signature, hallucination_report,
    matched_arms, normalization_shift, rescore_cross_model,
)

REFS = {
    "u02": "call maria at four zero five nine one two seven seven",
    "u05": "ship the package to fourteen hundred shattuck avenue berkeley",
}


def _row(model, clip, cond, wer, conf, transcript, *, failed=False, **kw):
    d = {"model": model, "clip_id": clip, "condition_name": cond,
         "wer": wer, "mean_conf": conf, "transcript": transcript,
         "failed": failed, "rt60": 0.5, "snr_db": 10.0, "noise_type": "babble",
         "codec": "none", "mic_rolloff": 0.0,
         "n_ref": 9, "n_sub": 0, "n_del": 0, "n_ins": 0, "n_match": 9,
         "utterance_conf": conf}
    d.update(kw)
    return d


# --- coverage: a ragged comparison must fail loudly, never quietly ------------
def test_arms_are_intersected_to_common_cells():
    """
    The whisper arm ran on 10 clips; nova-3 ran on 40. Comparing them as-is would
    confound the model with the clip set, and the resulting table would look
    perfectly well-formed.
    """
    rows = [
        _row("nova-3", "u02", "c1", 0.1, 0.9, "a"),
        _row("nova-3", "u05", "c1", 0.1, 0.9, "a"),
        _row("nova-3", "u09", "c1", 0.1, 0.9, "a"),      # whisper never ran u09
        _row("whisper-base", "u02", "c1", 0.2, 0.5, "a"),
        _row("whisper-base", "u05", "c1", 0.2, 0.5, "a"),
    ]
    arms = matched_arms(rows)
    assert len(arms["nova-3"]) == len(arms["whisper-base"]) == 2
    assert {r["clip_id"] for r in arms["nova-3"]} == {"u02", "u05"}
    print("ok: arms intersected to the cells both models ran")


def test_failed_rows_are_dropped_not_scored():
    """
    A failure sentinel is not a low-confidence prediction. Counting one as a
    prediction would corrupt the dead-zone rate and the confidence shape at once.
    """
    rows = [
        _row("nova-3", "u02", "c1", 0.1, 0.9, "a"),
        _row("nova-3", "u05", "c1", 0.9, 0.0, "", failed=True),
        _row("whisper-base", "u02", "c1", 0.2, 0.5, "a"),
        _row("whisper-base", "u05", "c1", 0.2, 0.5, "a"),
    ]
    arms = matched_arms(rows)
    assert len(arms["nova-3"]) == len(arms["whisper-base"]) == 1
    print("ok: failed rows dropped before intersection")


def test_wholesale_arm_failure_raises_with_a_useful_message():
    """
    The whisper arm failed wholesale once already (model weights blocked by a
    missing CA bundle). Every row `failed`, so an inner join would return an
    empty table and every downstream mean would be NaN -- which reads as 'no
    result' rather than 'the run never happened'.
    """
    rows = [
        _row("nova-3", "u02", "c1", 0.1, 0.9, "a"),
        _row("whisper-base", "u02", "c1", 0.0, 0.0, "", failed=True),
    ]
    try:
        matched_arms(rows)
        assert False, "expected RaggedArmsError"
    except RaggedArmsError as exc:
        assert "whisper" in str(exc).lower()
    print("ok: a wholesale arm failure raises instead of returning NaNs")


# --- the normalization audit --------------------------------------------------
def test_normalization_moves_whisper_and_leaves_nova3_alone():
    """
    This is the audit that makes the cross-model correction trustworthy. Both
    arms get re-scored; the word-form arm should barely move. If nova-3's WER
    shifted materially, the normalizer would be changing more than orthography
    and the correction could not be trusted in either direction.
    """
    rows = [
        # nova-3 is already word-form and exactly right
        _row("nova-3", "u02", "c1", 0.0, 0.9, REFS["u02"]),
        _row("nova-3", "u05", "c1", 0.0, 0.9, REFS["u05"]),
        # whisper heard the same words but wrote them as digits
        _row("whisper-base", "u02", "c1", 0.8, 0.5, "Call Maria at 405-912-77."),
        _row("whisper-base", "u05", "c1", 0.22, 0.5,
             "Ship the package to 1400 Shattuck Avenue, Berkeley."),
    ]
    arms = rescore_cross_model(matched_arms(rows), REFS)
    shift = normalization_shift(arms)

    assert shift["nova-3"]["mean_shift"] == 0.0, shift["nova-3"]
    assert shift["nova-3"]["wer_crossmodel_mean"] == 0.0
    # whisper's apparent error was entirely formatting, so it collapses to zero
    assert shift["whisper-base"]["wer_crossmodel_mean"] == 0.0
    assert shift["whisper-base"]["mean_shift"] > 0.4, shift["whisper-base"]
    print("ok: normalization moves the digit-form arm, not the word-form arm")


def test_normalization_does_not_launder_a_real_error():
    """The other half of the audit: a genuine miss must survive re-scoring."""
    rows = [
        _row("nova-3", "u02", "c1", 0.0, 0.9, REFS["u02"]),
        _row("whisper-base", "u02", "c1", 0.9, 0.5,
             "Call Mario at 405-912-77."),          # maria -> mario, a real sub
    ]
    arms = rescore_cross_model(matched_arms(rows), REFS)
    assert arms["whisper-base"][0]["wer_xm"] > 0.0
    assert arms["whisper-base"][0]["n_sub_xm"] == 1
    print("ok: a real substitution survives cross-model re-scoring")


# --- aggregation --------------------------------------------------------------
def test_condition_table_averages_over_clips():
    rows = [_row("nova-3", "u02", "c1", 0.2, 0.8, "a"),
            _row("nova-3", "u05", "c1", 0.4, 0.6, "a"),
            _row("nova-3", "u02", "c2", 0.1, 0.9, "a"),
            _row("nova-3", "u05", "c2", 0.1, 0.9, "a")]
    table = condition_table(rows)
    assert len(table) == 2
    c1 = next(r for r in table if r["condition_name"] == "c1")
    assert abs(c1["wer"] - 0.3) < 1e-9
    assert abs(c1["mean_conf"] - 0.7) < 1e-9
    assert c1["n_clips"] == 2
    print("ok: condition table averages over the clip set")


def test_edit_signature_is_a_fraction_of_reference_words():
    rows = [_row("nova-3", "u02", "c1", 0.3, 0.8, "a",
                 n_ref=10, n_sub=1, n_del=2, n_ins=0),
            _row("nova-3", "u05", "c1", 0.3, 0.8, "a",
                 n_ref=10, n_sub=1, n_del=0, n_ins=3)]
    sig = edit_signature(rows)
    assert abs(sig["sub"] - 2 / 20) < 1e-9
    assert abs(sig["del"] - 2 / 20) < 1e-9
    assert abs(sig["ins"] - 3 / 20) < 1e-9
    print("ok: edit signature normalized by reference words")


# --- the whisper-specific failure mode ---------------------------------------
def test_hallucination_detector_separates_length_blowup_from_confusion():
    """
    Whisper invents fluent sentences under heavy degradation. That is a different
    mechanism from acoustic confusion, and an insertion count alone cannot tell
    them apart -- both just look like 'more insertions'.
    """
    plain_miss = _row("whisper-base", "u02", "c1", 0.3, 0.4,
                      "call mario at four zero five nine one two seven seven")
    hallucinated = _row("whisper-base", "u02", "c2", 0.9, 0.4,
                        "Thank you for watching this video please subscribe to "
                        "the channel and hit the bell icon for more updates "
                        "coming very soon to this channel")
    rep = hallucination_report([plain_miss, hallucinated], REFS, top_k=2)

    top = rep["examples"][0]
    assert top["condition_name"] == "c2", rep["examples"]
    assert top["len_ratio"] > 2.0, top
    assert top["foreign_frac"] > 0.8, top
    # the ordinary substitution must NOT be flagged as a hallucination
    other = rep["examples"][1]
    assert other["len_ratio"] < 1.5, other
    print("ok: hallucination separated from ordinary substitution")


def test_a_duplicated_cell_in_one_arm_raises():
    """
    The intersection is computed over a SET of cells, so a cell that appears
    twice in one arm contributes one set member and two rows. The size check
    between arms cannot see it if both arms happen to carry one duplicate each,
    and `condition_table` then averages that clip in twice — its WER and
    confidence carry double weight in the condition, `n_clips` just reads one
    higher, and the table is otherwise perfect. Nothing downstream can tell.
    """
    base = [
        _row("nova-3", "u02", "c1", 0.1, 0.9, "a"),
        _row("nova-3", "u05", "c1", 0.1, 0.9, "a"),
        _row("whisper-base", "u02", "c1", 0.2, 0.5, "a"),
        _row("whisper-base", "u05", "c1", 0.2, 0.5, "a"),
    ]
    matched_arms(base)                                    # clean: no false positive

    # one duplicate per arm, so the arm SIZES still match — the case the old
    # equal-length check was blind to
    poisoned = base + [_row("nova-3", "u02", "c1", 0.9, 0.4, "a"),
                       _row("whisper-base", "u02", "c1", 0.9, 0.4, "a")]
    try:
        matched_arms(poisoned)
        assert False, "expected RaggedArmsError on a duplicated cell"
    except RaggedArmsError as exc:
        assert "3 rows for 2 common" in str(exc), exc
        assert "u02" in str(exc) and "double weight" in str(exc), exc
    print("ok: a duplicated (clip, condition) cell raises even when both arms "
          "carry one, so the equal-size check alone would have passed")


def test_condition_table_refuses_a_repeated_clip():
    """`condition_table` is reachable without going through `matched_arms`."""
    rows = [_row("nova-3", "u02", "c1", 0.2, 0.8, "a"),
            _row("nova-3", "u02", "c1", 0.9, 0.3, "a")]   # same clip, twice
    try:
        condition_table(rows)
        assert False, "expected a duplicate-clip raise"
    except ValueError as exc:
        assert "u02" in str(exc) and "2 rows for 1 distinct clips" in str(exc), exc
    print("ok: condition_table refuses a repeated clip inside one condition")


def test_a_clip_with_no_manifest_reference_raises():
    """
    An unknown clip_id used to score against an EMPTY reference, and
    `classify_errors("", hyp)` returns wer=1.0 with n_ref=0 — a perfect-looking
    total failure attributed to the model, for a row whose ground truth was
    simply never loaded. That is indistinguishable from a real acoustic collapse
    once it is in the table, and it drags the arm's cross-model mean toward 1.0.
    """
    rows = [_row("nova-3", "u99", "c1", 0.0, 0.9, "some transcript"),
            _row("whisper-base", "u99", "c1", 0.0, 0.5, "some transcript")]
    try:
        rescore_cross_model(matched_arms(rows), REFS)     # REFS has no u99
        assert False, "expected a missing-reference raise"
    except ValueError as exc:
        assert "u99" in str(exc) and "wer=1.0" in str(exc), exc
    print("ok: a clip with no manifest reference raises instead of scoring "
          "against an empty reference (which returns a clean-looking WER 1.0)")


def test_edit_signature_refuses_a_zero_denominator():
    """
    `... or 1` turned an empty denominator into a 0/0/0 composition, which reads
    as 'this model destroyed no words' — the exact inversion of 'there was
    nothing to score'.
    """
    for label, rows in (
            ("no rows", []),
            ("no reference words",
             [_row("nova-3", "u02", "c1", 0.0, 0.9, "a", n_ref=0,
                   n_sub=0, n_del=0, n_ins=0)])):
        try:
            edit_signature(rows)
            assert False, f"expected a raise for {label}"
        except ValueError as exc:
            # both messages must say what the 0/0/0 would have been MISREAD as
            assert "no errors" in str(exc) or "no data" in str(exc) \
                or "destroyed no words" in str(exc), exc
    print("ok: an empty edit signature raises instead of printing a clean 0/0/0")


def test_hallucination_report_survives_an_empty_transcript():
    """A model that returns nothing must not divide by zero."""
    rep = hallucination_report(
        [_row("whisper-base", "u02", "c1", 1.0, 0.0, "")], REFS)
    assert rep["examples"][0]["n_hyp"] == 0
    assert np.isfinite(rep["median_len_ratio"])
    print("ok: empty transcript handled")


if __name__ == "__main__":
    test_arms_are_intersected_to_common_cells()
    test_failed_rows_are_dropped_not_scored()
    test_wholesale_arm_failure_raises_with_a_useful_message()
    test_normalization_moves_whisper_and_leaves_nova3_alone()
    test_normalization_does_not_launder_a_real_error()
    test_condition_table_averages_over_clips()
    test_edit_signature_is_a_fraction_of_reference_words()
    test_hallucination_detector_separates_length_blowup_from_confusion()
    test_a_duplicated_cell_in_one_arm_raises()
    test_condition_table_refuses_a_repeated_clip()
    test_a_clip_with_no_manifest_reference_raises()
    test_edit_signature_refuses_a_zero_denominator()
    test_hallucination_report_survives_an_empty_transcript()
    print("\nL1 comparison layer verified on planted structure.")
