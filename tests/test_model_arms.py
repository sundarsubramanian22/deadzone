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
import csv
import os
import tempfile

import numpy as np

from deadzone.analysis.model_arms import (
    RaggedArmsError, augment_divergence_regions, condition_table, edit_signature,
    hallucination_report, matched_arms, model_arms_report, normalization_shift,
    rescore_cross_model,
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


def _silent_row(model, clip, cond, **kw):
    """
    A clip the model returned NOTHING for: WER 1.0, 100% deletions, no
    hypothesis word, and therefore NO confidence at all.

    This is a real MEASUREMENT (not a failure sentinel — `failed` stays False),
    and it is the row that makes the two WER estimands diverge: it lands in the
    all-clips mean at 1.0 while contributing nothing whatever to `mean_conf`.
    """
    kw.setdefault("n_ref", 9)
    n_ref = kw.pop("n_ref")
    return _row(model, clip, cond, 1.0, float("nan"), "",
                n_ref=n_ref, n_del=n_ref, n_sub=0, n_ins=0, n_match=0, **kw)


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


# =============================================================================
# THE SILENT-CLIP ESTIMAND MISMATCH — L1's instance of the D1 headline defect
# =============================================================================
# `condition_table` averaged `wer` over EVERY clip and `mean_conf` over only the
# clips that produced words, and `dead_zone_flags` then thresholded one against
# the other. A clip whose transcript comes back EMPTY scores WER 1.0 with 100%
# deletions and carries no per-word confidence, so it lifts the WER term while
# contributing nothing to the confidence term. Nothing changes shape, nothing
# goes NaN, no row count is wrong — only an assertion on the ESTIMANDS can catch
# it. Every test below is paired with a zero-silence negative control, so the
# guard is pinned to the violation and not to some incidental property of the
# fixture.

# 10 clips per condition; the two conditions below share a confidence and a
# spoke-WER and differ ONLY in how many clips came back empty.
CLIPS10 = tuple(f"u{i:02d}" for i in range(10))
PLANTED_CONF = 0.85


def _condition_rows(model, cond, n_silent, wer_spoke, conf=PLANTED_CONF, **kw):
    """
    One condition x 10 clips: `n_silent` empty transcripts, the rest at
    `wer_spoke` with confidence `conf`.

      wer_spoke      == wer_spoke                     (by construction)
      wer_all_clips  == ((10-n)*wer_spoke + n*1.0)/10
    """
    rows = []
    for i, clip in enumerate(CLIPS10):
        if i < n_silent:
            rows.append(_silent_row(model, clip, cond, **kw))
        else:
            rows.append(_row(model, clip, cond, wer_spoke, conf, "some words", **kw))
    return rows


def test_condition_table_reports_both_wers_and_the_silence_between_them():
    """
    n_spoke is mean_conf's denominator; wer_spoke is the only accuracy that
    denominator may be compared against.

    NEGATIVE CONTROL: `clean` is planted with the SAME confidence and the SAME
    spoke-WER and zero silent clips, so its two WERs must be bit-identical. If a
    future edit made `wer_spoke` anything other than "the same average over a
    subset", this control moves and the test fails on a cell with no silence in
    it at all.
    """
    rows = (_condition_rows("nova-3", "partial", n_silent=2, wer_spoke=0.20)
            + _condition_rows("nova-3", "clean", n_silent=0, wer_spoke=0.20))
    table = {r["condition_name"]: r for r in condition_table(rows)}
    p, c = table["partial"], table["clean"]

    # --- the cell WITH silence: the two estimands separate, and by exactly the
    #     silent clips
    assert p["n_clips"] == 10 and p["n_silent"] == 2 and p["n_spoke"] == 8
    assert abs(p["silent_frac"] - 0.2) < 1e-12
    assert abs(p["wer_spoke"] - 0.20) < 1e-12                    # 8 clips at 0.20
    assert abs(p["wer_all_clips"] - 0.36) < 1e-12                # (8*.2 + 2*1)/10
    assert p["wer"] == p["wer_all_clips"], "`wer` must alias the all-clips WER"
    assert p["wer_spoke"] != p["wer"]
    # the confidence was never touched by the silent clips — they never entered
    # it. An imputation (to 0, to 0.5, to anything) would show up right here.
    assert abs(p["mean_conf"] - PLANTED_CONF) < 1e-12
    assert p["mute"] is False

    # --- NEGATIVE CONTROL: no silence => the two pairings are the SAME NUMBER
    assert c["n_silent"] == 0 and c["n_spoke"] == c["n_clips"] == 10
    assert c["silent_frac"] == 0.0 and c["mute"] is False
    assert c["wer_spoke"] == c["wer_all_clips"] == c["wer"], (
        "with zero silent clips the paired subset IS the whole clip set, so the "
        "two WERs must be bit-identical — not merely close")
    assert abs(c["wer_spoke"] - p["wer_spoke"]) < 1e-12, (
        "both cells were planted with the same spoke-WER; only the all-clips "
        "number may differ between them")
    print(f"ok: wer_spoke {p['wer_spoke']:.3f} vs wer_all_clips "
          f"{p['wer_all_clips']:.3f} on a cell with 2/10 silent clips; the "
          f"0-silence control is bit-identical under both pairings")


def test_a_silent_row_is_a_measurement_not_a_failure():
    """
    The two exclusions are different things and must not be conflated: a
    `failed=True` row is a MISSING measurement (dropped by `matched_arms`),
    while a silent row is a real, severe measurement that must stay in the
    all-clips WER and stay out of the confidence average.
    """
    rows = (_condition_rows("nova-3", "c1", n_silent=3, wer_spoke=0.10)
            + _condition_rows("whisper-base", "c1", n_silent=0, wer_spoke=0.10))
    arms = matched_arms(rows)
    assert len(arms["nova-3"]) == 10, "silent rows were dropped like failures"
    t = condition_table(arms["nova-3"])[0]
    assert t["n_silent"] == 3 and t["n_clips"] == 10
    assert abs(t["wer_all_clips"] - (7 * 0.10 + 3 * 1.0) / 10) < 1e-12
    print("ok: a silent clip stays in the corpus WER and out of the confidence "
          "average; only `failed` rows are dropped")


# --- the flip: what the mismatched pairing calls a dead zone -----------------

def _write_csv(path, rows):
    cols = ["clip_id", "condition_name", "rt60", "snr_db", "noise_type", "codec",
            "mic_rolloff", "model", "transcript", "wer", "n_ref", "n_sub",
            "n_del", "n_ins", "n_match", "mean_conf", "utterance_conf", "failed"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def _grid(model, spec):
    """spec: {condition_name: (n_silent, wer_spoke, conf, factors)}."""
    rows = []
    for name, (n_sil, wer_spoke, conf, factors) in spec.items():
        rows += _condition_rows(model, name, n_sil, wer_spoke, conf, **factors)
    return rows


# Five conditions. `conf_pct_hi` is a within-model PERCENTILE (top 40%), so the
# fixture needs four confidence-bearing conditions for the top two to clear it —
# with three, the runner-up sits at percentile 0.5 and nothing is flagged for a
# reason that has nothing to do with the defect under test.
#
#   silence_driven : 6/10 silent -> all-clips WER 0.52 (over the 0.30 threshold)
#                    while spoke-WER is 0.20 (under it). Confident. THIS is the
#                    cell the old pairing called a dead zone. Real-grid analogue:
#                    rt60-0.7_snr-5_babble_opus-lowrate_roll-1, the one condition
#                    that flipped on the actual nova-3 arm.
#   genuine        : 0 silent, spoke-WER 0.55, most confident -> a dead zone
#                    under BOTH pairings. The correction must not simply empty
#                    the table.
#   quiet / quiet2 : 0 silent, low WER, low confidence -> never flagged; they
#                    exist to give the percentile a distribution to rank against.
#   mute           : 10/10 silent -> no confidence exists at all.
FLIP_SPEC = {
    "silence_driven": (6, 0.20, 0.95, {"rt60": 0.7, "snr_db": 5.0}),
    "genuine":        (0, 0.55, 0.97, {"rt60": 0.9, "snr_db": 2.0}),
    "quiet":          (0, 0.05, 0.30, {"rt60": 0.3, "snr_db": 18.0}),
    "quiet2":         (0, 0.08, 0.25, {"rt60": 0.5, "snr_db": 12.0}),
    "mute":           (10, 0.00, 0.00, {"rt60": 1.0, "snr_db": 0.0}),
}
CLEAN_SPEC = {k: (0, v[1], v[2], v[3]) for k, v in FLIP_SPEC.items()
              if k != "mute"}


def _report(spec, tmp):
    """Run the real L1 entry point over a fabricated master table."""
    rows = _grid("nova-3", spec) + _grid("whisper-base", spec)
    master = os.path.join(tmp, "master.csv")
    manifest = os.path.join(tmp, "manifest.csv")
    _write_csv(master, rows)
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "ground_truth"])
        w.writeheader()
        for c in CLIPS10:
            w.writerow({"id": c, "ground_truth": REFS["u02"]})
    return model_arms_report(master, manifest)


def test_dead_zones_are_flagged_on_the_paired_wer():
    """
    THE HEADLINE ASSERTION for L1. The cell whose all-clips WER only clears the
    threshold because 6 of its 10 clips came back EMPTY must not be called
    'confidently wrong': on the clips it spoke on it was 80% accurate at 0.95
    confidence. It is a SILENCE failure, whose fix (endpointing, VAD,
    dereverberation) is not the confidently-wrong fix.
    """
    with tempfile.TemporaryDirectory() as tmp:
        res = _report(FLIP_SPEC, tmp)
    d = res["per_model"]["nova-3"]

    # the flip, in both directions
    assert "silence_driven" in d["dead_zones_all_clips_pairing"], d
    assert "silence_driven" not in d["dead_zones"], d
    assert d["silence_driven"] == ["silence_driven"], d
    assert d["n_silence_driven"] == 1

    # ...and the genuine dead zone (zero silent clips) SURVIVES — the correction
    # must not just empty the table
    assert d["dead_zones"] == ["genuine"], d["dead_zones"]
    assert "quiet" not in d["dead_zones_all_clips_pairing"]

    # the counts say out loud what the old pairing would have claimed
    assert d["n_dead_zones"] == 1 and d["n_dead_zones_all_clips_pairing"] == 2
    assert d["dead_zone_rate"] < d["dead_zone_rate_all_clips_pairing"]

    # every category is accounted for; nothing is dropped
    assert (d["n_dead_zones"] + d["n_silence_driven"] + d["n_mute_zones"]
            <= d["n_conditions"])
    print(f"ok: the 6/10-silent cell is reclassified dead_zone -> silence_driven "
          f"({d['n_dead_zones_all_clips_pairing']} dead zones under the old "
          f"pairing vs {d['n_dead_zones']} under the corrected one); the "
          f"0-silence 'genuine' cell survives")


def test_zero_silence_makes_the_two_pairings_bit_identical():
    """
    THE NEGATIVE CONTROL, as its own test on the full public path.

    Same conditions, same confidences, same spoke-WERs — but with every clip
    speaking. The two pairings must now agree on EVERY published number. This is
    what pins the guard to the violation rather than to some other property of
    the fixture: if the fix had (say) changed a threshold or a ranking rule
    instead of the estimand, this test would fail even though no clip is silent.
    """
    with tempfile.TemporaryDirectory() as tmp:
        res = _report(CLEAN_SPEC, tmp)

    for m, d in res["per_model"].items():
        assert d["silence"]["n_silent"] == 0, (m, d["silence"])
        assert d["n_conditions_with_silence"] == 0 and d["mean_silent_frac"] == 0.0
        assert d["n_mute_zones"] == 0 and d["n_silence_driven"] == 0, (m, d)
        assert d["dead_zones"] == d["dead_zones_all_clips_pairing"], (m, d)
        assert d["n_dead_zones"] == d["n_dead_zones_all_clips_pairing"], (m, d)
        assert d["dead_zone_rate"] == d["dead_zone_rate_all_clips_pairing"], (m, d)
        assert d["wer_mean_strict"] == d["wer_mean_strict_spoke"], (m, d)
        assert d["shape"]["spearman"] == d["shape_all_clips_pairing"]["spearman"]
        assert d["shape"]["n"] == d["shape_all_clips_pairing"]["n"] == d["n_conditions"]
        assert d["dead_zones"], "the control must still contain a dead zone, or "\
                                "it cannot detect a fix that empties the table"

    ov = res["dead_zone_overlap"]
    assert ov["jaccard"] == ov["jaccard_all_clips_pairing"], ov
    for r in res["divergence_regions"]:
        assert (r["dead_zone_rate_by_model"]
                == r["dead_zone_rate_by_model_all_clips_pairing"]), r
        assert all(v == 0 for v in r["n_mute_by_model"].values()), r
    print("ok: with zero silent clips every corrected number is bit-identical to "
          "its all-clips twin, across both arms, the overlap and every region")


def test_a_mute_condition_is_never_a_dead_zone_and_is_still_named():
    """
    A condition that returned NOTHING on every clip has no confidence, so it
    cannot be *confidently* wrong — but it is the worst cell in the table and a
    confidence-based monitor is blind to it. It must be reported as its own
    category, never silently absorbed into 'not a dead zone'.
    """
    with tempfile.TemporaryDirectory() as tmp:
        res = _report(FLIP_SPEC, tmp)
    d = res["per_model"]["nova-3"]

    assert d["mute_zones"] == ["mute"], d["mute_zones"]
    assert d["n_mute_zones"] == 1
    assert "mute" not in d["dead_zones"]
    assert "mute" not in d["dead_zones_all_clips_pairing"]
    assert "mute" not in d["silence_driven"]
    # a mute condition is a MEASUREMENT, not a failure
    assert "mute" not in d["silence"]["silent_by_condition"] or \
        "mute" in d["silence"]["fully_silent_conditions"]
    assert "mute" in d["silence"]["fully_silent_conditions"]
    # and the formatted block names it
    from deadzone.analysis.model_arms import format_report
    text = format_report(res)
    assert "mute" in text and "silence-driven" in text, text
    print("ok: the fully-silent condition is surfaced as its own category, "
          "present in the report, and never counted as a dead zone")


def test_divergence_regions_carry_the_corrected_rate_and_the_old_one():
    """
    `find_divergence_regions` computes its dead-zone rates internally with the
    all-clips WER and exposes no wer_key, so the region rates are re-derived
    over the SAME bins and the old value is kept under an explicit name. The
    `wer_gap` itself stays all-clips on purpose: it is a corpus-severity
    comparison with no confidence term in it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        res = _report(FLIP_SPEC, tmp)
    regions = res["divergence_regions"]
    assert regions, "fixture produced no populated regions"

    moved = [r for r in regions
             if r["dead_zone_rate_by_model"]
             != r["dead_zone_rate_by_model_all_clips_pairing"]]
    assert moved, ("no region's dead-zone rate moved — the fixture no longer "
                   "exercises the correction inside the divergence scan")
    for r in regions:
        assert set(r["dead_zone_rate_by_model"]) == {"nova-3", "whisper-base"}
        assert r["dead_zone_rate_by_model"] == r["dead_zone_rate_by_model_spoke"]
        assert r["wer_pairing"] == "all_clips"
        # the severity comparison is untouched by the dead-zone fix
        assert np.isfinite(r["wer_gap"])
    print(f"ok: {len(moved)}/{len(regions)} divergence region(s) carry a corrected "
          f"dead-zone rate, with the all-clips value preserved beside it")


def test_augment_is_a_no_op_when_nothing_is_silent():
    """The unit-level negative control for the region augmentation."""
    from deadzone.model_compare import find_divergence_regions
    cond = {m: condition_table(_grid(m, CLEAN_SPEC)) for m in
            ("nova-3", "whisper-base")}
    before = find_divergence_regions(cond)
    rates_before = [dict(r["dead_zone_rate_by_model"]) for r in before]
    after = augment_divergence_regions(before, cond)
    for r, was in zip(after, rates_before):
        assert r["dead_zone_rate_by_model"] == was, (r, was)
    print("ok: augment_divergence_regions is a no-op on a silence-free table")


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
    test_condition_table_reports_both_wers_and_the_silence_between_them()
    test_a_silent_row_is_a_measurement_not_a_failure()
    test_dead_zones_are_flagged_on_the_paired_wer()
    test_zero_silence_makes_the_two_pairings_bit_identical()
    test_a_mute_condition_is_never_a_dead_zone_and_is_still_named()
    test_divergence_regions_carry_the_corrected_rate_and_the_old_one()
    test_augment_is_a_no_op_when_nothing_is_silent()
    print("\nL1 comparison layer verified on planted structure — including that a "
          "confidence is never thresholded against an accuracy measured on a "
          "different set of clips (both pairings published, silence-driven and "
          "mute conditions named, and every corrected number bit-identical to "
          "its all-clips twin when nothing is silent).")
