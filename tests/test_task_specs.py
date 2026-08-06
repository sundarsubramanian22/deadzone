"""
Loader/validation tests for `task_specs.json` (SPEC A.R5.2) — the entity-slot
annotation of `recording_manifest.csv`. Fully offline: a CSV, a JSON file, and
`agent_eval`/`audio_pipeline` pure functions. No audio, no API.

WHY THIS FILE EXISTS. `task_specs.json` is hand-authored data, and hand-authored
data is exactly where a silent bug lives: a slot value with a typo, a stray
capital, a digit written "405" instead of "four zero five" never matches the
transcript, so `evaluate_task` scores it a MISS on a *perfect* transcript. That
looks identical to a real entity failure, and it would land in every grid cell —
a constant, condition-independent entity-error floor that the D2 fingerprint
layer would happily attribute to acoustics. Nothing else in the repo can catch
it, because the spec is the ground truth.

So the load-bearing assertion is: on the EXACT ground truth, every spec must
score 1.0 slot accuracy / 0.0 entity-error rate. If an entity cannot be
recovered from a perfect transcript, the spec is wrong, not the model.

Run:  python3 tests/test_task_specs.py
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
from pathlib import Path

from deadzone.agent_eval import TaskSpec, evaluate_task, entity_error_rate, _contains_subsequence
from deadzone.audio_pipeline import classify_errors, normalize_text

ROOT = Path(_REPO_ROOT)
MANIFEST = ROOT / "recording_manifest.csv"
SPECS = ROOT / "task_specs.json"

# The slot-name vocabulary. Small and CLOSED on purpose: D2 aggregates entity
# errors BY SLOT TYPE ("codes degrade faster than names under codec"), which is
# only possible if the same semantics get the same name across clips. A new name
# here is a deliberate decision, not a typo — hence the test.
SLOT_VOCAB = {
    "name",      # person names (incl. multi-token and multi-person spans)
    "org",       # organisations, departments, business entities
    "phone",     # phone numbers and extensions
    "address",   # street addresses
    "location",  # places and place designators (room / gate / terminal / route)
    "code",      # alphanumeric codes: confirmation, passphrase, unit, plate, serial
    "id",        # numeric record identifiers: invoice, PO, card, tracking, page
    "amount",    # numeric quantities: currency, percentage, dosage, count
    "time",      # clock times and time-of-day expressions
    "date",      # dates and weekdays
}


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _no_dupes(pairs):
    """json object hook that REFUSES duplicate keys (json.load silently keeps
    the last one, so a duplicated clip id would vanish without a trace)."""
    seen, out = set(), {}
    for k, v in pairs:
        assert k not in seen, f"duplicate key in task_specs.json: {k!r}"
        seen.add(k)
        out[k] = v
    return out


def load_specs() -> dict:
    with open(SPECS) as f:
        return json.load(f, object_pairs_hook=_no_dupes)


def load_manifest() -> dict:
    with open(MANIFEST) as f:
        return {r["id"]: r["ground_truth"] for r in csv.DictReader(f)}


def as_taskspec(entry: dict) -> TaskSpec:
    """The exact construction downstream code (D2, R7) must use: `critical`
    arrives from JSON as a list and has to become a tuple for the frozen
    dataclass's declared type."""
    return TaskSpec(slots=dict(entry["slots"]), critical=tuple(entry["critical"]))


def _contiguous(hay: list[str], needle: list[str]) -> bool:
    """Independent re-implementation of agent_eval._contains_subsequence, so the
    coverage assertion isn't validated by the same code it's validating."""
    if not needle:
        return True
    return f" {' '.join(needle)} " in f" {' '.join(hay)} "


SPECS_DATA = load_specs()
MANIFEST_DATA = load_manifest()


# ---------------------------------------------------------------------------
# 1. structure: every clip, exactly once, nothing invented
# ---------------------------------------------------------------------------

def test_ids_match_manifest_exactly():
    assert len(MANIFEST_DATA) == 40, f"manifest should have 40 rows, has {len(MANIFEST_DATA)}"
    missing = sorted(set(MANIFEST_DATA) - set(SPECS_DATA))
    extra = sorted(set(SPECS_DATA) - set(MANIFEST_DATA))
    assert not missing, f"manifest ids with no task spec: {missing}"
    assert not extra, f"task specs for ids not in the manifest: {extra}"
    # _no_dupes already fired on a duplicated key during load; pin the count too.
    assert len(SPECS_DATA) == 40, len(SPECS_DATA)
    print(f"ok: all {len(SPECS_DATA)} manifest ids present exactly once, no extras")


def test_schema_is_well_formed():
    for cid, entry in SPECS_DATA.items():
        assert set(entry) == {"slots", "critical"}, f"{cid}: unexpected keys {set(entry)}"
        slots, crit = entry["slots"], entry["critical"]
        assert isinstance(slots, dict) and slots, f"{cid}: needs at least one slot"
        assert isinstance(crit, list), f"{cid}: 'critical' must be a list"
        for k, v in slots.items():
            assert isinstance(k, str) and isinstance(v, str) and v.strip(), f"{cid}/{k}"
        assert len(set(crit)) == len(crit), f"{cid}: duplicate names in critical: {crit}"
    print("ok: every entry is {slots: {name: value}, critical: [...]} with >=1 slot")


def test_slot_names_are_drawn_from_the_vocabulary():
    used = {}
    for cid, entry in SPECS_DATA.items():
        for slot in entry["slots"]:
            assert slot in SLOT_VOCAB, (
                f"{cid}: slot name {slot!r} is outside the vocabulary {sorted(SLOT_VOCAB)} — "
                f"reuse an existing name or extend SLOT_VOCAB deliberately")
            used[slot] = used.get(slot, 0) + 1
    hist = ", ".join(f"{k}={v}" for k, v in sorted(used.items(), key=lambda kv: -kv[1]))
    assert len(used) >= 8, f"vocabulary barely used — aggregation by type is thin: {used}"
    print(f"ok: {sum(used.values())} slots over {len(used)} reused names — {hist}")


# ---------------------------------------------------------------------------
# 2. the authoring traps: normalization parity and coverage
# ---------------------------------------------------------------------------

def test_slot_values_are_normalization_stable():
    """A value must already be in canonical spoken form: lowercase, unpunctuated,
    numbers as words. If normalize_text() changes it, the file and the scorer
    disagree about what the entity even is."""
    for cid, entry in SPECS_DATA.items():
        for slot, value in entry["slots"].items():
            canon = " ".join(normalize_text(value))
            assert canon == value, (
                f"{cid}/{slot}: {value!r} normalizes to {canon!r} — write the "
                f"canonical spoken form (lowercase, no punctuation, digits as words)")
    print("ok: every slot value survives normalize_text() unchanged")


def test_every_slot_value_occurs_in_its_ground_truth():
    """THE typo catcher: the value must appear CONTIGUOUSLY in that clip's
    normalized ground truth — the exact test evaluate_task applies."""
    for cid, entry in SPECS_DATA.items():
        ref = normalize_text(MANIFEST_DATA[cid])
        for slot, value in entry["slots"].items():
            want = normalize_text(value)
            assert _contains_subsequence(ref, want), (
                f"{cid}/{slot}: {value!r} is not a contiguous span of "
                f"{' '.join(ref)!r} — typo, paraphrase, or wrong clip")
            # cross-check against an independent implementation
            assert _contiguous(ref, want), f"{cid}/{slot}: mirror check disagrees"
    print("ok: every slot value is a contiguous span of its clip's ground truth")


def test_critical_names_exist_in_slots():
    for cid, entry in SPECS_DATA.items():
        unknown = [c for c in entry["critical"] if c not in entry["slots"]]
        assert not unknown, f"{cid}: critical names not present in slots: {unknown}"
        # An empty list silently means "ALL slots are critical" (TaskSpec.critical_slots),
        # which is never what an author intends to say implicitly.
        assert entry["critical"], f"{cid}: 'critical' is empty — state it explicitly"
    print("ok: every critical name refers to a real slot, and none is left implicit")


# ---------------------------------------------------------------------------
# 3. end to end through the real scorer
# ---------------------------------------------------------------------------

def test_taskspec_constructs_and_scores_perfect_on_ground_truth():
    for cid, entry in SPECS_DATA.items():
        spec = as_taskspec(entry)
        assert isinstance(spec.slots, dict) and isinstance(spec.critical, tuple)
        out = evaluate_task(MANIFEST_DATA[cid], spec)
        assert out["slot_accuracy"] == 1.0, f"{cid}: {out['slot_results']}"
        assert out["entity_error_rate"] == 0.0, f"{cid}: {out}"
        assert not out["critical_failure"], f"{cid}: {out['critical_missed']}"
        assert entity_error_rate(MANIFEST_DATA[cid], spec) == 0.0, cid
    print(f"ok: all {len(SPECS_DATA)} specs score 1.0 slot accuracy on their exact "
          f"ground truth (so a MISS downstream is acoustics, not annotation)")


def test_perfect_score_survives_casing_and_punctuation():
    """A real ASR hypothesis arrives capitalized and punctuated; scoring must
    canonicalize it the same way the manifest was canonicalized."""
    for cid, entry in SPECS_DATA.items():
        noisy = MANIFEST_DATA[cid].upper().replace(" ", ", ", 1) + "."
        out = evaluate_task(noisy, as_taskspec(entry))
        assert out["entity_error_rate"] == 0.0, f"{cid}: {out['slot_results']}"
    print("ok: specs still score perfectly on a cased/punctuated transcript")


# ---------------------------------------------------------------------------
# 4. the negative cases — a spec that can never fail is not a spec
# ---------------------------------------------------------------------------

def _drop_span(tokens: list[str], span: list[str]) -> list[str]:
    """Delete every contiguous occurrence of `span` — a degraded transcript that
    lost exactly that entity."""
    out, i, n = [], 0, len(span)
    while i < len(tokens):
        if tokens[i:i + n] == span:
            i += n
        else:
            out.append(tokens[i])
            i += 1
    return out


def test_dropping_a_critical_slot_is_flagged():
    for cid, entry in SPECS_DATA.items():
        spec = as_taskspec(entry)
        ref = normalize_text(MANIFEST_DATA[cid])
        for slot in spec.critical_slots():
            hyp = " ".join(_drop_span(ref, normalize_text(spec.slots[slot])))
            out = evaluate_task(hyp, spec)
            assert out["critical_failure"], f"{cid}: dropping {slot!r} was not flagged: {out}"
            assert slot in out["critical_missed"], f"{cid}/{slot}: {out['critical_missed']}"
            assert out["entity_error_rate"] > 0.0, f"{cid}/{slot}: {out}"
    print("ok: for every clip, deleting any critical entity trips critical_failure")


def test_dropping_a_noncritical_slot_is_an_entity_error_but_not_a_task_failure():
    """The distinction has to actually bite somewhere, or `critical` is decoration."""
    checked = 0
    for cid, entry in SPECS_DATA.items():
        spec = as_taskspec(entry)
        ref = normalize_text(MANIFEST_DATA[cid])
        for slot in spec.slots:
            if slot in spec.critical_slots():
                continue
            hyp = " ".join(_drop_span(ref, normalize_text(spec.slots[slot])))
            out = evaluate_task(hyp, spec)
            assert out["entity_error_rate"] > 0.0, f"{cid}/{slot}: {out}"
            assert not out["critical_failure"], (
                f"{cid}: losing non-critical {slot!r} must not be a task failure: {out}")
            checked += 1
    assert checked >= 5, (
        f"only {checked} non-critical slots in the whole corpus — critical-slot "
        f"failure would be indistinguishable from entity-error rate")
    print(f"ok: {checked} non-critical slots score as entity errors WITHOUT a "
          f"critical failure (the two metrics are separable)")


def test_critical_failure_diverges_from_wer():
    """The point of the layer, stated on real corpus data: one substituted digit
    is a rounding error in WER and a total task failure."""
    cid = "u02"
    ref = MANIFEST_DATA[cid]                       # ...four zero five nine one two seven seven
    hyp = " ".join(normalize_text(ref)[:-1] + ["zero"])   # last digit misheard
    wer = classify_errors(ref, hyp)["wer"]
    out = evaluate_task(hyp, as_taskspec(SPECS_DATA[cid]))
    assert wer < 0.10, wer                          # WER says "essentially perfect"
    assert out["critical_failure"], out             # the number is unusable
    assert out["slot_results"]["name"] == "hit", out
    print(f"ok: {cid} — one wrong digit -> WER {wer:.3f} but the phone slot is lost "
          f"(critical_failure=True)")


# ---------------------------------------------------------------------------
# 5. corpus-level shape (a report line, asserted so it can't silently rot)
# ---------------------------------------------------------------------------

def test_entity_stress_cases_are_covered():
    """The manifest was built around specific stress cases; the annotation has to
    actually point at them or D2 measures nothing interesting."""
    values = {cid: set(e["slots"].values()) for cid, e in SPECS_DATA.items()}
    for cid, needle in [("u04", "priya nair"), ("u07", "okafor"), ("u11", "sofia martinez"),
                        ("u14", "wei zhang"), ("u24", "nguyen and obrien"),
                        ("u26", "kowalski"), ("u40", "yamamoto"),
                        ("u06", "a seven x four two"), ("u17", "q nine j zero five"),
                        ("u33", "one z nine nine a w five"), ("u39", "seven h d k nine one"),
                        ("u02", "four zero five nine one two seven seven"),
                        ("u05", "fourteen hundred shattuck avenue"),
                        ("u13", "forty seven dollars and fifty cents"),
                        ("u18", "fifteen milligrams"), ("u28", "two fifteen")]:
        assert needle in values[cid], f"{cid}: expected a slot with value {needle!r}"
    n_slots = sum(len(e["slots"]) for e in SPECS_DATA.values())
    n_crit = sum(len(e["critical"]) for e in SPECS_DATA.values())
    print(f"ok: name / spelled-code / digit-string / address / currency / dosage / "
          f"time stress cases all annotated ({n_slots} slots, {n_crit} critical)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} TASK-SPEC TESTS PASSED — task_specs.json is consistent "
          f"with recording_manifest.csv and scores clean through evaluate_task")
