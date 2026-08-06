"""
Offline validation for Layer 4 (agent_eval.py). No agent, no LLM/TTS — synthetic
transcripts and synthetic turn-event streams with PLANTED errors, and we assert
the scaffold recovers them.

Task metrics — the point is that entity-error rate and WER DISAGREE:
  * a low-WER transcript that drops a critical entity is a task FAILURE,
  * a high-WER transcript (padded with filler) that keeps every entity is a task
    SUCCESS.
Turn-taking — planted false endpoint / missed turn / barge-in each get flagged,
and correctly-handled turns do NOT.

Deterministic, numpy-free logic. Run:  python3 test_agent_eval.py
"""
from agent_eval import (
    TaskSpec, evaluate_task, entity_error_rate,
    analyze_turns, summarize_turn_findings,
    USER_START, USER_END, AGENT_ENDPOINT, AGENT_START, AGENT_END,
)
from audio_pipeline import classify_errors

REF = "call maria at four zero five"
SPEC = TaskSpec(slots={"name": "maria", "code": "four zero five"},
                critical=("name",))


# ---- 1: all entities present -> perfect task score -------------------------

def test_all_entities_hit():
    out = evaluate_task("call maria at four zero five", SPEC)
    assert out["slot_accuracy"] == 1.0 and out["entity_error_rate"] == 0.0, out
    assert not out["critical_failure"], out
    assert out["slot_results"] == {"name": "hit", "code": "hit"}
    print("OK 1: all entities present -> slot_accuracy 1.0, no critical failure")


# ---- 2: a missed entity is caught (with normalization parity) --------------

def test_missed_entity():
    # code spoken with a different filler word -> "four oh five" != "four zero five"
    out = evaluate_task("call maria at four oh five", SPEC)
    assert out["slot_results"] == {"name": "hit", "code": "miss"}, out
    assert out["entity_error_rate"] == 0.5, out
    assert not out["critical_failure"], out       # code isn't critical here
    # normalization parity: case/punctuation don't matter
    out2 = evaluate_task("Call MARIA, at four zero five!", SPEC)
    assert out2["entity_error_rate"] == 0.0, out2
    print("OK 2: missed entity -> 'miss'; matching canonicalizes case/punctuation")


# ---- 3: entity-error rate and WER DISAGREE (the whole point) ---------------

def test_entity_vs_wer_disagree():
    # A) LOW WER but a critical entity flips: maria -> sarah (1 sub of 6 words)
    hyp_a = "call sarah at four zero five"
    wer_a = classify_errors(REF, hyp_a)["wer"]
    task_a = evaluate_task(hyp_a, SPEC)
    assert wer_a < 0.25, wer_a                                   # WER says "fine"
    assert task_a["critical_failure"] and task_a["entity_error_rate"] >= 0.5, task_a

    # B) HIGH WER (filler insertions) but every entity intact
    hyp_b = "um yeah so like call maria at four zero five please thanks"
    wer_b = classify_errors(REF, hyp_b)["wer"]
    task_b = evaluate_task(hyp_b, SPEC)
    assert wer_b > 0.5, wer_b                                    # WER says "bad"
    assert task_b["entity_error_rate"] == 0.0 and not task_b["critical_failure"], task_b

    print(f"OK 3: WER/entity disagree — A) WER={wer_a:.2f} but CRITICAL fail; "
          f"B) WER={wer_b:.2f} but task perfect")


# ---- 4: clean turn-taking -> no findings -----------------------------------

def _ev(t, ty):
    return {"t": t, "type": ty}


def test_clean_dialog():
    events = [
        _ev(0.0, USER_START), _ev(2.0, USER_END),
        _ev(2.1, AGENT_ENDPOINT),                     # endpoint AFTER user done
        _ev(2.2, AGENT_START), _ev(4.0, AGENT_END),
    ]
    assert analyze_turns(events) == [], analyze_turns(events)
    print("OK 4: clean dialog -> no turn-taking findings")


# ---- 5: planted turn-taking failures each get flagged ----------------------

def test_false_endpoint():
    events = [
        _ev(0.0, USER_START), _ev(2.0, USER_END),
        _ev(1.0, AGENT_ENDPOINT),                     # fires mid-user-turn
        _ev(2.1, AGENT_START), _ev(4.0, AGENT_END),   # responds -> no missed turn
    ]
    f = analyze_turns(events)
    assert [x.kind for x in f] == ["false_endpoint"], f
    assert abs(f[0].t - 1.0) < 1e-9
    print(f"OK 5a: false endpoint flagged — {f[0].detail}")


def test_missed_turn():
    events = [
        _ev(0.0, USER_START), _ev(2.0, USER_END),
        # nothing until 5.0 -> agent missed the turn
        _ev(5.0, AGENT_START), _ev(6.0, AGENT_END),
    ]
    f = analyze_turns(events, response_timeout=2.0)
    assert [x.kind for x in f] == ["missed_turn"], f
    assert abs(f[0].t - 2.0) < 1e-9
    print(f"OK 5b: missed turn flagged — {f[0].detail}")


def test_barge_in_error():
    events = [
        _ev(1.0, AGENT_START), _ev(5.0, AGENT_END),   # agent talks 1..5
        _ev(2.0, USER_START), _ev(3.0, USER_END),     # user barges in at 2
    ]
    f = analyze_turns(events, barge_in_tol=0.3)
    kinds = [x.kind for x in f]
    assert kinds == ["barge_in_error"], f             # not a missed turn (agent busy)
    assert abs(f[0].t - 2.0) < 1e-9
    print(f"OK 5c: barge-in flagged — {f[0].detail}")


def test_barge_in_handled_ok():
    # user barges in but the agent YIELDS within tolerance -> no error
    events = [
        _ev(1.0, AGENT_START), _ev(2.15, AGENT_END),  # yields 0.15s after barge
        _ev(2.0, USER_START), _ev(3.0, USER_END),
        _ev(3.1, AGENT_START), _ev(4.0, AGENT_END),   # responds to the user
    ]
    f = analyze_turns(events, barge_in_tol=0.3)
    assert f == [], f
    print("OK 5d: agent yields to barge-in within tolerance -> no finding")


# ---- 6: summary payload -----------------------------------------------------

def test_summary():
    events = [
        _ev(0.0, USER_START), _ev(2.0, USER_END), _ev(1.0, AGENT_ENDPOINT),
        _ev(2.1, AGENT_START), _ev(2.3, AGENT_END),
        _ev(10.0, USER_START), _ev(12.0, USER_END),   # then a missed turn
    ]
    s = summarize_turn_findings(analyze_turns(events))
    assert s["false_endpoint"] == 1 and s["missed_turn"] == 1, s
    assert s["total"] == 2, s
    print(f"OK 6: summary counts by kind -> {s}")


if __name__ == "__main__":
    test_all_entities_hit()
    test_missed_entity()
    test_entity_vs_wer_disagree()
    test_clean_dialog()
    test_false_endpoint()
    test_missed_turn()
    test_barge_in_error()
    test_barge_in_handled_ok()
    test_summary()
    print("\nAll agent-eval tests passed — task/entity metrics diverge from WER "
          "as intended, and the turn-taking analyzer flags planted failures only.")
