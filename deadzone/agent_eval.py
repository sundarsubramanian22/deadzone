"""
agent_eval.py — Layer 4, the voice-agent evaluation scaffold (SPEC §5).

WER is the wrong final metric for a task-taking voice agent: an order can be
transcribed at 15% WER and still be *actioned correctly* (every entity intact), or
at 3% WER and be *wrong* (one killed proper-noun flips the order). This layer is
the metric + analysis surface a live STT->LLM->TTS agent would feed into — built
and validated NOW so the agent reframe (real build later) is drop-in. It does NOT
run an agent and makes NO LLM/TTS calls.

Two independent pieces:
  1. TASK-COMPLETION metrics — given a transcript + a task spec (the slots/entities
     that had to survive), compute slot accuracy and entity-error rate, the
     task-level signal WER can't see. Entity matching reuses audio_pipeline's
     normalize_text so it canonicalizes exactly like the WER path.
  2. TURN-TAKING failure analyzer — given timestamped turn events, detect false
     endpoints (agent cut the user off), missed turns (agent never responded), and
     barge-in errors (agent kept talking over the user).

No factor-space use here (this layer is transcript/timeline-level, not acoustic),
so nothing is redefined. numpy only; no audio, no API, no network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from deadzone.audio_pipeline import normalize_text


# ===========================================================================
# 1. TASK-COMPLETION METRICS
# ===========================================================================

@dataclass(frozen=True)
class TaskSpec:
    """
    The entities that had to survive for the task to be actioned correctly.
    `slots` maps a slot name -> the expected spoken value (scored through the same
    normalize_text() as everything else). `critical` slots (default: all) are the
    ones whose loss means task failure regardless of overall WER.
    """
    slots: dict[str, str]
    critical: tuple[str, ...] = ()

    def critical_slots(self) -> tuple[str, ...]:
        return self.critical if self.critical else tuple(self.slots)


def _contains_subsequence(hay: Sequence[str], needle: Sequence[str]) -> bool:
    """True if `needle` tokens appear CONTIGUOUSLY in `hay` (normalized tokens)."""
    if not needle:
        return True
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if list(hay[i:i + n]) == list(needle):
            return True
    return False


def evaluate_task(transcript: str, spec: TaskSpec) -> dict:
    """
    Score a transcript against a task spec. Per slot: 'hit' if the expected entity
    value appears (normalized, contiguous) in the transcript, else 'miss'.

    Returns slot_results, slot_accuracy, and entity_error_rate = fraction of slots
    missed — the task-level number. Also reports critical_failure: any critical
    slot missed (the order is wrong no matter how low the WER).
    """
    hyp = normalize_text(transcript)
    results: dict[str, str] = {}
    for slot, value in spec.slots.items():
        want = normalize_text(value)
        results[slot] = "hit" if _contains_subsequence(hyp, want) else "miss"

    n = len(spec.slots)
    n_hit = sum(v == "hit" for v in results.values())
    crit_missed = [s for s in spec.critical_slots() if results.get(s) == "miss"]
    return {
        "slot_results": results,
        "n_slots": n,
        "n_hit": n_hit,
        "slot_accuracy": n_hit / n if n else float("nan"),
        "entity_error_rate": (n - n_hit) / n if n else float("nan"),
        "critical_failure": bool(crit_missed),
        "critical_missed": crit_missed,
    }


def entity_error_rate(transcript: str, spec: TaskSpec) -> float:
    """Convenience: just the entity-error rate (fraction of slots missed)."""
    return evaluate_task(transcript, spec)["entity_error_rate"]


# ===========================================================================
# 2. TURN-TAKING FAILURE ANALYZER
# ===========================================================================
#
# Event stream: a list of dicts, each {"t": float_seconds, "type": str}, where
# type is one of the constants below. user/agent speech come as start/end pairs.

USER_START = "user_speech_start"
USER_END = "user_speech_end"
AGENT_ENDPOINT = "agent_endpoint"      # agent decided the user finished
AGENT_START = "agent_speech_start"
AGENT_END = "agent_speech_end"


@dataclass
class TurnFinding:
    kind: str                          # false_endpoint | missed_turn | barge_in_error
    t: float
    detail: str


def _segments(events, start_type, end_type):
    """Pair start/end events into (start, end) segments (robust to minor disorder)."""
    starts = sorted(e["t"] for e in events if e["type"] == start_type)
    ends = sorted(e["t"] for e in events if e["type"] == end_type)
    segs, ei = [], 0
    for s in starts:
        while ei < len(ends) and ends[ei] < s:
            ei += 1
        if ei < len(ends):
            segs.append((s, ends[ei]))
            ei += 1
    return segs


def analyze_turns(events: Sequence[dict], response_timeout: float = 2.0,
                  barge_in_tol: float = 0.3, endpoint_margin: float = 0.05
                  ) -> list[TurnFinding]:
    """
    Detect turn-taking failures from a timestamped event stream:

      * false_endpoint — an agent_endpoint fires while the user is still speaking
        (inside a user segment, more than `endpoint_margin` before its end): the
        agent cut the user off.
      * missed_turn — a user segment ends but the agent neither endpoints nor
        starts speaking within `response_timeout`: the agent missed the turn.
      * barge_in_error — the user starts speaking during an agent segment and the
        agent keeps talking for more than `barge_in_tol` after (never yields).

    Returns the findings sorted by time. An empty list means clean turn-taking.
    """
    ev = sorted(events, key=lambda e: e["t"])
    user_segs = _segments(ev, USER_START, USER_END)
    agent_segs = _segments(ev, AGENT_START, AGENT_END)
    endpoints = sorted(e["t"] for e in ev if e["type"] == AGENT_ENDPOINT)
    agent_starts = sorted(s for s, _ in agent_segs)
    findings: list[TurnFinding] = []

    # false endpoint: endpoint strictly inside a user segment
    for t in endpoints:
        for (s, e) in user_segs:
            if s <= t < e - endpoint_margin:
                findings.append(TurnFinding(
                    "false_endpoint", t,
                    f"agent endpointed at {t:.2f}s, {e - t:.2f}s before the user "
                    f"finished (user turn {s:.2f}-{e:.2f}s)"))
                break

    # missed turn: user finished, no endpoint or agent speech within the timeout.
    # NOT a missed turn if the agent was already mid-utterance at the user's end
    # (that is a barge-in situation, handled separately).
    for (s, e) in user_segs:
        if any(a_s <= e <= a_e for (a_s, a_e) in agent_segs):
            continue
        responded = (any(e <= t <= e + response_timeout for t in endpoints)
                     or any(e <= a <= e + response_timeout for a in agent_starts))
        if not responded:
            findings.append(TurnFinding(
                "missed_turn", e,
                f"user turn ended at {e:.2f}s but the agent did not respond "
                f"within {response_timeout:.1f}s"))

    # barge-in error: user starts inside an agent segment, agent keeps going
    for (us, _ue) in user_segs:
        for (as_, ae) in agent_segs:
            if as_ < us < ae and (ae - us) > barge_in_tol:
                findings.append(TurnFinding(
                    "barge_in_error", us,
                    f"user barged in at {us:.2f}s but the agent kept speaking "
                    f"until {ae:.2f}s ({ae - us:.2f}s, tol {barge_in_tol:.2f}s)"))
                break

    return sorted(findings, key=lambda f: f.t)


def summarize_turn_findings(findings: Sequence[TurnFinding]) -> dict:
    """Counts per failure kind (the dashboard/report payload)."""
    kinds = ("false_endpoint", "missed_turn", "barge_in_error")
    counts = {k: sum(f.kind == k for f in findings) for k in kinds}
    counts["total"] = len(findings)
    return counts
