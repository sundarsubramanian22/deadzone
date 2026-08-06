"""probe_elevenlabs.py — the SPEC §12 day-one gate for an ElevenLabs Scribe arm.

ONE clip, ONE call. Nothing else in this project may be built on the ElevenLabs
arm until this prints a definitive yes/no, because the answer changes what the
arm *is*:

  per-word confidence PRESENT -> a full arm. It joins the headline (D1 dead-zone
      map), L2 calibration, and the confidence-shape comparison.
  per-word confidence ABSENT  -> a WER-only arm. It can still feed D2
      fingerprints and the L1 divergence regions, but it CANNOT enter the
      dead-zone map, the calibrator, or any confidence claim. That is the same
      labelling discipline the Whisper arm already carries.

The point of this script is to print the RAW `words[]` payload, unedited, so the
decision is made against what the vendor actually returned rather than against
what an adapter's parsing made of it. Every field name that comes back is
enumerated, because "confidence" here could plausibly be called `confidence`,
`logprob`, `logprobs`, `probability`, `score`, or `avg_logprob`, and guessing
wrong is how an arm silently gets built on a field that means something else.

    ./.venv/bin/python scripts/probe_elevenlabs.py            # default clip u02
    ./.venv/bin/python scripts/probe_elevenlabs.py data/recordings/u17.wav

The key is read from .env as ELEVENLABS_API_KEY and is NEVER printed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_CLIP = ROOT / "data" / "recordings" / "u02.wav"

# Field names that would plausibly carry a per-word confidence. Checked against
# whatever actually comes back, so an unexpected name is reported rather than
# silently missed.
CONFIDENCE_CANDIDATES = (
    "confidence", "logprob", "logprobs", "log_prob", "probability",
    "prob", "score", "avg_logprob", "conf",
)

# What the documentation says to expect, recorded here so the probe's job is to
# CONFIRM a specific prediction rather than to go fishing. Per the API reference
# for POST /v1/speech-to-text, each words[] entry carries `logprob`, documented
# range [-inf, 0], "higher logprobs indicate higher confidence" -- a genuine
# log-probability, NOT a [0,1] score. The realtime websocket variant
# (scribe_v2_realtime) documents the identical field in its
# CommittedTranscriptWithTimestamps message.
#
# Two things the probe must still settle, because the docs cannot:
#   (a) the capabilities page shows an abbreviated example response WITHOUT
#       logprob, so the field's real presence needs one live call to confirm;
#   (b) `language_probability` is a document-level LANGUAGE-detection confidence
#       on a 0-1 scale. It is NOT per-word confidence. Do not conflate them --
#       that mistake would silently give every word in a clip the same value.
EXPECTED_FIELD = "logprob"
EXPECTED_SCALE = "log-probability, <= 0, exp() to get a probability"


def load_key() -> str:
    """Read ELEVENLABS_API_KEY from the environment or .env. Never printed."""
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, val = line.partition("=")
            if name.strip() == "ELEVENLABS_API_KEY":
                return val.strip().strip("'\"")
    raise SystemExit(
        "ELEVENLABS_API_KEY not found in the environment or .env — "
        "the probe cannot run. (Never hardcode it; CLAUDE.md.)"
    )


def transcribe(path: Path, key: str, model_id: str) -> dict:
    with open(path, "rb") as fh:
        resp = requests.post(
            ENDPOINT,
            headers={"xi-api-key": key},
            files={"file": (path.name, fh, "audio/wav")},
            data={"model_id": model_id},
            timeout=180,
        )
    if resp.status_code != 200:
        # Print the vendor's own error verbatim: a 401 vs a 422 vs an unknown
        # model_id are three completely different problems and the body says which.
        raise SystemExit(
            f"HTTP {resp.status_code} from {ENDPOINT}\n{resp.text[:2000]}"
        )
    return resp.json()


def main() -> int:
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CLIP
    if not clip.is_file():
        raise SystemExit(f"clip not found: {clip}")

    key = load_key()
    model_id = os.environ.get("ELEVENLABS_MODEL_ID", "scribe_v2")

    print(f"clip     : {clip}")
    print(f"model_id : {model_id}")
    print(f"endpoint : {ENDPOINT}")
    print("-" * 78)

    # NOTE: there is deliberately NO fallback to scribe_v1. Per the ElevenLabs
    # changelog (2026-06-08) scribe_v1 was deprecated and removed on 2026-07-09,
    # so a fallback would only convert one clear error into a second, more
    # confusing one. scribe_v2 is the current batch literal; scribe_v2_realtime
    # is the websocket variant and carries the SAME per-word schema.
    payload = transcribe(clip, key, model_id)

    print("TOP-LEVEL KEYS:", sorted(payload.keys()))
    print()
    print("TRANSCRIPT:", json.dumps(payload.get("text", ""))[:400])
    print()

    words = payload.get("words") or []
    print(f"words[] present : {'words' in payload}")
    print(f"words[] length  : {len(words)}")
    print()

    if not words:
        print("RAW PAYLOAD (truncated):")
        print(json.dumps(payload, indent=2)[:3000])
        print()
        print("=" * 78)
        print("GATE RESULT: NO per-word array returned -> WER-ONLY ARM.")
        print("=" * 78)
        return 0

    print("RAW words[0:8] — verbatim, unedited:")
    print(json.dumps(words[:8], indent=2))
    print()

    # Enumerate every field that actually appeared, across all words (not just
    # the first): vendors sometimes omit a field on punctuation/spacing tokens.
    seen: dict[str, int] = {}
    for w in words:
        if isinstance(w, dict):
            for k in w:
                seen[k] = seen.get(k, 0) + 1
    print(f"ALL FIELDS SEEN ACROSS {len(words)} word entries (field -> count):")
    for k in sorted(seen):
        print(f"    {k:20} {seen[k]}")
    print()

    found = [k for k in seen if k.lower() in CONFIDENCE_CANDIDATES]
    # Also catch a confidence-like field under a name we did not anticipate.
    numeric_extra = [
        k for k in seen
        if k not in found
        and k not in ("start", "end", "text", "type", "speaker_id")
        and any(isinstance(w.get(k), (int, float)) for w in words if isinstance(w, dict))
    ]

    print(f"PREDICTION FROM THE DOCS: field {EXPECTED_FIELD!r} ({EXPECTED_SCALE})")
    print(f"  confirmed by this call: {EXPECTED_FIELD in seen}")
    if "language_probability" in payload:
        print(f"  NOTE: top-level language_probability = "
              f"{payload['language_probability']} -- this is LANGUAGE-detection "
              f"confidence, NOT per-word confidence. Do not use it as one.")
    print()

    print("=" * 78)
    if found:
        field = found[0]
        vals = [w[field] for w in words if isinstance(w, dict) and field in w
                and isinstance(w[field], (int, float))]
        print(f"GATE RESULT: PER-WORD CONFIDENCE PRESENT -> FULL ARM")
        print(f"  field name : {field!r}   (all matches: {found})")
        print(f"  coverage   : {len(vals)}/{len(words)} word entries carry it")
        if vals:
            lo, hi = min(vals), max(vals)
            print(f"  range      : [{lo:.6f}, {hi:.6f}]")
            # The scale decides the transform. A logprob is <= 0; a probability
            # sits in [0, 1]. Getting this backwards would feed the calibrator
            # nonsense that still looks like a number.
            if hi <= 0.0:
                print("  scale      : LOG-PROBABILITY (all values <= 0) -> use exp(logprob)")
            elif 0.0 <= lo and hi <= 1.0:
                print("  scale      : PROBABILITY in [0, 1] -> use as-is")
            else:
                print("  scale      : UNRECOGNISED — inspect before transforming")
    else:
        print("GATE RESULT: NO per-word confidence field -> WER-ONLY ARM")
        print(f"  fields present were: {sorted(seen)}")
        if numeric_extra:
            print(f"  numeric fields worth a human look: {numeric_extra}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
