"""probe_scribe_orthography.py — the ORTHOGRAPHY evidence for the Scribe arm.

`scripts/probe_elevenlabs.py` is the day-one gate: does the vendor return
per-word confidence at all. It passed, so `transcribe_elevenlabs` exists. This
script answers the NEXT question, and it is a gate of its own: **does Scribe's
text agree with the corpus's orthography, or does the arm carry a
condition-independent formatting offset that would be indistinguishable from an
acoustic effect once it is in the master table?**

That trap is not hypothetical here. It cost the Whisper arm 0.20-0.60 WER before
`cross_model_norm.py` existed (SPEC B.2 item 8): Whisper writes numbers as digits
and has no formatting switch, while the Deepgram adapter disables
smart_format/punctuate/numerals, so its output is already word-form. On a corpus
deliberately loaded with phone numbers, spelled codes, addresses and amounts, a
constant offset of that size lands identically in every grid cell — and a
constant clean-condition error looks exactly like a dead zone.

So the audit is: run BOTH scorings on the SAME transcripts and read the shift.

    strict   classify_errors            — the spine scoring, normalize_text only
    x-model  cross_model_classify_errors — plus the Whisper authors' normalizer
                                           and symmetric digit-run splitting

An arm whose raw output is already word-form should shift by ~0 (nova-3 measured
-0.014). A large shift is not an accuracy change; it is the normalizer recovering
orthography, and it is the size of the offset the arm would otherwise have
injected. **This must be run and read before a single Scribe row enters
`results/master.csv`.**

Six clips, six calls, ~$0.001 at $0.22/hr. They are chosen to span exactly the
entity classes the corpus stresses: a spoken phone number (u02), a written-form
quantity (u05), a spelled alphanumeric code (u06), a two-digit number plus a
personal name (u11), a serial with a leading zero (u17), and a mixed
letter/digit tracking id (u33).

    ./.venv/bin/python scripts/probe_scribe_orthography.py
    ./.venv/bin/python scripts/probe_scribe_orthography.py u02 u17

Writes NOTHING. In particular it never touches results/master.csv or
results/cache.jsonl — merging Scribe rows into the master table is gated on the
audit this script produces, not on this script having run.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deadzone.audio_pipeline import (                            # noqa: E402
    classify_errors, is_failed, normalize_text, transcribe_elevenlabs,
)
from deadzone.cross_model_norm import (                          # noqa: E402
    cross_model_classify_errors, cross_model_normalize,
)
from scripts.probe_elevenlabs import load_key                    # noqa: E402

# Entity classes, one per clip, so the table says WHAT each row is testing rather
# than leaving the reader to infer it from the sentence.
EVIDENCE_CLIPS: dict[str, str] = {
    "u02": "spoken digit string (phone)",
    "u05": "written-form quantity (fourteen hundred)",
    "u06": "spelled alphanumeric code",
    "u11": "two-digit number + personal name",
    "u17": "serial with a leading zero",
    "u33": "mixed letter/digit tracking id",
}


def load_manifest() -> dict[str, str]:
    """id -> normalized ground truth, from the one versioned source of truth."""
    with open(ROOT / "recording_manifest.csv", newline="", encoding="utf-8") as f:
        return {r["id"]: r["ground_truth"] for r in csv.DictReader(f)}


def main(argv: list[str]) -> int:
    ids = argv or list(EVIDENCE_CLIPS)
    refs = load_manifest()
    os.environ.setdefault("ELEVENLABS_API_KEY", load_key())   # never printed

    rows = []
    for cid in ids:
        ref = refs[cid]
        path = ROOT / "data" / "recordings" / f"{cid}.wav"
        res = transcribe_elevenlabs(str(path))
        if is_failed(res):
            print(f"{cid}: FAILED — {res['error']}")
            continue
        raw = res["transcript"]
        strict = classify_errors(ref, raw)
        xmodel = cross_model_classify_errors(ref, raw)
        rows.append({
            "id": cid, "what": EVIDENCE_CLIPS.get(cid, ""), "ref": ref, "raw": raw,
            "norm": " ".join(normalize_text(cross_model_normalize(raw))),
            "wer_strict": strict["wer"], "wer_xmodel": xmodel["wer"],
            "mean_conf": res["mean_conf"], "n_conf": len(res["word_confidences"]),
        })

    for r in rows:
        print("=" * 78)
        print(f"{r['id']}  ({r['what']})   mean_conf {r['mean_conf']:.4f} "
              f"over {r['n_conf']} words")
        print(f"  ground truth : {r['ref']}")
        print(f"  RAW scribe   : {r['raw']}")
        print(f"  normalized   : {r['norm']}")
        print(f"  WER  strict {r['wer_strict']:.3f}   x-model {r['wer_xmodel']:.3f}   "
              f"shift {r['wer_xmodel'] - r['wer_strict']:+.3f}")

    if rows:
        s = sum(r["wer_strict"] for r in rows) / len(rows)
        x = sum(r["wer_xmodel"] for r in rows) / len(rows)
        print("=" * 78)
        print(f"NORMALIZATION AUDIT over {len(rows)} clips: "
              f"WER strict {s:.3f} -> x-model {x:.3f}  (shift {x - s:+.3f})")
        print("A large NEGATIVE shift is the normalizer recovering ORTHOGRAPHY, not")
        print("accuracy — and it is the size of the constant offset this arm would")
        print("otherwise have injected into every grid cell. Compare against the")
        print("measured reference points: nova-3 -0.014 (already word-form),")
        print("whisper-base +0.090 (SPEC C.3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
