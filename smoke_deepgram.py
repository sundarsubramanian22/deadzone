"""
One real end-to-end Deepgram call, to eyeball the adapter's return shape.

    export DEEPGRAM_API_KEY=...
    ./.venv/bin/python smoke_deepgram.py [path/to.wav]

Defaults to the say-generated sample. Prints the full contract dict so you can
confirm word_confidences come back NON-EMPTY (the headline signal) with raw
output (smart_format/punctuate/numerals off).
"""
import sys
from audio_pipeline import transcribe_deepgram, is_failed

path = sys.argv[1] if len(sys.argv) > 1 else "data/recordings/sample_order.wav"
out = transcribe_deepgram(path)          # key from DEEPGRAM_API_KEY env

if is_failed(out):
    print("FAILED (sentinel):", out["error"])
    sys.exit(1)

print(f"transcript      : {out['transcript']!r}")
print(f"len(word_confidences): {len(out['word_confidences'])}")
print(f"mean_conf       : {out['mean_conf']:.4f}")
print(f"utterance_conf  : {out['utterance_conf']:.4f}")
# raw output (formatting off) => transcript tokens align 1:1 with word_confidences
pairs = list(zip(out["transcript"].split(), out["word_confidences"]))
print("first 5 (word, confidence):")
for w, c in pairs[:5]:
    print(f"    ({w!r}, {c:.3f})")
assert out["word_confidences"], "per-word confidence came back EMPTY — headline signal blocked!"
import math
assert not math.isnan(out["mean_conf"]), "mean_conf is NaN — pass condition failed!"
print("\nOK: per-word confidence is non-empty; adapter shape verified end-to-end.")
