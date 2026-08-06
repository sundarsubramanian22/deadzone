"""
smoke_codec.py — REAL codec round-trip through ffmpeg (the one path the offline
test suite mocks out). Run it after installing ffmpeg to prove apply_codec()
actually imprints artifacts, not just that it fails loudly when ffmpeg is absent.

    python3 scripts/smoke_codec.py

Pass condition per codec: the round-trip returns finite audio, restored to the
input length, that DIFFERS from the input (the codec left its mark). AMR-NB
requires an ffmpeg built with the opencore/AudioToolbox AMR encoder; if this
build lacks it, apply_codec raises CodecUnavailableError and we report that
honestly (a real limitation, not a silent skip).
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

from deadzone.conditions import apply_codec, CodecUnavailableError, _CODEC_SPECS

FS = 16000


def _probe_signal():
    """1 s of speech-like sum-of-formants + light noise — enough spectral content
    that a low-rate codec has something to throw away."""
    t = np.arange(FS) / FS
    x = (0.5 * np.sin(2 * np.pi * 220 * t)
         + 0.3 * np.sin(2 * np.pi * 1200 * t)
         + 0.2 * np.sin(2 * np.pi * 2600 * t))
    x += 0.02 * np.random.default_rng(0).standard_normal(FS)
    return x / np.max(np.abs(x)) * 0.9


def run_one(x, codec):
    y = apply_codec(x, FS, codec)
    assert np.all(np.isfinite(y)), f"{codec}: non-finite samples"
    y = y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    diff = float(np.sqrt(np.mean((y - x) ** 2)))
    assert diff > 1e-4, f"{codec}: output identical to input — codec did nothing"
    print(f"  {codec:<13} OK  round-trip RMS delta = {diff:.4f}, "
          f"len {len(y)} (in {len(x)})")


if __name__ == "__main__":
    x = _probe_signal()
    print(f"probe: {len(x)} samples @ {FS} Hz\n")

    # 'none' must be an exact passthrough
    assert np.array_equal(apply_codec(x, FS, "none"), x.astype(np.float64))
    print("  none          OK  exact passthrough")

    failures, unavailable = [], []
    for codec in _CODEC_SPECS:                      # g726, opus-lowrate
        try:
            run_one(x, codec)
        except CodecUnavailableError as e:
            unavailable.append(codec)
            print(f"  {codec:<13} UNAVAILABLE in this ffmpeg build:\n      {e}")
        except AssertionError as e:
            failures.append(codec)
            print(f"  {codec:<13} FAIL: {e}")

    print()
    if failures:
        raise SystemExit(f"SMOKE FAILED for: {failures}")
    ran = [c for c in _CODEC_SPECS if c not in unavailable]
    if not ran:
        raise SystemExit("no codec could run — install an ffmpeg with the encoders")
    print(f"SMOKE OK: real round-trip verified for {ran}"
          + (f"; unavailable (need encoder): {unavailable}" if unavailable else ""))
