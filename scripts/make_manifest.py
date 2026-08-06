"""
Freeze the experiment (SPEC A.R4.6) into `results/MANIFEST.json`.

WHY THIS EXISTS. A commercial model literal is a server-side moving target. The
string "nova-3" resolved to some particular set of weights on the day the grid
ran, and it will resolve to different ones later without any announcement or any
version bump we can see. A re-run in three months is therefore **not the same
experiment**, and without this file we could not say what we measured — only what
we asked for. The same argument applies, more weakly, to ffmpeg's codec
implementations and to every pinned dependency.

So: record the git SHA, the exact model literals, the asset checksums, the
dependency set, the ffmpeg build, and the realised call/cost totals. This file is
also the reproducibility appendix (R8.2 §10) in machine-readable form -- the
write-up quotes it rather than restating it.

    ./.venv/bin/python scripts/make_manifest.py
"""
from __future__ import annotations

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

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path("results")
OUT = RESULTS / "MANIFEST.json"


def _sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as exc:                       # never let provenance crash
        return f"<unavailable: {exc}>"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _asset_hashes() -> dict:
    """
    Hash the acoustic inputs, not just name them. An RIR set that silently
    changed between runs would move every reverb number in the study, and the
    filename would not tell you.
    """
    out: dict[str, dict] = {}
    for sub in ("rirs", "rirs_sim", "noise", "recordings"):
        root = Path("data") / sub
        if not root.is_dir():
            continue
        files = sorted(p for p in root.rglob("*.wav") if p.is_file())
        out[sub] = {
            "n_files": len(files),
            "files": {str(p.relative_to(root)): _sha256(p) for p in files},
        }
    return out


def _cache_totals() -> dict:
    """
    Realised call counts, read off the append-only cache rather than estimated.
    The cache IS the record of what was actually asked of the API.
    """
    totals: dict[str, dict] = {}
    for label, path in (("real", RESULTS / "cache.jsonl"),
                        ("sim", Path("results_sim") / "cache.jsonl")):
        if not path.is_file():
            continue
        per_model: dict[str, dict[str, int]] = {}
        for line in path.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = per_model.setdefault(row.get("model", "?"), {"ok": 0, "failed": 0})
            d["failed" if row.get("failed") else "ok"] += 1
        totals[label] = per_model
    return totals


def _cost_estimate(totals: dict) -> dict:
    """
    Deepgram bills per minute of audio, not per call. Clips average ~4.1 s, so a
    call is ~0.068 min. Whisper is local: zero marginal cost, non-zero wall clock.
    The rate is quoted, not assumed to be current -- vendor pricing moves and a
    stale number stated as fact is worse than a number stated as of a date.
    """
    dg_calls = sum(v.get("nova-3", {}).get("ok", 0) + v.get("nova-3", {}).get("failed", 0)
                   for v in totals.values())
    mins = dg_calls * (4.1 / 60.0)
    rate = 0.0043
    return {
        "deepgram_calls": dg_calls,
        "audio_minutes_est": round(mins, 1),
        "usd_per_minute_quoted": rate,
        "usd_total_est": round(mins * rate, 2),
        "rate_as_of": "2026-08-04",
        "note": ("Deepgram bills per minute of audio. Whisper runs locally at zero "
                 "marginal cost. Re-check vendor pricing before quoting this."),
    }


def build() -> dict:
    totals = _cache_totals()
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "sha": _sh("git", "rev-parse", "HEAD"),
            "branch": _sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_sh("git", "status", "--porcelain")),
            "describe": _sh("git", "describe", "--tags", "--always"),
        },
        "models": {
            # The literal is the thing that matters; the family name is not enough.
            "streaming_commercial": {
                "provider": "deepgram",
                "literal": "nova-3",
                "api": "pre-recorded (listen.v1.media.transcribe_file)",
                "exposes_word_confidence": True,
                "caveat": ("Server-side literal. Deepgram may re-point 'nova-3' at "
                           "new weights without notice; this run is only reproducible "
                           "as of the date above."),
            },
            "open_baseline": {
                "provider": "openai-whisper (local)",
                "literal": "base",
                "exposes_word_confidence": "derived, not native -- see "
                                           "_parse_whisper_result in audio_pipeline.py",
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": _sh("uname", "-srm"),
            "ffmpeg": _sh("ffmpeg", "-version").splitlines()[0] if _sh("ffmpeg", "-version") else "<none>",
            "requirements_lock": ("requirements.lock.txt"
                                  if Path("requirements.lock.txt").is_file() else None),
        },
        "codec_decision": {
            "narrowband_codec": "g726",
            "ffmpeg_encoder": "adpcm_g726 @ 16 kbit/s, 8 kHz",
            "why_not_amr": ("Stock ffmpeg ships AMR-NB decode-only. Building a "
                            "homebrew-tap ffmpeg with --with-opencore-amr would make "
                            "the grid depend on a source build that may not exist on "
                            "the next machine, for a codec-family difference the "
                            "write-up can simply state. G.726 is a genuine telephony "
                            "codec present in every stock ffmpeg."),
            "wideband_codec": "opus-lowrate",
        },
        "corpus": {
            "n_utterances": 40,
            "manifest": "recording_manifest.csv",
            "clean_wer_floor": 0.0165,
            "clean_wer_floor_note": "6 errors / 363 reference words; 35/40 clips exact; "
                                    "every non-zero row adjudicated by ear",
            "speakers": 1,
        },
        "grid": {
            "design": "interaction_grid()",
            "n_conditions": 176,
            "core": "complete 4x4x3x3 factorial (rt60 x snr_db x codec x mic_rolloff) "
                    "at noise_type=babble, 40 clips/cell",
            "noise_arm": "32 cells over engine/road",
            "snr_db_ceiling": 20.0,
            "snr_db_ceiling_why": "corpus inherent SNR measures ~25-28 dB, so a 25 dB "
                                  "request under-delivers by ~2.5 dB",
        },
        "calls": totals,
        "cost": _cost_estimate(totals),
        "assets": _asset_hashes(),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    man = build()
    OUT.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    calls = man["cost"]["deepgram_calls"]
    print(f"wrote {OUT}")
    print(f"  git      {man['git']['sha'][:12]} ({'dirty' if man['git']['dirty'] else 'clean'})")
    print(f"  deepgram {calls} calls  ~${man['cost']['usd_total_est']}")
    for kind, d in man["assets"].items():
        print(f"  {kind:<12} {d['n_files']} files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
