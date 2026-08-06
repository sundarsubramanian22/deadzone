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


# ---------------------------------------------------------------------------
# VENDOR RATES — one row per model arm, quoted WITH the date it was read
# ---------------------------------------------------------------------------
#
# WHY A TABLE AND NOT INLINE LITERALS. The previous `_cost_estimate` summed the
# `nova-3` key only, with `0.0043` written inline. That was correct for a
# two-arm grid where the second arm was free -- and silently wrong the moment a
# third, BILLED arm appeared: an ElevenLabs Scribe arm would have transcribed
# thousands of clips and contributed exactly $0.00 to the one file whose job is
# recording what the experiment cost, with no error and no missing field. A
# silent zero is this project's documented signature failure, so an unknown
# model is now named LOUDLY as un-costed instead of contributing nothing.
#
# Every row carries the date its rate was read. Vendor pricing moves; a rate
# stated as fact is worse than a rate stated as of a date. Keys are
# model_compare.MODEL_REGISTRY names, because that is what run_experiment.py
# writes into the `model` column and therefore what lands in the cache.
#
# LOCAL ARMS ARE LISTED, NOT OMITTED. Whisper and Vosk cost $0 in money and a
# lot in wall clock; leaving them out of the table would make "absent from the
# rates" mean two different things (free, or forgotten), which is precisely the
# ambiguity the un-costed warning exists to remove.

MINUTES_PER_CALL = 4.1 / 60.0        # corpus mean clip duration ~4.1 s

MODEL_RATES: dict[str, dict] = {
    "nova-3": {
        "provider": "deepgram",
        "billing": "per minute of audio",
        "usd_per_minute": 0.0043,
        "rate_as_of": "2026-08-04",
        "source": "Deepgram pricing, Nova-3 pre-recorded (listen.v1)",
    },
    "whisper-base": {
        "provider": "openai-whisper (local)",
        "billing": "local — no vendor charge",
        "usd_per_minute": 0.0,
        "rate_as_of": "n/a (local)",
        "source": "runs on this machine; the cost is wall clock, not money",
    },
    "vosk": {
        "provider": "vosk / kaldi (local)",
        "billing": "local — no vendor charge",
        "usd_per_minute": 0.0,
        "rate_as_of": "n/a (local)",
        "source": "runs on this machine; the cost is wall clock, not money",
    },
    "elevenlabs-scribe": {
        "provider": "elevenlabs",
        "billing": "per hour of audio",
        "usd_per_hour": 0.22,
        "usd_per_minute": 0.22 / 60.0,
        "rate_as_of": "2026-08-05",
        "source": ("ElevenLabs pricing, Scribe BATCH (scribe_v2). The websocket "
                   "arm scribe_v2_realtime bills at $0.39/hr — a DIFFERENT rate, "
                   "so a realtime arm needs its own row here rather than reusing "
                   "this one."),
    },
}


def _cost_estimate(totals: dict) -> dict:
    """
    Per-model spend, summed across every model present in the caches.

    Vendors bill per minute of audio, not per call; clips average ~4.1 s, so a
    call is ~0.068 min. A model with no row in MODEL_RATES is reported under
    `uncosted_models` and excluded from the total -- never folded in at zero,
    which would look identical to a genuinely free local arm.
    """
    calls: dict[str, dict[str, int]] = {}
    for per_model in totals.values():                # 'real' and 'sim' caches
        for model, d in per_model.items():
            c = calls.setdefault(str(model), {"ok": 0, "failed": 0})
            c["ok"] += int(d.get("ok", 0))
            c["failed"] += int(d.get("failed", 0))

    per_model_cost: dict[str, dict] = {}
    uncosted: list[str] = []
    usd_total = 0.0
    for model in sorted(calls):
        n = calls[model]["ok"] + calls[model]["failed"]
        mins = n * MINUTES_PER_CALL
        rate = MODEL_RATES.get(model)
        row = {
            "calls_ok": calls[model]["ok"],
            "calls_failed": calls[model]["failed"],
            "calls": n,
            "audio_minutes_est": round(mins, 1),
        }
        if rate is None:
            uncosted.append(model)
            row.update({
                "costed": False,
                "usd_est": None,
                "warning": (f"NO RATE ON FILE for model {model!r}. Its spend is "
                            f"NOT in usd_total_est. Add a row to "
                            f"make_manifest.MODEL_RATES (keyed by the "
                            f"model_compare.MODEL_REGISTRY name) before quoting "
                            f"a total."),
            })
        else:
            usd = mins * float(rate["usd_per_minute"])
            usd_total += usd
            row.update({
                "costed": True,
                "provider": rate["provider"],
                "billing": rate["billing"],
                "usd_per_minute_quoted": rate["usd_per_minute"],
                "usd_per_hour_quoted": rate.get("usd_per_hour"),
                "rate_as_of": rate["rate_as_of"],
                "rate_source": rate["source"],
                "usd_est": round(usd, 2),
            })
        per_model_cost[model] = row

    billed = [m for m, r in per_model_cost.items()
              if r.get("costed") and r.get("usd_per_minute_quoted")]
    dg_calls = sum(r["calls"] for m, r in per_model_cost.items()
                   if MODEL_RATES.get(m, {}).get("provider") == "deepgram")
    dg_mins = dg_calls * MINUTES_PER_CALL

    return {
        "per_model": per_model_cost,
        "billed_models": billed,
        "uncosted_models": uncosted,
        "total_calls": sum(r["calls"] for r in per_model_cost.values()),
        "total_audio_minutes_est": round(
            sum(r["audio_minutes_est"] for r in per_model_cost.values()), 1),
        "usd_total_est": round(usd_total, 2),
        "minutes_per_call_assumed": round(MINUTES_PER_CALL, 5),
        # LEGACY, DELIBERATELY VENDOR-SCOPED. The write-up quotes "11,086
        # Deepgram calls"; these two keys keep that number addressable by the
        # name it has always had. They are the DEEPGRAM arm only — the whole-grid
        # figures are `total_calls` / `total_audio_minutes_est` above, and the
        # two must not be confused.
        "deepgram_calls": dg_calls,
        "audio_minutes_est": round(dg_mins, 1),
        "usd_per_minute_quoted": MODEL_RATES["nova-3"]["usd_per_minute"],
        "rate_as_of": MODEL_RATES["nova-3"]["rate_as_of"],
        "legacy_field_note": ("`deepgram_calls` / `audio_minutes_est` / "
                              "`usd_per_minute_quoted` / `rate_as_of` describe the "
                              "DEEPGRAM arm alone and are retained under their "
                              "original names because the write-up quotes them. "
                              "For the whole experiment read `total_calls`, "
                              "`total_audio_minutes_est` and `usd_total_est`."),
        "note": ("Vendors bill per minute of audio, not per call. Local arms "
                 "(whisper, vosk) are listed at $0 rather than omitted, so an "
                 "absent model means 'no rate on file', not 'free'. Re-check "
                 "vendor pricing before quoting any of this."),
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
            # Renamed from "streaming_commercial" (SPEC Appendix K.5): the key
            # contradicted its own `api` field two lines below, which has always
            # read "pre-recorded". No arm in this project ever streamed. Nothing
            # consumes this key by name -- verified by grep across the repo before
            # the rename -- so the taxonomy is free to tell the truth.
            "batch_commercial": {
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
    cost = man["cost"]
    print(f"wrote {OUT}")
    print(f"  git      {man['git']['sha'][:12]} ({'dirty' if man['git']['dirty'] else 'clean'})")
    print(f"  calls    {cost['total_calls']} across {len(cost['per_model'])} arm(s), "
          f"~{cost['total_audio_minutes_est']} min audio  ~${cost['usd_total_est']}")
    for model, row in cost["per_model"].items():
        usd = "UN-COSTED" if not row["costed"] else f"~${row['usd_est']:.2f}"
        print(f"    {model:<20} {row['calls']:>6} calls  "
              f"{row['audio_minutes_est']:>7.1f} min  {usd}")
    # An un-costed arm is the one thing here that must not scroll past quietly:
    # its spend is absent from the freeze, which is the file that exists to say
    # what was spent.
    for model in cost["uncosted_models"]:
        print(f"  !! NO RATE ON FILE for {model!r} — its spend is NOT in "
              f"${cost['usd_total_est']}. Add it to make_manifest.MODEL_RATES.",
              file=sys.stderr)
    for kind, d in man["assets"].items():
        print(f"  {kind:<12} {d['n_files']} files hashed")
    return 1 if cost["uncosted_models"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
