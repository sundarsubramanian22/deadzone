"""
Generate the two audio sets that DO need to exist on disk.

    ./.venv/bin/python scripts/make_audio_sets.py            # both
    ./.venv/bin/python scripts/make_audio_sets.py --listen   # listening set only
    ./.venv/bin/python scripts/make_audio_sets.py --sweep    # L3 sweep only

WHY THE GRID DOESN'T SAVE AUDIO. `run_one` composes a degraded clip into a temp
file, transcribes it, and deletes it. Persisting all 7040 cells would be ~1.4 GB
of regenerable data: `apply_condition` seeds its noise crop from the condition
NAME, so any (clip, condition) reproduces bit-identically at zero cost. The cache
holds transcripts, not wavs (SPEC R4.2). Nothing is lost by not keeping them.

Two things still need real files:

  1. THE LISTENING SET (A.R3.5). The unit tests prove the maths; only your ears
     prove the RESULT is physically plausible — that reverb sounds like a room
     rather than a comb filter, that 0 dB SNR really buries the speech, that the
     codec sounds like a phone line, and that nothing is onset-shifted. This
     generates a curated ladder plus the ACTUAL dead zones the grid found, so
     you are listening to the conditions the write-up makes claims about rather
     than to arbitrary cells.

  2. THE L3 SWEEP (A.R5.9). Unlike every other layer, the paralinguistic
     decoupling analysis reads audio, not the results table: it needs one factor
     varied over a ladder with everything else held fixed, so a feature curve and
     a WER curve can be compared level for level. Filenames encode
     `<factor>_<level>.wav` because `analysis.layers.sweep_from_dir` parses the
     level back out of the name.

No API calls. Pure DSP, a few seconds.
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

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from deadzone.conditions import Condition, DiskAssetLibrary, apply_condition
from scripts.run_experiment import load_clip, load_manifest, write_degraded_wav

FS = 16000
LISTEN_DIR = Path("results/audio/listen")
SWEEP_ROOT = Path("results/audio/sweep")

# One clip carries the listening ladder so differences you hear are the CONDITION
# and not the speaker or the sentence. u02 is the smoke clip: digits + a name.
LISTEN_CLIP = "u02"

# A ladder from clean to destroyed, each step changing as few factors as possible
# so you can attribute what you hear. Labels are what you should be listening for.
LADDER: list[tuple[str, Condition]] = [
    ("01_benign",        Condition(0.2, 20.0, "babble", "none", 0.0)),
    ("02_reverb_only",   Condition(1.0, 20.0, "babble", "none", 0.0)),
    ("03_noise_only",    Condition(0.2, 0.0, "babble", "none", 0.0)),
    ("04_rolloff_only",  Condition(0.2, 20.0, "babble", "none", 1.0)),
    ("05_g726_only",     Condition(0.2, 20.0, "babble", "g726", 0.0)),
    ("06_opus_only",     Condition(0.2, 20.0, "babble", "opus-lowrate", 0.0)),
    ("07_engine_0dB",    Condition(0.2, 0.0, "engine", "none", 0.0)),
    ("08_road_0dB",      Condition(0.2, 0.0, "road", "none", 0.0)),
    ("09_reverb_x_codec", Condition(1.0, 20.0, "babble", "opus-lowrate", 1.0)),
    ("10_destroyed",     Condition(1.0, 0.0, "babble", "g726", 1.0)),
]

# The L3 sweeps: one factor laddered, everything else pinned at the benign end so
# the curve is attributable to that factor alone.
SWEEPS: dict[str, dict] = {
    "rt60":   {"levels": [0.2, 0.31, 0.43, 0.54, 0.66, 0.77, 0.89, 1.0],
               "fixed": dict(snr_db=20.0, noise_type="babble", codec="none",
                             mic_rolloff=0.0)},
    "snr_db": {"levels": [20.0, 17.1, 14.3, 11.4, 8.6, 5.7, 2.9, 0.0],
               "fixed": dict(rt60=0.2, noise_type="babble", codec="none",
                             mic_rolloff=0.0)},
}
SWEEP_CLIPS = ["u02", "u06", "u17", "u24", "u36"]


def _dead_zone_conditions(path: str = "results/dead_zones.csv") -> list[Condition]:
    """The conditions D1 actually flagged, so you hear what the claims are about."""
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for row in csv.DictReader(open(p)):
        name = row.get("condition_name") or row.get("condition")
        if not name:
            continue
        try:
            out.append(Condition.from_name(name))
        except Exception:
            continue
    return out


def make_listening_set(assets, manifest) -> int:
    LISTEN_DIR.mkdir(parents=True, exist_ok=True)
    audio = load_clip(LISTEN_CLIP, target_fs=FS)
    n = 0

    # the untouched original, so every comparison has a reference point
    sf.write(LISTEN_DIR / "00_RAW_original.wav", audio, FS, subtype="PCM_16")
    n += 1

    for label, cond in LADDER:
        y = apply_condition(audio, cond, assets, FS)
        write_degraded_wav(LISTEN_DIR / f"{label}.wav", y, FS)
        n += 1

    for i, cond in enumerate(_dead_zone_conditions(), start=1):
        y = apply_condition(audio, cond, assets, FS)
        write_degraded_wav(LISTEN_DIR / f"DEADZONE_{i:02d}_{cond.name}.wav", y, FS)
        n += 1

    (LISTEN_DIR / "WHAT_TO_LISTEN_FOR.md").write_text(
        "# Listening set (SPEC A.R3.5)\n\n"
        f"All from clip `{LISTEN_CLIP}` so what changes is the CONDITION, not the\n"
        "speaker or sentence. Start with `00_RAW_original.wav`.\n\n"
        "Check four things, in this order:\n\n"
        "1. **Onset alignment.** Every file must start when the original starts.\n"
        "   A late start means `apply_rir`'s direct-path trim is wrong and every\n"
        "   WER in the study carries a pure alignment artifact.\n"
        "2. **Reverb sounds like a room** (`02_reverb_only`) — not a delay, not a\n"
        "   metallic comb, no audible repeat.\n"
        "3. **SNR is believable.** `01_benign` (20 dB) should be barely noisy;\n"
        "   `03_noise_only` (0 dB) should nearly bury the speech. If 0 dB sounds\n"
        "   mild, the calibration is off.\n"
        "4. **Codecs sound like a phone line** (`05_g726_only`, `06_opus_only`) —\n"
        "   bandlimited and gritty, not just quieter.\n\n"
        "Then the `DEADZONE_*` files: these are the conditions D1 flagged as\n"
        "confidently-wrong. The question worth asking is whether they sound as bad\n"
        "as their WER says. If they sound *intelligible to you* while the model\n"
        "scored 0.3-0.4 WER at high confidence, that is the finding, audible.\n",
        encoding="utf-8")
    return n


def make_sweeps(assets, manifest) -> int:
    n = 0
    index: dict[str, list] = {}
    for factor, spec in SWEEPS.items():
        d = SWEEP_ROOT / factor
        d.mkdir(parents=True, exist_ok=True)
        entries = []
        for clip_id in SWEEP_CLIPS:
            audio = load_clip(clip_id, target_fs=FS)
            # the RAW clip is the paralinguistic baseline -- NOT a near-clean
            # composed cell, which already has an RIR on it (A.R5.9 gotcha).
            sf.write(d / f"{clip_id}__baseline_raw.wav", audio, FS, subtype="PCM_16")
            for lvl in spec["levels"]:
                kw = dict(spec["fixed"]); kw[factor] = lvl
                cond = Condition(**kw)
                y = apply_condition(audio, cond, assets, FS)
                # sweep_from_dir parses the level back out of <factor>_<level>.wav
                fn = f"{clip_id}__{factor}_{lvl:g}.wav"
                write_degraded_wav(d / fn, y, FS)
                entries.append({"clip_id": clip_id, "factor": factor,
                                "level": float(lvl), "file": str(d / fn),
                                "condition_name": cond.name})
                n += 1
        index[factor] = entries
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    (SWEEP_ROOT / "index.json").write_text(json.dumps(index, indent=2))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    do_listen = a.listen or not (a.listen or a.sweep)
    do_sweep = a.sweep or not (a.listen or a.sweep)

    assets = DiskAssetLibrary(root="data", target_fs=FS)
    manifest = load_manifest()

    if do_listen:
        print(f"listening set -> {LISTEN_DIR}: {make_listening_set(assets, manifest)} files")
    if do_sweep:
        print(f"L3 sweeps     -> {SWEEP_ROOT}: {make_sweeps(assets, manifest)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
