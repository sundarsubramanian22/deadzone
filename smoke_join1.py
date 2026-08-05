"""
⋈ JOIN 1 — the one-clip end-to-end smoke test and VALIDATION GATE (SPEC §11.3-4,
appendix A.R3).

    export DEEPGRAM_API_KEY=...
    ./.venv/bin/python smoke_join1.py                 # default clips u02,u17,u36
    ./.venv/bin/python smoke_join1.py --clips u02

NOTHING DOWNSTREAM IS TRUSTED UNTIL THIS PASSES. It is the only check that the
whole chain -- real recording -> real RIR -> real noise -> real codec -> real ASR
-> scoring -- produces a sane row, and it deliberately runs through
`run_experiment.run_one`, the SAME function the grid will use, so a green gate
here is evidence about the actual production path rather than a parallel
reimplementation of it.

THREE GATES, and each isolates a different failure:

  A  TRUE CLEAN      raw wav, `apply_condition` BYPASSED entirely.
                     There is no "clean" Condition -- the composer always applies
                     an RIR and always mixes noise, so the only true null is the
                     untouched file. Expect WER ~= 0. If this fails, the problem
                     is the reference transcript or the adapter, not acoustics.

  B  NEAR-CLEAN      the most benign Condition the factor space allows.
                     Expect WER to stay low. This separates the COMPOSER from the
                     MODEL: if A is 0.00 and B spikes, the bug is in composition
                     (asset resolution, SNR calibration, onset alignment), not in
                     the ASR.

  C  GARBAGE         the harshest Condition. Expect WER to spike hard.
                     If it does NOT, the composer silently no-op'd somewhere --
                     the single most dangerous outcome, because it produces a
                     complete, clean-looking results table full of nothing.

Gate C also carries the project's thesis in miniature: if mean confidence stays
high while WER > 0.5, the headline effect has been observed on a single clip
before a single grid cell has run.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

from audio_pipeline import (apply_rir, classify_errors, measured_snr_db,
                            transcribe_deepgram)
from conditions import Condition, DiskAssetLibrary, apply_condition
from run_experiment import (MASTER_COLUMNS, load_clip, load_manifest, run_one,
                            write_degraded_wav)

FS = 16000
DEFAULT_CLIPS = ["u02", "u17", "u36"]

# --- the three gates -------------------------------------------------------
NEAR_CLEAN = Condition(rt60=0.2, snr_db=25.0, noise_type="babble",
                       codec="none", mic_rolloff=0.0)
GARBAGE = Condition(rt60=1.0, snr_db=0.0, noise_type="babble",
                    codec="g726", mic_rolloff=1.0)

# Thresholds. Deliberately loose -- this gate exists to catch a BROKEN chain, not
# to grade the model. A tight threshold here just produces false alarms.
GATE_A_MAX_WER = 0.05      # allowed only with a written R1.10 adjudication
GATE_B_MAX_WER = 0.15
GATE_C_MIN_WER = 0.50
SNR_TOLERANCE_DB = 1.0


def _fake_id(tag: str) -> str:
    return f"smoke-{tag}"


def gate_a_raw(clip_id: str, path: Path, ref: str) -> dict:
    """Transcribe the untouched recording. No composer in the loop at all."""
    res = transcribe_deepgram(str(path))
    if res.get("transcript") is None:
        return {"gate": "A", "clip_id": clip_id, "failed": True,
                "error": res.get("error"), "wer": None, "mean_conf": None,
                "transcript": None, "n_conf": 0}
    e = classify_errors(ref, res["transcript"])
    return {"gate": "A", "clip_id": clip_id, "failed": False, "error": None,
            "wer": e["wer"], "n_ref": e["n_ref"], "counts": e["counts"],
            "mean_conf": res["mean_conf"], "transcript": res["transcript"],
            "n_conf": len(res["word_confidences"])}


def gate_composed(tag: str, clip_id: str, audio: np.ndarray, ref: str,
                  cond: Condition, assets, audio_dir: Path) -> dict:
    """One composed condition, through the SAME run_one the grid uses."""
    row = run_one(clip_id=clip_id, audio=audio, ref_text=ref, condition=cond,
                  assets=assets, fs=FS,
                  transcribe_fn=lambda p: transcribe_deepgram(p),
                  model="nova-3", run_id=_fake_id(tag))

    # Save a listenable copy. Non-negotiable manual step: your ears are the only
    # test for "did the composer produce something physically plausible".
    degraded = apply_condition(audio, cond, assets, FS)
    out = audio_dir / f"{clip_id}__{cond.name}.wav"
    write_degraded_wav(out, degraded, FS)

    row["_gate"] = tag
    row["_wav"] = str(out)
    row["_n_conf"] = len(row.get("word_confidences") or "[]")
    return row


def check_snr_calibration(audio: np.ndarray, assets, snr_targets=(0.0, 10.0, 25.0)) -> list[dict]:
    """
    Independent numeric check that the delivered SNR is the requested SNR.

    Reconstructs the reverberant signal the composer mixes against (RIR applied,
    noise not yet added), so the difference isolates exactly the added noise.
    This is the one place the calibration is verified against ground truth
    instead of trusted.
    """
    out = []
    for target in snr_targets:
        cond = Condition(rt60=0.5, snr_db=target, noise_type="babble",
                         codec="none", mic_rolloff=0.0)
        resolved = assets.resolve(cond)
        rir, _ = assets.load_rir(resolved.rir)
        wet = apply_rir(audio, rir, FS)                 # what noise is mixed onto
        composed = apply_condition(audio, cond, assets, FS)
        added = composed - wet                          # codec/rolloff are no-ops here
        got = measured_snr_db(wet, added, FS)
        out.append({"requested": target, "measured": got,
                    "ok": abs(got - target) <= SNR_TOLERANCE_DB})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    if not os.environ.get("DEEPGRAM_API_KEY"):
        print("DEEPGRAM_API_KEY not set", file=sys.stderr)
        return 2

    clip_ids = [c.strip() for c in a.clips.split(",")]
    manifest = load_manifest()
    assets = DiskAssetLibrary(root="data", target_fs=FS)
    res_dir = Path(a.results)
    audio_dir = res_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"library: {len(assets.rirs)} RIRs, {len(assets.noise)} noise clips")
    print(f"gates on: {clip_ids}\n")

    rows: list[dict] = []
    verdicts: list[tuple[str, str, bool, str]] = []

    for cid in clip_ids:
        ref = manifest[cid]
        path = Path("data/recordings") / f"{cid}.wav"
        audio = load_clip(cid, target_fs=FS)

        # ---- Gate A -------------------------------------------------------
        a_res = gate_a_raw(cid, path, ref)
        ok_a = (not a_res["failed"]) and a_res["wer"] <= GATE_A_MAX_WER \
            and a_res["n_conf"] > 0
        verdicts.append(("A", cid, ok_a,
                         f"WER {a_res['wer']:.3f} conf {a_res['mean_conf']:.3f} "
                         f"n_conf {a_res['n_conf']}" if not a_res["failed"]
                         else f"FAILED {a_res['error']}"))

        # ---- Gates B and C -------------------------------------------------
        for tag, cond, lo, hi in (("B", NEAR_CLEAN, None, GATE_B_MAX_WER),
                                  ("C", GARBAGE, GATE_C_MIN_WER, None)):
            r = gate_composed(tag, cid, audio, ref, cond, assets, audio_dir)
            rows.append(r)
            if r["failed"]:
                verdicts.append((tag, cid, False, f"FAILED {r['error']}"))
                continue
            w = r["wer"]
            ok = (lo is None or w >= lo) and (hi is None or w <= hi)

            # A NULL confidence is not a missing value -- it means the model
            # returned NO WORDS at all. That is a qualitatively different failure
            # from being confidently wrong: here the model effectively signals
            # distress, which is the SAFE failure this project contrasts against
            # the dangerous one. Report it as such rather than as a blank cell.
            mc = r["mean_conf"]
            conf = f"conf {mc:.3f}" if mc is not None else "conf --(EMPTY transcript)"
            rt = r["rir_rt60_measured"]
            rt_s = f"{rt:.2f}" if rt is not None else "--"
            note = (f"WER {w:.3f} {conf} "
                    f"sub/del/ins {r['n_sub']}/{r['n_del']}/{r['n_ins']} "
                    f"rt60_delivered {rt_s}")
            verdicts.append((tag, cid, ok, note))

    # ---- SNR calibration cross-check --------------------------------------
    print("SNR calibration (independent of the gates):")
    snr_ok = True
    for c in check_snr_calibration(load_clip(clip_ids[0], target_fs=FS), assets):
        flag = "ok" if c["ok"] else "OFF"
        snr_ok &= c["ok"]
        print(f"  requested {c['requested']:5.1f} dB -> measured "
              f"{c['measured']:6.2f} dB   [{flag}]")

    # ---- report ------------------------------------------------------------
    print("\ngate results:")
    for tag, cid, ok, note in verdicts:
        print(f"  [{'PASS' if ok else 'FAIL'}] gate {tag} {cid}: {note}")

    if rows:
        out = res_dir / "smoke_join1.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(MASTER_COLUMNS))
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in MASTER_COLUMNS})
        print(f"\nwrote {out} ({len(rows)} rows, master schema)")
        print(f"degraded audio in {audio_dir}/ -- LISTEN TO IT (A.R3.5)")

    all_ok = all(v[2] for v in verdicts) and snr_ok
    print("\n" + ("VALIDATION GATE PASSED — cleared to run the grid"
                  if all_ok else
                  "VALIDATION GATE FAILED — do NOT run the grid (playbook: A.R3.8)"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
