#!/usr/bin/env python3
"""
dashboard/make_synthetic.py — SPEC A.R6.2.

Emit ``results/synthetic_master.csv`` in the FROZEN master-table schema
(``run_experiment.MASTER_COLUMNS``) with PLANTED structure, so the dashboard can
be built, tested and demoed before any real grid exists.

Why a generator and not a checked-in fixture: the dashboard's job is to make a
*finding* legible, and you cannot tell whether it does that unless the finding is
known in advance. Everything the dashboard should show is planted here:

  * a DEAD ZONE — high reverb AND good SNR. WER is driven up by reverb; the
    commercial arm's confidence is driven almost entirely by SNR, so in the
    high-rt60 / high-SNR corner the model stays confident while it is wrong.
    That corner is the hero panel's whole point.
  * FINGERPRINTS — reverb produces DELETIONS, babble produces SUBSTITUTIONS plus
    a few insertions (competing speech), codec produces substitutions on
    entity-ish words. Each family therefore has a different edit signature.
  * A MODEL SPLIT — ``nova-3`` is overconfident and its confidence ignores
    reverb; ``whisper`` is less confident overall but its confidence tracks WER.
    The model toggle should visibly change the dead-zone map.
  * A SIM-VS-REAL GAP — simulated RIRs are systematically a little kinder than
    the measured ones, and the bias grows with reverb, but the ORDER of the
    conditions is largely preserved (high rank correlation, non-zero level gap).

The numbers in the table are NOT invented directly: a hypothesis transcript is
constructed by corrupting the real reference text, and ``wer`` / ``n_sub`` /
``n_del`` / ``n_ins`` / ``edits`` all come from the real
``audio_pipeline.classify_errors`` alignment of that pair. So the table is
internally consistent in exactly the way a real run's table is, and the hero
panel's transcript diff shows a genuine alignment.

Usage:
    python3 dashboard/make_synthetic.py [--out results/synthetic_master.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from audio_pipeline import classify_errors          # noqa: E402  (repo-root import)

# The schema is imported, never re-declared — two copies of a "frozen" schema is
# exactly how one of them goes stale (run_experiment.py, same discipline).
try:
    from run_experiment import MASTER_COLUMNS, AL_CLIPS, main_grid
    _HAVE_RUNNER = True
except Exception:                                    # pragma: no cover - defensive
    _HAVE_RUNNER = False
    MASTER_COLUMNS = (
        "clip_id", "condition_name", "rt60", "snr_db", "noise_type", "codec",
        "mic_rolloff", "model", "transcript", "wer", "n_ref", "n_sub", "n_del",
        "n_ins", "n_match", "mean_conf", "utterance_conf", "word_confidences",
        "edits", "rir_key", "rir_rt60_measured", "noise_key", "failed", "error",
        "run_id", "ts",
    )
    AL_CLIPS = ["u02", "u05", "u06", "u11", "u17", "u22", "u24", "u33", "u36", "u39"]
    main_grid = None


MODELS = ("nova-3", "whisper")
RUN_ID = "synthetic-r6"
TS = "2026-08-05T00:00:00+00:00"

# Words a degraded ASR plausibly emits: the substitution pool is deliberately
# entity-flavoured (names, digits, letters) because that is the failure mode the
# corpus was built to stress (SPEC §8).
_CONFUSIONS = {
    "four": "for", "five": "fire", "nine": "night", "two": "to", "seven": "heaven",
    "one": "won", "three": "free", "eight": "ate", "zero": "sarah", "six": "sticks",
    "b": "d", "a": "eight", "x": "ex", "maria": "mariah", "daniel": "danielle",
    "priya": "freya", "nair": "nare", "okafor": "aker for", "berkeley": "barclay",
    "shattuck": "shadduck", "avenue": "have you", "friday": "fry day",
    "thursday": "thirsty", "room": "rome", "lease": "least", "signed": "sign",
}
_FOREIGN = ["yeah", "okay", "uh", "and", "the", "so", "right", "hey"]

# ---------------------------------------------------------------------------
# The reference corpus — real manifest if present, else a small built-in stand-in
# ---------------------------------------------------------------------------

_FALLBACK_REFS = {
    "u02": "call maria at four zero five nine one two seven seven",
    "u05": "the package goes to fourteen hundred shattuck avenue in berkeley",
    "u06": "the access code is a seven x four two",
    "u11": "book the room for eleven fifteen on tuesday morning",
    "u17": "transfer nine hundred dollars to account three three eight one",
    "u22": "her flight lands at gate c nineteen at six forty",
    "u24": "spell the surname okafor for the record please",
    "u33": "confirm the order number q four seven seven zero six",
    "u36": "the invoice total is one thousand two hundred and forty dollars",
    "u39": "send the signed lease to daniel by friday afternoon",
}


def load_references(manifest_path: str = "recording_manifest.csv",
                    clips=AL_CLIPS) -> dict[str, str]:
    """clip_id -> ground-truth text. Falls back to a built-in set off-repo."""
    path = manifest_path if os.path.isabs(manifest_path) else os.path.join(REPO, manifest_path)
    refs: dict[str, str] = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("id") in clips and row.get("ground_truth"):
                    refs[row["id"]] = row["ground_truth"].strip()
    for c in clips:
        refs.setdefault(c, _FALLBACK_REFS.get(c, f"reference utterance {c} with code a seven x"))
    return {c: refs[c] for c in clips}


# ---------------------------------------------------------------------------
# The plant
# ---------------------------------------------------------------------------

def _norm(v: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def planted_rates(rt60: float, snr_db: float, noise_type: str, codec: str,
                  mic_rolloff: float, sim: bool = False) -> dict:
    """
    The ground truth of this synthetic world. Every downstream claim the dashboard
    makes should be traceable back to a term in here.
    """
    r = _norm(rt60, 0.2, 1.0)                    # 0 = dry, 1 = very reverberant
    s = 1.0 - _norm(snr_db, 0.0, 25.0)           # 0 = quiet, 1 = very noisy
    babble = 1.0 if noise_type == "babble" else 0.0
    coded = 0.0 if codec == "none" else (0.6 if codec == "g726" else 1.0)

    # Sim RIRs are systematically KINDER than measured ones, and the bias grows
    # with reverb — the sim2real finding, planted.
    sim_relief = (0.25 + 0.35 * r) if sim else 0.0

    del_rate = 0.55 * r * (1.0 - 0.5 * sim_relief)          # REVERB -> deletions
    sub_rate = (0.22 + 0.30 * babble) * s + 0.18 * coded + 0.10 * mic_rolloff
    ins_rate = 0.09 * babble * s                            # babble -> insertions
    return {"del_rate": del_rate, "sub_rate": sub_rate, "ins_rate": ins_rate}


def planted_confidence(model: str, rt60: float, snr_db: float, noise_type: str,
                       codec: str, mic_rolloff: float) -> float:
    """
    THE headline plant. `nova-3` confidence is essentially a function of SNR and
    channel — it barely notices reverb — so reverb-driven errors arrive with the
    confidence still high. `whisper` is lower and flatter but DOES track reverb,
    so its dead zone is smaller. The two models therefore disagree about where the
    danger is, which is what the model toggle exists to show.
    """
    r = _norm(rt60, 0.2, 1.0)
    s = 1.0 - _norm(snr_db, 0.0, 25.0)
    coded = 0.0 if codec == "none" else (0.5 if codec == "g726" else 1.0)
    if model == "nova-3":
        c = 0.97 - 0.42 * s - 0.10 * coded - 0.05 * r - 0.03 * mic_rolloff
    else:
        c = 0.88 - 0.30 * s - 0.08 * coded - 0.30 * r - 0.05 * mic_rolloff
    return max(0.05, min(0.995, c))


# `whisper` is a slightly weaker transcriber than the commercial arm, uniformly —
# so L1's "which model is weaker where" question has an answer to find.
_MODEL_WER_SCALE = {"nova-3": 1.0, "whisper": 1.18}


# ---------------------------------------------------------------------------
# Reference -> hypothesis, then score it for real
# ---------------------------------------------------------------------------

def corrupt(ref: str, n_sub: int, n_del: int, n_ins: int, rng: random.Random) -> str:
    words = ref.split()
    n = len(words)
    idx = list(range(n))
    rng.shuffle(idx)
    sub_at = set(idx[:n_sub])
    del_at = set(idx[n_sub:n_sub + n_del])

    out: list[str] = []
    for i, w in enumerate(words):
        if i in del_at:
            continue
        if i in sub_at:
            out.append(_CONFUSIONS.get(w, rng.choice(_FOREIGN)))
        else:
            out.append(w)
    for _ in range(n_ins):
        pos = rng.randrange(len(out) + 1) if out else 0
        out.insert(pos, rng.choice(_FOREIGN))
    return " ".join(out)


def make_row(clip_id: str, ref: str, cond: dict, model: str,
             rng: random.Random) -> dict:
    rates = planted_rates(cond["rt60"], cond["snr_db"], cond["noise_type"],
                          cond["codec"], cond["mic_rolloff"], sim=cond["sim"])
    scale = _MODEL_WER_SCALE.get(model, 1.0)
    n = len(ref.split())
    n_sub = min(n, round(rates["sub_rate"] * scale * n))
    n_del = min(n - n_sub, round(rates["del_rate"] * scale * n))
    n_ins = round(rates["ins_rate"] * scale * n)

    hyp = corrupt(ref, n_sub, n_del, n_ins, rng)
    # The TRUTH of the row comes from the real aligner, not from the plant — the
    # same function the runner uses, so the table is internally consistent.
    scored = classify_errors(ref, hyp)
    conf = planted_confidence(model, cond["rt60"], cond["snr_db"],
                              cond["noise_type"], cond["codec"], cond["mic_rolloff"])
    hyp_words = hyp.split()
    word_confs = [round(max(0.02, min(0.999, conf + rng.gauss(0.0, 0.05))), 4)
                  for _ in hyp_words]
    mean_conf = round(sum(word_confs) / len(word_confs), 4) if word_confs else None

    return {
        "clip_id": clip_id,
        "condition_name": cond["name"],
        "rt60": cond["rt60"], "snr_db": cond["snr_db"],
        "noise_type": cond["noise_type"], "codec": cond["codec"],
        "mic_rolloff": cond["mic_rolloff"],
        "model": model,
        "transcript": hyp,
        "wer": round(scored["wer"], 6),
        "n_ref": scored["n_ref"],
        "n_sub": scored["counts"]["sub"], "n_del": scored["counts"]["del"],
        "n_ins": scored["counts"]["ins"], "n_match": scored["counts"]["match"],
        "mean_conf": mean_conf,
        "utterance_conf": mean_conf,
        "word_confidences": json.dumps(word_confs),
        "edits": json.dumps(scored["edits"]),
        "rir_key": cond["rir_key"],
        "rir_rt60_measured": cond["rir_rt60_measured"],
        "noise_key": f"musan/{cond['noise_type']}_01.wav",
        "failed": False, "error": "",
        "run_id": RUN_ID, "ts": TS,
    }


def failed_row(clip_id: str, cond: dict, model: str) -> dict:
    """
    The runner's failure sentinel: no transcript, no confidence, WER 1.0. Planted
    on purpose — every panel must survive it, and the dashboard must SAY how many
    rows it dropped rather than quietly averaging a failure in as a dead zone.
    """
    return {
        "clip_id": clip_id, "condition_name": cond["name"],
        "rt60": cond["rt60"], "snr_db": cond["snr_db"],
        "noise_type": cond["noise_type"], "codec": cond["codec"],
        "mic_rolloff": cond["mic_rolloff"], "model": model,
        "transcript": "", "wer": 1.0, "n_ref": 0, "n_sub": 0, "n_del": 0,
        "n_ins": 0, "n_match": 0, "mean_conf": "", "utterance_conf": "",
        "word_confidences": "[]", "edits": "[]",
        "rir_key": cond["rir_key"], "rir_rt60_measured": cond["rir_rt60_measured"],
        "noise_key": f"musan/{cond['noise_type']}_01.wav",
        "failed": True, "error": "TimeoutError: adapter timed out",
        "run_id": RUN_ID, "ts": TS,
    }


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

_CORE_RT60 = (0.2, 0.45, 0.7, 1.0)
_CORE_SNR = (0.0, 10.0, 25.0)
_CORE_NOISE = ("babble", "engine", "road")
_CHANNEL_ANCHORS = ((0.2, 25.0), (0.45, 10.0), (1.0, 0.0))
_CHANNEL_CODECS = ("none", "g726", "opus-lowrate")
_CHANNEL_ROLLOFF = (0.0, 0.5, 1.0)


def _name(rt60, snr, noise, codec, roll) -> str:
    if _HAVE_RUNNER:
        try:
            from conditions import Condition
            return Condition(rt60, snr, noise, codec, roll).name
        except Exception:                            # pragma: no cover
            pass
    return (f"rt{rt60:g}_snr{snr:g}_{noise}"
            + (f"_{codec}" if codec != "none" else "")
            + (f"_roll{roll:g}" if roll else ""))


def _cond(rt60, snr, noise, codec, roll, sim=False) -> dict:
    # Measured RT60 wobbles around the request — the sim/real pairing is done on
    # the MEASURED value (analysis/sim2real.py), never the requested one.
    jitter = 0.012 if not sim else -0.018
    return {
        "rt60": rt60, "snr_db": snr, "noise_type": noise, "codec": codec,
        "mic_rolloff": roll, "sim": sim,
        "name": _name(rt60, snr, noise, codec, roll) + ("_sim" if sim else ""),
        "rir_key": (f"data/rirs_sim/sim_rt60-{rt60:.2f}_booth.wav" if sim
                    else f"data/rirs/but_rt60-{rt60:.2f}_q301.wav"),
        "rir_rt60_measured": round(rt60 + jitter, 4),
    }


def grid() -> list[dict]:
    """~60 real-RIR conditions (mirrors run_experiment.main_grid) + a sim arm."""
    conds: list[dict] = []
    for rt60 in _CORE_RT60:
        for snr in _CORE_SNR:
            for noise in _CORE_NOISE:
                conds.append(_cond(rt60, snr, noise, "none", 0.0))
    for rt60, snr in _CHANNEL_ANCHORS:
        for codec in _CHANNEL_CODECS:
            for roll in _CHANNEL_ROLLOFF:
                conds.append(_cond(rt60, snr, "babble", codec, roll))
    # sim arm: the acoustic core only (D4 pairs on non-reverb factors + measured
    # RT60, so the channel block adds nothing to the sim-vs-real question).
    for rt60 in _CORE_RT60:
        for snr in _CORE_SNR:
            for noise in _CORE_NOISE:
                conds.append(_cond(rt60, snr, noise, "none", 0.0, sim=True))

    seen, out = set(), []
    for c in conds:
        if c["name"] not in seen:
            seen.add(c["name"])
            out.append(c)
    return out


def build_rows(clips=None, models=MODELS, seed: int = 7,
               n_failed: int = 4) -> list[dict]:
    refs = load_references(clips=list(clips) if clips else AL_CLIPS)
    conds = grid()
    rows: list[dict] = []
    for model in models:
        for cond in conds:
            for i, (clip_id, ref) in enumerate(sorted(refs.items())):
                rng = random.Random(f"{seed}|{model}|{cond['name']}|{clip_id}")
                rows.append(make_row(clip_id, ref, cond, model, rng))
    # A handful of adapter failures, all in ONE clean cell — the trap
    # analysis/__init__ exists to avoid, so the dashboard gets to prove it too.
    clean = next(c for c in conds if c["rt60"] == 0.2 and c["snr_db"] == 25.0
                 and not c["sim"] and c["codec"] == "none")
    for model in models:
        for clip_id in sorted(refs)[:n_failed]:
            rows.append(failed_row(clip_id, clean, model))
    return rows


def write_csv(rows: list[dict], out: str) -> str:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MASTER_COLUMNS), extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in MASTER_COLUMNS})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic master table for the dashboard")
    ap.add_argument("--out", default=os.path.join(REPO, "results/synthetic_master.csv"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    rows = build_rows(seed=args.seed)
    write_csv(rows, args.out)
    n_cond = len({r["condition_name"] for r in rows})
    print(f"wrote {len(rows)} rows / {n_cond} conditions / "
          f"{len({r['model'] for r in rows})} models -> {args.out}")
    print(f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
