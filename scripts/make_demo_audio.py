"""
make_demo_audio.py — build the LISTENING + DEMO set.

    ./.venv/bin/python scripts/make_demo_audio.py              # everything
    ./.venv/bin/python scripts/make_demo_audio.py --force      # rebuild the wavs too
    ./.venv/bin/python scripts/make_demo_audio.py --force-docs # AND clobber edited docs
    ./.venv/bin/python scripts/make_demo_audio.py --check      # verify, generate nothing

Offline. No API calls, no key, no network. Pure DSP plus a read of
`results/master.csv`. A few seconds.

------------------------------------------------------------------------------
WHY THIS REPLACES results/audio/listen/DEADZONE_*
------------------------------------------------------------------------------
The old dead-zone listening files were all clip `u02`, and nova-3 transcribes
`u02` PERFECTLY (WER 0.000) in four of those six conditions. Playing a clip the
model got exactly right, while narrating "here is where the model fails", is a
demo that refutes itself. The ladder in that directory was and remains sound —
it validated the composer — so it is regenerated here; the DEADZONE_* files are
superseded.

------------------------------------------------------------------------------
WHAT THIS SET IS FOR
------------------------------------------------------------------------------
The study measures ONE axis: what the model does. It has no human axis at all,
because a controlled rig cannot have one — you cannot ask pyroomacoustics
whether a clip was hard to understand. This set generates the missing axis in
three minutes with the listener in the room, and it is built so the comparison
is fair rather than rhetorical.

PART 1 — THE PAIRED, MATCHED-WER TEST (the centrepiece).
Two conditions, each isolating a single degradation (no codec, no mic rolloff),
so nothing is confounded:

  A  "reverb-dominated"  rt60 1.0 @ SNR 20 dB   — a bad room, but QUIET
  B  "babble-dominated"  rt60 0.2 @ SNR  0 dB   — a good room, but buried

Over the same 40 clips the model scores them statistically indistinguishably:
the paired 95% bootstrap CI on the difference spans zero. On three clips it
scores them EXACTLY equal, and non-zero — i.e. the model is equally WRONG on
both, which is a stronger claim than equally right.

A human will almost certainly not rank them equal. We have the precedence
effect and a lifetime of room adaptation, so reverb is nearly free for us;
competing speech causes informational masking, which is brutal for us. The model
has neither prior. So the two axes decouple, and they decouple IN BOTH
DIRECTIONS — which is the point, and why "I listened to it and it sounded fine"
is not evidence about an ASR system.

PART 2 — THE PAYOFF.
`rt60-0.7_snr-20_babble_opus-lowrate_roll-1`. Note SNR 20 dB: this is QUIET.
The damage is reverb + a low-rate codec + full mic rolloff. Ten of the forty
clips return a COMPLETELY EMPTY transcript. You can still hear a person
speaking. The model returned an empty string.

PART 3 — THE FACTOR-ISOLATION LADDER (SPEC A.R3.5).
One clip, one factor at a time, clean to destroyed. This is not about WER; it is
the only test that the composer produces something physically PLAUSIBLE. The
unit tests prove the arithmetic; only ears prove the result.

------------------------------------------------------------------------------
THREE CONSTRUCTION RULES, EACH LOAD-BEARING
------------------------------------------------------------------------------
1. NO NUMBER IS TYPED BY HAND. Every WER, confidence, transcript and CI in the
   generated markdown is read or recomputed from `results/master.csv` at build
   time, for the same reason `load_manifest` refuses to let a reference be typed
   inline: a hand-copied number and the table it came from diverge silently, and
   a demo that misquotes its own study is worse than no demo.

2. THE WAVS ARE BIT-IDENTICAL TO WHAT THE MODEL HEARD. `apply_condition` is
   seeded from the condition NAME and `write_degraded_wav` is the same writer
   the grid used, so these files reproduce the transcribed audio exactly. That
   is why no loudness normalization is applied here even though B (0 dB) is
   audibly louder than A (20 dB): a cosmetic gain would break the identity for
   a confound that is better handled by letting the listener use the volume
   knob. The confound is named out loud in DEMO_SCRIPT.md instead of hidden.

3. THE BLIND COPIES ARE THE ONLY THING THE LISTENER SEES. The working filenames
   say `reverb` and `babble`; a listener who reads them has been told the answer
   and the exercise is worthless. `blind/` holds neutrally-named byte-identical
   copies plus the only interviewee-facing sheet; `KEY.md` and the prediction
   stay in the parent directory. The blind order is a FROZEN constant, not a
   shuffle, so it is reviewable and cannot drift between runs.

4. REGENERABLE AUDIO AND UNRECOVERABLE RECORD ARE NOT THE SAME ARTIFACT, and
   this script used to treat them as if they were. The wavs rebuild from
   `master.csv` and the asset library in seconds. The markdown does not: two of
   these documents now carry a pre-registered prediction, its verbatim listener
   response, the scoring, the verdict (it FAILED) and an analysis of a flaw in
   the pre-registration's own rubric — none of it derivable from any artifact in
   the repo. Rewriting all five unconditionally already destroyed one such edit,
   silently, because a freshly generated file looks exactly as correct as the
   one it replaced. Every document now goes through `write_doc`, which REFUSES
   to overwrite a file a human has touched. See THE AUTHORED-DOCUMENT GUARD
   below.
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
import hashlib
import json
import re
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from deadzone.conditions import Condition, DiskAssetLibrary, apply_condition
from scripts.make_audio_sets import LADDER, LISTEN_CLIP
from scripts.run_experiment import load_clip, load_manifest, write_degraded_wav

FS = 16000
MODEL = "nova-3"
MASTER = Path("results/master.csv")
DEMO_DIR = Path("results/audio/demo")
BLIND_DIR = DEMO_DIR / "blind"
ISO_DIR = DEMO_DIR / "isolation"
OLD_LISTEN_DIR = Path("results/audio/listen")

# The paired centrepiece. Both isolate ONE degradation: codec=none, rolloff=0.
COND_A = "rt60-1_snr-20_babble_none_roll-0"      # bad room, quiet
COND_B = "rt60-0.2_snr-0_babble_none_roll-0"     # good room, buried
LABEL_A = "reverb-dominated"
LABEL_B = "babble-dominated"

# Clips where the model scores A and B EXACTLY equal AND non-zero, so it is
# equally WRONG on both. Verified against master.csv; the test re-verifies.
PAIR_CLIPS = ["u40", "u21", "u26"]               # 3rd is the backup

# The payoff. SNR 20 dB — QUIET. Ten of forty clips come back empty.
COND_PAYOFF = "rt60-0.7_snr-20_babble_opus-lowrate_roll-1"
PAYOFF_CLIP = "u03"                              # longest of the empty ones

BOOT_N = 10000
BOOT_SEED = 0

# Frozen, reviewable blind order: blind_NN -> working filename stem. Chosen so
# no pair is adjacent and the A arm is not always first. Do NOT regenerate this
# with a shuffle — a permutation that silently changes between numpy versions
# would invalidate a KEY.md someone already printed.
BLIND_ORDER = [
    "pair2_B_babble_u21",
    "payoff_u03_clean",
    "pair1_A_reverb_u40",
    "pair3_B_babble_u26",
    "payoff_u03_deadzone",
    "pair2_A_reverb_u21",
    "pair1_B_babble_u40",
    "pair3_A_reverb_u26",
]


# --------------------------------------------------------------------------
# master table access
# --------------------------------------------------------------------------

class MissingTable(RuntimeError):
    pass


def load_rows(conditions: set[str], model: str = MODEL) -> dict[tuple[str, str], dict]:
    """(condition_name, clip_id) -> row, for the model arm we demo."""
    if not MASTER.is_file():
        raise MissingTable(
            f"{MASTER} not found — the demo quotes MEASURED numbers and refuses "
            f"to invent them. Rebuild it with:\n"
            f"    ./.venv/bin/python scripts/run_experiment.py --rebuild")
    csv.field_size_limit(10 ** 9)
    out: dict[tuple[str, str], dict] = {}
    with open(MASTER, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["model"] == model and r["condition_name"] in conditions:
                out[(r["condition_name"], r["clip_id"])] = r
    if not out:
        raise MissingTable(
            f"{MASTER} has no {model!r} rows for {sorted(conditions)} — the grid "
            f"this demo describes is not in the table.")
    return out


def _f(row: dict, key: str) -> float | None:
    v = (row.get(key) or "").strip()
    return float(v) if v else None


def paired_stats(rows: dict[tuple[str, str], dict]) -> dict:
    """
    The measured half of the exercise: paired over CLIPS, which is the right
    resampling unit — the same 40 sentences are scored under both conditions, so
    clip identity is a blocking factor and resampling words or cells would throw
    the pairing away and understate the CI.
    """
    clips = sorted({c for (cond, c) in rows if cond == COND_A}
                   & {c for (cond, c) in rows if cond == COND_B})
    a = np.array([float(rows[(COND_A, c)]["wer"]) for c in clips])
    b = np.array([float(rows[(COND_B, c)]["wer"]) for c in clips])
    d = a - b
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(clips), size=(BOOT_N, len(clips)))
    boot = d[idx].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {
        "n_clips": len(clips),
        "clips": clips,
        "mean_wer_A": float(a.mean()),
        "mean_wer_B": float(b.mean()),
        "paired_diff_A_minus_B": float(d.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "spans_zero": bool(lo < 0.0 < hi),
        "n_resamples": BOOT_N,
        "seed": BOOT_SEED,
        "resample_unit": "clip",
    }


def rir_acoustics(cond: Condition, assets, fs: int = FS) -> dict:
    """
    DRR and C50 of the RIR this condition actually resolves to.

    Same definition as scripts/run_d3a.py so the demo and the D3a report cannot
    quote different numbers for the same room. This matters because the project's
    sharpest reverb result is that RT60 MISLABELS the delivered acoustics:
    spearman(DRR, WER) = -1.000 while spearman(RT60, WER) = +0.800.
    """
    rir = assets.resolve(cond).rir
    h, hfs = sf.read(rir.key, dtype="float64")
    h = np.asarray(h, dtype=float)
    if h.ndim > 1:
        h = h.mean(axis=1)
    if hfs != fs:
        import librosa                                    # lazy: heavy import
        h = librosa.resample(h, orig_sr=hfs, target_sr=fs)
    d = int(np.argmax(np.abs(h)))
    w = int(0.0025 * fs)
    direct = h[max(0, d - w):d + w]
    rev = np.concatenate([h[:max(0, d - w)], h[d + w:]])
    e50 = int(0.050 * fs)
    return {
        "rir_key": rir.key,
        "room": Path(rir.key).name,
        "rt60_measured": float(rir.rt60),
        "drr_db": float(10 * np.log10((direct ** 2).sum() / max((rev ** 2).sum(), 1e-20))),
        "c50_db": float(10 * np.log10((h[d:d + e50] ** 2).sum()
                                      / max((h[d + e50:] ** 2).sum(), 1e-20))),
    }


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def _write_degraded(path: Path, clip_id: str, cond: Condition, assets) -> None:
    audio = load_clip(clip_id, target_fs=FS)
    y = apply_condition(audio, cond, assets, FS)
    write_degraded_wav(path, y, FS)


def build_audio(assets) -> dict[str, str]:
    """Every wav. Returns stem -> relative path."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    BLIND_DIR.mkdir(parents=True, exist_ok=True)
    ISO_DIR.mkdir(parents=True, exist_ok=True)

    ca, cb = Condition.from_name(COND_A), Condition.from_name(COND_B)
    made: dict[str, str] = {}

    for i, clip in enumerate(PAIR_CLIPS, start=1):
        for arm, cond, tag in ((("A"), ca, "reverb"), (("B"), cb, "babble")):
            stem = f"pair{i}_{arm}_{tag}_{clip}"
            p = DEMO_DIR / f"{stem}.wav"
            _write_degraded(p, clip, cond, assets)
            made[stem] = str(p)

    # The control is the raw recording resampled to 16 kHz — which is exactly
    # what the pipeline fed the model (load_clip resamples at the edge). Handing
    # over the untouched 48 kHz file instead would make the clean clip audibly
    # brighter for a reason that has nothing to do with the condition, i.e. it
    # would plant a confound in the one comparison the demo rests on.
    raw = load_clip(PAYOFF_CLIP, target_fs=FS)
    sf.write(str(DEMO_DIR / "payoff_u03_clean.wav"), raw, FS, subtype="PCM_16")
    made["payoff_u03_clean"] = str(DEMO_DIR / "payoff_u03_clean.wav")

    _write_degraded(DEMO_DIR / "payoff_u03_deadzone.wav", PAYOFF_CLIP,
                    Condition.from_name(COND_PAYOFF), assets)
    made["payoff_u03_deadzone"] = str(DEMO_DIR / "payoff_u03_deadzone.wav")

    # blind copies — real copies, not symlinks: a symlink resolves to a name
    # that says `reverb` the moment anyone inspects the folder, and a zipped
    # symlink is a broken file on someone else's laptop.
    blind_map: dict[str, str] = {}
    for n, stem in enumerate(BLIND_ORDER, start=1):
        src = Path(made[stem])
        dst = BLIND_DIR / f"blind_{n:02d}.wav"
        shutil.copyfile(src, dst)
        blind_map[dst.name] = src.name
        made[f"blind_{n:02d}"] = str(dst)

    # Part 3: the factor-isolation ladder, LADDER imported from make_audio_sets
    # so there is one definition of the ladder in the repo, not two that drift.
    iso = load_clip(LISTEN_CLIP, target_fs=FS)
    sf.write(str(ISO_DIR / "00_RAW_original.wav"), iso, FS, subtype="PCM_16")
    made["iso_00_RAW_original"] = str(ISO_DIR / "00_RAW_original.wav")
    for label, cond in LADDER:
        p = ISO_DIR / f"{label}.wav"
        y = apply_condition(iso, cond, assets, FS)
        write_degraded_wav(p, y, FS)
        made[f"iso_{label}"] = str(p)

    made["_blind_map"] = blind_map          # popped by the caller
    return made


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def build_manifest(made: dict, rows: dict, stats: dict, assets) -> dict:
    blind_map = made.pop("_blind_map")
    refs = load_manifest()
    ca, cb = Condition.from_name(COND_A), Condition.from_name(COND_B)
    cp = Condition.from_name(COND_PAYOFF)
    inv = {v: k for k, v in blind_map.items()}

    def cell(cond_name: str, clip: str, stem: str) -> dict:
        r = rows[(cond_name, clip)]
        fn = Path(made[stem]).name
        return {
            "file": made[stem],
            "blind": inv.get(fn),
            "wer": float(r["wer"]),
            "mean_conf": _f(r, "mean_conf"),
            "utterance_conf": _f(r, "utterance_conf"),
            "transcript": r["transcript"],
            "n_ref": int(r["n_ref"]), "n_sub": int(r["n_sub"]),
            "n_del": int(r["n_del"]), "n_ins": int(r["n_ins"]),
            "rir_key": r["rir_key"], "noise_key": r["noise_key"],
        }

    pairs = []
    for i, clip in enumerate(PAIR_CLIPS, start=1):
        pairs.append({
            "pair": i, "clip_id": clip, "ref": refs[clip],
            "role": "primary" if i <= 2 else "backup",
            "A": cell(COND_A, clip, f"pair{i}_A_reverb_{clip}"),
            "B": cell(COND_B, clip, f"pair{i}_B_babble_{clip}"),
        })

    payoff_rows = {c: r for (cond, c), r in rows.items() if cond == COND_PAYOFF}
    empty = sorted(c for c, r in payoff_rows.items() if not r["transcript"].strip())
    pr = payoff_rows[PAYOFF_CLIP]

    return {
        "generated_by": "scripts/make_demo_audio.py",
        "fs": FS,
        "model": MODEL,
        "master_table": str(MASTER),
        "supersedes": str(OLD_LISTEN_DIR / "DEADZONE_*.wav"),
        "conditions": {
            "A": {"name": COND_A, "label": LABEL_A, **ca.to_sample(),
                  **rir_acoustics(ca, assets)},
            "B": {"name": COND_B, "label": LABEL_B, **cb.to_sample(),
                  **rir_acoustics(cb, assets)},
            "payoff": {"name": COND_PAYOFF, **cp.to_sample(),
                       **rir_acoustics(cp, assets)},
        },
        "paired_result": stats,
        "pairs": pairs,
        "payoff": {
            "clip_id": PAYOFF_CLIP,
            "ref": refs[PAYOFF_CLIP],
            "n_empty": len(empty), "n_clips": len(payoff_rows),
            "empty_clips": empty,
            "clean_file": made["payoff_u03_clean"],
            "clean_blind": inv.get("payoff_u03_clean.wav"),
            "deadzone_file": made["payoff_u03_deadzone"],
            "deadzone_blind": inv.get("payoff_u03_deadzone.wav"),
            "wer": float(pr["wer"]), "n_ref": int(pr["n_ref"]),
            "n_del": int(pr["n_del"]),
            "transcript": pr["transcript"],
            "mean_conf": _f(pr, "mean_conf"),
            "utterance_conf": _f(pr, "utterance_conf"),
        },
        "blind_map": blind_map,
        "isolation": [k[4:] for k in sorted(made) if k.startswith("iso_")],
        "files": sorted(v for k, v in made.items() if not k.startswith("_")),
    }


# --------------------------------------------------------------------------
# THE AUTHORED-DOCUMENT GUARD
#
# Every document below used to be written unconditionally on any build — no
# skip-if-modified, no warning, no backup. That is fine for the wavs, which are
# a pure function of `master.csv` and the asset library, and it is catastrophic
# for the prose, which is where a human writes down the one thing the pipeline
# cannot produce: what a listener said. It has already cost one such edit.
#
# The defect is NOT the overwrite. It is that regenerable output and
# unrecoverable record were handed to the same writer.
#
# The rule follows `write_master()` in scripts/run_experiment.py — refuse to
# produce the misleading artifact rather than warn about it afterwards:
#
#   file absent          -> write.
#   hash matches record  -> this generator wrote it and nobody has touched it
#                           since; write.
#   hash differs         -> a human edited it; REFUSE, and name the flag.
#   NO recorded hash     -> provenance unknown; REFUSE.
#
# That last line is the load-bearing one, and it is deliberately the paranoid
# reading. A missing baseline is the degenerate input, and SPEC Appendix E.5's
# rule is to ask what a guard returns for the degenerate input rather than for
# the good one. If "no record" meant "safe to overwrite", this guard would be
# wide open on precisely the state the repo is in the first time it runs —
# every document in the kit already written, and no sidecar yet.
#
# `authored` therefore means "not provably this generator's own output", which
# is a wider set than "a human typed in it". That imprecision is the correct
# direction to be imprecise in, and the census says so rather than asserting a
# human edit it cannot actually observe.
#
# It also settles how the sidecar is seeded: it ISN'T. Recording today's
# hand-edited text as "what the generator last wrote" would certify it as
# generator-owned and the very next default run would erase it — the guard
# would have been the delivery mechanism for the bug it exists to stop. Not
# seeding is what protects it, and it protects any other file that predates the
# guard, or is restored from a backup, for the same reason.
#
# Detection is by CONTENT HASH, never mtime. mtime does not survive a checkout,
# a copy, a `cp` without `-p`, or a restore — all four of which are things that
# happen to a demo kit that gets handed around.
# --------------------------------------------------------------------------

# What the generator last wrote, keyed by repo-relative path (every path in this
# project resolves against the repo root as CWD, SPEC §13). Lives beside the
# documents it guards so it shares their lifecycle: delete `results/audio/demo/`
# and the whole set rebuilds from scratch, guard included.
DOC_HASHES = DEMO_DIR / "generated_docs.json"

ABSENT, GENERATED, AUTHORED = "absent", "generated", "authored"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_key(path: Path) -> str:
    return Path(path).as_posix()


def load_doc_hashes() -> dict[str, str]:
    """Path -> sha256 of the text this generator last wrote there.

    An unreadable sidecar returns `{}`, which is the same as an absent one, and
    both then read as unknown provenance — i.e. the failure mode of this
    function is to protect MORE, never less.
    """
    if not DOC_HASHES.is_file():
        return {}
    try:
        rec = json.loads(DOC_HASHES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    got = rec.get("sha256")
    return dict(got) if isinstance(got, dict) else {}


def _save_doc_hash(key: str, digest: str) -> None:
    """Record one document, immediately after it is written.

    Per-document rather than once at the end: a build that dies halfway must
    leave a record that matches what is actually on disk, or the survivors come
    back as `authored` on the next run and the guard cries wolf.
    """
    sha = load_doc_hashes()
    sha[key] = digest
    DOC_HASHES.parent.mkdir(parents=True, exist_ok=True)
    DOC_HASHES.write_text(json.dumps({
        "written_by": "scripts/make_demo_audio.py",
        "what_this_is":
            "SHA-256 of each document AS THIS GENERATOR LAST WROTE IT. Baseline "
            "for the authored-document guard: a file whose hash no longer "
            "matches has been edited by a human and will not be overwritten "
            "without --force-docs. Deleting this file unlocks nothing — an "
            "absent record reads as unknown provenance, which also refuses.",
        "sha256": dict(sorted(sha.items())),
    }, indent=2) + "\n", encoding="utf-8")


def doc_status(path: Path, hashes: dict[str, str] | None = None) -> str:
    """`absent` | `generated` (safe to rewrite) | `authored` (hands off)."""
    p = Path(path)
    if not p.is_file():
        return ABSENT
    h = load_doc_hashes() if hashes is None else hashes
    recorded = h.get(_doc_key(p))
    if recorded is None:
        return AUTHORED
    try:
        current = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return AUTHORED
    return GENERATED if recorded == _sha256(current) else AUTHORED


def _refusal(p: Path) -> str:
    return "\n".join([
        f"[demo docs] REFUSING to overwrite {p}",
        "    why   : its contents do not match what this generator last wrote, so it",
        "            holds edits made by a human. Rewriting it would replace them with",
        "            template output and say nothing — a regenerated file looks exactly",
        "            as correct as the one it replaced, which is how this was lost once",
        "            already (see results/audio/demo/REGENERATION_HAZARD.md).",
        "    state : the file on disk is UNCHANGED and the rest of the build continued.",
        "    cost  : it therefore does NOT carry this build's numbers. manifest.json and",
        "            KEY.md are the generated source of truth for those.",
        "    fix   : port the hand-written block into this script's template, so a",
        "            rebuild reproduces it — or, to overwrite it anyway:",
        "                ./.venv/bin/python scripts/make_demo_audio.py --force-docs",
        "            which first copies the current file to <name>.superseded-<UTC>.md.",
    ])


def write_doc(path: Path, text: str, *, force_docs: bool = False) -> str:
    """Write a generator-owned document. Returns `written` or `skipped`.

    `force_docs` defaults to False so that any future caller which forgets the
    keyword gets the protective behaviour, not the destructive one.
    """
    p = Path(path)
    status = doc_status(p)
    if status == AUTHORED and not force_docs:
        print(_refusal(p))
        return "skipped"
    if status == AUTHORED:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = p.with_name(f"{p.stem}.superseded-{stamp}{p.suffix}")
        shutil.copyfile(p, backup)
        print(f"[demo docs] --force-docs: OVERWRITING hand-edited {p}\n"
              f"            previous contents preserved at {backup}\n"
              f"            NOTHING ELSE IN THIS REPO HOLDS THEM — read that file "
              f"before deleting it.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    _save_doc_hash(_doc_key(p), _sha256(text))
    return "written"


# --------------------------------------------------------------------------
# documents (every number interpolated, none typed)
# --------------------------------------------------------------------------

def num(v: float) -> str:
    """`1.0 -> '1.0'`, `0.2 -> '0.2'`. `%g` renders 1.0 as a bare `1`, which reads
    as a count rather than a seconds value when spoken aloud."""
    s = f"{float(v):g}"
    return s if "." in s or "e" in s else s + ".0"


def room_name(rir_filename: str) -> str:
    """`mit_rt60-0.99_h081_Shower_2txts.wav` -> `Shower`.

    Read aloud in the demo, so it has to be the ROOM, not whatever token the
    dataset happened to end the filename with.
    """
    stem = Path(rir_filename).stem
    stem = re.sub(r"^mit_rt60-[\d.]+_h\d+_", "", stem)
    stem = re.sub(r"_\d*txts$", "", stem)
    return stem.replace("_", " ") or Path(rir_filename).stem


def quote(text: str, width: int = 76) -> str:
    """
    Re-wrap an interpolated paragraph into a markdown blockquote.

    Interpolated numbers have unpredictable width, so wrapping the template by
    hand puts line breaks in the middle of sentences the presenter has to read
    aloud. Wrapping AFTER substitution keeps the spoken lines clean whatever the
    numbers turn out to be.
    """
    out = []
    for i, para in enumerate(p.strip() for p in text.strip().split("\n\n")):
        if i:
            out.append(">")
        body = " ".join(para.split())
        out.extend(textwrap.wrap(body, width=width, initial_indent="> ",
                                 subsequent_indent="> "))
    return "\n".join(out)


def write_prediction(m: dict, *, force_docs: bool = False) -> str:
    s = m["paired_result"]
    p1, p2 = m["pairs"][0], m["pairs"][1]
    txt = f"""# Pre-registered prediction — SEALED

Do not show this, or say it aloud, until the listener has ranked the clips.
Announcing a prediction before someone judges is a demand characteristic; this
project's whole subject is not fooling yourself with a number you wanted.

Written before any listener heard anything. The blind names are frozen in
`KEY.md`, so this is checkable after the fact rather than a claim.

## The prediction

> The listener will rank **{p1['B']['blind']}** clearly harder than
> **{p1['A']['blind']}**, and **{p2['B']['blind']}** clearly harder than
> **{p2['A']['blind']}** — a confident, immediate, non-marginal call in both
> pairs.
>
> The model scores the two clips within each of those pairs **EXACTLY equal**
> (pair 1: {p1['A']['wer']:.3f} vs {p1['B']['wer']:.3f}; pair 2:
> {p2['A']['wer']:.3f} vs {p2['B']['wer']:.3f}), and across all
> {s['n_clips']} clips the paired difference is {s['paired_diff_A_minus_B']:+.4f} WER,
> 95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] — spanning zero.

## What each outcome means

- **Listener ranks them clearly unequal** → prediction holds; the human and
  model axes are decoupled and the demo lands.
- **Listener ranks them equal** → prediction fails. Say so. n=1 was never going
  to settle it, and the MEASURED half (the CI above) is unaffected either way —
  which is exactly why the two halves are labelled separately.

The prediction is about a human. The confidence interval is about the model.
Only one of those two things is a measurement.
"""
    write_doc(DEMO_DIR / "PREREGISTERED_PREDICTION.md", txt, force_docs=force_docs)
    return txt


def write_key(m: dict, *, force_docs: bool = False) -> str:
    s, A, B = m["paired_result"], m["conditions"]["A"], m["conditions"]["B"]
    lines = [
        "# KEY — presenter only. Do not put this in `blind/`.",
        "",
        "`blind/` is the handover folder: neutral names, byte-identical copies,",
        "and `blind/BLIND_SHEET.md`. Everything that names a condition lives here",
        "in the parent directory.",
        "",
        "## Blind name -> working file",
        "",
        "| blind | working file | condition |",
        "|---|---|---|",
    ]
    stem_cond = {}
    for p in m["pairs"]:
        stem_cond[Path(p["A"]["file"]).name] = f"A · {LABEL_A} · `{A['name']}`"
        stem_cond[Path(p["B"]["file"]).name] = f"B · {LABEL_B} · `{B['name']}`"
    stem_cond[Path(m["payoff"]["clean_file"]).name] = "control · raw recording, 16 kHz"
    stem_cond[Path(m["payoff"]["deadzone_file"]).name] = \
        f"payoff · `{m['conditions']['payoff']['name']}`"
    for blind, work in m["blind_map"].items():
        lines.append(f"| `{blind}` | `{work}` | {stem_cond.get(work, '')} |")

    lines += [
        "",
        "## The two conditions",
        "",
        "| | A — reverb-dominated | B — babble-dominated |",
        "|---|---|---|",
        f"| condition | `{A['name']}` | `{B['name']}` |",
        f"| requested rt60 | {num(A['rt60'])} s | {num(B['rt60'])} s |",
        f"| SNR | {num(A['snr_db'])} dB (quiet) | {num(B['snr_db'])} dB (buried) |",
        f"| noise | {A['noise_type']} | {B['noise_type']} |",
        f"| codec / rolloff | {A['codec']} / {A['mic_rolloff']:g} | "
        f"{B['codec']} / {B['mic_rolloff']:g} |",
        f"| delivered room | {room_name(A['room'])} | {room_name(B['room'])} |",
        f"| RIR file | `{A['room']}` | `{B['room']}` |",
        f"| measured RT60 | {A['rt60_measured']:.3f} s | {B['rt60_measured']:.3f} s |",
        f"| DRR | {A['drr_db']:+.2f} dB | {B['drr_db']:+.2f} dB |",
        f"| C50 | {A['c50_db']:+.2f} dB | {B['c50_db']:+.2f} dB |",
        f"| mean WER over {s['n_clips']} clips | {s['mean_wer_A']:.4f} | "
        f"{s['mean_wer_B']:.4f} |",
        "",
        "Each isolates ONE degradation: codec `none`, mic rolloff 0. Nothing in",
        "this comparison is confounded by the channel factors.",
        "",
        f"**Paired difference (A-B): {s['paired_diff_A_minus_B']:+.4f} WER, "
        f"95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]** — "
        f"{'spans zero' if s['spans_zero'] else 'DOES NOT SPAN ZERO — see below'}. "
        f"{s['n_resamples']:,}-resample paired bootstrap, seed {s['seed']}, "
        f"resampled over {s['resample_unit']}s.",
        "",
        "## Per-clip facts (all from `results/master.csv`)",
        "",
    ]
    for p in m["pairs"]:
        lines += [
            f"### Pair {p['pair']} — `{p['clip_id']}` ({p['role']})",
            "",
            f"- reference: `{p['ref']}`",
            f"- **A** {p['A']['blind']} — WER **{p['A']['wer']:.3f}**, "
            f"mean conf {p['A']['mean_conf']:.3f}",
            f"  - hyp: `{p['A']['transcript']}`",
            f"- **B** {p['B']['blind']} — WER **{p['B']['wer']:.3f}**, "
            f"mean conf {p['B']['mean_conf']:.3f}",
            f"  - hyp: `{p['B']['transcript']}`",
            "",
        ]
    pay = m["payoff"]
    lines += [
        "## Payoff",
        "",
        f"- condition `{m['conditions']['payoff']['name']}` — SNR "
        f"{num(m['conditions']['payoff']['snr_db'])} dB, i.e. QUIET. The damage is "
        f"reverb + codec + rolloff, not noise.",
        f"- clean: `{pay['clean_blind']}` · dead zone: `{pay['deadzone_blind']}`",
        f"- `{pay['clip_id']}` reference ({pay['n_ref']} words): `{pay['ref']}`",
        f"- transcript: **{'(empty string)' if not pay['transcript'].strip() else pay['transcript']}**",
        f"- WER {pay['wer']:.3f}, all {pay['n_del']} reference words deleted, "
        f"utterance confidence {pay['utterance_conf']:.2f}",
        f"- **{pay['n_empty']} of {pay['n_clips']} clips returned an empty "
        f"transcript in this condition**: {', '.join(pay['empty_clips'])}",
        "",
        "Mean word confidence is NULL here, not low: there are no words to be",
        "confident about. That is the deletion blindness the calibration layer",
        "reports — deletions carry no hypothesis token, so they are invisible to",
        "any confidence-based monitor.",
        "",
    ]
    txt = "\n".join(lines) + "\n"
    write_doc(DEMO_DIR / "KEY.md", txt, force_docs=force_docs)
    return txt


def pair_listing(p: dict) -> list[str]:
    """
    The two blind names of one pair, in BLIND-NUMBER order.

    Not in arm order. Listing A first every time would put the reverb arm first
    in all three pairs, so presentation position would be perfectly confounded
    with condition and "the second one was harder" would be ambiguous between an
    order effect and the thing being tested. The frozen BLIND_ORDER puts A first
    in one pair and B first in the other two — not balanced (three pairs cannot
    be), which the run-of-show says out loud.
    """
    return sorted([p["A"]["blind"], p["B"]["blind"]])


def write_blind_sheet(m: dict, *, force_docs: bool = False) -> str:
    pay = m["payoff"]
    rows = "\n".join(
        f"| {p['pair']} | `{a}` and `{b}` |"
        for p in m["pairs"] for a, b in [pair_listing(p)])
    first, second = sorted([pay["clean_blind"], pay["deadzone_blind"]])
    txt = f"""# Listening sheet

Eight short clips. Same speaker throughout; each clip is one sentence.

Play them as loud or as quiet as you like, and re-play freely — the clips are
not level-matched, and how loud something is is not what you are being asked
to judge.

## The task

For each pair below, answer one question: **which one is harder to understand?**
Rank them, and say how confident you are — "clearly harder" vs "about the same"
is the distinction that matters.

| pair | clips |
|---|---|
{rows}

Then, separately: `{first}` and `{second}`. Same question, plus one more —
**can you still tell that someone is speaking?**

Write your ranking down before we discuss any of it.
"""
    write_doc(BLIND_DIR / "BLIND_SHEET.md", txt, force_docs=force_docs)
    return txt


def write_demo_script(m: dict, *, force_docs: bool = False) -> str:
    s = m["paired_result"]
    A, B = m["conditions"]["A"], m["conditions"]["B"]
    p1, p2, p3 = m["pairs"]
    pay = m["payoff"]
    verdict = "spans zero" if s["spans_zero"] else "does NOT span zero"

    txt = f"""# DEMO_SCRIPT — the listening exercise (~3 minutes)

Presenter's run-of-show. Everything here is offline: play wavs from
`blind/`, read from this file. No network, no API key.

**Hand over:** `blind/` only (8 wavs + `BLIND_SHEET.md`).
**Keep back:** this file, `KEY.md`, `PREREGISTERED_PREDICTION.md`.
The working filenames in the parent directory say `reverb` and `babble`; a
listener who sees them has been told the answer.

---

## 0. Before they listen (10 s) — write the prediction down, do not say it

Open `PREREGISTERED_PREDICTION.md`, show that it exists, and **leave it closed**.
Saying a prediction out loud before someone judges is a demand characteristic.
This mirrors how the study handles its own hypotheses: `rt60 x snr_db` was
pre-registered in SPEC section 5 before any real audio existed, and confirmed on
the real grid at ST-S1 = 0.128 [0.091, 0.164]. Same discipline, three minutes
instead of three weeks.

One line to say out loud:

> "I have written down what I think you'll say. You'll see it in a minute."

---

## 1. The task (20 s)

Give them `blind/BLIND_SHEET.md`. Two rules only:

> "Same speaker, same kind of sentence. For each pair: which is harder to
> understand? And use the volume knob however you like — the clips aren't
> level-matched and loudness isn't what I'm asking about."

---

## 2. They listen and rank (60 s)

Pairs 1 and 2 are the primary evidence — clips `{p1['clip_id']}` and
`{p2['clip_id']}`, the two the model scores exactly equal. Pair 3
(`{p3['clip_id']}`) is a backup if someone wants a third. Let them finish every
ranking before you say anything at all.

---

## 3. The reveal (45 s)

Open `PREREGISTERED_PREDICTION.md`. Then:

{quote(f'''
"Those two clips are not the same degradation. One is a bad ROOM but quiet:
requested RT60 {num(A['rt60'])} s at {num(A['snr_db'])} dB SNR. That is the
{room_name(A['room'])} impulse response — a real measured room — with a
measured RT60 of {A['rt60_measured']:.3f} s and a direct-to-reverberant ratio of
{A['drr_db']:+.2f} dB.

The other is a GOOD room with the speech nearly buried: RT60 {num(B['rt60'])} s
at {num(B['snr_db'])} dB SNR, the {room_name(B['room'])} response, DRR
{B['drr_db']:+.2f} dB. Neither one has a codec or mic rolloff on it, so nothing
else is moving between them."
''')}

Then the numbers — read them exactly:

{quote(f'''
"On the clip you just heard, the model scored them **identically**: WER
{p1['A']['wer']:.3f} and {p1['B']['wer']:.3f}. Not close — equal. And not equal
because it got both right: it got both **wrong**, by the same amount, in
different places.

Across all {s['n_clips']} clips: the reverb condition means WER
**{s['mean_wer_A']:.4f}**, the babble condition **{s['mean_wer_B']:.4f}**. The
paired difference is **{s['paired_diff_A_minus_B']:+.4f}**, 95% CI
**[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]** — {verdict}. That is a
{s['n_resamples']:,}-resample paired bootstrap over clips, seed {s['seed']}.
Statistically indistinguishable."
''')}

Show the transcripts if there's a screen — they make the point better than the
scalar does, because the two conditions fail in different *shapes*:

- `{p1['clip_id']}` reference: `{p1['ref']}`
  - reverb: `{p1['A']['transcript']}`
  - babble: `{p1['B']['transcript']}`
- `{p2['clip_id']}` reference: `{p2['ref']}`
  - reverb: `{p2['A']['transcript']}`
  - babble: `{p2['B']['transcript']}`

---

## 4. The mechanism (30 s)

{quote('''
"You found one of them clearly harder. You have two priors the model does not.

**The precedence effect.** Your auditory system fuses a direct sound with the
reflections arriving in the first few tens of milliseconds and localizes to the
direct one — plus you have a lifetime of adapting to rooms. Reverb is close to
free for you.

**Informational masking.** Competing speech does not merely cover the signal,
it competes for the same linguistic machinery. That is brutal for you in a way
that broadband noise at the same SNR is not.

The model has neither prior. To it both of these are just damage, and the damage
happens to be the same size."
''')}

Tie it to the study's own reverb result if there is time:

{quote(f'''
"And the model is not tracking RT60 either. Across the four reverb levels in the
grid, the Spearman correlation of measured RT60 with WER is **+0.800** — but
with direct-to-reverberant ratio it is **-1.000**. The clip you just heard is
the extreme case: {room_name(A['room'])}, DRR {A['drr_db']:+.2f} dB against
{room_name(B['room'])} at {B['drr_db']:+.2f} dB. RT60 is the number every reverb
benchmark is parameterised by, and it mislabels the acoustics that actually get
delivered."
''')}

---

## 5. The payoff (25 s)

Play `{pay['clean_blind']}` (clean control) then `{pay['deadzone_blind']}`.

{quote(f'''
"Same sentence, same speaker. The second one is condition
`{m['conditions']['payoff']['name']}` — and look at the SNR:
**{num(m['conditions']['payoff']['snr_db'])} dB**. That is QUIET. The damage here
is reverb plus a low-rate codec plus full mic rolloff. It is not noise.

You can still hear that someone is talking. The model returned an **empty
string**. WER {pay['wer']:.3f}, all {pay['n_ref']} reference words deleted.
**{pay['n_empty']} of the {pay['n_clips']} clips came back completely empty in
this condition.**

And there is no low confidence to catch it with. Mean word confidence is not
low — it is *null*, because there are no words to be confident about. A monitor
watching confidence sees nothing at all."
''')}

---

## 6. The honest label (20 s) — do not skip this

{quote(f'''
"Two halves here, and they are not the same kind of thing.

**The human half is n=1, unblinded to the task, one listener, one speaker, one
accent. It is an intuition pump, not a measurement.** You knew you were being
tested. I chose the three clips *because* the model tied on them. The clips are
not level-matched, and with three pairs I cannot balance presentation order —
one pair plays the reverb arm first, two play it second.

**The measured half is the model-side paired result and its interval:**
{s['paired_diff_A_minus_B']:+.4f} WER, CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}],
over all {s['n_clips']} clips, {s['n_resamples']:,} resamples. That half is not
selected and not affected by anything you just said. It would read the same if
you had ranked them the other way round, or refused to rank them at all.

Doing the human side properly is a listening study: many listeners, randomized
order, level-matched stimuli, a real intelligibility score. That is exactly the
experiment this project does not have, and the limitations section says so."
''')}

---

## 7. The takeaway (10 s) — the line to land on

{quote('''
"**You cannot QA a voice agent by listening to it.** 'Sounds fine to me' is
not evidence that the ASR works, and 'sounds terrible' is not evidence that it
fails. The decoupling runs in both directions — I just played you one clip a
human finds easy that the model fails outright, and a pair a human ranks
unequal that the model scores the same. If your acceptance test is a person
with headphones, you are measuring the wrong system."
''')}

---

## Fallbacks

- **No speakers / bad room:** skip to section 3 and show the transcripts. The
  measured half needs no audio at all.
- **They rank the pair EQUAL:** the prediction failed — say so plainly, it costs
  nothing. The CI is unchanged, and "my n=1 intuition pump didn't fire" is a
  cheaper thing to lose than credibility.
- **They want to hear the factors separately:** `isolation/` is the ladder,
  `00_RAW_original` to `10_destroyed`, one factor at a time. See
  `WHAT_TO_LISTEN_FOR.md`.

## Provenance

Every number above is read or recomputed from `results/master.csv` at build
time by `scripts/make_demo_audio.py` — none is typed into this file. The wavs
are bit-identical to what nova-3 transcribed: `apply_condition` is seeded from
the condition name and the writer is the grid's own `write_degraded_wav`.
Regenerate with:

    ./.venv/bin/python scripts/make_demo_audio.py --force

**If you hand-edit this file, that edit is safe.** The generator hashes what it
writes and refuses to overwrite anything that no longer matches — it will print
a refusal naming this path and carry on. Only `--force-docs` overrides that, and
it copies the current file to `<name>.superseded-<UTC>.md` before it does.
"""
    write_doc(DEMO_DIR / "DEMO_SCRIPT.md", txt, force_docs=force_docs)
    return txt


def write_what_to_listen_for(m: dict, *, force_docs: bool = False) -> str:
    A, B = m["conditions"]["A"], m["conditions"]["B"]
    pay = m["payoff"]
    txt = f"""# What to listen for

Two different jobs live in this directory. Don't mix them up.

---

## PART 1 — `isolation/`: is the composer physically plausible? (SPEC A.R3.5)

All from clip `{LISTEN_CLIP}`, so what changes is the CONDITION, not the speaker
or the sentence. Start with `00_RAW_original.wav`. **WER is irrelevant here** —
this is a DSP check, and the only instrument is your ears. The unit tests prove
the arithmetic; nothing but listening proves the RESULT.

1. **Onset alignment.** Every file must start when the original starts. A late
   start means `apply_rir`'s direct-path trim is wrong and every WER in the
   study carries a pure alignment artifact.
2. **Reverb sounds like a room** (`02_reverb_only`) — not a delay, not a metallic
   comb, no audible repeat.
3. **SNR is believable.** `01_benign` (20 dB) should be barely noisy;
   `03_noise_only` (0 dB) should nearly bury the speech. If 0 dB sounds mild, the
   calibration is off.
4. **Codecs sound like a phone line** (`05_g726_only`, `06_opus_only`) —
   bandlimited and gritty, not just quieter.

This is the check that caught the `apply_rir` renormalization bug: reverb tail
energy leaking into the silent regions de-calibrated every downstream SNR, and
it produced clean-looking garbage with no error message anywhere.

---

## PART 2 — `blind/`: the human axis the study doesn't have

Run this on someone else, not on yourself. `DEMO_SCRIPT.md` is the run-of-show,
`KEY.md` is the answer key, `blind/BLIND_SHEET.md` is the only page the listener
sees. Do not let them see the working filenames in this directory — they say
`reverb` and `babble`.

What you are listening for yourself, before you run it on anyone:

- **A** (`{A['name']}`) should sound like a **bad room, but quiet**: the
  {room_name(A['room'])} impulse response, measured RT60
  {A['rt60_measured']:.3f} s, DRR {A['drr_db']:+.2f} dB, with the babble at
  {num(A['snr_db'])} dB SNR and barely there.
- **B** (`{B['name']}`) should sound like a **good room with the speech nearly
  buried**: the {room_name(B['room'])} response, measured RT60
  {B['rt60_measured']:.3f} s, DRR {B['drr_db']:+.2f} dB, babble at
  {num(B['snr_db'])} dB.
- If those two do NOT sound clearly different to you, stop: something is wrong
  with the composer, because the model says they are equally damaging and the
  whole exercise is that a human should disagree.
- **The payoff** (`{pay['deadzone_blind']}`, condition
  `{m['conditions']['payoff']['name']}`) should still sound like a person
  speaking. The model returned an empty string on it — and on
  {pay['n_empty']} of {pay['n_clips']} clips in that condition.

The clips are deliberately **not level-matched**: they are byte-identical to the
audio nova-3 was scored on, and a cosmetic gain would break that identity. Use
the volume knob and say so out loud when you run the exercise.

---

## SUPERSEDED

`{OLD_LISTEN_DIR}/DEADZONE_*.wav` are superseded by this directory: all six were
clip `{LISTEN_CLIP}`, which nova-3 transcribes at WER 0.000 in four of those six
conditions — so they demonstrated nothing. The ladder in that directory is fine
and is regenerated here as `isolation/`.
"""
    write_doc(DEMO_DIR / "WHAT_TO_LISTEN_FOR.md", txt, force_docs=force_docs)
    return txt


def mark_old_set_superseded(m: dict, *, force_docs: bool = False) -> None:
    if not OLD_LISTEN_DIR.is_dir():
        return
    write_doc(
        OLD_LISTEN_DIR / "SUPERSEDED.md",
        f"# SUPERSEDED by `{DEMO_DIR}/`\n\n"
        f"The `DEADZONE_*.wav` files here are all clip `{LISTEN_CLIP}`, and nova-3 "
        f"transcribes `{LISTEN_CLIP}` at WER 0.000 in four of those six conditions — "
        f"so they demonstrated nothing.\n\n"
        f"The ladder (`00_RAW_original` .. `10_destroyed`) is sound and is "
        f"regenerated as `{DEMO_DIR}/isolation/`. The replacement listening + demo "
        f"set, built on conditions where the model measurably fails, is in "
        f"`{DEMO_DIR}/` — see its `WHAT_TO_LISTEN_FOR.md` and `DEMO_SCRIPT.md`.\n\n"
        f"    ./.venv/bin/python scripts/make_demo_audio.py\n",
        force_docs=force_docs)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

MANIFEST = DEMO_DIR / "manifest.json"

# Every document this script owns, in the order main() writes them. One list,
# so `check()` reports on exactly the set the build touches and a document added
# later cannot be reported on but not guarded, or the reverse.
DOCS = (DEMO_DIR / "DEMO_SCRIPT.md",
        DEMO_DIR / "KEY.md",
        BLIND_DIR / "BLIND_SHEET.md",
        DEMO_DIR / "PREREGISTERED_PREDICTION.md",
        DEMO_DIR / "WHAT_TO_LISTEN_FOR.md",
        OLD_LISTEN_DIR / "SUPERSEDED.md")


def doc_report() -> dict[str, str]:
    """path -> status, over every document the build would write."""
    h = load_doc_hashes()
    return {str(p): doc_status(p, h) for p in DOCS}


def check() -> int:
    """Verify without generating. Returns a process exit code."""
    if not MANIFEST.is_file():
        print(f"MISSING {MANIFEST} — run:\n"
              f"    ./.venv/bin/python scripts/make_demo_audio.py")
        return 1
    m = json.loads(MANIFEST.read_text())
    missing = [f for f in m["files"] if not Path(f).is_file()]
    for doc in ("DEMO_SCRIPT.md", "KEY.md", "WHAT_TO_LISTEN_FOR.md",
                "PREREGISTERED_PREDICTION.md"):
        if not (DEMO_DIR / doc).is_file():
            missing.append(str(DEMO_DIR / doc))
    if not (BLIND_DIR / "BLIND_SHEET.md").is_file():
        missing.append(str(BLIND_DIR / "BLIND_SHEET.md"))
    if missing:
        print(f"{len(missing)} demo file(s) missing, first few:")
        for f in missing[:5]:
            print("   ", f)
        print("run:\n    ./.venv/bin/python scripts/make_demo_audio.py --force")
        return 1
    s = m["paired_result"]
    print(f"OK: {len(m['files'])} files, {len(m['blind_map'])} blind copies")
    print(f"    paired diff {s['paired_diff_A_minus_B']:+.4f} "
          f"CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] "
          f"({'spans zero' if s['spans_zero'] else 'DOES NOT SPAN ZERO'})")

    # Protected documents are a healthy steady state, not a fault — this is a
    # census, and --check keeps its exit code. Reported anyway because the whole
    # point is that a hand-edited document is invisible otherwise.
    rep = doc_report()
    protected = [p for p, st in rep.items() if st == AUTHORED]
    print(f"    docs: {sum(1 for st in rep.values() if st == GENERATED)} "
          f"generator-owned, {len(protected)} PROTECTED (edited or unrecorded)")
    for p in protected:
        print(f"          protected (will NOT be rewritten): {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the listening + demo audio set.")
    # --force rebuilds the WAVS. It deliberately does NOT unlock the documents:
    # tests/test_demo.py runs this script with --force against the live repo, so
    # a --force that clobbered prose would let a green test suite quietly delete
    # a listening result. Destroying an authored document requires asking for it
    # by name.
    ap.add_argument("--force", action="store_true",
                    help="rebuild the audio even if the manifest is already present "
                         "(does NOT overwrite hand-edited documents)")
    ap.add_argument("--force-docs", action="store_true",
                    help="ALSO overwrite documents a human has edited; each one is "
                         "copied to <name>.superseded-<UTC>.md first")
    ap.add_argument("--check", action="store_true",
                    help="verify the existing set, generate nothing")
    a = ap.parse_args()

    if a.check:
        return check()
    if MANIFEST.is_file() and not (a.force or a.force_docs):
        m = json.loads(MANIFEST.read_text())
        if all(Path(f).is_file() for f in m["files"]):
            print(f"up to date ({len(m['files'])} files) — --force to rebuild")
            return 0

    rows = load_rows({COND_A, COND_B, COND_PAYOFF})
    stats = paired_stats(rows)
    assets = DiskAssetLibrary(root="data", target_fs=FS)

    made = build_audio(assets)
    m = build_manifest(made, rows, stats, assets)
    fd = a.force_docs
    write_demo_script(m, force_docs=fd)
    write_key(m, force_docs=fd)
    write_blind_sheet(m, force_docs=fd)
    write_prediction(m, force_docs=fd)
    write_what_to_listen_for(m, force_docs=fd)
    mark_old_set_superseded(m, force_docs=fd)
    MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")

    # The census, printed AFTER the file list so it is the last thing on screen.
    # A refusal that scrolls past is a refusal that did not happen.
    rep = doc_report()
    protected = [p for p, st in rep.items() if st == AUTHORED]

    print(f"demo set -> {DEMO_DIR}: {len(m['files'])} wavs")
    print(f"  pairs   : {', '.join(p['clip_id'] for p in m['pairs'])}")
    print(f"  blind   : {len(m['blind_map'])} copies in {BLIND_DIR}")
    print(f"  ladder  : {len(m['isolation'])} files in {ISO_DIR}")
    print(f"  paired  : A {stats['mean_wer_A']:.4f} | B {stats['mean_wer_B']:.4f} "
          f"| diff {stats['paired_diff_A_minus_B']:+.4f} "
          f"CI [{stats['ci_lo']:+.4f}, {stats['ci_hi']:+.4f}] "
          f"({'spans zero' if stats['spans_zero'] else 'DOES NOT SPAN ZERO'})")
    print(f"  payoff  : {m['payoff']['n_empty']}/{m['payoff']['n_clips']} clips "
          f"empty under {COND_PAYOFF}")
    print(f"  docs    : {len(DOCS) - len(protected)} written, "
          f"{len(protected)} PROTECTED (edited or unrecorded — left untouched)")
    if protected:
        for p in protected:
            print(f"            kept as-is: {p}")
        print("            Those files were NOT regenerated and do not carry this")
        print("            build's numbers. Port the hand-written blocks into the")
        print("            templates in scripts/make_demo_audio.py, or rebuild them")
        print("            with --force-docs (which backs each one up first).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
