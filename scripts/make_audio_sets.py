"""
Generate the two audio sets that DO need to exist on disk.

    ./.venv/bin/python scripts/make_audio_sets.py              # both
    ./.venv/bin/python scripts/make_audio_sets.py --listen     # listening set only
    ./.venv/bin/python scripts/make_audio_sets.py --sweep      # L3 sweep only
    ./.venv/bin/python scripts/make_audio_sets.py --force-docs # AND clobber edited docs

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

REGENERABLE OUTPUT AND UNRECOVERABLE RECORD ARE NOT THE SAME ARTIFACT, and this
script used to hand both to the same writer. The wavs and `index.json` rebuild
from `data/` and the condition names in seconds — losing them costs nothing. The
one markdown file does not: `results/audio/listen/WHAT_TO_LISTEN_FOR.md` is the
instructions page sitting in the directory where the listening pass actually
happens, so it is the likeliest file in this repo for a human to annotate with
WHAT THEY HEARD. In this project that is not hypothetical — the listening pass is
what invalidated the published headline (SPEC Appendix G: the estimand mismatch,
dead zones 6 -> 2, rho restated), and nothing in the repo can regenerate a
listener's observation. It now goes through `write_doc`, which REFUSES to
overwrite a file this generator cannot prove it wrote. See THE AUTHORED-DOCUMENT
GUARD below. The audio is untouched by that and keeps regenerating freely.
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
import shutil
from datetime import datetime, timezone
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


# --------------------------------------------------------------------------
# THE AUTHORED-DOCUMENT GUARD
#
# `make_listening_set` used to `.write_text()` WHAT_TO_LISTEN_FOR.md on every
# run — no skip-if-modified, no warning, no backup. That is correct for the wavs,
# which are a pure function of `data/` and the condition names, and it is
# catastrophic for the prose, which is where a human writes down the one thing
# the pipeline cannot produce: what a listener heard. The defect is NOT the
# overwrite; it is that REGENERABLE OUTPUT and UNRECOVERABLE RECORD were handed
# to the same writer.
#
# This is not a new problem and this is deliberately not a new solution. The
# identical hazard in `scripts/make_demo_audio.py` cost a real edit — a
# pre-registered prediction's verbatim listener response, its scoring and its
# verdict, erased minutes after being written — and was closed in `aea58e2`. The
# mechanism below is that one: same four states, same fail-closed rule, same
# no-seeding decision, same hash-not-mtime detection, same backup-before-force.
# `tests/test_make_audio_sets.py::BothGuardsAgree` runs the two implementations
# through the same scenario matrix and asserts they answer identically, so they
# cannot drift into being two different guards for one hazard.
#
# It is a sibling rather than an import for two reasons, both mechanical:
# `make_demo_audio` imports LADDER and LISTEN_CLIP from THIS module, so importing
# back inverts the dependency and cycles; and a refusal has to name the override
# flag of the script the user actually ran, or it is a warning pointing at the
# wrong door.
#
# The rule, following `write_master()` in scripts/run_experiment.py — refuse to
# produce the misleading artifact rather than warn about it afterwards:
#
#   file absent          -> write.
#   hash matches record  -> this generator wrote it and nobody has touched it
#                           since; write.
#   hash differs         -> a human edited it; REFUSE, and name the flag.
#   NO recorded hash     -> provenance unknown; REFUSE.
#
# That last line is the load-bearing one and it is deliberately the paranoid
# reading. A missing record is the DEGENERATE input, and SPEC Appendix E.5's rule
# is to ask what a guard returns for the degenerate input rather than for the
# good one. If "no record" meant "safe to overwrite", this guard would be wide
# open on exactly the state the repo is in the first time it runs — the document
# already on disk, hand-edited, and no sidecar yet.
#
# It also settles how the sidecar is seeded: it ISN'T. Recording today's bytes as
# "what the generator last wrote" would certify hand-written text as
# generator-owned, and the very next default run would erase it — the guard would
# have been the delivery mechanism for the bug it exists to stop. Not seeding is
# what protects the file that is on disk right now, and any file restored from a
# backup or predating the guard, for the same reason.
#
# Detection is by CONTENT HASH, never mtime. mtime does not survive a checkout, a
# `cp` without `-p`, a zip round-trip or a restore — all four of which happen to
# a listening kit that gets handed around. A `touch` is not an edit and a
# byte-for-byte restore is not an edit; only the bytes decide.
# --------------------------------------------------------------------------

# What the generator last wrote, keyed by repo-relative path (every path in this
# project resolves against the repo root as CWD, SPEC §13). Lives beside the
# document it guards so it shares its lifecycle: delete `results/audio/listen/`
# and the whole set rebuilds from scratch, guard included.
DOC_HASHES = LISTEN_DIR / "generated_docs.json"

ABSENT, GENERATED, AUTHORED = "absent", "generated", "authored"

# Every document this script owns. `index.json` and the wavs are deliberately
# NOT here: they are machine data and regenerable audio, which is the whole
# distinction this guard encodes.
DOCS = (LISTEN_DIR / "WHAT_TO_LISTEN_FOR.md",)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_key(path: Path) -> str:
    return Path(path).as_posix()


def load_doc_hashes() -> dict[str, str]:
    """Path -> sha256 of the text this generator last wrote there.

    An unreadable or malformed sidecar returns `{}`, which is the same as an
    absent one, and both then read as unknown provenance — i.e. the failure mode
    of this function is to protect MORE, never less.
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
        "written_by": "scripts/make_audio_sets.py",
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
        f"[listening docs] REFUSING to overwrite {p}",
        "    why   : its contents do not match what this generator last wrote, so it",
        "            holds edits made by a human — most likely notes from an actual",
        "            listening pass, which nothing in this repo can reproduce.",
        "            Rewriting it would replace them with template output and say",
        "            nothing: a regenerated file looks exactly as correct as the one",
        "            it replaced. That is how the sibling kit lost an edit already",
        "            (scripts/make_demo_audio.py, fixed in aea58e2).",
        "    state : the file on disk is UNCHANGED and the rest of the build continued.",
        "    cost  : it therefore does NOT carry this build's content — the ladder and",
        "            dead-zone filenames it describes may have moved on. The wavs in",
        "            this directory are the generated source of truth for those.",
        "    fix   : port the hand-written block into this script's template, so a",
        "            rebuild reproduces it — or, to overwrite it anyway:",
        "                ./.venv/bin/python scripts/make_audio_sets.py --force-docs",
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
        print(f"[listening docs] --force-docs: OVERWRITING hand-edited {p}\n"
              f"                 previous contents preserved at {backup}\n"
              f"                 NOTHING ELSE IN THIS REPO HOLDS THEM — read that "
              f"file before deleting it.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    _save_doc_hash(_doc_key(p), _sha256(text))
    return "written"


def doc_report() -> dict[str, str]:
    """path -> status, over every document the build would write."""
    h = load_doc_hashes()
    return {str(p): doc_status(p, h) for p in DOCS}


def make_listening_set(assets, manifest, *, force_docs: bool = False) -> int:
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

    # The audio above regenerates unconditionally — it is a pure function of the
    # clip and the condition names, so a rebuild costs seconds and loses nothing.
    # The document does not: see THE AUTHORED-DOCUMENT GUARD.
    write_doc(
        LISTEN_DIR / "WHAT_TO_LISTEN_FOR.md",
        "# Listening set (SPEC A.R3.5)\n\n"
        "> **The `DEADZONE_*` files here are SUPERSEDED** by\n"
        "> `results/audio/demo/` (`scripts/make_demo_audio.py`): all of them are\n"
        f"> clip `{LISTEN_CLIP}`, which nova-3 transcribes at WER 0.000 in four of\n"
        "> the six flagged conditions, so they demonstrated nothing. The ladder\n"
        "> below is sound and is what this set is for.\n\n"
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
        force_docs=force_docs)
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
    # --listen and --sweep SELECT which set to build. Neither unlocks a document,
    # and that separation is the whole lesson from the sibling script: there,
    # --force rebuilt the wavs and tests/test_demo.py passed it against the live
    # tree on every `make test`, so a GREEN TEST SUITE was the delivery mechanism
    # that destroyed a hand-written record. Nothing in this repo currently runs
    # make_audio_sets.py automatically — but --listen is exactly the flag a future
    # `make listen-prep` would use, and if it doubled as the override the same
    # trap would be rebuilt through a different door. Destroying an authored
    # document has to be asked for by name.
    ap.add_argument("--listen", action="store_true",
                    help="listening set only (does NOT overwrite hand-edited documents)")
    ap.add_argument("--sweep", action="store_true",
                    help="L3 sweep only")
    ap.add_argument("--force-docs", action="store_true",
                    help="ALSO overwrite documents a human has edited; each one is "
                         "copied to <name>.superseded-<UTC>.md first")
    a = ap.parse_args()
    do_listen = a.listen or not (a.listen or a.sweep)
    do_sweep = a.sweep or not (a.listen or a.sweep)

    assets = DiskAssetLibrary(root="data", target_fs=FS)
    manifest = load_manifest()

    if do_listen:
        n = make_listening_set(assets, manifest, force_docs=a.force_docs)
        print(f"listening set -> {LISTEN_DIR}: {n} files")
        # The census, printed AFTER the file count so it is the last thing on
        # screen. A refusal that scrolls past is a refusal that did not happen.
        rep = doc_report()
        protected = [p for p, st in rep.items() if st == AUTHORED]
        print(f"  docs        : {len(DOCS) - len(protected)} written, "
              f"{len(protected)} PROTECTED (edited or unrecorded — left untouched)")
        for p in protected:
            print(f"                kept as-is: {p}")
        if protected:
            print("                Those files were NOT regenerated and do not carry")
            print("                this build's content. Port the hand-written blocks")
            print("                into the templates in scripts/make_audio_sets.py,")
            print("                or rebuild with --force-docs (backs each one up).")
    if do_sweep:
        print(f"L3 sweeps     -> {SWEEP_ROOT}: {make_sweeps(assets, manifest)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
