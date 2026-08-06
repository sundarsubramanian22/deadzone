"""
R1.7 gate — verify the recorded corpus against `recording_manifest.csv`.

Run this before trusting a single downstream number:

    python3 check_recordings.py                # whole corpus
    python3 check_recordings.py --only u01,u02 # the R1.8 two-clip checkpoint

WHY THIS EXISTS. The recordings are the one asset in this repo that no unit test
can validate. `test_pipeline.py` proves `classify_errors` aligns two strings
correctly; nothing anywhere proves the wav on disk is mono, unclipped, padded,
or paired with the right manifest row. Each of those failures is SILENT and
each corrupts every layer downstream:

  * stereo / wrong rate  -> the composer averages channels or resamples in a way
    you never modelled, and `DiskAssetLibrary` rate assertions fire mid-grid.
  * clipping             -> a hard nonlinearity sprays broadband harmonics that
    get convolved and coded, and you read the result as codec sensitivity.
  * missing head/tail padding -> `active_speech_mask()` loses its non-speech
    reference, `mix_at_snr()` calibrates against the wrong energy, and
    `apply_rir()`'s onset trim can shave the first phoneme -- which becomes a
    deletion in EVERY reverb condition and reads as a beautiful, entirely fake
    "reverb causes deletions" fingerprint.
  * a manifest row whose ground truth is not normalization-stable -> a permanent
    WER offset in every grid cell, indistinguishable from an acoustic effect.

ERRORS fail the gate (exit 1). WARNINGS are reported and do not fail, but every
one of them is something you should be able to explain.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audio_pipeline import active_speech_mask, normalize_text, rms

# ============================================================================
# THRESHOLDS — every one of these traces to a SPEC appendix requirement
# ============================================================================

SAMPLE_RATES_OK = (16000, 48000)        # R1.2
PCM_SUBTYPES_OK = ("PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE")

CLIP_ABS_MAX = 0.99                     # R1.4 — hard clipping
CLIP_NEAR_LEVEL = 0.95
CLIP_NEAR_COUNT_MAX = 10

PEAK_MIN_DBFS = -14.0                   # R1.4 — target -12..-6, tolerance -14..-4
PEAK_MAX_DBFS = -4.0
PEAK_SPREAD_WARN_DB = 6.0               # drift from the corpus median

PAD_MIN_S = 0.25                        # R1.5 — head and tail room tone

DUR_MIN_S = 1.5                         # R1.7 — sane utterance length
DUR_MAX_S = 12.0
DUR_OUTLIER_FACTOR = 2.0                # vs corpus median

SNR_WARN_DB = 20.0                      # inherent (in-clip) speech-to-floor SNR

# Files that legitimately live in the recordings dir without a manifest row.
IGNORE_STEMS = {"_roomtone", "sample", "public_sample"}


# ============================================================================
# MEASUREMENT — pure, so the test suite can drive it on synthetic signals
# ============================================================================

@dataclass
class ClipStats:
    clip_id: str
    fs: int
    channels: int
    subtype: str
    n_samples: int
    duration_s: float
    peak: float
    peak_dbfs: float
    n_near_clip: int
    lead_s: float
    tail_s: float
    active_dbfs: float
    floor_dbfs: float
    inherent_snr_db: float


def measure_clip(clip_id: str, x: np.ndarray, fs: int,
                 channels: int, subtype: str) -> ClipStats:
    """Everything the gate needs from one clip's samples. No IO, no judgement."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:                       # measure what the composer would see
        x = x.mean(axis=1)

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    mask = active_speech_mask(x, fs)
    idx = np.flatnonzero(mask)

    if idx.size:
        lead = float(idx[0]) / fs
        tail = float(len(x) - 1 - idx[-1]) / fs
        active = rms(x[mask])
        floor = rms(x[~mask]) if (~mask).any() else 0.0
    else:                                # no active region at all
        lead = tail = 0.0
        active = floor = 0.0

    def _db(v: float) -> float:
        return 20.0 * np.log10(v + 1e-12)

    snr = _db(active) - _db(floor) if (active > 0 and floor > 0) else float("nan")

    return ClipStats(
        clip_id=clip_id, fs=fs, channels=channels, subtype=subtype,
        n_samples=len(x), duration_s=len(x) / fs,
        peak=peak, peak_dbfs=_db(peak),
        n_near_clip=int(np.sum(np.abs(x) > CLIP_NEAR_LEVEL)),
        lead_s=lead, tail_s=tail,
        active_dbfs=_db(active), floor_dbfs=_db(floor), inherent_snr_db=snr,
    )


# ============================================================================
# EVALUATION — per-clip rules, then corpus-wide consistency rules
# ============================================================================

@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: list[ClipStats] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def evaluate_clip(s: ClipStats) -> tuple[list[str], list[str]]:
    """Per-clip rules. Returns (errors, warnings)."""
    err: list[str] = []
    warn: list[str] = []
    cid = s.clip_id

    if s.channels != 1:
        err.append(f"{cid}: {s.channels} channels, expected mono (R1.2)")
    if s.fs not in SAMPLE_RATES_OK:
        err.append(f"{cid}: {s.fs} Hz, expected one of {SAMPLE_RATES_OK} (R1.2)")
    if s.subtype not in PCM_SUBTYPES_OK:
        err.append(f"{cid}: subtype {s.subtype} is not uncompressed PCM (R1.2) "
                   f"— a lossy source applies an uncontrolled codec to the "
                   f"`codec` independent variable")

    if s.peak >= CLIP_ABS_MAX:
        err.append(f"{cid}: CLIPPED (max|x|={s.peak:.4f} >= {CLIP_ABS_MAX}) (R1.4)")
    elif s.n_near_clip > CLIP_NEAR_COUNT_MAX:
        err.append(f"{cid}: {s.n_near_clip} samples above {CLIP_NEAR_LEVEL} "
                   f"— clipping risk (R1.4)")

    if s.lead_s < PAD_MIN_S:
        err.append(f"{cid}: lead-in silence {s.lead_s:.2f}s < {PAD_MIN_S}s — the "
                   f"VAD has no non-speech reference and apply_rir's onset trim "
                   f"can shave the first phoneme (R1.5)")
    if s.tail_s < PAD_MIN_S:
        err.append(f"{cid}: tail silence {s.tail_s:.2f}s < {PAD_MIN_S}s (R1.5)")

    if not (DUR_MIN_S <= s.duration_s <= DUR_MAX_S):
        err.append(f"{cid}: duration {s.duration_s:.2f}s outside "
                   f"[{DUR_MIN_S}, {DUR_MAX_S}]s (R1.7)")

    if not (PEAK_MIN_DBFS <= s.peak_dbfs <= PEAK_MAX_DBFS):
        warn.append(f"{cid}: peak {s.peak_dbfs:+.1f} dBFS outside "
                    f"[{PEAK_MIN_DBFS}, {PEAK_MAX_DBFS}] (R1.4)")

    if np.isfinite(s.inherent_snr_db) and s.inherent_snr_db < SNR_WARN_DB:
        warn.append(f"{cid}: inherent SNR {s.inherent_snr_db:+.1f} dB "
                    f"< {SNR_WARN_DB} dB — room noise counts as SIGNAL in "
                    f"mix_at_snr, so the benign end of the snr_db axis will "
                    f"under-deliver (R1.3)")

    return err, warn


def evaluate_corpus(stats: list[ClipStats]) -> tuple[list[str], list[str]]:
    """Consistency rules that only make sense across the whole corpus."""
    err: list[str] = []
    warn: list[str] = []
    if not stats:
        return err, warn

    rates = sorted({s.fs for s in stats})
    if len(rates) > 1:
        err.append(f"MIXED sample rates across the corpus: {rates}. Every clip "
                   f"must agree — apply_condition raises on a rate mismatch.")

    peaks = np.array([s.peak_dbfs for s in stats])
    med_peak = float(np.median(peaks))
    for s in stats:
        if abs(s.peak_dbfs - med_peak) > PEAK_SPREAD_WARN_DB:
            warn.append(f"{s.clip_id}: peak {s.peak_dbfs:+.1f} dBFS is "
                        f"{s.peak_dbfs - med_peak:+.1f} dB from the corpus median "
                        f"({med_peak:+.1f}) — level drift across the session (R1.4)")

    durs = np.array([s.duration_s for s in stats])
    med_dur = float(np.median(durs))
    for s in stats:
        if s.duration_s > DUR_OUTLIER_FACTOR * med_dur:
            warn.append(f"{s.clip_id}: duration {s.duration_s:.2f}s is more than "
                        f"{DUR_OUTLIER_FACTOR}x the median ({med_dur:.2f}s) — "
                        f"check for a doubled take (R1.7)")

    return err, warn


def check_manifest_rows(rows: list[dict]) -> list[str]:
    """
    The ground-truth column must already be normalization-stable, i.e. survive
    normalize_text() unchanged. If it doesn't, scoring silently compares against
    a different string than the one you proofread.
    """
    err: list[str] = []
    seen: set[str] = set()
    for r in rows:
        cid, gt = r["id"], r["ground_truth"]
        if cid in seen:
            err.append(f"manifest: duplicate id {cid!r}")
        seen.add(cid)
        canon = " ".join(normalize_text(gt))
        if canon != gt:
            err.append(f"manifest[{cid}]: ground_truth is not normalization-stable\n"
                       f"    stored : {gt!r}\n"
                       f"    becomes: {canon!r}")
    return err


# ============================================================================
# IO + DRIVER
# ============================================================================

def check_corpus(manifest_path: str = "recording_manifest.csv",
                 rec_dir: str = "data/recordings",
                 only: list[str] | None = None) -> Report:
    import soundfile as sf

    rep = Report()
    d = Path(rec_dir)
    if not d.is_dir():
        rep.errors.append(f"recordings directory {d} does not exist")
        return rep

    rows = list(csv.DictReader(open(manifest_path)))
    rep.errors.extend(check_manifest_rows(rows))
    if only:
        rows = [r for r in rows if r["id"] in set(only)]

    expected = {r["id"] for r in rows}

    for r in rows:
        cid = r["id"]
        p = d / f"{cid}.wav"
        if not p.is_file():
            rep.errors.append(f"{cid}: MISSING {p}")
            continue
        info = sf.info(str(p))
        x, fs = sf.read(str(p), dtype="float64", always_2d=False)
        s = measure_clip(cid, x, fs, info.channels, info.subtype)
        rep.stats.append(s)
        e, w = evaluate_clip(s)
        rep.errors.extend(e)
        rep.warnings.extend(w)

    # stray wavs — a file here with no manifest row usually means a mislabelled
    # export, and a mislabelled export is a clip paired to the WRONG transcript.
    if not only:
        for p in sorted(d.glob("*.wav")):
            if p.stem not in expected and p.stem not in IGNORE_STEMS:
                rep.errors.append(f"stray file {p.name} has no manifest row — "
                                  f"a mislabelled export pairs audio to the "
                                  f"wrong ground truth")

    e, w = evaluate_corpus(rep.stats)
    rep.errors.extend(e)
    rep.warnings.extend(w)
    return rep


def _print_table(stats: list[ClipStats]) -> None:
    hdr = (f"{'id':<6}{'dur s':>8}{'peak dB':>9}{'lead s':>8}{'tail s':>8}"
           f"{'active dB':>11}{'floor dB':>10}{'SNR dB':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in stats:
        print(f"{s.clip_id:<6}{s.duration_s:>8.2f}{s.peak_dbfs:>9.1f}"
              f"{s.lead_s:>8.2f}{s.tail_s:>8.2f}"
              f"{s.active_dbfs:>11.1f}{s.floor_dbfs:>10.1f}"
              f"{s.inherent_snr_db:>8.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--manifest", default="recording_manifest.csv")
    ap.add_argument("--dir", default="data/recordings")
    ap.add_argument("--only", default=None,
                    help="comma-separated ids, e.g. u01,u02 (R1.8 checkpoint)")
    ap.add_argument("--table", action="store_true", help="print per-clip measurements")
    a = ap.parse_args()

    only = [s.strip() for s in a.only.split(",")] if a.only else None
    rep = check_corpus(a.manifest, a.dir, only)

    if a.table and rep.stats:
        _print_table(rep.stats)
        print()

    for w in rep.warnings:
        print(f"WARN  {w}")
    for e in rep.errors:
        print(f"ERROR {e}")

    n_expected = len(only) if only else len(list(csv.DictReader(open(a.manifest))))
    n_ok = len(rep.stats)

    if rep.stats:
        snrs = [s.inherent_snr_db for s in rep.stats if np.isfinite(s.inherent_snr_db)]
        if snrs:
            worst = min(snrs)
            print(f"\ninherent SNR: min {worst:+.1f} dB, median "
                  f"{float(np.median(snrs)):+.1f} dB")
            print(f"  -> the snr_db axis saturates near {worst:.0f} dB; requesting "
                  f"more than that under-delivers (see report/measurements.md)")

    print()
    if rep.ok:
        print(f"{n_ok}/{n_expected} OK" +
              (f"  ({len(rep.warnings)} warning(s))" if rep.warnings else ""))
        return 0
    print(f"{n_ok}/{n_expected} checked — {len(rep.errors)} ERROR(S). Gate FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
