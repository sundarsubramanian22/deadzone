"""
R2 — fetch and CURATE the real acoustic ingredients (SPEC §8).

    python3 fetch_assets.py                 # everything
    python3 fetch_assets.py --rirs-only
    python3 fetch_assets.py --noise-only --minimal
    python3 fetch_assets.py --verify        # re-check what's on disk, download nothing

Idempotent: anything already present with a matching SHA-256 is skipped.

WHY THIS CURATES INSTEAD OF DUMPING. Two traps make "download and extract"
the wrong move:

 1. RT60 COVERAGE. `AssetLibrary.resolve()` picks the RIR whose *measured* RT60
    is CLOSEST to the requested value. If the library clusters (the MIT survey's
    mode sits near 0.36 s), then requests at 0.2 and 1.0 both snap to whatever is
    nearest and the rt60 axis silently degrades into a step function with dead
    ranges -- after which Sobol reports that reverb barely matters. That is a
    wrong result with no error message. So we measure RT60 for all 270 IRs and
    SELECT a subset that evenly tiles [0.2, 1.0], then assert the largest gap.

 2. MEMORY. `DiskAssetLibrary.__init__` reads, resamples and CACHES every file
    it finds in RAM. Pointing it at an unzipped MUSAN or SLR28 tree exhausts
    memory and takes minutes to construct. We copy in a curated handful only.

Sources
  RIRs  : MIT Acoustical Reverberation Scene Statistics Survey (Traer &
          McDermott) -- 270 real measured IRs @ 32 kHz, 11.7 MB, no registration.
  Noise : DEMAND (Zenodo 1227121) -- multi-channel real environmental recordings,
          5 min per channel. Two environments per noise_type so the categorical
          level means "this kind of place", not "this one recording".
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# ============================================================================
# SOURCES
# ============================================================================

MIT_URL = "http://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"

DEMAND_BASE = "https://zenodo.org/record/1227121/files"

# noise_type -> DEMAND environments. Directory names MUST match the categorical
# levels in DEFAULT_FACTOR_SPACE exactly; a typo yields a library holding a type
# nothing ever requests, and resolve() raises mid-grid.
DEMAND_ENVS: dict[str, list[str]] = {
    "babble": ["PCAFETER", "PSTATION"],   # cafeteria, train station
    "engine": ["TCAR", "TBUS"],           # car interior, bus interior
    "road":   ["STRAFFIC", "SPSQUARE"],   # street traffic, public square
}

CHANNELS_PER_ENV = 2        # distinct mic channels kept from each environment
TARGET_FS = 16000           # what DiskAssetLibrary resamples to anyway

# RIR curation
N_RIRS = 16
RT60_LO, RT60_HI = 0.2, 1.0          # must match DEFAULT_FACTOR_SPACE's rt60 bounds
MAX_RT60_GAP = 0.15                  # largest acceptable hole in the tiling

ROOT = Path("data")
RIR_DIR = ROOT / "rirs"
NOISE_DIR = ROOT / "noise"
MANIFEST = Path("results") / "asset_manifest.json"


# ============================================================================
# HELPERS
# ============================================================================

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ssl_context():
    """
    macOS Python.framework builds ship WITHOUT a usable CA bundle, so HTTPS via
    urllib dies with CERTIFICATE_VERIFY_FAILED even though `curl` to the same URL
    works (curl uses the system keychain). Point OpenSSL at certifi's bundle.
    Returns None if certifi is unavailable, in which case download() falls back
    to curl. We never disable verification -- an unverified download of a file we
    then checksum and feed into the experiment is not a tradeoff worth making.
    """
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _download_curl(url: str, tmp: Path, label: str) -> bool:
    if shutil.which("curl") is None:
        return False
    print(f"\r  downloading {label} ... (curl)", end="", flush=True)
    r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(tmp), url])
    return r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0


def download(url: str, dest: Path, label: str) -> Path:
    """Stream to disk with a progress line. Skips if already present."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {label} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {label} ...", end="", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        ctx = _ssl_context()
        with urllib.request.urlopen(url, timeout=180, context=ctx) as r, \
                open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  downloading {label} ... {100 * got / total:5.1f}%",
                          end="", flush=True)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"\r  urllib failed for {label} ({type(e).__name__}); trying curl")
        if not _download_curl(url, tmp, label):
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"could not download {label} from {url}: {e}")

    tmp.rename(dest)
    print(f"\r  downloaded {label}      ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _to_mono_16k(x: np.ndarray, fs: int) -> tuple[np.ndarray, int]:
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != TARGET_FS:
        import librosa
        x = librosa.resample(np.asarray(x, dtype=np.float64),
                             orig_sr=fs, target_sr=TARGET_FS)
    return np.asarray(x, dtype=np.float64), TARGET_FS


def _safe_stem(s: str, maxlen: int = 48) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_") else "-" for c in s)
    return keep[:maxlen].strip("-")


# ============================================================================
# RIRs — download, measure, SELECT for coverage
# ============================================================================

def fetch_rirs(stage: Path, n_rirs: int = N_RIRS) -> list[dict]:
    import soundfile as sf
    from conditions import _rt60_schroeder

    print("\nRIRs — MIT Acoustical Reverberation Survey")
    zp = download(MIT_URL, stage / "mit_ir_survey.zip", "MIT IR survey")

    print("  measuring RT60 for every IR (this is what makes selection possible)...")
    z = zipfile.ZipFile(zp)
    names = [n for n in z.namelist()
             if n.startswith("Audio/") and n.lower().endswith(".wav")
             and "__MACOSX" not in n]

    cands: list[dict] = []
    for n in names:
        try:
            x, fs = sf.read(io.BytesIO(z.read(n)))
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        rt = _rt60_schroeder(x, fs)
        if not np.isfinite(rt) or not (0.0 < rt < 5.0):
            continue
        cands.append({"name": n, "rt60": float(rt), "fs": int(fs)})

    in_band = [c for c in cands if RT60_LO <= c["rt60"] <= RT60_HI]
    print(f"  {len(cands)} IRs with an estimable RT60; "
          f"{len(in_band)} inside [{RT60_LO}, {RT60_HI}] s")
    if len(in_band) < n_rirs:
        raise SystemExit(f"only {len(in_band)} in-band IRs, need {n_rirs}")

    # Greedy tiling: for each evenly-spaced target, take the closest unused IR.
    targets = np.linspace(RT60_LO, RT60_HI, n_rirs)
    used: set[int] = set()
    chosen: list[dict] = []
    for t in targets:
        i = min((i for i in range(len(in_band)) if i not in used),
                key=lambda i: abs(in_band[i]["rt60"] - t))
        used.add(i)
        chosen.append(in_band[i])
    chosen.sort(key=lambda c: c["rt60"])

    RIR_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for c in chosen:
        orig = Path(c["name"]).stem
        x, fs = sf.read(io.BytesIO(z.read(c["name"])))
        x, fs = _to_mono_16k(x, fs)
        dest = RIR_DIR / f"mit_rt60-{c['rt60']:.2f}_{_safe_stem(orig)}.wav"
        sf.write(dest, x, fs, subtype="PCM_24")
        out.append({"path": str(dest), "source": "MIT IR Survey",
                    "original": orig, "rt60_measured": round(c["rt60"], 4),
                    "fs": fs, "sha256": sha256(dest)})

    # The coverage assertion this whole function exists for.
    got = np.array([o["rt60_measured"] for o in out])
    gaps = np.diff(got)
    worst = float(gaps.max()) if len(gaps) else 0.0
    print(f"  selected {len(out)} IRs spanning {got.min():.2f}-{got.max():.2f} s")
    print(f"  largest RT60 gap: {worst:.3f} s "
          f"({'OK' if worst <= MAX_RT60_GAP else 'TOO WIDE'}, limit {MAX_RT60_GAP})")
    if worst > MAX_RT60_GAP:
        print("  WARNING: a gap this wide means requests inside it snap to a "
              "distant RIR — the rt60 axis will under-resolve there. Raise "
              "--n-rirs or accept and document it.")
    return out


# ============================================================================
# Noise — download per environment, keep a couple of channels each
# ============================================================================

def fetch_noise(stage: Path, minimal: bool = False) -> list[dict]:
    import soundfile as sf

    print("\nNoise — DEMAND (Zenodo 1227121)")
    out: list[dict] = []
    for ntype, envs in DEMAND_ENVS.items():
        use = envs[:1] if minimal else envs
        dest_dir = NOISE_DIR / ntype
        dest_dir.mkdir(parents=True, exist_ok=True)
        for env in use:
            zp = download(f"{DEMAND_BASE}/{env}_16k.zip",
                          stage / f"{env}_16k.zip", f"{env} ({ntype})")
            z = zipfile.ZipFile(zp)
            wavs = sorted(n for n in z.namelist()
                          if n.lower().endswith(".wav") and "__MACOSX" not in n)
            if not wavs:
                print(f"    WARN: no wavs inside {env}_16k.zip — skipped")
                continue
            for ch in wavs[:CHANNELS_PER_ENV]:
                x, fs = sf.read(io.BytesIO(z.read(ch)))
                x, fs = _to_mono_16k(x, fs)
                dest = dest_dir / f"demand_{env}_{_safe_stem(Path(ch).stem)}.wav"
                sf.write(dest, x, fs, subtype="PCM_16")
                out.append({"path": str(dest), "source": f"DEMAND/{env}",
                            "noise_type": ntype, "duration_s": round(len(x) / fs, 1),
                            "fs": fs, "sha256": sha256(dest)})
                print(f"    {dest.name}  ({len(x) / fs:.0f}s)")
    return out


# ============================================================================
# VERIFY — construct the real library and check it against the factor space
# ============================================================================

def verify() -> bool:
    from collections import Counter

    from conditions import DiskAssetLibrary
    from design import DEFAULT_FACTOR_SPACE

    print("\nVerifying the library as the composer will see it")
    try:
        lib = DiskAssetLibrary(root=str(ROOT), target_fs=TARGET_FS)
    except Exception as e:
        print(f"  FAILED to construct DiskAssetLibrary: {e}")
        return False

    rts = np.array(sorted(a.rt60 for a in lib.rirs))
    counts = Counter(a.noise_type for a in lib.noise)
    want = {f.name: f.levels for f in DEFAULT_FACTOR_SPACE.factors
            if f.name == "noise_type"}["noise_type"]

    ok = True
    print(f"  RIRs : {len(rts)}  RT60 {rts.min():.2f}-{rts.max():.2f} s")
    in_band = rts[(rts >= RT60_LO) & (rts <= RT60_HI)]
    gap = float(np.diff(in_band).max()) if len(in_band) > 1 else float("inf")
    print(f"         in-band {len(in_band)}, largest gap {gap:.3f} s "
          f"(limit {MAX_RT60_GAP})")
    if gap > MAX_RT60_GAP:
        print("         ^ rt60 axis will under-resolve inside that gap")
        ok = False

    print(f"  Noise: {dict(counts)}")
    for lvl in want:
        n = counts.get(lvl, 0)
        if n < 3:
            print(f"         {lvl}: only {n} clip(s) — need >=3 so the seeded "
                  f"per-condition pick actually varies")
            ok = False
    extra = set(counts) - set(want)
    if extra:
        print(f"         stray noise types not in the factor space: {sorted(extra)}")
        ok = False

    print(f"\n  {'LIBRARY OK' if ok else 'LIBRARY INCOMPLETE'}")
    return ok


# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rirs-only", action="store_true")
    ap.add_argument("--noise-only", action="store_true")
    ap.add_argument("--minimal", action="store_true",
                    help="one DEMAND environment per noise type (~360 MB instead of ~715)")
    ap.add_argument("--n-rirs", type=int, default=N_RIRS)
    ap.add_argument("--stage", default=None,
                    help="where to keep downloaded archives (default: a temp dir)")
    ap.add_argument("--verify", action="store_true",
                    help="only check what is already on disk")
    a = ap.parse_args()

    if a.verify:
        return 0 if verify() else 1

    tmp = None
    if a.stage:
        stage = Path(a.stage)
        stage.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory()
        stage = Path(tmp.name)

    entries: list[dict] = []
    try:
        if not a.noise_only:
            entries += fetch_rirs(stage, a.n_rirs)
        if not a.rirs_only:
            entries += fetch_noise(stage, a.minimal)
    finally:
        if tmp is not None:
            tmp.cleanup()

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(
        {"target_fs": TARGET_FS, "rt60_band": [RT60_LO, RT60_HI],
         "n_entries": len(entries), "entries": entries}, indent=2))
    print(f"\nwrote {MANIFEST} ({len(entries)} assets)")

    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
