"""
make_sim_rirs.py — R2.5: the SYNTHETIC half of the sim-vs-real leg (SPEC §4, §5.4).

`data/rirs/` holds 16 REAL measured RIRs (MIT Acoustical Reverberation Survey)
tiling measured RT60 across [0.2, 1.0] s. This script builds a *paired* synthetic
set in `data/rirs_sim/` with pyroomacoustics, so the grid can be re-run with only
the reverb ingredient swapped from measured to simulated. The difference between
the two runs IS the sim2real finding (`analysis/sim2real.py`), which is only
interpretable if the two libraries are matched on the thing that matters: how much
reverb actually reaches the microphone.

  python3 scripts/make_sim_rirs.py                 # generate + pair + write rir_pairs.json

THE TRAP THIS FILE EXISTS TO AVOID
----------------------------------
pyroomacoustics' *realized* RT60 is NOT the RT60 you asked for. `inverse_sabine`
inverts Sabine's formula, which is a diffuse-field approximation; the image-source
model then truncates at `max_order`, and both effects bias the decay. Measured on
the 13x9x4.5 m `hall` geometry in this file: asking for 1.00 s yields a Schroeder
T20 of **1.241 s** (+24%); asking the same room for 0.25 s yields **0.168 s**
(-33%). The bias is not even one-signed, so it cannot be corrected with a constant.

So pairing a real RIR to "the synthetic one I asked for the same RT60 for" would
silently compare a 1.01 s real room against a 1.24 s simulated one, and the
sim2real gap would be reporting a reverb mismatch we introduced ourselves. On the
16-pair set this script actually produces, **8 of 16 pairs would breach the
+/-0.05 s tolerance** if they were matched on the Sabine target — the trap is the
common case, not a corner case. Every pairing decision here is therefore made on
the MEASURED Schroeder T20 of both files (`conditions._rt60_schroeder`, the same
estimator the asset library uses), never on the Sabine target, which is recorded
in the manifest purely as provenance.

EVERYTHING HAPPENS AT 16 kHz — ON PURPOSE
-----------------------------------------
`DiskAssetLibrary` resamples every asset to `target_fs` (16 kHz) ON LOAD and then
re-measures RT60 from the RESAMPLED signal, which shifts the estimate slightly
versus measuring at the file's native rate. The pairing must hold for the library
the grid actually uses, so the synthetic RIRs are GENERATED at 16 kHz
(`pra.ShoeBox(fs=16000)`), WRITTEN at 16 kHz, and re-measured by reading them back
through `DiskAssetLibrary(target_fs=16000)` — no resampling anywhere, and the
real set is measured through that same path rather than trusting the RT60 in the
filename. The numbers in `rir_pairs.json` are therefore the numbers the experiment
will deliver, not numbers from a different rate.

One further deliberate choice: 4 room geometries x 2 source-mic distances,
assigned by RT60 band, so the synthetic set mirrors the real set's SPREAD of rooms
and placements rather than being one room with the absorption dialled up and down
(SPEC §8: each RIR proxies a room + a distance).

Deps: numpy, scipy, soundfile, pyroomacoustics (lazily imported — this module
stays importable, and its pairing logic testable, without pra installed).
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
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from deadzone.conditions import DiskAssetLibrary, _rt60_schroeder


# ============================================================================
# CONSTANTS
# ============================================================================

TARGET_FS = 16000            # the rate the real library is resampled to
PAIR_TOL_S = 0.05            # DoD: every pair within +/-0.05 s MEASURED RT60
ACCEPT_TOL_S = 0.02          # in-loop acceptance; margin for the on-disk re-measure
MAX_CALIBRATION_ITERS = 6
MAX_IMAGE_ORDER = 400        # refuse to generate an ISM that would take minutes

ROOT = Path("data")
REAL_SUBDIR = "rirs"
SIM_SUBDIR = "rirs_sim"
PAIRS_JSON = Path("results") / "rir_pairs.json"

# Placement constants — a source and a mic in a room, not two points in a box.
_SRC_HEIGHT = 1.55           # standing mouth height (m)
_MIC_HEIGHT = 1.15           # table / device height (m)
_WALL_MARGIN = 0.40          # keep both endpoints this far off every surface (m)
_AXIS_ANGLE = 0.6            # rad; the src->mic axis is deliberately off-grid


# ============================================================================
# ROOM GEOMETRIES — 4 rooms x 2 distances, mirroring the real set's spread
# ============================================================================

@dataclass(frozen=True)
class Geometry:
    """One shoebox + the source-mic distances we place inside it."""
    name: str
    dims: tuple[float, float, float]      # (Lx, Ly, Lz) metres
    distances: tuple[float, ...]          # horizontal src->mic separations (m)


# Ordered small -> large. Small rooms carry the short-RT60 end of the real set,
# large rooms the long end: that is both physically sensible AND the cheap corner
# of the image-source model (a 1 s RT60 in a 2.6 m booth needs max_order ~210).
GEOMETRIES: tuple[Geometry, ...] = (
    Geometry("booth",   (2.6, 2.2, 2.4), (0.5, 1.0)),
    Geometry("office",  (4.5, 3.6, 2.8), (1.0, 2.0)),
    Geometry("meeting", (7.5, 5.5, 3.0), (1.5, 3.0)),
    Geometry("hall",   (13.0, 9.0, 4.5), (2.5, 5.0)),
)


def config_for(rank: int, n_targets: int) -> tuple[Geometry, float]:
    """
    The (geometry, distance) to use for the `rank`-th shortest real RT60.

    The room comes from the RT60 band (short -> booth, long -> hall) and the
    distance alternates within the band; together that spreads geometry and
    placement across the synthetic set. There is deliberately NO fallback to
    another room if this one cannot realize the target: silently substituting a
    different geometry would change what the synthetic set represents without
    changing anything you can see. Failure raises and tells you to fix GEOMETRIES.
    """
    n_geo = len(GEOMETRIES)
    band = min(rank * n_geo // max(n_targets, 1), n_geo - 1)
    geo = GEOMETRIES[band]
    return geo, geo.distances[rank % len(geo.distances)]


# ============================================================================
# GENERATION — one pyroomacoustics RIR
# ============================================================================

def _placement(dims: tuple[float, float, float],
               distance: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Source and mic positions for a given horizontal separation.

    Deliberately OFF-CENTRE and off-axis: a source and mic on the room's centre
    line make whole families of image sources coincide, which produces a
    comb-filtered, unphysically regular RIR. Returns (src, mic, true_3d_distance).
    Raises if the requested separation does not fit with wall margin — a silent
    clamp would change the geometry without changing the label.
    """
    Lx, Ly, Lz = dims
    u = np.array([math.cos(_AXIS_ANGLE), math.sin(_AXIS_ANGLE)])
    mid = np.array([Lx * 0.45, Ly * 0.55])
    half = 0.5 * float(distance) * u
    src = np.array([*(mid - half), _SRC_HEIGHT], dtype=float)
    mic = np.array([*(mid + half), _MIC_HEIGHT], dtype=float)

    lo = np.array([_WALL_MARGIN] * 3)
    hi = np.array([Lx, Ly, Lz]) - _WALL_MARGIN
    for label, p in (("source", src), ("mic", mic)):
        if np.any(p < lo) or np.any(p > hi):
            raise ValueError(
                f"{label} at {np.round(p, 2).tolist()} falls outside "
                f"{dims} with a {_WALL_MARGIN} m margin at distance {distance} m — "
                f"fix GEOMETRIES rather than clamping the position"
            )
    return src, mic, float(np.linalg.norm(src - mic))


def synth_rir(geo: Geometry, distance: float, rt60_target: float,
              fs: int = TARGET_FS) -> dict:
    """
    Generate ONE synthetic RIR: inverse_sabine -> ShoeBox -> compute_rir.

    `rt60_target` is the Sabine TARGET, i.e. what we ask for — not what we get.
    The returned dict carries both it and the realized Schroeder T20 so callers
    can never confuse them.
    """
    import pyroomacoustics as pra          # lazy: keeps this module import-safe

    e_absorption, max_order = pra.inverse_sabine(float(rt60_target), list(geo.dims))
    max_order = int(max_order)
    if max_order > MAX_IMAGE_ORDER:
        raise ValueError(
            f"max_order {max_order} > {MAX_IMAGE_ORDER} for {geo.name} at "
            f"target {rt60_target:.3f}s — image-source count would explode"
        )
    src, mic, d3 = _placement(geo.dims, distance)

    room = pra.ShoeBox(list(geo.dims), fs=int(fs),
                       materials=pra.Material(float(e_absorption)),
                       max_order=max_order)
    room.add_source(src.tolist())
    room.add_microphone(mic.reshape(3, 1))
    room.compute_rir()
    h = np.asarray(room.rir[0][0], dtype=np.float64)

    return {
        "h": h,
        "geometry": geo.name,
        "room_dims": list(geo.dims),
        "distance_m": float(distance),
        "distance_3d_m": round(d3, 3),
        "rt60_sabine_target": float(rt60_target),
        "rt60_realized": float(_rt60_schroeder(h, fs)),
        "e_absorption": float(e_absorption),
        "max_order": max_order,
        "n_samples": int(len(h)),
    }


def calibrate_rir(want: float, geo: Geometry, dist: float,
                  fs: int = TARGET_FS, accept_tol: float = ACCEPT_TOL_S,
                  max_iter: int = MAX_CALIBRATION_ITERS,
                  verbose: bool = True) -> dict:
    """
    Find a synthetic RIR whose MEASURED RT60 lands within `accept_tol` of `want`.

    The Sabine target is a lever, not the answer: we generate, MEASURE, and then
    correct the target proportionally (target *= want / realized) until the
    measured value converges. Realized RT60 is close to monotone in the target,
    so this converges in 1-3 iterations. Returns the closest candidate found;
    whether it is close ENOUGH is the caller's call (see PAIR_TOL_S).
    """
    best: dict | None = None
    target = float(want)
    for it in range(1, max_iter + 1):
        cand = synth_rir(geo, dist, target, fs)     # ValueError => fix GEOMETRIES
        realized = cand["rt60_realized"]
        if not np.isfinite(realized) or realized <= 0:
            raise ValueError(
                f"{geo.name} d={dist}m target={target:.3f}s produced a RIR with no "
                f"estimable RT60 — the decay is degenerate, fix the geometry")
        cand["calibration_iters"] = it
        if best is None or abs(realized - want) < abs(best["rt60_realized"] - want):
            best = cand
        if verbose:
            print(f"      {geo.name:<8s} d={dist:>3.1f}m  it{it}  "
                  f"target={target:.3f}  realized={realized:.3f}  "
                  f"err={realized - want:+.3f}")
        if abs(realized - want) <= accept_tol:
            break
        # proportional correction on the TARGET, judged by the MEASUREMENT
        target = float(np.clip(target * want / realized, 0.03, 4.0))
    return best


# ============================================================================
# PAIRING — on measured RT60, one-to-one, never on the Sabine target
# ============================================================================

def pair_by_measured(real: Sequence[dict], sim: Sequence[dict],
                     tol: float = PAIR_TOL_S) -> dict:
    """
    Pair each real RIR with exactly one synthetic RIR by MEASURED RT60.

    `real` / `sim` entries need at least {'key': str, 'rt60': float}, where
    `rt60` is the Schroeder T20 measured off that file. Any other keys are
    carried through into the pair record.

    One-to-one by construction (Hungarian assignment minimizing total |delta|):
    a greedy nearest-match could hand the same synthetic RIR to two real ones and
    leave a third unpaired, which would double-count one simulated room in the
    sim2real average. Pairs whose |delta| exceeds `tol` are reported UNMATCHED
    rather than quietly accepted — an out-of-tolerance pair is a reverb mismatch
    masquerading as a sim2real gap.
    """
    from scipy.optimize import linear_sum_assignment

    real = list(real)
    sim = list(sim)
    if not real or not sim:
        return {"pairs": [], "unmatched_real": [r["key"] for r in real],
                "unmatched_sim": [s["key"] for s in sim],
                "max_abs_delta": float("nan"), "tol": float(tol), "ok": False}

    rt_r = np.array([float(r["rt60"]) for r in real])
    rt_s = np.array([float(s["rt60"]) for s in sim])
    cost = np.abs(rt_r[:, None] - rt_s[None, :])
    ri, si = linear_sum_assignment(cost)

    pairs, unmatched_real, unmatched_sim = [], [], []
    used_r, used_s = set(), set()
    for i, j in zip(ri, si):
        delta = float(rt_s[j] - rt_r[i])
        used_r.add(int(i))
        used_s.add(int(j))
        if abs(delta) > tol:
            unmatched_real.append(real[i]["key"])
            unmatched_sim.append(sim[j]["key"])
            continue
        rec = {
            "real_key": real[i]["key"],
            "real_rt60_measured": round(float(rt_r[i]), 4),
            "sim_key": sim[j]["key"],
            "sim_rt60_measured": round(float(rt_s[j]), 4),
            "delta_s": round(delta, 4),
        }
        for k, v in sim[j].items():
            if k not in ("key", "rt60", "h"):
                rec[k] = v
        pairs.append(rec)

    unmatched_real += [r["key"] for i, r in enumerate(real) if i not in used_r]
    unmatched_sim += [s["key"] for j, s in enumerate(sim) if j not in used_s]
    pairs.sort(key=lambda p: p["real_rt60_measured"])
    deltas = [abs(p["delta_s"]) for p in pairs]
    return {
        "pairs": pairs,
        "unmatched_real": unmatched_real,
        "unmatched_sim": unmatched_sim,
        "max_abs_delta": float(max(deltas)) if deltas else float("nan"),
        "tol": float(tol),
        "ok": (not unmatched_real) and (not unmatched_sim) and len(pairs) == len(real),
    }


# ============================================================================
# DISK IO
# ============================================================================

def read_real_rirs(root: Path = ROOT, subdir: str = REAL_SUBDIR,
                   fs: int = TARGET_FS) -> list[dict]:
    """
    The real set, measured through exactly the path the grid will use
    (`DiskAssetLibrary` -> resample -> `_rt60_schroeder`). Reading the RT60 out
    of the FILENAME instead would trust a number nobody re-derives.
    """
    lib = DiskAssetLibrary(root=str(root), target_fs=fs, rir_subdir=subdir)
    return sorted(({"key": a.key, "rt60": float(a.rt60)} for a in lib.rirs),
                  key=lambda d: d["rt60"])


def write_rir(h: np.ndarray, path: Path, fs: int = TARGET_FS) -> None:
    """
    Write one synthetic RIR mono @ fs, PCM_24 — the SAME container the real
    library uses. A format difference between the two arms would be a confound
    in the very comparison this set exists for.

    Peak-normalized only to avoid clipping on write: absolute RIR gain is
    irrelevant downstream because `apply_rir` renormalizes the convolved signal
    over the input's active-speech region (audio_pipeline trap 2).
    """
    import soundfile as sf
    peak = float(np.max(np.abs(h)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError(f"degenerate RIR (peak={peak}) for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), (h / peak) * 0.99, int(fs), subtype="PCM_24")


def clear_generated(sim_dir: Path, prefix: str = "sim_") -> int:
    """
    Remove wavs THIS script wrote (prefix-matched) so a regeneration cannot leave
    a stale file behind and inflate the library count. Files we did not write are
    left alone and reported.
    """
    if not sim_dir.is_dir():
        return 0
    removed = 0
    for p in sorted(sim_dir.glob("*.wav")):
        if p.name.startswith(prefix):
            p.unlink()
            removed += 1
        else:
            print(f"  NOTE: leaving foreign file {p} in place")
    return removed


# ============================================================================
# DRIVER
# ============================================================================

def generate_paired_set(root: Path = ROOT, real_subdir: str = REAL_SUBDIR,
                        sim_subdir: str = SIM_SUBDIR, fs: int = TARGET_FS,
                        tol: float = PAIR_TOL_S, accept_tol: float = ACCEPT_TOL_S,
                        max_iter: int = MAX_CALIBRATION_ITERS,
                        verbose: bool = True) -> dict:
    """
    Full R2.5: measure the real set -> calibrate one synthetic twin per real RIR
    -> write them -> RE-MEASURE FROM DISK -> pair on measured RT60. Returns the
    pairing result plus the per-file metadata that goes into rir_pairs.json.
    """
    real = read_real_rirs(root, real_subdir, fs)
    n = len(real)
    print(f"\nreal set: {n} RIRs, measured RT60 "
          f"{real[0]['rt60']:.3f}-{real[-1]['rt60']:.3f} s  ({root / real_subdir})")

    sim_dir = root / sim_subdir
    removed = clear_generated(sim_dir)
    if removed:
        print(f"  cleared {removed} previously generated file(s) from {sim_dir}")

    print(f"\ncalibrating {n} synthetic twins "
          f"(accept |measured - target| <= {accept_tol:.3f} s)")
    meta_by_key: dict[str, dict] = {}
    for rank, r in enumerate(real):
        want = r["rt60"]
        geo, dist = config_for(rank, n)
        print(f"  [{rank + 1:>2}/{n}] want {want:.3f} s")
        cand = calibrate_rir(want, geo, dist, fs=fs, accept_tol=accept_tol,
                             max_iter=max_iter, verbose=verbose)
        path = sim_dir / (f"sim_rt60-{cand['rt60_realized']:.2f}_"
                          f"{cand['geometry']}_d{cand['distance_m']:.1f}"
                          f"_r{rank:02d}.wav")
        write_rir(cand["h"], path, fs)
        meta = {k: v for k, v in cand.items() if k != "h"}
        meta["rt60_realized_inmemory"] = round(meta.pop("rt60_realized"), 4)
        meta["paired_to_real_rt60"] = round(float(want), 4)
        meta_by_key[str(path)] = meta

    # RE-MEASURE FROM DISK, through `DiskAssetLibrary` at the grid's target rate.
    # This IS the R2.5 DoD check that the sim library constructs: the number the
    # experiment delivers is the one read back through the library, not the one we
    # held in memory before writing.
    sim_on_disk = read_real_rirs(root, sim_subdir, fs)
    for s in sim_on_disk:
        s.update(meta_by_key.get(s["key"], {}))

    res = pair_by_measured(real, sim_on_disk, tol=tol)
    res["n_real"] = n
    res["n_sim"] = len(sim_on_disk)
    return res


def print_summary(res: dict) -> None:
    print("\n" + "=" * 78)
    print("PAIRED RIR SET — matched on MEASURED Schroeder T20, never on the "
          "Sabine target")
    print("=" * 78)
    print(f"  {'real RT60':>9} {'sim RT60':>9} {'delta':>7}  "
          f"{'geometry':<8} {'dist':>5} {'sabine tgt':>10} {'its':>4}  file")
    for p in res["pairs"]:
        print(f"  {p['real_rt60_measured']:>9.3f} {p['sim_rt60_measured']:>9.3f} "
              f"{p['delta_s']:>+7.3f}  {p.get('geometry', '?'):<8} "
              f"{p.get('distance_m', float('nan')):>4.1f}m "
              f"{p.get('rt60_sabine_target', float('nan')):>10.3f} "
              f"{p.get('calibration_iters', 0):>4}  {Path(p['sim_key']).name}")
    print(f"\n  pairs: {len(res['pairs'])}/{res.get('n_real', '?')}   "
          f"sim library: {res.get('n_sim', '?')} RIRs read back from disk   "
          f"max |delta| = {res['max_abs_delta']:.4f} s   "
          f"(tolerance {res['tol']:.3f} s)")


def write_pairs_json(res: dict, out: Path, root: Path, real_subdir: str,
                     sim_subdir: str, fs: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fs": int(fs),
        "real_dir": str(root / real_subdir),
        "sim_dir": str(root / sim_subdir),
        "tolerance_s": res["tol"],
        "matched_on": ("measured Schroeder T20 RT60 of BOTH files "
                       "(conditions._rt60_schroeder), never the Sabine target "
                       "handed to pyroomacoustics"),
        "n_pairs": len(res["pairs"]),
        "max_abs_delta_s": res["max_abs_delta"],
        "unmatched_real": res["unmatched_real"],
        "unmatched_sim": res["unmatched_sim"],
        "geometries": [{"name": g.name, "dims": list(g.dims),
                        "distances": list(g.distances)} for g in GEOMETRIES],
        "pairs": res["pairs"],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out} ({len(res['pairs'])} pairs)")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--real-subdir", default=REAL_SUBDIR)
    ap.add_argument("--sim-subdir", default=SIM_SUBDIR)
    ap.add_argument("--fs", type=int, default=TARGET_FS)
    ap.add_argument("--tol", type=float, default=PAIR_TOL_S,
                    help="max |measured RT60| difference allowed within a pair")
    ap.add_argument("--accept-tol", type=float, default=ACCEPT_TOL_S,
                    help="in-loop calibration tolerance (tighter, leaves margin)")
    ap.add_argument("--max-iter", type=int, default=MAX_CALIBRATION_ITERS)
    ap.add_argument("--out", default=str(PAIRS_JSON))
    ap.add_argument("--quiet", action="store_true",
                    help="hide the per-iteration calibration trace")
    a = ap.parse_args(list(argv) if argv is not None else None)

    res = generate_paired_set(Path(a.root), a.real_subdir, a.sim_subdir, a.fs,
                              tol=a.tol, accept_tol=a.accept_tol,
                              max_iter=a.max_iter, verbose=not a.quiet)
    print_summary(res)

    if res["n_sim"] != res["n_real"]:
        print(f"\nFAILED: the sim library read back {res['n_sim']} RIRs but the "
              f"real set has {res['n_real']} — the sim arm must mirror the real "
              f"arm one-for-one (stale or unreadable files in {a.sim_subdir}?)")
        return 1

    if not res["ok"]:
        bad = ", ".join(f"{Path(k).name}" for k in res["unmatched_real"])
        print("\nFAILED: these real RIRs have no synthetic partner within "
              f"{a.tol:.3f} s of their MEASURED RT60: {bad}")
        print("Fix by adding a room geometry that can carry those RT60s, or "
              "raising --max-iter. Do NOT relax --tol: the sim2real comparison "
              "assumes the two arms deliver the same amount of reverb.")
        return 1

    write_pairs_json(res, Path(a.out), Path(a.root), a.real_subdir,
                     a.sim_subdir, a.fs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
