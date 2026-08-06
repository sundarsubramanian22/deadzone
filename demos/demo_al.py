#!/usr/bin/env python3
"""
demo_al.py — watch the active-learning surrogate walk onto the failure boundary.

    ./.venv/bin/python demos/demo_al.py            # ~20 s, offline, deterministic
    ./.venv/bin/python demos/demo_al.py --fast     # short rehearsal run
    ./.venv/bin/python demos/demo_al.py --no-anim  # print frames sequentially (piping / logs)

WHAT THIS RUNS AGAINST, AND WHY IT IS NOT THE API. The oracle is a Gaussian
process fitted to the TRAINING half of the measured grid (`results/master.csv`),
with a disjoint half held back to score fidelity. That is the protocol SPEC
A.R4.7 mandates for the cheap replicates: a live API oracle costs ~5,000 calls
and minutes per seed, which is not demoable, and the held-out conditions are real
measurements the arms have never seen. Nothing here touches the network.

WHAT THIS DEMO CLAIMS, AND WHAT IT DOES NOT. It shows the MECHANISM: straddle
acquisition concentrating its evaluations on the failure contour instead of
spreading them evenly over the cube. It does NOT claim a savings number — the
active-vs-random gap has large seed variance (SPEC A.R5.5), so the headline
savings figure is a multi-seed result in the write-up, not a 20-second demo.

WHAT TO WATCH. The map is a 2-D slice of the 5-D factor space (reverb on x, SNR
on y) held at a harsh channel chosen so the failure contour crosses the middle of
the plane -- see SLICE_FIXED for why that is not the #1 dead zone's channel and
why it was not moved to match. Shading is the
surrogate's predicted WER; the bright cells are where it predicts WER within a
hair of the failure threshold. As evaluations are spent, watch two things move
together: the bright band sharpens, and the median distance of each new pick from
the failure contour falls.
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
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")           # sklearn GP length-scale bound chatter

from deadzone.active_learning import (DEFAULT_THRESHOLD, GPSurrogate, active_learn,
                             boundary_rmse, random_raw)
from deadzone.design import DEFAULT_FACTOR_SPACE as SPACE

MASTER = Path("results/master.csv")
MODEL = "nova-3"

# The slice we draw: a deliberately harsh channel -- babble + opus-lowrate + full
# mic rolloff -- chosen because the 0.5 WER contour runs across the MIDDLE of this
# plane, which is the only reason the picture is legible.
#
# THIS IS NOT THE #1 DEAD ZONE'S CHANNEL, AND IT USED TO CLAIM IT WAS. The comment
# here previously said "held at the channel settings of the #1 measured dead zone".
# That was true of the OLD #1 (rt60-0.7_snr-20_babble_opus-lowrate_roll-1), which
# ranked first only under the confidence/WER pairing defect and is now classified
# `silence_driven` (see demo_break.py). The corrected #1 is
# rt60-0.45_snr-0_engine_g726_roll-0 -- engine / g726 / flat mic.
#
# The slice was NOT moved to match it, and the reason is measured rather than
# assumed. On the engine/g726/roll-0 plane the surrogate's predicted WER spans
# -0.224 to 0.520 over the rt60 x SNR grid: it only grazes the 0.5 threshold in one
# corner (1.7% of cells inside the draw band, against 6.8% here), and a large part
# of the plane goes NEGATIVE, which is unphysical for a WER and reads as a broken
# render on a projector. Moving the slice would trade a legible contour for a corner
# artifact. So this plane is chosen for legibility of the MECHANISM -- straddle
# acquisition concentrating on a contour -- and it is not a depiction of the
# headline condition. demo_break.py is the demo that shows the headline condition.
SLICE_FIXED = {"noise_type": "babble", "codec": "opus-lowrate", "mic_rolloff": 1.0}

RAMP = " .:-=+*#%@"
DRAW_BAND = 0.03     # |predicted WER - threshold| <= this  ==  drawn as boundary
NEAR_BAND = 0.05     # "landed on the contour" for the concentration statistic
DEGENERATE_SD = 0.06  # a surrogate flatter than this is not worth drawing


class Ink:
    def __init__(self, on: bool):
        self.on = on

    def _w(self, c, s):
        return f"\033[{c}m{s}\033[0m" if self.on else s

    def dim(self, s):    return self._w("2", s)
    def bold(self, s):   return self._w("1", s)
    def red(self, s):    return self._w("1;31", s)
    def cyan(self, s):   return self._w("36", s)
    def yellow(self, s): return self._w("33", s)
    def green(self, s):  return self._w("32", s)
    def inv(self, s):    return self._w("7", s)


# --------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------

def load_master_rows(path: Path = MASTER) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"demo_al: {path} not found -- run the grid first (SPEC A.R4).")
    csv.field_size_limit(10**9)
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["model"] == MODEL]


def _encode(sample) -> np.ndarray:
    row = []
    for f in SPACE.factors:
        v = sample[f.name]
        row.append(float(v) if f.kind == "continuous"
                   else float(f.levels.index(v)) + 0.5)
    return np.asarray(row, dtype=float)


def _local_split(rows: list[dict], seed: int, holdout_frac: float = 0.4) -> dict:
    """Fallback split, used only if analysis/al_savings.py is mid-edit: condition-level
    mean WER, split into an oracle-training half and a disjoint held-out half."""
    agg: dict[str, list] = {}
    for r in rows:
        if r.get("failed", "False") == "True":
            continue
        try:
            wer = float(r["wer"])
        except (TypeError, ValueError):
            continue
        rt60 = float(r.get("rir_rt60_measured") or r["rt60"])
        agg.setdefault(r["condition_name"],
                       [[], {"rt60": rt60, "snr_db": float(r["snr_db"]),
                             "noise_type": r["noise_type"], "codec": r["codec"],
                             "mic_rolloff": float(r["mic_rolloff"])}])[0].append(wer)
    names = sorted(agg)
    X = np.array([_encode(agg[n][1]) for n in names])
    y = np.array([float(np.mean(agg[n][0])) for n in names])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    n_test = max(8, int(round(holdout_frac * len(names))))
    te, tr = order[:n_test], order[n_test:]
    return {"X_train": X[tr], "y_train": y[tr], "X_test": X[te], "y_test": y[te],
            "n_train": int(len(tr)), "n_test": int(len(te)), "n_conditions": len(names)}


def build_oracle(seed: int = 0):
    """
    Surrogate oracle + a disjoint held-out test set, both from the measured grid.
    Prefers analysis/al_savings.py (the owner of this split); falls back to the
    same construction locally if that module cannot be imported.
    """
    rows = load_master_rows()
    n_conditions = len({r["condition_name"] for r in rows})
    try:
        from deadzone.analysis.al_savings import (master_split_for_al,
                                         surrogate_oracle_from_master)
        split = master_split_for_al(rows, SPACE, model=MODEL, seed=seed)
        oracle, meta = surrogate_oracle_from_master(space=SPACE, split=split, seed=seed)
        owner = "deadzone.analysis.al_savings"
    except Exception as exc:                      # noqa: BLE001 -- demo must survive
        print(f"demo_al: deadzone.analysis.al_savings unavailable ({type(exc).__name__}); "
              f"using the local fallback split.", file=sys.stderr)
        split = _local_split(rows, seed)
        gp = GPSurrogate(SPACE, seed=seed).fit(split["X_train"], split["y_train"])

        def oracle(sample):                        # noqa: D401
            return float(np.ravel(gp.predict(_encode(sample).reshape(1, -1),
                                             return_std=False))[0])
        meta, owner = {"provenance": "surrogate"}, "demo_al fallback"
    split.setdefault("n_conditions", n_conditions)
    return oracle, split, meta, owner


def surrogate_spread(oracle, n: int = 300, seed: int = 99) -> float:
    """How much structure the fitted oracle actually has. A GP that collapses to a
    constant makes every downstream picture meaningless, so we check and say so."""
    X = random_raw(SPACE, n, seed)
    return float(np.std([oracle(SPACE.decode(x)) for x in X]))


# --------------------------------------------------------------------------
# the slice map
# --------------------------------------------------------------------------

def slice_grid(nx: int, ny: int):
    rt, snr = SPACE.factors[0], SPACE.factors[1]
    xs = np.linspace(rt.low, rt.high, nx)
    ys = np.linspace(snr.high, snr.low, ny)          # top row = quiet (high SNR)
    pts = []
    for y in ys:
        for x in xs:
            s = dict(SLICE_FIXED)
            s["rt60"], s["snr_db"] = float(x), float(y)
            pts.append(_encode(s))
    return xs, ys, np.asarray(pts)


def render_map(gp: GPSurrogate, xs, ys, pts, ink: Ink):
    mean, std = gp.predict(pts, return_std=True)
    nx, ny = len(xs), len(ys)
    mean, std = mean.reshape(ny, nx), std.reshape(ny, nx)
    on_band = np.abs(mean - DEFAULT_THRESHOLD) <= DRAW_BAND
    band_sigma = float(std[on_band].mean()) if on_band.any() else float("nan")

    lines = []
    for i, y in enumerate(ys):
        cells = ""
        for j in range(nx):
            k = int(np.clip(mean[i, j], 0, 1) * (len(RAMP) - 1))
            cells += ink.red(ink.bold("█")) if on_band[i, j] else ink.dim(RAMP[k])
        lines.append(f"  {y:5.1f} │{cells}│")
    lines.append("        └" + "─" * nx + "┘")
    lines.append("         " + f"{xs[0]:.2g}" + " " * max(1, nx - 8) + f"{xs[-1]:.2g}")
    return lines, band_sigma


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Terminal active-learning demo "
                                             "(offline, surrogate oracle).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seed", type=int, default=12, help="initial LHS design size")
    ap.add_argument("--budget", type=int, default=48, help="acquisitions after the seed")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--fast", action="store_true", help="short rehearsal run")
    ap.add_argument("--no-anim", action="store_true",
                    help="print frames one after another instead of redrawing")
    ap.add_argument("--no-color", dest="color", action="store_false")
    ap.add_argument("--hold", type=float, default=1.0, help="seconds per frame")
    args = ap.parse_args(argv)

    if args.fast:
        args.n_seed, args.budget, args.frames, args.hold = 10, 24, 4, 0.3

    color = args.color and sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")
    ink = Ink(color)
    anim = color and not args.no_anim
    width = max(76, min(shutil.get_terminal_size((96, 24)).columns, 110))
    t0 = time.time()

    print()
    print(ink.inv(("  DEADZONE  —  the grid is expensive, so let the surrogate choose"
                   + " " * width)[:width]))
    print()
    oracle, split, meta, owner = build_oracle(args.seed)
    X_test, y_test = split["X_test"], split["y_test"]
    spread = surrogate_spread(oracle)

    print(f"  oracle      GP fitted to {split['n_train']} of "
          f"{split['n_conditions']} measured conditions ({owner}); "
          f"{split['n_test']} held out")
    print(f"  provenance  {ink.yellow(meta.get('provenance', 'surrogate'))} — "
          f"a replication device, not a measurement")
    print(f"  slice       noise={SLICE_FIXED['noise_type']} "
          f"codec={SLICE_FIXED['codec']} mic_rolloff={SLICE_FIXED['mic_rolloff']:g}"
          f"   ·   failure threshold WER = {DEFAULT_THRESHOLD:g}")
    if spread < DEGENERATE_SD:
        print(ink.yellow(f"  WARNING     this seed's surrogate is nearly flat "
                         f"(predictive sd {spread:.3f}); the map will be "
                         f"uninformative. Try --seed 0."))
    print()

    # --- the active arm: the acquisition order comes from active_learning.py --
    print(ink.dim("  running straddle (boundary-seeking) acquisition ..."))
    traj = active_learn(oracle, SPACE, strategy="boundary", n_seed=args.n_seed,
                        budget=args.budget, seed=args.seed)

    # --- the passive arm: same oracle, same budget, nested so the curve is fair
    n_max = args.n_seed + args.budget
    X_rand = random_raw(SPACE, n_max, args.seed)
    y_rand = np.array([float(oracle(SPACE.decode(x))) for x in X_rand])

    xs, ys, pts = slice_grid(nx=34, ny=11)
    checkpoints = sorted({args.n_seed} | {
        int(round(args.n_seed + (args.budget * k) / (args.frames - 1)))
        for k in range(args.frames)})

    def dist(v):                       # distance from the failure contour
        return float(np.median(np.abs(np.asarray(v) - DEFAULT_THRESHOLD)))

    frame_h = 0
    for n in checkpoints:
        gp_a = GPSurrogate(SPACE, seed=args.seed).fit(traj.X_raw[:n], traj.y[:n])
        body, band_sigma = render_map(gp_a, xs, ys, pts, ink)
        acquired = traj.y[args.n_seed:n]
        d_acq = dist(acquired) if len(acquired) else dist(traj.y[:n])
        d_rand = dist(y_rand[:n])

        lines = [
            "  " + ink.bold(f"evaluations spent: {n:3d}")
            + ink.dim(f"   (the full grid cost {split['n_conditions']} conditions "
                      f"x 40 clips)"),
            "  " + ink.dim("SNR dB"),
            *body,
            "  " + ink.dim("        reverb RT60 (s) →"),
            "",
            "  " + ink.red("█") + ink.dim(f" = predicted failure boundary (within "
                                          f"{DRAW_BAND:g} of WER {DEFAULT_THRESHOLD:g})"
                                          f"    shading = predicted WER"),
            "  uncertainty on the boundary (mean GP sigma)  " + ink.cyan(f"{band_sigma:.4f}"),
            "  median |WER - threshold| of the picks        "
            + ink.green(f"{d_acq:.4f}") + ink.dim(f"   (uniform random: {d_rand:.4f})"),
        ]
        if n > args.n_seed:
            st = traj.steps[n - args.n_seed - 1]["chosen"]
            lines.append("  last pick   " + ink.cyan(
                f"rt60 {st['rt60']:.2f}  snr {st['snr_db']:.1f}  {st['noise_type']}"
                f"  {st['codec']}  roll {st['mic_rolloff']:.2f}"))
        else:
            lines.append(ink.dim("  last pick   (space-filling seed design, "
                                 "no acquisition yet)"))

        if anim and frame_h:
            sys.stdout.write(f"\033[{frame_h}A")
        for ln in lines:
            sys.stdout.write(("\033[2K" if anim else "") + ln + "\n")
        sys.stdout.flush()
        frame_h = len(lines)
        time.sleep(args.hold)

    # --- the honest close ---------------------------------------------------
    acq = traj.y[args.n_seed:]
    thirds = max(1, len(acq) // 3)
    blocks = [acq[i:i + thirds] for i in range(0, thirds * 3, thirds)]
    gp_a = GPSurrogate(SPACE, seed=args.seed).fit(traj.X_raw, traj.y)
    gp_r = GPSurrogate(SPACE, seed=args.seed).fit(X_rand, y_rand)
    rmse_a = boundary_rmse(gp_a, X_test, y_test)
    rmse_r = boundary_rmse(gp_r, X_test, y_test)
    near = lambda v: float(np.mean(np.abs(np.asarray(v) - DEFAULT_THRESHOLD) <= NEAR_BAND))

    print()
    print("  " + ink.bold("WHERE THE EVALUATIONS WENT"))
    print(ink.dim("    median distance from the failure contour, by block of picks"))
    print("    {:<38}{:.3f}".format("seed design (space-filling LHS)",
                                    dist(traj.y[:args.n_seed])))
    for i, b in enumerate(blocks, start=1):
        if len(b):
            print("    {:<38}{:.3f}".format(f"acquisitions, third {i}", dist(b)))
    print("    {:<38}{}".format(f"{n_max} uniform-random evaluations",
                                ink.yellow("{:.3f}".format(dist(y_rand)))))
    print()
    print(f"    Straddle put {near(acq):.0%} of its picks within {NEAR_BAND:g} WER of the"
          f" failure")
    print(f"    threshold; uniform random managed {near(y_rand):.0%}. That is the "
          f"mechanism —")
    print("    the loop spends its budget on the edge instead of on the cube.")
    print()
    print(ink.dim(f"    Held-out boundary RMSE this seed: active {rmse_a:.3f}, "
                  f"random {rmse_r:.3f}."))
    print(ink.dim("    ONE seed against a surrogate oracle. Seed variance in "
                  "active-vs-random is"))
    print(ink.dim("    large (SPEC A.R5.5), so the savings headline is a multi-seed "
                  "result in the"))
    print(ink.dim("    write-up, not this demo. What this demo shows is the "
                  "acquisition behaviour."))
    print(ink.dim(f"    [{time.time() - t0:.1f}s total, no network, no API key]"))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
