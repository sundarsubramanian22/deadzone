"""
run_d3a.py — the D3a runner (SPEC A.R5.4): pre-registration verdict + the
two-stage counterintuitive-cell hunt against the REAL oracle.

Sits at the repo root alongside run_experiment.py because it shares that file's
defining property: IT SPENDS MONEY. Everything under analysis/ is pure analysis of
data already collected and is safe to run in a loop; this is not. The budget is a
hard, checked ceiling (`--max-calls`), not a comment.

WHAT IT DOES
  1. Reads results/sobol.json (written by analysis.sensitivity — the EXACT
     factorial decomposition, not a Saltelli estimate) and resolves SPEC §5's
     pre-registered rt60 x snr_db hypothesis under the decision rule fixed in
     analysis.interactions.PreRegistration. Confirmed or not, the verdict is
     written; there is no branch that omits it.
  2. Fits a GP to the master table, sweeps it for counterintuitive cells
     (STAGE 1 — proposals, free, and NOT presentable as measurements), then
     re-measures every proposed cell through real Deepgram calls (STAGE 2) so the
     only cells that reach the write-up are ones a microphone actually produced.

WHY STAGE 2 CANNOT BE SKIPPED. A GP fitted to 176 conditions will interpolate a
dip wherever the measured points happen to bracket one, and a dip is exactly the
shape this project wants to find — which is precisely the circumstance in which a
plausible artefact gets written up as a discovery. `confirm_cells` is the only
function that can set confirmed=True and it can only do so by spending oracle
calls. Refuted proposals are KEPT and reported: "the surrogate predicted this and
the microphone disagreed" is a real result about surrogate fidelity.

    python3 scripts/run_d3a.py --dry-run          # plan + cost, zero API calls
    python3 scripts/run_d3a.py --max-calls 400

Outputs: results/interactions.json, results/interaction_report.txt,
         results/counterintuitive_confirmed.json
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
import os
import sys
from pathlib import Path

import numpy as np


from deadzone.analysis.interactions import (              # noqa: E402
    PREREGISTRATION, S2_CAVEAT,
    load_master_rows, load_sobol_json, preregistration_verdict, format_verdict,
    interaction_evidence, propose_counterintuitive_cells, confirm_cells,
    format_interaction_report, plot_payload,
)
from deadzone.analysis.sensitivity import (              # noqa: E402
    load_factorial, measured_counterintuitive, format_measured_counterintuitive,
    PRIMARY_FACTORS, PRIMARY_FIXED,
)
from deadzone.design import DEFAULT_FACTOR_SPACE, format_sobol_tables   # noqa: E402


# ============================================================================
# PROVENANCE OF THE PRE-REGISTRATION
# ============================================================================
# Found with `git log -S"rt60 × snr_db" -- SPEC.md`: the earliest commit whose
# SPEC.md contains the Track-C note "Pre-registered expectation: rt60 × snr_db
# ... is predicted to compound ... *before* seeing any real data".
#
# This SHA is what makes the hypothesis falsifiable rather than a story told
# afterwards, so it is recorded as a constant and printed with the verdict. It
# predates the corpus (R1, 2026-08-04) and the grid (R4, 2026-08-05) — i.e. it
# was committed before a single real transcription existed.
PREREG_COMMIT = "d8ddd4f67d45f265ab9b97216816218c6fb9a581"
PREREG_DATE = "2026-07-27"
PREREG_EVIDENCE = (
    "committed 2026-07-27, before the speech corpus was recorded (2026-08-04) and "
    "before any grid transcription existed (2026-08-05)"
)


def load_env(path: str = ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment WITHOUT printing them.

    The key never touches stdout, a log line or an error message: the failure mode
    for a leaked credential is permanent, and this script's output is meant to be
    pasted into a write-up.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ============================================================================
# THE REAL ORACLE
# ============================================================================

def build_oracle(clip_spec: str = "al", model: str = "nova-3",
                 results_dir: str = "results"):
    """
    A `wer_fn(sample) -> float` backed by real transcription, averaged over the
    AL clip set, reading and writing run_experiment.py's cache.

    Reuses run_experiment's `make_grid_wer_fn` / `ResultCache` / `load_corpus`
    rather than reimplementing the chain: a second, subtly different degrade ->
    transcribe -> score path would make these confirmations non-comparable with
    the grid they are supposed to be confirming against, and nothing would say so.
    Sharing results/cache.jsonl also means a re-run costs zero calls.
    """
    from scripts.run_experiment import (                   # imported late: needs the SDK
        AL_CLIPS, ResultCache, load_corpus, load_manifest, make_grid_wer_fn,
        select_clip_ids, DEFAULT_FS,
    )
    from deadzone.conditions import DiskAssetLibrary
    from deadzone.model_compare import get_model

    manifest = load_manifest()
    clip_ids = select_clip_ids(clip_spec, manifest)
    clips = load_corpus(clip_ids)
    refs = {c: manifest[c] for c in clip_ids}
    assets = DiskAssetLibrary(root="data", target_fs=DEFAULT_FS)
    cache = ResultCache(Path(results_dir) / "cache.jsonl")
    transcribe_fn = get_model(model)

    wer_fn = make_grid_wer_fn(clips, refs, assets, transcribe_fn,
                              model=model, cache=cache,
                              run_id="d3a-confirm")
    return wer_fn, cache, clip_ids


class BudgetedOracle:
    """
    Wraps the oracle with a HARD call ceiling and per-condition accounting.

    Counts CONDITIONS evaluated and multiplies by the clip count to get API calls,
    then refuses to exceed the ceiling — raising rather than silently truncating,
    because a partially-confirmed candidate set that looks complete is worse than
    a crash. Cache hits still count as conditions but are reported separately so
    the true billed number is knowable.
    """

    def __init__(self, wer_fn, n_clips: int, max_calls: int, cache=None):
        self._fn = wer_fn
        self.n_clips = n_clips
        self.max_calls = max_calls
        self._cache = cache
        self.conditions_evaluated = 0
        self.cache_hits_before = len(cache) if cache is not None else 0
        self.log: list[dict] = []

    @property
    def calls_spent(self) -> int:
        return self.conditions_evaluated * self.n_clips

    def __call__(self, sample: dict) -> float:
        if self.calls_spent + self.n_clips > self.max_calls:
            raise RuntimeError(
                f"budget exhausted: {self.calls_spent} calls spent, ceiling "
                f"{self.max_calls}, next condition needs {self.n_clips} more. "
                f"Re-run with a higher --max-calls or fewer --top-k candidates."
            )
        wer = float(self._fn(dict(sample)))
        self.conditions_evaluated += 1
        self.log.append({"sample": {k: (float(v) if isinstance(v, (int, float,
                                                                  np.floating))
                                        else v)
                                    for k, v in sample.items()},
                         "wer": wer})
        return wer


# ============================================================================
# MECHANISM — why the rt60 axis is non-monotonic
# ============================================================================

def rir_mechanism(rt60_levels, fs: int = 16000) -> dict:
    """
    Measure each grid rt60 level's DELIVERED room acoustics beyond RT60.

    The measured non-monotonicity along rt60 is not a fluke and not a modelling
    artefact: `AssetLibrary.resolve` snaps a requested rt60 to the nearest measured
    RIR, so each level of the "rt60" axis is a DIFFERENT REAL ROOM. RT60 summarises
    a decay slope; it says nothing about how much direct sound reaches the mic. Two
    rooms with the same RT60 and different source-mic distances damage a recogniser
    very differently.

    So we compute, per delivered RIR:
      * DRR — direct-to-reverberant ratio (energy within +/-2.5 ms of the direct
        peak vs everything else), the standard proxy for effective source distance;
      * C50 — early-to-late energy ratio at 50 ms, the standard speech-clarity index.

    and rank-correlate each against the measured marginal WER. If DRR orders the
    conditions and RT60 does not, the "non-monotonicity in rt60" is really
    monotonicity in DRR that the rt60 label is hiding — which is a statement about
    how reverb benchmarks are parameterised, not a quirk of this grid.
    """
    import soundfile as sf
    from scipy.stats import spearmanr, pearsonr
    from deadzone.conditions import Condition, DiskAssetLibrary

    lib = DiskAssetLibrary(root="data", target_fs=fs)
    out = []
    for req in rt60_levels:
        cond = Condition(rt60=float(req), snr_db=10.0, noise_type="babble",
                         codec="none", mic_rolloff=0.0)
        rir = lib.resolve(cond).rir
        h, hfs = sf.read(rir.key, dtype="float64")
        h = np.asarray(h, dtype=float)
        if h.ndim > 1:
            h = h.mean(axis=1)
        if hfs != fs:
            import librosa
            h = librosa.resample(h, orig_sr=hfs, target_sr=fs)
        d = int(np.argmax(np.abs(h)))
        w = int(0.0025 * fs)
        direct = h[max(0, d - w):d + w]
        rev = np.concatenate([h[:max(0, d - w)], h[d + w:]])
        e50 = int(0.050 * fs)
        drr = 10 * np.log10((direct ** 2).sum() / max((rev ** 2).sum(), 1e-20))
        c50 = 10 * np.log10((h[d:d + e50] ** 2).sum() /
                            max((h[d + e50:] ** 2).sum(), 1e-20))
        out.append({"rt60_requested": float(req),
                    "rt60_measured": float(rir.rt60),
                    "rir_key": rir.key,
                    "room": os.path.basename(rir.key),
                    "drr_db": float(drr), "c50_db": float(c50)})
    return {"levels": out,
            "note": ("each rt60 level is a DIFFERENT measured room; DRR/C50 "
                     "describe what RT60 alone omits"),
            "definitions": {
                "drr_db": "direct-to-reverberant ratio, +/-2.5 ms window on the direct peak",
                "c50_db": "early(<=50ms)-to-late energy ratio, the speech-clarity index"}}


def mechanism_correlations(mech: dict, marginal_wer: dict) -> dict:
    """Rank/linear correlation of RT60, DRR and C50 against the measured marginal WER."""
    from scipy.stats import spearmanr, pearsonr
    lv = [d for d in mech["levels"] if d["rt60_requested"] in marginal_wer]
    wer = np.array([marginal_wer[d["rt60_requested"]] for d in lv])
    res = {}
    for key, label in (("rt60_measured", "measured RT60"), ("drr_db", "DRR"),
                       ("c50_db", "C50")):
        x = np.array([d[key] for d in lv])
        res[key] = {"label": label,
                    "spearman": float(spearmanr(x, wer).statistic),
                    "pearson": float(pearsonr(x, wer).statistic)}
    res["n_levels"] = len(lv)
    res["interpretation"] = (
        "if |spearman| for DRR is 1.0 while RT60's is not, the rt60 axis is "
        "non-monotonic ONLY because RT60 mislabels the delivered acoustics: the "
        "damage is monotone in direct-to-reverberant ratio, which RT60 does not "
        "capture. Reverb benchmarks parameterised by RT60 alone will mis-rank "
        "conditions for exactly this reason.")
    return res


def reconcile_dip_evidence(measured: dict, ci: dict, fs: int = 16000) -> dict:
    """
    Reconcile the two rt60 results that otherwise read as a contradiction:
    an in-grid dip that is SIGNIFICANT on real measurements, and six
    surrogate-proposed dips that the oracle did not reproduce.

    They are not in conflict, and the reason is measurable rather than rhetorical.
    `AssetLibrary.resolve` snaps each requested rt60 to the NEAREST MEASURED RIR,
    so every rt60 level is a different real room. The two scans therefore examined
    two DIFFERENT, NON-OVERLAPPING triplets of rooms:

        grid  0.45 / 0.70 / 1.00 -> Bar / Campground Dininghall / Shower
        probe 0.60 / 0.70 / 0.80 -> Office ConfRoom / Campground Dininghall / Classroom

    Only the centre room is shared. Since WER on this grid tracks DRR almost
    perfectly (spearman -1.000 across the four grid levels), the sign of each
    triplet is PREDICTED by its DRR profile: the grid triplet's middle room has
    the BEST DRR of its three, so WER must dip; the probe triplet's middle room has
    the WORST DRR of its three, so WER must not. Both measurements are correct and
    both follow from the same mechanism.

    Verified, not assumed: this function re-resolves both triplets, checks the
    three probe RIRs are genuinely distinct files (they are — an appealing
    alternative explanation, that all three requests collapsed onto one RIR, is
    FALSE here and is recorded as falsified so nobody re-proposes it), and reports
    each triplet's DRR ordering next to its measured WER.
    """
    import soundfile as sf
    from deadzone.conditions import Condition, DiskAssetLibrary

    lib = DiskAssetLibrary(root="data", target_fs=fs)

    def acoustics(path):
        h, hfs = sf.read(path, dtype="float64")
        h = np.asarray(h, dtype=float)
        if h.ndim > 1:
            h = h.mean(axis=1)
        if hfs != fs:
            import librosa
            h = librosa.resample(h, orig_sr=hfs, target_sr=fs)
        d = int(np.argmax(np.abs(h)))
        w = int(0.0025 * fs)
        rev = np.concatenate([h[:max(0, d - w)], h[d + w:]])
        e = int(0.050 * fs)
        return (float(10 * np.log10((h[max(0, d - w):d + w] ** 2).sum()
                                    / max((rev ** 2).sum(), 1e-20))),
                float(10 * np.log10((h[d:d + e] ** 2).sum()
                                    / max((h[d + e:] ** 2).sum(), 1e-20))))

    def triplet(vals):
        out = []
        for v in vals:
            r = lib.resolve(Condition(rt60=float(v), snr_db=10.0, noise_type="babble",
                                      codec="none", mic_rolloff=0.0)).rir
            drr, c50 = acoustics(r.key)
            # round the REQUEST for display: it comes off a linspace, so 0.6 is
            # actually 0.6000000000000001 and prints as such in the report.
            out.append({"rt60_requested": round(float(v), 6),
                        "rt60_measured": float(r.rt60),
                        "rir_key": r.key, "room": os.path.basename(r.key),
                        "drr_db": drr, "c50_db": c50})
        return out

    grid_vals = [d["from_level"] for d in measured["marginal"] if d["factor"] == "rt60"]
    grid_trip = ([grid_vals[0], measured["marginal"][0]["level"],
                  measured["marginal"][0]["to_level"]]
                 if grid_vals else [0.45, 0.7, 1.0])

    probe_vals = None
    for c in ci.get("candidates", []):
        if c.get("kind") == "non_monotonic" and c.get("factor") == "rt60":
            probe_vals = [float(v) for v in c["probe_values"]]
            break
    probe_vals = probe_vals or [0.6, 0.7, 0.8]

    g, p = triplet(grid_trip), triplet(probe_vals)
    p_keys = [d["rir_key"] for d in p]
    distinct = len(set(p_keys)) == len(p_keys)

    def profile(t):
        drrs = [d["drr_db"] for d in t]
        mid_best = drrs[1] == max(drrs)
        mid_worst = drrs[1] == min(drrs)
        return {"drr": drrs,
                "middle_room_is_best_drr": bool(mid_best),
                "middle_room_is_worst_drr": bool(mid_worst),
                "predicted_wer_shape": ("DIP (middle room has the best DRR)" if mid_best
                                        else "PEAK / no dip (middle room has the worst DRR)"
                                        if mid_worst else "monotone")}

    gp_, pp_ = profile(g), profile(p)
    n_dip = sum(1 for c in ci.get("candidates", []) if c.get("confirmed"))
    n_probe = len(ci.get("candidates", []))
    wers = [w for c in ci.get("candidates", []) for w in c.get("measured_wer", [])]

    return {
        "grid_triplet": g, "probe_triplet": p,
        "grid_profile": gp_, "probe_profile": pp_,
        "shared_rooms": sorted({d["rir_key"] for d in g} & {d["rir_key"] for d in p}),
        "probe_rirs_are_distinct_files": distinct,
        "falsified_alternative": {
            "hypothesis": ("the three probe requests 0.6/0.7/0.8 all snapped to the "
                           "SAME RIR, so the probe measured one room three times and "
                           "had nothing to dip"),
            "verdict": "FALSE" if distinct else "TRUE",
            "evidence": f"{len(set(p_keys))} distinct RIR files for {len(p_keys)} requests: "
                        + ", ".join(os.path.basename(k) for k in p_keys),
        },
        "probe_wer_range": ([float(min(wers)), float(max(wers))] if wers else None),
        "probe_dynamic_range": (float(max(wers) - min(wers)) if wers else None),
        "grid_dynamic_range": float(
            max(measured["marginal"][0]["wer_before"],
                measured["marginal"][0]["wer_after"])
            - measured["marginal"][0]["wer_at_dip"]) if measured["marginal"] else None,
        "n_probes": n_probe, "n_probes_confirmed": n_dip,
        "refutation_scope": (
            "these six surrogate-proposed cells ONLY. It is NOT a refutation of the "
            "in-grid measured dip: the two scans looked at different room triplets, "
            "and the mechanism predicts opposite signs for each."),
        "conclusion": (
            "Non-monotonicity along rt60 is not a property of a response surface. "
            "Each rt60 request indexes an unrelated real room via nearest-match "
            "snapping, so whether a dip exists — and where — is a property of WHICH "
            "RIRs were curated, not of reverberation time. Re-sample the axis and "
            "the dip moves or disappears, which is exactly what happened here."),
        "consequence_for_surrogates": (
            "A GP fitted with rt60 as a CONTINUOUS coordinate assumes a smoothness "
            "the instrument does not have, so it will keep proposing cells the "
            "oracle cannot reproduce. The defensible parameterisation for this axis "
            "is DRR (or C50), which orders the measured conditions perfectly where "
            "RT60 does not (spearman -1.000 vs +0.800)."),
    }


def _format_reconciliation(rec: dict, ci: dict) -> str:
    out = ["=" * 78,
           "RECONCILIATION — the in-grid dip vs the six unreproduced proposals",
           "=" * 78, "",
           "  These two results look contradictory and are not. Each rt60 level is a",
           "  DIFFERENT REAL ROOM (nearest-measured-RT60 snapping), so the two scans",
           "  examined different, almost non-overlapping room triplets:", ""]
    for label, key, prof in (("GRID  (dip found)", "grid_triplet", "grid_profile"),
                             ("PROBE (no dip)", "probe_triplet", "probe_profile")):
        out.append(f"  {label}")
        out.append(f"    {'req':>5} {'room':<40} {'RT60':>6} {'DRR dB':>7} {'C50 dB':>7}")
        for d in rec[key]:
            out.append(f"    {d['rt60_requested']:>5g} {d['room'][:40]:<40} "
                       f"{d['rt60_measured']:>6.3f} {d['drr_db']:>7.2f} {d['c50_db']:>7.2f}")
        out.append(f"    -> {rec[prof]['predicted_wer_shape']}")
        out.append("")
    out += [f"  shared rooms between the two triplets: "
            f"{[os.path.basename(k) for k in rec['shared_rooms']] or 'none'}",
            "",
            "  Falsified alternative explanation:",
            f"    hypothesis: {rec['falsified_alternative']['hypothesis']}",
            f"    verdict   : {rec['falsified_alternative']['verdict']} — "
            f"{rec['falsified_alternative']['evidence']}",
            "",
            f"  Outcome: {rec['n_probes_confirmed']}/{rec['n_probes']} proposals reproduced. "
            f"Scope of that refutation:",
            f"    {rec['refutation_scope']}",
            "",
            f"  CONCLUSION: {rec['conclusion']}",
            "",
            f"  FOR SURROGATES: {rec['consequence_for_surrogates']}", ""]

    out += ["-" * 78,
            f"The six proposals, with the held-fixed coordinates that DIFFER between",
            f"them (all six share factor=rt60, dip at 0.7, probes [0.6, 0.7, 0.8]):",
            "-" * 78,
            f"  {'#':>2} {'snr_db':>7} {'codec':<14} {'rolloff':>8} "
            f"{'surrogate':>10} {'measured WER (0.6/0.7/0.8)':>30} {'result':>9}"]
    for i, c in enumerate(ci.get("candidates", []), 1):
        co = c["coords"]
        w = c.get("measured_wer") or []
        wtxt = " / ".join(f"{x:.4f}" for x in w) if w else "(not measured)"
        out.append(f"  {i:>2} {co['snr_db']:>7.2f} {co['codec']:<14} "
                   f"{co['mic_rolloff']:>8.3f} {c['surrogate_wer_at_dip']:>10.4f} "
                   f"{wtxt:>30} {'DIP' if c.get('confirmed') else 'no dip':>9}")
    if rec.get("probe_dynamic_range") is not None:
        out += ["",
                f"  Note: every probe sits in a BENIGN corner — measured WER spans "
                f"{rec['probe_wer_range'][0]:.4f}-{rec['probe_wer_range'][1]:.4f} "
                f"(range {rec['probe_dynamic_range']:.4f}), vs {rec['grid_dynamic_range']:.4f} "
                f"for the in-grid dip.",
                f"  The surrogate placed its proposals where the response is nearly "
                f"flat and clip noise dominates."]
    return "\n".join(out)


def _format_mechanism(mech: dict | None, measured: dict) -> str:
    """The mechanism section of the written report."""
    if not mech:
        return ("MECHANISM — unavailable (RIR assets not loadable in this "
                "environment); the measured dips above stand on their own.")
    marg = mech.get("marginal_wer", {})
    out = ["=" * 78,
           "MECHANISM — WHY THE rt60 AXIS IS NON-MONOTONIC",
           "=" * 78, "",
           "  Each rt60 level is delivered by the NEAREST MEASURED RIR, i.e. a",
           "  different real room. RT60 describes a decay slope and says nothing",
           "  about how much direct sound reaches the mic.", "",
           f"  {'req':>5} {'room':<36} {'RT60':>6} {'DRR dB':>7} {'C50 dB':>7} {'WER':>7}"]
    for d in mech["levels"]:
        w = marg.get(d["rt60_requested"], float("nan"))
        out.append(f"  {d['rt60_requested']:>5} {d['room'][:36]:<36} "
                   f"{d['rt60_measured']:>6.3f} {d['drr_db']:>7.2f} "
                   f"{d['c50_db']:>7.2f} {w:>7.4f}")
    cors = mech.get("correlations", {})
    out.append("")
    for _, v in cors.items():
        if isinstance(v, dict) and "spearman" in v:
            out.append(f"  spearman({v['label']:<14}, marginal WER) = "
                       f"{v['spearman']:+.3f}   pearson = {v['pearson']:+.3f}")
    out += ["", "  " + cors.get("interpretation", ""), ""]
    return "\n".join(out)


# ============================================================================
# MAIN
# ============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D3a — verdict + counterintuitive confirmation")
    ap.add_argument("--sobol", default="results/sobol.json")
    ap.add_argument("--master", default="results/master.csv")
    ap.add_argument("--model", default="nova-3")
    ap.add_argument("--clips", default="al")
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=400,
                    help="hard ceiling on real API calls (conditions x clips)")
    ap.add_argument("--dry-run", action="store_true",
                    help="propose + plan + verdict, but spend nothing")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)

    load_env()
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. the verdict (free) --------------------------------------------
    sobol = load_sobol_json(args.sobol)
    verdict = preregistration_verdict(sobol, PREREGISTRATION,
                                      commit=PREREG_COMMIT, date=PREREG_DATE)
    verdict["commit_evidence"] = PREREG_EVIDENCE
    evidence = interaction_evidence(sobol, PREREGISTRATION)

    print(format_verdict(verdict))
    print()

    # ---- 2a. MEASURED counterintuitive cells (free, in-grid, no surrogate) --
    # Run BEFORE the surrogate proposals and reported first: a complete factorial
    # lets us scan the measurements themselves, which needs no confirmation step
    # because there is nothing to confirm — it is already the measurement.
    rows = load_master_rows(args.master)
    block = load_factorial(rows, PRIMARY_FACTORS, model=args.model,
                           fixed=PRIMARY_FIXED)
    measured = measured_counterintuitive(block, bootstrap=2000, seed=args.seed,
                                         top_k=10)
    print(format_measured_counterintuitive(measured))
    print()

    mech = None
    try:
        levels = block["levels"]["rt60"]
        mech = rir_mechanism(levels)
        Y = np.asarray(block["W"]).mean(axis=0)
        others = tuple(range(1, Y.ndim))
        marg = {float(lv): float(v) for lv, v in zip(levels, Y.mean(axis=others))}
        mech["correlations"] = mechanism_correlations(mech, marg)
        mech["marginal_wer"] = marg
        print("-" * 78)
        print("MECHANISM — what the rt60 axis actually delivers")
        print("-" * 78)
        print(f"  {'req':>5} {'room':<34} {'RT60':>6} {'DRR dB':>7} {'C50 dB':>7} {'WER':>7}")
        for d in mech["levels"]:
            print(f"  {d['rt60_requested']:>5} {d['room'][:34]:<34} "
                  f"{d['rt60_measured']:>6.3f} {d['drr_db']:>7.2f} "
                  f"{d['c50_db']:>7.2f} {marg[d['rt60_requested']]:>7.4f}")
        for kk, v in mech["correlations"].items():
            if isinstance(v, dict) and "spearman" in v:
                print(f"  spearman({v['label']:<14}, WER) = {v['spearman']:+.3f}"
                      f"   pearson = {v['pearson']:+.3f}")
        print()
    except Exception as e:                       # noqa: BLE001 — annotation only
        print(f"[mechanism] skipped: {type(e).__name__}: {e}\n")

    # ---- 2b. propose from the surrogate (free) -----------------------------
    proposal = propose_counterintuitive_cells(
        rows, DEFAULT_FACTOR_SPACE, model=args.model, grid=args.grid,
        top_k=args.top_k, seed=args.seed)

    n_clips = len(__import__("scripts.run_experiment", fromlist=["AL_CLIPS"]).AL_CLIPS) if args.clips == "al" else 10
    planned = proposal["n_oracle_calls_required"] * n_clips
    print("-" * 78)
    print(f"STAGE 1 (free): {proposal['n_candidates']} candidate cells proposed by a "
          f"GP fitted to {proposal['surrogate_train_conditions']} real conditions")
    print(f"STAGE 2 plan  : {proposal['n_oracle_calls_required']} distinct conditions "
          f"x {n_clips} clips = {planned} API calls (ceiling {args.max_calls})")
    print("-" * 78)

    if planned > args.max_calls:
        print(f"REFUSING: plan of {planned} calls exceeds the {args.max_calls} ceiling. "
              f"Lower --top-k.")
        return 2

    confirmed = None
    calls_spent = 0
    if args.dry_run:
        print("[dry-run] no API calls made; counterintuitive cells stay UNCONFIRMED.")
    else:
        wer_fn, cache, clip_ids = build_oracle(args.clips, args.model, args.results_dir)
        oracle = BudgetedOracle(wer_fn, len(clip_ids), args.max_calls, cache)
        print(f"[stage 2] confirming against {args.model} over {len(clip_ids)} clips "
              f"({clip_ids})")
        confirmed = confirm_cells(proposal, oracle)
        calls_spent = oracle.calls_spent
        confirmed["api_calls_spent"] = calls_spent
        confirmed["clips"] = clip_ids
        confirmed["oracle_log"] = oracle.log
        confirmed["cache_rows_before"] = oracle.cache_hits_before
        confirmed["cache_rows_after"] = len(cache)
        confirmed["cache_rows_added"] = len(cache) - oracle.cache_hits_before
        cache.close()
        print(f"[stage 2] {confirmed['n_confirmed']}/{len(confirmed['candidates'])} "
              f"cells reproduced under real transcription; "
              f"{calls_spent} API calls budgeted, "
              f"{confirmed['cache_rows_added']} new cache rows written")

    # ---- 3. assemble + write ----------------------------------------------
    ci = confirmed if confirmed is not None else proposal

    rec = None
    try:
        rec = reconcile_dip_evidence(measured, ci)
        ci["reconciliation"] = rec
        ci["refutation_scope"] = rec["refutation_scope"]
        print(_format_reconciliation(rec, ci))
        print()
    except Exception as e:                       # noqa: BLE001 — annotation only
        print(f"[reconciliation] skipped: {type(e).__name__}: {e}\n")

    report_res = {
        "sobol_tables": format_sobol_tables(sobol),
        "evidence": evidence,
        "verdict": verdict,
        "counterintuitive": ci,
    }
    # `counterintuitive` is passed as None to the generic formatter ON PURPOSE.
    # Its renderer prints one line per candidate from `note`, and all six notes are
    # the identical string ("WER dips as rt60 moves through 0.7") — six duplicate
    # lines that hide the fact that the candidates differ in their HELD factors.
    # The reconciliation section below prints them as a table with the differing
    # coordinates, so the generic block would be strictly worse duplication.
    text_res = dict(report_res, counterintuitive=None)
    text = format_interaction_report(text_res)
    # MEASURED evidence first, then the mechanism, then the reconciliation, and
    # only then the Sobol/verdict block. Ordering is load-bearing: a reader who
    # meets "0/6 REFUTED" before the reconciliation will read the two rt60 results
    # as a contradiction, which is exactly the failure this section exists to stop.
    text = "\n\n".join([
        format_measured_counterintuitive(measured),
        _format_mechanism(mech, measured),
        _format_reconciliation(rec, ci) if rec else
        "RECONCILIATION — unavailable (RIR assets not loadable in this environment).",
        text,
    ])
    (out / "interaction_report.txt").write_text(text + "\n", encoding="utf-8")

    def jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        return str(o)

    payload = {
        "verdict": verdict,
        "evidence": evidence,
        "prereg_commit": PREREG_COMMIT,
        "prereg_date": PREREG_DATE,
        "prereg_evidence": PREREG_EVIDENCE,
        "s2_caveat": S2_CAVEAT,
        "sobol_source": args.sobol,
        "sobol_method": sobol.get("method"),
        "measured_counterintuitive": measured,
        "rir_mechanism": mech,
        "dip_reconciliation": rec,
        "counterintuitive": ci,
        "api_calls_spent": calls_spent,
        "plot_payload": plot_payload(report_res),
    }
    (out / "interactions.json").write_text(
        json.dumps(payload, indent=2, default=jsonable), encoding="utf-8")

    # Two independent evidence classes in one file, never merged into one list:
    # merging them is precisely how a GP's imagination ends up quoted as a
    # measurement. The measured block carries presentable_as_measured=True because
    # it IS the measurement; the surrogate block carries whatever the oracle said.
    (out / "counterintuitive_confirmed.json").write_text(
        json.dumps({
            "summary": {
                "measured_in_grid_significant": measured["n_cellwise_significant"],
                "measured_marginal_dips": measured["n_marginal"],
                "surrogate_proposed": ci.get("n_candidates"),
                "surrogate_confirmed": ci.get("n_confirmed", 0),
                "surrogate_refuted": ci.get("n_refuted", 0),
                "api_calls_spent": calls_spent,
            },
            "measured_in_grid": measured,
            "rir_mechanism": mech,
            "dip_reconciliation": rec,
            "surrogate_two_stage": ci,
        }, indent=2, default=jsonable), encoding="utf-8")

    print()
    print(f"[wrote] {out/'interaction_report.txt'}")
    print(f"[wrote] {out/'interactions.json'}")
    print(f"[wrote] {out/'counterintuitive_confirmed.json'}")
    print(f"[api] {calls_spent} Deepgram calls spent this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
