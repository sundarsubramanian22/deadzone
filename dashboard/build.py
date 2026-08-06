#!/usr/bin/env python3
"""
dashboard/build.py — SPEC A.R6. Emit ONE self-contained static HTML file.

    python3 dashboard/build.py                 # auto: real master.csv if present
    python3 dashboard/build.py --master results/synthetic_master.csv
    python3 dashboard/build.py --no-al         # skip the (slow) AL loop

STACK, decided and not relitigated (A.R6.1): a single HTML file with the results
inlined as JSON and vanilla JS. It opens from file:// with wifi off — no server,
no port, no pip install on someone else's laptop. Nothing in the emitted page
references an external host.

TWO PROPERTIES THIS FILE EXISTS TO GUARANTEE
--------------------------------------------
1. ONE DATA DOOR. Everything the page shows is assembled here into a single JSON
   blob and read in the browser through exactly one function, `loadData()`.
   Swapping synthetic results for real ones is `--master <path>` and nothing else.

2. NOTHING IS ALLOWED TO BLOCK THE BUILD. The analysis layers are separate
   modules with separate lifecycles; some may not exist yet, some may raise on a
   table shape they have not met. Every panel is therefore built inside
   `_panel()`, which converts ANY import error, missing file or exception into a
   payload of {"status": "missing", "reason": "<what went wrong>"} that the page
   renders as a labelled empty state. A dashboard that refuses to build because
   one leg is late is worse than useless the morning of a demo.

The corollary of (2): a panel that is empty must SAY WHY. There is no code path
here that emits an empty chart with no explanation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEFAULT_REAL = os.path.join(REPO, "results/master.csv")
DEFAULT_SYNTH = os.path.join(REPO, "results/synthetic_master.csv")
DEFAULT_OUT = os.path.join(HERE, "deadzone.html")

# How the sim arm is identified inside one master table. The runner records the
# RIR it actually used; the synthetic RIRs live in their own directory.
SIM_MARKERS = ("rirs_sim", "/sim_", "sim_rt60")


# ===========================================================================
# JSON hygiene — NaN and numpy do not survive a browser
# ===========================================================================

def jsonable(obj):
    """
    Recursively make `obj` JSON-safe. NaN / inf become null, NOT 0.0 — a missing
    measurement that renders as zero is the single most dangerous thing a
    dashboard can do (analysis/__init__ makes the same choice for the same
    reason: the page must be able to say "n/a", never "perfect").
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, int):
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    # numpy et al — duck-typed so numpy stays an optional import here
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        try:
            return jsonable(obj.item())
        except Exception:
            return str(obj)
    if hasattr(obj, "tolist"):
        try:
            return jsonable(obj.tolist())
        except Exception:
            return str(obj)
    return str(obj)


def _ok(data: dict) -> dict:
    return {"status": "ok", "reason": None, "data": data}


def _missing(reason: str) -> dict:
    return {"status": "missing", "reason": reason, "data": None}


def _panel(name: str, fn, notes: list) -> dict:
    """Run one panel builder. ANY failure becomes an explained empty state."""
    try:
        out = fn()
        if out is None:
            return _missing(f"{name}: the builder returned no payload.")
        return out
    except ImportError as e:
        msg = (f"{name}: an analysis module is not importable yet ({e}). "
               f"This panel fills in as soon as that module lands; rebuild then.")
        notes.append(msg)
        return _missing(msg)
    except FileNotFoundError as e:
        msg = f"{name}: a required results file is absent ({e})."
        notes.append(msg)
        return _missing(msg)
    except Exception as e:
        msg = (f"{name}: {type(e).__name__}: {e}\n"
               f"(built {datetime.now(timezone.utc).isoformat(timespec='seconds')}; "
               f"the other panels are unaffected)")
        notes.append(f"{name}: {type(e).__name__}: {e}")
        if os.environ.get("DEADZONE_DEBUG"):
            traceback.print_exc()
        return _missing(msg)


# ===========================================================================
# THE ONE DATA DOOR
# ===========================================================================

def load_rows(master_path: str) -> list[dict]:
    """Everything on the page comes from here. One table, one call."""
    from deadzone.analysis import load_master_table
    return load_master_table(master_path)


def is_sim_row(row: dict) -> bool:
    key = str(row.get("rir_key") or "")
    name = str(row.get("condition_name") or "")
    return any(m in key for m in SIM_MARKERS) or name.endswith("_sim")


def split_real_sim(rows):
    real = [r for r in rows if not is_sim_row(r)]
    sim = [r for r in rows if is_sim_row(r)]
    return real, sim


# ===========================================================================
# PANEL 1 (+2, derived) — the silent-failure map
# ===========================================================================

MAX_EDITS_IN_DIFF = 60


def _example_for_condition(rows, cond_name, model):
    """
    A representative clip row for the hover diff: the one whose WER is closest to
    the cell's mean. Not the worst clip — the worst is a cherry-pick, and the
    hover is supposed to show what that condition typically does to a transcript.
    """
    grp = [r for r in rows
           if str(r.get("condition_name")) == cond_name
           and (model is None or r.get("model") == model)
           and not r.get("failed")
           and r.get("transcript")]
    if not grp:
        return None
    wers = [r.get("wer") for r in grp if isinstance(r.get("wer"), float)
            and math.isfinite(r.get("wer"))]
    if not wers:
        return None
    mean = sum(wers) / len(wers)
    best = min(grp, key=lambda r: abs((r.get("wer") if isinstance(r.get("wer"), float)
                                       and math.isfinite(r.get("wer")) else 1e9) - mean))
    edits = best.get("edits")
    if isinstance(edits, str):
        try:
            edits = json.loads(edits)
        except Exception:
            edits = []
    edits = edits or []
    return {
        "clip_id": best.get("clip_id"),
        "wer": best.get("wer"),
        "transcript": best.get("transcript"),
        "edits": [list(e)[:3] for e in edits[:MAX_EDITS_IN_DIFF]],
        "truncated": len(edits) > MAX_EDITS_IN_DIFF,
    }


def build_silent_failure(rows_real, model):
    import deadzone.analysis.confidence_gap as cg
    report = cg.confidence_gap_report(rows_real, model=model)
    if not report.get("per_condition"):
        return _missing(
            f"no usable conditions for model {model!r}: every row either failed or "
            f"carried no confidence. The headline finding needs per-word confidence "
            f"— verify the adapter returns it (SPEC §12, day-one check).")
    payload = cg.plot_payload(report)
    for p in payload["points"]:
        p["example"] = _example_for_condition(rows_real, p["condition_name"], model)
    payload["correlation"] = report.get("correlation") or {}
    payload["gap_summary"] = report.get("gap_summary") or {}
    payload["n_clip_rows_used"] = report.get("n_clip_rows_used")
    return _ok(payload)


# ===========================================================================
# PANEL 3 — fingerprints
# ===========================================================================

def _family_rows(rows, family: str):
    """
    The degraded side of a family, reproducing analysis.fingerprints' own contrast
    (continuous -> degraded tercile, categorical -> that level) so the entity rate
    we attach to a family describes the SAME rows its signature was computed on.
    """
    from deadzone.design import DEFAULT_FACTOR_SPACE as SPACE
    if "=" in family:
        fname, level = family.split("=", 1)
        return [r for r in rows if str(r.get(fname)) == level]
    f = next((x for x in SPACE.factors if x.name == family), None)
    if f is None:
        return []
    vals = sorted(v for v in (r.get(family) for r in rows)
                  if isinstance(v, (int, float)) and math.isfinite(v))
    if not vals:
        return []
    lo_t = vals[max(0, int(len(vals) / 3) - 1)]
    hi_t = vals[min(len(vals) - 1, int(2 * len(vals) / 3))]
    if f.degradation == "down":
        return [r for r in rows if isinstance(r.get(family), (int, float))
                and r.get(family) <= lo_t]
    return [r for r in rows if isinstance(r.get(family), (int, float))
            and r.get(family) >= hi_t]


def build_fingerprints(rows_real, model):
    import deadzone.analysis.fingerprints as fp
    report = fp.fingerprint_report(
        rows_real, model=model,
        manifest_path=os.path.join(REPO, "recording_manifest.csv"),
        task_specs_path=os.path.join(REPO, "task_specs.json"))
    sigs = report.get("signatures") or []
    if not sigs:
        return _missing(
            f"no factor family had enough rows on both sides of its contrast for "
            f"model {model!r} — fingerprints need at least a few conditions per level.")

    # entity error rate per condition, so it can be averaged onto a family
    ent = report.get("entities") or {}
    ent_by_cond = {d["condition_name"]: d.get("entity_error_rate")
                   for d in (ent.get("by_condition") or [])}
    model_rows = [r for r in rows_real if model is None or r.get("model") == model]

    families = []
    for s in sigs:
        rd = s.get("rates_degraded") or {}
        sub_rows = _family_rows(model_rows, s["family"])
        conds = {str(r.get("condition_name")) for r in sub_rows}
        vals = [ent_by_cond[c] for c in conds
                if c in ent_by_cond and isinstance(ent_by_cond[c], float)
                and math.isfinite(ent_by_cond[c])]
        families.append({
            "family": s["family"],
            "factor": s.get("factor"),
            "contrast": s.get("contrast"),
            "sub_rate": rd.get("sub"), "del_rate": rd.get("del"), "ins_rate": rd.get("ins"),
            "wer": s.get("wer_degraded"),
            "dominant_edit": s.get("dominant_edit"),
            "degrades": s.get("degrades"),
            "effect_size_d": s.get("effect_size_d"),
            "implied_fix": s.get("implied_fix"),
            "caveat": s.get("caveat"),
            "n_conditions": len(conds),
            "n_ref": sum(int(r.get("n_ref") or 0) for r in sub_rows),
            "entity_error_rate": (sum(vals) / len(vals)) if vals else None,
        })

    overall = (ent.get("overall") or {})
    if ent.get("available"):
        note = ("Entity error rate is scored against task_specs.json slots. Mean entity ER "
                f"{overall.get('mean_entity_error_rate'):.2f} vs mean WER "
                f"{overall.get('mean_wer'):.2f} — entities degrade "
                + ("FASTER" if overall.get("entities_degrade_faster") else "NO faster")
                + " than the transcript as a whole.") if overall.get("mean_wer") is not None else None
    else:
        note = f"Entity error rate unavailable: {ent.get('reason')}"

    return _ok({
        "families": families,
        "signature_notes": [s["signature"] for s in sigs[:8]],
        "entity_note": note,
        "n_clip_rows_used": report.get("n_clip_rows_used"),
    })


# ===========================================================================
# PANEL 4 — sensitivity + the pre-registration verdict
# ===========================================================================

def build_sensitivity(rows_real, model, sobol_n: int, sobol_path: str):
    import deadzone.analysis.interactions as itx
    provenance = None

    if os.path.exists(sobol_path):
        sobol = itx.load_sobol_json(sobol_path)
        provenance = (f"Sobol indices loaded from {os.path.relpath(sobol_path, REPO)} "
                      f"(computed by the experiment runner).")
    else:
        # No saved Sobol run: fit the GP surrogate to the measured grid and take
        # the indices off THAT. Cheap and honest, as long as the page says so —
        # which is what `provenance` is for. It is NOT a measured Sobol run.
        from deadzone.design import DEFAULT_FACTOR_SPACE as SPACE, sobol_indices
        gp, mat = itx.fit_surrogate_from_master(rows_real, model=model)
        wer_fn = itx.surrogate_wer_fn(gp, SPACE, grid=7)
        sobol = sobol_indices(SPACE, wer_fn, N=sobol_n, seed=0)
        provenance = (
            f"No results/sobol.json in this build, so the indices come from a GP "
            f"surrogate fitted to the {mat['n_conditions']} measured conditions "
            f"(N={sobol_n}, {sobol['n_eval']} surrogate evaluations) — a surrogate "
            f"reading of the measured surface, NOT a measured Sobol run. Re-run the "
            f"real Sobol pass before quoting these numbers.")

    report = itx.interaction_report(sobol, rows_real, model=model)
    payload = itx.plot_payload(report)
    payload["provenance"] = provenance
    return _ok(payload)


# ===========================================================================
# PANEL 5 — the active-learning loop (steppable) + the savings curve
# ===========================================================================

AL_SLICE = {"noise_type": "babble", "codec": "none", "mic_rolloff": 0.0}
AL_NX, AL_NY = 8, 6


def _posterior_frames(oracle, space, seed=0, n_seed=10, budget=16, max_frames=14):
    """
    Re-fit the GP at each acquisition and record its posterior over the
    rt60 x SNR slice, so the panel can be STEPPED: the point of the panel is
    watching the acquisition walk onto the failure boundary, which a final-state
    plot cannot show.
    """
    import numpy as np
    from deadzone.active_learning import GPSurrogate, active_learn
    from deadzone.analysis.interactions import encode_sample

    traj = active_learn(oracle, space, strategy="boundary", n_seed=n_seed,
                        budget=budget, pool_size=300, seed=seed)

    rt = next(f for f in space.factors if f.name == "rt60")
    snr = next(f for f in space.factors if f.name == "snr_db")
    xs = [rt.low + i * (rt.high - rt.low) / (AL_NX - 1) for i in range(AL_NX)]
    ys = [snr.high - j * (snr.high - snr.low) / (AL_NY - 1) for j in range(AL_NY)]
    grid = np.asarray([encode_sample(space, dict(AL_SLICE, rt60=x, snr_db=y))
                       for y in ys for x in xs], dtype=float)

    samples = traj.samples(space)
    steps = [s["n_evals"] for s in traj.steps]
    keep = set(steps[:: max(1, len(steps) // max_frames)] or steps)
    keep.add(n_seed)
    if steps:
        keep.add(steps[-1])

    frames = []
    for n in sorted(keep):
        gp = GPSurrogate(space, seed=seed).fit(traj.X_raw[:n], traj.y[:n])
        mu = np.asarray(gp.predict(grid, return_std=False), dtype=float).ravel()
        pts = [{"rt60": s["rt60"], "snr_db": s["snr_db"], "wer": float(traj.y[i]),
                "new": (i == n - 1 and n > n_seed)}
               for i, s in enumerate(samples[:n])]
        st = next((s for s in traj.steps if s["n_evals"] == n), None)
        frames.append({
            "step": max(0, n - n_seed), "n_evals": int(n), "nx": AL_NX, "ny": AL_NY,
            "mu": [float(v) for v in mu], "points": pts,
            "chosen_wer": (st or {}).get("oracle_wer"),
        })
    return frames, [rt.low, rt.high], [snr.low, snr.high]


def build_active_learning(rows_real, model, seeds=(0, 1, 2), enabled=True):
    if not enabled:
        return _missing("The active-learning loop was skipped for this build "
                        "(`--no-al`). Re-run `python3 dashboard/build.py` without "
                        "it to populate this panel; it takes ~15 s.")
    import deadzone.analysis.al_savings as als
    from deadzone.design import DEFAULT_FACTOR_SPACE as SPACE
    from deadzone.active_learning import DEFAULT_THRESHOLD

    split = als.test_set_from_master(rows_real, SPACE, model=model)
    oracle, meta = als.surrogate_oracle_from_master(split=split, model=model)

    frames, xdom, ydom = _posterior_frames(oracle, SPACE)

    curves = None
    try:
        multi = als.multi_seed_curves(oracle, SPACE, X_test=split["X_test"],
                                      y_test=split["y_test"], seeds=list(seeds),
                                      n_seed=10, budget=20, pool_size=300)
        curves = als.plot_payload(als.al_savings_report(multi))
    except Exception as e:                      # the frames alone still teach it
        curves = None
        frames_note = f"active-vs-random curves unavailable: {type(e).__name__}: {e}"
        frames[-1]["note"] = frames_note if frames else None

    return _ok({
        "frames": frames, "curves": curves,
        "x_domain": xdom, "y_domain": ydom,
        "threshold": DEFAULT_THRESHOLD,
        "slice": AL_SLICE,
        "oracle": meta,
    })


# ===========================================================================
# PANEL 6 — sim vs real
# ===========================================================================

def build_sim2real(rows_real, rows_sim, model):
    if not rows_sim:
        return _missing(
            "This table holds no simulated-RIR rows, so there is no sim-vs-real "
            "comparison to make. The sim arm is identified by its rir_key "
            "(data/rirs_sim/...); run the grid against the synthetic RIRs "
            "(make_sim_rirs.py) and rebuild.")
    if not any(r.get("model") == model for r in rows_sim):
        have = sorted({str(r.get("model")) for r in rows_sim})
        return _missing(
            f"The simulated-RIR arm was only run for {', '.join(have)}, not for "
            f"'{model}'. The sim2real comparison is per-model by construction — the "
            f"gap is a property of one model's response to one room — so there is "
            f"nothing to show here until '{model}' is run against data/rirs_sim.")
    import deadzone.analysis.sim2real as s2r
    res = s2r.sim2real_report(rows_real, rows_sim, model=model)
    payload = s2r.plot_payload(res)
    head = res.get("headline") or {}
    payload["verdict_sentence"] = head.get("advice") or head.get("sentence")
    if not payload["scatter"]:
        return _missing(
            "No measured/simulated pair matched within the RT60 tolerance, so the "
            "gap cannot be computed. Pairing is on MEASURED RT60 of both files — "
            "check that rir_rt60_measured is populated on both arms.")
    return _ok(payload)


# ===========================================================================
# PANEL 7 — L1 multi-model comparison   (cross-cutting: not per-model)
# ===========================================================================

def build_model_arms(path: str) -> dict:
    """
    Read the L1 comparison written by `analysis/model_arms.py`.

    Read rather than recomputed, unlike panels 1-6. Re-scoring every row through
    the cross-model normalizer costs seconds and pulls in openai-whisper purely
    for its text normalizer — a heavy import to put on the critical path of a
    dashboard build that must not fail. The analysis owns the numbers; the
    dashboard displays them.

    NOTE FOR THE READER OF THE PAGE, enforced by the payload below: the two arms'
    confidences are NOT on a common scale (Deepgram returns acoustic confidence,
    Whisper the decoder's token probability), so this panel never draws them on a
    shared axis. It shows each model's dead-zone rate, its own confidence-vs-WER
    shape, and where the two disagree.
    """
    if not os.path.isfile(path):
        return _missing(
            "results/model_arms.json is absent, so the multi-model comparison has "
            "not been run. Produce it with `python3 -m deadzone.analysis.model_arms` once "
            "both arms are in the master table.")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)

    per = res.get("per_model") or {}
    if len(per) < 2:
        return _missing(
            f"Only {len(per)} model arm is present, so there is nothing to compare. "
            f"Run the grid for a second model (`--models whisper-base`) and re-run "
            f"the analysis.")

    ov = res.get("dead_zone_overlap") or {}
    hall = res.get("whisper_hallucination") or {}
    return _ok({
        "models": [
            {"model": m,
             "n_conditions": d.get("n_conditions"),
             "wer_strict": d.get("wer_mean_strict"),
             "wer_crossmodel": d.get("wer_mean_crossmodel"),
             "dead_zone_rate": d.get("dead_zone_rate"),
             "n_dead_zones": d.get("n_dead_zones"),
             "shape": d.get("shape"),
             "edits": d.get("edit_signature_crossmodel")}
            for m, d in per.items()
        ],
        "normalization_shift": res.get("normalization_shift"),
        "overlap": {"jaccard": ov.get("jaccard"),
                    "shared": ov.get("shared", []),
                    "only": {k: v for k, v in ov.items() if k.endswith("_only")}},
        "divergence_regions": (res.get("divergence_regions") or [])[:8],
        "hallucination": {
            "median_len_ratio": hall.get("median_len_ratio"),
            "p95_len_ratio": hall.get("p95_len_ratio"),
            "frac_rows_over_2x": hall.get("frac_rows_over_2x"),
            "examples": (hall.get("examples") or [])[:3],
        },
        "caption": ("Confidence is comparable WITHIN a model, never across two. "
                    "Each arm is ranked against its own confidence distribution; "
                    "absolute WER is shown both under the strict scoring and under "
                    "a symmetric cross-model normalization, because the two models "
                    "disagree about orthography as well as about acoustics."),
    })


# ===========================================================================
# PANEL 8 — L3 paralinguistic vs lexical decoupling   (cross-cutting)
# ===========================================================================

def build_decoupling(path: str) -> dict:
    """
    Read the L3 decoupling result. This is the one layer whose input is AUDIO
    rather than the results table, so it cannot be recomputed from `rows` here
    at all — the sweep wavs and their transcriptions are its source.
    """
    if not os.path.isfile(path):
        return _missing(
            "results/l3_decoupling.json is absent. This layer reads the sweep "
            "audio (results/audio/sweep/, from scripts/make_audio_sets.py), not the "
            "master table, so it must be run separately: "
            "`python3 -m deadzone.analysis.decoupling`.")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)

    factors = res.get("factors") or res.get("per_factor") or {}
    if not factors:
        return _missing(
            "The decoupling result contained no per-factor curves. Expected one "
            "entry per swept factor (rt60, snr_db).")

    out = []
    for name, f in factors.items():
        head = f.get("headline") or {}
        deg = f.get("lexical_degeneracy") or {}
        # A half-degradation level is only meaningful if the curve it was read
        # off actually travelled. The analysis flags the degenerate case; the
        # page must not quietly render a number the analysis refused to quote.
        quotable = not bool(deg.get("degenerate"))
        out.append({
            "factor": name,
            "levels": f.get("levels"),
            "severity_order": f.get("severity_order"),
            "wer_mean": f.get("wer_curve_mean"),
            "wer_std": f.get("wer_curve_std"),
            "wer_per_clip": f.get("wer_per_clip"),
            "primary_feature": f.get("primary_feature"),
            "feature_curve": (head.get("feature_curve")
                              or head.get("feature_drift_mean")),
            "per_feature": f.get("per_feature"),
            "verdict": f.get("verdict"),
            "statement": f.get("statement"),
            "leads": f.get("factor_leads"),
            "spearman": head.get("spearman"),
            "max_abs_gap": head.get("max_abs_gap"),
            "feature_half_level": head.get("feature_half_level"),
            "lexical_half_level": head.get("lexical_half_level"),
            "half_levels_quotable": quotable,
            "degeneracy": deg,
            "n_features_trend_reliable": f.get("n_features_trend_reliable"),
            "n_features": f.get("n_features"),
            "n_clips": f.get("n_clips"),
            "notes": f.get("notes"),
        })

    return _ok({
        "factors": out,
        "feature_keys": res.get("feature_keys"),
        "caption": ("Do paralinguistic features and lexical accuracy degrade at the "
                    "same rate, or decouple? If lexical accuracy leads, an agent "
                    "watching prosody would not notice its ASR had already failed. "
                    "Coupling is an equally valid result and is reported as one — "
                    "and where a curve never leaves its floor, no threshold is "
                    "quoted at all, because a min-max-normalized crossing point on "
                    "a flat curve is arithmetic without meaning."),
    })


# ===========================================================================
# ASSEMBLY
# ===========================================================================

def collect(master_path: str, with_al: bool = True, sobol_n: int = 1024,
            sim_master: str | None = None) -> dict:
    notes: list[str] = []
    rows = load_rows(master_path)

    # The sim arm is a separate table by necessity, not by preference: the run
    # cache is keyed on (clip_id, condition_name, model) and does not encode
    # which RIR library produced the row, so running the simulated arm against
    # the real results dir would be 100% cache hits and would report a sim2real
    # gap of exactly zero -- a wrong answer with no error message. Here the two
    # are concatenated and re-split by rir_key, which is the field that actually
    # records provenance.
    if sim_master and os.path.exists(sim_master):
        sim_rows = load_rows(sim_master)
        rows = rows + sim_rows
        notes.append(f"{len(sim_rows)} simulated-RIR rows joined from "
                     f"{os.path.relpath(sim_master, REPO)} for the sim-vs-real panel.")

    real, sim = split_real_sim(rows)

    from deadzone.analysis import split_by_model
    models = sorted(split_by_model(real).keys())
    if not models:
        notes.append("The master table contained no rows with a model column.")

    sobol_path = os.path.join(REPO, "results/sobol.json")
    per_model: dict[str, dict] = {}
    for m in models:
        per_model[m] = {
            "silent_failure": _panel("silent-failure map",
                                     lambda m=m: build_silent_failure(real, m), notes),
            "fingerprints": _panel("fingerprints",
                                   lambda m=m: build_fingerprints(real, m), notes),
            "sensitivity": _panel("sensitivity",
                                  lambda m=m: build_sensitivity(real, m, sobol_n, sobol_path), notes),
            "active_learning": _panel("active learning",
                                      lambda m=m: build_active_learning(real, m, enabled=with_al), notes),
            "sim2real": _panel("sim2real",
                               lambda m=m: build_sim2real(real, sim, m), notes),
        }

    # Panels 7 and 8 are cross-cutting: L1 compares the model arms against each
    # other and L3 reads audio rather than the table, so neither belongs under a
    # per-model key. They are built once and rendered outside the model toggle.
    cross = {
        "model_arms": _panel("multi-model comparison",
                             lambda: build_model_arms(
                                 os.path.join(REPO, "results/model_arms.json")), notes),
        "decoupling": _panel("paralinguistic decoupling",
                             lambda: build_decoupling(
                                 os.path.join(REPO, "results/l3_decoupling.json")), notes),
    }

    is_synth = "synthetic" in os.path.basename(master_path)
    if is_synth:
        notes.insert(0, "SYNTHETIC DATA: this build reads results/synthetic_master.csv, a "
                        "planted-structure table. Every number on this page is a demonstration "
                        "that the instrument reads its own plant back, not a measurement of any "
                        "real ASR model.")
    if sim:
        notes.append(f"{len(sim)} rows came from simulated RIRs and are excluded from panels 1-5; "
                     f"they are used only for the sim-vs-real comparison in panel 6.")

    return {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": os.path.relpath(master_path, REPO),
            "is_synthetic": is_synth,
            "n_rows": len(rows),
            "n_rows_real": len(real),
            "n_rows_sim": len(sim),
            "n_conditions": len({str(r.get("condition_name")) for r in real}),
            "models": models,
            "notes": notes,
        },
        "models": per_model,
        "cross": cross,
        "default_model": models[0] if models else None,
    }


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def emit_html(payload: dict, out_path: str,
              shell: str | None = None, css: str | None = None,
              js: str | None = None) -> str:
    """Inline CSS, data and JS into the shell. No external references, ever."""
    html = _read(shell or os.path.join(HERE, "shell.html"))
    style = _read(css or os.path.join(HERE, "app.css"))
    script = _read(js or os.path.join(HERE, "app.js"))

    blob = json.dumps(jsonable(payload), separators=(",", ":"), allow_nan=False)
    # A literal "</script>" inside a JS string ends the tag; escaping the slash
    # is the standard fix and is invisible to JSON.parse.
    blob = blob.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    data_js = "window.__DEADZONE_DATA__ = " + blob + ";"

    html = html.replace("/*__CSS__*/", style)
    html = html.replace("/*__DATA__*/", data_js)
    html = html.replace("/*__JS__*/", script)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


def default_master() -> str:
    return DEFAULT_REAL if os.path.exists(DEFAULT_REAL) else DEFAULT_SYNTH


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the self-contained Deadzone dashboard")
    ap.add_argument("--master", default=None,
                    help="master results table (default: results/master.csv if it "
                         "exists, else results/synthetic_master.csv)")
    ap.add_argument("--sim-master", default=None,
                    help="second table holding the simulated-RIR arm, appended to "
                         "--master before the split (default: results_sim/master_sim.csv "
                         "if present). The sim arm lives in its own results dir because "
                         "the run cache is keyed on (clip, condition, model) and does NOT "
                         "encode which RIR library produced the row -- one shared cache "
                         "would be 100%% false hits and would report a sim2real gap of "
                         "exactly zero.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-al", action="store_true", help="skip the active-learning loop (~15 s)")
    ap.add_argument("--sobol-n", type=int, default=1024)
    args = ap.parse_args(argv)

    master = args.master or default_master()
    if not os.path.exists(master):
        print(f"no master table at {master}; generating the synthetic one first",
              file=sys.stderr)
        from dashboard.make_synthetic import build_rows, write_csv  # noqa: WPS433
        write_csv(build_rows(), DEFAULT_SYNTH)
        master = DEFAULT_SYNTH

    sim_master = args.sim_master
    if sim_master is None:
        cand = os.path.join(REPO, "results_sim/master_sim.csv")
        sim_master = cand if os.path.exists(cand) else None

    payload = collect(master, with_al=not args.no_al, sobol_n=args.sobol_n,
                      sim_master=sim_master)
    out = emit_html(payload, args.out)

    size = os.path.getsize(out)
    print(f"built {out} ({size/1024:.0f} KB) from {payload['meta']['source']}")
    print(f"  {payload['meta']['n_rows']} rows · {payload['meta']['n_conditions']} conditions "
          f"· models: {', '.join(payload['meta']['models']) or 'none'}")
    for m, panels in payload["models"].items():
        state = ", ".join(f"{k}={'ok' if v['status'] == 'ok' else 'EMPTY'}"
                          for k, v in panels.items())
        print(f"  [{m}] {state}")
    cross_state = ", ".join(f"{k}={'ok' if v['status'] == 'ok' else 'EMPTY'}"
                            for k, v in payload.get("cross", {}).items())
    if cross_state:
        print(f"  [cross-model] {cross_state}")
    for n in payload["meta"]["notes"]:
        print(f"  note: {n.splitlines()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
