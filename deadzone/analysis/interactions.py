"""
analysis/interactions.py — D3a: the interaction hunt + the PRE-REGISTRATION
verdict (SPEC §5 third leg, A.R5.4).

design.py owns the *machinery* (Saltelli sampling, Sobol S1/ST/S2, the dense
counterintuitive scan). This file owns the *reading* of it against real data:
it turns a `design.sobol_indices` result plus the master results table into the
two things the write-up needs and nothing else can produce.

  1. THE PRE-REGISTERED VERDICT. SPEC §5 registered `rt60 × snr_db` as a genuine
     compounding two-way interaction *before any real audio existed*. That is the
     whole point of pre-registering: the answer becomes a result either way. This
     module emits ONE unambiguous sentence with real numbers and a
     confirmed / not-confirmed call, under a decision rule that was also fixed in
     advance (`PreRegistration.rule`) so the threshold can't drift to fit the data.

  2. CANDIDATE COUNTERINTUITIVE CELLS, EXPLICITLY UNCONFIRMED. The "huh" cells
     (WER *improves* with more degradation; an effect that flips sign) are the
     one-well-explained-surprise the write-up wants — but only if they're real.

THE TWO TRAPS THIS FILE EXISTS TO AVOID
---------------------------------------
TRAP 1 — LEADING WITH S2. SPEC §5's Track-C note is explicit: even with a *strong
planted* interaction at N=1024 the second-order S2 bootstrap CI crossed zero. S2
is a difference of noisy variance estimates and it is simply not precise enough at
affordable N to be quoted as a magnitude. So the hierarchy here is structural, not
stylistic: the ST−S1 gap is the PRIMARY evidence (it is a within-factor comparison
of two much better-estimated quantities), and S2 is demoted to answering only
*which* pair — a rank and a sign, never a size. The returned dict names the S2
entry `secondary_which_pair_only` and stamps it `magnitude_usable=False`; the
printed report prints `S2_CAVEAT` next to it every single time. If you find
yourself writing "the interaction explains S2 = 0.12 of the variance", the API is
telling you not to.

TRAP 2 — PRESENTING A SURROGATE PREDICTION AS A MEASUREMENT.
`design.find_counterintuitive_cells` sweeps a DENSE grid (grid^5 cells) over
`wer_fn`. Against the real API oracle that is thousands of calls per scan, i.e.
ruinous — so we sweep a GP surrogate fitted to the master table instead. A cell
found that way is a *prediction of a model fitted to ~60 conditions*, and a GP
will happily interpolate a dip that no microphone ever heard. The two-stage
protocol (SPEC A.R5.4) is therefore enforced in the data structure, not left to
the author's memory: every proposal comes back with `confirmed=False`,
`evidence="surrogate-proposed (NOT measured)"`, and the exact list of conditions
to re-evaluate; `confirm_cells()` is the only thing that can flip
`confirmed=True`, and it can only do so by consuming real oracle calls. The
formatter prints proposals under an UNCONFIRMED banner and refuses to drop it.

An unconfirmed pre-registration is a RESULT, not a failure (SPEC A.R5.4 gotcha):
the not-confirmed rendering states what was learned ("to the resolution of this
grid the two degradations are additive"), it does not hedge or apologise, and
there is no code path that omits the verdict.

    python3 -m deadzone.analysis.interactions results/sobol.json --master results/master.csv

Deps: numpy, scikit-learn (via active_learning.GPSurrogate).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Callable, Mapping, Sequence

import numpy as np

# Allow `python deadzone/analysis/interactions.py` as well as `import deadzone.analysis.interactions`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.design import (                          # noqa: E402  (after the path shim)
    DEFAULT_FACTOR_SPACE, FactorSpace,
    find_counterintuitive_cells, format_sobol_tables,
)
from deadzone.active_learning import GPSurrogate       # noqa: E402  — REUSED, not reimplemented


# ============================================================================
# THE PRE-REGISTRATION — fixed in SPEC §5 before any real audio existed
# ============================================================================

S2_CAVEAT = (
    "S2 is used for DIRECTION ONLY (which pair), never magnitude: SPEC §5 records "
    "that the second-order bootstrap CI crossed zero even with a strongly planted "
    "interaction at N=1024. Quote the ST-S1 gap for size."
)


@dataclass(frozen=True)
class PreRegistration:
    """
    A hypothesis committed to before the data existed, plus the decision rule
    that resolves it. Both halves matter: a pre-registered *claim* with a
    post-hoc *threshold* is not pre-registered at all, so `min_gap` and the
    direction check live in here and are printed with the verdict.
    """
    factor_a: str = "rt60"
    factor_b: str = "snr_db"
    claim: str = ("reverb and noise COMPOUND — a genuine two-way interaction, "
                  "not two independent main effects")
    source: str = "SPEC §5, Track-C design notes (registered before any real audio)"
    # --- the decision rule, fixed in advance -------------------------------
    min_gap: float = 0.02          # ST-S1 must exceed this AND its CI must clear it
    require_both_factors: bool = True   # both members of the pair must show the gap
    require_top_s2_pair: bool = True    # S2 must agree on WHICH pair (rank 1)

    @property
    def pair(self) -> tuple[str, str]:
        return (self.factor_a, self.factor_b)

    @property
    def rule(self) -> str:
        who = "both factors" if self.require_both_factors else "either factor"
        s2 = (" AND the registered pair ranks first in S2 (direction check only)"
              if self.require_top_s2_pair else "")
        return (f"CONFIRMED iff the ST-S1 interaction gap of {who} in "
                f"{self.factor_a} x {self.factor_b} exceeds {self.min_gap:.3f} with "
                f"its 95% bootstrap CI entirely above it{s2}.")


PREREGISTRATION = PreRegistration()


# ============================================================================
# SOBOL RESULT NORMALIZATION — in-memory result OR results/sobol.json
# ============================================================================

def normalize_sobol_result(res: Mapping) -> dict:
    """
    Accept either a live `design.sobol_indices` return value or the same thing
    round-tripped through JSON (numpy arrays -> nested lists, tuples -> lists),
    and hand back the live shape. Rebuilds `interaction_gap` / `s2_ranked` if the
    serializer dropped them, so `format_sobol_tables` works on both.

    Written because R4.5 persists to `results/sobol.json` and R5.4 reads it back:
    the analysis layer must not care which side of the disk it is on.
    """
    out = dict(res)
    names = list(out["names"])
    for key in ("S1", "ST", "S1_conf", "ST_conf", "S2", "S2_conf"):
        if key in out and out[key] is not None:
            out[key] = np.asarray(out[key], dtype=float)

    if not out.get("interaction_gap"):
        S1, ST = out["S1"], out["ST"]
        S1c, STc = out["S1_conf"], out["ST_conf"]
        out["interaction_gap"] = sorted(
            ({"factor": names[i], "S1": float(S1[i]), "ST": float(ST[i]),
              "gap": float(ST[i] - S1[i]), "S1_conf": float(S1c[i]),
              "ST_conf": float(STc[i])} for i in range(len(names))),
            key=lambda d: d["gap"], reverse=True,
        )
    else:
        out["interaction_gap"] = [dict(d) for d in out["interaction_gap"]]

    if not out.get("s2_ranked"):
        S2, S2c = out.get("S2"), out.get("S2_conf")
        pairs = []
        if S2 is not None:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if np.isnan(S2[i, j]):
                        continue
                    pairs.append({"pair": (names[i], names[j]), "S2": float(S2[i, j]),
                                  "S2_conf": float(S2c[i, j])})
        out["s2_ranked"] = sorted(pairs, key=lambda d: d["S2"], reverse=True)
    else:
        # JSON turns the ("a","b") pair tuple into ["a","b"]; put it back.
        out["s2_ranked"] = [{**d, "pair": tuple(d["pair"])} for d in out["s2_ranked"]]
    return out


def load_sobol_json(path: str) -> dict:
    """Read results/sobol.json (R4.5) into the live `sobol_indices` shape."""
    with open(path, "r", encoding="utf-8") as f:
        return normalize_sobol_result(json.load(f))


def sobol_to_json(res: Mapping) -> dict:
    """The inverse: a JSON-serializable view of a `sobol_indices` result."""
    out = {}
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif k == "s2_ranked":
            out[k] = [{**d, "pair": list(d["pair"])} for d in v]
        else:
            out[k] = v
    return out


# ============================================================================
# PRIMARY EVIDENCE — the ST-S1 interaction gap (this is the headline)
# ============================================================================

def _gap_ci(s1_conf: float, st_conf: float) -> float:
    """
    95% half-width for (ST - S1).

    SALib returns per-index bootstrap half-widths, not the bootstrap replicates,
    so the covariance between S1 and ST is unavailable and we add in quadrature.
    S1 and ST are computed from the SAME Saltelli sample and are positively
    correlated, so quadrature OVER-states the width of the gap's CI. That error is
    in the conservative direction for a pre-registered test — it makes CONFIRMED
    harder to reach, never easier — which is the only direction we're willing to
    be wrong in here.
    """
    return float(math.hypot(float(s1_conf), float(st_conf)))


def interaction_gap_table(sobol: Mapping, min_gap: float = 0.0) -> list[dict]:
    """
    PRIMARY interaction evidence, per factor: ST - S1 with a CI.

    A large gap means the factor moves WER mostly THROUGH interactions rather
    than on its own. Sorted strongest-first — this is the table the write-up
    leads with, and the only place a magnitude may be quoted.
    """
    rows = []
    for d in sobol["interaction_gap"]:
        conf = _gap_ci(d["S1_conf"], d["ST_conf"])
        gap = float(d["gap"])
        rows.append({
            "factor": d["factor"],
            "S1": float(d["S1"]), "S1_conf": float(d["S1_conf"]),
            "ST": float(d["ST"]), "ST_conf": float(d["ST_conf"]),
            "gap": gap, "gap_conf": conf,
            "gap_lo": gap - conf, "gap_hi": gap + conf,
            # "significant" = the whole CI sits above the pre-set threshold
            "significant": bool(gap - conf > min_gap),
        })
    rows.sort(key=lambda r: r["gap"], reverse=True)
    return rows


def s2_direction_only(sobol: Mapping, pair: Sequence[str]) -> dict:
    """
    SECONDARY evidence: where the registered pair sits in the S2 ranking.

    Deliberately returns a RANK, a SIGN and a "does the CI cross zero" flag —
    plus `magnitude_usable=False` — rather than a headline number. See S2_CAVEAT.
    """
    want = frozenset(pair)
    ranked = sobol["s2_ranked"]
    for rank, d in enumerate(ranked, start=1):
        if frozenset(d["pair"]) == want:
            s2, conf = float(d["S2"]), float(d["S2_conf"])
            return {
                "pair": tuple(d["pair"]), "rank": rank, "n_pairs": len(ranked),
                "is_top_pair": rank == 1,
                "S2": s2, "S2_conf": conf,
                "S2_lo": s2 - conf, "S2_hi": s2 + conf,
                "ci_crosses_zero": bool((s2 - conf) <= 0.0 <= (s2 + conf)),
                "sign": "positive" if s2 > 0 else ("negative" if s2 < 0 else "zero"),
                "magnitude_usable": False,          # structural, not advisory
                "usable_for": "which pair interacts — never how much",
                "caveat": S2_CAVEAT,
            }
    return {"pair": tuple(pair), "rank": None, "n_pairs": len(ranked),
            "is_top_pair": False, "S2": float("nan"), "S2_conf": float("nan"),
            "S2_lo": float("nan"), "S2_hi": float("nan"),
            "ci_crosses_zero": True, "sign": "unavailable",
            "magnitude_usable": False,
            "usable_for": "which pair interacts — never how much",
            "caveat": "pair absent from the S2 matrix (second-order pass not run?)"}


def interaction_evidence(sobol: Mapping,
                         prereg: PreRegistration = PREREGISTRATION) -> dict:
    """
    Both legs of the interaction hunt, in the ONLY order they may be read:
    `primary_gap` (magnitude, quotable) then `secondary_which_pair_only`.
    """
    gaps = interaction_gap_table(sobol, min_gap=prereg.min_gap)
    return {
        "primary_gap": gaps,
        "primary_note": ("ST-S1 interaction gap — PRIMARY evidence (SPEC §5). "
                         "Quote sizes from here."),
        "secondary_which_pair_only": sobol["s2_ranked"],
        "secondary_note": S2_CAVEAT,
        "top_gap_factors": [g["factor"] for g in gaps if g["significant"]],
        "top_s2_pair": tuple(sobol["s2_ranked"][0]["pair"]) if sobol["s2_ranked"] else None,
        "N": sobol.get("N"), "n_eval": sobol.get("n_eval"),
    }


# ============================================================================
# THE VERDICT — one unambiguous sentence, either way
# ============================================================================

def preregistration_verdict(sobol: Mapping,
                            prereg: PreRegistration = PREREGISTRATION,
                            commit: str | None = None,
                            date: str | None = None) -> dict:
    """
    Resolve the SPEC §5 pre-registration against the real Sobol result.

    The call is made on the PRIMARY evidence (both factors' ST-S1 gaps clearing
    `min_gap` with their CIs) and only *corroborated* by S2's ranking of which
    pair. There is no code path in which S2's magnitude enters the decision.

    Returns a dict whose `sentence` is publishable verbatim, `confirmed` is a
    plain bool, and `reasons` explains the call in every branch — including the
    not-confirmed one, which is a RESULT and is worded as one.
    """
    sobol = normalize_sobol_result(sobol)
    a, b = prereg.pair
    gaps = {g["factor"]: g for g in interaction_gap_table(sobol, prereg.min_gap)}
    missing = [f for f in (a, b) if f not in gaps]
    s2 = s2_direction_only(sobol, prereg.pair)

    if missing:
        # e.g. the screen dropped a registered factor before Stage 2 ran.
        sentence = (
            f"PRE-REGISTERED ({prereg.source}): {a} x {prereg.factor_b} — "
            f"{prereg.claim}. UNRESOLVED: {', '.join(missing)} is absent from the "
            f"Sobol result, so the registered pair was never tested. This is a "
            f"design gap, not a negative result: re-run the second-order pass on a "
            f"factor space containing both registered factors."
        )
        return {"prereg": asdict(prereg), "pair": (a, b), "status": "UNRESOLVED",
                "confirmed": False, "resolvable": False, "sentence": sentence,
                "primary": {}, "secondary_which_pair_only": s2,
                "gap_criterion_met": False, "pair_criterion_met": False,
                "power_limited": False,
                "reasons": [f"{m} missing from Sobol factor list" for m in missing],
                "rule": prereg.rule, "N": sobol.get("N"), "n_eval": sobol.get("n_eval")}

    ga, gb = gaps[a], gaps[b]
    sig = [ga["significant"], gb["significant"]]
    gap_ok = all(sig) if prereg.require_both_factors else any(sig)
    pair_ok = (not prereg.require_top_s2_pair) or bool(s2["is_top_pair"])
    confirmed = bool(gap_ok and pair_ok)
    # "not confirmed" splits into two very different findings: a gap estimated at
    # ~zero (genuinely additive) vs a positive gap the CI can't resolve (not enough
    # samples). Conflating them would report a null result the data can't support.
    power_limited = (not gap_ok) and any(g["gap"] > prereg.min_gap for g in (ga, gb))

    reasons: list[str] = []
    for g in (ga, gb):
        reasons.append(
            f"{g['factor']}: ST-S1 = {g['gap']:.3f} [{g['gap_lo']:.3f}, "
            f"{g['gap_hi']:.3f}] — {'clears' if g['significant'] else 'does NOT clear'} "
            f"the pre-set {prereg.min_gap:.3f} threshold"
        )
    top_pair_name = (" x ".join(sobol["s2_ranked"][0]["pair"])
                     if sobol["s2_ranked"] else "n/a")
    if prereg.require_top_s2_pair:
        if s2["rank"] is None:
            reasons.append("S2: the registered pair is absent from the second-order matrix")
        else:
            where = ("top pair" if s2["is_top_pair"]
                     else f"NOT the top pair — the strongest S2 pair is {top_pair_name}")
            reasons.append(
                f"S2 direction check: {a} x {b} ranks {s2['rank']}/{s2['n_pairs']} ({where})"
            )

    stem = (
        f"We pre-registered {a} x {b} as a genuine two-way interaction "
        f"({prereg.source}"
        + (f", committed {commit}" if commit else "")
        + (f", {date}" if date else "")
        + f"). On the real grid (N={sobol.get('N')}, {sobol.get('n_eval')} model "
          f"evaluations) the ST-S1 gap is {ga['gap']:.3f} [{ga['gap_lo']:.3f}, "
          f"{ga['gap_hi']:.3f}] for {a} and {gb['gap']:.3f} [{gb['gap_lo']:.3f}, "
          f"{gb['gap_hi']:.3f}] for {b}, and S2({a}, {b}) = {s2['S2']:.3f} "
          f"±{s2['S2_conf']:.3f} (rank {s2['rank']}/{s2['n_pairs']}, direction only)"
    )

    if confirmed:
        sentence = (
            stem + " — CONFIRMED. Reverb and noise compound: both factors carry a "
            "substantial share of their influence through interaction rather than "
            "alone, and the second-order term agrees the partner is each other."
        )
    else:
        # An unconfirmed pre-registration is a RESULT. Say what was learned; do
        # not hedge, do not bury it, do not quietly retire the hypothesis.
        if not gap_ok and power_limited:
            # The honest third case: a positive point estimate whose CI is simply
            # too wide at this N. Calling that "additive" would be a stronger
            # claim than the data supports, in the opposite direction.
            learned = (
                f"The point estimates are positive but their CIs do not clear the "
                f"pre-set {prereg.min_gap:.3f} threshold: at N={sobol.get('N')} this "
                f"grid is UNDERPOWERED for an interaction of this size, which is not "
                f"the same as the factors being additive. SPEC §5's prescription "
                f"applies — re-run the second-order pass at higher N before calling "
                f"it either way."
            )
        elif not gap_ok:
            learned = (
                f"To the resolution of this grid the two degradations are ADDITIVE: "
                f"each factor's total-order effect is essentially its own first-order "
                f"effect, so no interaction term is needed to describe WER. That is "
                f"the useful, reportable finding — a practitioner can reason about "
                f"reverb and noise budgets independently."
            )
        else:
            other = top_pair_name
            learned = (
                f"The gap evidence says these factors DO act through interactions, "
                f"but the second-order term attributes the strongest pairing to "
                f"{other}, not to the registered pair. The registered hypothesis is "
                f"not supported as stated; the interaction structure is real but "
                f"differently shaped."
            )
        sentence = stem + " — NOT CONFIRMED. " + learned

    return {
        "prereg": asdict(prereg), "pair": (a, b),
        "status": "CONFIRMED" if confirmed else "NOT CONFIRMED",
        "confirmed": confirmed, "resolvable": True,
        "sentence": sentence,
        "primary": {a: ga, b: gb},          # magnitude lives here
        "secondary_which_pair_only": s2,    # direction only — see S2_CAVEAT
        "gap_criterion_met": bool(gap_ok), "pair_criterion_met": bool(pair_ok),
        "power_limited": bool(power_limited),
        "reasons": reasons, "rule": prereg.rule,
        "N": sobol.get("N"), "n_eval": sobol.get("n_eval"),
    }


def format_verdict(v: Mapping) -> str:
    """The verdict as it goes into the write-up — banner, sentence, rule, reasons."""
    bar = "=" * 78
    head = {"CONFIRMED": "PRE-REGISTRATION: CONFIRMED",
            "NOT CONFIRMED": "PRE-REGISTRATION: NOT CONFIRMED  (this is a RESULT, "
                             "not a failure)",
            "UNRESOLVED": "PRE-REGISTRATION: UNRESOLVED"}[v["status"]]
    lines = [bar, head, bar, "", v["sentence"], "",
             f"Decision rule (fixed in advance): {v['rule']}"]
    if v["reasons"]:
        lines.append("Evidence:")
        lines += [f"  - {r}" for r in v["reasons"]]
    lines += ["", f"NOTE: {S2_CAVEAT}"]
    return "\n".join(lines)


# ============================================================================
# MASTER TABLE -> factor matrix  (shared with analysis/al_savings.py, D3b)
# ============================================================================
#
# Deliberately a local, minimal reader: D3 depends only on design.py /
# active_learning.py, so the third leg keeps working even while the other Track-D
# layers are in flight. The factor COLUMN NAMES are not re-declared — they are
# read off DEFAULT_FACTOR_SPACE, which run_experiment.py's MASTER_COLUMNS asserts
# itself against (`MASTER_COLUMNS[2:7] == DEFAULT_FACTOR_SPACE.names`). One source
# of truth, no drift possible.

_NEEDED_COLUMNS = ("wer", "failed", "model", "condition_name")


def _as_float(v, default: float = float("nan")) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_bool(v) -> bool:
    """
    CSV round-trips booleans as strings, and the string "False" is TRUTHY. A
    plain `if row["failed"]` on a CSV-loaded table therefore treats every row as
    failed — or, if someone "fixes" it by dropping the cast, treats none as
    failed and averages the failure sentinels straight into the response surface.
    Both are silent. Hence this explicit coercion at the door.
    """
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "t"}
    return bool(v)


def load_master_rows(path: str = "results/master.csv") -> list[dict]:
    """Read the master table CSV into plain dicts (stdlib only, no pandas)."""
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows:
        have = set(rows[0])
        missing = [c for c in tuple(DEFAULT_FACTOR_SPACE.names) + _NEEDED_COLUMNS
                   if c not in have]
        if missing:
            raise ValueError(f"master table {path} is missing columns: {missing}")
    return rows


def encode_sample(space: FactorSpace, sample: Mapping) -> np.ndarray:
    """
    A sample dict -> one raw SALib-space row (the inverse of `space.decode`).

    design.py owns decode; encode is genuinely missing there, so it lives here —
    and categoricals are placed at the CENTRE of their level cell (idx + 0.5) so
    that `space.decode(encode_sample(space, s)) == s` exactly. Encoding at the
    left edge (idx + 0.0) round-trips too, but sits on the floor() boundary where
    a float wobble flips the level; the centre is the safe representative point.
    """
    row = []
    for f in space.factors:
        v = sample[f.name]
        if f.kind == "continuous":
            row.append(float(v))
        else:
            if v not in f.levels:
                raise ValueError(f"{v!r} is not a level of {f.name} ({f.levels})")
            row.append(float(f.levels.index(v)) + 0.5)
    return np.asarray(row, dtype=float)


def condition_matrix(rows: Sequence[Mapping], space: FactorSpace = DEFAULT_FACTOR_SPACE,
                     model: str | None = None, use_measured_rt60: bool = True,
                     ) -> dict:
    """
    Collapse the master table to ONE point per condition: mean WER over clips,
    encoded into the factor space the surrogate and the design layer share.

    Two decisions worth stating out loud:

    * FAILED ROWS ARE DROPPED, NOT AVERAGED. A `failed=True` row is a missing
      measurement carrying a sentinel WER (a null transcript scores 1.0). Averaging
      it in manufactures a failure region exactly where the study is most
      interesting — the harshest conditions are both the likeliest to fail and the
      ones the map is about. Their count is returned so it can be reported.
    * `rir_rt60_measured` OVERRIDES `rt60` when present. `AssetLibrary.pick_rir`
      snaps a requested rt60 to the closest measured RIR, so a request for 1.0 s
      may be delivered as 0.45 s. The surrogate must be fitted against the reverb
      the model actually HEARD, not the coordinate we asked for (SPEC A.R4.2 §2).
      Requested values are kept alongside for traceability.
    """
    names = list(space.names)
    buckets: dict[tuple, dict] = {}
    n_failed = n_skipped = 0

    for r in rows:
        if model is not None and r.get("model") != model:
            continue
        if _as_bool(r.get("failed")):
            n_failed += 1
            continue
        wer = _as_float(r.get("wer"))
        if not math.isfinite(wer):
            n_skipped += 1
            continue
        sample: dict = {}
        ok = True
        for f in space.factors:
            if f.kind == "continuous":
                v = _as_float(r.get(f.name))
                if f.name == "rt60" and use_measured_rt60:
                    m = _as_float(r.get("rir_rt60_measured"))
                    if math.isfinite(m):
                        v = m
                if not math.isfinite(v):
                    ok = False
                    break
                sample[f.name] = v
            else:
                v = r.get(f.name)
                if v not in (f.levels or ()):
                    ok = False
                    break
                sample[f.name] = v
        if not ok:
            n_skipped += 1
            continue
        key = tuple(round(sample[n], 6) if isinstance(sample[n], float) else sample[n]
                    for n in names)
        b = buckets.setdefault(key, {"sample": sample, "wers": [],
                                     "condition_name": r.get("condition_name"),
                                     "rt60_requested": _as_float(r.get("rt60"))})
        b["wers"].append(wer)

    conditions = []
    X = []
    y = []
    for b in buckets.values():
        conditions.append({
            "sample": b["sample"], "condition_name": b["condition_name"],
            "rt60_requested": b["rt60_requested"],
            "wer": float(np.mean(b["wers"])), "n_clips": len(b["wers"]),
        })
        X.append(encode_sample(space, b["sample"]))
        y.append(float(np.mean(b["wers"])))

    return {
        "X_raw": np.asarray(X, dtype=float).reshape(len(X), len(names)),
        "y": np.asarray(y, dtype=float),
        "conditions": conditions,
        "n_conditions": len(conditions),
        "n_rows_used": sum(c["n_clips"] for c in conditions),
        "n_failed_rows": n_failed, "n_skipped_rows": n_skipped,
        "model": model, "use_measured_rt60": use_measured_rt60,
        "space_names": names,
    }


def fit_surrogate_from_master(rows: Sequence[Mapping],
                              space: FactorSpace = DEFAULT_FACTOR_SPACE,
                              model: str | None = None,
                              use_measured_rt60: bool = True,
                              seed: int = 0, min_conditions: int = 8,
                              ) -> tuple[GPSurrogate, dict]:
    """
    Fit the GP surrogate (active_learning.GPSurrogate — reused, not reimplemented)
    to the real measured response surface. This is the object that makes the dense
    counterintuitive sweep affordable; it is NOT a measurement device.
    """
    mat = condition_matrix(rows, space, model=model,
                           use_measured_rt60=use_measured_rt60)
    if mat["n_conditions"] < min_conditions:
        raise ValueError(
            f"only {mat['n_conditions']} usable conditions in the master table "
            f"(need >= {min_conditions}); a GP fitted to fewer than that will "
            f"invent structure the grid never measured"
        )
    gp = GPSurrogate(space, seed=seed).fit(mat["X_raw"], mat["y"])
    return gp, mat


# ============================================================================
# STAGE 1 of 2 — PROPOSE counterintuitive cells from the surrogate (UNCONFIRMED)
# ============================================================================

PROPOSED = "PROPOSED_UNCONFIRMED"
CONFIRMED = "CONFIRMED_BY_ORACLE"
REFUTED = "REFUTED_BY_ORACLE"

_SURROGATE_EVIDENCE = "surrogate-proposed (NOT measured)"


def _grid_key(v) -> object:
    return round(float(v), 9) if isinstance(v, (int, float, np.floating)) else v


def surrogate_wer_fn(gp: GPSurrogate, space: FactorSpace, grid: int
                     ) -> Callable[[dict], float]:
    """
    A `wer_fn(sample) -> float` backed by the GP, with the whole dense grid
    predicted in ONE batched call and served from a lookup.

    `find_counterintuitive_cells` calls wer_fn once per cell (grid^5 ≈ thousands);
    a GP `predict` per call is milliseconds each and turns a 3-second scan into a
    multi-minute one. Same numbers, batched.
    """
    names = list(space.names)
    axes = [space.axis_values(f, grid) for f in space.factors]
    combos = list(itertools.product(*axes))
    X = np.asarray([encode_sample(space, dict(zip(names, c))) for c in combos],
                   dtype=float)
    mu = np.asarray(gp.predict(X, return_std=False), dtype=float).ravel()
    table = {tuple(_grid_key(v) for v in c): float(m) for c, m in zip(combos, mu)}

    def wer_fn(sample: Mapping) -> float:
        key = tuple(_grid_key(sample[n]) for n in names)
        hit = table.get(key)
        if hit is not None:
            return hit
        # off-grid (confirmation neighbourhoods) — single predict, still the GP
        x = encode_sample(space, sample).reshape(1, -1)
        return float(np.asarray(gp.predict(x, return_std=False)).ravel()[0])

    wer_fn.table = table          # exposed for tests / debugging
    return wer_fn


def _held_defaults(space: FactorSpace, grid_axes: Sequence[Sequence]) -> dict:
    """Mid-grid values for factors not named in a reversal (see the caveat below)."""
    out = {}
    for f, ax in zip(space.factors, grid_axes):
        out[f.name] = ax[len(ax) // 2]
    return out


def _neighbour_triplet(axis: Sequence, value) -> list:
    """The (p-1, p, p+1) grid values around `value` — what a dip needs to be real."""
    vals = [float(v) for v in axis]
    p = int(np.argmin([abs(v - float(value)) for v in vals]))
    p = min(max(p, 1), len(vals) - 2)
    return [vals[p - 1], vals[p], vals[p + 1]]


def propose_counterintuitive_cells(rows: Sequence[Mapping],
                                   space: FactorSpace = DEFAULT_FACTOR_SPACE,
                                   *, model: str | None = None,
                                   grid: int = 9, top_k: int = 6, seed: int = 0,
                                   use_measured_rt60: bool = True,
                                   gp: GPSurrogate | None = None,
                                   ) -> dict:
    """
    STAGE 1 of the two-stage protocol (SPEC A.R5.4): fit a GP to the master table,
    sweep `design.find_counterintuitive_cells` over the SURROGATE, and return the
    hits as CANDIDATES — every one flagged `confirmed=False` — together with the
    exact condition list to re-evaluate against the real oracle.

    Nothing in here is a measurement. The returned dict says so in three places
    (`status`, per-candidate `evidence`, and `presentable_as_measured=False`)
    because the failure mode is not a crash, it is a plausible sentence in a
    write-up about a dip that only ever existed inside a GP.
    """
    if gp is None:
        gp, mat = fit_surrogate_from_master(rows, space, model=model,
                                            use_measured_rt60=use_measured_rt60,
                                            seed=seed)
    else:
        mat = condition_matrix(rows, space, model=model,
                               use_measured_rt60=use_measured_rt60)

    wer_fn = surrogate_wer_fn(gp, space, grid)
    cells = find_counterintuitive_cells(space, wer_fn, grid=grid, top_k=top_k)

    axes = [space.axis_values(f, grid) for f in space.factors]
    axis_by_name = {f.name: ax for f, ax in zip(space.factors, axes)}
    held = _held_defaults(space, axes)

    candidates: list[dict] = []

    # --- (a) non-monotonic dips: confirmable with a 3-point sweep -------------
    for c in cells["non_monotonic"]:
        coords = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                  for k, v in c["coords"].items()}
        triplet = _neighbour_triplet(axis_by_name[c["factor"]], c["at_value"])
        probes = []
        for v in triplet:
            s = dict(coords)
            s[c["factor"]] = v
            probes.append(s)
        candidates.append({
            "kind": "non_monotonic",
            "factor": c["factor"], "at_value": float(c["at_value"]),
            "surrogate_wer_at_dip": float(c["wer_at_dip"]),
            "surrogate_depth": float(c["depth"]),
            "coords": coords, "probe_conditions": probes, "probe_values": triplet,
            "confirmed": False, "status": PROPOSED, "evidence": _SURROGATE_EVIDENCE,
            "note": c["note"],
            "confirm_test": (f"measure WER at {c['factor']} = {triplet} with the "
                             f"other factors fixed; the middle point must be the "
                             f"lowest of the three"),
        })

    # --- (b) effect reversals: confirmable with a 2x2 corner design -----------
    for c in cells["reversals"]:
        a_name, b_name = c["factor"], c["moderator"]
        a_ax, b_ax = axis_by_name[a_name], axis_by_name[b_name]
        corners = []
        for b_val in (b_ax[0], b_ax[-1]):
            for a_val in (a_ax[0], a_ax[-1]):
                s = dict(held)
                s[a_name] = a_val
                s[b_name] = b_val
                corners.append(s)
        candidates.append({
            "kind": "reversal",
            "factor": a_name, "moderator": b_name,
            "surrogate_effect_at_low_moderator": float(c["effect_at_low_moderator"]),
            "surrogate_effect_at_high_moderator": float(c["effect_at_high_moderator"]),
            "surrogate_swing": float(c["swing"]),
            "coords": dict(held), "probe_conditions": corners,
            "confirmed": False, "status": PROPOSED, "evidence": _SURROGATE_EVIDENCE,
            "note": c["note"],
            "confirm_test": (f"measure the 2x2 corners of {a_name} x {b_name} (other "
                             f"factors held at {held}); the sign of {a_name}'s effect "
                             f"must flip between the two {b_name} levels"),
            "caveat": ("the surrogate detects reversals on MARGINAL means over the "
                       "other factors; the 2x2 confirmation holds them fixed, so a "
                       "failure to confirm may mean the reversal is marginal-only"),
        })

    # de-duplicate the oracle work: one call per distinct condition
    to_eval: list[dict] = []
    seen: set[tuple] = set()
    for cand in candidates:
        for s in cand["probe_conditions"]:
            k = tuple(_grid_key(s[n]) for n in space.names)
            if k not in seen:
                seen.add(k)
                to_eval.append(dict(s))

    return {
        "status": PROPOSED,
        "presentable_as_measured": False,
        "source": "GP surrogate fitted to the master results table",
        "warning": ("SURROGATE-PROPOSED, NOT MEASURED. Confirm each cell with real "
                    "oracle calls (confirm_cells) before it appears in the write-up."),
        "candidates": candidates,
        "n_candidates": len(candidates),
        "conditions_to_evaluate": to_eval,
        "n_oracle_calls_required": len(to_eval),
        "grid": grid, "top_k": top_k,
        "space_names": list(space.names),
        "surrogate_train_conditions": mat["n_conditions"],
        "surrogate_train_rows": mat["n_rows_used"],
        "model": model,
    }


# ============================================================================
# STAGE 2 of 2 — CONFIRM against the real oracle
# ============================================================================

def confirm_cells(proposal: Mapping, oracle: Callable[[dict], float],
                  eps: float = 1e-3) -> dict:
    """
    STAGE 2: re-evaluate every proposed cell's probe conditions through the REAL
    oracle and flip `confirmed` only where the measurement reproduces the pattern.

    The only function that may set `confirmed=True`, and it can only do it by
    spending oracle calls — which is exactly the property that keeps a GP's
    imagination out of the write-up. Refuted candidates are KEPT (status
    REFUTED_BY_ORACLE) with their measurements: "the surrogate predicted a dip
    here and the microphone disagreed" is itself worth a line about surrogate
    fidelity.
    """
    measured: dict[tuple, float] = {}
    names = list(proposal.get("space_names") or DEFAULT_FACTOR_SPACE.names)
    n_calls = 0

    def measure(sample: Mapping) -> float:
        nonlocal n_calls
        key = tuple(_grid_key(sample[n]) for n in names)
        if key not in measured:
            measured[key] = float(oracle(dict(sample)))
            n_calls += 1
        return measured[key]

    out_candidates = []
    for cand in proposal["candidates"]:
        c = dict(cand)
        vals = [measure(s) for s in c["probe_conditions"]]
        c["measured_wer"] = vals
        if c["kind"] == "non_monotonic":
            lo, mid, hi = vals
            ok = (mid < lo - eps) and (mid < hi - eps)
            c["measured_depth"] = float(min(lo, hi) - mid)
        else:
            e_lo = vals[1] - vals[0]      # effect of `factor` at low moderator
            e_hi = vals[3] - vals[2]      # ... and at high moderator
            ok = (e_lo * e_hi < 0) and abs(e_lo) > eps and abs(e_hi) > eps
            c["measured_effect_at_low_moderator"] = float(e_lo)
            c["measured_effect_at_high_moderator"] = float(e_hi)
        c["confirmed"] = bool(ok)
        c["status"] = CONFIRMED if ok else REFUTED
        c["evidence"] = "measured by the real oracle" if ok else \
                        "surrogate proposed it; the oracle did NOT reproduce it"
        out_candidates.append(c)

    n_conf = sum(1 for c in out_candidates if c["confirmed"])
    return {
        **{k: v for k, v in proposal.items() if k != "candidates"},
        "status": CONFIRMED if n_conf else REFUTED,
        "presentable_as_measured": True,
        "warning": (f"{n_conf}/{len(out_candidates)} proposed cells reproduced under "
                    f"real transcription; only those may be presented as findings."),
        "candidates": out_candidates,
        "n_confirmed": n_conf,
        "n_refuted": len(out_candidates) - n_conf,
        "n_oracle_calls_spent": n_calls,
    }


# ============================================================================
# THE REPORT
# ============================================================================

def interaction_report(sobol: Mapping, master_rows: Sequence[Mapping] | None = None,
                       space: FactorSpace = DEFAULT_FACTOR_SPACE,
                       *, prereg: PreRegistration = PREREGISTRATION,
                       model: str | None = None, grid: int = 9, top_k: int = 6,
                       seed: int = 0, commit: str | None = None,
                       date: str | None = None) -> dict:
    """
    The whole D3a deliverable: Sobol tables, PRIMARY gap evidence, the verdict,
    and (if the master table is supplied) surrogate-proposed counterintuitive
    cells marked unconfirmed.
    """
    sobol = normalize_sobol_result(sobol)
    res = {
        "sobol_tables": format_sobol_tables(sobol),
        "evidence": interaction_evidence(sobol, prereg),
        "verdict": preregistration_verdict(sobol, prereg, commit=commit, date=date),
        "counterintuitive": None,
    }
    if master_rows is not None:
        try:
            res["counterintuitive"] = propose_counterintuitive_cells(
                master_rows, space, model=model, grid=grid, top_k=top_k, seed=seed)
        except ValueError as e:                   # too few conditions to fit a GP
            res["counterintuitive"] = {
                "status": "UNAVAILABLE", "presentable_as_measured": False,
                "candidates": [], "n_candidates": 0,
                "conditions_to_evaluate": [], "n_oracle_calls_required": 0,
                "warning": f"surrogate not fitted: {e}",
            }
    return res


def format_interaction_report(res: Mapping) -> str:
    """Printed report. Ordering is load-bearing: verdict, then PRIMARY gap, then
    S2 (labelled direction-only), then the UNCONFIRMED proposals."""
    out = [format_verdict(res["verdict"]), ""]

    ev = res["evidence"]
    out += ["-" * 78,
            "PRIMARY INTERACTION EVIDENCE — ST-S1 gap (quote sizes from HERE)",
            "-" * 78,
            f"  {'factor':<12} {'S1':>16} {'ST':>16} {'ST-S1':>8} {'95% CI':>18}  sig"]
    for g in ev["primary_gap"]:
        out.append(
            f"  {g['factor']:<12} {g['S1']:>7.3f}±{g['S1_conf']:<7.3f} "
            f"{g['ST']:>7.3f}±{g['ST_conf']:<7.3f} {g['gap']:>8.3f} "
            f"[{g['gap_lo']:>7.3f},{g['gap_hi']:>7.3f}]  {'YES' if g['significant'] else '-'}"
        )
    out += ["",
            "-" * 78,
            "SECONDARY — S2, WHICH PAIR ONLY (never a magnitude)",
            "-" * 78,
            f"  {S2_CAVEAT}", ""]
    for i, d in enumerate(ev["secondary_which_pair_only"], start=1):
        a, b = d["pair"]
        crosses = (d["S2"] - d["S2_conf"]) <= 0.0 <= (d["S2"] + d["S2_conf"])
        out.append(f"  {i:>2}. {a + ' x ' + b:<26} S2={d['S2']:>7.3f}±{d['S2_conf']:<7.3f}"
                   f"  {'(CI crosses 0 — direction unreliable)' if crosses else ''}")

    out += ["", "-" * 78, "FULL SOBOL TABLES", "-" * 78, res["sobol_tables"]]

    ci = res.get("counterintuitive")
    if ci:
        out += ["", "!" * 78,
                f"COUNTERINTUITIVE CELLS — {ci['status']}",
                f"  {ci.get('warning', '')}",
                "!" * 78]
        if ci.get("candidates"):
            for i, c in enumerate(ci["candidates"], start=1):
                mark = "CONFIRMED" if c.get("confirmed") else "unconfirmed"
                out.append(f"  {i:>2}. [{mark}] {c['kind']}: {c['note']}")
                out.append(f"      evidence: {c['evidence']}")
                out.append(f"      confirm by: {c['confirm_test']}")
            if not ci.get("presentable_as_measured", False):
                out.append(f"  -> {ci['n_oracle_calls_required']} real oracle calls "
                           f"needed to confirm these; NONE of them may be presented "
                           f"as a measured finding until then.")
        else:
            out.append("  (no candidates found)")
    return "\n".join(out)


def plot_payload(res: Mapping) -> dict:
    """
    JSON-safe payload for the dashboard (E2). STABLE SHAPE — a parallel consumer
    depends on these exact keys; add, never rename or remove:

      gap_bars   list[dict], sorted strongest-gap first — THE PRIMARY CHART:
                   factor, S1, ST, gap (= ST-S1), gap_lo, gap_hi, significant
                 gap/gap_lo/gap_hi are the only interaction MAGNITUDES that may
                 be rendered with a number attached.
      s2_pairs   list[dict]: pair (2-list), S2, S2_conf, magnitude_usable (always
                 False). Render as a RANKING only — no bar length, no axis, no
                 "explains X% of variance". Show `s2_caveat` beside it.
      s2_caveat  str — must be displayed wherever s2_pairs is.
      verdict    {status: "CONFIRMED"|"NOT CONFIRMED"|"UNRESOLVED", confirmed: bool,
                  sentence: str (publishable verbatim), pair: 2-list}
                 NOT CONFIRMED is a RESULT: render it with the same prominence as
                 CONFIRMED, never greyed out or hidden behind a toggle.
      counterintuitive
                 {status, presentable_as_measured: bool, n_candidates, n_confirmed,
                  cells: [{kind, factor, moderator, confirmed, note}]}
                 If presentable_as_measured is False, every cell is a GP
                 prediction: label it UNCONFIRMED in the UI.
    """
    ev = res["evidence"]
    v = res["verdict"]
    ci = res.get("counterintuitive") or {}
    return {
        "gap_bars": [{"factor": g["factor"], "S1": g["S1"], "ST": g["ST"],
                      "gap": g["gap"], "gap_lo": g["gap_lo"], "gap_hi": g["gap_hi"],
                      "significant": g["significant"]}
                     for g in ev["primary_gap"]],
        "s2_pairs": [{"pair": list(d["pair"]), "S2": float(d["S2"]),
                      "S2_conf": float(d["S2_conf"]),
                      "magnitude_usable": False}
                     for d in ev["secondary_which_pair_only"]],
        "s2_caveat": S2_CAVEAT,
        "verdict": {"status": v["status"], "confirmed": v["confirmed"],
                    "sentence": v["sentence"], "pair": list(v["pair"])},
        "counterintuitive": {
            "status": ci.get("status"),
            "presentable_as_measured": bool(ci.get("presentable_as_measured", False)),
            "n_candidates": ci.get("n_candidates", 0),
            "n_confirmed": ci.get("n_confirmed", 0),
            "cells": [{"kind": c["kind"], "factor": c.get("factor"),
                       "moderator": c.get("moderator"),
                       "confirmed": bool(c.get("confirmed")),
                       "note": c.get("note")}
                      for c in ci.get("candidates", [])],
        },
    }


# ============================================================================
# CLI
# ============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D3a — interaction hunt + pre-registration verdict")
    ap.add_argument("sobol", nargs="?", default="results/sobol.json",
                    help="results/sobol.json from design.sobol_indices (R4.5)")
    ap.add_argument("--master", default=None,
                    help="results/master.csv — enables surrogate-proposed cells")
    ap.add_argument("--model", default=None, help="restrict the master table to one model arm")
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--commit", default=None, help="SHA the pre-registration was committed at")
    ap.add_argument("--date", default=None)
    ap.add_argument("--json-out", default=None, help="write the plot payload here")
    args = ap.parse_args(argv)

    sobol = load_sobol_json(args.sobol)
    rows = load_master_rows(args.master) if args.master else None
    res = interaction_report(sobol, rows, model=args.model, grid=args.grid,
                             top_k=args.top_k, commit=args.commit, date=args.date)
    print(format_interaction_report(res))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(plot_payload(res), f, indent=2)
        print(f"\n[wrote] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
