#!/usr/bin/env python3
"""
scripts/make_readme_figures.py — the five figures the README embeds, generated
from `results/` so they are reproducible rather than hand-drawn.

    ./.venv/bin/python scripts/make_readme_figures.py            # write them
    ./.venv/bin/python scripts/make_readme_figures.py --check    # fail if stale

Outputs (paths are a contract with README.md — do NOT rename):

    docs/assets/confidence-vs-error.svg     D1 hero: confidence vs error, nova-3
    docs/assets/overconfidence.svg          D1 threshold-free: the gap distribution
    docs/assets/fingerprints.svg            D2: typed-edit composition per factor
    docs/assets/model-comparison.svg        L1: the three arms, matched population
    docs/assets/whisper-hallucination.svg   L1 exhibit: one hallucinated row
    docs/assets/figures.json                every number drawn, machine-readable

===========================================================================
THE POPULATION RULE — why every figure carries its clip count on its face
===========================================================================
nova-3 ran 40 clips; whisper-base and elevenlabs-scribe ran the 10-clip AL
subset. So nova-3's dead-zone rate is 1.14% (2/176) on its own 40 clips and
0.57% (1/176) on the 10 clips where the three arms are comparable. BOTH ARE
CORRECT; quoting one without its clip count is the bug, and it is the bug this
project has committed at least three times (SPEC Appendix G; UNDERSTANDING
4.12), twice inside a chart or a table.

So: figures 1-3 are nova-3 on 40 clips and say so. Figure 4 is the three arms
on the matched 10-clip intersection (`model_arms.arm_intersection`, which takes
the intersection and returns the census) and says so, INCLUDING the sentence
that nova-3's rate reads differently on its own 40. Nothing is drawn without a
population label, and `figures.json` records the population of every number.

===========================================================================
THREE THINGS THIS SCRIPT REFUSES TO DRAW
===========================================================================
1. MUTE CONDITIONS ARE NOT PLOTTED ON THE CONFIDENCE AXIS. Seven nova-3
   conditions return an empty transcript on every clip: WER 1.0 and NO
   confidence at all. Placing them at confidence 0 would fabricate seven points
   at the ideal corner of a negative correlation — which is exactly the defect
   SPEC G.7 found in `overall_correlation` (a published rho of -0.957 that was
   an artifact of those seven points). They get a counted annotation instead.

2. ELEVENLABS-SCRIBE GETS NO DEAD-ZONE BAR. `dead_zone_flags` thresholds an
   ABSOLUTE WER, so it is not scale-free, and Scribe's orthography is
   non-deterministic across identical calls (SPEC I.2/I.5): its 7 strict dead
   zones fall to 0 under the cross-model normalizer because they were spelling,
   not confident error. Rank statistics survive the contamination (attenuated,
   so they are a lower bound) and Scribe keeps its rho bar. The dead-zone slot
   is drawn as an explicit exclusion with the reason, not left blank.

3. NO FIGURE IMPLIES THE DEAD-ZONE COUNT IS ROBUST. Both thresholds are
   hardcoded defaults and `results/dead_zone_sensitivity.json` returns FRAGILE
   for all three arms (nova-3's count runs 0-86 across the swept grid). Figure 1
   labels the box an operating point; figure 2 is the threshold-free version of
   the same claim; figure 4 quotes the one ordering that IS threshold-free,
   recomputed here on the matched population because the artifact's own sweep
   mixes nova-3's 40 clips with the others' 10.

===========================================================================
RENDERING CHOICES
===========================================================================
* Explicit LIGHT background on every figure (surface #fcfcfb + a hairline
  ring), not a theme-adaptive one. GitHub renders READMEs in both themes and
  serves README images through an <img> tag, so a transparent background with
  dark ink vanishes in dark mode. Presentation attributes only -- no <style>
  block, no CSS, no media query, no external font -- so nothing depends on what
  GitHub's sanitizer allows through.
* Sized for a shared screen: 900-unit viewBox, body type >= 20 units, so at the
  ~60-70% of README width a projected browser gives you the smallest text still
  lands around 13px.
* Palette is the validated categorical set (slots 1-3: #2a78d6 / #eb6834 /
  #1baf7a) plus the diverging blue-red pair; all-pairs CVD dE 9.2, normal-vision
  dE 24.0 on this surface. Aqua sits at 2.74:1 contrast, so every aqua mark
  carries a visible direct label (the relief rule).

Deps: numpy, scipy (via the analysis layer). No API, no audio, no network.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Sequence
from xml.sax.saxutils import escape

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.analysis import as_float, load_master_table            # noqa: E402
from deadzone.analysis.confidence_gap import (                       # noqa: E402
    CONF_PCT_HI, WER_HI, condition_flags, confidence_gap_report, plot_payload,
)
from deadzone.analysis.model_arms import (                           # noqa: E402
    SPINE_MODEL, arm_intersection, condition_table,
)
from deadzone.cross_model_norm import cross_model_normalize          # noqa: E402
from deadzone.model_compare import (                                 # noqa: E402
    confidence_wer_shape, is_wer_comparable,
)

# --------------------------------------------------------------------------
# paths — the five are a contract with README.md
# --------------------------------------------------------------------------
OUT_DIR = os.path.join(_REPO_ROOT, "docs", "assets")
FIG_CONFIDENCE = "confidence-vs-error.svg"
FIG_OVERCONF = "overconfidence.svg"
FIG_FINGERPRINTS = "fingerprints.svg"
FIG_MODELS = "model-comparison.svg"
FIG_HALLUCINATION = "whisper-hallucination.svg"
FIG_NUMBERS = "figures.json"
FIGURES = (FIG_CONFIDENCE, FIG_OVERCONF, FIG_FINGERPRINTS, FIG_MODELS,
           FIG_HALLUCINATION)

MASTER = os.path.join(_REPO_ROOT, "results", "master.csv")
FINGERPRINTS_JSON = os.path.join(_REPO_ROOT, "results", "fingerprints.json")
DZ_SENSITIVITY_JSON = os.path.join(_REPO_ROOT, "results",
                                   "dead_zone_sensitivity.json")
MANIFEST_CSV = os.path.join(_REPO_ROOT, "recording_manifest.csv")

# The one row figure 5 exhibits. Named as a constant because the figure quotes
# it verbatim and the test re-reads the same cell out of master.csv.
HALLU_CLIP, HALLU_COND = "u02", "rt60-1_snr-5_babble_opus-lowrate_roll-1"
HALLU_MODEL = "whisper-base"
HALLU_PHRASE = "you have a file"

# --------------------------------------------------------------------------
# palette + type scale (see the module docstring)
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
RING = "#e3e2dc"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"        # categorical slot 1
ORANGE = "#eb6834"      # categorical slot 2
AQUA = "#1baf7a"        # categorical slot 3  (2.74:1 -> always direct-labelled)
RED = "#e34948"         # diverging pole / the danger accent
RED_DEEP = "#b3282e"

FONT = ("system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, "
        "Arial, sans-serif")
MONO = ("ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
        "'Liberation Mono', monospace")

T_TITLE, T_SUB, T_AXIS, T_TICK, T_ANNO, T_FOOT, T_HERO = 36, 23, 23, 21, 22, 20, 46
W = 900          # viewBox width, shared by every figure


# ==========================================================================
# 1. a very small SVG writer (presentation attributes only)
# ==========================================================================
class Svg:
    """Append-only SVG builder. No <style>, no CSS, no external anything."""

    def __init__(self, width: int, height: int, title: str, desc: str):
        self.w, self.h = width, height
        self.parts: list[str] = []
        self._head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{esc(title)}">'
            f'<title>{esc(title)}</title><desc>{esc(desc)}</desc>'
            # The explicit light background. Without it the ink disappears when
            # GitHub renders the README in dark mode.
            f'<rect x="0" y="0" width="{width}" height="{height}" rx="10" '
            f'fill="{SURFACE}"/>'
            f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" '
            f'rx="10" fill="none" stroke="{RING}" stroke-width="1"/>')

    def add(self, s: str) -> None:
        self.parts.append(s)

    # An SVG neither clips nor reflows: text past the canvas edge is simply
    # drawn outside it, the file stays valid, and nothing complains. So the
    # writer refuses instead — the same move as `write_master` refusing to
    # write a duplicated table (SPEC E.3): cheaper to refuse at the one
    # creation site than to hope somebody opens the file afterwards.
    MARGIN = 16.0

    def text(self, x: float, y: float, s: str, size: float = T_ANNO,
             fill: str = INK, weight: str = "400", anchor: str = "start",
             family: str = FONT, opacity: float | None = None,
             letter: float | None = None, rotate: float | None = None) -> None:
        extra = "" if opacity is None else f' opacity="{opacity}"'
        if letter is not None:
            extra += f' letter-spacing="{letter}"'
        if rotate is not None:
            extra += f' transform="rotate({f2(rotate)} {f2(x)} {f2(y)})"'
        else:
            self._assert_fits(x, y, s, size, weight, anchor, family)
        self.add(f'<text x="{f2(x)}" y="{f2(y)}" font-family="{family}" '
                 f'font-size="{f2(size)}" fill="{fill}" font-weight="{weight}" '
                 f'text-anchor="{anchor}"{extra}>{esc(s)}</text>')

    def _assert_fits(self, x, y, s, size, weight, anchor, family) -> None:
        tw = text_w(s, size, weight, mono=(family == MONO))
        left = {"start": x, "middle": x - tw / 2, "end": x - tw}[anchor]
        if left < self.MARGIN - 4 or left + tw > self.w - self.MARGIN + 4:
            raise SystemExit(
                f"text would overflow the canvas: "
                f"[{left:.0f}..{left + tw:.0f}] of {self.w} — {s[:70]!r}")
        if y - size > self.h or y > self.h - 4:
            raise SystemExit(
                f"text would fall off the bottom: baseline {y:.0f} of {self.h} "
                f"— {s[:70]!r}")

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "none",
             stroke: str | None = None, sw: float = 1, rx: float = 0,
             opacity: float | None = None, dash: str | None = None) -> None:
        a = (f'<rect x="{f2(x)}" y="{f2(y)}" width="{f2(max(w, 0))}" '
             f'height="{f2(max(h, 0))}" rx="{f2(rx)}" fill="{fill}"')
        if stroke:
            a += f' stroke="{stroke}" stroke-width="{f2(sw)}"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            a += f' opacity="{opacity}"'
        self.add(a + "/>")

    def line(self, x1: float, y1: float, x2: float, y2: float,
             stroke: str = GRID, sw: float = 1, dash: str | None = None,
             opacity: float | None = None, cap: str = "butt") -> None:
        a = (f'<line x1="{f2(x1)}" y1="{f2(y1)}" x2="{f2(x2)}" y2="{f2(y2)}" '
             f'stroke="{stroke}" stroke-width="{f2(sw)}" stroke-linecap="{cap}"')
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            a += f' opacity="{opacity}"'
        self.add(a + "/>")

    def circle(self, cx: float, cy: float, r: float, fill: str = "none",
               stroke: str | None = None, sw: float = 1,
               opacity: float | None = None) -> None:
        a = f'<circle cx="{f2(cx)}" cy="{f2(cy)}" r="{f2(r)}" fill="{fill}"'
        if stroke:
            a += f' stroke="{stroke}" stroke-width="{f2(sw)}"'
        if opacity is not None:
            a += f' opacity="{opacity}"'
        self.add(a + "/>")

    def poly(self, pts: Sequence[tuple[float, float]], fill: str,
             opacity: float | None = None) -> None:
        d = " ".join(f"{f2(x)},{f2(y)}" for x, y in pts)
        a = f'<polygon points="{d}" fill="{fill}"'
        if opacity is not None:
            a += f' opacity="{opacity}"'
        self.add(a + "/>")

    def render(self) -> str:
        return self._head + "".join(self.parts) + "</svg>\n"


def esc(s) -> str:
    return escape(str(s), {'"': "&quot;", "'": "&apos;"})


def f2(v: float) -> str:
    """Trim float noise so the files diff cleanly between runs."""
    return f"{float(v):.2f}".rstrip("0").rstrip(".") or "0"


# --------------------------------------------------------------------------
# text metrics — why this table exists rather than a "characters per line"
# guess. Wrapping on a character count treats "WWW" and "iii" as equal, so a
# caps-heavy footnote overflows the canvas while a lowercase one wastes half
# the width. Nothing in an SVG clips or reflows, so the overflow is silent: the
# file is valid, the text is simply drawn off the edge, and it looks fine until
# someone opens it. Advances are Helvetica's (which system-ui approximates on
# both macOS and Windows); measured against headless Chrome across 11 real
# strings from these figures, the table OVER-estimates by 2-10%, which is the
# safe direction — reserve slightly too much space, never too little.
# --------------------------------------------------------------------------
_ADV_REGULAR = (
    "278 278 355 556 556 889 667 191 333 333 389 584 278 333 278 278 "
    "556 556 556 556 556 556 556 556 556 556 278 278 584 584 584 556 1015 "
    "667 667 722 722 667 611 778 722 278 500 667 556 833 722 778 667 778 722 "
    "667 611 722 667 944 667 667 611 278 278 278 469 556 333 "
    "556 556 500 556 556 278 556 556 222 222 500 222 833 556 556 556 556 333 "
    "500 278 556 500 722 500 500 500 334 260 334 584")
_ADV_BOLD = (
    "278 333 474 556 556 889 722 238 333 333 389 584 278 333 278 278 "
    "556 556 556 556 556 556 556 556 556 556 333 333 584 584 584 611 975 "
    "722 722 722 722 667 611 778 722 278 556 722 611 833 722 778 667 778 722 "
    "667 611 722 667 944 667 667 611 333 278 333 584 556 333 "
    "556 611 556 611 556 333 611 611 278 278 556 278 889 611 611 611 611 389 "
    "556 333 611 556 778 556 556 500 389 280 389 584")
_ASCII = [chr(c) for c in range(32, 127)]
_REG = dict(zip(_ASCII, (int(v) for v in _ADV_REGULAR.split())))
_BOLD = dict(zip(_ASCII, (int(v) for v in _ADV_BOLD.split())))
# the non-ASCII glyphs these figures actually use
_EXTRA = {"·": 333, "×": 584, "—": 1000, "–": 556, "→": 1000, "≥": 584,
          "≤": 584, "“": 333, "”": 333, "’": 222, "−": 584, "±": 584}
MONO_ADV = 0.605          # measured: ui-monospace advance / font-size


def text_w(s: str, size: float, weight: str = "400",
           mono: bool = False) -> float:
    """Advance width of `s`, in the same units as the viewBox."""
    if mono:
        return len(s) * size * MONO_ADV
    table = _BOLD if int(weight) >= 600 else _REG
    return sum(_EXTRA.get(c, table.get(c, 556)) for c in s) / 1000.0 * size


def wrap_px(text: str, size: float, max_px: float, weight: str = "400",
            mono: bool = False) -> list[str]:
    """Greedy wrap on MEASURED width. A single over-long word is not split."""
    out, line = [], ""
    for word in text.split():
        cand = f"{line} {word}".strip()
        if line and text_w(cand, size, weight, mono) > max_px:
            out.append(line)
            line = word
        else:
            line = cand
    if line:
        out.append(line)
    return out


def footer(s: Svg, y: float, lines: Sequence[str], x: float = 30,
           max_px: float = W - 60) -> float:
    """Footnotes, wrapped to the canvas. Returns the y after the last line."""
    for para in lines:
        for ln in wrap_px(para, T_FOOT, max_px):
            s.text(x, y, ln, T_FOOT, MUTED)
            y += T_FOOT + 6
        y += 5
    return y


def footer_height(lines: Sequence[str], max_px: float = W - 60) -> float:
    n = sum(len(wrap_px(p, T_FOOT, max_px)) for p in lines)
    return n * (T_FOOT + 6) + 5 * len(lines)


def card_height(cards: list[dict], total_w: float, gap: float = 10) -> float:
    """Height `stat_cards` will use — needed before the canvas exists."""
    inner = (total_w - gap * (len(cards) - 1)) / len(cards) - 28
    return 76 + max(len(wrap_px(c["caption"], T_FOOT - 1, inner))
                    for c in cards) * 23 + 14


def stat_cards(s: Svg, x: float, y: float, total_w: float, cards: list[dict],
               gap: float = 10) -> float:
    """
    A row of equal-width stat tiles: a hero figure with its caption under it.

    The caption is wrapped to the tile, and the tile is sized to the tallest
    caption in the row — the alternative (a fixed height) is the anti-pattern
    where the last line of the longest caption falls outside its own box.
    """
    n = len(cards)
    cw = (total_w - gap * (n - 1)) / n
    inner = cw - 28
    wrapped = [wrap_px(c["caption"], T_FOOT - 1, inner) for c in cards]
    for c in cards:
        # A stat tile's label is a fixed short string chosen by the caller, so a
        # label that does not fit is a bug in the caller, not something to wrap
        # around silently: the failure mode is a label bleeding into the
        # neighbouring tile, which looks like part of it.
        lw = text_w(c["label"], T_FOOT - 2, "700")
        if lw > inner:
            raise SystemExit(f"stat tile label {c['label']!r} is {lw:.0f} wide "
                             f"but the tile only has {inner:.0f} — shorten it")
    h = 76 + max(len(w) for w in wrapped) * 23 + 14
    for i, (c, lines) in enumerate(zip(cards, wrapped)):
        cx = x + i * (cw + gap)
        s.rect(cx, y, cw, h, "#f4f3ef", RING, 1, 6)
        s.text(cx + 14, y + 26, c["label"], T_FOOT - 2, MUTED, "700", letter=0.5)
        s.text(cx + 14, y + 76, c["value"], T_HERO, c.get("color", INK), "700")
        for j, ln in enumerate(lines):
            s.text(cx + 14, y + 104 + j * 23, ln, T_FOOT - 1, INK2)
    return y + h


# ==========================================================================
# 2. data — every number comes from results/, none from a summary
# ==========================================================================
def load_rows() -> list[dict]:
    if not os.path.isfile(MASTER):
        raise SystemExit(f"missing {MASTER} — run scripts/run_experiment.py first")
    return load_master_table(MASTER)


def spine_payload(rows: Sequence[dict]) -> tuple[dict, dict]:
    """D1 for the spine arm on ITS OWN population (40 clips, 176 conditions)."""
    rep = confidence_gap_report(rows, model=SPINE_MODEL)
    return rep, plot_payload(rep)


def matched_arms_data(rows: Sequence[dict]) -> dict:
    """
    The three arms restricted to the cells EVERY arm ran, plus the census.

    `arm_intersection` is the project's own matching primitive (SPEC J.6: it
    discovers the arms from the table rather than defaulting to a hardcoded
    pair, which is how a third arm previously vanished from a complete-looking
    two-arm report). Recomputing the per-arm numbers here rather than reading
    them out of results/model_arms.json keeps the figure a pure function of
    master.csv; the test asserts the two agree.
    """
    inter = arm_intersection(rows)
    census = inter["census"]
    out = {"census": census, "arms": {}}
    for model, arm_rows in sorted(inter["arms"].items()):
        table = condition_table(arm_rows)
        flags = condition_flags(table, WER_HI, CONF_PCT_HI, "wer_spoke")
        paired = [r for r in table
                  if np.isfinite(as_float(r["mean_conf"]))
                  and np.isfinite(as_float(r["wer_spoke"]))]
        out["arms"][model] = {
            "n_conditions": len(table),
            "n_clips": census["n_common_clips"],
            "n_rows": len(arm_rows),
            "n_dead_zones": int(flags.sum()),
            "dead_zone_rate": float(flags.mean()),
            "n_mute": int(sum(1 for r in table if r["mute"])),
            "shape": confidence_wer_shape(paired, wer_key="wer_spoke"),
            "wer_comparable": bool(is_wer_comparable(model)),
        }
    out["threshold_ordering"] = threshold_ordering(inter["arms"])
    return out


# The same box `results/dead_zone_sensitivity.json` sweeps. Recomputed on the
# MATCHED arms because that artifact runs each arm on its own clip set (nova-3
# on 40, the others on 10) — fine for a within-arm fragility claim, a
# population mismatch for a cross-arm one.
WER_HI_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
CONF_PCT_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def threshold_ordering(arms: dict[str, Sequence[dict]]) -> dict:
    """
    Sweep both dead-zone thresholds on the matched arms and ask the only
    question that survives the sweep: is the ORDER of the arms stable even
    though the COUNT is not?
    """
    tables = {m: condition_table(r) for m, r in arms.items()}
    surf = {m: [[int(condition_flags(t, w, c, "wer_spoke").sum())
                 for c in CONF_PCT_GRID] for w in WER_HI_GRID]
            for m, t in tables.items()}
    n_pts = len(WER_HI_GRID) * len(CONF_PCT_GRID)
    spine, base = SPINE_MODEL, "whisper-base"
    wins = 0
    if spine in surf and base in surf:
        wins = sum(1 for i in range(len(WER_HI_GRID))
                   for j in range(len(CONF_PCT_GRID))
                   if surf[spine][i][j] < surf[base][i][j])
    flat = [v for row in surf.get(spine, []) for v in row]
    return {
        "n_grid_points": n_pts,
        "spine": spine, "baseline": base,
        "spine_strictly_below_baseline_at": wins,
        "spine_count_min": min(flat) if flat else None,
        "spine_count_max": max(flat) if flat else None,
        "wer_hi_grid": list(WER_HI_GRID), "conf_pct_hi_grid": list(CONF_PCT_GRID),
    }


def threshold_fragility(model: str = SPINE_MODEL) -> dict:
    """
    How far the dead-zone COUNT moves across the swept threshold box, for one
    arm on its OWN population.

    Read from the artifact rather than typed into the caption, because a
    hardcoded "0 to 86" is a claim about a number with nothing checking it —
    the drift J.7 found in the demo script, one file over.

    Deliberately NOT used for anything cross-arm: this artifact runs each arm
    on its own clip set (nova-3 on 40, the others on 10), which is right for a
    within-arm fragility statement and a population mismatch for a comparison.
    The cross-arm ordering claim is recomputed in `threshold_ordering`.
    """
    if not os.path.isfile(DZ_SENSITIVITY_JSON):
        return {}
    with open(DZ_SENSITIVITY_JSON, encoding="utf-8") as fh:
        arm = json.load(fh)["per_model"][model]
    st = arm["count_stats"]
    return {"min": int(st["min"]), "max": int(st["max"]),
            "n_grid_points": int(st["n_grid_points"]),
            "count_at_default": int(arm["count_at_default"]),
            "verdict": st.get("verdict")}


def fingerprint_data() -> dict:
    """Typed-edit composition for the spine arm, from results/fingerprints.json."""
    if not os.path.isfile(FINGERPRINTS_JSON):
        raise SystemExit(f"missing {FINGERPRINTS_JSON} — run "
                         f"`python -m deadzone.analysis.fingerprints` first")
    with open(FINGERPRINTS_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    arm = payload["by_model"][SPINE_MODEL]
    sigs = [{
        "family": s["family"], "label": s["label"],
        "sub": s["rates_degraded"]["sub"], "del": s["rates_degraded"]["del"],
        "ins": s["rates_degraded"]["ins"],
        "dominant": s["dominant_edit"], "delta": s["delta"],
        "degrades": bool(s["degrades"]), "n_degraded": s["n_degraded"],
    } for s in arm["signatures"]]
    return {"n_clip_rows_used": arm["n_clip_rows_used"], "signatures": sigs}


def overall_composition(rows: Sequence[dict], model: str) -> dict:
    """
    Grand totals summed straight off the table — the denominator is reference
    WORDS, not rows, so a long clip counts for more than a short one. `wer` is
    not used here at all: this is a composition, not an average of averages.
    """
    sub = sum(int(r["n_sub"]) for r in rows if r["model"] == model
              and not r.get("failed"))
    dele = sum(int(r["n_del"]) for r in rows if r["model"] == model
               and not r.get("failed"))
    ins = sum(int(r["n_ins"]) for r in rows if r["model"] == model
              and not r.get("failed"))
    ref = sum(int(r["n_ref"]) for r in rows if r["model"] == model
              and not r.get("failed"))
    if ref <= 0:
        raise SystemExit(f"no reference words for {model} — refusing to divide")
    return {"sub": sub / ref, "del": dele / ref, "ins": ins / ref,
            "n_ref": ref,
            "n_rows": sum(1 for r in rows if r["model"] == model
                          and not r.get("failed"))}


def hallucination_row() -> dict:
    """
    The one exhibited row, read out of master.csv by (clip, condition, model).

    THE COUNTING NOTE THAT HAS TO TRAVEL WITH THIS FIGURE. Elsewhere this row is
    published as "3 ref words -> 49 hyp words" (report/writeup.md 6.6 and D.9,
    results/model_arms.json). That ratio is computed by
    `model_arms.hallucination_report`, which normalizes both sides with
    `cross_model_normalize` — mapping the spoken number words to DIGITS — and
    then tokenizes with `_WORD = re.compile(r"[a-z']+")`, which matches letters
    only. So the eight digit words of the reference are created and then
    discarded, leaving 3 of 11. The hypothesis contains no digits, so it loses
    nothing, and the 16.3x ratio is part model and part tokenizer.
    Both counts are computed here and both are recorded in figures.json; the
    FIGURE draws the strict alignment (11 -> 47), which is the one the scorer
    that produced `wer` actually used.
    """
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as fh:
        refs = {r["id"]: r["ground_truth"] for r in csv.DictReader(fh)}
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    row = None
    with open(MASTER, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["model"] == HALLU_MODEL and r["clip_id"] == HALLU_CLIP
                    and r["condition_name"] == HALLU_COND):
                row = r
                break
    if row is None:
        raise SystemExit(f"{HALLU_MODEL}/{HALLU_CLIP}/{HALLU_COND} not in master.csv")
    ref, hyp = refs[HALLU_CLIP], row["transcript"]
    word = re.compile(r"[a-z']+")
    return {
        "clip_id": HALLU_CLIP, "condition_name": HALLU_COND,
        "model": HALLU_MODEL,
        "reference": ref, "transcript": hyp,
        "n_ref_words": len(ref.split()), "n_hyp_words": len(hyp.split()),
        "len_ratio": len(hyp.split()) / max(len(ref.split()), 1),
        "n_ref": int(row["n_ref"]), "n_match": int(row["n_match"]),
        "n_sub": int(row["n_sub"]), "n_del": int(row["n_del"]),
        "n_ins": int(row["n_ins"]), "wer": float(row["wer"]),
        "mean_conf": float(row["mean_conf"]),
        "n_repeats": len(re.findall(HALLU_PHRASE, hyp.lower())),
        "phrase": HALLU_PHRASE,
        # the alternative counting, computed rather than asserted
        "n_ref_tokens_crossmodel": len(word.findall(cross_model_normalize(ref))),
        "n_hyp_tokens_crossmodel": len(word.findall(cross_model_normalize(hyp))),
    }


# ==========================================================================
# 3. FIGURE 1 — the hero: confidence vs error
# ==========================================================================
def fig_confidence(pay: dict, rep: dict, frag: dict) -> tuple[str, dict]:
    pts = [p for p in pay["points"]
           if np.isfinite(p["x_mean_conf"]) and np.isfinite(p["y_wer"])]
    dz = sorted((p for p in pts if p["dead_zone"]),
                key=lambda p: -p["x_mean_conf"])
    mute = pay["mute_conditions"]
    gs, corr = rep["gap_summary"], rep["correlation"]
    n_over = int(round(gs["frac_overconfident"] * gs["n"]))
    conf_hi, wer_hi = pay["quadrant"]["conf_hi_raw"], pay["quadrant"]["wer_hi"]
    n_clips = max((p["n_clips"] or 0) for p in pts)

    sweep = (f"Sweeping both thresholds over their defensible box moves that "
             f"count from {frag['min']} to {frag['max']} across "
             f"{frag['n_grid_points']} settings, so read the count as a "
             f"presentation choice and the correlation and the gap as the "
             f"findings." if frag else
             "The count moves with the thresholds; the correlation and the gap "
             "do not.")
    notes = [
        f"The dashed box is an OPERATING POINT — WER ≥ {wer_hi:.2f} and "
        f"confidence in this model's own top "
        f"{100 * (1 - pay['quadrant']['conf_pct_hi']):.0f}% — not a "
        f"measurement. {sweep}",
        "gap = mean confidence − (1 − WER), with both halves averaged over the "
        "SAME clips. Source: results/master.csv via "
        "deadzone.analysis.confidence_gap.",
    ]

    L, R, TOP, BOT = 112, 868, 166, 456
    x0, x1 = 0.40, 1.00
    strip_y = BOT + 178
    cards = [
        {"label": "SPEARMAN RHO",
         "value": f"{corr['spearman_confpct_vs_wer']:.3f}", "color": BLUE,
         "caption": f"confidence rank vs error rank, over "
                    f"{corr['n']} conditions"},
        {"label": "OVERCONFIDENT",
         "value": f"{n_over}/{gs['n']}", "color": RED_DEEP,
         "caption": f"conditions above the line — mean gap +{gs['mean']:.3f}"},
        {"label": "NOT ON THIS CHART", "value": str(len(mute)), "color": INK,
         "caption": "conditions returned nothing on every clip: no confidence "
                    "exists, so a confidence monitor is blind to them"},
    ]
    H = int(strip_y + card_height(cards, W - 60) + 34 + footer_height(notes) + 18)

    s = Svg(W, H, "Does the model know when it is wrong?",
            f"Per-condition mean word confidence against word error rate for "
            f"{SPINE_MODEL}. {len(pts)} conditions with words, "
            f"{n_over} of them overconfident.")

    s.text(30, 54, "Does the model know when it's wrong?", T_TITLE, INK, "700")
    s.text(30, 88, "Mostly yes — which is what makes the exceptions dangerous.",
           T_SUB, INK2)
    s.text(30, 120,
           f"nova-3  ·  176 conditions × {n_clips} clips  ·  one dot = one "
           f"condition", T_FOOT, MUTED)

    def px(v): return L + (v - x0) / (x1 - x0) * (R - L)
    def py(v): return BOT - v * (BOT - TOP)

    # the "overconfident" half-plane: everything ABOVE wer = 1 - conf
    s.poly([(px(x0), py(1 - x0)), (px(x1), py(1 - x1)),
            (px(x1), TOP), (px(x0), TOP)], RED, 0.06)

    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        s.line(L, py(v), R, py(v), GRID, 1)
        s.text(L - 12, py(v) + 7, f"{v:.1f}", T_TICK, MUTED, anchor="end")
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.text(px(v), BOT + 30, f"{v:.1f}", T_TICK, MUTED, anchor="middle")
    s.line(L, BOT, R, BOT, AXIS, 1.5)
    s.line(L, TOP, L, BOT, AXIS, 1.5)

    # ---- the dead-zone box: an OPERATING POINT, labelled as one -----------
    s.rect(px(conf_hi), TOP, R - px(conf_hi), py(wer_hi) - TOP, RED, RED_DEEP,
           1.5, 3, 0.14, "6 5")
    s.text(px(conf_hi) + 14, TOP + 30, "DEAD ZONE", T_ANNO, RED_DEEP, "700")
    s.text(px(conf_hi) + 14, TOP + 54, "confident AND wrong",
           T_FOOT, RED_DEEP)

    # ---- the calibration diagonal, labelled in the empty wedge below it --
    s.line(px(x0), py(1 - x0), px(x1), py(1 - x1), INK, 2, "7 5", 0.5)
    s.text(px(x0) + 10, py(0.30),
           "perfect calibration  (confidence = accuracy)", T_FOOT, INK2)

    # ---- the points -------------------------------------------------------
    for p in pts:
        if not p["dead_zone"]:
            s.circle(px(p["x_mean_conf"]), py(p["y_wer"]), 5.5, BLUE,
                     opacity=0.45)

    # The two dead zones are ringed in the plot and named in full underneath.
    # Nothing is written next to them: at this scale the only clear space near
    # those two points is occupied by the cloud, and a label drawn over the
    # data would cost more than the leader line saves.
    for p in dz:
        cx, cy = px(p["x_mean_conf"]), py(p["y_wer"])
        s.circle(cx, cy, 11, "none", SURFACE, 5)
        s.circle(cx, cy, 9, RED_DEEP)

    # ---- axis titles ------------------------------------------------------
    s.text((L + R) / 2, BOT + 64, "mean word confidence returned by the model",
           T_AXIS, INK, "600", anchor="middle")
    s.text(40, (TOP + BOT) / 2, "word error rate", T_AXIS, INK, "600",
           anchor="middle", rotate=-90)

    # ---- legend (identity is never colour alone) -------------------------
    ly = BOT + 96
    lx = 30.0
    s.rect(lx, ly - 13, 26, 15, RED, RING, 1, 2, 0.18)
    s.text(lx + 34, ly, "overconfident region", T_FOOT, INK2)
    lx += 34 + text_w("overconfident region", T_FOOT) + 34
    s.circle(lx + 8, ly - 6, 6, BLUE, opacity=0.55)
    s.text(lx + 24, ly, f"condition ({len(pts)} that produced words)",
           T_FOOT, INK2)
    lx += 24 + text_w(f"condition ({len(pts)} that produced words)", T_FOOT) + 34
    s.circle(lx + 8, ly - 6, 7, RED_DEEP)
    s.text(lx + 24, ly, f"dead zone ({len(dz)})", T_FOOT, INK2)

    for k, p in enumerate(dz):
        yy = ly + 34 + k * 27
        s.circle(38, yy - 7, 7, RED_DEEP)
        head = f"conf {p['x_mean_conf']:.3f} at WER {p['y_wer']:.3f}"
        s.text(54, yy, head, T_FOOT, RED_DEEP, "700")
        s.text(54 + text_w(head, T_FOOT, "700") + 14, yy,
               f"rt60 {p['rt60']:g} s · SNR {p['snr_db']:g} dB · "
               f"{p['noise_type']} · {p['codec']} · rolloff "
               f"{p['mic_rolloff']:g}", T_FOOT, INK2)

    # ---- the three numbers, incl. the one that cannot be plotted ---------
    footer(s, stat_cards(s, 30, strip_y, W - 60, cards) + 34, notes)

    return s.render(), {
        "population": f"{SPINE_MODEL}, {n_clips} clips, 176 conditions",
        "n_points_plotted": len(pts),
        "n_conditions": 176,
        "n_clips": n_clips,
        "spearman": corr["spearman_confpct_vs_wer"],
        "spearman_n": corr["n"],
        "n_overconfident": n_over,
        "n_gap_conditions": gs["n"],
        "mean_gap": gs["mean"],
        "n_dead_zones": len(dz),
        "n_mute": len(mute),
        "conf_hi_raw": conf_hi, "wer_hi": wer_hi,
        "conf_pct_hi": pay["quadrant"]["conf_pct_hi"],
        "threshold_fragility": frag,
        "dead_zones": [{"condition_name": p["condition_name"],
                        "mean_conf": p["x_mean_conf"], "wer_spoke": p["y_wer"],
                        "gap": p["gap"]} for p in dz],
    }


# ==========================================================================
# 4. FIGURE 2 — the threshold-free headline: the gap distribution
# ==========================================================================
def fig_overconfidence(pay: dict, rep: dict) -> tuple[str, dict]:
    gaps = np.array([p["gap"] for p in pay["points"]
                     if np.isfinite(p.get("gap", np.nan))], dtype=float)
    gs = rep["gap_summary"]
    n_over = int((gaps > 0).sum())
    n_not = int((gaps <= 0).sum())
    n_clips = max((p["n_clips"] or 0) for p in pay["points"])

    lo, hi, nb = -0.10, 0.80, 18
    edges = np.linspace(lo, hi, nb + 1)
    counts, _ = np.histogram(np.clip(gaps, lo, hi - 1e-9), bins=edges)

    notes = [
        "No operating point is involved in this chart. Every condition that "
        "produced a word is here, and the claim is the shape of the "
        "distribution — not a count that moves when a threshold does.",
        "gap = mean word confidence − (1 − WER), with the confidence and the "
        "WER averaged over the SAME clips. Pairing a spoke-only confidence "
        f"against an all-clips WER would read +"
        f"{gs['all_clips_pairing']['mean']:.3f} instead — that mismatch is the "
        "defect this project found in its own headline (SPEC Appendix G).",
        "Source: results/master.csv via deadzone.analysis.confidence_gap "
        "(gap_spoke).",
    ]

    L, R, TOP, BOT = 108, 868, 196, 452
    strip_y = BOT + 100
    cards = [
        {"label": "OVERCONFIDENT", "value": f"{n_over}", "color": RED_DEEP,
         "caption": f"of {len(gaps)} conditions ({100 * n_over / len(gaps):.0f}%) "
                    f"— the model claimed more accuracy than it delivered"},
        {"label": "WELL / UNDER-CONFIDENT", "value": f"{n_not}", "color": BLUE,
         "caption": f"of {len(gaps)} conditions — the safe side of zero"},
    ]
    H = int(strip_y + card_height(cards, W - 60) + 34 + footer_height(notes) + 18)

    s = Svg(W, H, "How overconfident, without picking a threshold",
            f"Distribution of the per-condition confidence-minus-accuracy gap "
            f"for {SPINE_MODEL}: {n_over} of {len(gaps)} conditions overconfident.")

    s.text(30, 54, "How overconfident — with no threshold at all", T_TITLE,
           INK, "700")
    s.text(30, 88, "confidence minus observed accuracy, one value per condition",
           T_SUB, INK2)
    s.text(30, 120,
           f"nova-3  ·  {n_clips} clips  ·  {len(gaps)} of 176 conditions "
           f"(the other 7 never spoke at all, so they carry no gap)",
           T_FOOT, MUTED)

    def px(v): return L + (v - lo) / (hi - lo) * (R - L)

    ytop = int(np.ceil(max(int(counts.max()), 1) / 10.0) * 10)

    def py(c): return BOT - c / ytop * (BOT - TOP)

    for c in range(0, ytop + 1, 10):
        s.line(L, py(c), R, py(c), GRID, 1)
        s.text(L - 12, py(c) + 7, str(c), T_TICK, MUTED, anchor="end")

    bw = (R - L) / nb
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        mid = (edges[i] + edges[i + 1]) / 2
        s.rect(px(edges[i]) + 1, py(c), bw - 2, BOT - py(c),
               RED if mid > 0 else BLUE, rx=3)

    s.line(L, BOT, R, BOT, AXIS, 1.5)
    for v in (-0.1, 0.0, 0.2, 0.4, 0.6, 0.8):
        s.text(px(v), BOT + 30, f"{v:.1f}", T_TICK, MUTED, anchor="middle")

    # zero: the only line that means anything here
    s.line(px(0), TOP - 30, px(0), BOT + 6, INK, 2)
    s.text(px(0) - 12, TOP - 38, "well calibrated", T_FOOT, INK, "700",
           anchor="end")
    s.text(px(0) + 12, TOP - 38, "overconfident  →", T_FOOT, RED_DEEP, "700")

    # the mean, direct-labelled (never a number on every bar)
    s.line(px(gs["mean"]), TOP + 6, px(gs["mean"]), BOT, RED_DEEP, 2, "6 4")
    s.text(px(gs["mean"]) + 10, TOP + 24, f"mean  +{gs['mean']:.3f}",
           T_FOOT, RED_DEEP, "700")
    s.text(px(gaps.max()) - 10, py(1) - 10, f"worst  +{gaps.max():.3f}",
           T_FOOT, INK2, anchor="end")

    s.text((L + R) / 2, BOT + 66,
           "confidence − accuracy   (positive = it claimed more than it delivered)",
           T_AXIS, INK, "600", anchor="middle")
    s.text(42, (TOP + BOT) / 2, "conditions", T_AXIS, INK, "600",
           anchor="middle", rotate=-90)

    footer(s, stat_cards(s, 30, strip_y, W - 60, cards) + 34, notes)

    return s.render(), {
        "population": f"{SPINE_MODEL}, {n_clips} clips, "
                      f"{len(gaps)} of 176 conditions",
        "n_gap_conditions": int(len(gaps)),
        "n_overconfident": n_over,
        "n_not_overconfident": n_not,
        "mean_gap": gs["mean"],
        "max_gap": float(gaps.max()),
        "min_gap": float(gaps.min()),
        "mean_gap_all_clips_pairing": gs["all_clips_pairing"]["mean"],
        "frac_overconfident": n_over / len(gaps),
    }


# ==========================================================================
# 5. FIGURE 3 — typed-edit fingerprints
# ==========================================================================
def fig_fingerprints(fp: dict, overall: dict) -> tuple[str, dict]:
    sigs = sorted(fp["signatures"], key=lambda d: -d["del"])
    rows = [{"label": "ALL 176 conditions", "sub": overall["sub"],
             "del": overall["del"], "ins": overall["ins"],
             "note": f"{overall['n_ref']:,} reference words", "head": True}]
    for sg in sigs:
        rows.append({
            "label": _family_label(sg),
            "sub": sg["sub"], "del": sg["del"], "ins": sg["ins"],
            "note": ("substitutions" if sg["dominant"] == "sub"
                     else "deletions"),
            "head": False, "dominant": sg["dominant"],
        })

    n_sub_dom = sum(1 for sg in sigs if sg["dominant"] == "sub")
    sub_names = ", ".join(_family_short(sg) for sg in sigs
                          if sg["dominant"] == "sub")
    notes = [
        "The header row is every nova-3 row in the grid. Each row below it is "
        "that family's DEGRADED half against its own clean half (e.g. SNR ≤ 5 dB "
        "vs ≥ 10 dB) — separate populations, which is why they do not sum to the "
        "header row.",
        f"{len(sigs) - n_sub_dom} of {len(sigs)} families are deletion-dominant, "
        f"and the two readings imply opposite fixes: a word the acoustic model "
        f"never emitted cannot be recovered by keyword boosting, only by "
        f"front-end work. The {n_sub_dom} substitution-dominant families "
        f"({sub_names}) are the ones where boosting or entity-aware decoding "
        f"actually helps.",
        "Source: results/fingerprints.json (rates_degraded) for the family rows; "
        "a direct sum of results/master.csv for the header row.",
    ]

    rowh, gap = 42, 8
    TOPR = 232
    plot_h = len(rows) * (rowh + gap) - gap
    H = int(TOPR + plot_h + 62 + footer_height(notes) + 18)

    s = Svg(W, H, "Which kind of error, not how many",
            "Typed-edit composition as a fraction of reference words, per "
            "degradation family, for nova-3.")

    s.text(30, 54, "Which KIND of error, not how many", T_TITLE, INK, "700")
    s.text(30, 88, "Deletions are not one mechanism among several — they are "
                   "the failure mode.", T_SUB, INK2)
    s.text(30, 120,
           f"nova-3  ·  {fp['n_clip_rows_used']:,} clip-rows  ·  40 clips  ·  "
           f"edits as a fraction of reference words", T_FOOT, MUTED)

    L, R = 316, 720
    scale = 0.75          # bar axis runs 0 -> 0.75 edits per reference word

    def bx(v): return L + v / scale * (R - L)

    # legend — always present for >= 2 series; aqua is under 3:1 on this
    # surface, so every insertion value is also direct-labelled (relief rule)
    lx = 30.0
    for name, col in (("deletions", BLUE), ("substitutions", ORANGE),
                      ("insertions", AQUA)):
        s.rect(lx, 148, 24, 15, col, rx=3)
        s.text(lx + 32, 161, name, T_FOOT, INK2)
        lx += 32 + text_w(name, T_FOOT) + 30

    for v in (0.0, 0.25, 0.5, 0.75):
        s.text(bx(v), TOPR - 12, f"{v:.2f}", T_FOOT, MUTED, anchor="middle")
        s.line(bx(v), TOPR - 4, bx(v), TOPR + plot_h, GRID, 1)

    y = TOPR
    for r in rows:
        mid = y + rowh / 2 + 7
        if r["head"]:
            s.rect(24, y - 4, W - 48, rowh + 8, "#eef4fc", "#cfe0f5", 1, 5)
        s.text(L - 16, mid, r["label"], T_ANNO, INK,
               "700" if r["head"] else "400", anchor="end")
        x = bx(0.0)
        for key, col in (("del", BLUE), ("sub", ORANGE), ("ins", AQUA)):
            seg = bx(r[key]) - bx(0.0)
            if seg > 0.5:
                s.rect(x, y + 9, max(seg - 2, 0.5), rowh - 18, col, rx=2)
            x += seg
        # label inside a segment ONLY where it fits with padding; otherwise the
        # value moves to the bar end. A clipped in-bar label is an anti-pattern.
        dseg = bx(r["del"]) - bx(0)
        if dseg > text_w("0.00", T_FOOT, "700") + 18:
            s.text(bx(0) + dseg / 2, mid, f"{r['del']:.2f}", T_FOOT,
                   "#ffffff", "700", anchor="middle")
        sseg = bx(r["sub"]) - bx(0)
        if sseg > text_w("0.00", T_FOOT, "700") + 18:
            s.text(bx(r["del"]) + sseg / 2, mid, f"{r['sub']:.2f}", T_FOOT,
                   "#ffffff", "700", anchor="middle")
        s.text(x + 12, mid, f"ins {r['ins']:.3f}", T_FOOT - 1,
               INK2 if r["head"] else MUTED)
        y += rowh + gap

    s.text((L + R) / 2, y + 44, "edits per reference word", T_AXIS, INK, "600",
           anchor="middle")

    footer(s, y + 84, notes)

    return s.render(), {
        "population": f"{SPINE_MODEL}, 40 clips, 176 conditions, "
                      f"{overall['n_ref']} reference words",
        "overall": {k: overall[k] for k in ("sub", "del", "ins", "n_ref", "n_rows")},
        "n_families": len(sigs),
        "n_deletion_dominant": len(sigs) - n_sub_dom,
        "n_substitution_dominant": n_sub_dom,
        "families": [{"family": sg["family"], "label": sg["label"],
                      "sub": sg["sub"], "del": sg["del"], "ins": sg["ins"],
                      "dominant": sg["dominant"]} for sg in sigs],
    }


_FAMILY_LABEL = {
    "snr_db": "noise level  (SNR ≤ 5 dB)",
    "mic_rolloff": "mic rolloff  (full)",
    "rt60": "reverb  (RT60 ≥ 0.7 s)",
    "codec=opus-lowrate": "codec  opus-lowrate",
    "codec=g726": "codec  G.726",
    "codec=none": "codec  none",
    "noise_type=babble": "noise  babble",
    "noise_type=engine": "noise  engine",
    "noise_type=road": "noise  road",
}
_FAMILY_SHORT = {"codec=g726": "G.726", "noise_type=road": "road noise",
                 "codec=opus-lowrate": "opus-lowrate",
                 "noise_type=babble": "babble", "noise_type=engine": "engine",
                 "codec=none": "no codec", "snr_db": "noise level",
                 "rt60": "reverb", "mic_rolloff": "mic rolloff"}


def _family_label(sg: dict) -> str:
    return _FAMILY_LABEL.get(sg["family"], sg["family"])


def _family_short(sg: dict) -> str:
    return _FAMILY_SHORT.get(sg["family"], sg["family"])


# ==========================================================================
# 6. FIGURE 4 — the three arms, on the population where they are comparable
# ==========================================================================
ARM_COLOR = {"nova-3": BLUE, "elevenlabs-scribe": ORANGE,
             "whisper-base": RED}
ARM_KIND = {"nova-3": "commercial", "elevenlabs-scribe": "commercial",
            "whisper-base": "open"}


def fig_models(md: dict) -> tuple[str, dict]:
    cen, arms = md["census"], md["arms"]
    order = [m for m in (SPINE_MODEL, "elevenlabs-scribe", "whisper-base")
             if m in arms] + [m for m in sorted(arms) if m not in
                              (SPINE_MODEL, "elevenlabs-scribe", "whisper-base")]
    to = md["threshold_ordering"]

    banner = [
        f"POPULATION — the {cen['n_common_clips']}-clip intersection: the "
        f"{cen['n_common_cells']:,} cells all three arms actually ran "
        f"({cen['n_common_conditions']} conditions × {cen['n_common_clips']} "
        f"clips per arm).",
        f"nova-3 also ran 40 clips, and on THAT population its dead-zone rate "
        f"reads 1.14% (2 of 176), not the "
        f"{100 * arms[SPINE_MODEL]['dead_zone_rate']:.2f}% below. Both are "
        f"correct; quoting one without its clip count is the bug.",
    ]
    per_arm_n = " / ".join(f"{m} {arms[m]['shape']['n']}" for m in order)
    notes = [
        f"The rho row is over a DIFFERENT condition count per arm "
        f"({per_arm_n}) because each arm goes mute on a different set of "
        f"conditions. The clip set is shared; the condition population is not.",
        f"The dead-zone COUNT is threshold-fragile — but the ORDER is not: "
        f"{to['spine']} is strictly below {to['baseline']} at all "
        f"{to['spine_strictly_below_baseline_at']} of {to['n_grid_points']} "
        f"threshold pairs swept on this same population "
        f"(WER ≥ 0.10…0.50 × confidence percentile 0.30…0.90).",
        "Source: results/master.csv via "
        "deadzone.analysis.model_arms.arm_intersection, which takes the "
        "intersection and returns the census — never a cross-arm mean over "
        "unmatched clips.",
    ]

    L, R = 322, 700
    rowh = 46
    banner_lines = [wrap_px(b, T_FOOT, W - 88) for b in banner]
    banner_h = sum(len(b) for b in banner_lines) * 25 + 24

    s = Svg(W, 10, "", "")     # placeholder; rebuilt below once H is known

    # --- measure the layout before drawing so nothing lands off-canvas -----
    y_banner = 106
    y_a = y_banner + banner_h + 40
    y_a_rows = y_a + 62
    y_a_end = y_a_rows + 8 + len(order) * rowh
    scribe_note = wrap_px(
        "Why elevenlabs-scribe has no bar: it returns a DIFFERENT transcript for "
        "byte-identical audio on repeat calls (A7X42 vs “A seven X four two”), "
        "and a dead zone is defined by an ABSOLUTE WER threshold — so its 7 "
        "strict dead zones fall to 0 under the cross-model normalizer. They were "
        "spelling, not confident error. Rank statistics survive that noise, "
        "attenuated, so its rho in panel B is a lower bound.",
        T_FOOT, W - 60)
    y_b = y_a_end + len(scribe_note) * 24 + 56
    y_b_rows = y_b + 62
    y_b_end = y_b_rows + 8 + len(order) * rowh
    H = int(y_b_end + 46 + footer_height(notes) + 18)

    s = Svg(W, H, "Three ASR arms on the cells all three ran",
            "Dead-zone rate and confidence-vs-error rank correlation per arm, "
            "on the matched 10-clip intersection.")

    s.text(30, 54, "Three arms — on the cells all three actually ran", T_TITLE,
           INK, "700")
    s.text(30, 88,
           "Whisper is not merely worse. It is worse at knowing it is worse.",
           T_SUB, INK2)

    # the population banner is the point of the figure, so it is a banner
    s.rect(30, y_banner, W - 60, banner_h, "#eef4fc", "#cfe0f5", 1, 5)
    yy = y_banner + 30
    for i, lines in enumerate(banner_lines):
        for ln in lines:
            s.text(44, yy, ln, T_FOOT, INK if i == 0 else INK2,
                   "700" if i == 0 else "400")
            yy += 25

    # ---- panel A: dead-zone rate -----------------------------------------
    s.text(30, y_a, "A.  Dead-zone rate — conditions that are confident AND wrong",
           T_AXIS, INK, "700")
    s.text(30, y_a + 28,
           "share of the 176 conditions at WER ≥ 0.30 with confidence in that "
           "arm's own top 40%", T_FOOT, MUTED)

    amax = 0.45

    def bx(v): return L + v / amax * (R - L)

    for v in (0.0, 0.1, 0.2, 0.3, 0.4):
        s.text(bx(v), y_a_rows - 8, f"{100 * v:.0f}%", T_FOOT, MUTED,
               anchor="middle")
        s.line(bx(v), y_a_rows, bx(v), y_a_end, GRID, 1)

    y = y_a_rows + 8
    for m in order:
        a = arms[m]
        mid = y + rowh / 2 + 7
        s.text(L - 16, mid, m, T_ANNO, INK, "600", anchor="end")
        if a["wer_comparable"]:
            bw = max(bx(a["dead_zone_rate"]) - bx(0), 3)
            s.rect(bx(0), y + 10, bw, rowh - 20, ARM_COLOR[m], rx=3)
            s.text(bx(0) + bw + 14, mid,
                   f"{100 * a['dead_zone_rate']:.2f}%   "
                   f"({a['n_dead_zones']} of {a['n_conditions']})",
                   T_FOOT, INK, "700")
        else:
            s.rect(bx(0), y + 10, R - bx(0), rowh - 20, "none", MUTED, 1.5, 3,
                   dash="5 5")
            s.text(bx(0) + 14, mid, "NOT QUOTABLE — see the note below",
                   T_FOOT, INK2, "700")
        y += rowh

    yy = y_a_end + 30
    for ln in scribe_note:
        s.text(30, yy, ln, T_FOOT, MUTED)
        yy += 24

    # ---- panel B: rank correlation ---------------------------------------
    s.text(30, y_b, "B.  Does confidence track error?   Spearman rho — more "
                    "negative is better", T_AXIS, INK, "700")
    s.text(30, y_b + 28,
           "rank of that arm's own confidence against rank of its WER, over its "
           "non-mute conditions", T_FOOT, MUTED)

    rmin = -1.0

    def rx(v): return L + (v - rmin) / (0 - rmin) * (R - L)

    for v in (-1.0, -0.8, -0.6, -0.4, -0.2, 0.0):
        s.text(rx(v), y_b_rows - 8, f"{v:.1f}", T_FOOT, MUTED, anchor="middle")
        s.line(rx(v), y_b_rows, rx(v), y_b_end, GRID, 1)

    y = y_b_rows + 8
    for m in order:
        a = arms[m]
        rho = a["shape"]["spearman"]
        mid = y + rowh / 2 + 7
        s.text(L - 16, mid, m, T_ANNO, INK, "600", anchor="end")
        s.rect(rx(rho), y + 10, rx(0) - rx(rho), rowh - 20, ARM_COLOR[m], rx=3)
        s.text(rx(0) + 14, mid, f"{rho:.3f}   n = {a['shape']['n']}",
               T_FOOT, INK, "700")
        y += rowh

    footer(s, y_b_end + 46, notes)

    return s.render(), {
        "population": (f"matched {cen['n_common_clips']}-clip intersection, "
                       f"{cen['n_common_conditions']} conditions, "
                       f"{cen['n_common_cells']} rows per arm"),
        "census": {k: cen[k] for k in
                   ("n_arms", "n_common_cells", "n_common_clips",
                    "n_common_conditions")},
        "arms": {m: {"dead_zone_rate": arms[m]["dead_zone_rate"],
                     "n_dead_zones": arms[m]["n_dead_zones"],
                     "n_conditions": arms[m]["n_conditions"],
                     "spearman": arms[m]["shape"]["spearman"],
                     "spearman_n": arms[m]["shape"]["n"],
                     "n_mute": arms[m]["n_mute"],
                     "wer_comparable": arms[m]["wer_comparable"],
                     "dead_zone_rate_drawn": arms[m]["wer_comparable"]}
                 for m in order},
        "threshold_ordering": to,
    }


# ==========================================================================
# 7. FIGURE 5 — the hallucination exhibit
# ==========================================================================
def fig_hallucination(h: dict) -> tuple[str, dict]:
    fs = 19.5
    cw = fs * MONO_ADV
    BOXL, BOXR = 30, W - 30
    chars = int((BOXR - BOXL - 32) / cw)

    ref_lines = wrap_px(h["reference"], fs, BOXR - BOXL - 32, mono=True)
    hyp_lines, spans = _wrap_with_spans(h["transcript"], chars)

    n_digit_words = h["n_ref_words"] - h["n_ref_tokens_crossmodel"]
    notes = [
        f"WER caps damage at one error per reference word, so it reports "
        f"{h['wer']:.2f} for a transcript {h['len_ratio']:.1f}× too long. Fed to "
        f"a downstream agent, an invented sentence is unbounded harm — which is "
        f"the argument for why WER is not the deployment metric. Under the same "
        f"class of stress nova-3 goes silent instead; a silent transcript at "
        f"least announces itself.",
        f"mean word confidence on this row: {h['mean_conf']:.3f}. COUNTING NOTE — "
        f"this row is published elsewhere as “{h['n_ref_tokens_crossmodel']} ref "
        f"words → {h['n_hyp_tokens_crossmodel']} hyp words” (16.3×). That ratio "
        f"is measured after a normalizer rewrites the spoken numbers as digits "
        f"and a letters-only tokenizer then drops them, so {n_digit_words} of "
        f"the {h['n_ref_words']} spoken words disappear from the reference. The "
        f"alignment drawn above is the one that produced the stored WER.",
    ]

    y_ref_label = 178
    y_ref_box = y_ref_label + 12
    ref_h = 22 + len(ref_lines) * 27
    y_hyp_label = y_ref_box + ref_h + 34
    y_hyp_box = y_hyp_label + 12
    hyp_h = 22 + len(hyp_lines) * 27
    y_loop = y_hyp_box + hyp_h + 30
    y_bars = y_loop + 34
    y_strip = y_bars + 108
    cards = [
        {"label": "REFERENCE WORDS KEPT",
         "value": f"{h['n_match']} of {h['n_ref']}", "color": RED_DEEP,
         "caption": f"{h['n_sub']} were substituted, {h['n_del']} deleted"},
        {"label": "WORDS INVENTED", "value": str(h["n_ins"]), "color": RED_DEEP,
         "caption": "insertions — unbounded, which is why WER exceeds 1.0 here"},
        {"label": "WORD ERROR RATE", "value": f"{h['wer']:.2f}",
         "color": RED_DEEP,
         "caption": f"{100 * h['wer']:.0f}% — more errors than there were words "
                    f"to get wrong"},
    ]
    H = int(y_strip + card_height(cards, W - 60) + 34 + footer_height(notes) + 18)

    s = Svg(W, H, "What a hallucinated transcript looks like",
            f"whisper-base on {h['clip_id']} at {h['condition_name']}: "
            f"{h['n_ref_words']} spoken words returned as {h['n_hyp_words']}.")

    s.text(30, 54, f"What “WER {h['wer']:.2f}” actually looks like", T_TITLE,
           INK, "700")
    s.text(30, 88, "Under stress nova-3 goes quiet. whisper-base invents.",
           T_SUB, INK2)
    s.text(30, 118,
           f"whisper-base  ·  clip {h['clip_id']}  ·  {h['condition_name']}",
           T_FOOT, MUTED)
    s.text(30, 142, "one row of results/master.csv — an exhibit, not a rate",
           T_FOOT, MUTED)

    # ---- reference --------------------------------------------------------
    s.text(BOXL, y_ref_label, f"WHAT WAS SAID  —  {h['n_ref_words']} words",
           T_FOOT, INK, "700", letter=0.6)
    s.rect(BOXL, y_ref_box, BOXR - BOXL, ref_h, "#eef4fc", "#cfe0f5", 1, 5)
    y = y_ref_box + 34
    for ln in ref_lines:
        s.text(BOXL + 16, y, ln, fs, INK, family=MONO)
        y += 27

    # ---- hypothesis -------------------------------------------------------
    s.text(BOXL, y_hyp_label,
           f"WHAT WHISPER-BASE RETURNED  —  {h['n_hyp_words']} words",
           T_FOOT, RED_DEEP, "700", letter=0.6)
    s.rect(BOXL, y_hyp_box, BOXR - BOXL, hyp_h, "#fdeeee", "#f3c9c9", 1, 5)
    y = y_hyp_box + 34
    for i, ln in enumerate(hyp_lines):
        for a, b in spans.get(i, []):
            s.rect(BOXL + 16 + a * cw - 2, y - 16, (b - a) * cw + 4, 23,
                   RED, rx=3, opacity=0.22)
        s.text(BOXL + 16, y, ln, fs, INK, family=MONO)
        y += 27

    s.text(BOXL, y_loop,
           f"“{h['phrase']}” repeats {h['n_repeats']} times — a degenerate "
           f"loop, not a mishearing", T_FOOT, RED_DEEP, "700")

    # ---- length blowup, as two bars on one scale -------------------------
    bar_l, bar_r = 210, W - 296
    for k, (label, n, col, note) in enumerate((
            ("said", h["n_ref_words"], BLUE, ""),
            ("returned", h["n_hyp_words"], RED,
             f"  —  {h['len_ratio']:.1f}× longer"))):
        by = y_bars + k * 40
        s.text(bar_l - 16, by + 20, label, T_FOOT, INK2, anchor="end")
        bw = max((bar_r - bar_l) * n / h["n_hyp_words"], 4)
        s.rect(bar_l, by + 4, bw, 24, col, rx=3)
        s.text(bar_l + bw + 12, by + 23, f"{n} words{note}", T_FOOT,
               RED_DEEP if note else INK, "700")

    # ---- the three numbers -----------------------------------------------
    footer(s, stat_cards(s, 30, y_strip, W - 60, cards) + 34, notes)

    return s.render(), {
        "population": "n = 1 row of results/master.csv — an exhibit, not a rate",
        **{k: h[k] for k in (
            "clip_id", "condition_name", "model", "reference", "transcript",
            "n_ref_words", "n_hyp_words", "len_ratio", "n_ref", "n_match",
            "n_sub", "n_del", "n_ins", "wer", "mean_conf", "n_repeats",
            "phrase", "n_ref_tokens_crossmodel", "n_hyp_tokens_crossmodel")},
    }


def _wrap_with_spans(text: str, chars: int) -> tuple[list[str], dict]:
    """
    Wrap `text` and return, per line, the character spans of HALLU_PHRASE.

    The phrase can straddle a wrap, so matches are found on the FULL string and
    then split across lines — searching each wrapped line independently would
    silently miss any occurrence that crossed a break, and the figure would
    under-count the loop it exists to show.
    """
    lines, starts, line, start = [], [], "", 0
    pos = 0
    for word in text.split():
        idx = text.index(word, pos)
        pos = idx + len(word)
        cand = f"{line} {word}".strip()
        if len(cand) > chars and line:
            lines.append(line)
            starts.append(start)
            line, start = word, idx
        else:
            if not line:
                start = idx
            line = cand
    if line:
        lines.append(line)
        starts.append(start)

    spans: dict[int, list[tuple[int, int]]] = {}
    for m in re.finditer(re.escape(HALLU_PHRASE), text.lower()):
        for i, (ln, st) in enumerate(zip(lines, starts)):
            en = st + len(ln)
            a, b = max(m.start(), st), min(m.end(), en)
            if a < b:
                spans.setdefault(i, []).append((a - st, b - st))
    return lines, spans


# ==========================================================================
# 8. driver
# ==========================================================================
def build() -> tuple[dict[str, str], dict]:
    rows = load_rows()
    rep, pay = spine_payload(rows)

    svgs, numbers = {}, {}
    svgs[FIG_CONFIDENCE], numbers[FIG_CONFIDENCE] = fig_confidence(
        pay, rep, threshold_fragility())
    svgs[FIG_OVERCONF], numbers[FIG_OVERCONF] = fig_overconfidence(pay, rep)
    svgs[FIG_FINGERPRINTS], numbers[FIG_FINGERPRINTS] = fig_fingerprints(
        fingerprint_data(), overall_composition(rows, SPINE_MODEL))
    svgs[FIG_MODELS], numbers[FIG_MODELS] = fig_models(matched_arms_data(rows))
    svgs[FIG_HALLUCINATION], numbers[FIG_HALLUCINATION] = fig_hallucination(
        hallucination_row())

    numbers = {
        "artifact": "docs/assets/figures.json",
        "purpose": ("Every number drawn in the five README figures, with the "
                    "population it is over. tests/test_readme_figures.py "
                    "re-derives each one from results/ and fails if a figure "
                    "and its artifact disagree — prose and pictures are not "
                    "covered by any other test in this repo (SPEC J.7)."),
        "generator": "scripts/make_readme_figures.py",
        "sources": ["results/master.csv", "results/fingerprints.json",
                    "results/dead_zone_sensitivity.json",
                    "recording_manifest.csv"],
        "thresholds": {"wer_hi": WER_HI, "conf_pct_hi": CONF_PCT_HI},
        "figures": numbers,
    }
    return svgs, numbers


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=OUT_DIR, help="output directory")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any file on disk differs from a fresh build")
    a = ap.parse_args(argv)

    svgs, numbers = build()
    payload = json.dumps(numbers, indent=1, sort_keys=True,
                         default=_json_safe) + "\n"
    want = dict(svgs)
    want[FIG_NUMBERS] = payload

    if a.check:
        stale = []
        for name, body in want.items():
            path = os.path.join(a.out, name)
            if not os.path.isfile(path):
                stale.append(f"{name}: MISSING")
            elif open(path, encoding="utf-8").read() != body:
                stale.append(f"{name}: STALE")
        for line in stale:
            print(f"  {line}")
        print(f"[readme-figures] {'STALE' if stale else 'up to date'} "
              f"({len(want) - len(stale)}/{len(want)} current)")
        return 1 if stale else 0

    os.makedirs(a.out, exist_ok=True)
    for name, body in want.items():
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            fh.write(body)
        print(f"  wrote {os.path.relpath(os.path.join(a.out, name), _REPO_ROOT)} "
              f"({len(body):,} bytes)")
    print(f"[readme-figures] {len(want)} files -> "
          f"{os.path.relpath(a.out, _REPO_ROOT)}")
    return 0


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serializable: {type(obj)}")


if __name__ == "__main__":
    raise SystemExit(main())
