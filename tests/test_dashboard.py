#!/usr/bin/env python3
"""
test_dashboard.py — offline tests for the E2 dashboard (SPEC A.R6).

Everything here runs with the network off and without a browser. The
demo-critical properties, in the order they would ruin a demo:

  1. IT BUILDS AT ALL, from a table that is missing analysis payloads.
  2. IT IS SELF-CONTAINED. Not one src/href points at an external host; no
     fetch/XHR; no CDN. If any of these creep in, the page dies on a laptop with
     wifi off — which is precisely the situation it was chosen for.
  3. THE EMBEDDED JSON PARSES, with no NaN (JSON.parse rejects NaN; Python's
     json.dumps emits it happily by default, so this is a real trap).
  4. A MISSING PAYLOAD RENDERS AN EXPLAINED EMPTY STATE, never a blank chart.
     Verified by actually running app.js against a minimal DOM shim under node,
     so this is a rendering assertion and not a string search.
  5. THE FILE IS STANDALONE HTML.

    python3 tests/test_dashboard.py
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

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = _REPO_ROOT

from dashboard import build as B                       # noqa: E402
from dashboard import make_synthetic as MS             # noqa: E402

SYNTH_CSV = os.path.join(REPO, "results/synthetic_master.csv")
BUILT = os.path.join(REPO, "dashboard/deadzone.html")

# Panels 1-6 are per-model and re-render when the model toggle changes.
PANEL_IDS = ["panel-hero", "panel-heatmap", "panel-fingerprints",
             "panel-sensitivity", "panel-al", "panel-sim2real"]

# Panels 7-8 are cross-cutting and deliberately do NOT change with the toggle:
# panel 7 IS the comparison between arms, and panel 8 reads sweep audio rather
# than the per-model table. They are listed separately so the toggle test does
# not demand that they re-render, which would be the wrong requirement.
CROSS_PANEL_IDS = ["panel-model-arms", "panel-decoupling"]

_NODE = shutil.which("node")


# ===========================================================================
# A minimal DOM shim, so app.js can be executed for real without a browser
# ===========================================================================

_SHIM = r"""
class N {
  constructor(tag, ns) {
    this.tagName = String(tag || "").toUpperCase(); this.ns = ns || null;
    this.children = []; this.attrs = {}; this.style = {}; this._text = "";
    this.listeners = {};
  }
  appendChild(c) { this.children.push(c); c.parent = this; return c; }
  removeChild(c) { this.children = this.children.filter(function (x) { return x !== c; }); return c; }
  get firstChild() { return this.children[0] || null; }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "id") DOC._ids[v] = this; }
  getAttribute(k) { return this.attrs[k]; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() {
    return this._text + this.children.map(function (c) { return c.textContent; }).join(" ");
  }
  set innerHTML(v) { this._text = String(v); }
  set className(v) { this.attrs["class"] = String(v); }
  get className() { return this.attrs["class"] || ""; }
  set value(v) { this.attrs["value"] = String(v); }
  get value() { return this.attrs["value"]; }
  querySelector(sel) {
    var want = sel.replace(".", ""), hit = null;
    (function walk(n) {
      if (hit) return;
      (n.children || []).forEach(function (c) {
        if (hit) return;
        if ((c.className || "").split(/\s+/).indexOf(want) >= 0) { hit = c; return; }
        walk(c);
      });
    })(this);
    return hit;
  }
  countClass(cls) {
    var n = 0;
    (function walk(node) {
      (node.children || []).forEach(function (c) {
        if ((c.className || "").split(/\s+/).indexOf(cls) >= 0) n++;
        walk(c);
      });
    })(this);
    return n;
  }
  countTag(tag) {
    var n = 0, T = tag.toUpperCase();
    (function walk(node) {
      (node.children || []).forEach(function (c) { if (c.tagName === T) n++; walk(c); });
    })(this);
    return n;
  }
  click() { (this.listeners["click"] || []).forEach(function (f) { f({}); }); }
  fire(t) { (this.listeners[t] || []).forEach(function (f) { f({}); }); }
}

var DOC = {
  _ids: {}, readyState: "complete",
  documentElement: new N("html"),
  createElement: function (t) { return new N(t); },
  createElementNS: function (ns, t) { return new N(t, ns); },
  createTextNode: function (s) { var n = new N("#text"); n._text = String(s); return n; },
  getElementById: function (id) { return DOC._ids[id] || null; },
  addEventListener: function () {}
};
DOC.body = new N("body");

globalThis.document = DOC;
globalThis.window = globalThis;
globalThis.getComputedStyle = function () {
  return { getPropertyValue: function () { return ""; } };
};

// the hosts the shell provides
["source-badge", "built-at", "model-toggle", "build-notes"].concat(PANEL_IDS_JSON)
  .forEach(function (id) {
    var n = new N("div"); n.setAttribute("id", id); DOC.body.appendChild(n);
  });
"""

_REPORT = r"""
var out = { panels: {}, models: [], badge: DOC.getElementById("source-badge").textContent };
PANEL_IDS_JSON.forEach(function (id) {
  var host = DOC.getElementById(id);
  out.panels[id] = {
    children: host.children.length,
    empties: host.countClass("empty"),
    svgs: host.countTag("svg"),
    text: host.textContent.slice(0, 400)
  };
});
var mt = DOC.getElementById("model-toggle");
var buttons = mt.children.filter(function (c) { return c.tagName === "BUTTON"; });
out.models = buttons.map(function (c) { return c.textContent.trim(); });

// exercise the interactions a demo actually uses
var hero = DOC.getElementById("panel-hero");
var circles = [];
(function walk(n) {
  (n.children || []).forEach(function (c) { if (c.tagName === "CIRCLE") circles.push(c); walk(c); });
})(hero);
out.hero_points = circles.length;
if (circles.length) circles[circles.length - 1].fire("mouseenter");
var det = hero.querySelector(".detail");
out.detail_text = det ? det.textContent.slice(0, 600) : "";

var al = DOC.getElementById("panel-al");
var alButtons = [];
(function walk(n) {
  (n.children || []).forEach(function (c) { if (c.tagName === "BUTTON") alButtons.push(c); walk(c); });
})(al);
if (alButtons.length > 1) { alButtons[1].click(); alButtons[1].click(); }
out.al_after_step = al.textContent.slice(0, 200);

if (buttons.length > 1) {
  buttons[buttons.length - 1].click();               // switch model, re-render all
  out.after_toggle = {};
  out.after_toggle_rendered = {};
  out.after_toggle_text = {};
  PANEL_IDS_JSON.forEach(function (id) {
    var el = DOC.getElementById(id);
    out.after_toggle[id] = el.countClass("empty");
    // a panel that legitimately has no data for this model must still RENDER --
    // an explanation is a pass, a blank hole is not.
    out.after_toggle_rendered[id] = (el.children || []).length;
    out.after_toggle_text[id] = el.textContent.slice(0, 400);
  });
  out.after_toggle_model = buttons[buttons.length - 1].textContent;
}
console.log("__RESULT__" + JSON.stringify(out));
"""


def _scripts_from_html(html: str):
    bodies = re.findall(r"<script>(.*?)</script>", html, re.S)
    if len(bodies) < 2:
        raise AssertionError("expected two inline <script> blocks (data, app)")
    return bodies[0], bodies[1]


def render_with_node(html: str) -> dict:
    """Execute the page's real data + app scripts against the DOM shim."""
    data_js, app_js = _scripts_from_html(html)
    prog = ("var PANEL_IDS_JSON = " + json.dumps(PANEL_IDS) + ";\n"
            + _SHIM + "\n" + data_js + "\n" + app_js + "\n" + _REPORT)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "run.js")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(prog)
        p = subprocess.run([_NODE, path], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise AssertionError("app.js threw while rendering:\n" + p.stderr[-4000:])
    m = re.search(r"__RESULT__(\{.*\})", p.stdout)
    if not m:
        raise AssertionError("no render report; stdout=" + p.stdout[-2000:]
                             + "\nstderr=" + p.stderr[-2000:])
    return json.loads(m.group(1))


# ===========================================================================

class SyntheticTable(unittest.TestCase):

    def test_schema_is_the_frozen_one(self):
        rows = MS.build_rows(clips=["u02", "u05"], models=("nova-3",), n_failed=1)
        self.assertTrue(rows)
        for r in rows[:20]:
            self.assertEqual(set(r), set(MS.MASTER_COLUMNS),
                             "synthetic rows must carry exactly the frozen schema")

    def test_wer_comes_from_the_real_aligner(self):
        """n_sub+n_del+n_ins over n_ref must equal wer — the row is self-consistent."""
        rows = [r for r in MS.build_rows(clips=["u02", "u05"], models=("nova-3",),
                                         n_failed=0)]
        for r in rows:
            got = (r["n_sub"] + r["n_del"] + r["n_ins"]) / r["n_ref"]
            self.assertAlmostEqual(got, r["wer"], places=5)

    def test_the_dead_zone_is_actually_planted(self):
        """High reverb + good SNR must be high WER AND high confidence."""
        c_hi = MS.planted_confidence("nova-3", 1.0, 25.0, "babble", "none", 0.0)
        c_lo = MS.planted_confidence("nova-3", 0.2, 0.0, "babble", "none", 0.0)
        self.assertGreater(c_hi, 0.85, "the dead-zone corner must stay confident")
        self.assertLess(c_lo, c_hi)
        rates = MS.planted_rates(1.0, 25.0, "babble", "none", 0.0)
        self.assertGreater(rates["del_rate"], 0.3, "reverb must drive deletions")


class BuildProduct(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(SYNTH_CSV):
            MS.write_csv(MS.build_rows(), SYNTH_CSV)
        cls.tmp = tempfile.mkdtemp()
        cls.out = os.path.join(cls.tmp, "dz.html")
        # --no-al: keeps the suite fast AND exercises a deliberate empty state.
        payload = B.collect(SYNTH_CSV, with_al=False, sobol_n=256)
        B.emit_html(payload, cls.out)
        with open(cls.out, encoding="utf-8") as fh:
            cls.html = fh.read()
        cls.payload = payload

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_it_builds(self):
        self.assertTrue(os.path.getsize(self.out) > 20_000)

    def test_standalone_html(self):
        h = self.html
        self.assertTrue(h.lstrip().lower().startswith("<!doctype html>"))
        for tag in ("<html", "<head", "<body", "</html>"):
            self.assertIn(tag, h.lower())
        self.assertEqual(h.lower().count("<body"), 1)

    def test_no_external_references(self):
        """The demo-risk test: nothing may be fetched from anywhere."""
        for attr, val in re.findall(r'\b(src|href)\s*=\s*["\']([^"\']*)["\']', self.html, re.I):
            self.assertNotIn("http://", val, f"{attr} points off-host: {val}")
            self.assertNotIn("https://", val, f"{attr} points off-host: {val}")
            self.assertFalse(val.startswith("//"), f"{attr} is protocol-relative: {val}")
        for tag in ("<link", "<iframe", "<img"):
            self.assertNotIn(tag, self.html.lower(),
                             f"{tag} can reference an external asset; keep the page inline")
        # no runtime network either
        for bad in ("fetch(", "XMLHttpRequest", "new WebSocket", "importScripts",
                    "cdn.", "unpkg", "jsdelivr", "googleapis"):
            self.assertNotIn(bad, self.html, f"{bad!r} would need a network")

    def test_embedded_json_parses_and_has_no_nan(self):
        data = self._data()
        self.assertIn("meta", data)
        self.assertIn("models", data)
        self.assertTrue(data["models"], "no model made it into the payload")
        for tok in ("NaN", "Infinity", "-Infinity"):
            self.assertNotIn(f":{tok}", self._blob(),
                             f"{tok} is not valid JSON and JSON.parse will reject it")

    def test_one_data_door(self):
        """All data access is behind loadData()/one blob — assert there is exactly one."""
        self.assertEqual(self.html.count("window.__DEADZONE_DATA__ = "), 1)
        self.assertIn("function loadData()", self.html)

    def test_every_panel_has_a_caption(self):
        for pid in PANEL_IDS:
            i = self.html.index(f'id="{pid}"')
            head = self.html[max(0, i - 1400):i]
            self.assertIn('class="caption"', head,
                          f"{pid} has no 'what am I looking at' caption")

    def test_skipped_panel_carries_a_reason(self):
        m = self.payload["models"][self.payload["default_model"]]
        al = m["active_learning"]
        self.assertEqual(al["status"], "missing")
        self.assertIn("--no-al", al["reason"])

    def test_panel_helper_swallows_any_exception(self):
        notes = []

        def boom():
            raise RuntimeError("upstream module exploded")
        got = B._panel("demo", boom, notes)
        self.assertEqual(got["status"], "missing")
        self.assertIn("upstream module exploded", got["reason"])
        self.assertTrue(notes)

    def test_panel_helper_survives_a_missing_module(self):
        notes = []

        def missing_module():
            import deadzone.analysis.not_a_real_module_yet  # noqa: F401
        got = B._panel("future layer", missing_module, notes)
        self.assertEqual(got["status"], "missing")
        self.assertIn("not importable", got["reason"])

    def test_jsonable_nan_becomes_null_not_zero(self):
        got = B.jsonable({"a": float("nan"), "b": float("inf"), "c": 0.5})
        self.assertIsNone(got["a"])
        self.assertIsNone(got["b"])
        self.assertEqual(got["c"], 0.5)

    def test_sim_rows_are_split_out_of_the_main_panels(self):
        rows = B.load_rows(SYNTH_CSV)
        real, sim = B.split_real_sim(rows)
        self.assertTrue(sim, "the synthetic table should carry a sim arm")
        self.assertTrue(all("rirs_sim" not in str(r.get("rir_key")) for r in real))

    # -- helpers ----------------------------------------------------------
    def _blob(self):
        m = re.search(r"window\.__DEADZONE_DATA__ = (.*?);\n?</script>", self.html, re.S)
        self.assertIsNotNone(m, "the data blob is missing")
        return m.group(1)

    def _data(self):
        return json.loads(self._blob().replace("<\\/", "</"))


@unittest.skipIf(_NODE is None, "node not available; DOM rendering not exercised")
class Rendering(unittest.TestCase):
    """Run app.js for real against a DOM shim — no browser, no network."""

    def test_missing_payloads_render_explained_empty_states(self):
        payload = {
            "meta": {"generated": "now", "source": "none", "is_synthetic": True,
                     "n_rows": 0, "n_conditions": 0, "models": ["nova-3"], "notes": []},
            "models": {"nova-3": {
                "silent_failure": B._missing("confidence_gap has not landed yet"),
                "fingerprints": B._missing("fingerprints has not landed yet"),
                "sensitivity": B._missing("no results/sobol.json"),
                "active_learning": B._missing("al_savings has not landed yet"),
                "sim2real": B._missing("no simulated-RIR rows"),
            }},
            "default_model": "nova-3",
        }
        with tempfile.TemporaryDirectory() as td:
            out = B.emit_html(payload, os.path.join(td, "e.html"))
            with open(out, encoding="utf-8") as fh:
                rep = render_with_node(fh.read())
        for pid in PANEL_IDS:
            p = rep["panels"][pid]
            self.assertEqual(p["empties"], 1, f"{pid} must render exactly one empty state")
            self.assertGreater(len(p["text"].strip()), 40,
                               f"{pid} empty state must explain itself, not sit blank")
        self.assertIn("has not landed yet", rep["panels"]["panel-hero"]["text"])

    def test_absent_panel_key_still_renders_an_empty_state(self):
        """A model dict with NO key at all (older build) must not blank the page."""
        payload = {"meta": {"generated": "now", "source": "x", "is_synthetic": True,
                            "n_rows": 0, "n_conditions": 0, "models": ["m"], "notes": []},
                   "models": {"m": {}}, "default_model": "m"}
        with tempfile.TemporaryDirectory() as td:
            out = B.emit_html(payload, os.path.join(td, "e.html"))
            with open(out, encoding="utf-8") as fh:
                rep = render_with_node(fh.read())
        for pid in PANEL_IDS:
            self.assertEqual(rep["panels"][pid]["empties"], 1)

    def test_no_data_at_all_does_not_crash(self):
        payload = {"meta": {"generated": "now", "source": "x", "is_synthetic": True,
                            "n_rows": 0, "n_conditions": 0, "models": [], "notes": []},
                   "models": {}, "default_model": None}
        with tempfile.TemporaryDirectory() as td:
            out = B.emit_html(payload, os.path.join(td, "e.html"))
            with open(out, encoding="utf-8") as fh:
                rep = render_with_node(fh.read())
        for pid in PANEL_IDS:
            self.assertEqual(rep["panels"][pid]["empties"], 1)

    def test_committed_artifact_renders_every_panel(self):
        if not os.path.exists(BUILT):
            self.skipTest("dashboard/deadzone.html not built yet")
        with open(BUILT, encoding="utf-8") as fh:
            rep = render_with_node(fh.read())
        for pid in PANEL_IDS:
            p = rep["panels"][pid]
            self.assertEqual(p["empties"], 0,
                             f"{pid} is an empty state in the committed build: {p['text'][:200]}")
            self.assertGreater(p["svgs"] + p["children"], 0, f"{pid} rendered nothing")
        self.assertGreaterEqual(len(rep["models"]), 1, "the model toggle rendered no models")

    def test_hover_shows_factors_and_a_real_transcript_diff(self):
        """The hover IS the hero panel. It must produce factor values + a diff."""
        if not os.path.exists(BUILT):
            self.skipTest("dashboard/deadzone.html not built yet")
        with open(BUILT, encoding="utf-8") as fh:
            rep = render_with_node(fh.read())
        self.assertGreater(rep["hero_points"], 5, "the scatter drew almost no points")
        txt = rep["detail_text"]
        self.assertIn("transcript diff", txt)
        self.assertIn("rt60", txt)
        self.assertIn("snr_db", txt)
        self.assertIn("WER", txt)

    def test_stepping_the_al_panel_does_not_break_it(self):
        if not os.path.exists(BUILT):
            self.skipTest("dashboard/deadzone.html not built yet")
        with open(BUILT, encoding="utf-8") as fh:
            rep = render_with_node(fh.read())
        self.assertIn("oracle calls", rep["al_after_step"])
        self.assertIn("step 2/", rep["al_after_step"])

    # Panels that may legitimately be an empty state for SOME models, with the
    # reason. These are not exemptions from rendering -- see the assertions
    # below, which still require the panel to draw an explanation. They are
    # exemptions only from "must contain data for every model".
    #
    #   panel-sim2real: the simulated-RIR arm was only ever run for nova-3
    #     (results_sim/ holds 1760 nova-3 rows and nothing else). D4 compares
    #     measured vs synthetic RIRs for ONE model; re-running it per model
    #     would double the API spend for a question that is about RIR
    #     provenance, not about model family. A model with no sim arm therefore
    #     has genuinely nothing to plot, and build_sim2real returns an explained
    #     empty state rather than the raw KeyError it used to raise.
    MODEL_CONDITIONAL_EMPTY = {"panel-sim2real"}

    def test_model_toggle_re_renders_every_panel(self):
        """Switching model must re-render every panel.

        A panel with no data for the newly-selected model is allowed to be an
        empty state, but it must SAY SO: an unexplained blank panel and a
        crashed panel look identical to someone demoing this, and the whole
        point of the empty state is that it names the reason.
        """
        if not os.path.exists(BUILT):
            self.skipTest("dashboard/deadzone.html not built yet")
        with open(BUILT, encoding="utf-8") as fh:
            rep = render_with_node(fh.read())
        self.assertGreaterEqual(len(rep["models"]), 2, "need >=2 models to test the toggle")
        rendered = rep.get("after_toggle_rendered") or {}
        texts = rep.get("after_toggle_text") or {}
        for pid, empties in (rep.get("after_toggle") or {}).items():
            # Non-negotiable for every panel: it drew something.
            self.assertGreater(rendered.get(pid, 0), 0,
                               f"{pid} rendered NOTHING after switching model")
            if pid in self.MODEL_CONDITIONAL_EMPTY:
                if empties:
                    self.assertTrue(
                        (texts.get(pid) or "").strip(),
                        f"{pid} is an empty state after switching model but "
                        f"explains nothing -- an unexplained blank panel is a bug")
                continue
            self.assertEqual(empties, 0, f"{pid} went empty after switching model")


class ModelArmsPanelIsNArm(unittest.TestCase):
    """
    Panel 7 IS the multi-model comparison, so it is the panel a third arm most
    obviously belongs in — and the one where its absence is least visible,
    because the page renders identically either way.
    """

    THIRD = "elevenlabs-scribe"

    def _payload(self, tmp, arms):
        """A minimal results/model_arms.json in the shape the analysis writes."""
        res = {
            "arms": list(arms),
            "arm_census": {"n_arms": len(arms), "n_common_cells": 40,
                           "n_common_clips": 4, "n_common_conditions": 10,
                           "cells_matched": False, "arms_excluded": [],
                           "n_rows_dropped_by_intersection": 12},
            "per_model": {m: {"n_conditions": 10, "wer_mean_strict": 0.2,
                              "wer_mean_crossmodel": 0.2, "dead_zone_rate": 0.1,
                              "n_dead_zones": 1, "shape": {"spearman": -0.9},
                              "edit_signature_crossmodel": {"sub": 0.1, "del": 0.1,
                                                            "ins": 0.0}}
                          for m in arms},
            "normalization_shift": {m: {"mean_shift": 0.0} for m in arms},
            "dead_zone_overlap": {
                "models": list(arms), "jaccard": 0.0, "shared": [],
                "jaccard_definition": "all-arm Jaccard",
                "pairwise": {f"{a}|{b}": {"models": [a, b], "jaccard": 0.5,
                                          "n_shared": 1, "n_union": 2}
                             for i, a in enumerate(arms) for b in arms[i + 1:]},
                **{f"{m}_only": ["c1"] for m in arms},
            },
            "divergence_regions": [],
            # every arm measured; the THIRD is the one that hallucinates
            "hallucination_by_model": {
                m: {"median_len_ratio": 1.0, "p95_len_ratio": 1.0,
                    "frac_rows_over_2x": (0.8 if m == self.THIRD else 0.0),
                    "mean_foreign_frac": 0.1, "examples": []}
                for m in arms},
        }
        if "whisper-base" in arms:
            res["whisper_hallucination"] = res["hallucination_by_model"]["whisper-base"]
        path = os.path.join(tmp, "model_arms.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        return path

    def test_three_arms_reach_the_panel_and_the_worst_one_is_named(self):
        arms = ["nova-3", "whisper-base", self.THIRD]
        with tempfile.TemporaryDirectory() as tmp:
            panel = B.build_model_arms(self._payload(tmp, arms))
        self.assertEqual(panel["status"], "ok", panel.get("reason"))
        d = panel["data"]
        self.assertEqual({m["model"] for m in d["models"]}, set(arms))
        # the hallucination block is selected by MEASUREMENT and names its arm,
        # rather than being whichever arm a literal pointed at
        self.assertEqual(d["hallucination"]["model"], self.THIRD)
        self.assertAlmostEqual(d["hallucination"]["frac_rows_over_2x"], 0.8)
        self.assertEqual(set(d["hallucination_by_model"]), set(arms))
        # all three pairs travel, so an all-arm Jaccard of 0 cannot be mistaken
        # for "no two arms agree anywhere"
        self.assertEqual(len(d["overlap"]["pairwise"]), 3)
        self.assertEqual(set(d["overlap"]["only"]),
                         {f"{m}_only" for m in arms})
        self.assertEqual(d["census"]["n_arms"], 3)
        json.dumps(B.jsonable(panel))          # must survive the page's JSON door

    def test_two_arms_still_report_the_baseline_arm(self):
        """
        NEGATIVE CONTROL. Without it the test above would pass on a build that
        always reports the last arm, or the alphabetically-last one. With only
        the original two arms — and whisper the sole hallucinator — the panel
        must name whisper, not something else.
        """
        arms = ["nova-3", "whisper-base"]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._payload(tmp, arms)
            with open(path, encoding="utf-8") as fh:
                res = json.load(fh)
            res["hallucination_by_model"]["whisper-base"]["frac_rows_over_2x"] = 0.5
            res["whisper_hallucination"] = res["hallucination_by_model"]["whisper-base"]
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(res, fh)
            panel = B.build_model_arms(path)
        d = panel["data"]
        self.assertEqual(d["hallucination"]["model"], "whisper-base")
        self.assertEqual(len(d["overlap"]["pairwise"]), 1)
        self.assertEqual(len(d["models"]), 2)


class CommittedArtifact(unittest.TestCase):

    def test_it_exists_and_is_self_contained(self):
        self.assertTrue(os.path.exists(BUILT),
                        "dashboard/deadzone.html must be built and committed")
        with open(BUILT, encoding="utf-8") as fh:
            html = fh.read()
        for attr, val in re.findall(r'\b(src|href)\s*=\s*["\']([^"\']*)["\']', html, re.I):
            self.assertNotIn("http", val, f"{attr}={val} reaches off-host")
        m = re.search(r"window\.__DEADZONE_DATA__ = (.*?);\n?</script>", html, re.S)
        self.assertIsNotNone(m)
        json.loads(m.group(1).replace("<\\/", "</"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
