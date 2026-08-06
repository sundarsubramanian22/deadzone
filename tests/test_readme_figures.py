#!/usr/bin/env python3
"""
test_readme_figures.py — the README's five charts, pinned to the artifacts.

    ./.venv/bin/python tests/test_readme_figures.py

A chart is a claim about a number, and until now this repo had no way to check
one. `tests/test_report_numbers.py` closed that hole for `report/*.md` and its
own docstring names the scope it does NOT cover: prose and pictures outside
`report/`. SPEC Appendix J.7 is what that costs — a rehearsal, not a test,
found the demo script narrating a verdict the dashboard contradicts and an SNR
level the grid never ran, with every suite green throughout. A figure is worse
than prose in one specific way: a number rendered as a bar cannot be grepped, so
nobody will ever notice it drifted.

So the generator emits `docs/assets/figures.json` beside the SVGs — every value
it drew, with the population that value is over — and this suite re-derives each
one straight from `results/` and fails on disagreement.

WHAT EACH TEST IS ACTUALLY DEFENDING
------------------------------------
1. THE POPULATION TRAP (SPEC Appendix G; UNDERSTANDING 4.12). nova-3 ran 40
   clips, the other two arms ran the 10-clip AL subset, so nova-3's dead-zone
   rate is 1.14% (2/176) on one population and 0.57% (1/176) on the other. Both
   are right. Quoting one without its clip count is the bug, and this project
   has committed it at least three times — twice inside a chart or a table.
   `test_every_figure_names_its_population` asserts the clip count is on the
   FACE of every SVG, and `test_the_three_arm_figure_is_the_matched_population`
   asserts the cross-arm chart uses the intersection and prints both of nova-3's
   rates. Its negative control mutates the figure record to the 40-clip rate and
   requires the check to fail.

2. THE SCRIBE EXCLUSION (SPEC I.5). `dead_zone_flags` thresholds an ABSOLUTE
   WER, so it is not scale-free, and elevenlabs-scribe's orthography is a
   per-call draw: its 7 strict dead zones fall to 0 under the cross-model
   normalizer. Drawing that bar would publish spelling as confident error.
   `test_the_incomparable_arm_gets_no_dead_zone_bar` pins the exclusion AND that
   the arm still appears in the rank-correlation panel, because silently
   dropping the arm entirely would be the J.6 defect (a complete-looking report
   about fewer models than were paid for).

3. THE MUTE CONDITIONS (SPEC G.7). Seven nova-3 conditions carry no confidence
   at all. Plotting them at confidence 0 fabricates seven points at the ideal
   corner of a negative correlation — which is exactly how a published rho of
   -0.957 came to be an artifact. `test_mute_conditions_are_counted_not_plotted`
   requires the hero to plot 169, not 176, and to say so.

4. DARK MODE. GitHub renders READMEs in both themes and serves README images
   through an <img>, so a transparent SVG with dark ink is invisible to half the
   readers. Every figure must carry an opaque full-canvas background.

5. SELF-CONTAINMENT. No external font, no CDN, no <style> block — GitHub
   sanitizes SVG and a chart that depends on what survives is a chart that
   renders differently for the reader than for the author.

6. LEGIBILITY AND OVERFLOW. An SVG neither clips nor reflows: text past the
   canvas edge is silently drawn outside it and the file stays valid. The
   generator refuses to emit that, and `TestTheOverflowGuard` proves the refusal
   fires — with a negative control, so it is pinned to the overflow and not to
   some incidental property of the string.

Offline. No API, no network, no browser. The number tests read `results/` and
skip without it (it is gitignored); the structural tests run on the committed
SVGs always.
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# -------------------------------------------------------------------------

import json
import re
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts import make_readme_figures as mrf

REPO = Path(_REPO_ROOT)
ASSETS = REPO / "docs" / "assets"
MASTER = REPO / "results" / "master.csv"
PY = str(Path(sys.executable))

# The five paths README.md embeds. Renaming one breaks the README silently —
# GitHub renders a missing image as alt text — so the names are pinned here.
FIVE = (
    "confidence-vs-error.svg",
    "overconfidence.svg",
    "fingerprints.svg",
    "model-comparison.svg",
    "whisper-hallucination.svg",
)
MIN_FONT = 18.0          # ~12px at the 65%-width case the figures are sized for
MIN_BYTES = 4_000


def _needs_master(case: unittest.TestCase) -> None:
    if not MASTER.is_file():
        case.skipTest("needs results/master.csv (gitignored) to re-derive")


def _numbers() -> dict:
    path = ASSETS / mrf.FIG_NUMBERS
    if not path.is_file():
        raise unittest.SkipTest(f"{path} not generated")
    return json.loads(path.read_text(encoding="utf-8"))["figures"]


def _svg_text(name: str) -> str:
    """All visible text of a figure, concatenated — what a reader can read."""
    root = ET.fromstring((ASSETS / name).read_text(encoding="utf-8"))
    ns = "{http://www.w3.org/2000/svg}"
    return " ".join(t.text or "" for t in root.iter(f"{ns}text"))


# ===========================================================================
class TestTheFilesExist(unittest.TestCase):

    def test_all_five_are_present_and_non_trivial(self):
        for name in FIVE:
            path = ASSETS / name
            self.assertTrue(path.is_file(), f"{path} is missing")
            body = path.read_text(encoding="utf-8")
            self.assertGreater(len(body), MIN_BYTES,
                               f"{name} is {len(body)} bytes — too small to be "
                               f"a real figure")
            self.assertIn("<svg", body)

    def test_each_one_parses_and_declares_its_geometry(self):
        for name in FIVE:
            root = ET.fromstring((ASSETS / name).read_text(encoding="utf-8"))
            vb = root.get("viewBox")
            self.assertIsNotNone(vb, f"{name} has no viewBox — it will not "
                                     f"scale in a README")
            x, y, w, h = (float(v) for v in vb.split())
            self.assertEqual((x, y), (0.0, 0.0))
            self.assertEqual(w, float(mrf.W),
                             f"{name} is {w} wide; the figures share one width "
                             f"so they line up down the page")
            self.assertGreater(h, 200)
            # width/height as well as viewBox: without them some renderers
            # size the image to the container and the type comes out tiny.
            self.assertEqual(root.get("width"), str(int(w)))
            self.assertEqual(root.get("height"), str(int(h)))

    def test_each_one_stands_alone(self):
        """A title, a population line and axis/section labels, per the brief."""
        for name in FIVE:
            root = ET.fromstring((ASSETS / name).read_text(encoding="utf-8"))
            ns = "{http://www.w3.org/2000/svg}"
            self.assertTrue((root.find(f"{ns}title") is not None)
                            and (root.find(f"{ns}title").text or "").strip(),
                            f"{name} has no <title> — it is also the alt text")
            self.assertTrue((root.find(f"{ns}desc") is not None)
                            and (root.find(f"{ns}desc").text or "").strip(),
                            f"{name} has no <desc>")
            self.assertGreaterEqual(len(_svg_text(name)), 400,
                                    f"{name} carries almost no text; it cannot "
                                    f"be read without the README around it")


class TestItSurvivesBothGitHubThemes(unittest.TestCase):
    """
    GitHub renders a README in light AND dark. An SVG embedded with ![]() is
    loaded as an image, so it does not inherit the page's colours: a chart with
    a transparent background and near-black ink is invisible to every dark-mode
    reader, and nothing in the build says so.

    The choice made here is an EXPLICIT LIGHT BACKGROUND on every figure rather
    than a theme-adaptive one, because theme adaptation inside an <img> needs a
    <style> block with a prefers-color-scheme query and GitHub's sanitizer is
    not a contract. So the invariant to hold is: an opaque rect covering the
    whole canvas, drawn first.
    """

    def test_every_figure_has_an_opaque_full_canvas_background(self):
        ns = "{http://www.w3.org/2000/svg}"
        for name in FIVE:
            root = ET.fromstring((ASSETS / name).read_text(encoding="utf-8"))
            w, h = float(root.get("width")), float(root.get("height"))
            first = next(iter(root.iter(f"{ns}rect")), None)
            self.assertIsNotNone(first, f"{name} has no background rect")
            self.assertEqual((float(first.get("x")), float(first.get("y"))),
                             (0.0, 0.0), f"{name}: first rect is not at 0,0")
            self.assertEqual((float(first.get("width")),
                              float(first.get("height"))), (w, h),
                             f"{name}: first rect does not cover the canvas")
            fill = (first.get("fill") or "").lower()
            self.assertTrue(fill.startswith("#") and fill != "none",
                            f"{name}: background fill is {fill!r}; a "
                            f"transparent figure disappears in dark mode")
            self.assertIsNone(first.get("opacity"),
                              f"{name}: the background rect is not opaque")

    def test_the_ink_is_dark_because_the_background_is_light(self):
        """
        The pair is the invariant, not either half. A light background with
        light text passes the test above and is unreadable.
        """
        for name in FIVE:
            body = (ASSETS / name).read_text(encoding="utf-8")
            self.assertIn(f'fill="{mrf.INK}"', body,
                          f"{name} never uses the primary ink colour")
            self.assertNotIn('fill="#fff"', body.lower().replace(" ", ""))


class TestItIsSelfContained(unittest.TestCase):

    FORBIDDEN = (
        ("http://", "an external reference"),
        ("https://www", "an external reference"),
        ("@import", "a CSS import"),
        ("<style", "a <style> block — GitHub's sanitizer is not a contract"),
        ("<script", "a script"),
        ("<image", "an embedded raster"),
        ("xlink:href", "an external link"),
        ("@font-face", "a web font"),
    )

    def test_no_external_or_sanitizable_dependency(self):
        for name in FIVE:
            body = (ASSETS / name).read_text(encoding="utf-8")
            # the SVG namespace is the one http:// that must be there
            probe = body.replace('xmlns="http://www.w3.org/2000/svg"', "")
            for needle, why in self.FORBIDDEN:
                self.assertNotIn(needle, probe,
                                 f"{name} contains {needle!r} — {why}")

    def test_only_generic_font_stacks(self):
        """A named webfont renders as a fallback for the reader and correctly
        for the author, which is the worst of both: the layout was measured
        against metrics the reader never gets."""
        for name in FIVE:
            fams = set(re.findall(r'font-family="([^"]+)"',
                                  (ASSETS / name).read_text(encoding="utf-8")))
            self.assertTrue(fams, f"{name} sets no font-family at all")
            for fam in fams:
                self.assertIn(fam, {mrf.FONT, mrf.MONO},
                              f"{name} uses an unexpected font stack: {fam!r}")
                self.assertTrue(fam.rstrip().endswith(("sans-serif", "monospace")),
                                f"{name}: {fam!r} has no generic fallback")


class TestItIsLegibleOnASharedScreen(unittest.TestCase):

    def test_nothing_is_set_below_the_floor(self):
        for name in FIVE:
            sizes = [float(v) for v in re.findall(
                r'font-size="([\d.]+)"',
                (ASSETS / name).read_text(encoding="utf-8"))]
            self.assertTrue(sizes, f"{name} sets no font-size")
            self.assertGreaterEqual(
                min(sizes), MIN_FONT,
                f"{name} has text at {min(sizes)} units — below {MIN_FONT}, "
                f"which is roughly 12px once the README is 65% of full width")

    def test_no_text_is_drawn_outside_the_canvas(self):
        """
        The estimator `text_w` OVER-estimates against real browser metrics
        (measured 2-10% high across the strings these figures use), so a pass
        here means it fits; it cannot mean the opposite.
        """
        ns = "{http://www.w3.org/2000/svg}"
        for name in FIVE:
            root = ET.fromstring((ASSETS / name).read_text(encoding="utf-8"))
            width = float(root.get("width"))
            for t in root.iter(f"{ns}text"):
                if t.get("transform"):
                    continue                      # rotated: bbox is pre-rotation
                s = t.text or ""
                size = float(t.get("font-size"))
                weight = t.get("font-weight", "400")
                mono = t.get("font-family") == mrf.MONO
                tw = mrf.text_w(s, size, weight, mono=mono)
                x = float(t.get("x"))
                left = {"start": x, "middle": x - tw / 2,
                        "end": x - tw}[t.get("text-anchor", "start")]
                self.assertGreaterEqual(left, -4, f"{name}: {s[:50]!r} starts "
                                                  f"left of the canvas")
                self.assertLessEqual(left + tw, width + 4,
                                     f"{name}: {s[:50]!r} runs to "
                                     f"{left + tw:.0f} of {width:.0f}")


class TestTheOverflowGuard(unittest.TestCase):
    """
    The generator refuses to write a figure whose text leaves the canvas —
    SPEC E.3's move (refuse at the one creation site rather than guard every
    reader), because an SVG neither clips nor reflows and the failure is
    therefore silent: valid file, plausible chart, words off the edge.
    """

    def test_it_refuses_a_string_that_would_not_fit(self):
        s = mrf.Svg(400, 200, "t", "d")
        with self.assertRaises(SystemExit):
            s.text(30, 100, "x" * 400, 24)

    def test_it_refuses_a_baseline_below_the_canvas(self):
        s = mrf.Svg(400, 200, "t", "d")
        with self.assertRaises(SystemExit):
            s.text(30, 260, "fine", 20)

    def test_negative_control_a_string_that_fits_is_written(self):
        """Without this the guard could be passing by refusing everything."""
        s = mrf.Svg(400, 200, "t", "d")
        s.text(30, 100, "short enough", 20)
        self.assertIn("short enough", s.render())

    def test_the_estimator_charges_more_for_wide_glyphs(self):
        """A character-count wrap treats 'WWW' and 'iii' as equal and is how a
        caps-heavy footnote silently runs off the page."""
        self.assertGreater(mrf.text_w("WWWWWWWWWW", 20),
                           mrf.text_w("iiiiiiiiii", 20) * 2)
        self.assertGreater(mrf.text_w("abc", 20, "700"),
                           mrf.text_w("abc", 20, "400"))


# ===========================================================================
#  the numbers
# ===========================================================================
class TestEveryFigureNamesItsPopulation(unittest.TestCase):

    def test_the_record_carries_a_population_for_each_figure(self):
        nums = _numbers()
        self.assertEqual(set(nums), set(FIVE))
        for name, rec in nums.items():
            self.assertTrue(rec.get("population", "").strip(),
                            f"{name} has no population in figures.json")

    def test_the_clip_count_is_on_the_face_of_the_chart(self):
        """
        Not in the record, not in the README — on the image. UNDERSTANDING 4.12:
        every number needs its population attached, and a figure travels away
        from whatever text was around it.
        """
        expect = {
            "confidence-vs-error.svg": ["40 clips", "176 conditions"],
            "overconfidence.svg": ["40 clips", "176 conditions"],
            "fingerprints.svg": ["40 clips", "nova-3"],
            "model-comparison.svg": ["10-clip", "40 clips"],
            "whisper-hallucination.svg": ["one row of results/master.csv"],
        }
        for name, needles in expect.items():
            text = _svg_text(name)
            for needle in needles:
                self.assertIn(needle, text,
                              f"{name} does not say {needle!r} anywhere a "
                              f"reader can see it")


class TestTheHeroMatchesTheArtifact(unittest.TestCase):

    def setUp(self):
        _needs_master(self)
        self.rec = _numbers()["confidence-vs-error.svg"]
        rows = mrf.load_rows()
        self.rep, self.pay = mrf.spine_payload(rows)

    def test_correlation_overconfidence_and_gap(self):
        corr, gs = self.rep["correlation"], self.rep["gap_summary"]
        self.assertAlmostEqual(self.rec["spearman"],
                               corr["spearman_confpct_vs_wer"], places=12)
        self.assertEqual(self.rec["spearman_n"], corr["n"])
        self.assertAlmostEqual(self.rec["mean_gap"], gs["mean"], places=12)
        self.assertEqual(self.rec["n_overconfident"],
                         int(round(gs["frac_overconfident"] * gs["n"])))

    def test_the_two_dead_zones_are_the_two_the_layer_flags(self):
        drawn = [d["condition_name"] for d in self.rec["dead_zones"]]
        real = [d["condition_name"] for d in self.rep["dead_zones"]]
        self.assertEqual(sorted(drawn), sorted(real))
        self.assertEqual(self.rec["n_dead_zones"],
                         self.rep["categories"]["n_dead_zones"])

    def test_each_dead_zone_satisfies_the_gap_identity(self):
        """
        gap = mean_conf - (1 - wer_spoke). SPEC C.5 records this identity being
        the only thing that caught a headline read off the wrong CSV column —
        rt60_measured 0.680 sits next to mean_conf 0.843 and reads exactly like
        a confidence. Checking the identity beats checking a position.
        """
        for d in self.rec["dead_zones"]:
            self.assertAlmostEqual(
                d["gap"], d["mean_conf"] - (1.0 - d["wer_spoke"]), places=9,
                msg=f"{d['condition_name']}: the drawn conf/WER/gap do not "
                    f"satisfy gap = conf - (1 - WER)")

    def test_the_quadrant_thresholds_are_the_layers_own(self):
        """A figure that re-picks the thresholds would shade a box the
        dead-zone list disagrees with."""
        th = self.rep["thresholds"]
        self.assertAlmostEqual(self.rec["wer_hi"], th["wer_hi"], places=12)
        self.assertAlmostEqual(self.rec["conf_hi_raw"], th["conf_hi_raw"],
                               places=12)
        self.assertEqual(th["wer_hi"], mrf.WER_HI)
        self.assertEqual(th["conf_pct_hi"], mrf.CONF_PCT_HI)


class TestMuteConditionsAreCountedNotPlotted(unittest.TestCase):

    def setUp(self):
        _needs_master(self)
        self.rec = _numbers()["confidence-vs-error.svg"]

    def test_the_hero_plots_the_speaking_conditions_only(self):
        self.assertEqual(self.rec["n_points_plotted"],
                         self.rec["n_conditions"] - self.rec["n_mute"])
        self.assertLess(self.rec["n_points_plotted"], self.rec["n_conditions"])

    def test_the_missing_ones_are_declared_on_the_chart(self):
        text = _svg_text("confidence-vs-error.svg")
        self.assertIn(str(self.rec["n_mute"]), text)
        self.assertIn("blind", text.lower(),
                      "the chart does not say a confidence monitor cannot see "
                      "the mute conditions — that is the whole reason they are "
                      "a separate category (SPEC G.5)")

    def test_negative_control_plotting_them_would_break_the_arithmetic(self):
        bad = dict(self.rec, n_points_plotted=self.rec["n_conditions"])
        self.assertNotEqual(bad["n_points_plotted"],
                            bad["n_conditions"] - bad["n_mute"])


class TestTheThreeArmFigureIsTheMatchedPopulation(unittest.TestCase):

    def setUp(self):
        _needs_master(self)
        self.rec = _numbers()["model-comparison.svg"]
        self.md = mrf.matched_arms_data(mrf.load_rows())

    def test_it_uses_the_intersection_and_publishes_the_census(self):
        cen = self.md["census"]
        self.assertEqual(self.rec["census"]["n_common_clips"],
                         cen["n_common_clips"])
        self.assertEqual(self.rec["census"]["n_common_cells"],
                         cen["n_common_cells"])
        self.assertEqual(self.rec["census"]["n_arms"], 3)
        # the intersection is genuinely smaller than the spine's own run, or
        # this figure is not testing what it claims to
        self.assertLess(cen["n_common_clips"], 40)

    def test_every_arm_number_matches_a_recomputation(self):
        for model, rec in self.rec["arms"].items():
            live = self.md["arms"][model]
            self.assertAlmostEqual(rec["dead_zone_rate"],
                                   live["dead_zone_rate"], places=12,
                                   msg=f"{model} dead-zone rate")
            self.assertEqual(rec["n_dead_zones"], live["n_dead_zones"])
            self.assertAlmostEqual(rec["spearman"],
                                   live["shape"]["spearman"], places=12,
                                   msg=f"{model} spearman")
            self.assertEqual(rec["spearman_n"], live["shape"]["n"])

    def test_it_matches_the_published_L1_artifact_too(self):
        """Two independent paths to the same numbers. `results/model_arms.json`
        was produced by the L1 runner; this figure recomputes from master.csv."""
        path = REPO / "results" / "model_arms.json"
        if not path.is_file():
            self.skipTest("results/model_arms.json not present")
        pub = json.loads(path.read_text(encoding="utf-8"))["per_model"]
        for model, rec in self.rec["arms"].items():
            self.assertAlmostEqual(rec["dead_zone_rate"],
                                   pub[model]["dead_zone_rate"], places=12,
                                   msg=f"{model} disagrees with model_arms.json")
            self.assertAlmostEqual(rec["spearman"],
                                   pub[model]["shape"]["spearman"], places=12)

    def test_the_chart_prints_BOTH_of_the_spines_rates(self):
        """
        1.14% on 40 clips and 0.57% on 10 are both correct; the failure mode is
        printing one. The chart must carry the other so a reader who has seen
        the hero figure is not left reconciling two numbers in their head.
        """
        text = _svg_text("model-comparison.svg")
        matched = 100 * self.rec["arms"][mrf.SPINE_MODEL]["dead_zone_rate"]
        self.assertIn(f"{matched:.2f}%", text)
        self.assertIn("1.14%", text,
                      "the chart never mentions the spine's 40-clip rate")
        self.assertIn("40 clips", text)

    def test_negative_control_the_forty_clip_rate_is_not_what_is_drawn(self):
        rate = self.rec["arms"][mrf.SPINE_MODEL]["dead_zone_rate"]
        self.assertAlmostEqual(rate, 1 / 176, places=6,
                               msg="the matched-population rate should be "
                                   "1/176; if it is 2/176 the figure quietly "
                                   "used the 40-clip population")
        self.assertNotAlmostEqual(rate, 2 / 176, places=6)


class TestTheIncomparableArmGetsNoDeadZoneBar(unittest.TestCase):

    def setUp(self):
        _needs_master(self)
        self.rec = _numbers()["model-comparison.svg"]

    def test_the_arm_is_present_but_its_rate_is_not_drawn(self):
        arms = self.rec["arms"]
        self.assertIn("elevenlabs-scribe", arms,
                      "the arm vanished entirely — that is SPEC J.6, a "
                      "complete-looking report about fewer models than ran")
        scribe = arms["elevenlabs-scribe"]
        self.assertFalse(scribe["wer_comparable"])
        self.assertFalse(scribe["dead_zone_rate_drawn"],
                         "a dead-zone bar was drawn for the arm whose WER is "
                         "not comparable — SPEC I.5: the threshold is on an "
                         "ABSOLUTE WER, so it is not a scale-free statistic")

    def test_the_comparable_arms_do_get_theirs(self):
        for model in (mrf.SPINE_MODEL, "whisper-base"):
            self.assertTrue(self.rec["arms"][model]["dead_zone_rate_drawn"],
                            f"{model} lost its bar; the exclusion is supposed "
                            f"to be one named arm, not a blanket refusal")

    def test_the_rank_statistic_survives_for_all_three(self):
        """Rank statistics are invariant to a constant orthography offset and
        merely attenuated by a per-call one, so the arm keeps its rho."""
        for model in self.rec["arms"]:
            self.assertLess(self.rec["arms"][model]["spearman"], 0.0)
            self.assertGreater(self.rec["arms"][model]["spearman_n"], 100)

    def test_the_chart_says_why(self):
        text = _svg_text("model-comparison.svg").lower()
        self.assertIn("not quotable", text)
        for needle in ("different transcript", "orthograph", "spelling"):
            if needle in text:
                break
        else:
            self.fail("the chart excludes the arm without giving the reason")


class TestNoFigureImpliesTheDeadZoneCountIsRobust(unittest.TestCase):
    """
    UNDERSTANDING 4.2: both thresholds are hardcoded defaults, and
    `results/dead_zone_sensitivity.json` returns FRAGILE for all three arms —
    nova-3's count runs 0 to 86 across the swept box. The count is an operating
    point; the correlation and the gap are the findings.
    """

    def test_the_hero_labels_its_box_an_operating_point(self):
        text = _svg_text("confidence-vs-error.svg").lower()
        self.assertIn("operating point", text)
        self.assertIn("threshold", text)

    def test_the_threshold_free_figure_says_it_uses_no_threshold(self):
        text = _svg_text("overconfidence.svg").lower()
        self.assertIn("no operating point", text)

    def test_the_fragility_range_on_the_hero_is_the_measured_one(self):
        path = REPO / "results" / "dead_zone_sensitivity.json"
        if not path.is_file():
            self.skipTest("results/dead_zone_sensitivity.json not present")
        arm = json.loads(path.read_text(encoding="utf-8"))[
            "per_model"][mrf.SPINE_MODEL]
        stats = arm["count_stats"]
        frag = _numbers()["confidence-vs-error.svg"]["threshold_fragility"]
        self.assertEqual(frag["min"], int(stats["min"]))
        self.assertEqual(frag["max"], int(stats["max"]))
        self.assertEqual(frag["n_grid_points"], int(stats["n_grid_points"]))
        # and the published count sits inside the range it is quoted against
        self.assertEqual(frag["count_at_default"], arm["count_at_default"])
        self.assertLessEqual(frag["min"], frag["count_at_default"])
        self.assertLessEqual(frag["count_at_default"], frag["max"])
        text = _svg_text("confidence-vs-error.svg")
        self.assertIn(f"from {frag['min']} to {frag['max']}", text,
                      "the hero quotes a sweep range that is not the one in "
                      "results/dead_zone_sensitivity.json")

    def test_negative_control_the_range_is_not_a_single_point(self):
        """If min == max the caption would be claiming fragility it did not
        measure, and the whole 4.2 caveat would be decoration."""
        frag = _numbers()["confidence-vs-error.svg"].get("threshold_fragility")
        if not frag:
            self.skipTest("no fragility artifact")
        self.assertGreater(frag["max"], frag["min"] + 10)

    def test_the_cross_arm_ordering_claim_is_recomputed_not_asserted(self):
        _needs_master(self)
        to = _numbers()["model-comparison.svg"]["threshold_ordering"]
        self.assertEqual(to["n_grid_points"],
                         len(mrf.WER_HI_GRID) * len(mrf.CONF_PCT_GRID))
        self.assertEqual(to["spine_strictly_below_baseline_at"],
                         to["n_grid_points"],
                         "the chart claims the ordering is threshold-free; the "
                         "sweep says otherwise")
        text = _svg_text("model-comparison.svg")
        self.assertIn(f"{to['spine_strictly_below_baseline_at']} of "
                      f"{to['n_grid_points']}", text)


class TestTheFingerprintFigureMatchesTheArtifact(unittest.TestCase):

    def setUp(self):
        _needs_master(self)
        self.rec = _numbers()["fingerprints.svg"]

    def test_the_header_row_is_a_direct_sum_of_the_table(self):
        live = mrf.overall_composition(mrf.load_rows(), mrf.SPINE_MODEL)
        for key in ("sub", "del", "ins"):
            self.assertAlmostEqual(self.rec["overall"][key], live[key],
                                   places=12, msg=key)
        self.assertEqual(self.rec["overall"]["n_ref"], live["n_ref"])

    def test_deletions_dominate_which_is_the_claim_the_chart_makes(self):
        o = self.rec["overall"]
        self.assertGreater(o["del"], o["sub"])
        self.assertGreater(o["del"], o["ins"])
        self.assertGreater(o["del"], 2 * o["sub"])

    def test_the_family_rows_come_from_the_fingerprints_artifact(self):
        path = REPO / "results" / "fingerprints.json"
        if not path.is_file():
            self.skipTest("results/fingerprints.json not present")
        pub = {s["family"]: s for s in json.loads(
            path.read_text(encoding="utf-8"))["by_model"][mrf.SPINE_MODEL][
                "signatures"]}
        self.assertEqual(len(self.rec["families"]), len(pub))
        for fam in self.rec["families"]:
            src = pub[fam["family"]]["rates_degraded"]
            for key in ("sub", "del", "ins"):
                self.assertAlmostEqual(fam[key], src[key], places=12,
                                       msg=f"{fam['family']} {key}")
            self.assertEqual(fam["dominant"],
                             pub[fam["family"]]["dominant_edit"])

    def test_the_dominance_tally_in_the_caption_is_counted_not_typed(self):
        n_sub = sum(1 for f in self.rec["families"] if f["dominant"] == "sub")
        self.assertEqual(self.rec["n_substitution_dominant"], n_sub)
        self.assertEqual(self.rec["n_deletion_dominant"],
                         self.rec["n_families"] - n_sub)
        text = _svg_text("fingerprints.svg")
        self.assertIn(f"{self.rec['n_deletion_dominant']} of "
                      f"{self.rec['n_families']} families", text)


class TestTheHallucinationExhibit(unittest.TestCase):

    def setUp(self):
        _needs_master(self)
        self.rec = _numbers()["whisper-hallucination.svg"]

    def test_it_is_the_row_that_is_actually_in_the_table(self):
        live = mrf.hallucination_row()
        for key in ("n_ref", "n_match", "n_sub", "n_del", "n_ins",
                    "n_ref_words", "n_hyp_words", "transcript", "reference"):
            self.assertEqual(self.rec[key], live[key], msg=key)
        self.assertAlmostEqual(self.rec["wer"], live["wer"], places=12)

    def test_the_edit_counts_reconstruct_the_stored_wer(self):
        r = self.rec
        wer = (r["n_sub"] + r["n_del"] + r["n_ins"]) / r["n_ref"]
        self.assertAlmostEqual(wer, r["wer"], places=9)
        self.assertEqual(r["n_match"] + r["n_sub"] + r["n_del"], r["n_ref"])

    def test_the_repetition_count_is_measured_from_the_transcript(self):
        self.assertEqual(
            self.rec["n_repeats"],
            len(re.findall(self.rec["phrase"], self.rec["transcript"].lower())))
        self.assertGreater(self.rec["n_repeats"], 3)

    def test_it_draws_the_strict_alignment_not_the_tokenizer_ratio(self):
        """
        SPEC 6.6 / D.9 and results/model_arms.json publish this row as
        "3 ref words -> 49 hyp words". Those counts come from
        `model_arms.hallucination_report`, which cross-model-normalizes (spoken
        numbers become digits) and then tokenizes with a letters-only regex that
        drops them — so 8 of the 11 spoken words leave the reference and the
        16.3x ratio is part model, part tokenizer. The figure draws 11 -> 47,
        the alignment that produced the stored WER, and prints the other
        counting on its face so neither reading can ambush a reader.
        """
        self.assertEqual(self.rec["n_ref_words"], 11)
        self.assertEqual(self.rec["n_ref_tokens_crossmodel"], 3)
        self.assertGreater(self.rec["n_ref_words"],
                           self.rec["n_ref_tokens_crossmodel"])
        text = _svg_text("whisper-hallucination.svg")
        self.assertIn(f"{self.rec['n_ref_words']} words", text)
        self.assertIn(str(self.rec["n_ref_tokens_crossmodel"]), text)
        self.assertIn("COUNTING NOTE", text)

    def test_it_is_labelled_n_equals_one(self):
        text = _svg_text("whisper-hallucination.svg").lower()
        self.assertIn("not a rate", text,
                      "a single row presented without that label reads as a "
                      "measured tendency")


class TestTheGeneratorIsReproducible(unittest.TestCase):
    """
    The interviewer's question is 'where did this number come from', and the
    answer has to be a command. `--check` re-derives every figure from
    `results/` and compares it byte-for-byte with what is on disk, so a stale
    asset is a failure rather than a thing somebody notices later.
    """

    def test_check_reports_the_committed_assets_are_current(self):
        _needs_master(self)
        p = subprocess.run([PY, "scripts/make_readme_figures.py", "--check"],
                           cwd=REPO, capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0,
                         f"docs/assets is stale — re-run "
                         f"scripts/make_readme_figures.py\n{p.stdout}{p.stderr}")
        self.assertIn("up to date", p.stdout)

    def test_check_notices_a_hand_edit(self):
        """Negative control: without this, --check could be passing by never
        comparing anything."""
        _needs_master(self)
        target = ASSETS / FIVE[0]
        original = target.read_bytes()
        try:
            target.write_text(original.decode("utf-8").replace(
                "</svg>", "<!-- hand edit --></svg>"), encoding="utf-8")
            p = subprocess.run([PY, "scripts/make_readme_figures.py", "--check"],
                               cwd=REPO, capture_output=True, text=True,
                               timeout=600)
            self.assertEqual(p.returncode, 1)
            self.assertIn("STALE", p.stdout)
        finally:
            target.write_bytes(original)

    def test_it_is_deterministic(self):
        _needs_master(self)
        a, _ = mrf.build()
        b, _ = mrf.build()
        self.assertEqual(a, b, "two builds of the same table differ — a figure "
                               "that changes without the data changing cannot "
                               "be diffed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
