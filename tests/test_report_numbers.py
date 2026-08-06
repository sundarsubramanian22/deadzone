"""
Number-pinning for `report/writeup.md`: every load-bearing figure in the prose
is re-read from the artifact it came from, and this suite fails when the two
disagree.

WHY THIS EXISTS. Nothing else in this repo asserts anything about a prose
document. Every other layer is guarded — the trap functions, the loaders, the
merge paths — but the *deliverable* is a markdown file that quotes ~60 measured
numbers, and an artifact can be regenerated without the prose moving. That has
happened repeatedly and silently (SPEC C.7 lists five figures that survived
multiple reviews because they were copied forward from a progress log instead of
re-read from `results/`; SPEC C.5, C.8 and G are three more). It is this
project's signature failure mode — a computation that succeeds and emits a
plausible number — aimed at its own write-up.

WHAT IT PINS. The figures that would embarrass someone in an interview, not
every number in the document: the D1 headline and its dead-zone row, the
deletion-blindness figures, the exact-Sobol pre-registration gaps and their two
CI forms, the ECE triple, the sim2real level/order/transfer trio, the L1 and L1b
arm shapes, the active-learning null, the DRR mechanism and its four-room table,
the L3 decoupling thresholds, the "listening is not QA" pair, and the
reproducibility totals. 157 figures across 234 prose sites, from 13 artifacts,
in under half a second.

WHAT IT DELIBERATELY DOES NOT PIN. Numbers with no artifact behind them are left
alone rather than pinned against a constant: §6.7's repeat-call spread (a 6-clip
× 4-call probe the document itself flags as unpersisted), the NORMALIZED row of
§6.7's three-arm table (it needs both arms re-scored through `cross_model_norm`,
a much slower path than the raw rows support), and the AL split-robustness
medians. Prose that merely *describes* — verdict strings, mechanism claims — is
also out of scope; this suite pins numbers.

HOW IT PINS. Every expected value is READ FROM AN ARTIFACT — `results/*.json`,
`results/*.csv`, `results/*.txt`, or recomputed from `results/master.csv`. None
is typed as a constant, because a hardcoded expectation pins nothing; it only
moves the drift into the test. The prose is matched by anchored regex and
compared NUMERICALLY with a tolerance, so legitimate formatting (0.8294 →
"0.829", a rate written as a percentage, a gap written in points) passes and
only the value moving fails.

Three properties worth naming:

  * EVERY capture must agree. A figure quoted in §1, in §6.1 and again in an
    appendix table is captured at all three sites and all three must match the
    artifact — so the suite also catches the document disagreeing with *itself*.
  * MISSING ARTIFACTS SKIP, they do not fail. `results/` is gitignored, so a
    fresh checkout has none of it. A red suite there would teach people to
    ignore this file.
  * THE NEGATIVE CONTROL IS UNIVERSAL. `test_every_check_can_actually_fail`
    mutates the prose value behind EVERY check in turn and asserts that check
    then fails. A pinning test that passes against wrong prose is worse than no
    test, because it certifies the drift.

TWO DOCUMENTED TRAPS, both pinned explicitly (SPEC B.4, C.5, G):

  1. `results/dead_zones.csv` runs `..., mic_rolloff, rt60_measured, mean_conf,
     conf_pct, wer, ...` and on the headline row `rt60_measured = 0.4736` sits
     immediately before `mean_conf = 0.8294`. An off-by-one column read swaps
     delivered reverb time for headline confidence and still looks plausible; it
     happened once already. Columns are read BY NAME here, and
     `test_dead_zone_row_survives_the_column_trap` re-derives the row through the
     identity `gap_spoke = mean_conf - (1 - wer_spoke)` and asserts the prose's
     confidence is NOT the row's `rt60_measured`.
  2. That identity only holds under the same-subset pairing. `wer` aliases
     `wer_all_clips` while `gap` aliases `gap_spoke`, so on a row with
     `n_silent > 0` the two differ by construction — which is the whole §6.1
     correction. The check asserts `n_silent == 0` on the headline row rather
     than assuming it.

POPULATIONS ARE PART OF THE PIN. Several figures legitimately differ by
population and a mismatch there is a labelling question, not a wrong number:
D1 is 40 clips, L1 is the 10-clip subset every arm ran, D4 is scoped to the
10-clip intersection. Each check names its source artifact, and the two
dead-zone rates (1.14 % over 40 clips, 0.57 % over 10) are pinned SEPARATELY
against their own artifacts, with `test_the_two_dead_zone_rates_are_reconciled`
asserting the document still explains why they differ.

Offline, no API, no audio. Run:
    ./.venv/bin/python tests/test_report_numbers.py
"""

# --- repo-root bootstrap -------------------------------------------------
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)
# -------------------------------------------------------------------------

import csv
import json
import re
from collections import defaultdict

WRITEUP = "report/writeup.md"

# Every document this suite is willing to pin. Adding a prose file here is the
# only step needed to bring it under the gate.
DOCS = [WRITEUP]

# Tolerances, named by the decimal place the prose rounds to. A figure printed
# to 3 dp is pinned to half a unit in the last place, so 0.8294 -> "0.829"
# passes and 0.829 -> "0.831" does not.
TOL3 = 5e-4
TOL2 = 5e-3
TOL1 = 5e-2
TOL0 = 0.5
EXACT = 1e-9

csv.field_size_limit(10 ** 8)


class Drift(AssertionError):
    """A prose figure no longer matches the artifact it was read from."""


class MissingArtifact(Exception):
    """`results/` is gitignored; absence is a skip, never a failure."""


# =========================================================================
# check plumbing
# =========================================================================

class Check:
    """One pinned figure: prose sites on the left, artifact value on the right."""

    __slots__ = ("key", "doc", "patterns", "expected", "tol", "source", "where")

    def __init__(self, key, patterns, expected, tol, source, where, doc=WRITEUP):
        self.key = key
        self.doc = doc
        self.patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        self.expected = float(expected)
        self.tol = float(tol)
        self.source = source          # artifact path + field, for the failure message
        self.where = where            # section of the document, for the failure message

    def __repr__(self):
        return "Check(%s)" % self.key


_MINUS = "−"       # U+2212 MINUS SIGN, which the prose uses instead of '-'
_NBSP = " "
_THIN = " "


def parse_num(text):
    """'−0.980' / '1,760' / '91 %' / '12.1' -> float. Prose typography only."""
    s = text.strip().replace(_MINUS, "-").replace(_NBSP, " ").replace(_THIN, " ")
    s = s.replace(",", "").replace("%", "").replace("$", "").strip()
    return float(s)


def _line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def apply_check(text, chk):
    """Run one check against a document body. Raises Drift with a fixable message."""
    hits = []
    for pat in chk.patterns:
        found = list(re.finditer(pat, text))
        if not found:
            raise Drift(
                "\n  document : %s  (%s)"
                "\n  check    : %s"
                "\n  problem  : the prose site this figure was pinned at is GONE."
                "\n             pattern %r matched nothing."
                "\n  artifact : %s says %.6g"
                "\n  fix      : if the sentence was reworded, update the pattern in"
                "\n             tests/test_report_numbers.py; if the figure was"
                "\n             deleted, delete the check with it."
                % (chk.doc, chk.where, chk.key, pat, chk.source, chk.expected))
        for m in found:
            hits.append((m.group(1), m.start(1)))

    bad = []
    for raw, pos in hits:
        try:
            val = parse_num(raw)
        except ValueError:
            bad.append((raw, pos, None))
            continue
        if abs(val - chk.expected) > chk.tol:
            bad.append((raw, pos, val))

    if bad:
        sites = "; ".join(
            "line %d says %r" % (_line_of(text, pos), raw) for raw, pos, _ in bad)
        raise Drift(
            "\n  document : %s  (%s)"
            "\n  check    : %s"
            "\n  prose    : %s"
            "\n  artifact : %s says %.6g  (tolerance %g)"
            "\n  fix      : the artifact is the source of truth. Re-read it and"
            "\n             correct the document, or if the pipeline changed"
            "\n             deliberately, re-derive every figure in %s."
            % (chk.doc, chk.where, chk.key, sites, chk.source, chk.expected,
               chk.tol, chk.where))

    return len(hits)


# =========================================================================
# artifact readers  (every expected value below is READ, never typed)
# =========================================================================

def _read(path):
    if not os.path.exists(path):
        raise MissingArtifact(path)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _read_json(path):
    return json.loads(_read(path))


def _doc(path):
    if not os.path.exists(path):
        raise MissingArtifact(path)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _cg_block(model):
    """The `results/confidence_gap.txt` section for one model arm."""
    txt = _read("results/confidence_gap.txt")
    marker = "D1 confidence-accuracy gap — model '%s'" % model
    i = txt.find(marker)
    if i < 0:
        raise MissingArtifact("confidence_gap.txt has no block for %r" % model)
    j = txt.find("D1 confidence-accuracy gap — model '", i + len(marker))
    return txt[i: j if j > 0 else len(txt)]


def _grab(block, pattern, group=1):
    m = re.search(pattern, block)
    if m is None:
        raise MissingArtifact("artifact layout changed: %r not found" % pattern)
    return m.group(group)


def _dead_zone_rows(model):
    """`results/dead_zones.csv` rows for one model, read BY COLUMN NAME."""
    path = "results/dead_zones.csv"
    if not os.path.exists(path):
        raise MissingArtifact(path)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["model"] == model]
    if not rows:
        raise MissingArtifact("dead_zones.csv has no rows for %r" % model)
    return rows


def _headline_dead_zone(model="nova-3"):
    """The #1 dead zone: highest same-subset gap among rows categorised dead_zone."""
    rows = [r for r in _dead_zone_rows(model) if r["category"] == "dead_zone"]
    if not rows:
        raise MissingArtifact("no dead_zone rows for %r" % model)
    return max(rows, key=lambda r: float(r["gap_spoke"]))


_MASTER_CACHE = {}


def _master_nova(columns):
    """Non-failed nova-3 rows from `results/master.csv`, a few columns only."""
    key = tuple(sorted(columns))
    if key in _MASTER_CACHE:
        return _MASTER_CACHE[key]
    path = "results/master.csv"
    if not os.path.exists(path):
        raise MissingArtifact(path)
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["model"] != "nova-3":
                continue
            if str(r["failed"]).strip().lower() in ("true", "1"):
                continue
            out.append({c: r[c] for c in columns})
    _MASTER_CACHE[key] = out
    return out


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


# =========================================================================
# the checks, grouped by the artifact they are read from
# =========================================================================

def checks_d1_headline():
    """§1 + §6.1 — the silent-failure map. Source: results/confidence_gap.txt."""
    b = _cg_block("nova-3")
    src = "results/confidence_gap.txt [nova-3 block]"

    rho = float(_grab(b, r"global spearman\(conf_pct, WER_spoke\) = (-?[\d.]+)"))
    rho_all = float(_grab(b, r"\[all-clips pairing: (-?[\d.]+)\]"))
    gap_mean = float(_grab(b, r"gap \(same subset\) mean \+?(-?[\d.]+)"))
    over_pct = float(_grab(b, r"overconfident in (\d+)% of conditions"))
    n_dz = int(_grab(b, r"categories: (\d+) dead zone"))
    dz_pct = float(_grab(b, r"categories: \d+ dead zone \(([\d.]+)%\)"))
    n_sd = int(_grab(b, r"(\d+) silence-driven"))
    n_mute = int(_grab(b, r"(\d+) conditions silent on EVERY clip"))
    n_cond = int(_grab(b, r"conditions: (\d+)"))
    n_silent_rows = int(_grab(b, r"SILENT clip-rows[^:]*: (\d+) / \d+"))
    n_rows = int(_grab(b, r"SILENT clip-rows[^:]*: \d+ / (\d+)"))
    silent_pct = float(_grab(b, r"SILENT clip-rows[^:]*: \d+ / \d+ \(([\d.]+)%\)"))
    gap_mismatched = float(_grab(b, r"all-clips pairing, MISMATCHED\) mean \+?(-?[\d.]+)"))
    n_dz_mismatched = int(_grab(b, r"would have called (\d+) of them dead zones"))

    return [
        Check("D1 spearman(conf, WER), paired",
              [r"Spearman ρ = (−?-?[\d.]+)",
               r"its WER is \*\*(−?-?[\d.]+)\*\*",
               r"came back at \*\*(−?-?[\d.]+)\*\*",
               r"pairing correctly gives \*\*(−?-?[\d.]+) \(n = \d+\)"],
              rho, TOL3, src + " global spearman(conf_pct, WER_spoke)", "§1 / §6.1 / §10"),
        Check("D1 spearman, all-clips pairing",
              [r"all-clips pairing at \*\*(−?-?[\d.]+) \(n = \d+\)"],
              rho_all, TOL3, src + " [all-clips pairing]", "Appendix D.11b"),
        Check("D1 conditions that returned words (n)",
              [r"\*\*(\d+) of 176\*\* conditions that returned any words",
               r"pairing correctly gives \*\*−?-?[\d.]+ \(n = (\d+)\)"],
              n_cond - n_mute, EXACT, src + " conditions - mute conditions", "§1 / §6.1"),
        Check("D1 mean confidence gap",
              [r"mean gap \*\*\+([\d.]+)\*\*",
               r"= \*\*\+([\d.]+)\*\*\), with",
               r"0\.256 → \*\*\+([\d.]+)\*\*"],
              gap_mean, TOL3, src + " gap (same subset) mean", "§1 / §6.1"),
        Check("D1 overconfident share",
              [r"overconfident in (\d+) %"],
              over_pct, TOL0, src + " overconfident in N% of conditions", "§1 / §6.1"),
        Check("D1 dead-zone count (40-clip table)",
              [r"\*\*(\d+) of 176 \(\d\.\d+ %\)\*\*"],
              n_dz, TOL0, src + " categories: N dead zone", "§1 / §6.1"),
        Check("D1 dead-zone rate % (40-clip table)",
              [r"\*\*\d+ of 176 \((\d\.\d+) %\)\*\*",
               r"\*\*(\d\.\d+) % \(\d+/176\)\*\* in §6\.1"],
              dz_pct, TOL2, src + " categories: N dead zone (P%)", "§1 / §6.1 / §6.6"),
        Check("D1 silence-driven count",
              [r"\*\*(\d+) silence-driven\*\*", r"silence-driven \((\d+)\)"],
              n_sd, EXACT, src + " categories: N silence-driven", "§1 / §6.1"),
        Check("D1 mute-zone count",
              [r"\*\*(\d+) mute\s*\n?zones\*\*", r"mute zone \((\d+)\)",
               r"\*\*(\d+) mute zones\*\* below"],
              n_mute, EXACT, src + " conditions silent on EVERY clip", "§1 / §6.1"),
        Check("D1 pre-correction dead-zone count",
              [r"reported \*\*(\d+)\*\* dead\s*\n?zones at a mean gap",
               r"the count (\d+) → \*\*2\*\*"],
              n_dz_mismatched, EXACT, src + " would have called N of them dead zones",
              "§1 / §6.1"),
        Check("D1 pre-correction mean gap",
              [r"mean gap of\s*\n?(\d+\.\d+)", r"It moves the mean gap\s*\n(\d+\.\d+) →"],
              gap_mismatched, TOL3, src + " gap (all-clips pairing, MISMATCHED) mean",
              "§1 / §6.1"),
        Check("D1 silent clip-row share",
              [r"\*\*(\d+\.\d) %\s*\n?\s*of clip-rows produced no words"],
              silent_pct, TOL1, src + " SILENT clip-rows", "§8 limitation 7"),
        Check("D1 silent clip-rows / total (corpus-wide)",
              [r"nova-3's\nsilent rate is (\d+\.\d) %"],
              silent_pct, TOL1, src + " SILENT clip-rows", "§6.7 finding 3"),
        Check("D1 nova-3 clip-rows scored",
              [r"= ([\d,]+) transcriptions", r"covers all (\d+) Deepgram\nrows"],
              n_rows, EXACT, src + " clip-rows used", "§1 / §10"),
        Check("D1 silent clip-rows (count)",
              [r"\*\*(\d+) of 7040 clip-rows"], n_silent_rows, EXACT,
              src + " SILENT clip-rows", "Appendix D"),
    ]


def checks_dead_zone_row():
    """§1 + §6.1 — the #1 dead zone. Source: results/dead_zones.csv, BY NAME."""
    r = _headline_dead_zone("nova-3")
    src = "results/dead_zones.csv [%s]" % r["condition_name"]
    return [
        Check("#1 dead zone: mean word confidence",
              [r"mean\nword confidence ([\d.]+) at WER", r"confidence ([\d.]+) at WER 0\.306"],
              float(r["mean_conf"]), TOL3, src + " mean_conf", "§1 / §6.1"),
        Check("#1 dead zone: WER on the clips it spoke on",
              [r"confidence 0\.829 at WER ([\d.]+)"],
              float(r["wer_spoke"]), TOL3, src + " wer_spoke", "§1 / §6.1"),
        Check("#1 dead zone: clips that came back empty",
              [r"\*\*(\d+) of 40\*\* clips coming back empty",
               r"\*\*(\d+) of those 40 came back empty\*\*"],
              float(r["n_silent"]), EXACT, src + " n_silent", "§1 / §6.1"),
        Check("#1 dead zone: same-subset gap",
              [r"\| engine \| g726 \|[^\n|]*\|[^\n|]*\|[^\n|]*\|[^\n|]*\|[^\n|]*\| "
               r"\*\*\+(\d\.\d+)\*\* \|"],
              float(r["gap_spoke"]), TOL3, src + " gap_spoke", "Appendix D.1"),
    ]


def checks_fingerprints():
    """§6.2 — typed failure fingerprints. Source: fingerprints.json + calibration.json."""
    f = _read_json("results/fingerprints.json")["by_model"]["nova-3"]
    d = _read_json("results/calibration.json")["deletion_blindness"]
    n_ref = float(d["n_ref_words"])
    ent = f["entities"]["overall"]
    cls = f["inventory"]["by_class"]
    babble = f["insertions"]["by_group"]["babble"]
    fsrc = "results/fingerprints.json [by_model/nova-3]"
    csrc = "results/calibration.json [deletion_blindness]"
    return [
        Check("D2 reference words scored",
              [r"Across the ([\d,]+) reference words"],
              n_ref, EXACT, csrc + " n_ref_words", "§6.2"),
        Check("D2 deletion rate",
              [r"\*\*deletions ([\d.]+)\*\*", r"deletions \*\*([\d.]+)\*\* of reference words"],
              float(d["deleted_fraction_of_reference"]), TOL3,
              csrc + " deleted_fraction_of_reference", "§1 / §6.2"),
        Check("D2 substitution rate",
              [r"substitutions ([\d.]+), insertions", r"against substitutions ([\d.]+),"],
              float(d["n_substitutions"]) / n_ref, TOL3,
              csrc + " n_substitutions / n_ref_words", "§1 / §6.2"),
        Check("D2 insertion rate",
              [r"insertions\n?([\d.]+) — deletion is not one mechanism"],
              float(d["n_insertions"]) / n_ref, TOL3,
              csrc + " n_insertions / n_ref_words", "§6.2"),
        Check("D2 entity error rate",
              [r"entity error rate \*\*(\d+\.\d+)\*\*",
               r"entity error rate (\d+\.\d+) against WER"],
              float(ent["mean_entity_error_rate"]), TOL3,
              fsrc + " entities/overall/mean_entity_error_rate", "§1 / §6.2"),
        Check("D2 WER paired with the entity error rate",
              [r"against WER \*\*(\d+\.\d+)\*\*",
               r"against WER (\d+\.\d+)\*\* \(D\.3\)"],
              float(ent["mean_wer"]), TOL3,
              fsrc + " entities/overall/mean_wer", "§1 / §6.2"),
        Check("D2 destroyed-word rate, proper nouns",
              [r"\*\*([\d.]+) for proper nouns"],
              float(cls["proper_noun"]["destruction_rate"]), TOL3,
              fsrc + " inventory/by_class/proper_noun/destruction_rate", "§6.2"),
        Check("D2 destroyed-word rate, digit words",
              [r"against\n([\d.]+) for digit words"],
              float(cls["digit_word"]["destruction_rate"]), TOL3,
              fsrc + " inventory/by_class/digit_word/destruction_rate", "§6.2"),
        Check("D2 babble insertions that are foreign tokens",
              [r"\*\*(\d+) % foreign tokens\*\*"],
              100.0 * float(babble["foreign_frac"]), TOL0,
              fsrc + " insertions/by_group/babble/foreign_frac", "§6.2"),
    ]


def checks_sobol():
    """§6.3 — exact functional-ANOVA Sobol + the pre-registration. Source: sobol.json."""
    s = _read_json("results/sobol.json")
    src = "results/sobol.json"
    idx = {n: i for i, n in enumerate(s["names"])}
    gap = {g["factor"]: g for g in s["interaction_gap"]}
    top2 = s["s2_ranked"][0]
    order = s["variance_share_by_order"]

    out = [
        Check("Sobol partition sums to 1",
              [r"`sum\(S_u\) = (\d\.\d+)`"],
              float(s["variance_explained_check"]), 1e-9,
              src + " variance_explained_check", "§1 / §5 / §6.3"),
        Check("S2(rt60, snr_db)",
              [r"S2\(`rt60`, `snr_db`\) = \*\*([\d.]+)"],
              float(top2["S2"]), TOL3, src + " s2_ranked[0].S2", "§6.3"),
        Check("S2(rt60, snr_db) half-width",
              [r"S2\(`rt60`, `snr_db`\) = \*\*[\d.]+ ± ([\d.]+)"],
              float(top2["S2_conf"]), TOL3, src + " s2_ranked[0].S2_conf", "§6.3"),
        Check("first-order variance share",
              [r"First-order terms carry ([\d.]+) of the variance"],
              float(order["1"]), TOL3, src + " variance_share_by_order['1']", "§6.3"),
        Check("second-order variance share",
              [r"second-order ([\d.]+);"],
              float(order["2"]), TOL3, src + " variance_share_by_order['2']", "§6.3"),
    ]

    # the S1 / ST table, one check per cell
    for fac in ("snr_db", "rt60", "mic_rolloff", "codec"):
        i = idx[fac]
        row = r"\| `%s` \| " % fac
        out += [
            Check("S1[%s]" % fac, [row + r"([\d.]+) ±"],
                  float(s["S1"][i]), TOL3, src + " S1[%s]" % fac, "§6.3 table"),
            Check("ST[%s]" % fac, [row + r"[\d.]+ ± [\d.]+ \| ([\d.]+) ±"],
                  float(s["ST"][i]), TOL3, src + " ST[%s]" % fac, "§6.3 table"),
            Check("ST-S1 gap[%s]" % fac,
                  [row + r"[\d.]+ ± [\d.]+ \| [\d.]+ ± [\d.]+ \| \*?\*?(\d+\.\d+)\*?\*? \|"],
                  float(gap[fac]["gap"]), TOL3, src + " interaction_gap[%s].gap" % fac,
                  "§6.3 table"),
        ]

    # the two pre-registered factors' quadrature CIs, quoted in the table AND in
    # the verdict blockquote — both sites captured, both must agree.
    for fac in ("rt60", "snr_db"):
        g = gap[fac]
        lo = g["gap"] - g["gap_conf_quadrature"]
        hi = g["gap"] + g["gap_conf_quadrature"]
        val = "%.3f" % g["gap"]
        out += [
            Check("pre-registration CI lo[%s]" % fac,
                  [r"\*\*%s\*\* \| \[([\d.]+), [\d.]+\]" % val,
                   r"\*\*%s \[([\d.]+), [\d.]+\]\*\*" % val],
                  lo, TOL3, src + " interaction_gap[%s] gap - gap_conf_quadrature" % fac,
                  "§1 / §6.3"),
            Check("pre-registration CI hi[%s]" % fac,
                  [r"\*\*%s\*\* \| \[[\d.]+, ([\d.]+)\]" % val,
                   r"\*\*%s \[[\d.]+, ([\d.]+)\]\*\*" % val],
                  hi, TOL3, src + " interaction_gap[%s] gap + gap_conf_quadrature" % fac,
                  "§1 / §6.3"),
        ]

    # the widening factors: the document claims the quadrature interval is N x
    # the direct one, per factor. Both forms are persisted, so the ratio is too.
    widen = {f: gap[f]["gap_conf_quadrature"] / gap[f]["gap_conf_direct"]
             for f in ("rt60", "snr_db", "mic_rolloff", "codec")}
    out += [
        Check("quadrature/direct widening [rt60]",
              [r"\*\*([\d.]+)× for `rt60`"], widen["rt60"], TOL2,
              src + " gap_conf_quadrature / gap_conf_direct", "§6.3"),
        Check("quadrature/direct widening [snr_db]",
              [r"([\d.]+)× for `snr_db`"], widen["snr_db"], TOL2,
              src + " gap_conf_quadrature / gap_conf_direct", "§6.3"),
        Check("quadrature/direct widening [mic_rolloff]",
              [r"([\d.]+)× for `mic_rolloff`"], widen["mic_rolloff"], TOL2,
              src + " gap_conf_quadrature / gap_conf_direct", "§6.3"),
        Check("quadrature/direct widening [codec]",
              [r"only ([\d.]+)× for `codec`"], widen["codec"], TOL2,
              src + " gap_conf_quadrature / gap_conf_direct", "§6.3"),
        Check("weakest clearance over the 0.020 threshold",
              [r"clears the pre-set 0\.020 threshold by\n?([\d.]+)×",
               r"still clears the threshold by \*\*([\d.]+)×\*\*"],
              (gap["snr_db"]["gap"] - gap["snr_db"]["gap_conf_quadrature"]) / 0.020,
              TOL2, src + " snr_db quadrature lower bound / 0.020", "§6.3"),
    ]
    return out


def checks_drr():
    """§6.3 — the DRR mechanism. Source: interactions.json + master.csv marginals."""
    it = _read_json("results/interactions.json")
    corr = it["rir_mechanism"]["correlations"]
    dip = [m for m in it["measured_counterintuitive"]["marginal"]
           if m["factor"] == "rt60"][0]
    src = "results/interactions.json [rir_mechanism]"

    # the rt60 marginal, recomputed from the measured grid rather than read from
    # a summary: the 144-cell babble factorial, cell means then level means.
    rows = _master_nova(["condition_name", "rt60", "snr_db", "noise_type",
                         "codec", "mic_rolloff", "wer"])
    cells = defaultdict(list)
    for r in rows:
        if r["noise_type"] != "babble":
            continue
        cells[(r["rt60"], r["snr_db"], r["codec"], r["mic_rolloff"])].append(float(r["wer"]))
    per_level = defaultdict(list)
    for (rt60, _s, _c, _m), wers in cells.items():
        per_level[float(rt60)].append(_mean(wers))
    marg = {lvl: _mean(v) for lvl, v in per_level.items()}
    msrc = "results/master.csv [nova-3, babble 144-cell factorial marginal]"

    out = [
        Check("spearman(DRR, WER)",
              [r"ρ\(DRR, WER\) = (−?-?[\d.]+)",
               r"`spearman\(DRR, WER\) = (−?-?[\d.]+)`"],
              float(corr["drr_db"]["spearman"]), TOL3,
              src + " correlations.drr_db.spearman", "§1 / §6.3"),
        Check("spearman(RT60, WER)",
              [r"ρ\(RT60, WER\) = \+?(−?-?[\d.]+)",
               r"`spearman\(RT60, WER\) = \+?(−?-?[\d.]+)`"],
              float(corr["rt60_measured"]["spearman"]), TOL3,
              src + " correlations.rt60_measured.spearman", "§1 / §6.3"),
        Check("rt60 dip depth",
              [r"a dip at 0\.7 of depth \*\*([\d.]+)"],
              float(dip["depth"]), TOL3, src + " marginal[rt60].depth", "§6.3"),
        Check("rt60 dip depth CI lo",
              [r"depth \*\*[\d.]+\n\[([\d.]+), [\d.]+\]\*\*"],
              float(dip["depth_ci_lo"]), TOL3, src + " marginal[rt60].depth_ci_lo", "§6.3"),
        Check("rt60 dip depth CI hi",
              [r"depth \*\*[\d.]+\n\[[\d.]+, ([\d.]+)\]\*\*"],
              float(dip["depth_ci_hi"]), TOL3, src + " marginal[rt60].depth_ci_hi", "§6.3"),
    ]

    # the four-room table: measured RT60, DRR, C50 per requested level, plus the
    # marginal WER the level actually delivered.
    for lvl in it["rir_mechanism"]["levels"]:
        req = lvl["rt60_requested"]
        # the table prints the requested level as the document writes it: at
        # least one decimal, trailing zeros stripped (1.0, 0.45, 0.7, 0.2)
        req_s = ("%.2f" % req).rstrip("0")
        req_s = req_s + "0" if req_s.endswith(".") else req_s
        row = r"\| %s \| [A-Za-z ]+ \| " % re.escape(req_s)
        out += [
            Check("room table measured RT60 @ rt60=%s" % req_s,
                  [row + r"([\d.]+) \|"], float(lvl["rt60_measured"]), TOL3,
                  src + " levels[%s].rt60_measured" % req_s, "§6.3 room table"),
            Check("room table DRR @ rt60=%s" % req_s,
                  [row + r"[\d.]+ \| \*?\*?(−?-?[\d.]+)\*?\*? \|"],
                  float(lvl["drr_db"]), TOL2,
                  src + " levels[%s].drr_db" % req_s, "§6.3 room table"),
            Check("room table C50 @ rt60=%s" % req_s,
                  [row + r"[\d.]+ \| \*?\*?−?-?[\d.]+\*?\*? \| (−?-?[\d.]+) \|"],
                  float(lvl["c50_db"]), TOL2,
                  src + " levels[%s].c50_db" % req_s, "§6.3 room table"),
            Check("rt60 marginal WER @ rt60=%s" % req_s,
                  [row + r"[\d.]+ \| \*?\*?−?-?[\d.]+\*?\*? \| −?-?[\d.]+ \| ([\d.]+) \|"],
                  marg[float(req)], TOL3,
                  msrc + " level %s" % req_s, "§6.3 room table"),
        ]

    # and the same marginal quoted as a sequence in §1 and §6.3
    seq = [marg[k] for k in sorted(marg)]
    out.append(Check(
        "rt60 marginal sequence",
        [r"— ([\d.]+) → [\d.]+ → [\d.]+ → [\d.]+ —",
         r"non-monotonic — ([\d.]+) →"],
        seq[0], TOL3, msrc + " level 0.2", "§1 / §6.3"))
    return out


def checks_calibration():
    """§6.4 + limitation 7 — ECE and deletion blindness. Source: calibration.json."""
    c = _read_json("results/calibration.json")
    p = c["primary"]
    d = c["deletion_blindness"]
    src = "results/calibration.json"
    return [
        Check("ECE raw",
              [r"cuts ECE from ([\d.]+) raw", r"ECE ([\d.]+) → 0\.008",
               r"\| raw confidence \| \*\*([\d.]+)\*\*"],
              float(p["ece_raw"]["median"]), TOL3,
              src + " primary.ece_raw.median", "§1 / §6.4 / D.10"),
        Check("ECE after temperature scaling",
              [r"against\n([\d.]+) for a global temperature",
               r"\| \+ temperature scaling[^|]*\| \*\*([\d.]+)\*\*"],
              float(p["ece_temperature"]["median"]), TOL3,
              src + " primary.ece_temperature.median", "§6.4 / D.10"),
        Check("ECE after the feature-conditioned calibrator",
              [r"raw to ([\d.]+)\*\*", r"ECE 0\.051 → ([\d.]+)",
               r"\| \+ feature-conditioned \| \*\*([\d.]+)\*\*"],
              float(p["ece_feature"]["median"]), TOL3,
              src + " primary.ece_feature.median", "§1 / §6.4 / D.10"),
        Check("calibration conditions (groups)",
              [r"fit on the (\d+) conditions"],
              float(p["n_groups"]), EXACT, src + " primary.n_groups", "§6.4"),
        Check("deleted fraction of reference words",
              [r"(\d+\.\d) % of reference words",
               r"(\d+\.\d) % of the [\d,]+ reference words"],
              100.0 * float(d["deleted_fraction_of_reference"]), TOL1,
              src + " deletion_blindness.deleted_fraction_of_reference",
              "§6.7 / §8 limitation 7"),
        Check("deleted fraction of all errors",
              [r"(\d+\.\d) % of all errors", r"(\d+\.\d) % of errors"],
              100.0 * float(d["deleted_fraction_of_errors"]), TOL1,
              src + " deletion_blindness.deleted_fraction_of_errors",
              "§6.7 / §8 limitation 7"),
        Check("emitted-word accuracy",
              [r"emitted-word accuracy \*\*([\d.]+)\*\*",
               r"\| emitted-word accuracy \| \*\*([\d.]+)\*\*"],
              float(d["emitted_word_accuracy"]), TOL3,
              src + " deletion_blindness.emitted_word_accuracy", "§6.1 / D.11"),
        Check("reference-word recovery",
              [r"reference recovery \*\*([\d.]+)\*\*",
               r"\| reference recovery \| \*\*([\d.]+)\*\*"],
              float(d["reference_word_recovery"]), TOL3,
              src + " deletion_blindness.reference_word_recovery", "§6.1 / D.11"),
        Check("survivor-bias overstatement",
              [r"overstatement of \*\*([\d.]+)\*\*", r"\| overstatement \| \*\*([\d.]+)\*\*"],
              float(d["emitted_word_accuracy"]) - float(d["reference_word_recovery"]),
              TOL3, src + " emitted_word_accuracy - reference_word_recovery",
              "§6.1 / D.11"),
        Check("deleted reference words (count)",
              [r"are \*\*([\d,]+) words = \d+\.\d % of the"],
              float(d["n_deletions"]), EXACT,
              src + " deletion_blindness.n_deletions", "D.11"),
    ]


def checks_calibration_statement():
    """§6.4 — the plain-language discount. Source: calibration.json statement."""
    st = _read_json("results/calibration.json")["statement"]
    src = "results/calibration.json [statement]"
    m = re.search(r"Above rt60 = 0\.7[^.]*?~([\d.]+) to become a calibrated probability "
                  r"\(([\d.]+) reported vs ([\d.]+) observed accuracy on (\d+) held-out words",
                  st)
    if m is None:
        raise MissingArtifact("calibration statement layout changed")
    disc, rep, obs, nw = (float(m.group(1)), float(m.group(2)),
                          float(m.group(3)), float(m.group(4)))
    return [
        Check("rt60 confidence discount",
              [r"discounted by\n?~([\d.]+)\*\* to become"], disc, TOL2, src, "§6.4"),
        Check("rt60 reported confidence",
              [r"\(([\d.]+) reported vs [\d.]+ observed, \d+ held-out words\)"],
              rep, TOL2, src, "§6.4"),
        Check("rt60 observed accuracy",
              [r"\([\d.]+ reported vs ([\d.]+) observed, \d+ held-out words\)"],
              obs, TOL2, src, "§6.4"),
        Check("rt60 held-out words",
              [r"\([\d.]+ reported vs [\d.]+ observed, (\d+) held-out words\)"],
              nw, EXACT, src, "§6.4"),
    ]


def checks_active_learning():
    """§6.5 — the AL null. Source: al_savings.json."""
    a = _read_json("results/al_savings.json")
    h = a["headline"]
    src = "results/al_savings.json [headline]"
    return [
        Check("AL boundary_rmse target",
              [r"target\n([\d.]+) was reached"],
              float(h["target"]), TOL3, src + " target", "§6.5"),
        Check("AL seeds reaching target, active arm",
              [r"\*\*(\d+) of 8 seeds\*\*", r"reached by (\d+) of 8 active seeds"],
              float(h["n_seeds_reaching_target"]["active_boundary"]), EXACT,
              src + " n_seeds_reaching_target.active_boundary", "§1 / §6.5"),
        Check("AL seeds reaching target, random arm",
              [r"random's \*\*(\d+) of 8\*\*", r"and (\d+) of 8 random seeds"],
              float(h["n_seeds_reaching_target"]["random"]), EXACT,
              src + " n_seeds_reaching_target.random", "§1 / §6.5"),
        Check("AL evaluation budget",
              [r"inside a (\d+)-evaluation budget", r"inside the (\d+)-evaluation"],
              float(a["n_total"]), EXACT, "results/al_savings.json n_total", "§1 / §6.5"),
        Check("AL seed count",
              [r"all (\d+) seeds\nran against the surrogate oracle"],
              float(a["n_seeds"]), EXACT, "results/al_savings.json n_seeds", "§6.5"),
    ]


def checks_model_arms():
    """§6.6 + §6.7 — the multi-model comparison. Source: model_arms.json."""
    m = _read_json("results/model_arms.json")
    pm = m["per_model"]
    nova, wh = pm["nova-3"], pm["whisper-base"]
    scribe = pm.get("elevenlabs-scribe")
    src = "results/model_arms.json [per_model]"
    pair = m["dead_zone_overlap"]["pairwise"]["nova-3|whisper-base"]

    out = [
        Check("L1 nova-3 confidence-vs-WER shape",
              [r"nova-3 ρ = (−?-?[\d.]+) \(n = \d+\)"],
              float(nova["shape"]["spearman"]), TOL3,
              src + " nova-3.shape.spearman", "§6.6"),
        Check("L1 nova-3 shape n",
              [r"nova-3 ρ = −?-?[\d.]+ \(n = (\d+)\)"],
              float(nova["shape"]["n"]), EXACT, src + " nova-3.shape.n", "§6.6"),
        Check("L1 whisper-base confidence-vs-WER shape",
              [r"whisper-base ρ = (−?-?[\d.]+) \(n = \d+\)"],
              float(wh["shape"]["spearman"]), TOL3,
              src + " whisper-base.shape.spearman", "§6.6"),
        Check("L1 whisper-base shape n",
              [r"whisper-base ρ = −?-?[\d.]+ \(n = (\d+)\)"],
              float(wh["shape"]["n"]), EXACT, src + " whisper-base.shape.n", "§6.6"),
        Check("L1 nova-3 dead-zone rate (10-clip subset)",
              [r"\*\*(\d\.\d+) % \(\d+/176\)\*\* here", r"\*\*(\d\.\d+) % vs \d+\.\d+ %\*\*"],
              100.0 * float(nova["dead_zone_rate"]), TOL2,
              src + " nova-3.dead_zone_rate", "§6.6"),
        Check("L1 whisper-base dead-zone rate",
              [r"\*\*\d\.\d+ % vs (\d+\.\d+) %\*\*"],
              100.0 * float(wh["dead_zone_rate"]), TOL2,
              src + " whisper-base.dead_zone_rate", "§6.6"),
        Check("L1 nova|whisper dead-zone Jaccard",
              [r"shared dead\nzones 0, Jaccard (\d\.\d+)\.", r"Jaccard \*\*(\d\.\d+)\*\*\)"],
              float(pair["jaccard"]), EXACT,
              "results/model_arms.json dead_zone_overlap.pairwise[nova-3|whisper-base]",
              "§1 / §6.6"),
        Check("L1 nova-3 all-clips WER",
              [r"nova-3's WER \(([\d.]+) →"],
              float(nova["wer_mean_strict"]), TOL3,
              src + " nova-3.wer_mean_strict", "§6.6"),
        Check("L1 nova-3 spoke-subset WER",
              [r"nova-3's WER \([\d.]+ → \*\*([\d.]+)\*\*\)"],
              float(nova["wer_mean_strict_spoke"]), TOL3,
              src + " nova-3.wer_mean_strict_spoke", "§6.6"),
        Check("L1 whisper-base all-clips WER",
              [r"raises\* whisper-base's \(([\d.]+) →"],
              float(wh["wer_mean_strict"]), TOL3,
              src + " whisper-base.wer_mean_strict", "§6.6"),
        Check("L1 whisper-base spoke-subset WER",
              [r"whisper-base's \([\d.]+ → \*\*([\d.]+)\*\*\)"],
              float(wh["wer_mean_strict_spoke"]), TOL3,
              src + " whisper-base.wer_mean_strict_spoke", "§6.6"),
        Check("L1 matched rows per arm",
              [r"\*\*n = (\d+) rows per\nmodel\*\*"],
              float(m["arm_census"]["n_common_cells"]), EXACT,
              "results/model_arms.json arm_census.n_common_cells", "§6.6"),
    ]

    # Edit signatures. §6.6 compares the arms on the CROSS-MODEL scoring (the
    # only one in which absolute rates are comparable across arms) and §6.7
    # quotes the STRICT one plus what normalization does to it. Both keys are
    # required rather than defaulted: a `.get()` here would silently drop the
    # check if the artifact's layout moved, which is the failure mode this whole
    # file exists to catch.
    if "edit_signature_crossmodel" not in nova or "edit_signature_strict" not in nova:
        raise MissingArtifact("model_arms.json per_model has no edit_signature_* keys")
    xm = {k: v["edit_signature_crossmodel"] for k, v in pm.items()}
    st = {k: v["edit_signature_strict"] for k, v in pm.items()}
    out.append(Check(
        "L1 whisper insertion rate over nova-3's",
        [r"insertions \*\*(\d+\.\d+)×\*\* nova-3's rate"],
        float(xm["whisper-base"]["ins"]) / float(xm["nova-3"]["ins"]), TOL1,
        "results/model_arms.json per_model[*].edit_signature_crossmodel ins ratio",
        "§6.6"))
    if scribe is not None:
        out += [
            Check("L1b scribe strict substitution rate",
                  [r"strict sub (\d\.\d+) / del"], float(st["elevenlabs-scribe"]["sub"]),
                  TOL3, src + " elevenlabs-scribe.edit_signature_strict.sub", "§6.7"),
            Check("L1b scribe strict deletion rate",
                  [r"strict sub \d\.\d+ / del (\d\.\d+) against",
                   r"Scribe's deletions \((\d\.\d+) → \d\.\d+\)"],
                  float(st["elevenlabs-scribe"]["del"]), TOL3,
                  src + " elevenlabs-scribe.edit_signature_strict.del", "§6.7"),
            Check("L1b nova-3 strict substitution rate",
                  [r"against nova-3's (\d\.\d+) / \d\.\d+\)"],
                  float(st["nova-3"]["sub"]), TOL3,
                  src + " nova-3.edit_signature_strict.sub", "§6.7"),
            Check("L1b nova-3 strict deletion rate",
                  [r"against nova-3's \d\.\d+ / (\d\.\d+)\)",
                   r"not at all \((\d\.\d+) → \d\.\d+\)"],
                  float(st["nova-3"]["del"]), TOL3,
                  src + " nova-3.edit_signature_strict.del", "§6.7"),
            Check("L1b scribe deletion rate after normalization",
                  [r"Scribe's deletions \(\d\.\d+ → (\d\.\d+)\)"],
                  float(xm["elevenlabs-scribe"]["del"]), TOL3,
                  src + " elevenlabs-scribe.edit_signature_crossmodel.del", "§6.7"),
            Check("L1b nova-3 deletion rate after normalization",
                  [r"not at all \(\d\.\d+ → (\d\.\d+)\)"],
                  float(xm["nova-3"]["del"]), TOL3,
                  src + " nova-3.edit_signature_crossmodel.del", "§6.7"),
        ]

    if scribe is not None:
        n_rows_nova = float(nova["silence"]["n_rows"])
        out += [
            Check("L1b nova-3 empty-transcript rate (matched subset)",
                  [r"empty transcript on (\d+\.\d) %"],
                  100.0 * float(nova["silence"]["silent_rate"]), TOL1,
                  src + " nova-3.silence.silent_rate", "§6.7 finding 3"),
            Check("L1b nova-3 silent rows (matched subset)",
                  [r"\((\d+)/1757\) and goes fully"],
                  float(nova["silence"]["n_silent"]), EXACT,
                  src + " nova-3.silence.n_silent", "§6.7 finding 3"),
            Check("L1b nova-3 mute conditions (matched subset)",
                  [r"goes fully \*\*mute on\n(\d+)\*\* conditions",
                   r"after its (\d+) hardest were dropped"],
                  float(nova["n_mute_zones"]), EXACT,
                  src + " nova-3.n_mute_zones", "§6.7 finding 3"),
            Check("L1b scribe empty-transcript rate",
                  [r"Scribe on \*\*(\d+\.\d) %\*\*"],
                  100.0 * float(scribe["silence"]["silent_rate"]), TOL1,
                  src + " elevenlabs-scribe.silence.silent_rate", "§6.7 finding 3"),
            Check("L1b scribe silent rows",
                  [r"\((\d+)/1757\) and \*\*2\*\*"],
                  float(scribe["silence"]["n_silent"]), EXACT,
                  src + " elevenlabs-scribe.silence.n_silent", "§6.7 finding 3"),
            Check("L1b scribe dead zones, strict",
                  [r"not quotable\*\*: (\d+) of 176 \(\d\.\d+ %\)\nunder strict scoring"],
                  float(scribe["n_dead_zones"]), EXACT,
                  src + " elevenlabs-scribe.n_dead_zones", "§6.7"),
            Check("L1b scribe dead-zone rate, strict",
                  [r"not quotable\*\*: \d+ of 176 \((\d\.\d+) %\)\nunder strict scoring"],
                  100.0 * float(scribe["dead_zone_rate"]), TOL2,
                  src + " elevenlabs-scribe.dead_zone_rate", "§6.7"),
            Check("L1b scribe shape n",
                  [r"Scribe's over (\d+) after 2"],
                  float(scribe["shape"]["n"]), EXACT,
                  src + " elevenlabs-scribe.shape.n", "§6.7 finding 3"),
            Check("L1b nova-3 shape n on the matched subset",
                  [r"over (\d+) conditions after its 12 hardest"],
                  float(nova["shape"]["n"]), EXACT,
                  src + " nova-3.shape.n", "§6.7 finding 3"),
        ]
        rate_ratio = (float(nova["silence"]["silent_rate"])
                      / float(scribe["silence"]["silent_rate"]))
        out.append(Check(
            "L1b failure-mode ratio",
            [r"failure modes differ ([\d.]+)×"], rate_ratio, TOL1,
            src + " nova-3.silence.silent_rate / elevenlabs-scribe.silence.silent_rate",
            "§6.7 finding 3"))
    return out


def checks_sim2real():
    """§7 — the sim-vs-real gap. Source: sim2real.json."""
    h = _read_json("results/sim2real.json")["nova-3"]["headline"]
    src = "results/sim2real.json [nova-3.headline]"
    return [
        Check("D4 level gap (points)",
              [r"underestimates WER by ([\d.]+) points",
               r"read \*\*([\d.]+) points optimistic\*\*"],
              abs(float(h["mean_gap"])) * 100.0, TOL1, src + " mean_gap", "§1 / §7"),
        Check("D4 level gap CI lo",
              [r"\[−(\d+\.\d), −\d+\.\d\]"],
              abs(float(h["ci"][0])) * 100.0, TOL1, src + " ci[0]", "§1 / §7"),
        Check("D4 level gap CI hi",
              [r"\[−\d+\.\d, −(\d+\.\d)\]"],
              abs(float(h["ci"][1])) * 100.0, TOL1, src + " ci[1]", "§1 / §7"),
        Check("D4 rank correlation",
              [r"rank conditions well \(\*\*ρ = ([\d.]+)\*\*\)",
               r"Spearman \*\*ρ = ([\d.]+)\*\*"],
              float(h["spearman"]), TOL3, src + " spearman", "§1 / §7"),
        Check("D4 Kendall tau",
              [r"Kendall τ = ([\d.]+)"],
              float(h["kendall"]), TOL3, src + " kendall", "§7"),
        Check("D4 dead-zone Jaccard",
              [r"\*\*Jaccard ([\d.]+), recall [\d.]+\*\*",
               r"\(\*\*Jaccard ([\d.]+)\*\*\)"],
              float(h["dead_zone_jaccard"]), EXACT, src + " dead_zone_jaccard", "§1 / §7"),
        Check("D4 real dead zones (10-clip scope)",
              [r"\| real (\d+), sim \d+, both \d+ →"],
              float(h["n_dead_zones_real"]), EXACT, src + " n_dead_zones_real", "§7"),
        Check("D4 sim dead zones (10-clip scope)",
              [r"\| real \d+, sim (\d+), both \d+ →"],
              float(h["n_dead_zones_sim"]), EXACT, src + " n_dead_zones_sim", "§7"),
        Check("D4 real mute zones",
              [r"real \*\*(\d+)\*\*,? sim"],
              float(h["n_mute_real"]), EXACT, src + " n_mute_real", "§7"),
        Check("D4 sim mute zones",
              [r"sim \*\*(\d+)\*\* — the simulation"],
              float(h["n_mute_sim"]), EXACT, src + " n_mute_sim", "§7"),
        Check("D4 paired conditions",
              [r"\*\*n_pairs = (\d+)\*\*"],
              float(h["n_pairs"]), EXACT, src + " n_pairs", "§7"),
    ]


def checks_l3():
    """§6.8 — paralinguistic/lexical decoupling. Source: l3_decoupling.json."""
    f = _read_json("results/l3_decoupling.json")["factors"]
    a = f["rt60@opus_roll1"]["headline"]
    b = f["snr_db@g726_roll1"]["headline"]
    src = "results/l3_decoupling.json [factors]"
    return [
        Check("L3 f0 half-degradation level (rt60)",
              [r"`f0` collapses at rt60 ≈ (\d+\.\d+)"],
              float(a["feature_half_level"]), TOL2,
              src + " rt60@opus_roll1.headline.feature_half_level", "§6.8"),
        Check("L3 lexical half-degradation level (rt60)",
              [r"while WER only halves at\n≈ (\d+\.\d+)\*\*",
               r"`f0` collapses at rt60 ≈ \d+\.\d+, WER halves at ≈ (\d+\.\d+) "],
              float(a["lexical_half_level"]), TOL2,
              src + " rt60@opus_roll1.headline.lexical_half_level", "§6.8"),
        Check("L3 rms half-degradation level (snr_db)",
              [r"`rms` collapses at ≈ (\d+\.\d+) dB"],
              float(b["feature_half_level"]), TOL2,
              src + " snr_db@g726_roll1.headline.feature_half_level", "§6.8"),
        Check("L3 lexical half-degradation level (snr_db)",
              [r"`rms` collapses at ≈ \d+\.\d+ dB while WER halves at ≈ (\d+\.\d+) dB",
               r"`rms` collapses at ≈ \d+\.\d+ dB, WER halves at ≈ (\d+\.\d+) dB"],
              float(b["lexical_half_level"]), TOL2,
              src + " snr_db@g726_roll1.headline.lexical_half_level", "§6.8"),
    ]


def checks_indistinguishable_pair():
    """§6.3 corollary — 'listening is not QA'. Recomputed from master.csv."""
    rows = _master_nova(["clip_id", "condition_name", "wer"])
    A = "rt60-1_snr-20_babble_none_roll-0"      # Shower RIR, quiet
    B = "rt60-0.2_snr-0_babble_none_roll-0"     # Restaurant RIR, buried
    a = {r["clip_id"]: float(r["wer"]) for r in rows if r["condition_name"] == A}
    b = {r["clip_id"]: float(r["wer"]) for r in rows if r["condition_name"] == B}
    common = sorted(set(a) & set(b))
    if len(common) < 10:
        raise MissingArtifact("master.csv has no A/B pair for the §6.3 corollary")
    diffs = [a[c] - b[c] for c in common]
    n_same = sum(1 for d in diffs if d == 0.0)
    src = "results/master.csv [%s vs %s, nova-3, paired over %d clips]" % (A, B, len(common))
    return [
        Check("A/B corollary: WER of the drenched-but-quiet condition",
              [r"drenched but quiet\), WER \*\*(\d+\.\d+)\*\*",
               r"the model\*\* — WER \*\*(\d+\.\d+)\*\*\n\s*against"],
              _mean(a[c] for c in common), TOL3, src + " condition A mean WER", "§1 / §6.3"),
        Check("A/B corollary: WER of the dry-but-buried condition",
              [r"dry but buried in babble\), WER \*\*(\d+\.\d+)\*\*",
               r"against \*\*(\d+\.\d+)\*\*, paired difference"],
              _mean(b[c] for c in common), TOL3, src + " condition B mean WER", "§1 / §6.3"),
        Check("A/B corollary: paired difference",
              [r"paired difference \*\*−(\d+\.\d+),"],
              abs(_mean(diffs)), TOL3, src + " mean paired difference", "§1 / §6.3"),
        Check("A/B corollary: clips scoring identically",
              [r"with (\d+) of 40 clips scoring \*identically\*",
               r"\*\*(\d+) of 40\*\*\n\s*clips scoring identically"],
              float(n_same), EXACT, src + " clips with zero paired difference", "§1 / §6.3"),
    ]


def _rank(xs):
    """Average ranks, ties shared — the rank transform Spearman is defined on."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys):
    rx, ry = _rank(xs), _rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy)


def checks_three_arm_shape():
    """§6.7 Finding 2, STRICT row — recomputed from master.csv.

    This table exists in no artifact: it was computed while the section was
    written, on the conditions all three arms spoke on, which is the only
    population in which three arms can be ranked. So it is re-derived here from
    the raw rows rather than read from a summary — which is the stronger pin, and
    the reason this group is longer than the others.

    Only the STRICT row is pinned. The NORMALIZED row needs both arms re-scored
    through `cross_model_norm`, which is a different (and much slower) code path;
    the strict row is what the raw table supports.
    """
    path = "results/master.csv"
    if not os.path.exists(path):
        raise MissingArtifact(path)

    # every non-failed row, all arms, only the fields the shape needs
    by_model = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r["failed"]).strip().lower() in ("true", "1"):
                continue
            by_model[r["model"]].append(
                (r["clip_id"], r["condition_name"], r["wer"], r["mean_conf"]))
    if len(by_model) < 3:
        raise MissingArtifact("master.csv has %d arms, needs 3" % len(by_model))

    clips = set.intersection(*[{c for c, _n, _w, _m in rows}
                               for rows in by_model.values()])

    # "spoke" = the arm returned a confidence, i.e. it emitted words. That is
    # the same population rule §6.1's correction turns on, applied to three arms.
    per_model = {}
    for model, rows in by_model.items():
        agg = defaultdict(list)
        for clip, cond, wer, conf in rows:
            if clip not in clips or conf in ("", "nan", None):
                continue
            agg[cond].append((float(wer), float(conf)))
        per_model[model] = {
            cond: (_mean(w for w, _c in v), _mean(c for _w, c in v))
            for cond, v in agg.items() if v}

    common = sorted(set.intersection(*[set(t) for t in per_model.values()]))
    src = ("results/master.csv [%d conditions all %d arms spoke on, %d common clips]"
           % (len(common), len(per_model), len(clips)))

    out = [Check("L1b three-arm common conditions",
                 [r"Read on the (\d+)\nconditions \*\*all three arms spoke on\*\*"],
                 float(len(common)), EXACT, src + " condition count", "§6.7 finding 2")]
    row = r"\| \*\*strict\*\* \(spine scorer\) \| "
    cols = {"nova-3": row + r"(−?-?[\d.]+) \|",
            "elevenlabs-scribe": row + r"−?-?[\d.]+ \| (−?-?[\d.]+) \|",
            "whisper-base": row + r"−?-?[\d.]+ \| −?-?[\d.]+ \| (−?-?[\d.]+) \|"}
    for model, pat in cols.items():
        if model not in per_model:
            continue
        wers = [per_model[model][c][0] for c in common]
        confs = [per_model[model][c][1] for c in common]
        out.append(Check(
            "L1b strict confidence-vs-WER shape [%s]" % model,
            [pat], _spearman(confs, wers), TOL3,
            src + " spearman(mean_conf, wer) for %s" % model, "§6.7 table"))
    return out


def checks_manifest():
    """§1 + §10 — the reproducibility totals. Source: MANIFEST.json."""
    man = _read_json("results/MANIFEST.json")
    cost = man["cost"]
    per = cost["per_model"]
    src = "results/MANIFEST.json [cost]"
    out = [
        Check("Deepgram calls",
              [r"\*\*([\d,]+) Deepgram calls", r"\(([\d,]+) Deepgram calls"],
              float(cost["deepgram_calls"]), EXACT, src + " deepgram_calls", "§1 / §10"),
        Check("Deepgram audio minutes",
              [r"Deepgram calls ≈ ([\d.]+) min"],
              float(cost["audio_minutes_est"]), TOL1, src + " audio_minutes_est", "§10"),
        Check("Deepgram spend",
              [r"min of audio\n≈ \$(\d\.\d\d)\*\*", r"Deepgram calls ≈ \$(\d\.\d\d);",
               r"\$(\d\.\d\d) is Deepgram"],
              float(per["nova-3"]["usd_est"]), TOL2,
              src + " per_model['nova-3'].usd_est", "§1 / §10"),
        Check("total calls, all arms",
              [r"experiment is (1[\d,]+) calls", r"; ([\d,]+) calls\n≈ \$[\d.]+ across all three arms"],
              float(cost["total_calls"]), EXACT, src + " total_calls", "§1 / §10"),
        Check("total spend, all arms",
              [r"min ≈ \$(\d\.\d\d)\*\*, of which", r"≈ \$(\d\.\d\d) across all three arms"],
              float(cost["usd_total_est"]), TOL2, src + " usd_total_est", "§1 / §10"),
        Check("total audio minutes, all arms",
              [r"calls\n?≈ ([\d.]+) min ≈ \$3\.70"],
              float(cost["total_audio_minutes_est"]), TOL1,
              src + " total_audio_minutes_est", "§10"),
    ]
    if "elevenlabs-scribe" in per:
        out += [
            Check("ElevenLabs calls",
                  [r"\*\*([\d,]+) ElevenLabs calls"],
                  float(per["elevenlabs-scribe"]["calls"]), EXACT,
                  src + " per_model['elevenlabs-scribe'].calls", "§10"),
            Check("ElevenLabs spend",
                  [r"\$(\d\.\d\d)\*\* at the quoted"],
                  float(per["elevenlabs-scribe"]["usd_est"]), TOL2,
                  src + " per_model['elevenlabs-scribe'].usd_est", "§10"),
        ]
    return out


def checks_clean_baseline():
    """§8 limitation 3 — the clean-condition floor. Source: clean_baseline.csv."""
    path = "results/clean_baseline.csv"
    if not os.path.exists(path):
        raise MissingArtifact(path)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n_ref = sum(float(r["n_ref"]) for r in rows)
    micro = sum(float(r["wer"]) * float(r["n_ref"]) for r in rows) / n_ref
    return [
        Check("clean-condition WER floor",
              [r"clean floor \*\*WER (\d+\.\d+) %\*\*"],
              100.0 * micro, TOL2,
              "results/clean_baseline.csv [word-weighted mean WER]", "§8 limitation 3"),
        Check("reference words per condition",
              [r"\*\*(\d+) reference words per condition\*\*"],
              n_ref, EXACT, "results/clean_baseline.csv [sum of n_ref]", "§5"),
    ]


CHECK_GROUPS = [
    ("D1 headline", checks_d1_headline),
    ("D1 dead-zone row", checks_dead_zone_row),
    ("D2 fingerprints", checks_fingerprints),
    ("D3a sensitivity", checks_sobol),
    ("D3a DRR mechanism", checks_drr),
    ("L2 calibration", checks_calibration),
    ("L2 calibration statement", checks_calibration_statement),
    ("D3b active learning", checks_active_learning),
    ("L1 model arms", checks_model_arms),
    ("D4 sim2real", checks_sim2real),
    ("L3 decoupling", checks_l3),
    ("§6.3 corollary", checks_indistinguishable_pair),
    ("L1b three-arm shape", checks_three_arm_shape),
    ("reproducibility", checks_manifest),
    ("corpus floor", checks_clean_baseline),
]


def _collect():
    """Build every check whose artifact is present. Returns (checks, skipped)."""
    checks, skipped = [], []
    for name, loader in CHECK_GROUPS:
        try:
            checks.extend(loader())
        except MissingArtifact as exc:
            skipped.append((name, str(exc)))
    return checks, skipped


def _docs():
    out = {}
    for path in DOCS:
        try:
            out[path] = _doc(path)
        except MissingArtifact:
            pass
    return out


# =========================================================================
# tests
# =========================================================================

def test_every_pinned_figure_matches_its_artifact():
    """THE point of the file: prose == artifact, for every load-bearing figure."""
    docs = _docs()
    if WRITEUP not in docs:
        print("SKIP: %s not present" % WRITEUP)
        return
    checks, skipped = _collect()
    if not checks:
        print("SKIP: results/ is absent (gitignored) — nothing to pin against")
        return

    failures, n_sites = [], 0
    for chk in checks:
        try:
            n_sites += apply_check(docs[chk.doc], chk)
        except Drift as exc:
            failures.append(str(exc))

    for name, why in skipped:
        print("  skipped group %-24s (%s)" % (name, why))

    if failures:
        raise Drift("\n\n%d of %d pinned figures no longer match their artifact:\n%s"
                    % (len(failures), len(checks), "\n".join(failures)))

    print("OK: %d pinned figures matched their artifacts across %d prose sites in %s"
          % (len(checks), n_sites, WRITEUP))


def test_every_check_can_actually_fail():
    """NEGATIVE CONTROL. Mutate the prose behind each check; each must then fail.

    A pinning test that passes against wrong prose is worse than no test — it
    certifies the drift. So every check is exercised against a document in which
    its own figure has been changed, and is required to notice.
    """
    docs = _docs()
    if WRITEUP not in docs:
        print("SKIP: %s not present" % WRITEUP)
        return
    checks, _ = _collect()
    if not checks:
        print("SKIP: results/ is absent (gitignored)")
        return

    text = docs[WRITEUP]
    blind = []
    for chk in checks:
        mutated = _mutate_first_site(text, chk)
        if mutated is None:
            blind.append("%s (could not build a mutation)" % chk.key)
            continue
        try:
            apply_check(mutated, chk)
        except Drift:
            continue
        blind.append(chk.key)

    if blind:
        raise AssertionError(
            "these checks did NOT notice a corrupted prose value, so they pin "
            "nothing:\n  " + "\n  ".join(blind))
    print("OK: all %d checks failed when their prose figure was corrupted — none "
          "is vacuous" % len(checks))


def _mutate_first_site(text, chk):
    """Rewrite the first captured number of `chk` to a clearly different value."""
    for pat in chk.patterns:
        m = re.search(pat, text)
        if m is None:
            continue
        raw = m.group(1)
        try:
            val = parse_num(raw)
        except ValueError:
            continue
        # a value far outside any tolerance, formatted like its neighbour so the
        # regex still matches and the parse still succeeds
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        bogus = abs(val) + max(7.0, 3.0 * chk.tol)
        rendered = ("%." + str(decimals) + "f") % bogus
        if raw.strip().startswith(("-", _MINUS)):
            rendered = raw.strip()[0] + rendered
        if rendered == raw:
            continue
        s, e = m.start(1), m.end(1)
        return text[:s] + rendered + text[e:]
    return None


def test_dead_zone_row_survives_the_column_trap():
    """The documented off-by-one: `rt60_measured` sits next to `mean_conf`.

    `results/dead_zones.csv` runs `..., mic_rolloff, rt60_measured, mean_conf,
    conf_pct, wer, ...`. Reading one column left of `mean_conf` yields the
    DELIVERED REVERB TIME, which reads exactly like a plausible confidence — it
    was mis-read that way once already (SPEC C.5). The settling check is the
    stored identity, not a column count.
    """
    try:
        r = _headline_dead_zone("nova-3")
    except MissingArtifact as exc:
        print("SKIP: %s" % exc)
        return

    conf = float(r["mean_conf"])
    wer_spoke = float(r["wer_spoke"])
    gap = float(r["gap_spoke"])
    rt60_measured = float(r["rt60_measured"])
    n_silent = int(r["n_silent"])

    # (2) the identity only holds under the same-subset pairing
    assert n_silent == 0, (
        "the #1 dead zone now has %d silent clips, so `wer` (all-clips) and "
        "`gap` (spoke) are no longer the same pairing — re-check §6.1's claim "
        "that 'confidence and WER cover the same clips'" % n_silent)
    assert abs(gap - (conf - (1.0 - wer_spoke))) < 1e-9, (
        "gap_spoke != mean_conf - (1 - wer_spoke) on %s: the row is not what "
        "the document describes" % r["condition_name"])
    assert abs(float(r["wer"]) - float(r["wer_all_clips"])) < 1e-12, (
        "`wer` no longer aliases `wer_all_clips`; the alias trap in SPEC G.4 "
        "has moved and every consumer of this file needs re-reading")

    doc = _doc(WRITEUP)
    quoted = re.search(r"mean\nword confidence ([\d.]+) at WER ([\d.]+)", doc)
    assert quoted, "§1's headline dead-zone sentence is gone"
    q_conf, q_wer = float(quoted.group(1)), float(quoted.group(2))
    assert abs(q_conf - conf) < TOL3, (
        "§1 quotes confidence %.3f; the artifact says %.4f" % (q_conf, conf))
    assert abs(q_conf - rt60_measured) > 0.05, (
        "§1's quoted 'confidence' %.3f equals the row's rt60_measured %.4f — "
        "this is the off-by-one column read, not a confidence" % (q_conf, rt60_measured))
    assert abs(q_wer - wer_spoke) < TOL3, (
        "§1 quotes WER %.3f; the artifact's wer_spoke is %.4f" % (q_wer, wer_spoke))
    print("OK: the headline dead-zone row reproduces gap = conf - (1 - wer_spoke) "
          "on 0 silent clips, and the quoted confidence is mean_conf, not rt60_measured")


def test_the_two_dead_zone_rates_are_reconciled():
    """Populations differ legitimately; the document must still say why.

    §6.1's 1.14 % is over 40 clips, §6.6's 0.57 % over the 10 clips every arm
    ran. Both are pinned against their own artifact above. What this asserts is
    that the *reconciliation sentence* survives, because two unexplained rates
    for one model in one document is how a reader loses trust.
    """
    doc = _doc(WRITEUP)
    try:
        b = _cg_block("nova-3")
        m = _read_json("results/model_arms.json")
    except MissingArtifact as exc:
        print("SKIP: %s" % exc)
        return

    rate_40 = float(_grab(b, r"categories: \d+ dead zone \(([\d.]+)%\)"))
    rate_10 = 100.0 * float(m["per_model"]["nova-3"]["dead_zone_rate"])
    assert abs(rate_40 - rate_10) > 1e-6, (
        "the two populations now agree; if that is real, §6.6's reconciliation "
        "paragraph is stale and should be simplified rather than left standing")

    assert re.search(r"which is why nova-3's dead-zone rate reads", doc), (
        "§6.6's sentence reconciling the 10-clip rate with §6.1's 40-clip rate "
        "is gone. The document now states %.2f %% and %.2f %% for the same model "
        "with no explanation." % (rate_10, rate_40))
    n_clips = int(m["arm_census"]["n_common_clips"])
    assert re.search(r"ran on the %d-clip AL subset" % n_clips, doc), (
        "§6.6 no longer names the %d-clip subset the L1 numbers are computed on; "
        "the population label is what makes the two rates compatible" % n_clips)
    print("OK: the 40-clip (%.2f %%) and %d-clip (%.2f %%) dead-zone rates are both "
          "present and still reconciled in the text" % (rate_40, n_clips, rate_10))


def test_retracted_numbers_are_not_quoted_as_findings():
    """Three figures this project retracted must never read as live results.

    Each was published, found wrong, and replaced (SPEC C.8, G.6, G.7). The risk
    is not that they are mentioned — the document discusses all three on purpose
    — it is that a later edit re-promotes one into a claim.
    """
    doc = _doc(WRITEUP)
    problems = []

    # 1. rho = -0.957: the artifact of mixing 176 conditions into an n = 169
    #    statistic. Allowed only where it is explicitly named as the old value.
    for m in re.finditer(r"[−-]0\.957", doc):
        ctx = doc[max(0, m.start() - 260): m.end() + 160]
        if not re.search(r"previously reported|earlier|superseded|was computed", ctx):
            problems.append("line %d quotes rho = -0.957 without marking it as "
                            "the retracted value" % _line_of(doc, m.start()))

    # 2. the 19.9-point sim2real gap: a corpus difference, not a simulation gap.
    for m in re.finditer(r"19\.9", doc):
        ctx = doc[max(0, m.start() - 200): m.end() + 220]
        if not re.search(r"masquerad|unmatched|not matched|artifact", ctx):
            problems.append("line %d quotes the 19.9-point gap without naming it "
                            "as the unmatched-clip artifact" % _line_of(doc, m.start()))

    # 3. the pair 0.843 / 0.387: the old #1 dead zone, now silence-driven. The
    #    document itself promises it "is not quoted as one again".
    for m in re.finditer(r"0\.843", doc):
        ctx = doc[max(0, m.start() - 400): m.end() + 400]
        if not re.search(r"silence|silence-driven|well calibrated|not a finding"
                         r"|correlat", ctx):
            problems.append("line %d quotes confidence 0.843 outside the "
                            "silence-driven context" % _line_of(doc, m.start()))

    if problems:
        raise AssertionError("retracted figures have been re-promoted:\n  "
                             + "\n  ".join(problems))
    print("OK: rho=-0.957, the 19.9-point gap and the 0.843/0.387 pair appear only "
          "where they are named as retracted")


def test_every_arm_the_document_discusses_is_in_the_artifact():
    """The arms in the write-up and the arms in the table must be the same set.

    The L1b checks above are built only when `model_arms.json` carries the
    ElevenLabs arm, so a run that dropped that arm while §6.7 still discussed it
    would quietly lose ~10 checks instead of failing. This closes that door: the
    membership itself is asserted, not assumed.
    """
    try:
        arms = set(_read_json("results/model_arms.json")["arms"])
    except MissingArtifact as exc:
        print("SKIP: %s" % exc)
        return
    doc = _doc(WRITEUP)
    # artifact key -> the token the prose uses for that arm
    naming = {"nova-3": r"[Nn]ova-3",
              "whisper-base": r"[Ww]hisper-base",
              "elevenlabs-scribe": r"`scribe_v2`"}
    unknown = arms - set(naming)
    assert not unknown, (
        "model_arms.json has arm(s) %s that this test does not know how to find "
        "in the prose — add them to `naming` and pin their figures" % sorted(unknown))
    for key, token in naming.items():
        in_doc = re.search(token, doc) is not None
        in_artifact = key in arms
        assert in_doc == in_artifact, (
            "arm %r is %s the artifact but %s the write-up — one of the two was "
            "changed without the other"
            % (key, "in" if in_artifact else "absent from",
               "discussed in" if in_doc else "absent from"))
    print("OK: the %d arms in model_arms.json are exactly the arms the write-up "
          "discusses (%s)" % (len(arms), ", ".join(sorted(arms))))


def test_the_limitation_count_is_self_consistent():
    """§1 claims a number of limitations; §8 has to actually contain that many.

    This is the drift that killed the standalone summary: it advertised "16
    limitations" while §8 had grown to 19. The count is not in any artifact —
    it is a property of the document — so it is checked against the document.
    """
    doc = _doc(WRITEUP)
    body = doc.split("## 8. Limitations")[1].split("## 9.")[0]
    actual = len(re.findall(r"^(\d+)\. \*\*", body, re.M))
    numbers = [int(n) for n in re.findall(r"^(\d+)\. \*\*", body, re.M)]
    assert numbers == list(range(1, actual + 1)), (
        "§8's limitations are not numbered 1..n consecutively: %s" % numbers)

    m = re.search(r"All (\d+)\s*\n?limitations are in §8", doc)
    assert m, "§1 no longer states how many limitations §8 carries"
    claimed = int(m.group(1))
    assert claimed == actual, (
        "§1 advertises %d limitations; §8 contains %d. Update the opener — this "
        "is the exact drift that retired report/SUMMARY.md, which still said 16 "
        "when §8 had reached 19." % (claimed, actual))
    print("OK: §1's claim of %d limitations matches the %d numbered items in §8"
          % (claimed, actual))


def test_the_pin_covers_every_report_document():
    """No prose file in `report/` may quote figures without being pinned.

    The failure this guards is a *new* summary/one-pager appearing beside the
    write-up and drifting — which is exactly why `report/SUMMARY.md` was folded
    into §1 rather than shipped alongside it. Reference logs are exempt by name.
    """
    exempt = {"report/measurements.md",       # capture-chain log, measured once
              "report/INTERVIEW_RUNBOOK.md"}  # rehearsal notes, not a deliverable
    present = sorted(os.path.join("report", f)
                     for f in os.listdir("report") if f.endswith(".md")) \
        if os.path.isdir("report") else []
    unpinned = [p for p in present if p not in exempt and p not in DOCS]
    assert not unpinned, (
        "these report documents quote numbers but nothing pins them:\n  %s\n"
        "Either fold them into %s (one surface, no drift) or add them to DOCS "
        "here with checks of their own." % ("\n  ".join(unpinned), WRITEUP))
    print("OK: every non-exempt document in report/ is under the pin (%s)"
          % ", ".join(DOCS))


if __name__ == "__main__":
    test_every_pinned_figure_matches_its_artifact()
    test_every_check_can_actually_fail()
    test_dead_zone_row_survives_the_column_trap()
    test_the_two_dead_zone_rates_are_reconciled()
    test_retracted_numbers_are_not_quoted_as_findings()
    test_every_arm_the_document_discusses_is_in_the_artifact()
    test_the_limitation_count_is_self_consistent()
    test_the_pin_covers_every_report_document()
    print("\nAll report-number tests passed — every load-bearing figure in "
          "report/writeup.md was re-read from the artifact that produced it, the "
          "documented column trap is checked by identity rather than by position, "
          "retracted figures stay retracted, and every check was proven able to "
          "fail.")
