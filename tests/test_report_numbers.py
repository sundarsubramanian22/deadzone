"""
Number-pinning for `report/writeup.md`, `README.md` and
`report/UNDERSTANDING.md`: every load-bearing figure in the prose is re-read
from the artifact it came from, and this suite fails when the two disagree.

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
reproducibility totals. 211 figures across 311 prose sites, from 14 artifacts,
in under half a second.

SOME FIGURES HAVE NO PRODUCER AND ARE RECOMPUTED HERE. §6.7's three-arm
overconfidence sentence is written by no script and stored in no artifact —
`model_compare.py` and `model_arms.py` contain no bootstrap code at all, so that
table's CIs exist only as prose. What can be re-derived from `results/master.csv`
is: `_clipped_gap_table` reimplements `confidence_gap.py`'s
`mean_conf - clip(1 - WER_spoke, 0, 1)` and pins both arms' means AND the two
populations they may legitimately be quoted over. The published +0.276 / +0.121
were reproducible under no pairing at all; the clamp is why (see that helper).

IT ALSO PINS THE SCOPE ON THE WEAKEST CLAIM. §6.3's "damage is monotone in DRR,
not RT60" is the one figure in the write-up with no interval, and it is the one
an adversarial reader can undercut using a column of the document's own table:
C50 sits at −0.800, the same magnitude as RT60's +0.800, separated from DRR by a
single discordant pair 0.19 dB apart, on a sample of four rooms. The hedge — the
room count, the exact permutation p-values, the C50 correlation, the gap — is
pinned alongside the claim so a later compression pass cannot drop the scope
while keeping the sentence.

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
    artifact — so the suite also catches the document disagreeing with *itself*,
    and (since README.md reads the SAME artifact objects) the two documents
    disagreeing with *each other*.
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
import itertools
import json
import re
from collections import defaultdict

WRITEUP = "report/writeup.md"
README = "README.md"
UNDERSTANDING = "report/UNDERSTANDING.md"

# Every document this suite is willing to pin. Adding a prose file here is the
# only step needed to bring it under the gate.
#
# README.md is here for the same reason `report/SUMMARY.md` was retired: it is a
# SUMMARY, it is the first document an outsider reads, and a summary drifts
# *away from* its body in one direction — toward the stronger claim. It carried
# three such drifts at once (an active-learning "far fewer oracle calls" that
# §6.5 reports as a null, a "Whisper is the outlier" that holds only under
# normalized scoring, and a scope note that reassured about clips while the four
# correlations it covered sat on three different condition populations). Pinning
# it to the SAME artifact objects the write-up is pinned to means the two
# documents cannot disagree without a failure here.
#
# report/UNDERSTANDING.md is here for a sharper reason: the EXEMPTION LIST IS
# WHERE NUMBERS GO TO ROT. `report/measurements.md`, exempt since it was written,
# has already drifted on five figures (22,411 for 22,416; 35.6 % for 35.1 %; a
# calibration discount of 0.74 on n = 7,980 for 0.75 on 8,144; an ECE triple
# rounded to 0.051/0.032/0.006; and a D4 sentence still describing the
# PRE-correction dead-zone sets). Exempting a second, larger prep document would
# repeat that. So its HEADLINE figures are pinned instead — the ones it presents
# as agreeing with `results/`. Its own new computations (the dead-zone threshold
# sensitivity sweep, the 0.962 confidence anchor) have no artifact and are
# deliberately NOT pinned, and neither are the passages where it records a
# disagreement with the write-up: that document exists partly to disagree, and a
# pin there would delete the disagreement rather than check it.
#
# report/INTERVIEW_INTERNAL.md is here for the same reason as UNDERSTANDING.md,
# with one twist that makes it MORE important rather than less: it is a private
# document, so nobody but its author will ever read it, and an unread document
# is where a stale number survives longest. It is also the only document in the
# repo written to be READ ALOUD from — a figure that drifts here is spoken, not
# published, and cannot be corrected by a later reader. Its cue cards and its
# numbers card are pinned to the SAME artifact objects the write-up is pinned to,
# so the script and the deliverable cannot disagree. Its own commentary (the
# threshold-box persistence figures, the re-derived cell-wise bootstrap
# comparison, the prose in the disagreement table) has no artifact and is
# deliberately NOT pinned — that document exists partly to record disagreements,
# and pinning them would delete them rather than check them.
STATUS = "report/STATUS.md"
INTERVIEW_INTERNAL = "report/INTERVIEW_INTERNAL.md"
DOCS = [WRITEUP, README, UNDERSTANDING, STATUS, INTERVIEW_INTERNAL]

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


# --- rank statistics on the four-room reverb axis -------------------------
# §6.3's DRR claim rests on n = 4 rooms, so its p-values are EXACT permutation
# tests over 4! = 24 orderings, not asymptotic approximations. Computed here in
# stdlib rather than read from an artifact because no artifact stores them —
# but the four (DRR, C50, WER) triples they are computed from ARE read from
# `results/interactions.json`, so nothing below is a typed constant.

def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    for pos, i in enumerate(order):
        out[i] = pos + 1.0
    return out


def _spearman(xs, ys):
    """Spearman rho. The four room values carry no ties, so the d^2 form holds."""
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _kendall_tau(xs, ys):
    n = len(xs)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (xs[j] - xs[i]) * (ys[j] - ys[i])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2.0)


def _perm_p(xs, ys, stat, two_sided):
    """Exact permutation p over every relabelling of `xs` against fixed `ys`."""
    obs = stat(xs, ys)
    hits = total = 0
    for perm in itertools.permutations(xs):
        total += 1
        val = stat(list(perm), ys)
        if (abs(val) >= abs(obs) - 1e-12) if two_sided else (val <= obs + 1e-12):
            hits += 1
    return hits / float(total)


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
    out += _checks_drr_scope(it, corr)
    return out


def _checks_drr_scope(it, corr):
    """§1 / §6.3 / limitation 6 / §9 — the SCOPE on the DRR claim.

    This is the one claim in the write-up with no interval, and it is the one an
    adversarial reader can undercut using a column of the document's own table:
    C50 sits at −0.800, the same magnitude as RT60's +0.800, separated from DRR
    by a single discordant pair 0.19 dB apart, on a sample of four rooms. Every
    figure that states that scope is pinned here, so the hedge cannot be quietly
    dropped in a later compression pass while the claim survives.
    """
    src = "results/interactions.json [rir_mechanism]"
    levels = sorted(it["rir_mechanism"]["levels"], key=lambda l: l["rt60_requested"])
    drr = [float(l["drr_db"]) for l in levels]
    c50 = [float(l["c50_db"]) for l in levels]
    wer = [float(it["rir_mechanism"]["marginal_wer"][_lvl_key(l)]) for l in levels]

    # the reverb axis is delivered by however many DISTINCT RIRs the grid used —
    # read from the measured table, not asserted, because "n = 4" is the whole
    # scope of the claim.
    n_rooms = len({r["rir_key"] for r in _master_nova(["rir_key"])})
    n_measured = float(_read_json("results/MANIFEST.json")["assets"]["rirs"]["n_files"])

    # C50's neighbouring pair: the single swap the DRR-vs-C50 separation rests on.
    gaps = sorted(abs(a - b) for a, b in itertools.combinations(c50, 2))

    minutes_per_call = float(
        _read_json("results/MANIFEST.json")["cost"]["minutes_per_call_assumed"])
    rate = float(_read_json("results/MANIFEST.json")["cost"]["usd_per_minute_quoted"])
    n_clips = float(_read_json("results/MANIFEST.json")["corpus"]["n_utterances"])
    unused_calls = (n_measured - n_rooms) * n_clips

    return [
        Check("spearman(C50, WER) — the coordinate that ties RT60's magnitude",
              [r"ρ\(C50, WER\) = (−?-?[\d.]+)",
               r"`spearman\(C50, WER\) = (−?-?[\d.]+)`",
               r"separation from C50 \((−?-?[\d.]+)\)"],
              float(corr["c50_db"]["spearman"]), TOL3,
              src + " correlations.c50_db.spearman", "§1 / §6.3 / limitation 6"),
        Check("distinct RIRs delivering the rt60 axis",
              [r"n = (\d+) rooms",
               r"exactly (\d+) distinct RIRs",
               r"exactly \*\*(\d+)\*\* distinct `rir_key` values",
               r"snapping onto (\d+) distinct RIRs",
               r"an \*\*n = (\d+)\*\*"],
              n_rooms, EXACT,
              "results/master.csv [nova-3, distinct rir_key]",
              "§1 / §6.3 / limitation 6"),
        Check("measured RIRs curated on disk",
              [r"(\d+) measured RIRs", r"F: (\d+) measured, \d+ used"],
              n_measured, EXACT,
              "results/MANIFEST.json assets.rirs.n_files", "§1 / §6.3 / §9 / F"),
        Check("exact one-sided permutation p for spearman(DRR, WER)",
              [r"one-sided permutation p = (\d+\.\d+)",
               r"worth `p = (\d+\.\d+)` one-sided"],
              _perm_p(drr, wer, _spearman, two_sided=False), TOL3,
              src + " [exact permutation over 4! orderings]", "§1 / §6.3 / limitation 6"),
        Check("Kendall tau on the four rooms",
              [r"Kendall τ = (−?-?\d+\.\d+) at two-sided"],
              _kendall_tau(drr, wer), TOL3,
              src + " [kendall tau, DRR vs marginal WER]", "§6.3"),
        Check("exact two-sided permutation p for Kendall tau",
              [r"at two-sided p = (\d+\.\d+)"],
              _perm_p(drr, wer, _kendall_tau, two_sided=True), TOL3,
              src + " [exact permutation over 4! orderings]", "§6.3"),
        Check("the C50 gap the DRR/C50 separation rests on",
              [r"differ by \*\*(\d+\.\d+) dB\*\*", r"pair (\d+\.\d+) dB apart"],
              gaps[0], TOL2,
              src + " [min |ΔC50| over the four rooms]", "§1 / §6.3"),
        Check("unmeasured rooms already on disk",
              [r"is (\d+) rooms × \d+ clips"],
              n_measured - n_rooms, EXACT,
              "MANIFEST assets.rirs.n_files − distinct rir_key in master.csv", "§9"),
        # NOTE the wording of these three sites is load-bearing: §10's own
        # `\*\*([\d,]+) Deepgram calls` and `Deepgram calls ≈ ([\d.]+) min`
        # checks would otherwise capture §9's *hypothetical* follow-up cost as
        # if it were the realized experiment total. Keep "further" and the bold
        # boundaries, or the two totals silently cross-pin.
        Check("§9's calls to finish the reverb axis",
              [r"clips = \*\*(\d+) further Deepgram calls"],
              unused_calls, EXACT,
              "(16 − 4 rooms) × 40 clips, from MANIFEST + master.csv", "§9"),
        Check("§9's audio minutes for that follow-up",
              [r"further Deepgram calls\*\* ≈ \*\*(\d+) min\*\*"],
              unused_calls * minutes_per_call, TOL0,
              "MANIFEST cost.minutes_per_call_assumed × 480", "§9"),
        Check("§9's cost for that follow-up",
              [r"of fresh\naudio ≈ \*\*\$(\d+\.\d+)\*\*"],
              unused_calls * minutes_per_call * rate, TOL2,
              "MANIFEST cost.usd_per_minute_quoted × 33 min", "§9"),
    ]


def _lvl_key(lvl):
    """`marginal_wer` is keyed by the requested rt60 as the JSON wrote it."""
    req = float(lvl["rt60_requested"])
    s = ("%.2f" % req).rstrip("0")
    return s + "0" if s.endswith(".") else s


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


def _clipped_gap_table(model, clip_filter=None):
    """Per-condition `mean_conf - clip(1 - WER_spoke, 0, 1)`, the module's formula.

    THE CLIP IS NOT COSMETIC. `deadzone/analysis/confidence_gap.py` clamps
    delivered accuracy into [0, 1] before subtracting, which matters for any arm
    whose WER can exceed 1.0 — i.e. any arm that hallucinates. Recomputing the
    gap without the clamp reproduces neither artifact, and the discrepancy
    (+0.2754 against the artifact's +0.272 for Scribe) is small enough to look
    like rounding. It was published as +0.276 once for exactly that reason.

    Only clips that emitted words contribute, so `mean_conf` and `WER` are over
    the SAME rows — §6.1's correction, applied at recomputation time.
    """
    path = "results/master.csv"
    if not os.path.exists(path):
        raise MissingArtifact(path)
    agg = defaultdict(lambda: {"conf": [], "wer": []})
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["model"] != model:
                continue
            if str(r["failed"]).strip().lower() in ("true", "1"):
                continue
            if r["mean_conf"] in ("", "nan", None):
                continue
            if clip_filter is not None and r["clip_id"] not in clip_filter:
                continue
            a = agg[r["condition_name"]]
            a["conf"].append(float(r["mean_conf"]))
            a["wer"].append(float(r["wer"]))
    return {c: _mean(a["conf"]) - min(max(1.0 - _mean(a["wer"]), 0.0), 1.0)
            for c, a in agg.items() if a["conf"]}


def _checks_l1b_gaps():
    """§6.7's overconfidence sentence — recomputed, because NOTHING produces it.

    The three-arm table in §6.7 is written by no script and stored in no
    artifact, and `model_arms.py` / `model_compare.py` contain no bootstrap code
    at all, so its CIs live only in the prose. What CAN be re-derived from
    `results/master.csv` is re-derived here: the two arms' mean gaps and the two
    populations they are legitimately quoted over.

    The population is the whole point. nova-3 emits nothing on 10 of Scribe's
    174 spoke-conditions, so "nova-3's gap on the same conditions" is only
    well-defined on the 164 BOTH arms spoke on — and the paired figure is pinned
    separately from each arm's own-population figure so the two cannot be
    silently interchanged. That interchange is §6.1's defect, and it recurred
    here between two arms instead of between two clip sets.
    """
    _pm, _common, clips = _three_arm_populations()
    scribe = _clipped_gap_table("elevenlabs-scribe")
    nova10 = _clipped_gap_table("nova-3", clips)
    nova40 = _clipped_gap_table("nova-3")
    both = sorted(set(scribe) & set(nova10))
    src = "results/master.csv [recomputed, clipped gap; see _clipped_gap_table]"

    return [
        Check("L1b Scribe spoke-conditions (its own population)",
              [r"positive in \*\*(\d+) of 174\*\* conditions strictly",
               r"over Scribe's own\n(\d+) spoke-conditions"],
              float(len(scribe)), EXACT, src + " [scribe spoke-conditions]", "§6.7"),
        Check("L1b Scribe strict mean gap (own 174)",
              [r"conditions strictly, mean\n\*\*\+([\d.]+)\*\*"],
              _mean(scribe.values()), TOL3, src + " [scribe, own population]", "§6.7"),
        Check("L1b conditions nova-3 is absent from",
              [r"nothing at all on \*\*(\d+) of\nScribe's 174\*\*"],
              float(len(scribe) - len(both)), EXACT,
              src + " [scribe spoke-conds minus the intersection]", "§6.7"),
        Check("L1b paired population, both arms",
              [r"On the \*\*(\d+) both arms spoke on\*\*"],
              float(len(both)), EXACT, src + " [scribe ∩ nova-3 spoke-conditions]", "§6.7"),
        Check("L1b Scribe mean gap on the paired 164",
              [r"Scribe runs \*\*\+([\d.]+)\*\*"],
              _mean(scribe[c] for c in both), TOL3,
              src + " [scribe, paired population]", "§6.7"),
        Check("L1b nova-3 mean gap on the paired 164",
              [r"against nova-3's\n\*\*\+([\d.]+)\*\*"],
              _mean(nova10[c] for c in both), TOL3,
              src + " [nova-3, paired population]", "§6.7"),
        Check("L1b nova-3 corpus-wide gap, named as the OTHER population",
              [r"corpus-wide gap is \*\*\+([\d.]+)\*\*"],
              _mean(nova40.values()), TOL3,
              src + " [nova-3, 40 clips — §6.1's population]", "§6.7"),
        Check("L1b nova-3 corpus-wide conditions",
              [r"over 40 clips and (\d+) conditions"],
              float(len(nova40)), EXACT,
              src + " [nova-3 spoke-conditions, 40 clips]", "§6.7"),
    ]


def _three_arm_populations():
    """(per_model, common conditions, common clips) for the three-arm table.

    "spoke" = the arm returned a confidence, i.e. it emitted words. That is the
    same population rule §6.1's correction turns on, applied to three arms.
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
    return per_model, common, clips


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
    per_model, common, clips = _three_arm_populations()
    src = ("results/master.csv [%d conditions all %d arms spoke on, %d common clips]"
           % (len(common), len(per_model), len(clips)))

    out = _checks_l1b_gaps() + [Check("L1b three-arm common conditions",
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


def _master_census():
    """(total rows, failed rows, distinct clips, distinct conditions) in master.csv.

    Counted rather than read from a summary: the README's scale callouts are the
    first numbers anyone sees, and a partial re-run that shortened the table is
    exactly the drift SPEC J.5 records.
    """
    path = "results/master.csv"
    if not os.path.exists(path):
        raise MissingArtifact(path)
    n = n_failed = 0
    clips, conds = set(), set()
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n += 1
            if str(r["failed"]).strip().lower() in ("true", "1"):
                n_failed += 1
            clips.add(r["clip_id"])
            conds.add(r["condition_name"])
    return n, n_failed, len(clips), len(conds)


def checks_readme():
    """README.md — the external artifact, pinned to the same artifact objects.

    A summary drifts *away from* its body in one direction: toward the stronger
    claim. This document is the first thing an outsider reads and it carries the
    headline, the three-arm table, the fingerprints and the appendix of things
    that did not work — so every load-bearing figure in it is re-read from the
    file that produced it, and the two documents cannot disagree without a
    failure here.

    TWO THINGS ARE PINNED THAT ARE NOT NUMBERS OF THE FIRST KIND, because they
    are the claims a summary most wants to round off:

      * the POPULATION of every three-arm figure (the 10-clip subset) is pinned
        against `arm_census`, and nova-3's two dead-zone rates (1.14 % over 40
        clips, 0.57 % over 10) are pinned SEPARATELY against their own artifacts;
      * the ACTIVE-LEARNING result is pinned as the null it is — seeds reaching
        target on both arms, plus the permutation control that killed its own
        obvious fix.

    A THIRD used to be here and is now RECORDED AS LOST, not quietly dropped:
    the dead-zone COUNT was pinned alongside the THRESHOLD SWEEP (13 / 2 / 0
    across one notch of `conf_pct`) so that "a later edit cannot keep the count
    and drop the fragility". The 2026-08-06 rewrite did exactly that — the count
    is still in §5, the sweep is not — so those five checks were deleted rather
    than repointed, and the pin no longer defends that pairing in README. It
    still holds in report/writeup.md. Anyone restoring the sweep to README
    should restore the checks with it (git log this file).
    """
    if not os.path.exists(README):
        raise MissingArtifact(README)
    b = _cg_block("nova-3")
    cg = "results/confidence_gap.txt [nova-3 block]"
    dz = _headline_dead_zone("nova-3")
    m = _read_json("results/model_arms.json")
    pm = m["per_model"]
    ma = "results/model_arms.json [per_model]"
    al = _read_json("results/al_savings.json")
    drr = _read_json("results/al_drr.json")["controls"]["control_a_permutation"]
    s2r = _read_json("results/sim2real.json")["nova-3"]["headline"]
    cal = {row["model"]: row for row in _read_json("results/calibration.json")["cross_arm"]}
    blocked = {row["model"]: row
               for row in _read_json("results/calibration.json")["blocked_arms"]}
    fp = _read_json("results/fingerprints.json")["by_model"]["nova-3"]
    delb = _read_json("results/calibration.json")["deletion_blindness"]
    conf = _read_json("results/confidence_char.json")["by_model"]["nova-3"]
    man = _read_json("results/MANIFEST.json")
    n_rows, n_failed, n_clips, n_conds = _master_census()

    n_mute = int(_grab(b, r"(\d+) conditions silent on EVERY clip"))
    n_cond = int(_grab(b, r"conditions: (\d+)"))
    ent = fp["entities"]["overall"]
    cls = fp["inventory"]["by_class"]

    # (the threshold-sensitivity surface used to be read here for the §5 sweep;
    # that whole paragraph left README, so results/dead_zone_sensitivity.json is
    # no longer a source for this document — see the removal note in §5 below.)

    def C(key, patterns, expected, tol, source, where):
        return Check(key, patterns, expected, tol, source, where, doc=README)

    out = [
        # ---- §3 scale callouts ------------------------------------------
        C("README total scored rows",
          [r"= \*\*([\d,]+) scored transcriptions\*\*",
           r"<b>([\d,]+)</b><br/>scored rows",
           r"results/master\.csv<br/>([\d,]+) rows"],
          n_rows, EXACT, "results/master.csv [row count]", "§3 scale"),
        C("README failed rows",
          [r"<sub>(\d+) failures = \d\.\d+%</sub>"],
          n_failed, EXACT, "results/master.csv [rows with failed=True]", "§3 scale"),
        C("README utterances",
          [r"<b>(\d+)</b><br/>utterances"],
          n_clips, EXACT, "results/master.csv [distinct clip_id]", "§1 / §3"),
        C("README conditions",
          [r"<b>(\d+)</b><br/>conditions"],
          n_conds, EXACT, "results/master.csv [distinct condition_name]", "§3 scale"),
        C("README total API calls",
          [r"<b>([\d,]+)</b><br/>API calls"],
          float(man["cost"]["total_calls"]), EXACT,
          "results/MANIFEST.json cost.total_calls", "§3 scale"),
        C("README total spend",
          [r"<b>\$([\d.]+)</b><br/>total spend"],
          float(man["cost"]["usd_total_est"]), TOL2,
          "results/MANIFEST.json cost.usd_total_est", "§3 scale"),
        C("README clean-condition WER floor",
          [r"clean-condition WER \*\*([\d.]+)%\*\*"],
          100.0 * float(man["corpus"]["clean_wer_floor"]), TOL2,
          "results/MANIFEST.json corpus.clean_wer_floor", "§3 scale"),

        # ---- §4 the listening beat --------------------------------------
        C("README A/B: drenched-but-quiet WER",
          [r"SNR 20 dB \| \*\*([\d.]+)\*\* \|"],
          _ab_pair()["mean_A"], TOL3,
          "results/master.csv [rt60-1_snr-20_babble_none_roll-0, 40 clips]", "§4"),
        C("README A/B: dry-but-buried WER",
          [r"SNR 0 dB \| \*\*([\d.]+)\*\* \|"],
          _ab_pair()["mean_B"], TOL3,
          "results/master.csv [rt60-0.2_snr-0_babble_none_roll-0, 40 clips]", "§4"),
        C("README A/B: paired difference",
          [r"→ \*\*−([\d.]+)\*\* paired difference",
           r"paired difference −([\d.]+) WER"],
          abs(_ab_pair()["diff"]), TOL2,
          "results/master.csv [A - B, paired over 40 clips]", "§4"),
        C("README A/B: clips scoring identically",
          [r"\*\*(\d+) of 40\*\* clips score exactly equal",
           r"(\d+) of 40 clips score identically"],
          float(_ab_pair()["n_same"]), EXACT,
          "results/master.csv [clips with zero paired difference]", "§4"),

        # ---- §5 the headline, threshold-free first ----------------------
        C("README D1 spearman, paired",
          [r"\*\*ρ = (−[\d.]+)\*\*, overconfident in"],
          float(_grab(b, r"global spearman\(conf_pct, WER_spoke\) = (-?[\d.]+)")),
          TOL3, cg + " global spearman(conf_pct, WER_spoke)", "§5 headline"),
        C("README D1 spearman, all-clips",
          [r"all-clips pairing ρ = (−?-?[\d.]+)"],
          float(_grab(b, r"\[all-clips pairing: (-?[\d.]+)\]")),
          TOL3, cg + " [all-clips pairing]", "§5 headline"),
        C("README D1 conditions that spoke (n)",
          [r"(\d+) of 176 conditions that returned words"],
          n_cond - n_mute, EXACT,
          cg + " conditions - mute conditions", "§5 headline"),
        C("README D1 overconfident share",
          [r"overconfident in \*\*(\d+)%\*\*"],
          float(_grab(b, r"overconfident in (\d+)% of conditions")), TOL0,
          cg + " overconfident in N% of conditions", "§5 headline"),
        # REMOVED: "README D1 overconfident count" — the stat-tile "154 of 169
        # conditions overconfident" left README; §5 now states the share only
        # ("overconfident in 91%"), which `README D1 overconfident share` pins.
        # REMOVED: "README D1 mean gap" — the "+0.147 mean gap" stat tile left
        # README with the same table; no prose or caption carries it.
        C("README D1 dead-zone count",
          [r"\*\*(\d+) of 176 \(\d\.\d+%\)\*\* conditions clear the published",
           r"\| \*\*dead zone\*\* \| \*\*(\d+)\*\* \|"],
          float(_grab(b, r"categories: (\d+) dead zone")), EXACT,
          cg + " categories: N dead zone", "§5 headline"),
        C("README D1 dead-zone rate % (40 clips)",
          [r"\*\*\d+ of 176 \((\d\.\d+)%\)\*\* conditions clear the published",
           r"dead-zone rate is \*\*([\d.]+)% \(\d+/176\)\*\* on 40 clips"],
          float(_grab(b, r"categories: \d+ dead zone \(([\d.]+)%\)")), TOL2,
          cg + " categories: N dead zone (P%)", "§5 headline / §7 population"),
        C("README D1 silence-driven count",
          [r"\| \*\*silence-driven\*\* \| \*\*(\d+)\*\* \|"],
          float(_grab(b, r"(\d+) silence-driven")), EXACT,
          cg + " categories: N silence-driven", "§5 categories"),
        C("README D1 mute-zone count",
          [r"\| \*\*mute zone\*\* \| \*\*(\d+)\*\* \|"], n_mute, EXACT,
          cg + " conditions silent on EVERY clip", "§5 categories"),
        C("README headline dead zone: confidence",
          [r"→ \*\*([\d.]+)\*\* confidence at WER"], float(dz["mean_conf"]), TOL3,
          "results/dead_zones.csv [#1 dead_zone row].mean_conf", "§5 headline"),
        C("README headline dead zone: WER",
          [r"confidence at WER \*\*([\d.]+)\*\*,"], float(dz["wer_spoke"]), TOL3,
          "results/dead_zones.csv [#1 dead_zone row].wer_spoke", "§5 headline"),
        C("README headline dead zone: clips silent",
          [r"\*\*(\d+) of 40\*\* clips silent"], float(dz["n_silent"]), EXACT,
          "results/dead_zones.csv [#1 dead_zone row].n_silent", "§5 headline"),
        C("README estimand defect: gap inflation, mean",
          [r"\(\+([\d.]+) mean inflation\)"],
          float(_grab(b, r"inflation mean \+([\d.]+),")), TOL3,
          cg + " inflation mean", "§5 the defect"),
        # REMOVED: "README estimand defect: gap inflation, max" — the "max
        # +0.524" half of the inflation pair left README; the callout now quotes
        # the mean only, which the check immediately above still pins.
        C("README estimand defect: the retracted count",
          [r"v1 reported \*\*(\d+)\*\* dead zones"],
          float(_grab(b, r"pairing alone would have called (\d+) of them dead zones")),
          EXACT, cg + " [the all-clips pairing would have called N]", "§5 the defect"),

        # ---- §5 the confidence anchor, so 0.829 is interpretable --------
        C("README clean-condition confidence",
          [r"clean-condition mean \*\*([\d.]+)\*\*"],
          float(conf["dynamic_range"]["clean_corner"]), TOL3,
          "results/confidence_char.json [nova-3.dynamic_range.clean_corner]", "§5"),
        C("README best-condition confidence",
          [r"best condition \*\*([\d.]+)\*\*"],
          float(conf["dynamic_range"]["highest_condition"]["value"]), TOL3,
          "results/confidence_char.json [nova-3.dynamic_range.highest_condition]", "§5"),
        C("README worst-condition confidence",
          [r"worst condition that still speaks \*\*([\d.]+)\*\*"],
          float(conf["dynamic_range"]["lowest_condition"]["value"]), TOL3,
          "results/confidence_char.json [nova-3.dynamic_range.lowest_condition]", "§5"),
        # REMOVED: "README aggregate AUROC" — §5's "arithmetic mean AUROC 0.944"
        # sentence left README. The same figure survives in the §7 table and is
        # pinned there by `README nova-3 utterance AUROC (mean aggregate)`; it is
        # NOT repointed here, because a §5 check landing on a §7 table row would
        # pin a sentence it was not written for.

        # ---- §5 the threshold sweep: the count is an OPERATING POINT ----
        # REMOVED, all five: the whole threshold-sensitivity sweep left README —
        # "13 at conf-pct 0.50", "2 at 0.60", "0 at 0.70", and the box range
        # "0 to 86". §5 now states the published operating point (WER >= 0.30,
        # top 40% confidence) and its count only. The sweep is still pinned in
        # report/writeup.md; nothing in README carries these five numbers, so
        # results/dead_zone_sensitivity.json is no longer a README source.
    ]

    # ---- §7 the three arms. EVERY ONE carries its population. -----------
    out += [
        C("README three-arm rows per model",
          [r"all three arms ran — ([\d,]+) rows per arm"],
          float(m["arm_census"]["n_common_cells"]), EXACT,
          "results/model_arms.json arm_census.n_common_cells", "§7 population"),
        C("README three-arm shared clips",
          [r"ran a \*\*(\d+)-clip subset\*\*"],
          float(m["arm_census"]["n_common_clips"]), EXACT,
          "results/model_arms.json arm_census.n_common_clips", "§7 population"),
        C("README nova-3 dead-zone count, 40-clip population",
          [r"dead-zone rate is \*\*[\d.]+% \((\d+)/176\)\*\* on 40 clips"],
          float(_grab(b, r"categories: (\d+) dead zone")), EXACT,
          cg + " categories: N dead zone", "§7 population"),
        # The per-arm dead-zone COLUMN left the §7 table — README now says
        # "Dead-zone rate and rho(confidence, WER) are in the chart above" and
        # ships them in docs/assets/model-comparison.svg. What survives in prose
        # is the population box's nova-3 pair, and Scribe's strict count inside
        # the "it was spelling, not confident error" caveat. Those two are
        # repointed; the rest are removed rather than aimed at a chart.
        C("README nova-3 dead-zone rate, 10-clip population",
          [r"and \*\*([\d.]+)% \(\d+/176\)\*\* on 10"],
          100.0 * float(pm["nova-3"]["dead_zone_rate"]), TOL2,
          ma + " nova-3.dead_zone_rate", "§7 population"),
        C("README nova-3 dead-zone count, 10-clip population",
          [r"and \*\*[\d.]+% \((\d+)/176\)\*\* on 10"],
          float(pm["nova-3"]["n_dead_zones"]), EXACT,
          ma + " nova-3.n_dead_zones", "§7 population"),
        # REMOVED: "README whisper dead-zone rate" (39.20%) — no prose site.
        # REMOVED: "README whisper dead-zone count" (69) — no prose site.
        # REMOVED: "README scribe dead-zone rate (flagged not quotable)"
        #          (3.98%) — no prose site.
        C("README scribe dead-zone count (flagged not quotable)",
          [r"its (\d+) strict dead zones fall to \*\*\d+\*\* under the normalizer"],
          float(pm["elevenlabs-scribe"]["n_dead_zones"]), EXACT,
          ma + " elevenlabs-scribe.n_dead_zones", "§7 table caveat"),
    ]

    # per-arm shape rho, each on ITS OWN condition population. The shape column
    # left the §7 table with the dead-zone column (both are in the chart now),
    # but every arm's rho survives in the prose that reads the chart against
    # itself — the "Scribe and Whisper swap" callout and the whisper-base
    # bullet. Each pattern below is anchored on the words of the sentence it was
    # written for, so no arm's rho can be satisfied by another arm's number.
    #
    # REMOVED with the table: "README nova-3 shape n" (164), "README
    # elevenlabs-scribe shape n" (174), "README whisper-base shape n" (171) —
    # the per-arm condition counts have no prose site left in README at all.
    shape_sites = {
        "nova-3": [r"vs nova-3's \*\*(−?-?[\d.]+)\*\* on the same rows"],
        "elevenlabs-scribe": [r"Scribe ahead on ρ \((−?-?[\d.]+) vs −?-?[\d.]+\)"],
        "whisper-base": [r"Scribe ahead on ρ \(−?-?[\d.]+ vs (−?-?[\d.]+)\)",
                         r"- ρ \*\*(−?-?[\d.]+)\*\* vs nova-3's"],
    }
    for model, rho_pats in shape_sites.items():
        if model not in pm:
            continue
        out.append(
            C("README %s strict shape" % model, rho_pats,
              float(pm[model]["shape"]["spearman"]), TOL3,
              ma + " %s.shape.spearman" % model, "§7"))

    silence_sites = {
        "nova-3": (r"\| ([\d.]+)% \| 12 \|", r"\| [\d.]+% \| (\d+) \|\n\| \*\*elevenlabs"),
        "elevenlabs-scribe": (r"\*\*([\d.]+)%\*\* \| 2 \|",
                              r"\*\*[\d.]+%\*\* \| (\d+) \|\n\| \*\*whisper"),
        "whisper-base": (r"\| ([\d.]+)% \| 5 \|",
                         r"\*\*whisper-base\*\* \|[^\n]*\| [\d.]+% \| (\d+) \|"),
    }
    for model, (rate_pat, mute_pat) in silence_sites.items():
        if model not in pm:
            continue
        out += [
            C("README %s silent-row rate (10-clip)" % model, [rate_pat],
              100.0 * float(pm[model]["silence"]["silent_rate"]), TOL1,
              ma + " %s.silence.silent_rate" % model, "§7 table"),
            C("README %s mute conditions (10-clip)" % model, [mute_pat],
              float(pm[model]["n_mute_zones"]), EXACT,
              ma + " %s.n_mute_zones" % model, "§7 table"),
        ]

    # The utterance-level AUROC column, and it is pinned for a REASON: it is the
    # one statistic on which the two non-spine arms SWAP (Whisper 0.888 above
    # Scribe 0.737, against Scribe leading on rho and dead-zone rate). That swap
    # is what refuses a "commercial beats open" reading, so a later edit must not
    # be able to drop it and keep the ordering.
    cc = {row["model"]: row
          for row in _read_json("results/confidence_char.json")["cross_arm"]}
    # AUROC is now the FIRST data column of the §7 table, so each pattern is
    # anchored on the arm name AND on the ECE cell that follows it — that pairing
    # exists only in this table, and cannot drift onto the "0.888 vs 0.737"
    # sentence below it, which quotes the same two numbers in the other order.
    auroc_sites = {
        "nova-3": r"\*\*nova-3\*\* \| \*\*([\d.]+)\*\* \| \*\*[\d.]+ →",
        "elevenlabs-scribe": r"\*\*elevenlabs-scribe\*\* \| ([\d.]+) \| [\d.]+ →",
        "whisper-base": r"\*\*whisper-base\*\* \| ([\d.]+) \| \*\*BLOCKED\*\*",
    }
    for model, pat in auroc_sites.items():
        if model not in cc:
            continue
        out.append(C("README %s utterance AUROC (mean aggregate)" % model, [pat],
                     float(cc[model]["mean_auroc"]), TOL3,
                     "results/confidence_char.json cross_arm[%s].mean_auroc" % model,
                     "§7 table"))
    if "elevenlabs-scribe" in cc and "nova-3" in cc:
        out += [
            C("README scribe words tied at the confidence ceiling",
              [r"([\d.]+)% of words within 0\.001 of 1\.0"],
              100.0 * float(cc["elevenlabs-scribe"]["frac_within_eps_of_one"]), TOL1,
              "results/confidence_char.json cross_arm[elevenlabs-scribe]"
              ".frac_within_eps_of_one", "§7 scribe"),
            # ANCHORED ON THE CEILING CLAUSE, NOT ON "(nova-3: N%)". README now
            # carries that parenthetical TWICE -- here at 15.3% (words tied at
            # the confidence ceiling) and again in the hallucination paragraph at
            # 0.1% (rows over 2x reference length). The bare "\(nova-3: ...\)"
            # pattern matched both and reported the hallucination figure against
            # the ceiling artifact. Each of the two now carries its own sentence.
            C("README nova-3 words tied at the confidence ceiling",
              [r"within 0\.001 of 1\.0 \(nova-3: ([\d.]+)%\)"],
              100.0 * float(cc["nova-3"]["frac_within_eps_of_one"]), TOL1,
              "results/confidence_char.json cross_arm[nova-3].frac_within_eps_of_one",
              "§7 scribe"),
        ]

    # calibration, per arm, from the cross-arm table (never the top-level keys,
    # which are nova-3's alone and would silently mislabel a second arm)
    ece_sites = {
        "nova-3": r"\*\*nova-3\*\* \|[^\n|]*\| "
                  r"\*\*([\d.]+) → ([\d.]+) → ([\d.]+)\*\*",
        "elevenlabs-scribe": r"\*\*elevenlabs-scribe\*\* \|[^\n|]*\| "
                             r"([\d.]+) → ([\d.]+) → ([\d.]+) \^",
    }
    # each row prints three ECEs in one cell, so the pattern is built with
    # exactly ONE capturing group at a time — three groups would make
    # `apply_check` read group(1) only and silently pin nothing for the other two
    for model, pat in ece_sites.items():
        if model not in cal:
            continue
        parts = pat.split("([\\d.]+)")
        if len(parts) != 4:
            raise MissingArtifact("the README ECE pattern for %r no longer has "
                                  "three slots" % model)
        for i, field in enumerate(("ece_raw", "ece_temperature", "ece_feature")):
            rebuilt = "".join(
                part + ("([\\d.]+)" if j == i else "[\\d.]+")
                for j, part in enumerate(parts[:-1])) + parts[-1]
            out.append(C("README %s %s" % (model, field), [rebuilt],
                         float(cal[model][field]), TOL3,
                         "results/calibration.json cross_arm[%s].%s" % (model, field),
                         "§7 table"))
    if "elevenlabs-scribe" in cal and "nova-3" in cal:
        out += [
            C("README scribe temperature",
              [r"harder correction: \*\*T = ([\d.]+)\*\* vs nova-3's"],
              float(cal["elevenlabs-scribe"]["temperature_T"]), TOL2,
              "results/calibration.json cross_arm[elevenlabs-scribe].temperature_T", "§7"),
            # README dropped the second "T =" ("T = 4.11 vs nova-3's 1.39"), so
            # nova-3's temperature is now a bare bolded number after "vs
            # nova-3's" -- a phrase README also uses for rho and for insertions.
            # The pattern therefore carries Scribe's "T = " prefix as its anchor.
            C("README nova-3 temperature",
              [r"\*\*T = [\d.]+\*\* vs nova-3's \*\*([\d.]+)\*\*"],
              float(cal["nova-3"]["temperature_T"]), TOL2,
              "results/calibration.json cross_arm[nova-3].temperature_T", "§7"),
        ]
    if "whisper-base" in blocked:
        w = blocked["whisper-base"]
        out += [
            C("README whisper misaligned rows",
              [r"uncomputable\*\*: (\d+) rows have confidence-list length"],
              float(w["n_misaligned_rows"]), EXACT,
              "results/calibration.json blocked_arms[whisper-base].n_misaligned_rows",
              "§7 whisper"),
        ]

    # the failure-mode contrast, and the hallucination exhibit
    xm = {k: v["edit_signature_crossmodel"] for k, v in pm.items()}
    hal = m["hallucination_by_model"]
    ex = m["whisper_hallucination"]["examples"][0]
    fig = _read_json("docs/assets/figures.json")["figures"]["whisper-hallucination.svg"]
    out += [
        C("README nova-3 silent-rate advantage over scribe",
          [r"\*\*([\d.]+)× less likely to go silent\*\*"],
          float(pm["nova-3"]["silence"]["silent_rate"])
          / float(pm["elevenlabs-scribe"]["silence"]["silent_rate"]), TOL1,
          ma + " silent_rate ratio nova-3 / elevenlabs-scribe", "§7 scribe"),
        # all three now live in one reworded sentence:
        #   "insertions **9.4x** nova-3's (0.197 vs 0.021)"
        # so each pattern spells out the whole sentence and captures its own
        # slot -- none of the three can be satisfied by another's number.
        C("README whisper normalized insertion rate",
          [r"insertions \*\*[\d.]+×\*\* nova-3's \(([\d.]+) vs [\d.]+\)"],
          float(xm["whisper-base"]["ins"]), TOL3,
          ma + " whisper-base.edit_signature_crossmodel.ins", "§7 whisper"),
        C("README nova-3 normalized insertion rate",
          [r"insertions \*\*[\d.]+×\*\* nova-3's \([\d.]+ vs ([\d.]+)\)"],
          float(xm["nova-3"]["ins"]), TOL3,
          ma + " nova-3.edit_signature_crossmodel.ins", "§7 whisper"),
        C("README whisper/nova-3 insertion ratio",
          [r"insertions \*\*([\d.]+)×\*\* nova-3's"],
          float(xm["whisper-base"]["ins"]) / float(xm["nova-3"]["ins"]), TOL1,
          ma + " edit_signature_crossmodel ins ratio", "§7 whisper"),
        # NOT pinned to model_arms.json's own n_ref/n_hyp, which are 3 and 49.
        # Those come from `hallucination_report`, which cross-model-normalizes
        # ("four zero five" -> "405") and then tokenizes with [a-z']+ — so it
        # MANUFACTURES eight digit tokens and immediately discards them. The
        # reference is 11 spoken words. The hypothesis contains no digits, loses
        # nothing, and the ratio inflates 16.3x against a true 4.3x. The figure
        # and the README both publish the strict alignment; these pin that, and
        # figures.json stores the discarded counting alongside it under
        # n_{ref,hyp}_tokens_crossmodel so the artifact still records both.
        # the corrected and the discarded counting now sit in one callout:
        #   "reported across the repo as **3 -> 49** ... Correct: **11 -> 47**"
        # The two are pinned APART on purpose -- that is the whole point of the
        # callout -- so "Correct:" anchors the strict pair and "reported across
        # the repo as" anchors the retracted one.
        C("README hallucination exhibit: reference words (STRICT, not the "
          "letters-only tokenization)",
          [r"Correct: \*\*(\d+) → \d+, WER"], float(fig["n_ref"]), EXACT,
          "docs/assets/figures.json [whisper-hallucination.svg] n_ref",
          "§7 exhibit"),
        C("README hallucination exhibit: hypothesis words (STRICT)",
          [r"Correct: \*\*\d+ → (\d+), WER"], float(fig["n_hyp_words"]), EXACT,
          "docs/assets/figures.json [whisper-hallucination.svg] n_hyp_words",
          "§7 exhibit"),
        C("README hallucination exhibit: the discarded counting is named as the "
          "artifact it is, never as the result",
          [r"reported across the repo as \*\*(\d+) → \d+\*\*"],
          float(fig["n_ref_tokens_crossmodel"]), EXACT,
          "docs/assets/figures.json n_ref_tokens_crossmodel", "§7 exhibit"),
        C("README whisper rows over 2x reference length",
          [r"\*\*([\d.]+)%\*\* of rows exceed 2× the reference length"],
          100.0 * float(hal["whisper-base"]["frac_rows_over_2x"]), TOL1,
          "results/model_arms.json hallucination_by_model[whisper-base]", "§7 exhibit"),
        # REMOVED: "README whisper p95 length ratio" (2.75) -- the p95 figure
        # left README entirely; the paragraph keeps only the over-2x share.
        # anchored on "of rows exceed 2x the reference length", NOT on a bare
        # "(nova-3: N%)" -- see the ceiling check above for why that matters.
        C("README nova-3 rows over 2x reference length",
          [r"of rows exceed 2× the reference length \(nova-3: ([\d.]+)%\)"],
          100.0 * float(hal["nova-3"]["frac_rows_over_2x"]), TOL1,
          "results/model_arms.json hallucination_by_model[nova-3]", "§7 exhibit"),
    ]

    # REMOVED, all four: the operational calibration-discount sentence left
    # README -- "above rt60 = 0.7, discount reported confidence by ~0.07 (0.81
    # reported vs. 0.75 observed on 8,144 held-out words)". §7's nova-3 bullets
    # keep the ECE pair only. The sentence is still pinned in report/writeup.md
    # against the same results/calibration.json [statement] field.

    # ---- §8 fingerprints ------------------------------------------------
    out += [
        # REMOVED: "README reference words scored" (63,888) and the three
        # edit-composition rates -- "deletions 0.351", "substitutions 0.136",
        # "insertions 0.020". §8's HTML edit-composition table left README; the
        # composition is now carried by docs/assets/fingerprints.svg, and the
        # surviving prose states the mechanism rather than the three rates.
        C("README deletions as a share of all errors",
          [r"Deletions are \*\*([\d.]+)%\*\* of nova-3's errors"],
          100.0 * float(delb["deleted_fraction_of_errors"]), TOL1,
          "results/calibration.json deletion_blindness.deleted_fraction_of_errors",
          "§3 limits"),
        C("README entity error rate",
          [r"entity error rate \*\*([\d.]+)\*\*"],
          float(ent["mean_entity_error_rate"]), TOL3,
          "results/fingerprints.json [nova-3] entities.overall.mean_entity_error_rate",
          "§8"),
        C("README WER paired with the entity error rate",
          [r"vs\. WER \*\*([\d.]+)\*\*"],
          float(ent["mean_wer"]), TOL3,
          "results/fingerprints.json [nova-3] entities.overall.mean_wer", "§8"),
        C("README babble insertions that are foreign",
          [r"\*\*(\d+)%\*\* are tokens absent from the reference"],
          100.0 * float(fp["insertions"]["by_group"]["babble"]["foreign_frac"]), TOL0,
          "results/fingerprints.json [nova-3] insertions.by_group.babble.foreign_frac",
          "§8"),
    ]
    for label, klass in (("proper nouns", "proper_noun"),
                         ("spelled letters", "spelled_letter"),
                         ("content words", "content_word"),
                         ("function words", "function_word"),
                         ("digit words", "digit_word")):
        out.append(C("README destroyed-word rate, %s" % label,
                     [r"\| \*?\*?%s\*?\*? \| \*?\*?([\d.]+)" % label],
                     float(cls[klass]["destruction_rate"]), TOL3,
                     "results/fingerprints.json [nova-3] inventory.by_class.%s" % klass,
                     "§8"))
    # REMOVED, all five: the per-family signature table left README -- the
    # deltas "falling SNR +0.344", "mic rolloff +0.264", "reverb +0.212",
    # "g726 +0.061" and "road noise +0.059", each with its dominant edit type.
    # docs/assets/fingerprints.svg carries the split now, and §8's surviving
    # prose states the 7-of-9 deletion / 2-of-9 substitution count rather than
    # the five effect sizes. The deltas remain pinned in report/writeup.md
    # against the same results/fingerprints.json signatures.

    # ---- §9 the comparability gate --------------------------------------
    shift = m["normalization_shift"]
    # the sign is part of the claim here — nova-3's shift is NEGATIVE and near
    # zero (the control), the other two are positive — so the pattern captures
    # it rather than assuming one direction
    for model, anchor in (("nova-3", r"\| nova-3 \| \*\*([−+-]?[\d.]+)\*\*"),
                          ("whisper-base", r"\| whisper-base \| \*\*([−+-]?[\d.]+)\*\*"),
                          ("elevenlabs-scribe",
                           r"\| elevenlabs-scribe \| \*\*([−+-]?[\d.]+)\*\*")):
        if model not in shift:
            continue
        out.append(C("README normalization shift [%s]" % model, [anchor],
                     float(shift[model]["mean_shift"]), TOL3,
                     "results/model_arms.json normalization_shift[%s].mean_shift" % model,
                     "§9"))

    # ---- appendix: the two nulls and the sim2real offset ----------------
    out += [
        C("README AL evaluation budget",
          [r"At a (\d+)-evaluation budget"],
          float(al["n_total"]), EXACT, "results/al_savings.json n_total", "appendix (a)"),
        C("README AL seeds reaching target, active",
          [r"reached by \*\*(\d+) of 8 active seeds\*\*"],
          float(al["headline"]["n_seeds_reaching_target"]["active_boundary"]), EXACT,
          "results/al_savings.json headline.n_seeds_reaching_target.active_boundary",
          "appendix (a)"),
        C("README AL seeds reaching target, random",
          [r"against random's \*\*(\d+) of 8\*\*"],
          float(al["headline"]["n_seeds_reaching_target"]["random"]), EXACT,
          "results/al_savings.json headline.n_seeds_reaching_target.random",
          "appendix (a)"),
        C("README AL seed count",
          [r"all (\d+) seeds ran against a \*\*surrogate oracle\*\*"],
          float(al["n_seeds"]), EXACT, "results/al_savings.json n_seeds",
          "appendix (a)"),
        C("README DRR permutation rank",
          [r"ranks \*\*(\d+)th of 24\*\*"], float(drr["drr_rank_of_n"]), EXACT,
          "results/al_drr.json controls.control_a_permutation.drr_rank_of_n",
          "appendix (a)"),
        C("README DRR permutation count",
          [r"ranks \*\*\d+th of (\d+)\*\*"], float(drr["n_permutations"]), EXACT,
          "results/al_drr.json controls.control_a_permutation.n_permutations",
          "appendix (a)"),
        C("README DRR permutation p",
          [r"permutation p = \*\*([\d.]+)\*\*"], float(drr["permutation_p_value"]),
          TOL2, "results/al_drr.json controls.control_a_permutation.permutation_p_value",
          "appendix (a)"),
        C("README D4 level gap",
          [r"underestimates WER by ([\d.]+) points\*\*"],
          abs(float(s2r["mean_gap"])) * 100.0, TOL1,
          "results/sim2real.json [nova-3.headline].mean_gap", "appendix (c)"),
        C("README D4 CI lo",
          [r"95% CI \[−([\d.]+), −[\d.]+\]"],
          abs(float(s2r["ci"][0])) * 100.0, TOL1,
          "results/sim2real.json [nova-3.headline].ci", "appendix (c)"),
        C("README D4 CI hi",
          [r"95% CI \[−[\d.]+, −([\d.]+)\]"],
          abs(float(s2r["ci"][1])) * 100.0, TOL1,
          "results/sim2real.json [nova-3.headline].ci", "appendix (c)"),
        C("README D4 rank correlation",
          [r"Spearman \*\*ρ = ([\d.]+)\*\*"], float(s2r["spearman"]), TOL3,
          "results/sim2real.json [nova-3.headline].spearman", "appendix (c)"),
        C("README D4 real dead zones",
          [r"— (\d+) real, \d+ found"], float(s2r["n_dead_zones_real"]), EXACT,
          "results/sim2real.json [nova-3.headline].n_dead_zones_real", "appendix (c)"),
        C("README D4 sim dead zones",
          [r"— \d+ real, (\d+) found"], float(s2r["n_dead_zones_sim"]), EXACT,
          "results/sim2real.json [nova-3.headline].n_dead_zones_sim", "appendix (c)"),
        C("README D4 dead-zone Jaccard",
          [r"dead-zone \*\*Jaccard ([\d.]+)\*\*"], float(s2r["dead_zone_jaccard"]),
          EXACT, "results/sim2real.json [nova-3.headline].dead_zone_jaccard",
          "appendix (c)"),
    ]
    ov = m["dead_zone_overlap"]["pairwise"]
    out += [
        C("README nova|whisper dead-zone Jaccard",
          [r"whisper-base \(Jaccard ([\d.]+)\)"],
          float(ov["nova-3|whisper-base"]["jaccard"]), EXACT,
          "results/model_arms.json dead_zone_overlap.pairwise[nova-3|whisper-base]",
          "appendix (c)"),
        C("README scribe|whisper shared dead zones",
          [r"share \*\*(\d+)\*\* \(Jaccard"],
          float(ov["elevenlabs-scribe|whisper-base"]["n_shared"]), EXACT,
          "results/model_arms.json dead_zone_overlap.pairwise"
          "[elevenlabs-scribe|whisper-base].n_shared", "appendix (c)"),
        C("README scribe|whisper dead-zone Jaccard",
          [r"share \*\*\d+\*\* \(Jaccard \*\*([\d.]+)\*\*\)"],
          float(ov["elevenlabs-scribe|whisper-base"]["jaccard"]), TOL3,
          "results/model_arms.json dead_zone_overlap.pairwise"
          "[elevenlabs-scribe|whisper-base].jaccard", "appendix (c)"),
    ]
    return out


_AB_CACHE = {}


def _ab_pair():
    """The two single-degradation conditions the listening demo turns on.

    A = Shower RIR at SNR 20 dB (drenched but quiet); B = Restaurant RIR at
    SNR 0 dB (dry but buried). Paired over the clips both ran — the same
    same-population rule §6.1's correction turns on.
    """
    if _AB_CACHE:
        return _AB_CACHE
    rows = _master_nova(["clip_id", "condition_name", "wer"])
    A = "rt60-1_snr-20_babble_none_roll-0"
    B = "rt60-0.2_snr-0_babble_none_roll-0"
    a = {r["clip_id"]: float(r["wer"]) for r in rows if r["condition_name"] == A}
    b = {r["clip_id"]: float(r["wer"]) for r in rows if r["condition_name"] == B}
    common = sorted(set(a) & set(b))
    if len(common) < 10:
        raise MissingArtifact("master.csv has no A/B pair for the listening demo")
    diffs = [a[c] - b[c] for c in common]
    _AB_CACHE.update({"mean_A": _mean(a[c] for c in common),
                      "mean_B": _mean(b[c] for c in common),
                      "diff": _mean(diffs),
                      "n_same": sum(1 for d in diffs if d == 0.0),
                      "n": len(common)})
    return _AB_CACHE


def _n_overconfident_conditions():
    """Conditions whose same-subset gap is positive — the README's 154 of 169.

    Recomputed with `confidence_gap.py`'s own formula (mean_conf minus clipped
    delivered accuracy, over the clips that emitted words only) because the
    artifact publishes the SHARE and not the count.
    """
    gaps = _clipped_gap_table("nova-3")
    return float(sum(1 for v in gaps.values() if v > 0))


def checks_understanding():
    """report/UNDERSTANDING.md — headline figures only, same artifacts.

    Scope is deliberate (see the DOCS comment): the figures this document quotes
    as AGREEING with `results/` are pinned; the ones it computes itself, and the
    ones where it records a disagreement with the write-up, are not.
    """
    if not os.path.exists(UNDERSTANDING):
        raise MissingArtifact(UNDERSTANDING)
    b = _cg_block("nova-3")
    cg = "results/confidence_gap.txt [nova-3 block]"
    dz = _headline_dead_zone("nova-3")
    cal = _read_json("results/calibration.json")["primary"]
    s2r = _read_json("results/sim2real.json")["nova-3"]["headline"]
    sob = _read_json("results/sobol.json")
    gap = {g["factor"]: g for g in sob["interaction_gap"]}
    n_mute = int(_grab(b, r"(\d+) conditions silent on EVERY clip"))
    n_cond = int(_grab(b, r"conditions: (\d+)"))

    def C(key, patterns, expected, tol, source):
        return Check(key, patterns, expected, tol, source, "UNDERSTANDING.md",
                     doc=UNDERSTANDING)

    return [
        C("U D1 spearman, paired",
          [r"\*\*Spearman ρ = (−?-?[\d.]+)\*\*", r"\*\*(−?-?[\d.]+) paired"],
          float(_grab(b, r"global spearman\(conf_pct, WER_spoke\) = (-?[\d.]+)")),
          TOL3, cg + " global spearman(conf_pct, WER_spoke)"),
        C("U D1 spearman, all-clips",
          [r"paired / (−?-?[\d.]+)"],
          float(_grab(b, r"\[all-clips pairing: (-?[\d.]+)\]")),
          TOL3, cg + " [all-clips pairing]"),
        # NOTE the anchors. A bare `mean gap \*\*\+(...)` also matches this
        # document's RECORD of §6.7's Scribe gap, which is a different arm and a
        # different population — the same conflation the pin exists to catch, so
        # the pattern names nova-3's context rather than the phrase alone.
        C("U D1 mean gap (nova-3, corpus-wide)",
          [r"\(154/169\)\*\*, mean gap \*\*\+([\d.]+)\*\*",
           r"mean overconfidence \*\*\+([\d.]+)\*\*"],
          float(_grab(b, r"gap \(same subset\) mean \+?(-?[\d.]+)")), TOL3,
          cg + " gap (same subset) mean"),
        C("U D1 overconfident share",
          [r"overconfident in (\d+) % of conditions"],
          float(_grab(b, r"overconfident in (\d+)% of conditions")), TOL0,
          cg + " overconfident in N% of conditions"),
        C("U D1 conditions that spoke",
          [r"overconfident in \*\*\d+ of (\d+)\*\*"], n_cond - n_mute, EXACT,
          cg + " conditions - mute conditions"),
        C("U D1 dead-zone count",
          [r"\*\*(\d+) of 176 conditions \(\d\.\d+ %\)\*\*",
           r"\*\*(\d+) of 176 \(\d\.\d+ %\)\*\*"],
          float(_grab(b, r"categories: (\d+) dead zone")), EXACT,
          cg + " categories: N dead zone"),
        C("U D1 dead-zone rate %",
          [r"\*\*\d+ of 176 conditions \((\d\.\d+) %\)\*\*",
           r"\*\*\d+ of 176 \((\d\.\d+) %\)\*\*"],
          float(_grab(b, r"categories: \d+ dead zone \(([\d.]+)%\)")), TOL2,
          cg + " categories: N dead zone (P%)"),
        C("U headline dead zone: confidence",
          [r"mean word confidence \*\*([\d.]+)\*\* at WER"],
          float(dz["mean_conf"]), TOL3,
          "results/dead_zones.csv [#1 dead_zone row].mean_conf"),
        C("U headline dead zone: WER",
          [r"at WER \*\*([\d.]+)\*\*"], float(dz["wer_spoke"]), TOL3,
          "results/dead_zones.csv [#1 dead_zone row].wer_spoke"),
        C("U ECE raw", [r"ECE \*\*([\d.]+) →", r"\| ([\d.]+) \| 0\.0346"],
          float(cal["ece_raw"]["median"]), TOL3, "results/calibration.json"),
        C("U ECE feature-conditioned",
          [r"\*\*→ ([\d.]+)\*\* \(feature-conditioned\)",
           r"0\.0346 \(T = 1\.39\) \| \*\*([\d.]+)\*\* \|"],
          float(cal["ece_feature"]["median"]), TOL3, "results/calibration.json"),
        C("U D4 level gap",
          [r"underestimates WER by ([\d.]+) points",
           r"correct number is ([\d.]+)\."],
          abs(float(s2r["mean_gap"])) * 100.0, TOL1, "results/sim2real.json"),
        C("U D4 rank correlation",
          [r"Spearman \*\*ρ = ([\d.]+)\*\*"], float(s2r["spearman"]), TOL3,
          "results/sim2real.json"),
        C("U ST-S1 gap, rt60",
          [r"`rt60` \*\*([\d.]+)\*\*, `snr_db`"],
          float(gap["rt60"]["gap"]), TOL3, "results/sobol.json interaction_gap[rt60]"),
        C("U ST-S1 gap, snr_db",
          [r"`snr_db` \*\*([\d.]+)\*\*"],
          float(gap["snr_db"]["gap"]), TOL3,
          "results/sobol.json interaction_gap[snr_db]"),
    ]


def checks_status():
    """report/STATUS.md — the handful of figures its state table quotes.

    A status document is the LEAST likely thing to be re-derived and the most
    likely to be copied forward from a log, which is exactly the drift SPEC C.7
    records costing five numbers. So its scalars are pinned to the same artifact
    objects the write-up uses. Its scope/limitation prose carries no figures and
    is deliberately not pinned.
    """
    if not os.path.exists(STATUS):
        raise MissingArtifact(STATUS)
    cal = _read_json("results/calibration.json")["primary"]
    s2r = _read_json("results/sim2real.json")["nova-3"]["headline"]
    b = _cg_block("nova-3")

    def C(key, patterns, expected, tol, source):
        return Check(key, patterns, expected, tol, source, "STATUS.md",
                     doc=STATUS)

    return [
        C("S dead-zone count after the G correction",
          [r"dead zones 6 → (\d+)"],
          float(_grab(b, r"categories: (\d+) dead zone")), EXACT,
          "results/confidence_gap.txt [nova-3] categories: N dead zone"),
        C("S D1 spearman, paired",
          [r"ρ to (−?-?[\d.]+) paired"],
          float(_grab(b, r"global spearman\(conf_pct, WER_spoke\) = (-?[\d.]+)")),
          TOL3, "results/confidence_gap.txt [nova-3] global spearman"),
        C("S ECE raw", [r"ECE ([\d.]+) → 0\.0346"],
          float(cal["ece_raw"]["median"]), TOL3,
          "results/calibration.json primary.ece_raw.median"),
        C("S ECE temperature", [r"ECE 0\.0507 → ([\d.]+) →"],
          float(cal["ece_temperature"]["median"]), TOL3,
          "results/calibration.json primary.ece_temperature.median"),
        C("S ECE feature-conditioned", [r"→ 0\.0346 → ([\d.]+)"],
          float(cal["ece_feature"]["median"]), TOL3,
          "results/calibration.json primary.ece_feature.median"),
        C("S D4 level gap (points)", [r"([\d.]+) pts optimistic"],
          abs(float(s2r["mean_gap"])) * 100.0, TOL1,
          "results/sim2real.json [nova-3.headline] mean_gap"),
        C("S D4 rank correlation", [r"pts optimistic, ρ ([\d.]+)"],
          float(s2r["spearman"]), TOL3,
          "results/sim2real.json [nova-3.headline] spearman"),
    ]


def _conf_char(model):
    """`results/confidence_char.json` block for one arm.

    NOTE the artifact carries bare `NaN` literals (an aggregate whose accuracy is
    undefined on an empty bucket). Python's `json` accepts them; a strict parser
    would not. Read with `_read_json` like everything else so the failure mode is
    a MissingArtifact rather than a silent skip.
    """
    d = _read_json("results/confidence_char.json")
    by = d.get("by_model") or {}
    if model not in by:
        raise MissingArtifact("confidence_char.json has no block for %r" % model)
    return by[model]


# --- §7's `u11` exhibit: the only worked false-alarm-AND-miss example --------
# INTERVIEW_INTERNAL §7 prints four per-word confidences for one clip in one
# condition, and the whole point of the table is the ORDERING between them:
# two wrong words above the utterance mean, and the single LOWEST confidence in
# the utterance sitting on the one word the model got RIGHT. That is the
# concrete answer to "is your confidence signal usable as a gate", so if any of
# those four numbers drifts the exhibit stops making its own argument.

_U11_CELL = ("u11", "rt60-0.45_snr-0_engine_g726_roll-0")


def _u11_live_words():
    """(row, [(op, ref, hyp, conf), ...]) for the demo-live exemplar cell.

    Rebuilt from `results/master.csv` rather than read from a demo cache, so the
    script is pinned to the GRID and not to a presentation artifact that could
    itself have drifted.

    The alignment is the load-bearing part. Hypothesis words are the edits that
    carry a hypothesis token (`match` / `sub` / `ins`), in order — that sequence
    is exactly the hypothesis, so it aligns 1:1 with `word_confidences`. It is
    deliberately NOT `len(word_confidences)`: the two disagree on 225 of 8,797
    rows because vendor confidences are per RAW token while edits are over
    NORMALIZED ones (SPEC G.9), and using the confidence-list length as a word
    count would reintroduce a population mismatch inside the pin for one.
    """
    rows = _master_nova(["clip_id", "condition_name", "edits",
                         "word_confidences", "wer", "mean_conf"])
    hit = [r for r in rows
           if r["clip_id"] == _U11_CELL[0] and r["condition_name"] == _U11_CELL[1]]
    if len(hit) != 1:
        raise MissingArtifact(
            "results/master.csv holds %d non-failed nova-3 rows for %s (expected "
            "exactly 1) — the §7 exhibit cannot be pinned" % (len(hit), _U11_CELL))
    r = hit[0]
    edits = json.loads(r["edits"])
    confs = json.loads(r["word_confidences"])
    hyp = [e for e in edits if e[2] is not None]
    if len(hyp) != len(confs):
        raise MissingArtifact(
            "u11 exemplar: %d hypothesis words vs %d confidences — the 1:1 "
            "alignment the §7 table is read off does not hold"
            % (len(hyp), len(confs)))
    return r, [(op, ref, h, float(c)) for (op, ref, h), c in zip(hyp, confs)]


def _u11_conf(ref_word):
    """Confidence on the hypothesis token aligned to `ref_word`, by name."""
    _, words = _u11_live_words()
    hits = [c for (_op, ref, _h, c) in words if ref == ref_word]
    if len(hits) != 1:
        raise MissingArtifact(
            "u11 exemplar: %d hypothesis tokens align to reference word %r "
            "(expected 1)" % (len(hits), ref_word))
    return hits[0]


def _clean_baseline_row(clip_id):
    path = "results/clean_baseline.csv"
    if not os.path.exists(path):
        raise MissingArtifact(path)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["id"] == clip_id:
                return r
    raise MissingArtifact("%s has no row for clip %s" % (path, clip_id))


def _ab_pair_wer():
    """Mean WER over all 40 clips for §5's two indistinguishable conditions.

    §5's condition card is a two-point counterfactual — neither condition has a
    codec or a mic rolloff — so the card asserts that too, and this recomputes
    both means from the table rather than trusting the card.
    """
    rows = _master_nova(["clip_id", "condition_name", "wer", "codec", "mic_rolloff"])
    out = {}
    for name in ("rt60-1_snr-20_babble_none_roll-0",
                 "rt60-0.2_snr-0_babble_none_roll-0"):
        g = [r for r in rows if r["condition_name"] == name]
        if not g:
            raise MissingArtifact("results/master.csv has no nova-3 rows for %s" % name)
        bad = [r for r in g
               if r["codec"] != "none" or float(r["mic_rolloff"]) != 0.0]
        if bad:
            raise MissingArtifact(
                "%s carries a codec or a mic rolloff, so §5's 'nothing else moves "
                "between them' is false" % name)
        out[name] = _mean(float(r["wer"]) for r in g)
    return out


def _deletion_share_of_errors_matched():
    """del / (sub+del+ins) per arm, on the cells ALL arms ran.

    §8's 63.2 % vs 33.7 % is the 10-clip MATCHED figure and nova-3's own 40-clip
    corpus number for the same quantity is 69.3 %. Six points apart, both
    correct — which is why the document prints the population beside it and why
    this is computed on the intersection rather than per arm in isolation.
    """
    path = "results/master.csv"
    if not os.path.exists(path):
        raise MissingArtifact(path)
    by_model, cells = defaultdict(list), defaultdict(set)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["clip_id"], r["condition_name"])
            by_model[r["model"]].append((key, r))
            cells[r["model"]].add(key)
    if len(cells) < 2:
        raise MissingArtifact("results/master.csv holds fewer than two arms")
    common = set.intersection(*cells.values())
    out = {}
    for model, rows in by_model.items():
        g = [r for key, r in rows if key in common]
        s = sum(float(r["n_sub"] or 0) for r in g)
        d = sum(float(r["n_del"] or 0) for r in g)
        i = sum(float(r["n_ins"] or 0) for r in g)
        if s + d + i <= 0:
            raise MissingArtifact("%s recorded no errors on the matched cells" % model)
        out[model] = d / (s + d + i)
    return out


def _silence_driven_payoff():
    """§6's payoff exhibit: the condition the estimand mismatch was hiding behind.

    It is the OLD #1 dead zone, now classified `silence_driven`, and both halves
    of the exhibit are the point: 10 of 40 clips returned nothing, and on the 30
    that spoke the model was well calibrated. Computed here over the same 40
    clips so the two halves cannot drift into different populations — which is
    the exact defect the exhibit is an example of.
    """
    rows = _master_nova(["clip_id", "condition_name", "transcript", "wer"])
    g = [r for r in rows
         if r["condition_name"] == "rt60-0.7_snr-20_babble_opus-lowrate_roll-1"]
    if not g:
        raise MissingArtifact(
            "results/master.csv has no nova-3 rows for the silence-driven payoff "
            "condition rt60-0.7_snr-20_babble_opus-lowrate_roll-1")
    spoke = [r for r in g if (r["transcript"] or "").strip()]
    if not spoke:
        raise MissingArtifact("payoff condition: no clip produced words")
    return {"n_clips": len(g),
            "n_silent": len(g) - len(spoke),
            "spoke_acc": 1.0 - _mean(float(r["wer"]) for r in spoke)}


def checks_interview_internal():
    """report/INTERVIEW_INTERNAL.md — the private interview script.

    Scope (see the DOCS comment): every figure this document QUOTES ALOUD is
    pinned to the artifact that produced it. Its own commentary — the threshold
    persistence sweep, the re-derived cell-wise bootstrap comparison, and the
    table of places where two documents disagree — is deliberately unpinned,
    because that material exists to record a disagreement and a pin would delete
    it rather than check it.

    Two of these figures exist in NO other pinned document and are pinned here
    for the first time: the clean-confidence anchor (0.962) and the aggregate
    AUROC comparison (mean 0.944 vs min 0.877). They are the answer to "what is
    the confidence score, actually", so they are exactly the figures that would
    be spoken from memory if they drifted.
    """
    if not os.path.exists(INTERVIEW_INTERNAL):
        raise MissingArtifact(INTERVIEW_INTERNAL)
    b = _cg_block("nova-3")
    cg = "results/confidence_gap.txt [nova-3 block]"
    dz = _headline_dead_zone("nova-3")
    cal = _read_json("results/calibration.json")["primary"]
    s2r = _read_json("results/sim2real.json")["nova-3"]["headline"]
    sob = _read_json("results/sobol.json")
    gap = {g["factor"]: g for g in sob["interaction_gap"]}
    ma = _read_json("results/model_arms.json")["per_model"]
    cc = _conf_char("nova-3")
    n_mute = int(_grab(b, r"(\d+) conditions silent on EVERY clip"))
    n_cond = int(_grab(b, r"conditions: (\d+)"))

    # the aggregate table's `min` row, read by name rather than by position
    per_agg = {a["aggregate"]: a for a in cc["separation"]["per_aggregate"]}
    if "min" not in per_agg:
        raise MissingArtifact("confidence_char.json separation has no `min` aggregate")

    # corpus-wide deletion rate: summed from the table, not read from a summary.
    # `results/fingerprints.txt` reports PER-CONDITION composition, so the 0.351
    # the script quotes has to be recomputed the way the document says it was.
    rows = _master_nova(["n_ref", "n_del"])
    del_rate = sum(int(float(r["n_del"])) for r in rows) / \
        float(sum(int(float(r["n_ref"])) for r in rows))

    xm = {k: v["edit_signature_crossmodel"] for k, v in ma.items()}
    ab = _ab_pair_wer()
    dele = _deletion_share_of_errors_matched()
    rooms = _read_json("results/al_drr.json")["rooms"]
    payoff = _silence_driven_payoff()

    def C(key, patterns, expected, tol, source):
        return Check(key, patterns, expected, tol, source, "INTERVIEW_INTERNAL.md",
                     doc=INTERVIEW_INTERNAL)

    return [
        # --- §6 the headline, and §A/Q2's pivot to the threshold-free form ----
        C("II D1 spearman, paired",
          [r"paired ρ = \*\*(−?-?[\d.]+)\*\*"],
          float(_grab(b, r"global spearman\(conf_pct, WER_spoke\) = (-?[\d.]+)")),
          TOL3, cg + " global spearman(conf_pct, WER_spoke)"),
        C("II D1 mean gap",
          [r"mean gap \*\*\+([\d.]+)\*\*"],
          float(_grab(b, r"gap \(same subset\) mean \+?(-?[\d.]+)")), TOL3,
          cg + " gap (same subset) mean"),
        C("II D1 conditions that spoke",
          [r"overconfident in \*\*154 of (\d+)\*\*",
           r"\*\*overconfident in 154 of (\d+)\*\*"],
          n_cond - n_mute, EXACT, cg + " conditions - mute conditions"),
        C("II D1 overconfident share",
          [r"conditions — \*\*(\d+) %\*\*"],
          float(_grab(b, r"overconfident in (\d+)% of conditions")), TOL0,
          cg + " overconfident in N% of conditions"),
        C("II D1 dead-zone count",
          [r"\*\*(\d+) of 176 conditions\*\*"],
          float(_grab(b, r"categories: (\d+) dead zone")), EXACT,
          cg + " categories: N dead zone"),
        # The rate is pinned WITH its population in the pattern. Quoting 1.14 %
        # without "40 clips" is the project's signature bug (SPEC G, J.8), so the
        # regex refuses to match a bare rate.
        C("II D1 dead-zone rate, 40-clip population",
          [r"\*\*(\d\.\d+) %\*\* \(2 of 176, 40 clips\)"],
          float(_grab(b, r"categories: \d+ dead zone \(([\d.]+)%\)")), TOL2,
          cg + " categories: N dead zone (P%)"),
        C("II headline dead zone: confidence",
          [r"mean word confidence \*\*([\d.]+)\*\* at WER \*\*[\d.]+\*\*"],
          float(dz["mean_conf"]), TOL3,
          "results/dead_zones.csv [#1 dead_zone row].mean_conf"),
        C("II headline dead zone: WER",
          [r"mean word confidence \*\*[\d.]+\*\* at WER \*\*([\d.]+)\*\*"],
          float(dz["wer_spoke"]), TOL3,
          "results/dead_zones.csv [#1 dead_zone row].wer_spoke"),

        # --- §A/Q3: what the confidence score empirically IS -----------------
        C("II clean-confidence anchor",
          [r"clean corner of the grid reads \*\*([\d.]+)\*\*",
           r"clean corner \*\*([\d.]+)\*\* \(WER"],
          float(cc["dynamic_range"]["clean_corner"]), TOL3,
          "results/confidence_char.json [nova-3].dynamic_range.clean_corner"),
        C("II best aggregate AUROC",
          [r"AUROC \*\*([\d.]+)\*\*"],
          float(cc["separation"]["best_auroc"]), TOL3,
          "results/confidence_char.json [nova-3].separation.best_auroc"),
        C("II min-aggregate AUROC",
          [r"min \*\*([\d.]+)\*\*"], float(per_agg["min"]["auroc"]), TOL3,
          "results/confidence_char.json [nova-3].separation.per_aggregate[min].auroc"),
        C("II ECE raw", [r"ECE \*\*([\d.]+)\*\* raw"],
          float(cal["ece_raw"]["median"]), TOL3,
          "results/calibration.json primary.ece_raw.median"),
        C("II ECE feature-conditioned",
          [r"\*\*([\d.]+)\*\* feature-conditioned"],
          float(cal["ece_feature"]["median"]), TOL3,
          "results/calibration.json primary.ece_feature.median"),

        # --- §9 the fingerprint headline -------------------------------------
        C("II corpus-wide deletion rate",
          [r"Deletions are \*\*([\d.]+)\*\* of reference words",
           r"- del \*\*([\d.]+)\*\* ·"],
          del_rate, TOL3,
          "results/master.csv sum(n_del)/sum(n_ref) over non-failed nova-3 rows"),

        # --- §11 the pre-registration verdict --------------------------------
        C("II ST-S1 gap, rt60", [r"\*\*ST − S1 = ([\d.]+)\*\*"],
          float(gap["rt60"]["gap"]), TOL3,
          "results/sobol.json interaction_gap[rt60]"),
        C("II ST-S1 gap, snr_db",
          [r"\*\*([\d.]+)\*\* for SNR", r"\*\*([\d.]+)\*\* for `snr_db`"],
          float(gap["snr_db"]["gap"]), TOL3,
          "results/sobol.json interaction_gap[snr_db]"),

        # --- §2 / §C sim-vs-real ---------------------------------------------
        C("II D4 level gap", [r"\*\*([\d.]+) points optimistic\*\*"],
          abs(float(s2r["mean_gap"])) * 100.0, TOL1,
          "results/sim2real.json [nova-3.headline] mean_gap"),
        C("II D4 rank correlation", [r"rank correlation \*\*([\d.]+)\*\*"],
          float(s2r["spearman"]), TOL3,
          "results/sim2real.json [nova-3.headline] spearman"),

        # --- §8 the three-arm section, every figure carrying its population ---
        C("II L1 nova-3 dead-zone rate, 10-clip population",
          [r"\*\*(\d\.\d+) %\*\* \(1 of 176, 10 clips\)"],
          100.0 * float(ma["nova-3"]["dead_zone_rate"]), TOL2,
          "results/model_arms.json per_model[nova-3].dead_zone_rate"),
        C("II L1 whisper-base dead-zone rate, 10-clip population",
          [r"\*\*(\d+\.\d+) %\*\* \(69 of 176, 10 clips\)"],
          100.0 * float(ma["whisper-base"]["dead_zone_rate"]), TOL2,
          "results/model_arms.json per_model[whisper-base].dead_zone_rate"),
        C("II L1 nova-3 confidence-vs-WER shape",
          [r"nova-3 ρ = \*\*(−?-?[\d.]+)\*\*"],
          float(ma["nova-3"]["shape"]["spearman"]), TOL3,
          "results/model_arms.json per_model[nova-3].shape.spearman"),
        C("II L1 whisper-base confidence-vs-WER shape",
          [r"whisper-base ρ = \*\*(−?-?[\d.]+)\*\*"],
          float(ma["whisper-base"]["shape"]["spearman"]), TOL3,
          "results/model_arms.json per_model[whisper-base].shape.spearman"),
        # Anchored to its own sentence, not to "any multiplier in the file":
        # a bare `**N.N×**` pattern silently acquires a second site the moment
        # another ratio is written anywhere in the document, and then the check
        # is about whichever one comes first (SPEC J.7).
        C("II L1 whisper insertion rate over nova-3's",
          [r"insertions\s*\n?>?\s*are \*\*(\d+\.\d+)×\*\*"],
          float(xm["whisper-base"]["ins"]) / float(xm["nova-3"]["ins"]), TOL1,
          "results/model_arms.json per_model[*].edit_signature_crossmodel ins ratio"),

        # --- §7 the `u11` exhibit: false alarm AND miss on one utterance -----
        # The ORDERING is the argument, so all four are pinned. If `martinez`
        # ever stops being the lowest number in the table the exhibit is making
        # the opposite point from the one the prose claims.
        C("II u11 wrong word `street`->`three`",
          [r"`street` → `three` \| \*\*wrong\*\* \| \*\*([\d.]+)\*\*"],
          _u11_conf("street"), TOL3,
          "results/master.csv [nova-3 u11 @ dead zone #1] word_confidences, "
          "aligned via edits"),
        C("II u11 wrong word `elm`->`l`",
          [r"`elm` → `l` \| \*\*wrong\*\* \| \*\*([\d.]+)\*\*"],
          _u11_conf("elm"), TOL3,
          "results/master.csv [nova-3 u11 @ dead zone #1] word_confidences, "
          "aligned via edits"),
        C("II u11 CORRECT word `martinez` (the lowest in the utterance)",
          [r"`martinez` \| \*\*right\*\* \| \*\*([\d.]+)\*\*"],
          _u11_conf("martinez"), TOL3,
          "results/master.csv [nova-3 u11 @ dead zone #1] word_confidences, "
          "aligned via edits"),
        C("II u11 utterance mean confidence",
          [r"Utterance mean \*\*([\d.]+)\*\*"],
          float(_u11_live_words()[0]["mean_conf"]), TOL3,
          "results/master.csv [nova-3 u11 @ dead zone #1].mean_conf"),
        C("II u11 clean control confidence",
          [r"clean control \*\*WER [\d.]+ at confidence ([\d.]+)\*\*"],
          float(_clean_baseline_row("u11")["mean_conf"]), TOL3,
          "results/clean_baseline.csv [u11].mean_conf"),

        # --- §5 the indistinguishable pair, as a two-point counterfactual ----
        C("II A/B condition A mean WER",
          [r"\*\*A\*\* \|.*\*\*([\d.]+)\*\* \|"],
          ab["rt60-1_snr-20_babble_none_roll-0"], TOL3,
          "results/master.csv mean wer over 40 nova-3 clips "
          "[rt60-1_snr-20_babble_none_roll-0]"),
        C("II A/B condition B mean WER",
          [r"\*\*B\*\* \|.*\*\*([\d.]+)\*\* \|"],
          ab["rt60-0.2_snr-0_babble_none_roll-0"], TOL3,
          "results/master.csv mean wer over 40 nova-3 clips "
          "[rt60-0.2_snr-0_babble_none_roll-0]"),

        # --- §8 the failure mode a confidence monitor cannot see -------------
        # Pinned WITH its population in the pattern, like the dead-zone rates:
        # the same quantity on nova-3's own 40-clip corpus is 69.3 %, six points
        # away, so a bare percentage here would be quotable as either.
        C("II deletion share of errors, nova-3, 10-clip matched",
          [r"\*\*([\d.]+) % of nova-3's errors carry no confidence at all\*\*"],
          100.0 * dele["nova-3"], TOL1,
          "results/master.csv del/(sub+del+ins) over the cells all arms ran"),
        C("II deletion share of errors, scribe, 10-clip matched",
          [r"\*\*([\d.]+) % for Scribe\.\*\*"],
          100.0 * dele["elevenlabs-scribe"], TOL1,
          "results/master.csv del/(sub+del+ins) over the cells all arms ran"),

        # --- §A/Q1 the DRR ordering, which IS the mechanistic claim ----------
        C("II DRR-WER rank correlation",
          [r"\*\*ρ\(DRR, WER\) = (−?-?[\d.]+)\*\*"],
          _spearman([r["drr_db"] for r in rooms],
                    [r["marginal_wer"] for r in rooms]), TOL3,
          "results/al_drr.json rooms[] spearman(drr_db, marginal_wer)"),
        C("II RT60-WER rank correlation",
          [r"\*\*ρ\(RT60, WER\) = \+([\d.]+)\*\*"],
          _spearman([r["rt60_measured"] for r in rooms],
                    [r["marginal_wer"] for r in rooms]), TOL3,
          "results/al_drr.json rooms[] spearman(rt60_measured, marginal_wer)"),

        # --- §11 why the pre-registration interval is deliberately wide ------
        C("II preregistration CI width ratio, rt60",
          [r"\*\*([\d.]+)× wider than the direct"],
          float(gap["rt60"]["gap_conf_ratio_quadrature_over_direct"]), TOL2,
          "results/sobol.json interaction_gap[rt60]"
          ".gap_conf_ratio_quadrature_over_direct"),
        C("II preregistration CI width ratio, snr_db",
          [r"and ([\d.]+)× for `snr_db`\*\*"],
          float(gap["snr_db"]["gap_conf_ratio_quadrature_over_direct"]), TOL2,
          "results/sobol.json interaction_gap[snr_db]"
          ".gap_conf_ratio_quadrature_over_direct"),
        C("II S1/ST bootstrap correlation, rt60",
          [r"\*\*\+([\d.]+)\*\* for `rt60`"],
          float(gap["rt60"]["s1_st_bootstrap_corr"]), TOL3,
          "results/sobol.json interaction_gap[rt60].s1_st_bootstrap_corr"),
        C("II S1/ST bootstrap correlation, snr_db",
          [r"\*\*\+([\d.]+)\*\* for `snr_db`"],
          float(gap["snr_db"]["s1_st_bootstrap_corr"]), TOL3,
          "results/sobol.json interaction_gap[snr_db].s1_st_bootstrap_corr"),

        # --- §6 the silence-driven payoff exhibit ----------------------------
        C("II payoff condition: silent clips",
          [r"\*\*(\d+) of the 40 clips returned nothing at all\.\*\*"],
          payoff["n_silent"], EXACT,
          "results/master.csv empty transcripts "
          "[nova-3 rt60-0.7_snr-20_babble_opus-lowrate_roll-1]"),
        C("II payoff condition: accuracy on the clips it spoke on",
          [r"it was \*\*([\d.]+) % accurate at [\d.]+\ns*confidence",
           r"it was \*\*([\d.]+) % accurate"],
          100.0 * payoff["spoke_acc"], TOL1,
          "results/master.csv 1 - mean(wer) over the non-empty rows "
          "[nova-3 rt60-0.7_snr-20_babble_opus-lowrate_roll-1]"),
    ]


CHECK_GROUPS = [
    ("D1 headline", checks_d1_headline),
    ("STATUS doc", checks_status),
    ("README summary", checks_readme),
    ("UNDERSTANDING prep doc", checks_understanding),
    ("INTERVIEW_INTERNAL script", checks_interview_internal),
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

    failures = []
    per_doc = defaultdict(lambda: [0, 0])   # doc -> [checks, prose sites]
    for chk in checks:
        if chk.doc not in docs:
            continue
        per_doc[chk.doc][0] += 1
        try:
            per_doc[chk.doc][1] += apply_check(docs[chk.doc], chk)
        except Drift as exc:
            failures.append(str(exc))

    for name, why in skipped:
        print("  skipped group %-24s (%s)" % (name, why))

    if failures:
        raise Drift("\n\n%d of %d pinned figures no longer match their artifact:\n%s"
                    % (len(failures), len(checks), "\n".join(failures)))

    for doc in DOCS:
        if doc in per_doc:
            n_chk, n_sites = per_doc[doc]
            print("OK: %d pinned figures matched their artifacts across %d prose "
                  "sites in %s" % (n_chk, n_sites, doc))


def test_every_check_can_actually_fail():
    """NEGATIVE CONTROL. Mutate the prose behind each check; each must then fail.

    A pinning test that passes against wrong prose is worse than no test — it
    certifies the drift. So every check is exercised against a document in which
    its own figure has been changed, and is required to notice.

    The mutation is applied to THE CHECK'S OWN DOCUMENT. Mutating a single
    document for every check would let a README check "pass" this control by
    noticing a corrupted *write-up* — which is precisely the cross-document
    confusion the README group exists to prevent.
    """
    docs = _docs()
    if WRITEUP not in docs:
        print("SKIP: %s not present" % WRITEUP)
        return
    checks, _ = _collect()
    if not checks:
        print("SKIP: results/ is absent (gitignored)")
        return

    blind = []
    for chk in checks:
        if chk.doc not in docs:
            continue
        mutated = _mutate_first_site(docs[chk.doc], chk)
        if mutated is None:
            blind.append("%s [%s] (could not build a mutation)" % (chk.key, chk.doc))
            continue
        try:
            apply_check(mutated, chk)
        except Drift:
            continue
        blind.append("%s [%s]" % (chk.key, chk.doc))

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
              # report/INTERVIEW_RUNBOOK.md was here and is now DELETED, not
              # exempted. It was the document the presenter read most and the
              # only one whose numbers nothing checked -- the J.7 gap, sitting
              # in the exempt set that J.7 itself warns is where numbers rot.
              # Superseded by report/INTERVIEW_INTERNAL.md, which is in DOCS
              # and pinned. One presenter document, checked, beats two.
              # Stage directions moved off the demo's screen. Same category as
              # the runbook -- not a deliverable, never quoted -- and it is
              # written to quote no figures at all, naming the artifact each
              # note depends on instead, so there is nothing in it to drift.
              "report/_demo_internal_notes.md",
              # Research note on what each arm's confidence FIELD is, sourced to
              # vendor docs/model cards/shipped source. Exempt on the SAME
              # ground as the two above and no other: it is written to quote no
              # `results/` figure at all -- every observation it explains is
              # named ordinally with the artifact that carries the number -- so
              # there is nothing in it to drift. The numerals it does contain
              # are vendor facts (74 M params, 680 k hours) and shipped-source
              # constants (whisper's compression_ratio_threshold=2.4), none of
              # which this project measures. If a `results/` figure is ever
              # pasted into it, delete this entry and give it checks instead.
              "report/model_architecture_notes.md",
              }
    # report/UNDERSTANDING.md is NOT exempt — it is in DOCS with its headline
    # figures pinned (see `checks_understanding`), which is the point of this
    # assertion rather than an exception to it.
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
