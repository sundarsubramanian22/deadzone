"""
test_provenance.py — offline tests for the two files that record what the
experiment USED and what it COST: `scripts/make_manifest.py` and
`requirements.txt`.

Neither is an analysis layer, so neither had a suite. Both failed an audit in the
same way, and it is the way this project keeps failing: a MISSING thing that
looks exactly like a zero.

  * `_cost_estimate` summed the `nova-3` key only, with the rate written inline.
    Correct for a two-arm grid whose second arm was free — and silently wrong the
    moment a third, BILLED arm exists. An ElevenLabs Scribe arm would have
    transcribed thousands of clips and contributed exactly $0.00 to the freeze,
    with no error, no missing field and a total that still looked plausible.
  * `requests` is imported by three sites in this repo and was declared in no
    requirements file — it resolved as a transitive dependency, which works right
    up until whichever package was dragging it in drops it.

Every test here constructs the violating input, asserts the loud behaviour, and
carries a negative control so it is pinned to the violation rather than to some
incidental property of the fixture.

Run: python3 tests/test_provenance.py
"""
from __future__ import annotations

# --- repo-root bootstrap -------------------------------------------------
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# -------------------------------------------------------------------------

import os
import re

from scripts.make_manifest import (                                # noqa: E402
    MINUTES_PER_CALL, MODEL_RATES, _cost_estimate,
)

_FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        _FAILS.append(name)


def totals(**per_model: int) -> dict:
    """One 'real' cache with the given ok-call counts per model."""
    return {"real": {m: {"ok": n, "failed": 0} for m, n in per_model.items()}}


# ===========================================================================
print("\n[1] the rate table covers every registry arm, and says so per row")
# ===========================================================================

from deadzone.model_compare import MODEL_REGISTRY                  # noqa: E402

missing = [m for m in MODEL_REGISTRY if m not in MODEL_RATES]
check("every MODEL_REGISTRY arm has a rate row", not missing, str(missing))
check("rates are keyed by REGISTRY names (what lands in the `model` column)",
      set(MODEL_RATES) >= set(MODEL_REGISTRY))
for m, r in MODEL_RATES.items():
    check(f"{m}: quotes a per-minute rate", "usd_per_minute" in r)
    check(f"{m}: says when the rate was read", bool(r.get("rate_as_of")))
    check(f"{m}: names the provider and the source", bool(r.get("provider"))
          and bool(r.get("source")))
# LOCAL ARMS ARE LISTED, NOT OMITTED — otherwise "absent from the table" would
# mean both 'free' and 'forgotten', which is the ambiguity the warning exists to
# remove.
check("local arms are present at $0 rather than omitted",
      MODEL_RATES["whisper-base"]["usd_per_minute"] == 0.0
      and MODEL_RATES["vosk"]["usd_per_minute"] == 0.0)
check("elevenlabs is billed per HOUR and converted, not guessed",
      abs(MODEL_RATES["elevenlabs-scribe"]["usd_per_minute"]
          - MODEL_RATES["elevenlabs-scribe"]["usd_per_hour"] / 60.0) < 1e-15)


# ===========================================================================
print("\n[2] THE VIOLATION: a billed arm with no rate row is named, not zeroed")
# ===========================================================================

t_unknown = totals(**{"nova-3": 1000, "some-new-arm": 5000})
res_unknown = _cost_estimate(t_unknown)

check("the unknown arm is listed under uncosted_models",
      res_unknown["uncosted_models"] == ["some-new-arm"],
      str(res_unknown["uncosted_models"]))
check("its usd_est is None, never 0.0 (a zero reads as 'free')",
      res_unknown["per_model"]["some-new-arm"]["usd_est"] is None
      and res_unknown["per_model"]["some-new-arm"]["costed"] is False)
check("it carries a warning naming itself and the fix",
      "NO RATE ON FILE" in res_unknown["per_model"]["some-new-arm"]["warning"]
      and "MODEL_RATES" in res_unknown["per_model"]["some-new-arm"]["warning"])
check("its CALLS are still counted (the arm is not made invisible)",
      res_unknown["per_model"]["some-new-arm"]["calls"] == 5000
      and res_unknown["total_calls"] == 6000)

# NEGATIVE CONTROL: drop the unknown arm and the money total is IDENTICAL. That
# is the point — the un-costed arm contributed exactly nothing to the dollar
# figure, which is why it has to be named out loud rather than folded in.
res_known = _cost_estimate(totals(**{"nova-3": 1000}))
check("neg control: removing the un-costed arm does not move usd_total_est",
      res_unknown["usd_total_est"] == res_known["usd_total_est"],
      f"{res_unknown['usd_total_est']} vs {res_known['usd_total_est']}")
check("neg control: with only known arms, uncosted_models is empty",
      res_known["uncosted_models"] == [])
check("neg control: total_calls DOES move (so the fixture really differed)",
      res_unknown["total_calls"] != res_known["total_calls"])


# ===========================================================================
print("\n[3] a SECOND billed vendor is summed, not dropped (the ElevenLabs case)")
# ===========================================================================

n_dg, n_el = 1000, 2000
res_two = _cost_estimate(totals(**{"nova-3": n_dg, "elevenlabs-scribe": n_el,
                                   "whisper-base": 500}))
want_dg = n_dg * MINUTES_PER_CALL * MODEL_RATES["nova-3"]["usd_per_minute"]
want_el = n_el * MINUTES_PER_CALL * MODEL_RATES["elevenlabs-scribe"]["usd_per_minute"]

check("both billed arms appear in billed_models",
      sorted(res_two["billed_models"]) == ["elevenlabs-scribe", "nova-3"],
      str(res_two["billed_models"]))
check("the scribe arm's spend is priced at $0.22/hr",
      abs(res_two["per_model"]["elevenlabs-scribe"]["usd_est"] - round(want_el, 2))
      < 1e-9, str(res_two["per_model"]["elevenlabs-scribe"]))
check("the total is the SUM over vendors, not the deepgram arm alone",
      abs(res_two["usd_total_est"] - round(want_dg + want_el, 2)) < 0.011,
      f"{res_two['usd_total_est']} vs {round(want_dg + want_el, 2)}")
check("the local arm is present at $0 and does not perturb the total",
      res_two["per_model"]["whisper-base"]["usd_est"] == 0.0
      and res_two["per_model"]["whisper-base"]["costed"] is True)
# THE REGRESSION THIS EXISTS TO CATCH: the old implementation would have returned
# exactly the deepgram number here.
check("the old deepgram-only total would have been strictly SMALLER",
      res_two["usd_total_est"] > round(want_dg, 2),
      f"{res_two['usd_total_est']} vs deepgram-only {round(want_dg, 2)}")


# ===========================================================================
print("\n[4] real + sim caches are summed per model, and legacy fields hold")
# ===========================================================================

both = {"real": {"nova-3": {"ok": 7220, "failed": 0},
                 "whisper-base": {"ok": 1757, "failed": 3}},
        "sim": {"nova-3": {"ok": 3855, "failed": 11}}}
res_both = _cost_estimate(both)

check("nova-3 calls sum across the real and sim caches",
      res_both["per_model"]["nova-3"]["calls"] == 7220 + 3855 + 11,
      str(res_both["per_model"]["nova-3"]["calls"]))
check("failed calls are counted (they were still billed)",
      res_both["per_model"]["nova-3"]["calls_failed"] == 11
      and res_both["per_model"]["whisper-base"]["calls_failed"] == 3)
# The write-up quotes "11,086 Deepgram calls, ~$3.26". Those two numbers must
# survive the generalization untouched, under the names they already have.
check("legacy `deepgram_calls` still reads 11086 on the real cache shape",
      res_both["deepgram_calls"] == 11086, str(res_both["deepgram_calls"]))
check("legacy `usd_total_est` still reads 3.26",
      res_both["usd_total_est"] == 3.26, str(res_both["usd_total_est"]))
check("legacy `audio_minutes_est` is still the DEEPGRAM arm's 757.5",
      res_both["audio_minutes_est"] == 757.5, str(res_both["audio_minutes_est"]))
check("the whole-grid figures are separate and larger",
      res_both["total_calls"] == 12846
      and res_both["total_audio_minutes_est"] > res_both["audio_minutes_est"])
check("the legacy fields say out loud that they are deepgram-only",
      "DEEPGRAM arm alone" in res_both["legacy_field_note"])


# ===========================================================================
print("\n[5] requirements.txt declares `requests` as the DIRECT dependency it is")
# ===========================================================================
#
# `requests` is imported by three sites here: audio_pipeline.transcribe_elevenlabs
# (lazy), scripts/probe_elevenlabs.py (module level) and tests/test_adapters.py.
# It resolved only transitively until now.

_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def declared(text: str) -> set[str]:
    """Distribution names declared in a requirements file, comments stripped."""
    out = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _REQ_LINE.match(line)
        if m:
            out.add(m.group(1).lower().replace("_", "-"))
    return out


req_txt = open(os.path.join(_REPO_ROOT, "requirements.txt"), encoding="utf-8").read()
req_names = declared(req_txt)

# the dependency is REAL: prove the import sites exist before demanding the pin,
# so this is not a decorative requirement nobody uses.
sites = []
for rel in ("deadzone/audio_pipeline.py", "scripts/probe_elevenlabs.py",
            "tests/test_adapters.py"):
    p = os.path.join(_REPO_ROOT, rel)
    if os.path.exists(p) and re.search(r"^\s*import requests\b",
                                       open(p, encoding="utf-8").read(), re.M):
        sites.append(rel)
check("`requests` is genuinely imported by repo code", len(sites) >= 2, str(sites))
check("requirements.txt declares requests", "requests" in req_names,
      str(sorted(req_names)))

# NEGATIVE CONTROL on the parser itself, both branches — otherwise a parser that
# returned every token would "pass" against any file at all.
check("parser: a file WITHOUT requests does not report it",
      "requests" not in declared("numpy\nscipy\n# requests is only a comment\n"))
check("parser: a commented-out pin does not count as a declaration",
      "requests" not in declared("numpy\n#requests\n"))
check("parser: an inline comment after the name still counts",
      "requests" in declared("requests  # ElevenLabs arm speaks plain HTTP\n"))

lock = os.path.join(_REPO_ROOT, "requirements.lock.txt")
if os.path.exists(lock):
    check("the lock file already pins requests (so the pin is not aspirational)",
          "requests" in declared(open(lock, encoding="utf-8").read()))
else:
    print("  skip requirements.lock.txt not present")


# ===========================================================================
print()
if _FAILS:
    print(f"FAILED {len(_FAILS)} check(s): {_FAILS}")
    raise SystemExit(1)
print("test_provenance.py — ALL CHECKS PASSED")
