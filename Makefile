# Deadzone — demo kit (SPEC A.R9)
#
# One command per demo, nothing multi-step.
#
#   make help          what you can run
#   make demo          THE HERO. One clip, played and transcribed LIVE, twice.
#   make demo-all      the full scripted path (hero + AL + dashboard), in order
#
# WHAT CHANGED, AND WHY IT IS STATED HERE. `make demo` used to chain
# test-core -> demo-break -> demo-al -> dashboard, and every step was offline by
# construction. It is now the single merged hero beat, which DOES call the API —
# because the one thing a cached demo cannot show is the payload arriving. The
# offline guarantee has not been dropped, it has been moved: `demo_hero.py`
# falls back to the archived measurements and exits 0 on a missing key, a dead
# network, a vendor error or a timeout, and `make demo-replay` runs the whole
# beat from cache with no network at all. The old chain is `make demo-all`.
#
# `demo-break` and `demo-live` survive as hidden fallback targets. They are not
# on the on-stage list any more (the hero does both jobs), but they still work,
# and they are what you reach for if the hero misbehaves on the day.
.DEFAULT_GOAL := help
.PHONY: help lock test test-core demo demo-all demo-replay demo-prep demo-break \
        demo-hero demo-al demo-check dashboard dashboard-build clean-demo \
        demo-live demo-listen

PY       := ./.venv/bin/python
# NOTE: `./.venv/bin/pip` is a broken console script on this machine — its shebang
# still points at the venv's pre-rename path. `python -m pip` is immune to that,
# so always go through the interpreter.
PIP      := $(PY) -m pip
DASH     := dashboard/deadzone.html

# Auto-discovered so a suite added tomorrow is picked up without editing this
# file, and a suite that does not exist yet simply is not run.
TESTS      := $(wildcard tests/test_*.py)
CORE_TESTS := $(wildcard tests/test_pipeline.py tests/test_conditions.py tests/test_design.py tests/test_demo.py)

# macOS `open`, Linux `xdg-open`; whichever exists.
OPEN := $(shell command -v open 2>/dev/null || command -v xdg-open 2>/dev/null)


help:
	@echo ""
	@echo "  Deadzone demo kit."
	@echo ""
	@echo "  THE HERO"
	@echo "    make demo             2 min  one clip, played and transcribed LIVE, twice:"
	@echo "                                 raw, then in a measured dead zone. You pick the"
	@echo "                                 clip. This is the ONE target that needs wifi +"
	@echo "                                 DEEPGRAM_API_KEY — and it is safe without them:"
	@echo "                                 no key, no network, a vendor error or a timeout"
	@echo "                                 each print one line, fall back to the archived"
	@echo "                                 measurements, and exit 0."
	@echo "    make demo-replay      2 min  the same beat entirely from cache — rehearsal"
	@echo "                                 mode, and the instant fallback. No network."
	@echo ""
	@echo "  ALSO ON STAGE"
	@echo "    make test-core        30 s  the trap functions + the demo kit, green"
	@echo "    make demo-al          30 s  the surrogate walking onto the boundary"
	@echo "    make demo-listen      3 min INTERACTIVE — they listen, rank, then see the tie"
	@echo "    make dashboard         —    open the self-contained HTML from file://"
	@echo "    make demo-all               test-core + hero + demo-al + dashboard, in order"
	@echo ""
	@echo "  BEFORE STAGE"
	@echo "    make demo-check             preflight: is every artifact present?"
	@echo "    make demo-prep              bake results/demo/ (offline DSP, no API)"
	@echo "    make lock                   refresh requirements.lock.txt"
	@echo "    make test                   every test_*.py ($(words $(TESTS)) suites, ~60 s)"
	@echo "    make dashboard-build        regenerate $(DASH) from results/master.csv"
	@echo ""
	@echo "  FALLBACKS (still supported; superseded on stage by 'make demo')"
	@echo "    make demo-break             the offline half alone: audio + cached numbers"
	@echo "    make demo-live              the live half alone: two calls, no audio"
	@echo ""
	@echo "  CLEANUP"
	@echo "    make clean-demo             delete results/demo/"
	@echo ""


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

## every discovered suite; non-zero exit if any one of them fails
test:
	@fail=0; \
	for t in $(TESTS); do \
	  log=/tmp/deadzone-$$(basename $$t).log; \
	  printf '  %-30s ' "$$t"; \
	  if $(PY) $$t >$$log 2>&1; then echo "ok"; \
	  else echo "FAIL  (see $$log)"; fail=1; fi; \
	done; \
	if [ $$fail -eq 0 ]; then echo ""; echo "  all $(words $(TESTS)) suites green"; \
	else echo ""; echo "  SUITES FAILED"; fi; \
	exit $$fail

## the demo opener: the correctness-critical core, fast and deterministic
test-core:
	@echo ""
	@echo "  The three trap functions produce clean-looking GARBAGE if subtly wrong,"
	@echo "  with no error message. These are the tests that stop that."
	@echo ""
	@fail=0; \
	for t in $(CORE_TESTS); do \
	  log=/tmp/deadzone-$$(basename $$t).log; \
	  printf '  %-30s ' "$$t"; \
	  if $(PY) $$t >$$log 2>&1; then echo "ok"; \
	  else echo "FAIL  (see $$log)"; fail=1; fi; \
	done; \
	echo ""; \
	exit $$fail


# --------------------------------------------------------------------------
# the demos
# --------------------------------------------------------------------------

## bake the offline demo artifacts: two wavs per exemplar + every cached number
## + the listening set (results/audio/demo/, run-of-show in its DEMO_SCRIPT.md)
demo-prep:
	$(PY) demos/demo_break.py --prepare
	$(PY) demos/demo_hero.py --prepare
	$(PY) scripts/make_demo_audio.py

results/demo/demo_cache.json:
	@$(MAKE) --no-print-directory demo-prep

results/demo/hero/hero_cache.json:
	@$(PY) demos/demo_hero.py --prepare

## THE HERO. One clip, played and transcribed LIVE, twice. The interviewer picks
## the clip from a menu of measured dead zones (or takes a random one). Both
## calls are real; every number on screen comes from the two responses that just
## arrived, and the archived grid row is shown afterwards as corroboration —
## never as a stand-in.
##
## It is safe to run with no key and no network: each failure prints ONE line,
## falls back to the archived measurements for the same clip and condition, and
## exits 0. `make demo-replay` is the same beat from cache, for rehearsal.
demo: results/demo/hero/hero_cache.json
	@$(PY) demos/demo_hero.py

demo-hero: demo

## rehearsal / fallback: the whole hero beat with no network at all
demo-replay: results/demo/hero/hero_cache.json
	@$(PY) demos/demo_hero.py --replay

## FALLBACK (hidden from `make help`'s on-stage list). The offline half of the
## hero on its own: audio plus the cached measurements, no API call. Kept
## working because on the day it is the thing that cannot fail.
demo-break: results/demo/demo_cache.json
	@$(PY) demos/demo_break.py

## the active-learning loop, against a GP surrogate fitted to the real grid
demo-al:
	@$(PY) demos/demo_al.py

## THE INTERACTIVE BEAT — the interviewer listens and ranks, THEN learns the
## model scored those clips exactly equal. Offline, no API key, plays wavs
## through afplay/aplay/play/ffplay. It waits for a human, so it is
## deliberately NOT in the `demo` chain, which must run unattended.
## The beat ENDS at the last pair's reveal — nothing auto-plays after it.
##   demos/demo_listen.py --check     preflight
##   demos/demo_listen.py --replay    rehearse with the recorded 2026-08-05 session
##   demos/demo_listen.py --full      + the measured half, the failed prediction,
##                                      the third pair and the closing (all opt-in;
##                                      also --measured/--prediction/--payoff/--closing)
demo-listen:
	@$(PY) demos/demo_listen.py

## FALLBACK (hidden from `make help`'s on-stage list). The live half of the hero
## on its own: two real nova-3 calls on one hardcoded clip, no audio playback.
## Superseded by `make demo`, which does this AND plays the audio AND lets the
## interviewer choose the clip. Kept because it is a second, independently
## written path to the same evidence, and on the day two paths beat one.
demo-live:
	@$(PY) demos/demo_live.py

## open the self-contained dashboard from file:// — no server, no wifi
dashboard:
	@if [ ! -f $(DASH) ]; then \
	  echo "  $(DASH) missing — run: make dashboard-build"; exit 1; fi
	@echo "  opening file://$(CURDIR)/$(DASH)"
	@if [ -n "$(OPEN)" ]; then $(OPEN) "file://$(CURDIR)/$(DASH)"; \
	 else echo "  no opener found; point a browser at the URL above"; fi

## regenerate the dashboard from the real grid (~20 s; NOT needed on stage)
dashboard-build:
	$(PY) dashboard/build.py --master results/master.csv

## the whole scripted path, in the rehearsed order. This is what `make demo`
## used to be, minus demo-break — the hero replaces it and does strictly more.
demo-all: test-core demo demo-al dashboard
	@echo ""
	@echo "  Done. The dashboard is open; the 3-minute path is in dashboard/DEMO.md."
	@echo ""


# --------------------------------------------------------------------------
# hygiene
# --------------------------------------------------------------------------

## is every artifact the demo needs actually on disk, right now, offline?
demo-check:
	@$(PY) demos/demo_hero.py --check
	@$(PY) demos/demo_break.py --check
	@$(PY) scripts/make_demo_audio.py --check

## pin exactly what is installed, so the demo machine is reproducible
lock:
	$(PIP) freeze > requirements.lock.txt
	@echo "  wrote requirements.lock.txt ($$(wc -l < requirements.lock.txt | tr -d ' ') packages)"

clean-demo:
	rm -rf results/demo
	@echo "  removed results/demo/ — run 'make demo-prep' to rebuild it"
