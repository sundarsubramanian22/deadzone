# Deadzone — demo kit (SPEC A.R9)
#
# One command per demo, nothing multi-step. The entire on-stage path — `make demo`
# and every target it chains — needs NO API key and touches NO network. Rehearse
# with wifi OFF.
#
# `make demo-live` is the sole exception: an OPTIONAL extra beat that needs both.
# It is deliberately not a prerequisite of `demo`, and it falls back to cache and
# exits 0 when either is missing.
#
#   make help          what you can run
#   make demo          the full scripted path, in order
#
.DEFAULT_GOAL := help
.PHONY: help lock test test-core demo demo-prep demo-break demo-al demo-check \
        dashboard dashboard-build clean-demo demo-live demo-listen

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
	@echo "  Deadzone demo kit — the whole ON-STAGE path runs OFFLINE, no API key required."
	@echo ""
	@echo "  ON STAGE (in this order)"
	@echo "    make test-core        30 s  the trap functions + the demo kit, green"
	@echo "    make demo-break       60 s  one clip, clean -> measured dead zone"
	@echo "    make demo-al          30 s  the surrogate walking onto the boundary"
	@echo "    make demo-listen      3 min INTERACTIVE — they listen, rank, then see the tie"
	@echo "    make dashboard         —    open the self-contained HTML from file://"
	@echo "    make demo                   all four, in sequence"
	@echo ""
	@echo "  OPTIONAL — the only target that touches the network"
	@echo "    make demo-live        20 s  the same beat, transcribed LIVE"
	@echo "                                NEEDS wifi + DEEPGRAM_API_KEY. Skippable:"
	@echo "                                it falls back to cache and exits 0."
	@echo ""
	@echo "  BEFORE STAGE"
	@echo "    make demo-check             preflight: is every artifact present?"
	@echo "    make demo-prep              bake results/demo/ (offline DSP, no API)"
	@echo "    make lock                   refresh requirements.lock.txt"
	@echo "    make test                   every test_*.py ($(words $(TESTS)) suites, ~60 s)"
	@echo "    make dashboard-build        regenerate $(DASH) from results/master.csv"
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
	$(PY) scripts/make_demo_audio.py

results/demo/demo_cache.json:
	@$(MAKE) --no-print-directory demo-prep

## THE 60 SECONDS. One clip clean, then in a measured dead zone.
demo-break: results/demo/demo_cache.json
	@$(PY) demos/demo_break.py

## the active-learning loop, against a GP surrogate fitted to the real grid
demo-al:
	@$(PY) demos/demo_al.py

## THE INTERACTIVE BEAT — the interviewer listens and ranks, THEN learns the
## model scored those clips exactly equal. Offline, no API key, plays wavs
## through afplay/aplay/play/ffplay. It waits for a human, so it is
## deliberately NOT in the `demo` chain, which must run unattended.
##   demos/demo_listen.py --check     preflight
##   demos/demo_listen.py --replay    rehearse with the recorded 2026-08-05 session
demo-listen:
	@$(PY) demos/demo_listen.py

## OPTIONAL LIVE BEAT — the ONLY target here that needs network + an API key.
## Two real nova-3 calls (~$0.001) on one clip: clean, then the #1 measured
## dead zone, showing the per-word confidences as they come back.
##
## Deliberately NOT a prerequisite of `demo`. The spine is offline by design and
## must stay that way — if this were in the chain, a conference wifi outage
## would take the whole demo down, which is the exact failure the kit exists to
## prevent. It is safe to run anyway: no key, no network, a vendor error or a
## timeout all print one explanatory line, fall back to the cached results, and
## exit 0. Rehearse it with `demos/demo_live.py --offline`; preflight it with
## `demos/demo_live.py --check` before you decide whether to schedule the beat.
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

## the whole scripted path, in the rehearsed order
demo: test-core demo-break demo-al dashboard
	@echo ""
	@echo "  Done. The dashboard is open; the 3-minute path is in dashboard/DEMO.md."
	@echo ""


# --------------------------------------------------------------------------
# hygiene
# --------------------------------------------------------------------------

## is every artifact the demo needs actually on disk, right now, offline?
demo-check:
	@$(PY) demos/demo_break.py --check
	@$(PY) scripts/make_demo_audio.py --check

## pin exactly what is installed, so the demo machine is reproducible
lock:
	$(PIP) freeze > requirements.lock.txt
	@echo "  wrote requirements.lock.txt ($$(wc -l < requirements.lock.txt | tr -d ' ') packages)"

clean-demo:
	rm -rf results/demo
	@echo "  removed results/demo/ — run 'make demo-prep' to rebuild it"
