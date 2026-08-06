# Deadzone — demo kit (SPEC A.R9)
#
# One command per demo. Nothing here is multi-step, nothing here needs an API
# key, and nothing here touches the network. Rehearse with wifi OFF.
#
#   make help          what you can run
#   make demo          the full scripted path, in order
#
.DEFAULT_GOAL := help
.PHONY: help lock test test-core demo demo-prep demo-break demo-al demo-check \
        dashboard dashboard-build clean-demo

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
	@echo "  Deadzone demo kit — every target runs OFFLINE, no API key required."
	@echo ""
	@echo "  ON STAGE (in this order)"
	@echo "    make test-core        30 s  the trap functions + the demo kit, green"
	@echo "    make demo-break       60 s  one clip, clean -> measured dead zone"
	@echo "    make demo-al          30 s  the surrogate walking onto the boundary"
	@echo "    make dashboard         —    open the self-contained HTML from file://"
	@echo "    make demo                   all four, in sequence"
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
demo-prep:
	$(PY) demos/demo_break.py --prepare

results/demo/demo_cache.json:
	@$(MAKE) --no-print-directory demo-prep

## THE 60 SECONDS. One clip clean, then in a measured dead zone.
demo-break: results/demo/demo_cache.json
	@$(PY) demos/demo_break.py

## the active-learning loop, against a GP surrogate fitted to the real grid
demo-al:
	@$(PY) demos/demo_al.py

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

## pin exactly what is installed, so the demo machine is reproducible
lock:
	$(PIP) freeze > requirements.lock.txt
	@echo "  wrote requirements.lock.txt ($$(wc -l < requirements.lock.txt | tr -d ' ') packages)"

clean-demo:
	rm -rf results/demo
	@echo "  removed results/demo/ — run 'make demo-prep' to rebuild it"
