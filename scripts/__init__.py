"""scripts/ — entry points that SPEND MONEY or PRODUCE ARTIFACTS.

Everything here is meant to be run from the repo root, e.g.

    ./.venv/bin/python scripts/run_experiment.py --dry-run

Relative data paths (`data/...`, `results/...`) are resolved against the
current working directory, so the repo root is the assumed CWD — the same
invariant the Makefile, the demos and the dashboard rely on.

This file exists so the modules stay importable (`from scripts.run_experiment
import load_manifest`), which the test suites and `analysis/decoupling.py` do.
"""
