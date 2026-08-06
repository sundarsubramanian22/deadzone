"""pytest bootstrap — put the repo root on sys.path, no install step required.

`python -m pytest tests/` must work straight from a clean checkout, with no
`pip install -e .` and no packaging metadata. pytest prepends the *test file's*
directory (tests/) to sys.path, not the repo root, so `import deadzone` would
fail without this.

Direct invocation (`python tests/test_pipeline.py`) does not load conftest at
all, so each test module additionally carries its own two-line repo-root
bootstrap. Both paths are gated in the README.

NOTE ON THE CANONICAL RUNNER. These suites are script-style: each module has an
`if __name__ == "__main__"` driver that calls its checks in a fixed order and
prints a pass banner. `make test` runs them that way, and that is the project's
canonical gate. pytest support is a convenience on top of it, which is what the
deselect list below exists to reconcile.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The suites read `data/...`, `results/...`, `recording_manifest.csv` and
# `task_specs.json` by paths relative to the CWD, exactly as the Makefile, the
# demos and the dashboard do. Pin the CWD to the repo root so `pytest tests/`
# behaves identically to `python tests/test_x.py` run from the root.
os.chdir(REPO_ROOT)


# ---------------------------------------------------------------------------
# Collection: four `test_*` names are not pytest tests.
#
# They are NOT renamed, because renaming would change a public API and this is a
# pure layout refactor. They are deselected instead, and every one of them still
# runs under the canonical runner (`make test` / `python tests/test_d3.py`), so
# nothing is silently dropped from the gate.
#
#   test_set_from_master   — PRODUCTION CODE in deadzone/analysis/al_savings.py.
#                            It builds the held-out *test set* the AL arms are
#                            scored against (ML terminology). pytest only sees
#                            it because tests/test_d3.py imports it by name.
#
#   the other three        — chained helpers in tests/test_d3.py. They take
#                            values threaded from earlier checks (`rows`, `prop`,
#                            `split`), which pytest misreads as fixture requests.
#                            The module's __main__ driver passes them explicitly.
# ---------------------------------------------------------------------------
_NOT_PYTEST_TESTS = {
    "test_set_from_master",
    "test_confirmation_needs_the_real_oracle",
    "test_single_seed_is_refused_or_flagged",
    "test_multi_seed_band_and_headline",
}


def pytest_collection_modifyitems(config, items):
    keep, drop = [], []
    for item in items:
        (drop if item.name in _NOT_PYTEST_TESTS else keep).append(item)
    if drop:
        items[:] = keep
        config.hook.pytest_deselected(items=drop)
