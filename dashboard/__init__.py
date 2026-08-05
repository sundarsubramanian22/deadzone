"""dashboard/ — E2 deliverable: a single self-contained static HTML file.

`build.py` is the only entry point; `make_synthetic.py` feeds it when no real
grid exists yet. Nothing in here is imported by the analysis layers — the
dependency runs one way only, so a dashboard change can never break a finding.
"""
