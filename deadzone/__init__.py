"""deadzone — the importable library: pipeline core, degradation composer,
experiment design, and the analysis layers.

Nothing in here has a CLI entry point of its own except the `analysis`
submodules, which are run as `python -m deadzone.analysis.<layer>`. Anything
that spends money or writes an artifact lives in `scripts/`; the demo kit lives
in `demos/`. The dependency runs one way only: `scripts/`, `demos/` and
`dashboard/` import from here, never the reverse.

Submodules are NOT imported eagerly — several pull in optional heavy deps
(librosa, pyroomacoustics, the vendor SDKs), and `import deadzone` must stay
cheap and dependency-light for the offline test suites.
"""
