# CLAUDE.md — Drive-Thru ASR Silent-Failure Map

Persistent project context. Read at the start of every session. Keep this lean;
full detail lives in `SPEC.md`.

## What this is
A controlled acoustic testbed that maps where Deepgram's Nova-3 fails **silently**
(confident but wrong) in a drive-thru regime, what *kind* of failure each
condition causes, and a cheap early-warning signal. Full brief: @SPEC.md

## Build / test / run commands
- Run unit tests: `python3 test_pipeline.py`  (offline, synthetic — must stay green)
- Python 3.11+, deps: `numpy scipy soundfile librosa pyroomacoustics deepgram-sdk`
  (add `silero-vad`, `pesq`, `pystoi` as those phases land)
- Deepgram key via env var `DEEPGRAM_API_KEY` — never hardcode.

## Non-negotiable conventions (the trap functions — do NOT regress these)
- SNR is computed on **active-speech energy only**, never whole-file power.
- After RIR convolution: **trim the direct-path delay** (keep onset-aligned) AND
  **renormalize level over the input's active region** (reverb tail de-calibrates
  SNR otherwise). Both are already correct in `audio_pipeline.py`; keep them.
- Scoring returns WER **and typed edits** (sub/del/ins) — the fingerprint layer
  needs the edit types, not just a scalar. Normalize ref & hyp identically.
- Always pull **per-word confidence** from Deepgram — it is the headline signal.

## Working rules
- Build in DAG order (see SPEC §9). Do not start an analysis layer until the
  master results table exists and passed the validation gate (SPEC §11 step 4).
- Every new pipeline stage ships with a test. Nothing downstream is trusted until
  the clean-clip≈0-WER / garbage-spikes gate passes.
- Prefer editing existing modules over new ones; match the layout in SPEC §13.

## Open decisions (ask before assuming)
- Third analysis leg: **predictor** vs **interaction hunt** — unset. See SPEC §12.

## Positioning (for the write-up)
This is a known evaluation genre (WildASR, Speech Robustness Bench, "When
Denoising Hinders"). Do **not** frame it as novel. Our angle = their commercial
models + drive-thru domain + actionable guidance. Cite prior work up front.
Name the Lombard-effect boundary explicitly (SPEC §4).
