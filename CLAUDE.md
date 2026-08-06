# CLAUDE.md — Deadzone: Streaming-ASR Silent-Failure Map

Persistent project context. Read at the start of every session. Keep this lean;
full detail lives in `SPEC.md`.

## What this is
Deadzone is a general, domain-neutral testbed that maps where a **streaming-capable
ASR model** fails **silently** (confident but wrong) under controlled acoustic
degradation — the "dead zones" — what *kind* of failure each condition causes,
and a cheap early-warning signal. Any streaming ASR, any acoustic domain. The grid
runs **three arms** (10,560 rows): Deepgram Nova-3, Whisper-base, ElevenLabs
Scribe. Note it is all **batch**: Deepgram via the pre-recorded endpoint
(`listen.v1.media.transcribe_file`), Whisper locally, Scribe via its batch
endpoint — no arm uses `listen.live`. Full brief: @SPEC.md

## Build / test / run commands
- Run unit tests: `python3 tests/test_pipeline.py`  (offline, synthetic — must stay green)
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
- Always pull **per-word confidence** from the streaming ASR model (the Deepgram
  adapter) — it is the headline signal.
- **`elevenlabs-scribe` is rank-only: never put it in a cross-model WER
  comparison.** Its orthography is non-deterministic across identical calls, so
  its offset is variance, not bias, and cannot be normalized away. Enforced in
  code (`model_compare` raises); measured within-arm only.

## Working rules
- Build in DAG order (see SPEC §9). Do not start an analysis layer until the
  master results table exists and passed the validation gate (SPEC §11 step 4).
- Every new pipeline stage ships with a test. Nothing downstream is trusted until
  the clean-clip≈0-WER / garbage-spikes gate passes.
- Prefer editing existing modules over new ones; match the layout in SPEC §13.

## Open decisions (ask before assuming)
- Third analysis leg is **resolved**: the active-learning surrogate (`active_learning.py`,
  D3) alongside the interaction hunt (`design.py`). See SPEC §5.

## Positioning (for the write-up)
This is a known evaluation genre (WildASR, Speech Robustness Bench, "When
Denoising Hinders"). Do **not** frame it as novel. The angle = the silent-failure
lens (confidence–accuracy gap) + typed failure fingerprints + the active-learning
surrogate + testing commercial streaming models that expose per-word confidence,
with actionable guidance. Cite prior work up front. Name the Lombard-effect
boundary explicitly (SPEC §4).
