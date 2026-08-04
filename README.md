# Deadzone — Streaming-ASR Acoustic-Robustness Evaluation

**Deadzone is a controlled evaluation framework that maps where streaming ASR
models fail *silently* under acoustic degradation — the "dead zones" where the
model is confidently wrong.** It is domain-neutral: any streaming ASR model, any
acoustic environment.

Aggregate WER reports a single number and hides *where* and *how* a model breaks.
Deadzone is a controlled instrument for **causal attribution of acoustic
failure**: it uses *real* ingredients (measured room impulse responses, recorded
noise, real codecs) but controls only the *assembly*, so it can hold every factor
fixed and turn one acoustic knob at a time. That **counterfactual isolation** is
the thing field recordings can't give you — in the wild, mic, placement, noise,
and codec all move at once. Fidelity to any one deployment is explicitly *not* the
goal; isolation is.

## The four findings

1. **Silent-failure map (headline).** Plot the model's word-level confidence
   against actual WER for every condition. The deliverable is the **danger zone**:
   the acoustic conditions where the model stays *confident while wrong* — the case
   a downstream system can't defend against, because it never asks for a repeat.

2. **Failure fingerprints (mechanism).** Don't count errors — *classify* them from
   aligned edits (substitutions / deletions / insertions). Each condition gets an
   error *signature* (reverb → deletions, babble → substitutions, codec → killed
   proper-nouns/entities), and each signature implies a concrete fix.

3. **Interaction hunt.** A Plackett–Burman screen finds which factors move WER at
   all, then Saltelli/Sobol sensitivity analysis (with bootstrap CIs) surfaces the
   **counterintuitive cells** — non-monotonic "sweet spots" and factor pairs that
   compound (e.g. reverb × SNR) — the surprises worth writing up.

4. **Active-learning surrogate.** A lightweight Gaussian-process surrogate predicts
   WER from condition parameters and *actively chooses* which condition to evaluate
   next (boundary-seeking **straddle** acquisition), mapping the failure boundary in
   **far fewer oracle calls than random or grid sampling** — validated against a
   random baseline on a planted synthetic boundary.

## Honest boundaries

Deadzone isolates *acoustic* factors. It deliberately brackets out **behavioral**
ones — accent, disfluency, speaking rate, and especially the **Lombard effect**
(people change how they *produce* speech in noise; no room simulator captures a
behavior). It also runs synthetic RIRs alongside measured ones and reports the
**sim-vs-real gap** as its own result. Naming where the instrument stops being
trustworthy is a first-class deliverable, not a footnote.

## Method

Real ingredients, controlled assembly. Every degraded clip is composed in one
fixed, physically-motivated order — **room reverb → additive noise at a calibrated
SNR → mic frequency response → transmission codec** — reusing three
correctness-critical "trap" functions (active-speech-gated SNR, onset-aligned
level-preserved RIR convolution, and typed-edit WER scoring) that produce
clean-looking garbage if implemented subtly wrong. Every stage ships with an
offline synthetic test; the statistics and active-learning layers are validated
against *planted* structure before ever touching real audio.

## Tech stack

- **DSP / audio:** `numpy`, `scipy`, `soundfile`, `librosa`, `pyroomacoustics`;
  real codec round-trips (AMR-NB / low-rate Opus) via `ffmpeg`.
- **Experiment design & sensitivity:** `SALib` (Plackett–Burman, Saltelli/Sobol
  S1/ST/S2, bootstrap CIs).
- **Active learning:** `scikit-learn` Gaussian-process regression (ARD Matérn
  kernel) with straddle acquisition; `scipy.stats.qmc` for LHS/Sobol sampling.
- **ASR adapters:** Deepgram (streaming commercial model exposing per-word
  confidence — the headline signal) and OpenAI Whisper (open, API-independent
  baseline), behind one shared contract so their WERs are directly comparable.

## Repo

| File | What |
|---|---|
| `audio_pipeline.py` | Trap functions (SNR / RIR / typed-edit WER) + transcription adapters |
| `conditions.py` | Named, reproducible degradation composer + disk-backed asset library |
| `design.py` | Plackett–Burman screen → Sobol sensitivity → counterintuitive-cell detector |
| `active_learning.py` | GP surrogate + straddle acquisition + active-vs-random comparison |
| `recording_manifest.csv` | ~40 domain-neutral utterances (entity/number stress cases) to record |
| `test_*.py` | Fully offline test suites — every stage validated on synthetic data |
| `SPEC.md` / `CLAUDE.md` | Full project brief and working conventions |

## Running the tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
for t in test_pipeline test_adapters test_design test_conditions test_active_learning; do
  python3 $t.py
done
```

All suites run offline (no API key, no GPU, no network) on synthetic signals.
Live smoke tests for the Deepgram adapter (`smoke_deepgram.py`) and the ffmpeg
codec path (`smoke_codec.py`) are separate and require a key / `ffmpeg`.
