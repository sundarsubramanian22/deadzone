# Deadzone — a silent-failure map for streaming-capable ASR

Deadzone is a controlled testbed that asks a different question of a speech
recognizer than the usual one. Not *how much does it break* under acoustic
degradation, but **does it know it is breaking?** For a streaming voice agent a
confidently wrong transcript is far more dangerous than a visibly uncertain one:
confidence is what decides whether the system commits or asks you to repeat.

**This is a well-trodden genre and this repo does not claim otherwise.** Controlled
acoustic-degradation testbeds for ASR already exist and are good — WildASR /
"Back to Basics: Revisiting ASR in the Age of Voice Agents" (fixed linguistic
content, image-source room simulation, RT60 sweeps), the **Speech Robustness
Bench** (noise, reverb, time transforms, adversarial perturbation), and **"When
Denoising Hinders"** (perceptually cleaner audio is not necessarily more
ASR-robust). Far-field RIR augmentation goes back to Ko et al. 2017 and the
REVERB / CHiME challenges; the simulator here is `pyroomacoustics` (Scheibler
2018). The method is theirs.

What is different is the lens, not the machinery:

1. **The confidence–accuracy gap as the headline deliverable**, rather than WER.
2. **Typed failure fingerprints** — errors are classified from the aligned edits
   (sub / del / ins) so each condition gets a *signature* that implies a fix,
   instead of collapsing to a scalar.
3. **An active-learning surrogate** asked whether the failure boundary can be
   mapped in fewer oracle calls than a grid (boundary-seeking straddle
   acquisition over a GP). **On this surface the answer was no** — the
   `boundary_rmse` target was reached by 2 of 8 active seeds against random's 4
   of 8, so no savings are claimed and the layer is reported as a **null**. The
   synthetic control still passes, which is what makes it a method meeting a
   surface it has no purchase on rather than a broken implementation. See the
   write-up §6.5, and D.8b for the re-run in DRR coordinates that also changed
   nothing.
4. **Commercial streaming-capable models that expose per-word confidence** —
   Deepgram Nova-3 as the primary arm and ElevenLabs Scribe as a second,
   independent vendor — where the literature mostly uses Whisper / Conformer /
   wav2vec. Without per-word confidence there is no silent-failure question to
   ask. **Measured in batch:** every row went through a pre-recorded endpoint
   (Deepgram via `listen.v1.media.transcribe_file`, *not* `listen.live`), and
   the Whisper arm runs locally with full-file lookahead — so all three arms are
   batch and the comparison is not mode-confounded, but what is mapped is
   acoustic robustness rather than streaming behaviour. See the write-up's
   limitation 17.

## The headline result

Across **176 controlled acoustic conditions** × 40 recorded utterances:

> Nova-3's mean word confidence tracks its own error rate almost perfectly —
> **Spearman ρ = −0.980** (paired on the clips it actually spoke on; −0.952
> against every clip, silent ones included; n = 169 either way) — and it is
> nonetheless **overconfident in 91% of conditions (154/169)**, mean gap
> **+0.147**. **2 of 176 (1.14%)** are genuine dead zones. The worst: mean word
> confidence **0.829** at **WER 0.306**, all 40 clips, none silent
> (`rt60 0.45 s · SNR 0 dB · engine · g726 · mic rolloff 0`).

These numbers moved from an earlier pass because of a real defect, and the fix
is the more interesting result. `mean_conf` was averaged only over clips that
returned words, while `wer` was averaged over every clip, including **empty**
ones with no confidence at all — two populations subtracted as if they were
one, manufacturing dead zones that weren't real. Caught by **listening**, not a
test. Conditions now split three ways: **dead zone** — confidently wrong (2) —
**silence-driven** — the gap was clips vanishing, not confident errors (4) — and
**mute zone** — nothing emitted on any of the 40 clips (7). A mute zone is not a
dead zone — confidently wrong vs. entirely absent — and **a confidence-based
monitor cannot see one at all**: WER 1.0, 100% deletions, no confidence to be
wrong with.

The point is not that the model is blind. It is that the model is *mostly*
self-aware, which is exactly what makes the residual dangerous: any system tuned
on its average self-awareness will trust it precisely where it should not.

## Three arms, and a verdict that depends on how you score

The grid runs three recognizers — **10,560 rows** in `results/master.csv`:
Nova-3 over all 40 clips (7,040 rows), and **Whisper-base** and **ElevenLabs
Scribe** over the shared 10-clip subset (1,760 rows each; Scribe: 176 conditions
× 10 clips, **0 failures**).

**Scribe is measured within itself and ranked only — it is excluded from every
cross-model WER comparison, and that exclusion is enforced in code and pinned by
a test, not left to convention.** The reason is not that it scores badly. It is
that **its orthography is non-deterministic**: four identical calls on
byte-identical audio returned different transcripts on **5 of 6 probe clips** —
`A7X42` vs. "A seven X four two", `Q9J05` vs. "Q nine J zero five", with u33
flipping the other way — worth up to **0.727 strict WER on identical input**.
*(Provenance, since it is weaker than everything else quoted here: the
repeat-call counts come from a 6-clip × 4-call probe that is **not persisted to
any artifact**, so unlike every other figure on this page they are not pinned by
`tests/test_report_numbers.py`. The 0.727 itself is recomputed from
`results/master.csv`. The grid was run once per cell like the other arms, so it
carries this variance unquantified — write-up limitation 18.)*
Whisper's formatting offset is a *constant* (+0.090): characterize it once and
subtract it, which is exactly what `deadzone/cross_model_norm.py` does. A
per-call draw cannot be subtracted, because it is variance, not bias.

**The headline is better than any ranking: the verdict reverses with scoring.**
Scored strictly, Scribe's confidence-vs-WER shape (ρ = **−0.820**) sits midway
between Nova-3 (**−0.970**) and Whisper (**−0.590**), and Nova-3 looks clearly
better calibrated. Apply the cross-model orthography normalization and Scribe
collapses onto Nova-3 (**−0.948** vs. −0.970) while Whisper barely moves
(**−0.603**). **Which commercial model appears to know when it is wrong depends
on a scoring choice most benchmarks make silently.** The surviving ~0.02 is an
*upper bound* and may well be zero — measurement error can only attenuate a rank
correlation — so this repo does **not** claim Nova-3 is better calibrated than
Scribe.

What *is* confirmed under **both** scorings is narrower: **Nova-3's
confidence-shape edge over Whisper is real** (+0.277 [0.171, 0.406] strict,
+0.262 [0.157, 0.390] normalized — both intervals clear of zero). The tempting
next sentence — *"and so is Scribe's, therefore Whisper is the outlier rather
than Nova-3"* — **is only true under normalized scoring, and the write-up
declines it.** Measured on the 159 conditions all three arms spoke on, Scribe's
lead over Whisper is **+0.227 [0.097, 0.376] normalized — separable**, but
**+0.074 [−0.112, +0.267] strict — not separable**. So "Whisper is the outlier"
is a claim *about a scoring choice*, not about the models: it survives the
normalizer and does not survive the spine scorer, which is the same reversal
this section is about. The caveat travels too: n = 3 models and only **one**
open model, so "commercial vs. open" and "vendor-specific" remain confounded.

*(Scope, stated precisely because getting it loosely right is this project's
signature bug. The four correlations in the previous paragraph — −0.970,
−0.820, −0.948, −0.590 — share the **10-clip subset** but **not a condition
population**: each is computed over the conditions *that arm* emitted words on,
n = **164** for Nova-3, **174** for Scribe, **171** for Whisper. They are
therefore fine as per-model shapes and **not** directly rankable against each
other, which is exactly why the write-up's three-arm table (§6.7) restricts to
the **159** conditions all three spoke on, and why the separability intervals
above are the 159-condition ones. Nova-3 reads −0.970 there against the −0.980
quoted for its full 40-clip table above: different scopes, not a discrepancy.)*

## Run the demo in three commands

These three — and the whole `make demo` path — run **offline**, with wifi off and
**no API key**. All numbers come from the already-measured grid; the audio is
regenerated by the same tested DSP that produced it.

```bash
make demo-break     # 60 s  one clip, clean -> a measured dead zone, out loud
make demo-al        # 30 s  the surrogate walking onto the failure boundary
make dashboard      #  —    the self-contained dashboard, opened from file://
```

`make demo` runs all three in the rehearsed order, preceded by the core test
suite. `make help` lists every target; `make demo-check` is the preflight to run
*before* you turn wifi off. First run only: `make demo-prep` bakes the cached
artifacts (it is invoked automatically if they are missing).

One **optional** extra beat is the sole exception to the offline guarantee:

```bash
make demo-live      # 20 s  the same beat, transcribed LIVE  [NEEDS wifi + key]
```

It needs network and `DEEPGRAM_API_KEY`, is deliberately **not** part of `make
demo`, and is safe to attempt anyway — no key, no network, a vendor error or a
timeout each print one explanatory line, fall back to the cached results, and
exit 0.

The 3-minute spoken path through the dashboard is in
[`dashboard/DEMO.md`](dashboard/DEMO.md); a sealed-prediction blind listening
exercise (reverb vs. babble) lives in `results/audio/demo/` (`DEMO_SCRIPT.md`).

## Honest boundaries

Deadzone isolates **acoustic** factors and deliberately brackets out
**behavioral** ones — accent, disfluency, speaking rate, and above all the
**Lombard effect**: in noise people change how they *produce* speech, and no room
simulator reproduces a behavior. It is one speaker and 40 utterances, so nothing
here generalizes across speakers. Synthetic RIRs are run alongside the measured
ones and the sim-vs-real gap is reported as its own result. Deletions carry no
confidence value at all, which is a substantive hole in the headline signal
rather than a footnote. The full limitations section is in the write-up.

## Method, in one paragraph

Real ingredients, controlled assembly. Every degraded clip is composed in one
fixed, physically-motivated order — **measured room impulse response → recorded
noise at a calibrated SNR → mic frequency response → transmission codec** —
through three correctness-critical "trap" functions that produce clean-looking
garbage if subtly wrong: SNR computed on **active-speech energy only**, RIR
convolution that trims the direct-path delay **and** renormalizes over the input's
active region, and WER scoring that returns the **typed edits** alongside the
scalar. Each ships with an offline test; the statistics and active-learning
layers were validated against *planted* synthetic structure before touching real
audio.

## Repo map

| Path | What |
|---|---|
| `deadzone/` | The importable library. Nothing in here spends money or writes an artifact. |
| `deadzone/audio_pipeline.py` | The three trap functions + the ASR adapters behind one contract |
| `deadzone/conditions.py` | Named, reproducible degradation composer + disk-backed asset library |
| `deadzone/design.py` | Plackett–Burman screen → Saltelli/Sobol sensitivity → counterintuitive cells |
| `deadzone/active_learning.py` | GP surrogate, straddle acquisition, active-vs-random comparison |
| `deadzone/analysis/` | The finding layers: confidence gap, fingerprints, sensitivity, sim2real, AL savings. Run as `./.venv/bin/python -m deadzone.analysis.<layer>`. |
| `scripts/` | Entry points that spend money or produce artifacts — `run_experiment.py` (the grid runner → `results/master.csv`), the asset fetchers, the smoke tests |
| `dashboard/` | The self-contained 8-panel HTML dashboard + its build script |
| `demos/` | `demo_break.py`, `demo_al.py` — the demo kit (offline, key-free) |
| `tests/` | Fully offline suites; every stage validated on synthetic signals |

Everything is run **from the repo root** — relative `data/`, `results/` and
`results_sim/` paths are resolved against it, and the Makefile, the demos, the
dashboard and the test suites all assume that CWD.

## Where to read further

- **[`SPEC.md`](SPEC.md)** — the full self-contained project brief: framing, prior
  work, experimental design, the task DAG, and the execution checklist.
- **[`report/writeup.md`](report/writeup.md)** — the write-up, with the results and
  the limitations section.
- **`results/MANIFEST.json`** — the experiment freeze: git SHA, exact model
  literals, ffmpeg version and codec decision, asset SHA-256s, realized call
  counts and cost. Commercial model literals are updated server-side, so a re-run
  months from now is **not the same experiment**; this file is what says what was
  actually measured.

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
make test          # every offline suite
```

`requirements.lock.txt` pins the exact demo-machine environment. Live smoke tests
(`scripts/smoke_deepgram.py`, `scripts/smoke_codec.py`) need a key / `ffmpeg` and are separate
from everything above. The Deepgram key is read from `DEEPGRAM_API_KEY` and is
never committed.
