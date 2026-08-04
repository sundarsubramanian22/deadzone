# Deadzone — Streaming-ASR Silent-Failure Map — Project Spec

> Self-contained brief. Anyone (human or agent) who reads only this file should
> understand what we're building, why, what's done, and what to do next.

---

## 0. TL;DR

Deadzone is a controlled instrument that maps **where a streaming ASR model fails
*silently*** under acoustic degradation — i.e. conditions where it stays
*confident while being wrong* — plus **what kind of failure** each condition
produces, and a **cheap early-warning signal** for it. Raw "how much does WER go
up" attribution is the baseline layer, not the point. Deliverables: a runnable
repo, a short technical write-up with an honest lit review + limitations, and an
interactive dashboard. It is domain-neutral: any streaming ASR model, any
acoustic environment. Target effort ~40 hrs (full version).

---

## 1. Why this project exists (context)

- **The problem.** Streaming-ASR robustness is under-characterized. Models are
  reported at an **aggregate WER**, but that single number hides *where* and *how*
  a model fails — which acoustic conditions break it, what kind of error each one
  produces, and whether the model *knows* it is failing. Deployments live and die
  on exactly the tail conditions the aggregate averages away.
- **The instrument.** Deadzone is a controlled rig for **causal attribution of
  acoustic failure**: hold everything fixed, turn one acoustic knob, and measure
  what moves. That is the one thing real-world recordings *can't* give you — in
  the field every factor (mic, placement, noise, codec) moves at once, so causes
  are confounded and un-isolable.
- **Why a simulation, not field data.** A controlled rig does the one thing real
  data can't: **counterfactual isolation** — vary a single factor with everything
  else held constant. Fidelity to any one deployment is explicitly *not* the goal;
  isolation is. (The sim-vs-real gap is then measured and reported honestly — §4.)
- **The bar the project sets itself.** (a) build & ship an end-to-end instrument
  independently, (b) engage honestly with the ASR-robustness literature, (c) know
  exactly where the instrument stops being trustworthy. Competence + judgment +
  honest boundaries over any claim of novelty.

## 2. Core idea / intellectual framing

Descriptive benchmarking is never interesting. Interesting work is **predictive**
(forecast the failure), **mechanistic** (know *why*), or **interventional**
(found a lever). We get all three off one testbed by asking harder questions of
the same generated data.

**The reframe that carries the project:** stop asking "how much does the model
break?" Ask **"does the model *know* it's breaking?"** For a streaming voice
agent, a *confidently wrong* model is far more dangerous than one that knows it's
struggling — confidence is what decides whether the system commits to a wrong
transcript or asks the speaker to repeat. That turns a benchmark into a study of
**silent failures** — the "dead zones" where the model is confidently wrong —
which is product-critical and under-studied.

## 3. Prior work (position honestly — do NOT claim novelty)

The controlled-degradation-testbed genre is well-trodden. Cite these up front so
the work reads as informed, not naive:

- **WildASR / "Back to Basics: Revisiting ASR in the Age of Voice Agents"** (2026)
  — controlled acoustic shifts with fixed linguistic content, image-source room
  sim (pyroomacoustics), RT60 sweeps. Closest to our method + framing.
- **Speech Robustness Bench (SRB)** — named benchmark: models under additive
  noise, reverb, time transforms, adversarial perturbations.
- **"When Denoising Hinders"** (2026) — perceptually cleaner audio is not
  necessarily more ASR-robust (distribution shift). This *is* the perception-
  mismatch hypothesis, already published.
- Speech-enhancement eval work showing ASR ranking ≠ perceptual-quality metric
  ranking; ASR-based intelligibility prediction showing low human/machine
  correlation; **Carlini & Wagner** audio adversarial examples (the extreme case).
- Far-field ASR lineage: RIR-augmentation (Ko et al. 2017), Google Home far-field
  (Kim et al. 2017), REVERB / CHiME challenges, pyroomacoustics (Scheibler 2018).

**Where the contribution actually lives (not novelty of method):** the framing
shift from *how much* to *silently* (the confidence–accuracy gap as the headline
deliverable), the mechanism layer (typed failure *fingerprints* rather than a
scalar WER), the active-learning surrogate that maps the failure boundary in far
fewer evaluations than a grid, and testing **commercial streaming models** that
expose per-word confidence — where the literature mostly uses Whisper / Conformer
/ wav2vec — with *actionable guidance* rather than mere characterization.

## 4. Scope

**In scope:** three cleanly-modelable factor families —
- **Reverb / placement** (via real measured RIRs; each RIR proxies a room+distance)
- **Noise type × SNR** (real recorded engine/road/babble)
- **Channel** (mic frequency-response filter + intercom/VoIP codec/bitrate)

**Out of scope — and stated out loud as the honest boundary:**
- **Behavioral / human factors we deliberately bracket out:** accent &
  code-switching, disfluency, speaking rate, head orientation, and especially the
  **Lombard effect** (people involuntarily change pitch/spectrum when it's loud —
  noise doesn't just mask the signal, it changes how it's *produced*). No acoustic
  simulator captures these because they're behaviors, not rooms. Naming this
  boundary is a bigger competence signal than any simulation detail. Framing:
  "here's what I isolated in sim; these human factors are exactly where real field
  recordings would earn their keep."

**Realism principle:** use *real* ingredients (measured RIRs, recorded noise),
control only the *assembly*. Every input is real; only the combination is held
fixed. Then run synthetic (pyroomacoustics) RIRs alongside the real ones and
report the **sim-vs-real gap** as its own finding (proves sim2real awareness).

## 5. Findings / analysis layers

> *Model arms:* a commercial streaming model exposing per-word confidence
> (**Nova-3** in the reference adapter) is the spine, vs open **Whisper** as the
> API-independent baseline. A domain-tuned model literal (e.g. `nova-2-drivethru`)
> is a planned third arm — not built, but a one-line model-literal swap later
> given the matched adapter shape (see §7).

1. **HEADLINE — confidence–accuracy gap (silent-failure map).** A streaming model
   that returns word-level confidence gives it up for free. Plot confidence vs
   actual WER per condition; the deliverable is the **danger zone** — conditions
   where the model stays confident while failing. "Here are the exact acoustic
   conditions where the model is confidently wrong." Near-zero added cost.
2. **MECHANISM — failure fingerprints.** Don't count errors, *classify* them from
   the aligned edits: reverb → deletions? babble → substitutions? codec → killed
   proper-nouns / entities? Each condition gets an error *signature*, and each
   signature implies a fix (proper-noun subs under babble → keyword boosting;
   deletions under reverb → dereverberation).
3. **THIRD LEG — surrogate + interaction hunt (BUILT).**
   - *Active-learning surrogate:* a lightweight GP predicts WER from condition
     parameters and *actively chooses* which conditions to evaluate next
     (straddle / boundary-seeking acquisition), mapping the failure boundary in
     far fewer oracle calls than random/grid sampling. Descriptive → predictive.
     (`active_learning.py`.)
   - *Interaction hunt:* find the non-monotonic / counterintuitive cells (mild
     denoise + reverb worse than either alone; a noise condition that *improves*
     WER by nudging toward the training distribution — the "denoising hinders"
     effect with a mechanism). One well-explained surprise > ten expected results.
     (`design.py`.)
4. **SIM-VS-REAL gap** (see §4) — synthetic vs measured RIRs, quantified.

> *Track-C design notes (from the synthetic-validation harness, `design.py`):*
> - **Pre-registered expectation:** `rt60 × snr_db` (reverb × noise) is predicted
>   to compound — a genuine 2-way interaction — *before* seeing any real data.
>   Registering it now makes its confirmation (or absence) on the real grid a
>   real result, not a post-hoc story.
> - **S2 is noisy; lead with the ST−S1 gap.** Even with a strong *planted*
>   interaction at N=1024, the second-order S2 bootstrap CI crossed zero. On the
>   real grid, run the second-order pass at a **higher N**, and treat the
>   **ST−S1 gap as the primary interaction evidence**, using S2 only to say
>   *which* pair interacts, not *how much*.

*Optional stretch (only if time):* does degradation hurt a streaming model's
**end-of-turn / turn-taking detection** more than its transcription? The WER
literature uses isolated utterances and misses turn-taking failure entirely.
Higher cost, higher reward. Flag as stretch, not spine.

## 6. Method / experimental design

- Factors × levels across the three in-scope families.
- **Screen first** with a fractional-factorial (or Plackett-Burman) design to
  find which factors even move WER — avoids the full-grid combinatorial blowup.
- **Sobol / Latin-hypercube sample** the survivors instead of a dense grid.
- Report **variance-based sensitivity (Sobol indices)** — this *is* Finding-1
  attribution, stated more rigorously than ANOVA — with **bootstrap CIs** on WER.
- Run every clip through the **commercial streaming model** (spine; confidence is
  the headline signal) **and Whisper** (open baseline — shows you benchmark, aren't
  API-dependent).

## 7. Pipeline core — ALREADY BUILT & TESTED (`audio_pipeline.py`)

Three correctness-critical "trap" functions. Each produces clean-looking GARBAGE
if subtly wrong, with no error message. Do **not** reintroduce these bugs:

- `mix_at_snr()` — SNR computed on **active-speech energy** (VAD-gated), not
  whole-file energy. Whole-file power is deflated by silence → your "10 dB" mix
  comes out wrong. Noise is tiled/random-cropped to length.
- `apply_rir()` — convolve, then **(a)** trim the RIR's leading direct-path delay
  so output stays onset-aligned to input (else WER inherits a pure alignment
  artifact), and **(b)** renormalize level over the input's active region (reverb
  tail leaks energy into silent regions and de-calibrates downstream SNR — this
  bug was caught by the test suite, not by any error).
- `classify_errors()` — word-level Levenshtein backtrace returning WER **plus the
  typed edits** (sub/del/ins) that the fingerprint analysis needs; a scalar WER
  can't tell you *what kind* of error. Normalizes both ref & hyp identically
  (casing/punctuation swing WER by points otherwise).

`transcribe_deepgram()` is the streaming-ASR API adapter (needs your key; kept out
of the tested core so DSP runs offline). It returns transcript **+
word_confidences** — the confidences are the headline; verify they come back
non-empty on day one.

Tests live in `test_pipeline.py`, run fully offline on synthetic signals:
`python3 test_pipeline.py` → all three verified.

## 8. Data assets + recording protocol

- **Speech:** record ~40–50 short domain-neutral utterances loaded with the same
  stress cases (proper nouns, personal names, addresses, alphanumeric codes,
  varied length) so it exercises entity/number failure modes — the interesting
  failure mode and where commercial models focus their entity handling. TTS is an
  acceptable fallback; skip generic read-speech corpora (no entity/number stress).
- **RIRs:** BUT ReverbDB (real measured) + pyroomacoustics (synthetic, for the gap).
- **Noise:** MUSAN + a few real traffic/engine clips (e.g. DEMAND).
- **Recording rules:** mono; 16 kHz or 48 kHz (resample-friendly); one utterance
  per file; **quiet room** — source must be CLEAN, all degradation is added
  synthetically, so captured room noise is contamination you can't remove.
  Transcribe each clip **as you record it** — batching later is where reference-
  transcript errors (the one bug tests can't catch) creep in.

## 9. Architecture — task DAG & parallelism

Rule: anything that **produces/downloads a raw asset** runs in parallel; anything
that **consumes multiple assets together** is a join point and must wait.

```
Track A  Data acquisition          ─┐  (fully parallel, START NOW, ~5h)
  A1 record utterances + transcripts│
  A2 download RIRs / noise          │
  A3 ASR key + confidence test      │
                                    │
Track B  Pipeline core            ─┤  (parallel w/ A, ~8–12h)
  B1 trap functions [DONE]          │
  B2 transcribe adapters (DG+Whisper)
  B3 condition-builder              │
                                    │
Track C  Experiment design        ─┘  (parallel w/ B — test on SYNTHETIC
  C1 fractional-factorial screen        WER data, no real audio needed, ~4–6h)
  C2 Sobol sampling
  C3 sensitivity indices + bootstrap CIs
            │
            ▼
        ⋈ JOIN 1  Smoke test  (1 clip → degrade → transcribe → score = 1 row)
            │      then VALIDATION GATE: clean≈0 WER, garbage spikes, edits sane
            ▼
        ⋈ JOIN 2  Run the grid  → master results table  (needs B + C + all A)
            │
            ▼
Track D  Analysis (all read the SAME table, parallel w/ each other, ~10–14h)
  D1 confidence-gap map (HEADLINE)
  D2 failure fingerprints
  D3 third leg (active-learning surrogate | interactions)
  D4 sim2real gap
            │
            ▼
Track E  Deliverables (~6–8h)
  E1 write-up  (lit-review + methods draftable ANYTIME; findings join after D)
  E2 dashboard (build shell on fake data early; plug real results at end)
```

Long pole: `A → ⋈1 → ⋈2 → D`. Highest-leverage move: run **A and B from hour
zero simultaneously**, slot C into any pipeline wait. Joins are the only true
bottlenecks.

## 10. Build plan + time budget (~35–50h, call it 40)

| Phase | Hrs | Notes |
|---|---|---|
| Pipeline robust across all conditions | 8–12 | biggest chunk; each new degradation = new silent-bug risk |
| Record + prep corpus | 3–5 | ground-truth transcription is the time sink |
| Experiment design (screen→Sobol→indices→CIs) | 4–6 | real applied stats |
| 4 analysis layers | 10–14 | each a mini-project |
| Sim2real validation | 3–4 | cheap to run, thoughtful to interpret |
| Deliverables (write-up + dashboard) | 6–8 | front-end always overruns |
| Inevitable gnarly bug | 2–4 | budget it |

*Ladder if time compresses:* ~15h = confidence-gap map only, done correctly.
~25h = add Whisper baseline + fingerprints (sweet spot). ~40h = full version.

## 11. First steps (in order)

1. **Own the pipeline core.** Read `audio_pipeline.py`, run `test_pipeline.py`,
   understand *why* each trap function does what it does. Swap the energy VAD for
   `silero-vad` before trusting real data.
2. **Start Track A now** (parallel, no code needed): record utterances, download
   RIRs/noise, get the ASR key and confirm word-confidences return non-empty.
3. **Smoke test (JOIN 1):** one clip → `apply_rir` → `mix_at_snr` → save wav →
   `transcribe_deepgram` → `classify_errors` → one correct
   `(condition, wer, mean_conf, edit_counts)` row.
4. **Validation gate:** clean→~0 WER, degraded→spike + sane edit types. Only then
   scale to the full grid.

## 12. Open decisions / risks

- **VERIFY day one:** the streaming ASR model returns per-word confidence on your
  plan. If empty, the headline finding is blocked — resolve before building further.
- **Risk:** ground-truth transcript errors are invisible to tests → transcribe
  carefully at record time.
- **Risk:** simulation ≠ real acoustics → this is *answered*, not hidden, by the
  sim2real gap finding + the §4 boundary section.

## 13. Suggested repo layout

```
deadzone/
  CLAUDE.md              # persistent agent context (points here)
  SPEC.md               # this file
  audio_pipeline.py     # trap functions + adapters  [DONE]
  test_pipeline.py      # offline unit tests          [DONE]
  conditions.py         # B3: named degradation configs
  design.py             # C: factorial screen + Sobol + sensitivity
  active_learning.py    # D3: active-learning surrogate + acquisition
  run_experiment.py     # ⋈2: loop → master results table (parquet/csv)
  analysis/
    confidence_gap.py   # D1 headline
    fingerprints.py     # D2 mechanism
    sim2real.py         # D4
  dashboard/            # E2
  data/                 # gitignored: rirs/, noise/, recordings/
  results/              # gitignored: master table + figures
  report/               # E1 write-up
```
