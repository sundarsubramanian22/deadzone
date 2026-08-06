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

### Advanced layers (built + synthetic-validated; real path wired, awaiting audio)

These consume the master results table / real recordings but their machinery is
built and validated against planted synthetic structure now, so they drop in the
moment audio lands.

- **L1 — Multi-model comparison** (`model_compare.py`): a one-line-per-model
  registry (Deepgram / Whisper / Vosk arms) + cross-model dead-zone comparison
  that normalizes confidence *within* each model (scales aren't comparable across
  families) and asks whether the confidence-vs-WER *shape* differs. Synthetic-
  validated: recovers a planted weaker-model region. *Needs real audio:* per-model
  grid results to name the real divergent regions.
- **L2 — Learned confidence calibration** (`calibration.py`): temperature-scaling
  baseline + a feature-conditioned calibrator (confidence logit + acoustic params)
  that corrects condition-dependent overconfidence; reports ECE before/after +
  reliability-diagram data. Synthetic-validated: recovers a planted overconfidence
  that grows with reverb (ECE cut to ~1/5). *Needs real audio:* word-level
  correctness + confidence per condition to fit the real calibrator.
- **L3 — Paralinguistic-parallel features** (`paralinguistic.py`): a dependency-
  light (numpy/scipy, CPU) extractor of energy/pitch/spectral features run
  alongside transcription, + an analysis that asks whether paralinguistic features
  and lexical accuracy degrade at the same rate or DECOUPLE. Synthetic-validated:
  recovers planted pitch + a feature degrading on a known noise schedule, and
  detects coupling/decoupling. *Needs real audio:* the recorded clips (per
  degradation level) to measure the real decoupling.
- **L4 — Voice-agent eval scaffold** (`agent_eval.py`): task-completion metrics
  (slot/entity accuracy, entity-error rate, critical-slot failure) that diverge
  from WER, + a turn-taking-failure analyzer (false endpoints / missed turns /
  barge-in) over timestamped events. Structure only — NO live STT->LLM->TTS agent,
  no LLM/TTS calls. Synthetic-validated: entity-error and WER disagree as designed
  and planted turn failures are flagged. *Needs real audio + a live agent:* real
  transcripts/timelines to score; makes the agent reframe drop-in.

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

> **⚠️ SUPERSEDED 2026-08-05 — the repo was reorganized into a conventional
> Python layout.** The flat original is kept below the fold for archaeology. The
> structure that follows is the one on disk; CLAUDE.md's "match the layout in
> SPEC §13" now means *this* one.

**The split is by ROLE, and one boundary is load-bearing:** everything under
`deadzone/` is importable library code that is **free to re-run**; everything
under `scripts/` **spends money or writes artifacts**. That is why `run_d3a.py`
and `run_experiment.py` live in `scripts/` and never under `analysis/` — the
budget ceiling (`--max-calls`) is a checked invariant, not a comment, and an
API-spending path must not hide behind a module whose contract is that it is
cheap to call in a loop.

```
deadzone/                    # repo root
  CLAUDE.md  SPEC.md  README.md  Makefile
  requirements.txt  requirements.lock.txt
  recording_manifest.csv     # 40 utterances + normalized ground truth
  task_specs.json            # entity slots per clip (D2 + agent layer)

  deadzone/                  # the importable package — FREE to re-run
    audio_pipeline.py        # the 3 trap functions + the model adapters
    conditions.py            # Condition, DiskAssetLibrary, apply_condition
    design.py                # factor space, PB screen, Sobol, counterintuitive cells
    active_learning.py       # GP surrogate + straddle acquisition
    calibration.py           # L2: temperature + feature-conditioned calibrators
    model_compare.py         # L1 primitives (within-model percentiles, dead zones)
    paralinguistic.py        # L3: feature extraction + drift
    agent_eval.py            # L4 scaffold (synthetic-validated; R7 out of scope)
    cross_model_norm.py      # cross-model orthography, L1 ONLY — never the trap fn
    analysis/                # all read the SAME master table; all free to re-run
      confidence_gap.py      #   D1 headline — the silent-failure map
      fingerprints.py        #   D2 mechanism — typed edit signatures
      sensitivity.py         #   exact functional-ANOVA Sobol + clip bootstrap
      interactions.py        #   D3a pre-registration verdict + dips
      sim2real.py            #   D4 — clip-matched, see Appendix C.8
      al_savings.py          #   D3b active-vs-random curves
      model_arms.py          #   L1 cross-model comparison
      calibration_report.py  #   L2 reporting
      decoupling.py          #   L3 reporting
      layers.py

  scripts/                   # entry points that SPEND MONEY or write artifacts
    run_experiment.py        #   ⋈2: the grid → master.{parquet,csv} + cache.jsonl
    run_d3a.py               #   D3a: two-stage counterintuitive hunt (--max-calls)
    probe_elevenlabs.py      #   the SPEC §12 day-one gate for a new vendor arm
    make_manifest.py  make_sim_rirs.py  make_audio_sets.py
    check_recordings.py  fetch_assets.py
    smoke_deepgram.py  smoke_codec.py  smoke_join1.py

  demos/                     # offline, no API key, no network
    demo_break.py  demo_al.py
  tests/                     # 21 suites, all offline; conftest.py bootstraps sys.path
  dashboard/                 # E2 — build.py → self-contained deadzone.html
  report/                    # E1 write-up
  data/                      # gitignored: recordings/ rirs/ rirs_sim/ noise/
  results/                   # gitignored: master table, artifacts, figures
  results_sim/               # gitignored: the sim-RIR arm's SEPARATE cache (B.2 §7)
```

Invocation follows the layout: `./.venv/bin/python -m deadzone.analysis.<mod>`
for analysis, `./.venv/bin/python scripts/<x>.py` for runners, and every path in
the code resolves **relative to the repo root as CWD**.

<details><summary>original flat layout (pre-2026-08-05, for archaeology)</summary>

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
</details>

---

# Appendix: Remaining Work — Granular Execution Checklist

> **Purpose.** Written 2026-08-04 so that a future session (mine or an agent's)
> can execute the rest of this project **without re-deriving a single decision**.
> Everything below is ordered by hard dependency: do it top to bottom. Each item
> has concrete sub-steps, a **DoD** (definition of done — the observable fact that
> proves it's finished), and the **gotcha** that silently ruins it.
>
> Nothing in §0–§13 above is superseded by this appendix; this is the execution
> layer under it.

## A.0 — State of the world (as of this appendix)

**Built, tested, merged to `main` (all suites green, all offline/synthetic):**

| Module | What's done |
|---|---|
| `audio_pipeline.py` | 3 trap functions (`mix_at_snr`, `apply_rir`, `classify_errors`) + 3 adapters (`transcribe_deepgram` live-verified, `transcribe_whisper`, `transcribe_vosk`) behind one contract |
| `conditions.py` | `Condition`, `DiskAssetLibrary` (scans `data/rirs`, `data/noise`), `apply_mic_rolloff`, `apply_codec` (real ffmpeg), `apply_condition` in fixed `COMPOSITION_ORDER` |
| `design.py` | `DEFAULT_FACTOR_SPACE` (5 factors), Plackett–Burman screen, Saltelli/Sobol S1/ST/S2 + bootstrap CIs, `find_counterintuitive_cells` |
| `active_learning.py` | `GPSurrogate`, straddle `acquire`, `active_learn`, `random_baseline`, `learning_curve`, `compare_arms`, `make_pipeline_oracle` |
| `model_compare.py` (L1) | model registry, `within_model_conf_percentile`, `dead_zone_flags`, `find_divergence_regions`, `compare_models` |
| `calibration.py` (L2) | `TemperatureScaler`, `FeatureCalibrator`, ECE, reliability curves, `calibration_report` |
| `paralinguistic.py` (L3) | `extract_features`, `feature_drift`, `compare_degradation_rates` |
| `agent_eval.py` (L4) | `TaskSpec`, `evaluate_task`, `entity_error_rate`, `analyze_turns`, `summarize_turn_findings` |
| `recording_manifest.csv` | 40 utterances `u01`–`u40` with normalized ground truth |
| `test_*.py` | 8 offline suites; every layer validated against *planted* synthetic structure |

**The single blocker: there is no real audio.** `data/recordings/` holds only two
`say`-generated placeholders; `data/rirs/` and `data/noise/` do not exist. Every
remaining item is downstream of that.

**Files that do not exist yet and must be written** (§13 layout):
`check_recordings.py`, `fetch_assets.py`, `smoke_join1.py`, `run_experiment.py`,
`task_specs.json`, `analysis/confidence_gap.py`, `analysis/fingerprints.py`,
`analysis/sim2real.py`, `analysis/al_savings.py`, `dashboard/`, `report/`,
`agent/` (final milestone). Directories `analysis/ results/ dashboard/ report/`
do not exist.

**Rough remaining budget:** R1–R2 ≈ 5 h · R3 ≈ 2 h · R4 ≈ 4 h (plus unattended
compute) · R5 ≈ 10 h · R6 ≈ 6 h · R7 ≈ 12–16 h · R8 ≈ 6 h · R9 ≈ 2 h.
R7 (live agent) is optional-by-design; see its off-ramp.

---

## A.R0 — Reconstitute the environment (15 min)

- [ ] **R0.1 — Rebuild and verify the venv.**
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt` (Python 3.11+; the repo was built on 3.11.9)
  - `pip install soundfile librosa` if not already resolved by the requirements pin.
  - **DoD:** `python3 -c "import numpy, scipy, soundfile, librosa, pyroomacoustics, SALib, sklearn, deepgram"` exits 0.
- [ ] **R0.2 — Green the whole suite before touching anything.**
  - `for t in test_pipeline test_adapters test_design test_conditions test_active_learning test_model_compare test_calibration test_paralinguistic test_agent_eval; do python3 $t.py || echo "FAIL $t"; done`
  - **DoD:** all nine print their pass banner, zero `FAIL` lines.
  - **Gotcha:** if a suite fails after a dependency upgrade, fix it *now*. Every
    later phase assumes the trap functions are trustworthy; a regression here
    poisons everything silently and no downstream number will look wrong.
- [ ] **R0.3 — Confirm the API key still works and confidences still come back.**
  - `export DEEPGRAM_API_KEY=...` (it lives in `.env`, gitignored — never inline it)
  - `python3 smoke_deepgram.py data/recordings/sample.wav`
  - **DoD:** prints a transcript, `len(word_confidences) > 0`, non-NaN `mean_conf`,
    and the final `OK:` line.
  - **Gotcha:** this is SPEC §12's day-one verification. If word confidences come
    back empty, **stop** — the headline finding is blocked and nothing else matters.
- [ ] **R0.4 — Create the missing directories.**
  - `mkdir -p analysis results/figures results/audio dashboard report data/rirs data/noise/{babble,engine,road} data/recordings data/rirs_sim`
  - `data/` and `results/` are already gitignored; `analysis/`, `dashboard/`,
    `report/` are code and *do* get committed.
  - **DoD:** `ls -d analysis results dashboard report data/rirs data/noise/babble` succeeds.

---

## A.R1 — Record the speech corpus (the long pole; ~3 h)

This is the irreplaceable asset. RIRs and noise can be re-downloaded; a bad
recording session costs a full re-record, and a bad *transcript* costs the
credibility of every number in the writeup.

- [ ] **R1.1 — Decide and commit to the clip count: record all 40 (`u01`–`u40`).**
  - **Why 40 and not 10, not 200.** WER precision per condition is governed by the
    total *reference word count*, not the clip count. The manifest averages ~8.5
    words/utterance, so 40 clips ≈ **340 reference words per condition**. Treating
    word errors as roughly binomial, the standard error on a per-condition WER of
    0.20 is ≈ √(0.2·0.8/340) ≈ **2.2 percentage points** — tight enough to
    distinguish adjacent grid cells, which is the entire point of a controlled rig.
    Drop to 15 clips (~130 words) and SE ≈ 3.5 pts, at which point neighbouring
    cells overlap and the sensitivity analysis is measuring noise. Go to 100 clips
    and SE only improves to ~1.4 pts (√N returns) while *every* API call count in
    the project multiplies by 2.5×.
  - 40 also buys **entity coverage**: the manifest deliberately spans personal
    names (Nguyen, Okafor, Yamamoto, Kowalski), spelled alphanumeric codes
    (`u06`, `u17`, `u33`, `u39`), phone/PO/card digit strings, addresses, currency,
    dosages and times — the failure modes the fingerprint layer classifies. Cutting
    clips cuts *categories*, not just samples.
  - **Sub-designations to fix now and never revisit:**
    - **Smoke clip:** `u02` (digits + a name, short, unambiguous) — used for R3.
    - **Active-learning clip set (10):** `u02, u05, u06, u11, u17, u22, u24, u33, u36, u39`
      — chosen to span names / digits / spelled codes / addresses. The AL oracle
      averages WER over these; a single-clip oracle is far too noisy for a GP.
    - **Sobol clip set:** same 10. (Sobol needs many samples; full-40 is reserved
      for the screen and the main grid.)
  - **DoD:** the three subsets are written into `run_experiment.py` as named
    constants, not chosen ad hoc at run time.

- [ ] **R1.2 — Pick the device and lock the settings.**
  - **Use:** a USB condenser or a decent dynamic mic (or, acceptably, the built-in
    mic in a *treated* quiet room) into software that writes **uncompressed PCM WAV**.
    - GUI: **Audacity** (free) — set Project Rate 48000, channels **1 (mono)**,
      export as *WAV 16-bit PCM* (or 24-bit).
    - CLI (preferred, fully reproducible): list devices with
      `ffmpeg -f avfoundation -list_devices true -i ""`, then record with
      `ffmpeg -f avfoundation -i ":<idx>" -ac 1 -ar 48000 -sample_fmt s16 -t 8 data/recordings/u01.wav`
  - **Exact settings:** `WAV / PCM · 48 kHz (or 16 kHz) · 16-bit (or 24-bit) · mono · no processing`.
    48 kHz is preferred: `DiskAssetLibrary` resamples everything to 16 kHz anyway,
    and downsampling from 48 is clean while upsampling from 8 invents nothing.
  - **Do NOT use, and why each one specifically breaks this project:**
    - **iPhone Voice Memos / any `.m4a` / `.mp3` / `.ogg` output** — AAC is a lossy
      codec. `codec` is one of our five *independent variables*. Recording through
      AAC applies an uncontrolled codec to every clip before the experiment starts,
      so the `codec` factor's effect is measured on top of a hidden constant and the
      `none` level isn't actually a null condition.
    - **AirPods / any Bluetooth headset** — HFP/mSBC bandlimits to 8–16 kHz *and*
      applies vendor noise suppression. Same confound as above, plus it partially
      applies the `mic_rolloff` factor for you.
    - **Zoom / Meet / Teams / Discord / any conferencing capture** — these apply
      denoising, AGC, echo cancellation and often dereverberation. The project's
      intellectual neighbour is literally "When Denoising Hinders"; recording
      through a denoiser means the source is already enhancement-processed and the
      whole premise collapses.
    - **macOS "Voice Isolation" / "Wide Spectrum" mic modes** — turn OFF (Control
      Centre → Mic Mode → **Standard**) . Also uncheck any AGC in your recorder.
    - **Stereo** — the composer collapses to mono anyway; capturing mono keeps the
      collapse from being an uncontrolled averaging step.
    - **Any plugin chain: compressor, limiter, normalizer, noise gate, EQ.** A gate
      deletes the room-tone padding the VAD needs (R1.5). A compressor changes the
      active-speech energy that `mix_at_snr` calibrates SNR against, so your
      requested 10 dB is not the delivered 10 dB.
  - **TTS fallback** (SPEC §8 permits it): if recording is impossible, render the
    manifest with a TTS engine **directly to WAV** (`say -o u01.aiff` → convert, or
    a TTS API with WAV output). Acceptable, but write it into the limitations
    section: TTS has no natural prosodic variation and no real capture chain, which
    makes the corpus *easier* than reality and makes the Lombard boundary argument
    (§4) even more load-bearing. Never use a TTS that only emits MP3.
  - **DoD:** one test file recorded, and `soundfile.info()` on it reports
    `PCM_16` (or `PCM_24`), `channels=1`, `samplerate=48000` (or 16000).

- [ ] **R1.3 — Set up the room, and understand why the pipeline cares.**
  - **Requirements:** small, soft, quiet. Carpet/curtains/soft furnishings, or a
    clothes-filled closet. HVAC, fridge, fans, air purifier **off**. Windows shut.
    Phone on airplane mode and off the desk. Laptop idle so the fan never spins up.
    Avoid kitchens, bathrooms, and any hard-surfaced room.
  - **Why it matters — two distinct mechanisms, both silent:**
    1. **Reverb multiplies.** `apply_condition` convolves a *measured* RIR onto the
       clip. Any reverb already in the recording is convolved with it, so a request
       for `rt60=0.3` delivers "0.3 s ⊗ your room". The `rt60` axis stops being the
       thing you set, and every rt60-attributed result is contaminated by a constant
       you never recorded.
    2. **SNR mis-calibrates.** `mix_at_snr` calibrates *added* noise against
       active-speech energy. Noise already present in the clip is counted as
       **signal**, so the delivered SNR is worse than requested, and worse by an
       amount that varies with how loud the fridge was that hour.
    - Both show up in the Sobol decomposition as unexplained residual variance —
      i.e. they attack the headline attribution directly, and no test can see them.
  - **Measure the room, don't trust it.** Record 10 s of silence to
    `data/recordings/_roomtone.wav`, then check with the repo's own primitives:
    ```python
    import soundfile as sf; from audio_pipeline import rms
    x, fs = sf.read("data/recordings/_roomtone.wav"); print(20*np.log10(rms(x)+1e-12), "dBFS")
    ```
  - **DoD:** room-tone RMS ≤ **−60 dBFS**, and speech-peak minus room-tone RMS ≥ **40 dB**.
    If you can't hit −60, you can still proceed but must record the number and state
    it in the limitations section.
  - **Gotcha:** keep the room *identical* across all 40 takes — same time of day,
    same doors closed. A corpus recorded half in the morning and half with the AC on
    has a hidden two-level factor in it.

- [ ] **R1.4 — Mic distance, level, and clipping.**
  - **Distance:** 15–20 cm (6–8 in), ~15–30° off-axis, or on-axis behind a pop
    filter. **Identical for all 40 clips** — distance is a factor the RIRs *simulate*
    (each RIR proxies a room+distance, §4). Varying it at record time injects a
    second, uncontrolled copy of a factor we're trying to isolate.
  - **Level:** peaks between **−12 and −6 dBFS**. Never approach 0.
    - **Why:** clipping is a hard nonlinearity that sprays broadband harmonics.
      Those harmonics then get convolved by the RIR and squeezed by the codec, and
      you will read the resulting WER rise as *codec sensitivity*. It is the most
      plausible-looking wrong result available to you.
  - **Post-hoc gain:** peak-normalizing all clips to a common target afterwards is
    **optional and safe** (`apply_rir` renormalizes over the active region and
    `mix_at_snr` calibrates on active-speech energy, so absolute level doesn't
    propagate) — but do it **uniformly to all 40 or to none**. Per-clip loudness
    normalization with different targets is a hidden factor.
  - **DoD:** `check_recordings.py` (R1.7) reports zero clips with `max|x| ≥ 0.99`
    and zero clips with more than 10 samples above 0.95, and all 40 peak within
    −14…−4 dBFS.

- [ ] **R1.5 — Silence padding: ~0.5 s of *room tone* at both ends.**
  - **Target:** 0.3–0.7 s of recorded room tone before the first phoneme and after
    the last. Start the recorder, breathe, then speak; stop a beat after finishing.
  - **Why the VAD and the traps need it — four separate reasons:**
    1. `active_speech_mask()` is a relative-energy VAD. A clip that is ~100 % speech
       gives it no non-speech reference, and the "active" energy estimate it hands
       to `mix_at_snr` degenerates — your SNR calibration loses its anchor.
    2. `mix_at_snr` *defines* SNR against active-speech energy only (the
       non-negotiable convention in CLAUDE.md). No silent region, no valid split.
    3. `apply_rir` trims the direct-path delay to stay onset-aligned. With no
       lead-in, the trim has no headroom and can shave the first phoneme — which
       becomes a deletion in *every* reverb condition and reads as a beautiful,
       entirely fake "reverb causes deletions" fingerprint.
    4. Deepgram's endpointing wants lead-in/lead-out; a hard cut at the first
       phoneme is a known cause of a dropped first word.
  - **Pad with recorded room tone, never digital zeros.** Exact zeros make the VAD
    threshold trivially easy (unlike any real signal) and are not physically
    realizable after convolution. If you must pad after the fact, tile a slice of
    `_roomtone.wav` — do not `np.pad` with zeros.
  - **DoD:** `check_recordings.py` confirms every clip has ≥ 0.25 s of
    below-threshold audio at both head and tail per `active_speech_mask()`.

- [ ] **R1.6 — What to say: the two rules that decide whether the project is valid.**
  - **Source of truth:** `recording_manifest.csv` — columns `id, say_this, ground_truth`,
    rows `u01`–`u40`. `say_this` is the display form (capitals, punctuation);
    `ground_truth` is the normalized reference (lowercase, no punctuation, numbers
    written as *spoken words*).
  - **Rule 1 — read it EXACTLY as written.** Including the spoken-digit form: `u02`
    is "four zero five, nine one two, seven seven" — **not** "four oh five", **not**
    "four hundred five".
    - **Why:** `ground_truth` is stored in spoken form and `normalize_text()`
      deliberately does *not* map digits ↔ words (see the normalization-parity note
      at `audio_pipeline.py:262`). Say "oh" where the reference says "zero" and you
      have manufactured a substitution that fires in **every single grid cell** — a
      constant WER offset that (a) fails the clean≈0 gate at R3 and (b) is
      mathematically indistinguishable from a real acoustic effect once it's in the
      table.
  - **Rule 2 — if you improvise, fix the transcript in the same breath.** If you
    stumble, self-correct, swap a word, or drop an article and you *keep the take*,
    you must immediately edit that row's `ground_truth` to what you actually said,
    normalized the same way (lowercase, punctuation stripped, digits spelled out).
    Preferably: just re-record. Either way, resolve it **before starting the next
    clip** — never batch this.
  - **Why ground-truth accuracy is the one unfixable thing.** Every test in this
    repo is offline and synthetic. `test_pipeline.py` proves `classify_errors`
    aligns two strings correctly; **nothing anywhere proves the reference string is
    what you said.** A wrong reference is permanently, silently wrong for every
    downstream layer:
    - it inflates WER uniformly → and since the model is *confident* about words it
      got right, a wrong reference looks exactly like a **dead zone**. You would
      publish a fabricated headline finding.
    - it invents substitutions that D2 will classify into a fingerprint and you will
      write a mechanistic story about.
    - it adds variance the Sobol decomposition attributes to whichever factor
      happens to correlate.
    - it teaches the L2 calibrator that confident-correct words are wrong, which
      inverts the calibration curve.
    - There is no test that catches it and no post-hoc detection except relistening
      to all 40 clips. This is SPEC §12's named risk.
  - **Delivery:** natural conversational pace, don't over-enunciate, don't perform.
    One utterance per file. Roughly consistent loudness across takes.
  - **DoD:** every row of `recording_manifest.csv` matches, word for word, what is
    on the corresponding wav.

- [ ] **R1.7 — File naming and placement.**
  - **Convention:** `data/recordings/<id>.wav`, where `<id>` is the manifest `id`
    verbatim — `u01.wav` … `u40.wav`. Lowercase, zero-padded two digits, `.wav`.
    Nothing else in the filename: no dates, no takes, no initials. The loader keys
    off the manifest `id` column, so the filename *is* the join key.
  - Rejected takes go to `data/recordings/rejects/` (or get deleted) — never leave
    `u07_take2.wav` next to `u07.wav`.
  - `data/` is gitignored (`.gitignore` covers `data/` and `*.wav`). The clips are
    **never committed**; only the manifest is versioned. **Back the recordings up
    off-machine** — they are the only irreplaceable artifact in this repo.
  - The two existing placeholders `data/recordings/sample.wav` and
    `public_sample.wav` are `say`-generated TTS from the adapter smoke test. They
    match no manifest id so an id-driven loop will never pick them up; keep them for
    `smoke_deepgram.py` or delete them, but don't confuse them for takes.
  - **Write `check_recordings.py`** — the gate script for this whole phase. It reads
    the manifest and asserts, per row:
    - the file `data/recordings/{id}.wav` exists (and flag any wav in the dir with
      no manifest row);
    - `channels == 1`, subtype is PCM, samplerate ∈ {16000, 48000}, and **all 40
      agree** on the rate;
    - `max|x| < 0.99` and ≤ 10 samples above 0.95 (clipping);
    - peak in −14…−4 dBFS (level consistency); flag any clip > 6 dB from the median;
    - ≥ 0.25 s of non-active lead-in and lead-out via `active_speech_mask()`;
    - duration in 1.5–8 s, and flag any clip > 2× the median duration;
    - `ground_truth` is already normalization-stable: `" ".join(normalize_text(gt)) == gt`.
  - **DoD:** `python3 check_recordings.py` prints `40/40 OK` with zero warnings.

- [ ] **R1.8 — Record two clips first, verify the whole chain, then record 38.**
  - Record `u01` and `u02`. Run `check_recordings.py --only u01,u02`. Run
    `python3 smoke_deepgram.py data/recordings/u02.wav`. Eyeball the transcript
    against `ground_truth`.
  - **Why:** discovering a wrong gain setting, a mic-mode processing flag, or a
    stereo file *after* 40 takes costs the entire session. This checkpoint costs
    10 minutes.
  - **DoD:** both clips pass the checker and `u02` transcribes to its ground truth
    with WER 0.0.

- [ ] **R1.9 — Record the remaining 38, transcribing/verifying as you go.**
  - After each take: listen back once, confirm it matches `say_this`, and only then
    move on. This is the "transcribe as you record" rule from §8, and the
    verification-as-you-go loop is what makes Rule 2 (R1.6) actually happen.
  - **DoD:** 40 files present, `check_recordings.py` → `40/40 OK`.

- [ ] **R1.10 — The clean-transcription adjudication pass (do not skip).**
  - Run all 40 clean clips through Deepgram and diff against `ground_truth`.
    Write it as a small loop reusing `transcribe_deepgram` + `classify_errors`; dump
    to `results/clean_baseline.csv` with `id, wer, edits`.
  - For **every** clip with WER > 0: listen to it and adjudicate by ear.
    - Transcript is wrong → fix `recording_manifest.csv`, commit the fix.
    - You misspoke → re-record.
    - The model genuinely missed a clean word → keep it, and **record the id and the
      word** — this is your measured clean-condition floor and it belongs in the
      writeup as a number, not a surprise.
  - **DoD:** `results/clean_baseline.csv` exists; mean clean WER ≤ **0.02**; every
    non-zero row has a one-line adjudication note.
  - **Gotcha:** this is the *only* place you let the ASR check your work, and it's
    safe only because **you adjudicate by ear** — never "fix" the manifest to match
    what the model heard. That would be training the ground truth on the system
    under test.
  - **Cost:** 40 calls ≈ 3 minutes of audio. Negligible.

---

## A.R2 — Fetch the real acoustic assets (~2 h, mostly download wait)

- [ ] **R2.1 — Measured RIRs → `data/rirs/`.**
  - **Primary:** **BUT ReverbDB** (Brno University of Technology Speech@FIT Reverb
    Database) — real measured RIRs across 8 rooms with multiple mic positions;
    per §8 this is the named source. Requires accepting terms on the project page.
  - **Easiest fallback (recommend starting here):** the **MIT Acoustical
    Reverberation Scene Statistics Survey** — 271 real measured IRs, single small
    download, no registration.
  - **Also acceptable:** OpenSLR **SLR28** (`rirs_noises.zip`, Ko et al. 2017 — the
    RIR-augmentation lineage cited in §3; bundles real RIRs from RWCP / REVERB / AIR
    plus simulated ones). **Note it is ~14 GB** — pull it only if the smaller sets
    don't give RT60 coverage.
  - **Curate to 10–16 files.** Copy only the ones you'll use into `data/rirs/`
    (subdirectories are fine — `_audio_files` uses `rglob`).
  - **The RT60-coverage gotcha (this one will quietly flatten a headline factor).**
    `AssetLibrary.resolve()` picks the RIR whose **measured** RT60 (Schroeder T20,
    `conditions.py:226`) is *closest* to the requested `rt60`. If your set clusters
    at 0.4–0.6 s, then requests for 0.2 and 1.0 both snap to the nearest available
    and the `rt60` axis silently becomes a coarse step function with dead ranges.
    Sobol will then report that reverb barely matters — a wrong result with no error
    message.
    - **Mitigation:** after populating, print the measured-RT60 histogram:
      ```python
      from conditions import DiskAssetLibrary
      lib = DiskAssetLibrary(root="data", target_fs=16000)
      print(sorted(round(a.rt60, 3) for a in lib.rirs))
      ```
    - Require coverage of **[0.2, 1.0] s with no gap larger than ~0.15 s**, and at
      least 2 RIRs near each end. Add files until that holds.
  - **DoD:** `DiskAssetLibrary` constructs without `MissingAssetError`, reports
    ≥ 10 usable RIRs, zero `RT60 unestimable` warnings on the curated set, and the
    RT60 histogram satisfies the coverage rule above. Record the list in
    `results/asset_manifest.json`.

- [ ] **R2.2 — Noise → `data/noise/{babble,engine,road}/`.**
  - **Directory name IS the `noise_type`** (`_scan_noise`, `conditions.py:303`), and
    it must match the categorical levels in `DEFAULT_FACTOR_SPACE` **exactly**:
    `babble`, `engine`, `road`. A typo (`traffic/`) produces a library with a type
    nothing ever requests, and `resolve()` raises at grid time.
  - **Recommended single source: DEMAND** (Zenodo; 16-channel real environmental
    recordings, 5 min each) — one download covers all three types:
    - `babble` ← `SCAFE` (cafeteria) + `PSTATION` (train station) + `PCAFETER`
    - `engine` ← `TCAR` (car interior) + `TBUS` (bus)
    - `road` ← `STRAFFIC` (street traffic) + `SPSQUARE` (public square)
    - Use channel 1 of each; convert to mono wav.
  - **MUSAN** (OpenSLR SLR17) is the §8-named alternative — good `noise/free-sound`
    and `noise/sound-bible` material, and its `speech/` corpus can be summed
    (6–8 random speakers, level-matched) to synthesize babble if DEMAND's cafeteria
    isn't babbly enough. **~11 GB** — curate, don't bulk-copy.
  - **Requirements per type:** ≥ 3 clips, each ≥ **60 s**.
    - **Why 60 s:** `mix_at_snr` tiles/random-crops the noise with a seed derived
      from the condition name. With a 5 s clip every condition sees nearly the same
      excerpt, the seeded crop stops decorrelating conditions, and the noise
      realization becomes a hidden constant confounded with everything else.
  - **The memory gotcha.** `DiskAssetLibrary.__init__` reads, resamples to 16 kHz
    and **caches every file in RAM** (`self._cache`). Pointing it at an
    uncurated MUSAN/SLR28 tree will exhaust memory and take many minutes to
    construct. Copy in only the ~9–12 noise clips and ~12 RIRs you actually want.
  - **DoD:** `lib = DiskAssetLibrary("data")` constructs in < 60 s, and
    `collections.Counter(a.noise_type for a in lib.noise)` prints exactly
    `{'babble': ≥3, 'engine': ≥3, 'road': ≥3}` with no fourth key.

- [ ] **R2.3 — Write `fetch_assets.py` (idempotent, reproducible).**
  - Downloads (curl/urllib) → unzips to a staging dir → copies the curated subset
    into `data/rirs/` and `data/noise/<type>/` → converts to mono wav → prints a
    summary table (file, source dataset, duration, measured RT60 where applicable)
    → writes `results/asset_manifest.json` with **SHA-256 per file**.
  - Must be safely re-runnable: skip anything already present with a matching hash.
  - **DoD:** deleting `data/rirs` and `data/noise` and running
    `python3 fetch_assets.py` restores a library that passes R2.1 and R2.2.
  - **Gotcha:** check free disk before starting; SLR28 + MUSAN unzipped is ~30 GB
    of staging even if you keep 200 MB.

- [x] **R2.4 — ffmpeg + the codec path.** ✅ **RESOLVED: option (b), G.726.**
  Stock ffmpeg ships AMR-NB decode-only, so the `codec` factor level was changed
  from `amr` to `g726` (ITU-T ADPCM at 16 kbit/s, present in every stock ffmpeg,
  2 bits/sample). Verified round-tripping through real ffmpeg via `smoke_codec.py`:
  RMS delta 0.073 vs 0.247 for opus-lowrate, exact length preserved. Option (a)
  (homebrew-ffmpeg tap with `--with-opencore-amr`) was rejected: it makes the grid
  depend on a source build that may not exist on the next machine, for a codec
  family difference the write-up can simply state. **Must be named in the methods
  section** — which narrowband codec was used is a methods fact.
  <details><summary>original decision text (kept for the reasoning)</summary>
  - Install: macOS `brew install ffmpeg` · Debian/Ubuntu `sudo apt install ffmpeg`.
  - Verify the two encoders the factor space needs:
    ```
    ffmpeg -hide_banner -encoders | grep -E "amr|opus"
    ```
  - **CONFIRMED ISSUE on this machine (2026-08-04):** stock Homebrew ffmpeg ships
    AMR-NB as **decode-only** (`amrnb`/`amr_nb_at` appear under `-decoders`; no
    `amr_nb` under `-encoders`). `libopus` **is** present. So `apply_codec(x, fs, "amr")`
    will raise `CodecUnavailableError` — correctly and loudly, per design — and any
    grid cell with `codec="amr"` dies. Pick one fix **before** R3:
    - **(a) Get a real AMR encoder** (most faithful to §4's "intercom codec"):
      `brew tap homebrew-ffmpeg/ffmpeg && brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-opencore-amr`
      then change `_CODEC_SPECS["amr"]["args"]` (`conditions.py:369`) from
      `-c:a amr_nb` to `-c:a libopencore_amrnb`, and re-verify.
    - **(b) Swap the level for a codec you already have.** G.726 32 kbit/s ADPCM is
      a genuine telephony codec and *is* available (`adpcm_g726`). Change the
      `codec` level in `DEFAULT_FACTOR_SPACE` (`design.py:130`) from `"amr"` to
      `"g726"` and add
      `"g726": {"rate": 8000, "args": ["-c:a", "adpcm_g726", "-b:a", "16k", "-f", "wav"]}`
      to `_CODEC_SPECS`. `Condition.__post_init__` validates against
      `design.py`'s levels, so the one edit propagates. Cheaper, and equally
      defensible in the writeup as long as you *say* which narrowband codec you used.
    - Whichever you pick, **record the choice and the reason in the writeup's
      methods section** — "AMR-NB where available, else G.726" is a fine sentence;
      a silent substitution is not.
  - Run the real round-trip smoke test: `python3 smoke_codec.py`
  - **DoD:** `smoke_codec.py` completes with no `CodecUnavailableError` for **every**
    level of the `codec` factor; each non-`none` codec returns audio the same length
    as its input, materially different from the input (non-trivial RMS difference),
    and audibly bandlimited when you listen to it.
  - **Gotcha:** don't "fix" this by deleting the codec factor. Channel/codec is one
    of the three in-scope factor families (§4) and the codec→entity-destruction
    fingerprint is one of D2's expected results.
  </details>

  *Outcome note:* on the real grid the codec split was worth keeping — `g726`
  produces **substitutions** (entity destruction) while `opus-lowrate` produces
  **deletions**. Two different mechanisms from one factor, which a single codec
  level would have hidden.

- [x] **R2.5 — Synthetic RIRs for the sim2real leg → `data/rirs_sim/`.** ✅ **DONE**
  (`make_sim_rirs.py`). 16 synthetic RIRs generated at 16 kHz, paired 1:1 with the
  measured set: **max |delta| 0.0191 s** against the ±0.05 s tolerance, verified by
  independent re-measurement. **The trap was the common case, not a corner case:**
  pairing on the Sabine target instead of the measured RT60 would have breached
  **8 of 16 pairs** (max 0.162 s), and the bias is not one-signed — the hall
  realizes 1.241 s for a 1.00 s request (+24%) and 0.168 s for a 0.25 s request
  (−33%), so no constant correction fixes it.
  - Generate with pyroomacoustics now (it costs nothing and R5.6 needs it):
    for each **measured** RT60 in your real set, build a `pra.ShoeBox` with
    `pra.inverse_sabine(rt60, room_dims)` → `room.compute_rir()` → save 16 kHz wav.
    Use 3–4 room geometries × source/mic distances to mirror the real set's spread.
  - **Match on *measured* RT60, not requested.** Run `_rt60_schroeder` on each
    generated RIR; pyroomacoustics' realized RT60 diverges from the Sabine target.
    Save a `results/rir_pairs.json` mapping each real RIR to its closest-realized
    synthetic partner, with both measured values.
  - **DoD:** `DiskAssetLibrary(root="data", rir_subdir="rirs_sim")` constructs, has
    the same count as the real set, and every pair in `rir_pairs.json` matches
    within **±0.05 s measured RT60**.

---

## A.R3 — JOIN 1: the one-clip smoke test and the validation gate (~2 h)

**Nothing downstream is trusted until this passes.** (CLAUDE.md working rules;
SPEC §11 step 4.)

- [ ] **R3.1 — Write `smoke_join1.py`.**
  - Steps, exactly:
    1. `sf.read("data/recordings/u02.wav")`; if 48 kHz, `librosa.resample` to 16000
       (the composer and the library must agree on one rate — `apply_condition`
       raises `MissingAssetError` on a rate mismatch, by design).
    2. `assets = DiskAssetLibrary(root="data", target_fs=16000)`.
    3. `ref = ` the `ground_truth` for `u02` read from the manifest (never typed inline).
    4. Run the three gate conditions below; for each: `apply_condition` →
       `sf.write` to `results/audio/<condition.name>.wav` → `transcribe_deepgram` →
       `classify_errors(ref, result["transcript"])`.
    5. Emit one row per condition to `results/smoke_join1.csv` with the full schema
       you'll use for the master table (R4.2) — this is also how you validate the schema.
  - **DoD:** the script runs end to end and writes 4 rows (raw + 3 conditions).

- [ ] **R3.2 — Gate A: true clean ≈ 0 WER.**
  - Transcribe the **raw** wav, bypassing `apply_condition` entirely.
  - **Important structural note:** there is no "clean" `Condition` — `apply_condition`
    *always* applies an RIR and *always* mixes noise. The only true null is the raw
    file. Build the control this way and say so in the methods section.
  - **DoD:** WER = 0.00 (accept ≤ 0.05 only with a written adjudication from R1.10),
    `word_confidences` non-empty, `mean_conf` high (expect > 0.9).

- [ ] **R3.3 — Gate B: near-clean condition stays low.**
  - `Condition(rt60=0.2, snr_db=25.0, noise_type="babble", codec="none", mic_rolloff=0.0)`
  - **DoD:** WER < **0.15**. This isolates the *composer* from the model: if the raw
    clip is 0.00 and the benign condition spikes, the bug is in the composition
    chain, not in the ASR.

- [ ] **R3.4 — Gate C: garbage spikes, and confidence does *not* fully follow.**
  - `Condition(rt60=1.0, snr_db=0.0, noise_type="babble", codec="amr"|"g726", mic_rolloff=1.0)`
  - **DoD:** WER > **0.5**, edit counts are non-degenerate (not 100 % insertions —
    see gotcha), and `word_confidences` still non-empty.
  - **Note for the writeup:** if `mean_conf` stays high while WER > 0.5, you have
    *already observed the headline effect on a single clip.* Screenshot it — that's
    your demo hook, and it's the first evidence the project's premise is real.

- [ ] **R3.5 — Listen to the degraded audio. Manually. Non-negotiable.**
  - Play `results/audio/*.wav` from R3.1. Check by ear:
    - **onset alignment** — the degraded clip starts where the original starts (if
      it lags, `apply_rir`'s direct-path trim is off and every WER carries a pure
      alignment artifact);
    - **the reverb sounds like a room**, not like a delay or a metallic comb;
    - **the noise sits at roughly the SNR you asked for** — 25 dB should be barely
      audible, 0 dB should nearly bury the speech;
    - **the codec sounds like a phone line** (bandlimited, gritty).
  - **Why:** your ears are the only test for "did the composer produce something
    physically plausible". The unit tests prove the maths; only listening proves the
    *result*. This is precisely how the `apply_rir` renormalization bug was the kind
    of thing that produced clean-looking garbage.
  - **DoD:** you have personally listened to all three degraded clips and can
    describe each one.

- [ ] **R3.6 — Numeric cross-check of the SNR calibration.**
  - Reconstruct one condition with the noise captured separately and assert
    `measured_snr_db(clean_after_rir, added_noise, fs)` is within **±1.0 dB** of the
    requested `snr_db`.
  - **DoD:** assertion passes at snr_db ∈ {0, 10, 25}.

- [ ] **R3.7 — Run gates A–C on 3 different clips (`u02`, `u17`, `u36`).**
  - **DoD:** all three gates pass on all three clips.
  - **Rule: do not start R4 until this holds.** A gate that passes on one clip and
    fails on another is a clip problem (padding, level) and it will look like an
    acoustic effect once it's averaged into the grid.

- [ ] **R3.8 — Failure playbook (what to do when a gate fails).**
  - **Clean WER > 0** → it's the transcript, not the model, ~90 % of the time.
    Relisten; fix the manifest. If the token is odd (`obrien`, `wifi`), check what
    `normalize_text()` does to both sides before concluding anything.
  - **`word_confidences` empty** → check the model literal, confirm the adapter is
    still sending `smart_format/punctuate/numerals = False`, check the plan/tier.
    **Blocked headline** — SPEC §12 says resolve this before building further.
  - **Degraded WER doesn't spike** → suspect a silent no-op. Print
    `assets.resolve(cond)` and the chosen RIR's measured `rt60` (the closest-match
    rule may have snapped 1.0 → 0.45; see R2.1). Confirm the codec actually ran.
    Compare degraded vs clean RMS and spectrum.
  - **Degraded audio silent / NaN** → codec length mismatch, or `apply_rir`
    renormalizing against an empty active region (a clip whose padding is digital
    silence — R1.5).
  - **WER floor at benign conditions** → onset misalignment; re-check `apply_rir`'s
    direct-path trim against `test_pipeline.py`.
  - **Gate C is ~100 % insertions** → under heavy babble the model is transcribing
    the *background speakers*. That's a real finding, not a bug (and it's the D2
    detail at R5.3) — but confirm the reference-side alignment is sane first.

---

## A.R4 — JOIN 2: run the real grid and the real active-learning loop

- [ ] **R4.1 — Fix the evaluation budget before writing the runner.**
  - **Cost model.** Clips average ~4 s = 0.067 min. Deepgram pre-recorded Nova-3 was
    ~$0.0043/min at time of writing (**re-check current pricing** — vendor rates
    move). That's ≈ **$0.0003 per clip-call**; 10,000 calls ≈ **$3**. Whisper-base
    runs locally: $0, but ~1–2× realtime on CPU.
  - **The binding constraint is wall-clock and concurrency, not money.** ~1–2 s per
    round trip serial. Use a thread pool of **8–16 workers** and respect the
    account's concurrency limit; 10,000 calls then lands in ~15–25 min instead of
    ~4 h.
  - **Recommended allocation (write these into the runner as constants):**
    | Pass | Design | Clips | Calls |
    |---|---|---|---|
    | PB screen | `screen_factors`, 5 factors → 8 runs | 40 | 320 |
    | Main grid | ~60 named conditions (rt60 × snr × 3 noise, codec/rolloff crossed at a reduced set) | 40 | 2,400 |
    | Sobol S1/ST | Saltelli N=128, D=5 → 1,536 samples | 10 | 15,360 |
    | Sobol S2 (higher N per §5) | N=256 on surviving factors | 10 | ~20,000 |
    | Active learn + random baseline | 80 + 80 evals | 10 | 1,600 |
    | Whisper arm | main grid re-run | 40 | 2,400 (local) |
    | **Total (Deepgram)** | | | **≈ 40 k calls ≈ 45 h of audio ≈ $10–12** |
  - **DoD:** a written budget in `results/BUDGET.md` with the numbers you actually
    intend, checked against current vendor pricing.
  - **Gotcha:** the Sobol passes dominate. If you need to cut, cut the Sobol clip
    set to 5 or N to 64 — never cut the main grid, which every analysis layer reads.

- [ ] **R4.2 — Write `run_experiment.py` (§13's missing file).**
  - **Inputs:** manifest CSV, recordings dir, `DiskAssetLibrary`, a list of
    `Condition`s (or a design generator), a list of model names.
  - **Core loop:** for each (clip, condition, model): `apply_condition` → write to a
    temp wav → `transcribe_fn(path)` → `classify_errors(ref, transcript)` → one row.
  - **Master table schema** (fix it now; every analysis layer and the dashboard
    depend on it):
    ```
    clip_id, condition_name, rt60, snr_db, noise_type, codec, mic_rolloff,
    model, transcript, wer, n_ref, n_sub, n_del, n_ins, n_match,
    mean_conf, utterance_conf, word_confidences (json), edits (json),
    rir_key, rir_rt60_measured, noise_key, failed (bool), error, run_id, ts
    ```
    Note `rir_rt60_measured` — the *delivered* reverb, which is what you should
    actually regress against (the requested `rt60` is only a request; see R2.1).
  - **Output:** `results/master.parquet` + a `results/master.csv` mirror for
    grep-ability and for the dashboard.
  - **Append-only JSONL cache** keyed by `(clip_id, condition_name, model)` →
    `results/cache.jsonl`. Check the cache before every call; append after every
    call. **This is not optional** — a crash 80 % into a 40 k-call run with no cache
    costs the whole run and the whole budget.
  - **Never drop failures silently.** Use `is_failed(result)`; write the row with
    `failed=True` and the error string. Count them at the end; if > 2 % of rows
    failed, stop and investigate before analysing.
  - **Determinism:** `apply_condition` is seeded from `condition.name`, so degraded
    audio is reproducible — cache the **transcript**, not the wav (regeneration is
    cheap; disk is not).
  - **Also expose `make_grid_wer_fn(clips, assets, transcribe_fn)`** returning
    `wer_fn(sample: dict) -> float` = mean WER over the clip set. That's the exact
    callable shape `design.screen_factors` and `design.sobol_indices` want, so the
    design layer plugs straight in with no adapter.
  - **DoD:** `python3 run_experiment.py --dry-run` prints the full call plan and
    cost estimate without calling anything; a 20-row real run completes, caches, and
    a second run of the same 20 rows makes **zero** API calls.

- [ ] **R4.3 — Run the Plackett–Burman screen on real audio.**
  - `screen_factors(DEFAULT_FACTOR_SPACE, make_grid_wer_fn(all_40, ...))`
  - **DoD:** `results/screen.json` with per-factor effect sizes and the survivor list.
  - **Gotcha:** PB is Resolution-III — two-factor interactions are aliased onto main
    effects (`SCREEN_CAVEAT`, `design.py:139`). **Do not drop a factor on the screen
    alone.** Drop conservatively, respect `min_survivors`, and quote `SCREEN_CAVEAT`
    verbatim in the writeup. A factor with a weak main effect and a strong
    interaction looks dead here; Stage-2 Sobol is what resolves that.

- [ ] **R4.4 — Run the main grid → `results/master.parquet`.**
  - ~60 conditions × 40 clips × {nova-3, whisper-base}.
  - Run it unattended (background/`nohup`) and monitor the cache line count.
  - **DoD:** `master.parquet` exists with ≈ 4,800 rows, `failed` rate < 2 %, and a
    sanity pass: clean-ish conditions cluster low-WER, harsh conditions cluster
    high-WER, no condition has *identical* WER across all 40 clips (that would mean
    the composer no-op'd).

- [ ] **R4.5 — Run the Sobol passes.**
  - `sobol_indices(space, wer_fn, N=128)` on the 10-clip set for S1/ST; a second
    pass at **N=256** for S2, per SPEC §5's explicit instruction.
  - **DoD:** `results/sobol.json` + `format_sobol_tables(res)` output saved to
    `results/sobol_tables.txt`, with bootstrap CIs on every index.

- [ ] **R4.6 — Freeze the experiment.**
  - Write `results/MANIFEST.json`: git SHA, date, exact Deepgram **model literal**,
    Whisper model size, `pip freeze`, ffmpeg version + codec choice (R2.4), asset
    SHA-256s, clip count, total calls, total cost.
  - `git tag grid-v1` at the SHA that produced it.
  - **Why:** commercial model literals are updated server-side. A re-run in three
    months is **not the same experiment**, and without this you cannot say what you
    measured. This paragraph also becomes the reproducibility appendix (R8).
  - **DoD:** `results/MANIFEST.json` exists and the tag is pushed.

- [ ] **R4.7 — Run active learning against the real oracle.**
  - **Small piece of code to write first:** `make_pipeline_oracle` (`active_learning.py:378`)
    takes a **single** clip. A single-clip WER is far too noisy an oracle — the GP
    will happily model the noise. Write `make_multiclip_oracle(clips, refs, assets,
    transcribe_fn)` mirroring it but returning the **mean WER over the 10-clip AL
    set**. Same signature otherwise, so `active_learn` is unchanged.
  - `traj = active_learn(oracle, DEFAULT_FACTOR_SPACE, strategy="boundary", n_seed=15, n_total=80)`
  - `rand = random_baseline(oracle, DEFAULT_FACTOR_SPACE, n_total=80)` — **same
    budget, same oracle**, or the comparison means nothing.
  - **Build the AL test set from the already-computed grid table, not from fresh
    oracle calls.** `make_test_set(oracle, space, n=512)` would cost 5,120 extra
    calls; the master table already contains real (params → WER) pairs. Reusing it is
    free *and* is the more honest evaluation (held-out real measurements).
  - **DoD:** `results/al_trajectory.json` + `results/al_curve.json` (from
    `learning_curve`), and `evals_to_target` gives a concrete number of evaluations
    for each arm at a fixed `boundary_rmse` target.
  - **Gotcha:** seed variance in AL-vs-random is large. **One seed is not evidence.**
    Run ≥ 3 seeds. Do the multi-seed replicates against a **GP surrogate fitted to
    the master table** (instant, free), and confirm one seed end-to-end against the
    true API oracle. State exactly that protocol in the writeup — it's honest and
    it's cheap.

---

## A.R5 — Generate the real findings, one layer at a time

All of these read the **same** `results/master.parquet` (SPEC §9, Track D) and are
independent of each other — do them in any order, or in parallel.

- [ ] **R5.1 — D1 HEADLINE: the confidence–accuracy gap / silent-failure map.**
  - **Write `analysis/confidence_gap.py`.** Reuse `model_compare.within_model_conf_percentile`
    and `dead_zone_flags` for the single-model case rather than reimplementing.
  - **Real output:**
    - a scatter of per-condition `mean_conf` vs `wer` (one point per condition,
      averaged over the 40 clips), with the **dead-zone quadrant** (high conf,
      high WER) shaded;
    - a **ranked table of named dead-zone conditions** with their exact factor
      values, `mean_conf`, `wer`, and n;
    - the *gap metric* per condition (`mean_conf − (1 − wer)`, or the percentile
      form) and its distribution across factor space;
    - the correlation between confidence and accuracy overall — **including where
      confidence tracks WER well**, which you report too (honesty beats narrative);
    - `results/dead_zones.csv`.
  - **DoD:** you can write, from real data, a sentence of this exact form —
    *"At rt60 = 0.8 s, SNR = 5 dB, babble, AMR, Nova-3 returns mean word confidence
    0.91 while WER is 0.47 (n = 40 clips, 340 ref words)."* If you cannot fill in
    the blanks with real numbers, this layer is not done.
  - **Gotcha:** confidence is only comparable **within** a model. Never put Nova-3
    and Whisper confidences on the same axis (that's what `within_model_conf_percentile`
    exists for). Also: exclude or separately account for `failed=True` rows — a
    failure sentinel is not a low-confidence prediction.

- [ ] **R5.2 — Annotate entity slots (shared prerequisite for D2 and the agent layer).**
  - Write `task_specs.json`: for each manifest id, the `TaskSpec` slots and which
    are critical. Example:
    ```json
    {"u02": {"slots": {"name": "maria",
                       "phone": "four zero five nine one two seven seven"},
             "critical": ["phone"]},
     "u06": {"slots": {"code": "a seven x four two"}, "critical": ["code"]}}
    ```
  - Values must be in the same normalized spoken form as `ground_truth`
    (`evaluate_task` scores through the same `normalize_text`).
  - **DoD:** all 40 ids present; a loader test asserts every slot value is a
    contiguous subsequence of that clip's `ground_truth` (catching typos).
  - **Why here:** D2's entity fingerprint and R7's agent task-accuracy both consume
    this. Author it once.

- [ ] **R5.3 — D2 MECHANISM: failure fingerprints.**
  - **Write `analysis/fingerprints.py`.**
  - **Real output:**
    - per-condition edit-type composition (sub / del / ins as a fraction of `n_ref`),
      as a stacked-bar chart faceted by factor;
    - a **signature per factor family**, stated as a claim with numbers:
      *reverb → deletions*, *babble → substitutions + insertions*,
      *codec → entity/proper-noun substitutions*;
    - **entity-specific error rate** per condition via `agent_eval.entity_error_rate`
      against `task_specs.json` — and the gap between entity error rate and overall
      WER (the entity layer should degrade *faster*; if it doesn't, that's the
      finding);
    - a **substituted-word inventory**: which reference words get destroyed most,
      grouped into {proper noun, digit, spelled letter, function word};
    - **each signature paired with its implied fix** (proper-noun subs under babble
      → keyword boosting; deletions under reverb → dereverberation; entity loss
      under codec → constrained/entity-aware decoding). §5.2 promises actionable
      guidance; this is where it's delivered.
  - **DoD:** a table with one row per factor family: *signature · dominant edit type
    · effect size · implied fix*, all populated from real numbers.
  - **Gotcha:** insertions inflate under babble because the model transcribes
    **background speakers**. That's a different mechanism from acoustic confusion,
    and merging them would make the fingerprint wrong. Separate them by checking
    whether inserted tokens plausibly come from the babble source (or simply report
    "insertions under babble are competing-speech capture, not confusion" and show
    an example transcript). One concrete example transcript is worth a paragraph.

- [ ] **R5.4 — D3a: the interaction hunt, and the pre-registration verdict.**
  - Inputs: `results/sobol.json` from R4.5.
  - **Real output:**
    - S1 / ST / S2 tables with bootstrap CIs (`format_sobol_tables`);
    - **the pre-registered verdict, stated explicitly.** §5 pre-registered
      `rt60 × snr_db` as compounding *before any real data existed*. Write the
      resolution as one unambiguous sentence: *"We pre-registered rt60 × snr_db as a
      genuine two-way interaction (SPEC §5, committed <SHA>, <date>). On the real
      grid the ST−S1 gap for rt60 is X [CI a, b] and S2(rt60, snr_db) = Y [CI c, d]
      → confirmed / not confirmed."*
    - **Lead with the ST−S1 gap**, per §5. Use S2 only to say *which* pair
      interacts, never *how much* — its CI crossed zero even with a strong planted
      effect at N=1024.
    - the counterintuitive cells from `find_counterintuitive_cells`.
  - **Protocol for the counterintuitive cells (important):** `find_counterintuitive_cells`
    sweeps a **dense grid** over `wer_fn`, which is ruinous against the real API
    oracle. Run it against the **GP surrogate fitted to the master table**, then
    **verify each candidate cell with real oracle calls** before it goes in the
    writeup. State this two-stage protocol explicitly — surrogate to *propose*,
    oracle to *confirm*.
  - **DoD:** the pre-registration sentence is written with real numbers, and every
    surprising cell in the writeup has been confirmed by real transcription.
  - **Gotcha:** an unconfirmed pre-registration is a **result**, not a failure. Write
    it up as one. Quietly dropping it would be the single worst thing you could do
    to this project's credibility.

- [ ] **R5.5 — D3b: the active-learning savings curve.**
  - **Write `analysis/al_savings.py`** consuming `results/al_curve.json`.
  - **Real output:** `boundary_rmse` (and `boundary_error`) vs number of oracle
    evaluations, active vs random, **with a band across ≥ 3 seeds**; plus the single
    headline number from `evals_to_target`.
  - **DoD:** *"Straddle-acquisition active learning reaches boundary RMSE ≤ 0.08 in
    N_active evaluations vs N_random for random sampling — a P % reduction in oracle
    calls (median over 3 seeds)."* with real N's.
  - **Gotcha:** report the **seed band**, not the best seed. And be explicit that the
    multi-seed replicates ran on the surrogate oracle with one seed confirmed
    end-to-end (R4.7) — don't let the reader assume all seeds were API-backed.

- [ ] **R5.6 — D4: the sim-vs-real gap.**
  - **Write `analysis/sim2real.py`.**
  - Build a second library `DiskAssetLibrary(root="data", rir_subdir="rirs_sim")`
    and re-run the **same condition list** on the **same clips** (R2.5 already
    produced the RIRs and `rir_pairs.json`).
  - **Real output:**
    - paired per-condition WER, measured vs simulated, at matched **measured** RT60;
    - the mean signed gap and its CI ("sim under/over-estimates WER by X points");
    - **Spearman rank correlation across conditions** — the more useful claim: does
      a pyroomacoustics-only testbed put the conditions in the *same order*, even if
      the absolute WERs differ? A high rank correlation with a large level offset is
      a genuinely useful, quotable result for anyone building sim-only benchmarks;
    - whether the dead-zone *set* (R5.1) is the same under sim RIRs.
  - **DoD:** two numbers (mean gap + CI, rank correlation) and a one-paragraph
    interpretation aimed at practitioners who use synthetic RIRs exclusively.
  - **Gotcha:** match pairs on **measured** Schroeder RT60 of both RIRs, never on the
    Sabine target you asked pyroomacoustics for. This is the whole methodological
    point of the comparison and it's easy to get wrong.
  - **Cost:** ~2,400 extra calls. Budget it.

- [ ] **R5.7 — L1: multi-model comparison, on real data.**
  - Feed `compare_models({"nova-3": rows_dg, "whisper-base": rows_ws})` (and `vosk`
    if you download a model dir — optional third arm).
  - **Real output:** per-model dead-zone rate; the confidence-vs-WER **shape** per
    model; and the ranked `find_divergence_regions` output naming the real factor
    regions where the models' dead zones differ.
  - **DoD:** *"In the high-reverb × AMR region, Nova-3's dead-zone rate is X vs
    Whisper's Y; the models disagree most at [named region]."*
  - **Gotchas:**
    - Whisper's "confidence" is not the same quantity as Deepgram's. Check what
      `_parse_whisper_result` (`audio_pipeline.py:420`) actually derives, and state
      it. All cross-model confidence claims must go through
      `within_model_conf_percentile` — the module already does this, don't bypass it.
    - **Whisper hallucinates fluent sentences under heavy noise.** That is a
      *qualitatively different* failure mode from Nova-3's, it will explode the
      insertion count, and it deserves its own callout with an example transcript —
      it's one of the more interesting things you'll find.
    - The §5 third arm (a domain-tuned literal like `nova-2-drivethru`) is a one-line
      model-literal swap if you want it; it is not required.

- [ ] **R5.8 — L2: learned confidence calibration, on real data.**
  - **Getting word-level correctness labels (the fiddly part — here's the recipe).**
    `classify_errors` returns `edits` as an ordered list of `(op, ref_word, hyp_word)`.
    Filter to the edits where `hyp_word is not None` (i.e. `match`, `sub`, `ins`) —
    that sequence is **exactly the hypothesis words in order**, so it aligns 1:1 with
    `word_confidences`. Label `correct = 1` for `match`, `0` for `sub` and `ins`.
    (Deletions have no hypothesis word and therefore no confidence — they are
    invisible to a confidence-calibration analysis, which is itself worth one honest
    sentence in the writeup.)
  - Assemble `(row_params, conf, correct)` per hypothesis word across the grid.
  - Fit `TemperatureScaler` (baseline) and `FeatureCalibrator` (feature-conditioned).
    Report `calibration_report(conf_before, conf_after, correct)` → ECE before/after
    + reliability-diagram data.
  - **DoD:** real ECE before and after for both calibrators, a reliability diagram,
    and a plain-language statement of what the calibrator learned — e.g.
    *"above rt60 = 0.7 s, reported confidence must be discounted by ~0.15 to become
    a calibrated probability."*
  - **Gotchas:**
    - **Split by condition or by clip, never randomly over words.** Words from the
      same clip+condition are highly correlated; a random word-level split leaks and
      will show a fake ECE improvement.
    - `_logit` assumes confidences strictly in (0, 1) — **clip** to `[eps, 1-eps]`;
      vendors do return exactly 1.0.
    - Vendor confidence is not a calibrated probability by construction. That's the
      *premise* of this layer, not a flaw — say so.

- [ ] **R5.9 — L3: paralinguistic decoupling, on real audio.**
  - This one needs **degraded audio on disk**, unlike every other layer (which only
    needs the table). Add a `--save-audio` flag to `run_experiment.py` and use it for
    a **sweep subset only**: one factor varied over ~8 levels with everything else
    fixed (do it for `snr_db` and for `rt60` separately), on ~5 clips.
    ~80 wavs. Don't save the full grid — that's tens of GB.
  - Run `extract_features_from_path` on clean + each degraded clip, `feature_drift`
    against the clean features, then `compare_degradation_rates(levels,
    feature_curve, lexical_curve)` where `lexical_curve` is WER at the same levels.
  - **Real output:** per-factor coupling verdict — spearman, `max_abs_gap`,
    each curve's half-degradation level, and which one leads.
  - **DoD:** a statement of this form, per swept factor —
    *"Under increasing reverb, pitch/energy features hold to rt60 ≈ 0.8 while WER
    halves at rt60 ≈ 0.45: lexical accuracy leads, i.e. a paralinguistics-driven
    agent would not notice its ASR had already failed."* Coupling is also a valid
    result; report it as such.
  - **Gotcha:** features must come from the **degraded** audio and the clean
    reference from the **raw recording** — not from the near-clean composed
    condition, which already has an RIR on it.

---

## A.R6 — The interactive dashboard (~6 h)

- [ ] **R6.1 — Choose the stack: a single self-contained HTML file.**
  - **Recommendation: one static HTML file** with the results inlined as JSON and
    vanilla JS (+ inline SVG or a bundled plotting lib). Rationale is demo risk:
    it opens offline from `file://`, has no server to crash, no port conflict, no
    `pip install` on someone else's laptop, and can be handed over as a link or an
    attachment.
  - Streamlit is faster to build if the interactivity gets complex; if you go that
    way, accept that a live demo now depends on a running Python process.
  - **DoD:** the choice is made and written down; don't relitigate it at hour 5.

- [ ] **R6.2 — Build the shell against synthetic results FIRST.**
  - Generate `results/synthetic_master.csv` in the master-table schema (R4.2) from
    the existing synthetic validators — the same planted-structure generators the
    test suites already use.
  - Build every panel against that file. Keep **all** data access behind one
    `loadData()` function / one embedded JSON blob, so swapping to real results is a
    one-line change.
  - **DoD:** the full dashboard renders on synthetic data before any real grid exists.
  - **Why:** this is §9's Track E guidance, and it means the dashboard is not on the
    critical path behind the grid.

- [ ] **R6.3 — Build the panels.**
  1. **Silent-failure map (hero).** Confidence vs WER scatter, dead-zone quadrant
     shaded, hover → factor values + the actual transcript diff. This panel *is* the
     project; give it the most space and the clearest caption.
  2. **Factor-space heatmap.** `rt60 × snr_db` grid, cell fill = WER, a second
     encoding (border weight / hatch) = the confidence gap. Faceted by `noise_type`
     and `codec`.
  3. **Fingerprint panel.** Stacked sub/del/ins bars per condition family + entity
     error rate overlaid, with the implied-fix text.
  4. **Sensitivity panel.** S1 / ST bars with CIs + the pre-registration verdict
     rendered as a prominent confirmed/not-confirmed statement.
  5. **Active-learning panel.** Steppable: GP posterior + selected points as the
     loop advances, plus the active-vs-random curve with the seed band.
  6. **Sim2real panel.** Paired measured-vs-sim WER, with the rank correlation.
  7. **Model toggle.** Switch the entire dashboard between nova-3 / whisper.
  8. **Audio players.** Embed clean + degraded audio for ~6 representative
     conditions with the transcript diff beside each. **Let people hear the dead
     zone** — this is worth more than any chart in a live setting. Encode as small
     compressed files or base64 data URIs; watch total page size.
  - Keep one visual system across panels: one categorical palette for factors, one
    sequential ramp for WER, one consistent "danger" accent for dead zones. Every
    panel gets a one-line "what am I looking at" caption.

- [ ] **R6.4 — Swap in the real results.**
  - Point `loadData()` at `results/master.parquet`-derived JSON. Verify every panel
    still renders and that no panel silently shows an empty chart when a field is
    missing.
  - **DoD:** all eight panels render on real data with no console errors.

- [ ] **R6.5 — Demo-ready criteria.**
  - Opens from `file://` with wifi **off**; loads in < 2 s; no console errors; legible
    at 1280×800 and on a projector (test at 150 % zoom); every panel captioned;
    graceful empty-state if a results file is absent; a scripted **3-minute path**
    through it written down in `dashboard/DEMO.md`.
  - **DoD:** you have run the 3-minute path start to finish, offline, without
    touching anything but the dashboard.

---

## A.R7 — FINAL MILESTONE: the live voice agent (~12–16 h)

> **⚠️ Highest effort and by far the highest demo risk in the project.** Everything
> else is offline, deterministic and reproducible. This is a real-time networked
> system with three vendors in the loop; it can fail live for reasons that have
> nothing to do with the quality of your work. **Build it last. Never let it be the
> only thing you can show. Never let it block the writeup.**
>
> **Discipline: build and validate on synthetic/replay first** — the same rule that
> got L1–L4 built before any audio existed. `agent_eval.py` is already
> synthetic-validated; the live system must reach the *same* scoring interface, not
> a new one.

- [ ] **R7.1 — Build the replay harness before anything live.**
  - Offline path: degraded wav file → STT → LLM → TTS-to-file, emitting the **same
    event stream** `analyze_turns` consumes. No microphone, no speakers, no realtime.
  - **DoD:** a full condition sweep runs unattended in replay mode and writes
    `results/agent_runs.jsonl`.
  - **Why:** if replay works and scores, the live path is only an IO swap. If you go
    live first, you'll be debugging audio devices and endpointing simultaneously.

- [ ] **R7.2 — STT leg: Deepgram streaming (`listen.live`, WebSocket).**
  - Interim results on; configure `endpointing`, `utterance_end_ms`, `vad_events`.
  - **The agent's endpoint decision is the `agent_endpoint` event** — that is the
    signal the whole turn-taking finding rests on.
  - **DoD:** streaming a wav into the socket produces interim + final transcripts and
    a timestamped endpoint event.
  - **Gotcha:** your recorded clips carry ~0.5 s of trailing room tone (R1.5), which
    may itself trip the endpointer. Set `utterance_end_ms` **deliberately**, write
    the value into the methods section, and treat it as a **configured parameter you
    chose**, not a property you discovered.

- [ ] **R7.3 — LLM leg: Claude, constrained to a slot-filling task.**
  - Use `claude-sonnet-5` (or `claude-haiku-4-5` if you need the lowest latency).
  - **Keep the task narrow and structured:** the agent's job is to extract the
    utterance's entities into a structured tool call / JSON object. That is what
    makes the run *scoreable* — the agent's extracted slots go straight into
    `evaluate_task(…, TaskSpec)` from `task_specs.json` (R5.2). A free-form chatty
    agent has no ground truth and produces no finding.
  - **DoD:** given a clean transcript, the agent emits slots that `evaluate_task`
    scores at 1.0 slot accuracy.

- [ ] **R7.4 — TTS leg + playback.**
  - Deepgram Aura keeps it to one vendor and one key; ElevenLabs or any streaming
    TTS also works.
  - In replay mode: render to file, don't play. In live mode: stream to the output
    device with barge-in support (kill playback the moment user speech starts).
  - **DoD:** replay mode produces a TTS wav per agent turn with timestamps.

- [ ] **R7.5 — Turn-taking loop + event log.**
  - Emit, with monotonic timestamps, into `results/agent_events.jsonl`:
    `user_speech_start`, `user_speech_end`, `agent_endpoint`, `agent_speech_start`,
    `agent_speech_end` — the exact constants at `agent_eval.py:102-106`. Do not
    invent new event names; the analyzer is already written and tested against these.
  - Handle: response timeout, barge-in (user speaks over agent), and missed turns.
  - **DoD:** `analyze_turns(events)` runs on a real log and returns findings without
    a schema error.

- [ ] **R7.6 — Build the multi-turn material (the corpus doesn't have it).**
  - The 40 manifest utterances are **single-turn**; turn-taking needs dialogues.
  - **Recommended approach — splice, don't improvise.** Author ~8–10 two-to-four-turn
    dialogues from manifest entities, and construct their audio by concatenating
    recorded clips with **controlled inter-turn gaps** (0.2 / 0.5 / 1.0 / 2.0 s) plus
    one deliberately **overlapping barge-in** case.
  - **Why splicing is better than recording conversations:** it gives you a
    *ground-truth turn timeline* — you know exactly when each turn started and ended,
    to the sample. Real conversation does not come with labels, and `analyze_turns`
    needs the truth to score false endpoints and missed turns against.
  - **DoD:** `data/dialogues/*.wav` + `data/dialogues/timeline.json` with
    ground-truth turn boundaries.

- [ ] **R7.7 — Run the degradation testbed against the full agent.**
  - **Injection method — file-injection, not acoustic playback:**
    - **(a) DO THIS:** stream the degraded WAV bytes directly into the Deepgram live
      socket, chunked and **paced to realtime**. No audio hardware, fully
      reproducible, and it still exercises the real streaming + endpointing path.
    - **(b) DO NOT DO THIS for measurement:** playing audio through speakers into a
      mic. It re-introduces uncontrolled room acoustics *on top of* your simulated
      ones, which directly contradicts the counterfactual-isolation premise the whole
      project rests on (§1). It's a fine party trick; it is not a measurement.
  - **Reduced condition set:** ~12 conditions spanning the dead zone and the safe
    zone (drawn from R5.1's real results) × 10 dialogues. Live runs are **realtime-
    paced**, so ~12 × 10 × 20 s ≈ 40 min of audio plus LLM/TTS latency ≈ **1.5–2 h**
    of wall clock. Don't try to run the full grid through the agent.
  - **DoD:** `results/agent_runs.jsonl` with one record per (dialogue, condition)
    containing slots, transcripts, the event log, and the scores.

- [ ] **R7.8 — The two agent findings.**
  1. **Task/entity accuracy diverges from WER.** Plot slot accuracy, entity error
     rate and critical-slot failure against WER across conditions. `agent_eval` is
     built so these *can* disagree; show whether they do on real data, and by how
     much. "WER 0.15 but 40 % of critical slots lost" is the sentence you want.
  2. **Turn-taking fails before transcription does (or doesn't).** Plot false
     endpoints / missed turns / barge-in failures against acoustic condition,
     overlaid on WER. **This is the money finding** — §5's stretch goal — because the
     WER literature uses isolated utterances and structurally cannot see it.
  - **DoD:** both plots exist with real numbers, and each has a one-sentence takeaway.

- [ ] **R7.9 — The off-ramp (decide this in advance, not under pressure).**
  - **If the live agent is not solid 3 days before the demo:** ship the **replay-mode**
    results as the finding, and present `agent_eval.py` as a validated scaffold.
    That is a completely respectable outcome and it is *exactly* what §5's L4 entry
    already claims — structure built and synthetic-validated, drop-in ready.
  - Regardless of outcome: **screen-record a successful live run** as soon as you get
    one, and keep it in the demo kit as the fallback.
  - **DoD:** the recording exists, or the off-ramp is taken and written up honestly.

---

## A.R8 — The write-up (~6 h; sections 1–5 and 8 are draftable NOW)

- [ ] **R8.1 — Draft the data-independent sections immediately.**
  - Motivation, prior work, method, experimental design, and limitations need **zero**
    real data. Write them during R1/R2 download waits (§9 says exactly this). By the
    time the grid lands you should only be filling in numbers.
  - **DoD:** `report/writeup.md` exists with sections 1–5 and 8 drafted before R4
    completes.

- [ ] **R8.2 — Section list (write them in this order).**
  1. **Abstract / TL;DR** — one sentence with the headline number in it.
  2. **Motivation** — aggregate WER hides *where* and *how*; the reframe from "how
     much does it break" to "does it know it's breaking" (§2).
  3. **Prior work — up front, before any of your own results.** WildASR / "Back to
     Basics", Speech Robustness Bench, "When Denoising Hinders", Ko et al. 2017,
     Kim et al. 2017, REVERB / CHiME, Scheibler 2018 (pyroomacoustics), Carlini &
     Wagner. **Say plainly that this genre is well-trodden** and state the delta
     modestly: the silent-failure lens, typed fingerprints, the AL surrogate, and
     testing a *commercial streaming model with per-word confidence* where the
     literature mostly uses Whisper/Conformer/wav2vec. Do **not** claim novelty
     (CLAUDE.md positioning).
  4. **Method** — the factor space; the composition order and *why it's physically
     motivated* (`COMPOSITION_ORDER`, `conditions.py:428`); and **the three trap
     functions with the bugs they prevent**. Write this section in detail — it is the
     strongest competence signal in the document. The `apply_rir` renormalization bug
     (reverb tail leaking energy into silent regions and de-calibrating downstream
     SNR, caught by the test suite and by *no* error message) is the anecdote to tell.
  5. **Experimental design** — PB screen with `SCREEN_CAVEAT` quoted verbatim;
     Saltelli/Sobol with bootstrap CIs; and the **pre-registered** `rt60 × snr_db`
     hypothesis, cited with its commit SHA and date to prove it predates the data.
  6. **Results** — one subsection per finding, D1 first: silent-failure map →
     fingerprints → interactions & pre-registration verdict → AL savings → L1 model
     comparison → L2 calibration → L3 decoupling → (L4 agent, if built).
  7. **Sim-vs-real gap** — its own section (§4 promises it as a first-class finding).
  8. **Limitations and honest boundaries — its own numbered section, not a footnote:**
     - behavioral factors deliberately bracketed out, with the **Lombard effect named
       and explained** (noise changes how speech is *produced*; no room simulator
       captures a behavior) — §4;
     - **one speaker, one accent, 40 utterances** — results do not generalize across
       speakers, and this is a real limit, not a caveat;
     - reference transcripts are human-verified but human (with the measured clean
       WER floor from R1.10 quoted);
     - **commercial model literals are moving targets** — quote the exact string and
       date from `results/MANIFEST.json`;
     - the RIR/noise sample is small and its RT60 coverage is discrete, so the `rt60`
       axis is realized by nearest-match snapping (R2.1) — state the achieved coverage;
     - **WER is not the deployment metric** — which is precisely why the agent layer
       exists;
     - vendor confidence is not a calibrated probability by construction (the premise
       of L2, not a defect);
     - composition order is fixed and order effects are unstudied;
     - if TTS was used for the corpus (R1.2 fallback), say so and say why it makes
       the corpus easier than reality.
  9. **What I'd do next / what field data would earn** — the §4 framing: "here's what
     I isolated in sim; these human factors are exactly where real field recordings
     would earn their keep."
  10. **Reproducibility appendix** — commands, `results/MANIFEST.json` contents,
      asset checksums, versions, total API cost, wall-clock.

- [ ] **R8.3 — Interview-ready criteria.**
  - Every claim carries a number **and** an interval.
  - You can state, **unprompted**, the strongest argument against your own headline
    finding.
  - Every figure has a one-sentence takeaway caption; the figure is readable without
    the body text.
  - The limitations section is the section you'd *most* want to be asked about.
  - Reads end-to-end in under 10 minutes.
  - Contains one honest sentence about what surprised you and one about what you got
    wrong first (the RIR renormalization bug is the natural candidate).
  - Positioning check: nowhere does the document claim novelty of method.
  - **DoD:** a colleague who has never seen the repo can read it and correctly state
    the headline finding, the main limitation, and what you'd do next.

---

## A.R9 — Demo kit (~2 h)

- [ ] **R9.1 — Live-demoable pieces (deterministic, offline, fast).**
  1. **The test suite** — `python3 test_pipeline.py` and friends, ~30 s. Optional
     opener; it shows the trap functions and the engineering discipline behind them.
  2. **One-clip degrade-and-break** — write `demo_break.py`: plays the clean clip,
     plays the dead-zone version, and shows the transcript diff **beside the
     confidence number**. This is the single most visceral 60 seconds you have.
     - **Pre-generate the audio and cache the transcripts.** Include an `--offline`
       flag that uses cached results so it needs **no network and no API key**.
  3. **The active-learning loop, running live** — run it against the **GP surrogate
     fitted to the master table**, not the API. Instant, deterministic, and it shows
     the boundary tightening as points are chosen. ~20 s.
  4. **The dashboard** — self-contained HTML, offline (R6.5).
  5. **The live agent** — *only if solid* (R7.9), always with the screen recording
     loaded and ready as fallback.

- [ ] **R9.2 — Presented as findings (too slow or too fragile to run live).**
  - The Sobol sensitivity tables (hours of compute).
  - The full grid run (hours; thousands of API calls).
  - The calibration ECE before/after numbers.
  - The sim-vs-real gap.
  - The paralinguistic decoupling curves.
  - The agent task/turn-taking results if the live path was off-ramped.

- [ ] **R9.3 — Demo hygiene.**
  - Every demo runs **offline** with wifi off — rehearse it that way.
  - **No API key required on stage** (everything cached).
  - `pip freeze > requirements.lock.txt` and commit it.
  - One command per demo (a `Makefile` target or a single script each); no
    multi-step recipes typed live.
  - Full dry run on the demo laptop, wifi off, projector attached, at least once.
  - Timings rehearsed: opener 30 s · degrade-and-break 60 s · AL loop 30 s ·
    dashboard 3 min · agent 2 min.
  - A `README` section: **"Run the demo in three commands."**
  - **DoD:** the complete demo has been run start to finish, offline, on the actual
    demo machine, once, without improvisation.

---

## A.10 — Pre-flight checklist (the short version)

- [ ] `test_*.py` all green (R0.2)
- [ ] `smoke_deepgram.py` returns non-empty word confidences (R0.3)
- [ ] `check_recordings.py` → 40/40 OK (R1.7)
- [ ] Clean baseline mean WER ≤ 0.02, every exception adjudicated (R1.10)
- [ ] RIR measured-RT60 coverage spans [0.2, 1.0] with no gap > 0.15 s (R2.1)
- [ ] `data/noise/{babble,engine,road}/` each ≥ 3 clips ≥ 60 s (R2.2)
- [ ] `smoke_codec.py` passes for **every** codec level (R2.4 — AMR encoder issue resolved)
- [ ] JOIN-1 gates A/B/C pass on 3 clips, and you have **listened** to the audio (R3)
- [ ] `results/master.parquet` exists, failure rate < 2 % (R4.4)
- [ ] `results/MANIFEST.json` written and `grid-v1` tagged (R4.6)
- [ ] A real dead-zone condition can be named with real numbers (R5.1)
- [ ] The pre-registration verdict is written, whichever way it went (R5.4)
- [ ] Dashboard opens offline and the 3-minute path is rehearsed (R6.5)
- [ ] Writeup limitations section names the Lombard boundary explicitly (R8.2)
- [ ] Full demo rehearsed offline on the demo machine (R9.3)

---

# Appendix B: Progress Log — state as of 2026-08-05 (evening)

> Written at a hard pause. **Read this FIRST when resuming**; it supersedes the
> checkbox states in Appendix A wherever the two disagree. Everything below is
> read off the actual artifacts in `results/`, not from memory.
>
> **Working tree is uncommitted and several files are mid-edit.** See B.7 before
> you commit anything.

## B.1 Phase status

| Phase | State | Evidence |
|---|---|---|
| **R0** environment | ✅ done | all suites green |
| **R1** corpus | ✅ done | `check_recordings.py` → 40/40 OK; clean-condition WER **1.65 %** (6 errors / 363 ref words), 35/40 clips exact, every non-zero row adjudicated by ear |
| **R2** assets | ✅ done | 16 curated RIRs, 12 noise clips (4/type, 300 s), G.726 codec verified |
| **R3** JOIN-1 gate | ✅ PASSED | 9/9 gates on u02/u17/u36; SNR calibration exact to 0.01 dB on real audio |
| **R4** grid | ✅ done | 176 conditions × 40 clips × nova-3 = **7040 rows, 0 failures**, + the 1760-row whisper arm = **8800 rows, 3 failures (0.03 %)**, well under the 2 % gate; `results/master.parquet` now written too (pyarrow installed — it was silently falling back to CSV only) |
| **R4.3/4.5** screen + sensitivity | ✅ **done, and better than planned** | exact functional-ANOVA decomposition of the real factorial — see B.3 |
| **R4.7 / R5.5** active learning | ✅ done — **a null, and it is robust** | see B.4 |
| **R5.1** D1 headline | ✅ real — ⚠️ **SUPERSEDED by Appendix G** | then: ρ = −0.957, 6/176 dead zones. **Now: ρ = −0.980 paired / −0.952 all-clips (n = 169), 2/176 dead zones** — the estimand mismatch |
| **R5.3** D2 fingerprints | ✅ real | deletions dominate; entity error 0.633 vs WER 0.511 |
| **R5.4** D3a interactions | ✅ real — **pre-registration CONFIRMED**, report reconciled | see B.3 |
| **R5.6** D4 sim2real | ✅ real | sim underestimates WER by 12.1 pts; Spearman 0.873; **dead-zone Jaccard 0.00** |
| **R5.7** L1 multi-model | ✅ **UNBLOCKED AND DONE** | whisper arm 1760/1760; `results/model_arms.{json,txt}` written — see B.9 |
| **R5.8** L2 calibration | ✅ **re-run and final** | see B.4 |
| **R5.9** L3 decoupling | ✅ **real, two quotable verdicts** | see B.4 |
| **R6** dashboard | ✅ **done** — all 8 panels `ok`, rebuilt on the full table | see B.6 |
| **R7** live agent | ⬜ **deliberately out of scope** | user-scoped out; `agent_eval.py` remains a synthetic-validated scaffold |
| **R8** write-up | 🟡 **compressed version being written** | see B.5 |
| **R9** demo kit | ✅ **done** — every `make` target passes offline, 21/21 suites green | see B.6 |
| **audit** sensitivity | ✅ **independently re-derived** | every headline Sobol number reproduced to 1e-16 without importing the module — see B.10 |

---

## B.2 Decisions made (settled — do not relitigate)

Carried forward from the previous log and still binding:

1. **Narrowband codec is G.726, not AMR-NB.** Stock ffmpeg is AMR decode-only.
   Must be named in the methods section.
2. **RIR source is the MIT Acoustical Reverberation Survey**, curated to 16 IRs by
   greedy RT60 tiling. BUT ReverbDB remains a documented upgrade path.
3. **Noise is DEMAND**, two environments per `noise_type`.
4. **The grid that runs is `interaction_grid()`, not `main_grid()`.**
5. **Sim/real pairing is on measured Schroeder RT60**, never the Sabine target.
6. **`normalize_text` deletes apostrophes and collapses a small compound map**,
   applied symmetrically to reference and hypothesis.
7. **The sim arm uses a separate `results_sim/` cache.** The cache key is
   `(clip_id, condition_name, model)` and does **not** encode which RIR library
   produced the row, so a shared cache would be 100 % false hits and would report
   a sim2real gap of exactly zero. Keep the directories separate. The dashboard
   now re-joins the two tables at build time via `--sim-master` and re-splits on
   `rir_key`, which is the field that actually records provenance.

New this session:

8. **Cross-model WER gets its own normalization module, and the trap function is
   left alone.** `cross_model_norm.py` applies the Whisper authors' published
   `EnglishTextNormalizer` plus symmetric digit-run splitting, to **both** arms,
   used **only** for L1. Rationale: Whisper writes numbers as digits and has no
   formatting switch, while the Deepgram adapter disables
   `smart_format`/`punctuate`/`numerals`, so its output is already word-form. The
   offset that creates is 0.20–0.60 depending on entity density and is
   condition-independent — mathematically indistinguishable from an acoustic
   effect once it is in the table. Adding digit mapping to `normalize_text` was
   rejected because the corpus itself uses contradictory conventions
   (`u02` "four zero five" → 405, but `u05` "fourteen hundred" → 1400 and
   `u11` "eighty eight" → 88): no single rule is correct, and guessing inside a
   trap function is the exact failure this project is about.
   Three residuals are **pinned by tests rather than patched**: letter runs in
   spelled codes (`AW` vs `a w`); the normalizer's asymmetry about leading zeros
   (`Q9J05` costs one spurious deletion); and the transform is deliberately **not
   idempotent** (`EnglishTextNormalizer` rewrites a standalone `1` as `one`, and
   digit-splitting manufactures standalone digits) so it must be applied exactly
   once, to raw text.
9. **Sensitivity is computed by exact functional ANOVA, not Saltelli sampling.**
   See B.3 — this supersedes R4.5's plan.
10. **`make_manifest.py` freezes the experiment** into `results/MANIFEST.json`
    (git SHA, model literals, ffmpeg build, codec rationale, asset SHA-256s,
    realized totals). The write-up quotes it rather than restating it.

---

## B.3 The methodological upgrade: the grid is a complete factorial

The single most consequential discovery this session. `interaction_grid()`'s
babble core turns out to be a **complete 4×4×3×3 factorial** —
`rt60` {0.2, 0.45, 0.7, 1.0} × `snr_db` {0, 5, 10, 20} × `codec` {none, g726,
opus-lowrate} × `mic_rolloff` {0.0, 0.5, 1.0}, 40 clips in every cell, 144 cells,
5760 real transcriptions — plus a 32-cell engine/road arm.

A complete factorial with equal cell counts admits an **exact** variance
decomposition. Main effects, every two-way interaction and the higher-order
remainder form a finite partition of the measured response. So:

- **Sobol indices are computed exactly, not estimated.** `analysis/sensitivity.py`
  decomposes the measured grid directly. Verified partition:
  `sum(S_u) = 1.000000000000`, max abs error 0.00e+00. No surrogate, no
  Monte-Carlo sampling error. This is strictly stronger than running Saltelli
  against a GP fitted to the same grid, which is what R4.5 originally planned.
- **CIs come from bootstrapping the 40 CLIPS**, 2000 replicates — clips are the
  right resampling unit because cells within a clip are correlated.
- **The Plackett–Burman screen is derived from the grid**, saving 320 fresh API
  calls. `SCREEN_CAVEAT` (Resolution-III aliasing) is quoted verbatim in the
  write-up **with a note that it does not bind here**, because a complete
  factorial has no aliasing.

**Results** (`results/sobol.json`, `results/sensitivity_report.txt`, total
variance of condition-mean WER 0.1266):

| factor | S1 | ST | ST−S1 | 95 % CI | sig |
|---|---|---|---|---|---|
| `rt60` | 0.347 ± 0.024 | 0.474 ± 0.027 | **0.128** | [0.091, 0.164] | YES |
| `snr_db` | 0.391 ± 0.030 | 0.503 ± 0.027 | **0.112** | [0.072, 0.152] | YES |
| `mic_rolloff` | 0.099 ± 0.013 | 0.183 ± 0.020 | 0.084 | [0.060, 0.107] | YES |
| `codec` | 0.023 ± 0.003 | 0.065 ± 0.009 | 0.042 | [0.032, 0.052] | YES |

S2 ranking (**direction only, never magnitude**, per SPEC §5):
`rt60×snr_db` 0.034 ± 0.006 (rank 1/6) > `rt60×mic_rolloff` 0.019 > the rest.

### The pre-registration: **CONFIRMED**

> We pre-registered `rt60 × snr_db` as a genuine two-way interaction (SPEC §5,
> Track-C design notes, committed `d8ddd4f`, 2026-07-27). On the real grid
> (N = 144, 5760 model evaluations) the ST−S1 gap is **0.128 [0.091, 0.164]** for
> `rt60` and **0.112 [0.072, 0.152]** for `snr_db`, and S2(`rt60`, `snr_db`) =
> 0.034 ± 0.006 (rank 1/6, direction only) — **CONFIRMED**. Reverb and noise
> compound.

Decision rule was fixed in advance: confirmed iff both registered factors' ST−S1
gaps exceed 0.020 with the 95 % bootstrap CI entirely above it, **and** the
registered pair ranks first in S2 (direction check only).

### The best mechanistic finding in the project: DRR, not RT60

The `rt60` axis is **non-monotonic** — marginal WER runs 0.2026 → 0.6359 → 0.4495
→ 0.7581 across the four levels, a significant dip at 0.7 (depth 0.1864
[0.1574, 0.2142], 36 cells × 40 clips; 33 cellwise dips have CIs entirely above
zero, the deepest 0.5765 [0.4683, 0.6707]).

The mechanism: each `rt60` level is delivered by the **nearest measured RIR**,
i.e. a *different real room*. RT60 describes a decay slope and says nothing about
how much direct sound reaches the mic.

| requested | room | RT60 | DRR dB | C50 dB | WER |
|---|---|---|---|---|---|
| 0.2 | Restaurant | 0.193 | 16.90 | 28.10 | 0.2026 |
| 0.45 | Bar | 0.474 | −2.05 | 10.22 | 0.6359 |
| 0.7 | Campground Dining | 0.680 | 4.26 | 10.03 | 0.4495 |
| 1.0 | Shower | 1.011 | −10.02 | 2.12 | 0.7581 |

`spearman(DRR, WER) = −1.000` (pearson −0.995) against `spearman(RT60, WER) =
+0.800`. **The damage is monotone in direct-to-reverberant ratio, which RT60 does
not capture.** Reverb benchmarks parameterised by RT60 alone will mis-rank
conditions for exactly this reason. This is a genuinely quotable result and it
should be prominent in the write-up.

### ✅ RESOLVED — the in-grid dip vs the six unreproduced proposals

`results/interaction_report.txt` reported **both** a significant in-grid rt60 dip
**and** `0/6 proposals reproduced`, which read as a contradiction. It now carries a
`RECONCILIATION` section, and the resolution is more interesting than the
contradiction.

**My working hypothesis was that the probe's three requests (0.6 / 0.7 / 0.8) all
snapped to the same RIR, so the probe had nothing to dip. That hypothesis was
tested and is FALSE** — it resolved to three *distinct* files. The real
explanation: the two scans examined **different, almost non-overlapping room
triplets**, sharing only one room.

| scan | requests → rooms | DRR dB | outcome |
|---|---|---|---|
| GRID | 0.45 Bar · 0.7 Campground · 1.0 Shower | −2.05 / **4.26** / −10.02 | **DIP** (middle room has the *best* DRR) |
| PROBE | 0.6 Office ConfRoom · 0.7 Campground · 0.8 Classroom | 7.76 / **4.26** / 9.42 | **PEAK** (middle room has the *worst* DRR) |

The mechanism predicts **opposite signs** for the two triplets, and that is
exactly what was measured. So the refutation is real but narrowly scoped: it
refutes *those six surrogate-proposed cells only*, and is **not** a refutation of
the in-grid measured dip.

**The conclusion is the finding:** non-monotonicity along `rt60` is not a property
of a response surface at all. Each `rt60` request indexes an unrelated real room
via nearest-match snapping, so whether a dip exists — and where — is a property of
**which RIRs were curated**, not of reverberation time. Re-sample the axis and the
dip moves or disappears, which is literally what happened between the two scans.

*Consequence for surrogates:* a GP fitted with `rt60` as a **continuous**
coordinate assumes a smoothness the instrument does not have, so it will keep
proposing cells the oracle cannot reproduce. The defensible parameterisation for
this axis is **DRR (or C50)**, which orders the measured conditions perfectly
where RT60 does not.

The six proposals now print with the held-fixed coordinates that differ between
them, instead of six identical lines.

---

## B.4 Real findings

> ⚠️ **The D1 block immediately below is SUPERSEDED by Appendix G.** The
> confidence–WER estimand mismatch inflated the mean gap by +0.109 and produced
> four of the six dead zones. Corrected: **2/176 dead zones**, ρ = **−0.980**
> paired / **−0.952** all-clips (n = 169), and the #1 condition quoted here is
> now classified `silence_driven`. D2, L2, D3b, D4 and L3 below are unaffected.

**D1 — the headline is more nuanced than the premise assumed.** Global
spearman(confidence, WER) = **−0.957**: nova-3 largely *does* know when it is
failing. But it is **overconfident in 92 % of conditions** (mean gap 0.256) and
**3.41 % (6/176) are genuine dead zones**. Ranked #1: `rt60 0.7 s, SNR 20 dB,
babble, opus-lowrate, rolloff 1.0` → mean word confidence **0.843** at WER
**0.387** (n = 40 clips, 363 ref words, 0 failed; `conf_pct` 0.661, `gap` 0.230).

> 📎 **Column-order note for `results/dead_zones.csv` — read this before quoting
> the file.** The schema is `…, mic_rolloff, rt60_measured, mean_conf, conf_pct,
> wer, …`. On the #1 row `rt60_measured = 0.680` sits immediately before
> `mean_conf = 0.843`, and 0.680 reads exactly like a plausible confidence, so
> an off-by-one column read silently swaps the *delivered reverb time* for the
> *headline confidence* — and the result still looks sane. This was mis-read once
> during the 2026-08-05 session and caught only by an arithmetic check.
> **The check that settles it:** `gap = mean_conf − (1 − wer)`, i.e.
> 0.843 − 0.613 = 0.2297, which reproduces the stored `gap` field exactly.
> Corroborated independently by `demo_break.py` ("rt60 0.7s (measured 0.68s) …
> mean confidence 0.843") and by `results/interaction_report.txt`, which lists
> the rt60 = 0.7 room (Campground Dininghall) at measured RT60 0.680.
> Verify any quoted row against that identity rather than by counting columns.

*Framing:* the danger is not that the model is blind — it is that it is *mostly*
self-aware, so a system calibrated on average behaviour will trust it precisely
where it shouldn't. The strongest counter-argument (and it is in the write-up):
`mean_conf` is **survivor-biased** — see the deletion blindness below.

**D2 — deletions dominate; entities are hit hardest.** `snr_db`, `mic_rolloff`,
`rt60`, `opus-lowrate` → **deletions**. `g726`, `road` → **substitutions**.
`engine` and `codec=none` correctly get **NO FIX**. Destroyed-word rate:
proper_noun **0.646**, spelled_letter 0.613, content 0.530, function 0.462.
Entity error rate **0.633 vs WER 0.511**. Babble insertions are **92 % foreign
tokens** — the model transcribing background speakers, a different mechanism from
confusion.

**L2 — calibration, re-run and now FINAL** (`results/calibration.{json,txt}`).
7040 rows, 42 732 hypothesis words, grouped by **condition** (169 groups), 5 seeds:

| | ECE, median [min, max] |
|---|---|
| raw confidence | **0.0507** [0.0496, 0.0586] |
| + temperature scaling (T = 1.385 [1.354, 1.435]) | **0.0346** [0.0312, 0.0391] |
| + feature-conditioned | **0.0077** [0.0045, 0.0106] |

Held-out **clips** as a robustness check: 0.0487 → 0.0396 → 0.0196.
A random word-level split is **not offered** by the code — it leaks, and the
symptom is a *better* ECE.

*Alignment-fix audit:* 123/7040 rows (1.75 %) re-aligned, 0 still misaligned;
hypothesis words fit on went 41 692 → **42 732** (+1040, +2.43 %). Effect on the
headline ECE: raw −0.0005, temperature −0.0000, feature +0.0001 — **negligible**.
The defect was real (a `zip()` would have bound confidences to the wrong words)
but the old path *skipped* rather than zipped, so the previously reported number
was already safe.

*Deletion blindness, quantified — this is a substantive hole, not a footnote:*
deletions are **22 416 words = 35.1 % of the reference and 69.3 % of ALL errors**,
and carry no hypothesis token and therefore no confidence. A perfectly calibrated
confidence converges on **emitted-word accuracy 0.767**, not on **reference
recovery 0.513** — an overstatement of **0.254** if read as the latter. At the
limit, **7 of 176 conditions returned an empty transcript on every clip**
(WER 1.00, 100 % deletions) and contribute *zero* words: the calibrator is fit on
169 conditions and is **silent about the worst 7**.

*Plain-language statement:* above `rt60 = 0.7` reported confidence must be
discounted by ~0.07 to become a calibrated probability (0.81 reported vs 0.75
observed on 8144 held-out words); above `mic_rolloff = 0.5`, by ~0.06 (0.82 vs
0.76 on 13 823 words).

**D3b — active learning is a NULL, and the null is robust.**
(`results/al_savings.{json,txt}`, `al_curve.json`, `al_trajectory.json`.)

> No savings claim: the `boundary_rmse` target 0.162 was reached by **2/8 active
> seeds and 4/8 random seeds** within the 45-evaluation budget. Report the budget,
> not a ratio.

8 seeds, all against the **surrogate oracle**; **no seed was confirmed end-to-end
against the live API** — the write-up must say so rather than let the reader
assume it. The test set is held-out **real measurements** from the master table
(0 oracle calls to build it). At every target threshold the whole curve is
published so it cannot be cherry-picked; random matches or beats straddle
acquisition throughout (e.g. at target 0.205: active 36 evals vs random 15).
This is a legitimate finding — a smooth, low-dimensional surface with a wide
boundary region gives boundary-seeking acquisition little to exploit. Do not
massage the threshold to manufacture a win.

**D4 — sim2real.** Simulated RIRs **underestimate WER by 12.1 points**; Spearman
rank correlation across conditions **0.873**; **dead-zone Jaccard 0.00** — a
pyroomacoustics-only testbed orders conditions well but recovers *none* of the
actual dead zones. Note both sim2real arms use the 10-clip subset, so their
dead-zone sets are computed within that subset and do not coincide with the
40-clip D1 table.

**L3 — decoupling, now with a harsh-region sweep and two real verdicts.**
The first two sweeps (round 1) pinned the other factor at its benign end and
therefore walked a **flat edge** of the response surface: WER never left the
floor. Round 2 added four sweeps that cross the interaction region. Six sweeps
total, in `results/l3_decoupling.{json,txt}`:

| sweep | verdict |
|---|---|
| `rt60` (snr 20) | LEXICAL FLOOR — features move, WER does not |
| `snr_db` (rt60 0.2) | LEXICAL FLOOR — features move, WER does not |
| `rt60@snr0` | NO SUPPORTABLE THRESHOLD — WER moves, feature curves unreadable |
| **`rt60@opus_roll1`** | **DECOUPLED** — f0 collapses at rt60 ≈ 0.62 while WER only halves at ≈ 0.85 |
| `rt60@g726_roll0.5` | NO SUPPORTABLE THRESHOLD |
| **`snr_db@g726_roll1`** | **DECOUPLED** — rms collapses at ≈ 4.46 dB while WER halves at ≈ 6.61 dB |

Both decoupled verdicts point the **same** way: **the paralinguistic stream leads,
so a feature-based monitor would alarm *before* the transcript measurably
degrades** — the opposite of the failure mode the layer was designed to look for.

*The degeneracy guard is itself a finding.* `compare_degradation_rates` min-max
normalizes both curves before finding the 0.5 crossing, so it cannot distinguish
a 0.0 → 1.0 collapse from a 0.000 → 0.054 wander. The first run duly reported
*"rolloff holds to 15.29 dB while WER halves at 11.58 dB"* — real arithmetic on
meaningless input, and exactly this project's signature failure mode. A
`_curve_degeneracy` guard (`MIN_LEXICAL_RANGE = 0.10`) plus a feature-side trend
guard (`MIN_TREND_RHO = 0.70` on spearman vs severity rank) now **refuses the
threshold instead of inventing one**; non-quotable half-levels print in
`[brackets]`; clips whose own WER is flat get no vote on `leads`; non-trending
features are labelled a **power limitation, not a finding of stability**. This
belongs in the write-up beside the `apply_rir` anecdote as a second worked
example.

*Second half of the baseline trap, found this session:* the raw captures in
`data/recordings/` are 48 kHz while all sweep audio is 16 kHz, and
`centroid`/`rolloff`/`flatness` integrate to Nyquist — so a 48 kHz baseline
injects a large constant offset unrelated to degradation. Documented in the
module header.

*L3 API spend:* 85 calls round 1 (0 failures, all cached to
`results/l3_transcripts.jsonl`, re-run is free), plus round 2.

---

## B.5 Blockers and known defects (start here on resume)

1. 🟡 **Whisper arm is at 1597/1760 rows** (163 left) — every row cached, so
   resuming costs nothing for work already done. **Resume with `--workers 1`:**
   `SSL_CERT_FILE=$(./.venv/bin/python -c 'import certifi;print(certifi.where())') ./.venv/bin/python run_experiment.py --models whisper-base --clips al --workers 1`
   Then `--rebuild --models nova-3,whisper-base` to refresh `results/master.csv`,
   then `./.venv/bin/python -m analysis.model_arms` to produce
   `results/model_arms.json` and unblock L1 and dashboard panel 7.

   **⚠️ Do NOT run the Whisper arm with more than one worker.** At `--workers 4`
   it aborted with exit 134:
   *"Numba workqueue threading layer is terminating: Concurrent access has been
   detected. The workqueue threading layer is not threadsafe."* Whisper pulls in
   Numba, and the default workqueue layer cannot be entered from multiple Python
   threads. Earlier multi-worker runs survived by luck. Whisper is CPU-bound
   anyway, so the threads bought little; serial is both safe and barely slower.
   (`NUMBA_THREADING_LAYER=tbb` is the documented alternative if parallelism is
   ever actually needed — untested here.)

   *(The earlier note here blaming a missing `./.venv/bin/pip` was wrong: pip
   exists; `nohup` failed to resolve the relative path. The first real blocker was
   `CERTIFICATE_VERIFY_FAILED` on the model-weights download — the same macOS
   missing-CA-bundle issue that bit the DEMAND fetch. Note that
   `./.venv/bin/pip` IS a broken console script on this machine for an unrelated
   reason — its shebang points at the venv's pre-rename path — so always use
   `python -m pip`.)*
2. ✅ **The `interaction_report.txt` contradiction is RESOLVED** — see B.3. The
   report now carries a `RECONCILIATION` section and the two results are shown to
   be consistent under the DRR mechanism.
3. ⚠️ **The six surrogate-proposed counterintuitive cells remain unreproduced**
   and must not be presented as measured surprises. The *in-grid* dips (B.3) are
   real measurements with CIs and stand on their own. The report now scopes the
   refutation explicitly to those six cells.
4. 🟡 **`report/writeup.md` is still the LONG version** — 885 lines, §1–10 plus
   Appendices A–F, **7 `[[PENDING: ...]]` markers**. **DECISION MADE: ship the
   compressed version only.** Target is **~3,500 words in the main body, under 10
   minutes end to end**; deep material moves to appendices rather than being
   deleted; do not expand §4 back out. The agent had all the final numbers in hand
   and was writing the compressed version when the pause landed, so none of that
   work is in the file yet. Every number it needs is in B.3/B.4 above.
5. ✅ **`test_demo.py` is 22/22 — the earlier "1 failing" note was STALE.**
   Re-verified 2026-08-05 with **zero code changes**: the fix bar was already met
   in the tree. `test_demo.py`'s `cache()` helper self-heals (runs
   `demo_break.py --prepare` itself if `results/demo/demo_cache.json` is absent),
   and `demo_break.py:play()` checks `is_file()` and prints
   `(audio missing: … -- run 'make demo-prep')` instead of raising. The line that
   was mistaken for a failure is that **intended** message. Verified robustly by
   moving `results/demo/` out of the repo entirely, re-running (it rebuilt from
   scratch, offline, no API key, 22/22), then restoring the original — so the pass
   is not a stale-cache illusion.
6. 🟡 **`test_dashboard.py`: 21/22, one remaining failure.** Confirmed
   2026-08-05 to be exactly the Whisper dependency: `test_model_toggle_re_renders_
   every_panel` → *"panel-hero went empty after switching model"* — the second
   model has no rows in `master.csv` yet. Do the final rebuild as
   `./.venv/bin/python dashboard/build.py` (**no flags** — `--no-al` was what
   broke one of the two already-fixed tests) after the Whisper arm lands.
7. ⬜ **Nobody has listened to the degraded audio yet** (A.R3.5). The files are in
   `results/audio/listen/` with `WHAT_TO_LISTEN_FOR.md`. Cheap, and it is the only
   test for "is this physically plausible" — the unit tests prove the maths, not
   the result.
8. ⬜ **`grid-v1` is not tagged** and the tree is uncommitted (B.7).
9. ✅ **`run_d3a.py` stays at the repo root — RESOLVED, do not relitigate.** It
   shares `run_experiment.py`'s defining property: **it spends money**. Everything
   under `analysis/` is pure analysis of already-collected data and is safe to run
   in a loop; `run_d3a.py` is not, and its budget ceiling (`--max-calls`) is a
   checked invariant rather than a comment. Folding it into
   `analysis/sensitivity.py:main()` would put an API-spending path behind a module
   whose whole contract is that it is free to re-run. The root/`analysis` split is
   the money boundary; keep it legible.

---

## B.6 Dashboard and demo kit

**Dashboard (R6) — 8 panels.** Panels 1–6 unchanged and per-model. Added this
session:
- **Panel 7 — L1 multi-model comparison** and **Panel 8 — L3 decoupling**, both
  **cross-cutting**: they sit outside the model toggle because panel 7 *is* the
  comparison between arms and panel 8 reads audio rather than the per-model
  table. They live under a new `cross` key in the payload, and
  `test_dashboard.py` now lists them separately in `CROSS_PANEL_IDS` so the
  toggle test does not wrongly demand they re-render.
- **`--sim-master`**: the sim arm is joined at build time from
  `results_sim/master_sim.csv` and re-split on `rir_key`, which fixed
  `sim2real=EMPTY`. Defaults to that path when it exists.
- Panel 8 draws WER on a **fixed 0–1 axis, never auto-scaled** — a flat curve must
  *look* flat, and auto-scaling would stretch a 0.05 wander to fill the panel.
  Half-degradation levels are suppressed entirely when the analysis refused to
  quote them.
- `build_sim2real` now returns an explained empty state when the requested model
  has no sim arm, instead of surfacing a raw `KeyError`.

Current build status (`--no-al`, Whisper not yet complete):
`[nova-3] silent_failure=ok, fingerprints=ok, sensitivity=ok, sim2real=ok` ·
`[cross-model] decoupling=ok, model_arms=EMPTY`.

**Demo kit (R9).** `Makefile` (auto-discovers `test_*.py` so a new suite is picked
up without editing it), `demo_break.py` (752 lines), `demo_al.py` (343 lines),
`requirements.lock.txt`, `test_demo.py` (22 tests, 1 failing — see B.5).
**`README.md` is now written** (128 lines) and `dashboard/DEMO.md` has been
updated — neither has been reviewed. Note DEMO.md must describe **8 panels, not
6**, and must not reference a live voice agent (R7 is out of scope).
The full sweep of every `make` target had not completed when work stopped.

---

## B.7 Repo state at the pause — READ BEFORE COMMITTING

Nothing is committed. `git status`:

```
 M SPEC.md                     M dashboard/app.css       M dashboard/app.js
 M dashboard/build.py          M dashboard/deadzone.html M dashboard/shell.html
 M analysis/al_savings.py      M test_calibration.py     M test_dashboard.py
 M test_paralinguistic.py
?? Makefile                    ?? cross_model_norm.py    ?? make_manifest.py
?? demo_al.py                  ?? demo_break.py          ?? requirements.lock.txt
?? report/writeup.md           ?? run_d3a.py
?? analysis/calibration_report.py  ?? analysis/decoupling.py
?? analysis/model_arms.py          ?? analysis/sensitivity.py
?? test_cross_model_norm.py    ?? test_demo.py           ?? test_model_arms.py
?? test_sensitivity.py
```

**✅ The mid-edit risk is largely retired.** A full sweep at this pause found
**every suite green** except the two named in B.5 (`test_demo` 1/22,
`test_dashboard` 1 remaining). Verified passing:

`test_pipeline` · `test_adapters` · `test_conditions` · `test_check_recordings`
`test_task_specs` · `test_cross_model_norm` · `test_model_arms` ·
`test_run_experiment` · `test_calibration` · `test_paralinguistic` ·
`test_analysis` · `test_layers` · `test_active_learning` · `test_design` ·
`test_model_compare` · `test_sim2real` · `test_agent_eval`

Still worth re-running before committing, since `test_sensitivity.py` and
`report/writeup.md` were being edited when work stopped:

```
for t in test_*.py; do ./.venv/bin/python $t >/dev/null 2>&1 || echo "FAIL $t"; done
```

**One result from that sweep is load-bearing for the write-up.**
`test_active_learning` passes with the banner *"active sampling reaches target
fidelity in far fewer oracle calls than random"* — on **planted synthetic
structure**, where the boundary is sharp. So the real-grid null (B.4) is **not a
broken implementation**: the machinery beats random on a surface with an
exploitable boundary and fails to beat it on the real one. That is the difference
between a broken method and a method meeting a surface it has no purchase on, and
a reader will assume the former unless it is said. The synthetic suite is the
control; cite it.

**Resume order (highest value first):**
1. **Write the compressed `report/writeup.md`** (~3,500-word main body) and fill
   its markers. This is now the single biggest remaining item and every number it
   needs is already in B.3/B.4 — no new computation required.
2. Finish the Whisper arm **at `--workers 1`** → `--rebuild` master →
   `python -m analysis.model_arms` → L1 lands and dashboard panel 7 fills.
3. Rebuild the dashboard with **no flags**; re-run `test_dashboard.py`.
4. Fix the one `test_demo.py` failure; review the generated `README.md` and
   `dashboard/DEMO.md`; run every `make` target end to end with wifi off.
5. `make_manifest.py` again on a clean tree, commit, then `git tag grid-v1`.
6. The listening pass (A.R3.5) — user's task, ~10 minutes, and the only test for
   "is this physically plausible".

**Time estimate at this pause:** ~1–2 h wall clock with agents in parallel, of
which ~45 min is the user's own attention (reading the compressed write-up,
eyeballing the dashboard, the listening pass). The mid-edit risk that widened the
earlier estimate has been retired by the green sweep above.

## B.8 Repo delta vs §13's layout

Beyond the previous log: `make_manifest.py`, `cross_model_norm.py` (+tests),
`analysis/{sensitivity,decoupling,model_arms,calibration_report}.py` (+tests),
`demo_break.py`, `demo_al.py`, `Makefile`, `requirements.lock.txt`,
`report/writeup.md`, `run_d3a.py`, and `results/audio/sweep/harsh/`.
`results_sim/` remains gitignored alongside `results/`.

---

# Appendix C: Session log — 2026-08-05 (resumed)

> Supersedes Appendix B wherever the two disagree. Everything here is read off
> artifacts regenerated during this session, not from memory.

## C.1 What closed

| item | before | now |
|---|---|---|
| Whisper arm | 1597/1760, `--workers 4` crashed on Numba | **1760/1760** at `--workers 1` |
| master table | 7040 nova-3 rows, CSV only | **8800 rows** (nova-3 + whisper), **3 failures = 0.03 %**, and `master.parquet` now actually written |
| L1 multi-model | blocked | **done** — see C.3 |
| dashboard | 8 panels wired, `model_arms=EMPTY` | **all 8 `ok`**, rebuilt on the full table |
| `test_dashboard.py` | 21/22 | **22/22** (see C.2) |
| `test_demo.py` | reported 1 failing | **22/22 — the report was stale**, fix was already in the tree |
| whole suite | 17 green + 2 known failures | **21/21 green** |
| `make` targets | never swept | `help`, `demo-check`, `demo-break`, `demo-al`, `dashboard-build` **all pass offline** |

**`pyarrow` was missing**, so `run_experiment.py --rebuild` had been silently
falling back to CSV-only and printing the reason. R4.4's DoD names
`results/master.parquet`; it exists now, and `requirements.lock.txt` was
re-frozen (65 packages) to include it.

## C.2 The last dashboard failure was a WRONG TEST, not a broken panel

`test_model_toggle_re_renders_every_panel` asserted that **no** panel is an empty
state after switching model. Once the whisper arm landed, the failure moved from
`panel-hero` (a genuine bug, fixed by the rebuild) to `panel-sim2real` — and
that one is **correct behaviour**: the simulated-RIR arm was only ever run for
nova-3, because D4 compares measured vs synthetic **RIR provenance**, not model
families, and re-running it per model would double the API spend to answer a
question nobody asked. A model with no sim arm has genuinely nothing to plot.

The test now encodes the real invariant instead of the convenient one:

- **every** panel must still *render something* after the toggle (a blank hole
  and a crashed panel are indistinguishable to someone demoing this), and
- a panel in `MODEL_CONDITIONAL_EMPTY` may be empty **only if it explains
  itself** — the assertion checks the empty state carries text.

`build_sim2real` emits *"The simulated-RIR arm was only run for nova-3, not for
whisper-base"*, so the test is not passing vacuously. The exemption set is one
named panel with the reason written next to it, not a blanket relaxation.

## C.3 L1 — the multi-model comparison (R5.7), real numbers

**Scope caveat that must travel with every number here:** the whisper arm ran on
the **10-clip AL subset**, so L1 is computed on cells *both* models ran —
n = 1757 rows per model (176 conditions × 10 clips − 3 whisper failures). This is
why nova-3's dead-zone rate reads **1.14 % (2/176)** here but **3.41 % (6/176)**
in the D1 headline table. Different clip subsets, not a contradiction — and an
unexplained 1.14-vs-3.41 in the same document would cost the reader's trust.

**The normalization audit is the validity check for the layer** — without it no
cross-model WER is quotable:

| model | WER strict | WER x-model | shift | n |
|---|---|---|---|---|
| nova-3 | 0.433 | 0.447 | **−0.014** | 1757 |
| whisper-base | 0.996 | 0.906 | **+0.090** | 1757 |

nova-3 is already word-form (the adapter disables `smart_format`/`punctuate`/
`numerals`), so its shift *should* be ~0 and is. Whisper's +0.090 is the
normalizer recovering its digit orthography, not inventing accuracy. A large
nova-3 shift would have meant the normalizer was changing more than spelling.

**The finding** (⚠️ **the dead-zone-rate and spearman columns are SUPERSEDED by
Appendix G.8** — `condition_table()` carried the same estimand mismatch as D1.
Corrected: nova-3 **0.57 % (1/176)**, ρ **−0.970** (n = 164); whisper **39.20 %**
unchanged, ρ **−0.590** (n = 171). The WER column stands as the all-clips
quantity; G.8 adds the spoke-subset figures, which move the two arms in
*opposite* directions):

| model | conds | WER | dead-zone rate | spearman(conf, WER) |
|---|---|---|---|---|
| nova-3 | 176 | 0.433 | **1.14 %** (2) | **−0.929** |
| whisper-base | 176 | 0.996 | **39.20 %** (69) | **−0.566** |

Whisper is not merely worse — it is **worse at knowing it is worse**, which is
the project's thesis restated across model families.

**Dead zones do not transfer at all:** shared 0, **Jaccard 0.000**, nova-only 2,
whisper-only 69. Pair this with D4's sim2real Jaccard of **0.00** and there are
now *two independent senses* in which a dead-zone map fails to transfer — across
RIR provenance and across model family. Quotable practitioner warning: **you
cannot borrow someone else's dead-zone map.**

**The mechanism differs, not just the magnitude** (fraction of reference words):

| model | sub | del | ins |
|---|---|---|---|
| nova-3 | 0.149 | 0.270 | 0.021 |
| whisper-base | 0.413 | 0.289 | **0.197** |

Deletions are comparable (0.270 vs 0.289); substitutions are **2.8×** and
insertions **9.4×**. Under degradation nova-3 goes quiet; **whisper invents.**

**Hallucination** (its own callout per R5.7): median hyp/ref length ratio 1.00,
p95 **2.75**, **9.9 %** of rows exceed 2× the reference length, mean foreign-token
fraction **0.528**. Whisper's WER exceeds 1.0 in two regions (g726 1.060,
rt60 0.4–0.6 1.066) — possible precisely because insertions are unbounded. The
example to quote, 3 reference words becoming 49:

> `[u02 @ rt60-1_snr-5_babble_opus-lowrate_roll-1]`
> **REF:** "call maria at"
> **HYP:** "I'm gonna go here and do a little bit of the work. You call her, you
> have a passport, you have a file. You have a file, you have a file, you have a
> file, you have a file, you have a file. You have a file."

Note the degenerate repetition loop. **WER understates this**: WER caps damage at
one error per reference word, while a 49-word hallucination handed to a
downstream LLM agent is unbounded harm. That is a second, independent argument
for why WER is not the deployment metric.

**Top divergence regions** (ranked by WER gap; whisper is the worse arm in every
one): `snr_db` 10–15 dB gap 0.638 (dead-zone rate 0 % vs 38.5 %) · `noise_type`
engine 0.627 · `snr_db` 15–20 dB 0.624 (0 % vs 58.3 %) · `rt60` 0.6–0.8 0.610
(2.8 % vs 47.2 %) · `codec` g726 0.593.

## C.4 Independent audit of the sensitivity decomposition

Everything B.3 claims was **re-derived from `results/master.csv` with plain
numpy, importing nothing from `analysis.sensitivity`** — the only evidence worth
having for a number this load-bearing.

- **Complete factorial: VERIFIED.** 5760 nova-3 babble rows = 144 × 40 exactly;
  cell-size histogram `{40: 144}`; zero missing cells, zero extras, zero
  duplicate `(cell, clip)` pairs, zero failed/non-finite rows. Every cell holds
  *literally the same* 40 clip ids. The precondition is **validated, not
  assumed** — injecting a deletion or flipping rows to `failed=True` both make
  `load_factorial` raise.
- **Exact ANOVA: VERIFIED.** A deliberately different implementation reproduced
  `V_total = 0.12658007899684545` with Δ = 0.00e+00, and every S1/ST to ≤1.1e-16.
  Centering and orthogonality checked *directly* rather than inferred from the
  sum: max |effect summed over its own axis| = 1.33e-15, max inner product over
  all 105 pairs = 1.21e-17.
- **Clip-level bootstrap: VERIFIED**, and the stated reason is empirically
  right. Re-running the bootstrap with explicit index resampling and a different
  seed matched every CI to Monte-Carlo noise. Building the **wrong** bootstrap
  (resampling clips *within* each cell, destroying the coupling) gives gap
  half-widths ~35–45 % too narrow — so the module is on the correct side of a
  choice that would otherwise have quietly manufactured significance.
- **Pre-registration rule: VERIFIED**, and it is applied *conservatively*. The
  gap CI adds S1 and ST half-widths in quadrature, which assumes independence;
  the bootstrap correlation is strongly positive (+0.86 / +0.88 / +0.68 / +0.86),
  so the published interval is **~2.5× wider than the honest one** (rt60
  half-width 0.0360 vs 0.0143). The code computes the tighter interval and
  deliberately declines to use it. Conservative in the only direction that
  matters for a pre-registered test. No branch can omit the verdict.

**One defect found and fixed.** `load_factorial`'s hole check cannot see a
**duplicate**: the fill loop assigns `W[clip, cell] = y`, so a second row for an
already-written cell silently overwrites the first and leaves no NaN behind —
the block still looks complete and the exact ANOVA decomposes whichever row
landed last. Inert on the current table (`n_rows == W.size == 5760`), but the
realizable cause is mundane (a partial re-run appending a fresh transcript for a
measured cell). Now guarded by asserting `n_used == W.size`, with a test pinning
both the raise and the negative case. **This is the project's signature failure
mode found in the project's own analysis code**, and it belongs in the write-up
beside the `apply_rir` and `_curve_degeneracy` anecdotes as a third worked
example.

**Two disclosures for the write-up** (neither changes a headline number):

1. **The ST−S1 gap carries a small upward finite-clip bias.** Clip-sampling
   noise lands disproportionately in high-order ANOVA terms (weight 0.25 for the
   4-way term vs 0.021 for a main effect), inflating ST and deflating S1.
   Measured two ways that agree: bootstrap bias +0.0041 for both `rt60` and
   `snr_db`; split-half (20 clips) gaps 0.1312/0.1156 vs full-sample
   0.1275/0.1120. Bias-corrected gaps ≈0.124 and ≈0.108, and the order-4 share
   (0.0112) is mostly noise floor. **Does not threaten the verdict** — the margin
   over the pre-set 0.020 threshold is ~5×, and the conservative CI lower bound
   (0.0915) is 4.5× it.
2. **The decomposed response is the *unweighted* mean of per-clip WER**, not
   word-weighted pooled corpus WER. Both were computed: max per-cell difference
   0.0194, `V_total` 0.12658 vs 0.12586, S1/ST shift ≤0.0033, rankings identical,
   verdict unchanged. Immaterial — but **name it in methods**, because A.R1.1
   justifies the corpus size in reference *words*, which implies the pooled
   quantity. The unweighted mean is the right estimand to pair with a clip-level
   bootstrap; that is the justification and it just needs saying.

## C.5 A correction worth keeping

Mid-session I "corrected" the headline dead-zone number from confidence **0.843**
to **0.680**, having miscounted the columns of `results/dead_zones.csv`. The
schema is `…, mic_rolloff, rt60_measured, mean_conf, conf_pct, wer, …`, and
`rt60_measured = 0.680` sits immediately before `mean_conf = 0.843` — 0.680 reads
exactly like a plausible confidence, so an off-by-one column read swaps the
*delivered reverb time* for the *headline confidence* and the result still looks
sane. The original number was right; the correction was wrong and was retracted.

**The check that settles it is an identity, not a column count:**
`gap = mean_conf − (1 − wer)` → 0.843 − 0.613 = 0.2297, reproducing the stored
`gap` field exactly. Corroborated by `demo_break.py` ("rt60 0.7s (measured
0.68s) … mean confidence 0.843") and by `results/interaction_report.txt`, which
lists the rt60 = 0.7 room (Campground Dininghall) at measured RT60 0.680.

Kept in the log deliberately: this is the same class of error the project exists
to study, committed against the project's own results file, and caught only
because a redundant arithmetic identity was available to check against. That is
an argument for storing derived fields like `gap` even though they are
recomputable — **redundancy is what makes a silent error loud.**

## C.6 What remains

1. **The compressed write-up** — the last substantive item. All numbers are in
   B.3, B.4, C.3 and C.4; no computation remains.
2. **`git tag grid-v1`** on the committed SHA, then regenerate
   `results/MANIFEST.json` so it records a clean tree (it currently reads
   `a6ece0c455af (dirty)`; 11,086 Deepgram calls, ~$3.26).
3. **The listening pass (A.R3.5)** — still nobody's ears on the degraded audio.
   Files in `results/audio/listen/` with `WHAT_TO_LISTEN_FOR.md`. ~10 minutes,
   the user's own task, and the only test for "is this physically plausible."

## C.7 Five numbers Appendix B had wrong (found while writing the report)

The write-up pass re-derived every quoted number from the artifacts rather than
from the progress log, and caught five discrepancies. **The write-up is correct;
B.3/B.4 above are stale on these points.**

1. **The sim2real gap of 12.1 points REQUIRES clip-matching, and the matching is
   load-bearing.** The sim arm ran on 10 clips; the real table has 40. Comparing
   them as-is gives **19.9 points** [−22.9, −17.1]. Restricting the real arm to
   the same 10 clips reproduces **12.1** [−15.0, −9.6] with ρ 0.873, τ 0.698,
   Jaccard 0.00 exactly. The 19.9 figure is an artifact of not matching, and the
   write-up names it as such. Anyone re-running D4 must match clips first.
   (Also: max |Δ| measured RT60 in the D4 pairing is **0.017 s**, distinct from
   the 0.019 s quoted for RIR *generation* in R2.5 — two different quantities.)
2. **ρ = −0.957 is the confidence-PERCENTILE correlation over 169 conditions,
   not 176.** Raw `mean_conf` over the same 169 gives −0.9523; the canonical
   `overall_correlation` is −0.95714, n = 169. The 7 missing conditions are the
   silent ones — they emit no words and therefore have no confidence at all.
   Stating n = 169 *strengthens* the deletion-blindness argument rather than
   weakening the headline.
   > ⚠️ **This resolution was WRONG, and Appendix G.7 replaces it.** The count
   > was right and the *computation* was still mixing populations: all 176
   > conditions were passed to `confidence_wer_shape` while n = 169 was
   > reported, so the 7 mute conditions entered as fabricated points at the
   > ideal corner. Corrected, both at n = 169: **−0.980 paired / −0.952
   > all-clips**. −0.957 was the artifact, sitting between the two honest
   > numbers. Recorded here because a partial explanation that reconciles the
   > arithmetic is the most dangerous kind — it closes the question.
3. **`digit_word` is the LEAST destroyed word class (0.361), below function
   words (0.462).** So "entities degrade faster than the transcript" is carried
   by **proper nouns (0.646) and spelled letters (0.613)**, not by numbers. The
   naive reading of the destroyed-word table gets this backwards.
4. **Competing-speech capture is not babble-specific.** Foreign-insertion
   fractions are also high for engine (0.94) and road (0.89), even though babble
   carries ~3× the insertions. The mechanism claim must be about *insertions
   under any competing source*, not about babble alone.
5. **Stale scalars corrected:** deletions **35.1 %** (not 35.6 %), **22,416**
   (not 22,411), the rt60 calibration discount is 0.81 vs **0.75** on **8,144**
   held-out words (not 0.74 / 7,980), and the clean-baseline floor is three
   proper-noun mis-spellings among **six errors across five clips** (not "three
   of five residual errors").

**Process note.** Four of these five survived multiple earlier reviews because
they were being copied forward from the progress log rather than re-read from
`results/`. The log is a *summary*, not a source. Quote artifacts.

## C.8 D4 defect: `sim2real.py` never enforced the clip half of its own premise

Found while reconciling the write-up against the artifacts, and it is the most
serious thing this session turned up.

`analysis/sim2real.py`'s docstring states the premise plainly: *"same clips, same
condition list, only the reverb ingredient swapped."* The code enforces the
**condition** half correctly — conditions are paired on **measured** Schroeder
RT60, never the Sabine target (B.2 item 5). It never enforced the **clip** half.
The real arm has **40 clips**; the sim arm has **10** (the AL subset, run small
deliberately to save API spend). So each condition's real mean WER was computed
over 40 clips and its simulated mean over a *different, smaller* set, and the
difference silently absorbed **clip difficulty** on top of the RIR-provenance
effect it exists to isolate.

Reproduced independently with plain pandas, no repo imports:

```
unmatched (as shipped)                       mean gap -0.1991   n = 176
clip-matched (real arm cut to the sim's 10)  mean gap -0.1212   n = 176
```

**The published 19.9-point gap was inflated by 7.8 points of pure confound.** The
defensible number is **12.1 points** — which is what B.4 quoted all along, so the
progress log was right and the stored artifact had drifted to the unmatched
computation.

Why this one stings: **counterfactual isolation is the entire premise of the
project** (§1) — vary one factor with everything else held constant. Comparing
different clip sets breaks exactly that, in the one layer whose job is to
validate the simulation. The failure signature is the house one: no exception,
no warning, a plausible number, and a docstring asserting the invariant the code
does not check.

**Fix in flight:** intersect the two arms' `clip_id` sets before any aggregation,
exclude failed rows, and make the mismatch **loud** — the output and the
formatted report must state each arm's clip count, the common count, and the rows
dropped. Report-and-proceed rather than raise, because a 10-vs-40 mismatch is the
*expected* situation given the sim arm was deliberately run on a subset; raise
only if the intersection is empty or implausibly small. Pinned by a test in which
the excluded clips are deliberately easier, so the unmatched and matched gaps
differ materially and only the matched one passes.

**Generalized lesson, worth one line in the write-up:** every layer that joins
two arms needs an explicit matching step with a loud mismatch report.
`analysis/model_arms.py` already got this right (`matched_arms`, `RaggedArmsError`,
"arms matched to the cells BOTH models ran; failed rows dropped") — D4 simply
never received the same treatment. When one module in a codebase has learned a
lesson, check whether its siblings have.

### C.8.1 — D4 defect RESOLVED

Fixed in `analysis/sim2real.py` (+210 lines) with three tests in
`test_sim2real.py` (8 → **11**, all green; full suite **21/21**). The RT60
condition-matching logic was deliberately left untouched.

**Corrected headline (nova-3), and every value SPEC B.4 recorded reproduces:**

| | unmatched (as shipped) | **matched (correct)** |
|---|---|---|
| mean gap | −0.1991 | **−0.12122** |
| 95 % CI | — | **[−0.15008, −0.09582]** |
| Spearman ρ | 0.8787 | **0.87326** (p = 3.2e−56) |
| Kendall τ | — | **0.69814** |
| dead-zone Jaccard | 0.00 | **0.00** (real 2, sim 1, both 0) — ⚠️ **the SET is superseded by G.8: real 1, sim 0, Jaccard still 0.00.** Level and order are bit-identical and did not move |
| n_pairs | 176 | **176** |

Clip census: real arm 40 clips, sim arm 10, **common 10** — exactly the AL subset
`u02, u05, u06, u11, u17, u22, u24, u33, u36, u39`. 5280 real rows dropped, 0
sim. Failures excluded (real 0/7040, sim 6/1760). Max |ΔRT60| **0.017 s**.
The verdict text is unchanged — **"ORDER PRESERVED, LEVEL OFFSET"** — but the
quotable sentence is now **"12.1 points optimistic," not 19.9.**
**The 19.9 figure is dead; it must not appear as a result anywhere.**

**Design decisions worth keeping:**

- **Report-and-proceed, with a raise as the floor.** A 10-vs-40 mismatch is the
  *designed* state (the sim arm was run small on purpose), so refusing it would
  delete a legitimate finding over a deliberate cost-saving choice. A partial
  mismatch restricts both arms and reports the census loudly. New
  `ClipSetMismatchError` fires only below `MIN_COMMON_CLIPS = 3`, where a
  per-condition mean stops describing an acoustic condition and starts describing
  which two utterances happened to overlap.
- **The census is unavoidable downstream.** `clip_match` is carried into the
  report, and `n_clips` / `clips_matched` are injected into `headline` itself —
  so a consumer reading only the headline still cannot quote the gap without
  seeing what it is a gap over. `results/sim2real.json` now records
  `"clips_matched": false, "n_clips": 10`.
- **`usable_rows()`** was extracted so the clip census is taken over exactly the
  rows that can contribute a measurement — a clip that only ever *failed* on one
  arm is correctly not counted as having run there.
- **New `results/sim2real.txt`** (the module previously wrote no text artifact),
  matching the sibling-`.txt` convention of `calibration.txt` / `model_arms.txt`.

**The test is the good part.** The sim arm gets a strict subset of the real
arm's clips, and the real-only clips are planted **0.30 WER points harder** — so
the two arithmetics differ enough to **flip the sign**: matched **+0.10** (the
pure planted RIR effect) vs unmatched **−0.0714** (RIR + clip difficulty). A test
that merely asserted "the gap is finite" could have passed by accident; this one
cannot.

## C.9 Write-up status — DONE, but longer than the ~3,500-word target

`report/writeup.md`, 1028 lines. **Zero `[[PENDING]]` markers** — every one
resolved from artifacts, including L1 (the whisper arm landed mid-write).

**Main body §1–§10 is ~4,550 prose words (~4,985 counting table cells and
numerals) — roughly a 17–20 minute read, against the "under 10 minutes" bar.**
Structure: §1–§10 plus Appendices **A–G** and result tables **D.1–D.10**.

Two compression passes ran (5,810 → 4,553 prose words, −21 %). The second pass
stopped after four successive rounds yielded 17, 11, 23 and 1 words: everything
remaining in the body is either a protected claim or a numbered result with its
interval, so further cuts **delete content rather than compress it**. That is a
scope decision, not an editing one, and it is the user's call.

**If a further cut is wanted, take these in order** (named by the editor, all
currently protected):
1. §6.3's pre-registration blockquote (78 words) — restates the table directly
   above it.
2. §8 item 1's final sentence.
3. §6.1's third paragraph.

**Honest read of the overshoot:** the ~3,500 target was set *before* three
mid-flight additions that did not exist then — the full L1 section with its scope
caveat, normalization audit, divergence table and hallucination callout; the four
sensitivity disclosures; and the AL split-seed robustness. Those are ~900–1,000
words of material the project genuinely produced this session. The body is dense,
not padded.

**Relocated, not deleted** (this is why the appendices grew): new **Appendix G**
(grid construction + the JOIN-1 gate probe and A/B/C table), new **D.10** (full
L2 calibration numbers), the GRID-vs-PROBE room-triplet table → D.7, L3 sweep
ranges and guard constants → Appendix E, the cross-model audit and divergence
tables → Appendix C / D.9, Sabine-vs-Schroeder detail → Appendix F.

**Deleted outright — redundancy only:** a verbatim duplicate of limitation 12,
a restatement of the clean-floor number already in limitation 3, and transitional
filler.

⚠️ **Two of the must-survive numbers below are now WRONG in the write-up** —
`0.843 @ WER 0.387` (that condition is `silence_driven`; the surviving #1 is
`rt60-0.45_snr-0_engine_g726_roll-0`, **0.829 @ WER 0.306, 0/40 silent**) and
`ρ = −0.957` (**−0.980 paired / −0.952 all-clips**). See Appendix G. The
write-up, dashboard and `results/` artifacts must be brought into agreement
before `grid-v1` is tagged (F.2).

**Verified present after compression** (normalized grep, every must-survive
item): 0.843 @ WER 0.387 · ρ = −0.957 over 169 conditions · "mostly self-aware"
· survivor bias + 35.1 % / 69.3 % / silent-about-the-worst-7 · DRR −1.000 vs
RT60 +0.800 and "a property of which RIRs were curated" · the pre-registration
verdict with `d8ddd4f` / 2026-07-27 / both gaps and CIs / the decision rule fixed
in advance · the AL null with "NO seed was confirmed" and the
`test_active_learning.py` control · both Jaccard 0.00 results and "you cannot
borrow someone else's dead-zone map" · the 3→49-word whisper transcript · prior
work with no novelty claim · Lombard named in §8 item 1 · both closing honest
sentences · all **16** limitations. Every cross-reference resolves.

**19.9 appears exactly once**, as *"a corpus difference masquerading as a
simulation gap"* — a named artifact, never a result. Correct treatment.

## C.10 Final state

- **21/21 test suites green** (test_sim2real 8 → **13**, test_sensitivity +2).
- **Dashboard**: all 8 panels `ok`, rebuilt on the full 10,560-row joined table.
- **Every `make` target passes offline**, no API key.
- **`results/MANIFEST.json`** regenerated: 11,086 Deepgram calls, ~$3.26.
  It currently records `a6ece0c455af (dirty)` — **regenerate it after the commit**
  so it names a clean tree, then `git tag grid-v1`.
- **Not committed.** The tree is uncommitted by choice; committing and tagging
  were left for the user. Note the repo is on `main`.
- **Still outstanding: the listening pass (A.R3.5).** Nobody has put ears on the
  degraded audio. `results/audio/listen/` + `WHAT_TO_LISTEN_FOR.md`, ~10 minutes.
  The unit tests prove the maths; only listening proves the *result*.
  > ✅ **DONE, and it invalidated the D1 headline.** The exemplar clips sounded
  > intelligible; the estimand mismatch (Appendix G) is what that observation
  > exposed. The last unchecked item in the spec was the one that mattered.

---

# Appendix D: ElevenLabs Scribe arm — the day-one gate (2026-08-05)

## D.1 GATE RESULT: BLOCKED on the key, zero spend

`scripts/probe_elevenlabs.py` (written, working, one clip / one call) cannot run.
The vendor rejected the credential before ever looking at the audio:

```
HTTP 400  {"type":"authentication_error","code":"invalid_api_key",
           "message":"API key must start with 'sk_'.",
           "status":"invalid_api_key_prefix"}
```

The `.env` value is **64 characters and does not begin with `sk_`**. Verified it
is not a parsing artifact — no stray quotes, no `export` prefix, and the
`DEEPGRAM_API_KEY` in the same file parses and authenticates fine. **Nothing was
spent.** Needs a fresh key from the ElevenLabs dashboard (Developers → API Keys).

## D.2 What the docs say to expect (so the probe CONFIRMS a prediction)

| question | answer | source |
|---|---|---|
| per-word confidence? | **YES — `logprob`**, inside `words[]` | API ref, `POST /v1/speech-to-text` |
| scale | **[−∞, 0]**, higher = more confident — a genuine **log-probability**, not a [0,1] score | ditto |
| batch model literal | **`scribe_v2`** (current flagship) | models overview |
| streaming? | **YES — `scribe_v2_realtime`**, websocket, ~150 ms | realtime API ref |
| streaming schema | **same per-word `logprob`**, in `CommittedTranscriptWithTimestamps` | ditto |
| `scribe_v1` | **deprecated, removed 2026-07-09** — dead as of today | changelog 2026-06-08 |
| price | **$0.22/hr batch**, $0.39/hr realtime | pricing/api |

**Two traps recorded in the probe itself:**

1. The **capabilities** page shows an abbreviated example response with **no
   `logprob`**; the full API reference does list it. So the field's real presence
   still needs one live call — documentation disagreement is exactly why the gate
   is a probe and not a doc read.
2. **`language_probability` is NOT per-word confidence.** It is a document-level
   *language-detection* score on a 0–1 scale. Mistaking it for confidence would
   assign every word in a clip the same value — a perfectly smooth, entirely
   fake confidence signal that no test would catch. The probe prints it with a
   warning label for exactly this reason.

The probe deliberately has **no `scribe_v1` fallback**: v1 is removed, so falling
back would convert one clear error into a second, more confusing one.

## D.3 Cost is ~7× cheaper than budgeted

At $0.22/hr batch and ~4 s/clip: the 10-clip subset (1,760 calls ≈ 2.0 hr audio)
is **≈ $0.43**, and the full 40-clip grid (7,040 calls ≈ 7.8 hr) is **≈ $1.72** —
against the ~$3 estimate. Cost is not a constraint on this arm; the gate is.

## D.4 A decision this opens up (needs the user)

`scribe_v2_realtime` exposes **the same per-word `logprob` over a websocket**.
The current Deepgram arm uses the **pre-recorded** endpoint, so the grid today has
**no genuinely streaming arm at all** — a framing gap the write-up handles by
labelling arms honestly. An ElevenLabs realtime arm would be the project's first
true streaming measurement with per-word confidence.

That is a **larger** piece of work than the batch arm (websocket, realtime pacing,
chunking) and it is **not** what was scoped. Default remains: **batch
`scribe_v2`, labelled batch, exactly as the Whisper arm is labelled.** Flagging
it because it is a genuine opportunity, not because it should be taken now.

## D.5 Order of operations once a valid key lands (unchanged)

1. Probe → yes/no on `logprob`, shown to the user before any adapter exists.
2. Adapter to the existing return contract. If `logprob`, transform `exp(logprob)`
   and keep it **strictly in its own scale** — never pool confidences across
   vendors (the `within_model_conf_percentile` rule the Whisper/Vosk arms follow).
3. **Normalization audit as a GATE**, before a single row enters `master`. Extend
   `cross_model_norm.py` for Scribe's orthography and show the arm's WER shift is
   **orthography, not accuracy** — the same trap that was worth 0.20–0.60 WER on
   the Whisper arm and is indistinguishable from an acoustic effect once in the
   table. Deepgram's shift was −0.014 (≈0, as predicted) and Whisper's +0.090;
   Scribe's must be explainable in the same terms.
4. Smoke-test one clip, then the 10-clip subset, verify, then decide on the grid.

---

# Appendix E: Invariant hardening — and three more real defects (2026-08-05)

A systematic pass over every loader and merge path, after three defects of the
same shape turned up in one session. **Three more latent defects were found**,
not merely guarded against. All headline artifacts re-verified byte-identical.

## E.1 The three defects

**1. The Sobol partition check was defeated by exactly what it exists to catch.**
`_check_partition` computed `worst = float(np.max(err))` then `if worst > tol:
raise`. With a NaN anywhere in the response `worst` is `nan`, and **`nan > tol`
is False** — so the guard *passes*, every Sobol index comes back `nan`, and the
report prints `sum(S_u) = nan` as though it were a measurement. **The guard's
success condition and its failure condition were indistinguishable.** This is
the single sharpest example the project has produced, because it is the failure
mode occurring *inside the instrument built to detect it*. The test now asserts
explicitly that the bare threshold **would** have passed.

**2. `rescore_cross_model` scored unknown clips against an empty reference.**
`refs.get(r["clip_id"], "")` — and `cross_model_classify_errors("", hyp)` returns
`wer=1.0, n_ref=0`. A clip-id typo or any manifest/table drift therefore produced
a clean-looking **total model failure** that dragged the arm's cross-model mean
toward 1.0, indistinguishable from a real acoustic collapse. Now raises, naming
the offending ids.

**3. `matched_arms`' equal-size check was blind to duplicates.** `common` is a
**set** of cells, so a cell appearing twice in one arm contributes one set member
and two rows. One duplicate per arm and `len(set(sizes.values())) == 1` still
holds, while `condition_table` double-weights that clip and `n_clips` merely
reads one higher. Replaced with the row-count identity `len(rows_m) ==
len(common)` per arm.

## E.2 Near-misses worth recording

- **`edit_signature`'s `... or 1`** turned a zero denominator into a `0/0/0`
  composition that reads as *"this model destroyed no words"* — the exact
  inversion of *"there was nothing to score."*
- **`split_robustness` guarded the dict field but not the SENTENCE.** The
  verdict string interpolated an unguarded `np.median(all_diffs)`, so it could
  print *"median paired difference nan"* as a measured tie.
- **Duplicate seeds.** The AL arms are deterministic in the seed, so
  `seeds=(0,0,0)` cleared the `MIN_SEEDS` gate and produced a **zero-width band
  labelled "median over 3 seeds"** — a single-seed anecdote wearing the strongest
  available form of the claim. Now rejected in both `seed_band` and
  `multi_seed_curves`.
- **`nan >= wer_hi` is False**, so an unmeasured cell silently counted as **not**
  a dead zone *and* stayed in the denominator, diluting D1's headline rate. The
  test measures the exact dilution: 0.333 → 0.311.

## E.3 The structural fix

`write_master` now **refuses to write** a table violating one-row-per-
`(clip_id, condition_name, model)` at all. That is the origin of this entire
family, and it is cheaper to refuse writing than to guard every reader — it also
leaves nothing on disk to be misread later. `load_manifest` refuses duplicate
ids (SPEC §12's unfixable bug: the later ground truth silently wins). `run_grid`
asserts `len(out) == len(plan)` — a `None` row filtered out would shorten the
table *exactly where harsh cells die*. `ResultCache` counts and **reports**
superseded rows: report-and-proceed, since last-write-wins is deliberate there.

23 guards, 9 new test functions across 7 suites. Every guard raises with the
count identity, the worst offenders, and the likely cause. Every test constructs
the violating input, asserts the raise, **and carries a negative control** so the
guard is pinned to its violation rather than to some incidental property of the
fixture.

## E.4 Regression evidence

Byte-identical after the pass: `sobol.json`, `sobol_5factor.json`, `screen.json`,
`model_arms.{json,txt}`, `l3_decoupling.json`, `calibration.{json,txt}`.
Headline numbers unchanged: rt60 S1 0.3466 / ST 0.4741, snr_db 0.3905 / 0.5025,
`sum(S_u) = 1.0`; nova-3 WER 0.433 / dead-zone 1.14 %, whisper 0.996 / 39.20 %;
ECE 0.0507 → 0.0346 → 0.0077. **21/21 suites green.**

## E.5 The generalized lesson

Every one of these six findings has the same shape: **a guard whose failure mode
is silence.** `nan > tol`, `nan >= wer_hi`, `... or 1`, a set-based size check, a
`.get(key, "")` default, a deterministic seed repeated three times. In each case
the degenerate input produced not an error but a *plausible value* — and in two
cases the plausible value was the more flattering one (no dead zone; no destroyed
words). When writing a guard, ask what it returns for the degenerate input, not
just for the good one.

---

# Appendix F: Closing decisions (2026-08-05)

## F.1 Write-up length — SETTLED at ~14 min. Do not relitigate.

`report/writeup.md` main body is **3,595 prose words (~14 min)** against an
original "under 10 minutes" target. **The user reviewed this and accepted it as
is.** Three compression passes took it 5,810 → 3,595 (−38 %), relocating ~1,135
words into appendices rather than deleting them.

**Why 3,000 was not reachable, measured rather than asserted.** 1,828 of the
3,595 words are protected content: the 16 limitations (393), the headline +
survivor-bias counter-argument (311), the DRR block + "which RIRs were curated"
(253), the trap-function paragraph (239), the AL null + its control (200), both
Jaccards + "cannot borrow someone else's dead-zone map" (101), the no-novelty
lineage (96), the two closing honest sentences (94), the pre-registration
blockquote (88), the abstract headline (53). The remaining 1,767 carry the
method, the factor table, and every claim's number-and-interval. Reaching 3,000
would mean cutting ~600 from *that* — a third of the surviving results text.
Saturation was tested by rewriting §6.1, §6.3 and §6.6 from a blank page instead
of editing; each fresh attempt yielded only 15–20 words.

**The general finding:** "under 10 minutes" and "every claim keeps its number and
interval, plus 16 limitations in the body" are **incompatible constraints**, not
an editing failure. If a future session wants it shorter, the honest move is a
**one-page executive summary prepended** to the existing document (the fast
reader stops there, the full text survives intact) — *not* another compression
pass. A fourth pass will only delete numbers.

## F.2 `grid-v1` tag — STILL HELD, now for a second reason

The tree was clean at **`46614d4`** and `results/MANIFEST.json` records that SHA
with `dirty: false` (11,086 Deepgram calls, ~$3.26, all assets SHA-256'd). The
tag was deliberately **not** placed: `grid-v1` should mark a *validated* grid,
and A.R3.5 — the listening pass — is the only check that the degraded audio is
physically plausible. The unit tests prove the maths, not the result.

**Update: the listening pass has now run, and it did not return clean.** It
found the estimand mismatch (Appendix G), and **the headline changed** — dead
zones 6 → 2, the #1 condition reclassified, ρ restated. So the hold stands for a
second and stronger reason: `grid-v1` must not be placed until the **write-up,
the dashboard and the `results/` artifacts all agree** on the corrected numbers.
A tag naming a state where the three disagree is worse than no tag — it makes an
inconsistency look ratified.

Order: repair the artifacts → rebuild the dashboard → reconcile
`report/writeup.md` (C.9 lists the two stale numbers) → regenerate
`results/MANIFEST.json` on the clean tree → **then** tag.

## F.3 State at close

**8 commits, clean tree, 21/21 suites green.**

| piece | commit |
|---|---|
| conventional package layout (pure renames) | `a5d39f6` |
| D4 clip matching — 7.8 pts of confound | `a5527aa` |
| analysis layers (exact Sobol, L1, L2/L3) | `8893a97` |
| demo kit, dep lock, ElevenLabs gate | `6061b9a` |
| path/invocation references | `06ba022` |
| invariant hardening + 3 more defects | `a3f92f2` |
| SPEC appendices C/D/E | `7b587c0` |
| the write-up | `46614d4` |

**Open, and each needs a human:** the ElevenLabs `sk_` key (Appendix D — probe
ready, zero spent), the listening pass, and then the tag.

> **Update:** the listening pass is **done** — see Appendix G. It cost the
> headline finding and bought a better one. Remaining: reconcile the write-up,
> dashboard and artifacts with G's corrected numbers, then tag (F.2); and the
> ElevenLabs key, unchanged.

---

# Appendix G: The estimand mismatch — found by ears, not by tests (2026-08-05)

> **The listening pass (A.R3.5) — the last unchecked item in the entire spec —
> invalidated the headline finding.** Everything below follows from that one
> user action. Supersedes B.4's D1 paragraph, C.3's L1 table and C.7 item 2
> wherever they disagree.

## G.1 The defect: two averages over different populations, subtracted

In `deadzone/analysis/confidence_gap.py:per_condition_table()`, per-condition
`mean_conf` was averaged over only the clips that **produced words**, while `wer`
was averaged over **all** clips — including clips that returned an **empty
transcript**, which contribute WER 1.0 and *no confidence at all* (no words, no
per-word confidences, nothing to average).

`add_gap_metrics` then computed `gap = mean_conf − (1 − wer)` across that seam.
`find_dead_zones` ranked on the gap. `model_compare.dead_zone_flags` used the
all-clips WER for **membership** too, so the shaded dead-zone quadrant — the
project's hero panel — inherited it.

For the #1 condition this paired **confidence over 30 clips with WER over 40**.

Neither average is wrong on its own. `mean_conf` over speaking clips is the only
`mean_conf` that exists; `wer` over all clips is the correct corpus WER. The
defect lives entirely in the **subtraction**, and it has the house signature:
clean arithmetic, correct row count, no NaN, no warning, a plausible number.

## G.2 How it was found — and this is the point

The user ran the listening pass (A.R3.5) and reported that the dead-zone exemplar
clips **sounded intelligible**. Checking per-clip data showed the exemplar clip
`u02` was transcribed **perfectly in four of the six dead zones**. That is what
prompted the estimand check.

**No test caught this. The audio did.** A.R3.5 has said from the beginning that
the unit tests prove the maths and only listening proves the *result*. That is
now demonstrated rather than asserted, and it is the single strongest sentence
available for the write-up's methods section: the last item on the checklist,
the one with no code behind it, is the one that found the headline error.

## G.3 Scale

- **116 / 176** conditions had **≥ 1** silent clip.
- **2,210 / 7,040** nova-3 rows (**31.4 %**) are silent, spanning **123**
  conditions — 116 partial plus the 7 that are silent on every clip.
- Mean gap inflation **+0.109**; **max +0.524**.

An inflation of that size is not a rounding artifact. It is comparable to the
entire published mean gap (0.256).

## G.4 The fix: publish BOTH pairings, never silently one

The correction is *not* "use the right one." It is to make the mismatched
quantity impossible to produce by accident and impossible to quote without a
name:

- `wer_spoke` / `wer_all_clips` — both stored, both published.
- `gap_spoke` / `gap_all_clips` / `gap_inflation` — the difference between the
  two pairings is itself a first-class field, so the size of the effect travels
  with the number.
- `n_silent` / `n_spoke` / `silent_frac` per condition, so a reader can always
  see which population a row describes.
- `gap` remains as an **alias of `gap_spoke`**; the mismatched quantity exists
  **only** under an explicit name (`gap_all_clips`). The default is now the
  same-population comparison, and asking for the other one is a deliberate act.
- `dead_zone_flags` and `confidence_wer_shape` take **`wer_key=`** — the estimand
  is named **at the call site**, so the choice is visible in the calling code
  rather than buried in a table schema.

This is the same design move as C.5's argument for storing derived fields:
**redundancy is what makes a silent error loud.** `gap_inflation` is
recomputable; storing it is what makes the mismatch legible.

## G.5 The new taxonomy — three categories, and dead zones are a *view* over it

New `condition_flags()` / `classify_conditions()`. `find_dead_zones()` is now a
thin view over `classify_conditions()`, so **you cannot obtain dead zones without
also being handed the other two categories** — a structural fix in the spirit of
E.3 (refuse to produce the misleading artifact rather than guard every reader).

| category | meaning |
|---|---|
| `dead_zone` | confidently **WRONG** — the model emits words and they are bad |
| `silence_driven` | the apparent gap came from the estimand mismatch |
| `mute_zone` | **empty transcript on EVERY clip** (nova-3: 7, whisper: 5) |

**A mute zone is deliberately NOT a dead zone.** Confidently wrong and entirely
absent are different mechanisms with different fixes — and, decisively, **a
confidence-based monitor cannot see a mute zone at all**: there is no confidence
to be low. Folding the two together would have hidden the one failure class the
project's proposed early-warning signal is structurally blind to. This is the
same hole L2's deletion-blindness analysis quantified (B.4), now surfaced at the
level of whole conditions instead of individual words.

## G.6 Results — D1

- **Dead zones: 6 → 2 of 176** (3.41 % → **1.14 %**).
- **New #1 `rt60-0.45_snr-0_engine_g726_roll-0`:** confidence **0.829** at WER
  **0.306**, **0/40 silent** — it needs no asterisk. The headline survives; it
  is smaller and it is clean.
- **Old #1 `rt60-0.7_snr-20_babble_opus-lowrate_roll-1` → `silence_driven`.**
  Gap **+0.230 → +0.025**. On its 30 speaking clips the model was **81.8 %
  accurate at 0.843 confidence** — i.e. **well calibrated**. The published
  headline dead zone was a condition where the model was behaving correctly.

## G.7 A SECOND instance in the same file

`overall_correlation` passed **all 176** conditions to `confidence_wer_shape`
while **reporting n = 169**. The 7 mute conditions have no confidence, land at
percentile 0, and carry WER 1.0 — **seven fabricated points sitting exactly at
the ideal corner of a negative correlation**, inflating the very statistic used
to argue the model knows when it is failing.

Corrected, both at n = 169: **ρ = −0.980 paired** / **−0.952 all-clips**.

**The published −0.957 was the artifact of those 7 points** — it sat between the
two honest numbers, which is precisely why nothing looked wrong. Note that C.7
item 2 already investigated this number, concluded "n = 169, not 176," and
recorded that as the resolution; the count was right and the *computation* was
still mixing populations. **A partial explanation that reconciles the arithmetic
is the most dangerous kind of resolution**, because it closes the question.

## G.8 Propagation — both found by the fix, both repaired

**L1 — `deadzone/analysis/model_arms.py:condition_table()` had the identical
mismatch.** Corrected:

| model | dead-zone rate | spearman(conf, WER) | silent rows | mute conds |
|---|---|---|---|---|
| nova-3 | 1.14 % → **0.57 % (1/176)** | −0.929 → **−0.970** (n = 164) | 431/1757 | 12 |
| whisper-base | **39.20 %** (unchanged) | **−0.590** (n = 171) | 336/1757 | 5 |

**The correction moves the two arms in OPPOSITE directions — and that is a
finding, not a bookkeeping note.** Restricting to the spoke subset **lowers**
nova-3's WER (0.433 → 0.307) but **raises** whisper's (0.996 → **1.128**).
Whisper's silent clips score exactly 1.0, so removing them removes its *best*
rows: its speaking clips hallucinate **past** 1.0. **One model goes quiet under
stress; the other invents.** This is C.3's sub/del/ins mechanism table
(insertions 9.4×) restated as a single number, arrived at independently.

> **A wrong guess, corrected by measurement, kept on the record.** The first
> explanation offered for the opposite-direction move was accuracy-clipping.
> It is wrong: `dead_zone_flags` thresholds WER **directly** and does not clip.
> The real cause is unbounded insertions. Recorded because the plausible
> mechanism and the true mechanism predicted the same sign, and only the
> measurement separated them.

**D4 — `deadzone/analysis/sim2real.py`: LEVEL and ORDER are bit-identical and
did not move.** Correctly so: they contain no confidence term. Mean gap
**−0.12122**, CI **[−0.15008, −0.09582]**, Spearman **0.8733**, Kendall
**0.6981** — all unchanged from C.8.1. Only the **dead-zone set** moved: real
**2 → 1**, sim **1 → 0**, **Jaccard still 0.00**, and the sim arm now invents
**0** false dead zones (was 1).

*A defect fix that changes some numbers and provably not others is the good
case.* The unchanged half is evidence the fix was scoped correctly.

D4's sets remain scoped to the **10-clip common subset** (C.8.1) and do **not**
coincide with D1's 40-clip table. Now stated in the payload and in the report
text, not just in this log.

**Cross-check that costs nothing and is worth a lot:** D4's surviving real dead
zone is **the same condition** as L1's surviving nova-3 dead zone, computed
through an independent path. Two layers, two code paths, one answer.

## G.9 `n_hyp_words` comes from the ALIGNMENT, not from the confidence list

`n_hyp_words` is derived as `n_ref − n_del + n_ins` from the edit alignment,
**not** as `len(word_confidences)`. The two disagree on **225 / 8,797** rows,
because vendor confidences are per **raw** token while edits are over
**normalized** tokens. Using the confidence-list length as the word count would
have silently reintroduced a population mismatch inside the fix for a population
mismatch.

## G.10 A new measured finding: two conditions the model cannot tell apart

Independent of the defect, and quotable on its own.

| | condition | room / mechanism | WER |
|---|---|---|---|
| **A** | `rt60-1_snr-20_babble_none_roll-0` | Shower, **DRR −10.02 dB**, SNR 20 dB | **0.1123** |
| **B** | `rt60-0.2_snr-0_babble_none_roll-0` | Restaurant, **DRR +16.90 dB**, SNR 0 dB | **0.1301** |

Paired over the **same 40 clips**: difference **−0.0178**, 95 % CI
**[−0.0654, +0.0310]** — spans zero. Exactly equal per-clip on `u40`, `u26`,
`u21`, `u10`. Two single-degradation conditions at opposite extremes of two
different axes are **statistically indistinguishable to the model**.

A human is not close to indifferent between them: **B** is dramatically harder
(informational masking — competing speech captures attention), **A** is easy
(the precedence effect fuses the reflections). Practical claim, and it is the
sharpest one-sentence argument in the project for why the instrument exists:

> **You cannot QA a voice agent by listening to it.**

New demo set at `results/audio/demo/` — blind filenames, a **sealed
pre-registered prediction**, and the key held separately.

**Discipline, stated so nobody over-claims it later: the human half is n = 1 and
unblinded.** It is an **intuition pump, never data**. The measured half — the
paired 40-clip model-side comparison with its CI — is the result. Anyone
presenting this must keep the two halves separate out loud.

## G.11 The generalized lesson

This is the same family Appendix E documents — *a guard whose failure mode is
silence* — with a new member:

> **Two averages over different populations, subtracted.**

Neither average is wrong. There is no degenerate input, no NaN, no `or 1`, no
`.get(key, "")`. The row counts are right. The failure is entirely in the
**join between two correct quantities**, which is a strictly harder thing to see
than a bad value.

**The rule to add to the checklist:**

1. When two quantities are **subtracted or compared**, assert they were computed
   over the **same rows** — and make the assertion executable, not a docstring
   (C.8's lesson, one level deeper: D4's premise was violated across *clips*,
   D1's across *rows within a clip set*).
2. **Name the estimand at the call site** (`wer_key=`), so a reader can see which
   population a number describes without reading the producer.
3. When both pairings are meaningful, **publish both** plus their difference.
   Choosing one silently is how the mismatch survived review.

And the meta-lesson, which is the one the write-up should carry: **the project's
signature failure mode has now been found three times inside the project's own
analysis code** — `load_factorial`'s duplicate-overwrite (C.4), `sim2real`'s
unenforced clip premise (C.8), and this. Every time, the instrument built to
detect silent failure was itself failing silently. The only thing that caught
*this* one was a human listening to the audio.
