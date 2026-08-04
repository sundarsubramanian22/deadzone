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

- [ ] **R2.4 — ffmpeg + the codec path (there is a live problem here — read this).**
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

- [ ] **R2.5 — Synthetic RIRs for the sim2real leg → `data/rirs_sim/`.**
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
