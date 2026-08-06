# Deadzone: mapping where a streaming ASR model fails *silently*

A controlled-degradation study of Deepgram Nova-3 across 176 acoustic conditions, with
the confidence–accuracy gap as the headline measurement.

*Sundar Subramanian · grid run 2026-08-05 · `run-20260805T070146Z-6a77c4`*

> Sections 1–10 are the argument. Derivations, mechanism detail and full tables are in
> the appendices and are not part of the read.

---

## 1. Abstract

**Across 176 controlled acoustic conditions, Nova-3's word confidence tracks its own error
rate almost perfectly (Spearman ρ = −0.957) — and is nonetheless overconfident in 92 % of
them (mean gap 0.256), with 6 of 176 (3.41 %) qualifying as genuine dead zones. The worst
reports mean word confidence 0.843 at WER 0.387.**

The model is not blind, it is *mostly* self-aware, and that makes the residual 3.4 %
dangerous: a system whose "ask the caller to repeat" threshold was tuned on average behaviour
will trust it precisely where it should not. The same grid — a **complete 4 × 4 × 3 × 3
factorial**, so its Sobol decomposition is *exact* rather than sampled — also gives typed
failure fingerprints, a **confirmed** pre-registered `rt60 × snr_db` interaction, the
mechanistic result that reverb damage is monotone in **direct-to-reverberant ratio, not RT60**
(ρ = −1.000 vs +0.800), and a **null** on active learning reported as a null. And a dead-zone
map does not transfer, in two independent senses: simulated RIRs rank conditions well
(ρ = 0.873) yet recover **none** of the real dead zones (Jaccard 0.00), and neither does a
second model family (Jaccard 0.00).

---

## 2. Motivation

Aggregate WER hides three things a deployment needs: *which* conditions break the model,
*what kind* of error each produces, and *whether the model knows it is failing*.

The third matters most and is least reported. A model that is wrong and knows it is an
engineering problem — ask the caller to repeat, fall back, escalate. A model that is wrong
and confident is a silent failure: the system commits and the error propagates. So the
question is not "how much does the model break" but **"does it know it is breaking, and where
does that self-knowledge fail?"** Answering it needs counterfactual isolation, which field
recordings cannot give: in the wild the mic, placement, noise and codec all move at once.
Isolation, not fidelity, is the goal; the gap to reality is measured (§7), not hidden.

---

## 3. Prior work — this is a well-trodden genre

**The controlled-degradation ASR testbed is an established genre and nothing in the method
here is novel.** The lineage, before any of my own results and annotated in Appendix I:
**WildASR / "Back to Basics" (2026)**, the closest neighbour in method and framing; **Speech
Robustness Bench**; **"When Denoising Hinders" (2026)**, which already publishes the
perception/recognition mismatch hypothesis; the far-field lineage of **Ko et al. 2017**, **Kim
et al. 2017**, **REVERB** and **CHiME**; **Scheibler et al. 2018 (pyroomacoustics)**, the
simulator used for §7; and **Carlini & Wagner** on audio adversarial examples.

**The delta is modest and specific — a lens, not a method:** the confidence–accuracy gap per
condition rather than WER per condition; typed fingerprints naming which *class* of reference
word dies, so each condition implies a fix; an active-learning surrogate, which here returns a
null (§6.5); and a **commercial streaming model that exposes per-word confidence**, where the
literature overwhelmingly uses Whisper, Conformer and wav2vec — none of which expose one, so
the question is unanswerable on them.

---

## 4. Method

| factor | type | levels | realizes |
|---|---|---|---|
| `rt60` | continuous | 0.2 – 1.0 s | reverb / talker distance, via **measured** RIRs |
| `snr_db` | continuous | 0 – 20 dB | additive noise level |
| `noise_type` | categorical | babble, engine, road | noise character (real recordings) |
| `codec` | categorical | none, g726, opus-lowrate | transmission channel |
| `mic_rolloff` | continuous | 0 – 1 | cheap-microphone frequency response |

**Realism principle: every ingredient is real, only the assembly is controlled** — 16 measured
RIRs (T20 RT60 **0.193 – 1.011 s**), DEMAND noise, real ffmpeg codec round-trips, with
composition order physically motivated and therefore a constant, not a knob (A.4, Appendix F).
**The narrowband codec is G.726, not AMR-NB** — stock ffmpeg ships AMR-NB decode-only — and
keeping *two* paid off: `g726` produces substitutions, `opus-lowrate` deletions (§6.2).

**The trap functions.** Three functions are correctness-critical in the same way: each
produces **clean-looking garbage** if subtly wrong — no exception, plausible audio, plausible
numbers. `mix_at_snr` computes SNR over **active-speech energy**, not whole-file power, which
the silent pad deflates by a clip-dependent amount, making it a per-clip confound rather than
a constant (requested vs delivered agreed to **0.01 dB** at the JOIN-1 gate).
`classify_errors` returns **typed edits** and normalizes both sides identically; apostrophe
handling alone caused 4 of 11 clean-condition errors, and such offsets land in every cell,
indistinguishable from a dead zone. And `apply_rir`, the one I got wrong first: it must trim
the RIR's direct-path delay — else a pure alignment artifact reads as "reverb causes
deletions" — *and* renormalize over the input's **active region**. My first version used
whole-file RMS, which the reverb tail inflates by dumping energy into the formerly-silent pad,
so the renormalizer scaled the signal *down* and `mix_at_snr` delivered a worse SNR than
requested **by an amount that grew with RT60**. No error, plausible audio, and a WER curve
that looked like a real reverb effect. A test that reconstructed the mix and asserted
delivered SNR against the request caught it (**Appendix A**). Three later recurrences of the
same shape (Appendix B; §6.7; A.5 — one found by audit in this project's *own* analysis code)
are named rather than quietly fixed, because the shape is the point.

---

## 5. Experimental design

**Corpus.** 40 domain-neutral utterances, one speaker, mono PCM in one sitting at fixed mic
distance and level, loaded with entity stress cases (H.2). **363 reference words per
condition**, putting the binomial standard error on a per-condition WER of 0.5 at ~2.6 points.
Every clean-condition error was adjudicated *by ear*, the reference never edited to match the
model.

**The grid.** A 13-call probe reallocated the design before it ran: **SNR alone barely moves
the model** while **the damage is an interaction** (Appendix G). What ran is a **complete
4 × 4 × 3 × 3 factorial** over `rt60 × snr_db × codec × mic_rolloff` on babble (144 cells)
plus a 32-cell engine/road arm — **176 conditions × 40 clips = 7040 rows, 0 failures, 394 s,
≈ $2.52**, per-condition WER spanning 0.006 – 1.000. That reallocated where we *sample*, not
what we *predicted* (§6.3). `snr_db` stops at 20 dB because the corpus's inherent SNR is
~25–28 dB, and nothing downstream was trusted until the JOIN-1 gate passed **9/9** (Appendix
G).

**Sensitivity is computed *exactly*.** A complete factorial with equal cell counts admits a
finite variance partition, so §6.3's Sobol indices are an **exact functional-ANOVA
decomposition of the measured grid**, not a Saltelli estimate (`sum(S_u) = 1.000000000000`) —
strictly stronger than the planned Saltelli-on-a-GP, which would have inherited both the
surrogate's bias and the sampler's variance. **CIs bootstrap the 40 clips**, not the cells.
The grid-derived Plackett–Burman screen costs 0 fresh API calls, all four factors survive,
and `SCREEN_CAVEAT` is quoted verbatim in **D.5** (with D.6 on why aliasing cannot bind here).

---

## 6. Results

### 6.1 D1 — the silent-failure map (headline)

**Reported first because it is the majority behaviour: Nova-3 largely does know when it is
failing.** Spearman between its within-model confidence percentile and its WER is **−0.957**
across the **169 of 176** conditions that returned any words. The danger is the residual:
**overconfident in 92 % of them** (155/169; mean gap `mean_conf − (1 − WER)` = **0.256**),
with **6 of 176 (3.41 %)** meeting the dead-zone criterion. Ranked #1: **rt60 0.7 s, SNR
20 dB, babble, opus-lowrate, rolloff 1.0 → mean word confidence 0.843 at WER 0.387** (n = 40
clips, 363 reference words). All six sit **mid-range, not at the harsh end** (D.1) — at the
harsh corners the model does the safe thing and emits nothing at confidence 0, so silent
failure lives in exactly the region a deployment considers acceptable. **The danger is not
that the model is blind but that it is mostly self-aware**, so a system calibrated on average
behaviour trusts it precisely where it should not.

**The strongest argument against this finding, stated unprompted.** `mean_conf` is
**survivor-biased**: deletions carry no hypothesis word and therefore no confidence, and
deletions are **22,416 words = 35.1 % of the reference and 69.3 % of ALL errors** — so a
perfectly calibrated confidence converges on emitted-word accuracy and overstates reference
recovery by **0.254**, while the **7 of 176 conditions that returned an empty transcript on
every clip** contribute zero words to any confidence statistic (D.11). The gap metric
therefore partly measures an artifact of how confidence is defined. That does not rescue the
deployment: a consumer reading `mean_conf = 0.843` cannot know 35 % of the utterance never
entered the denominator, so the *operational* hazard is real where the *statistical*
attribution is contestable — and the fix, an emission-rate signal alongside confidence, no
vendor exposes (limitation 7).

### 6.2 D2 — failure fingerprints, and the fix each implies

Across the 63,888 reference words scored: **deletions 0.351**, substitutions 0.136,
insertions 0.020 — deletion is not one mechanism among several, it is *the* failure mode. The
deletion/substitution split is the actionable part (D.2): `snr_db` (+0.344), `mic_rolloff`
(+0.264), `rt60` (+0.212) and `opus-lowrate` produce **deletions** → front-end fixes only;
`g726` and `road` produce **substitutions** → entity-aware decoding and boosting; `engine`
(−0.127) and `codec = none` correctly emit **NO FIX**.

**Entities degrade faster than words:** destroyed-word rate **0.646 for proper nouns against
0.361 for digit words**, and **entity error rate 0.633 against WER 0.511** (D.3). Insertions
under babble are a separate mechanism, **92 % foreign tokens** — competing-speech capture, not
confusion, so the fix is target-speaker extraction (D.3b).

### 6.3 D3a — sensitivity, the pre-registration verdict, and DRR

**Exact Sobol indices** (144-cell factorial, 5760 transcriptions; ±95 % clip-bootstrap
half-widths):

| factor | S1 | ST | ST − S1 | 95 % CI on the gap | sig |
|---|---|---|---|---|---|
| `snr_db` | 0.391 ± 0.031 | 0.503 ± 0.027 | **0.112** | [0.072, 0.152] | YES |
| `rt60` | 0.347 ± 0.024 | 0.474 ± 0.027 | **0.128** | [0.091, 0.164] | YES |
| `mic_rolloff` | 0.099 ± 0.013 | 0.183 ± 0.020 | 0.084 | [0.060, 0.107] | YES |
| `codec` | 0.023 ± 0.003 | 0.065 ± 0.009 | 0.042 | [0.032, 0.052] | YES |

First-order terms carry 0.860 of the variance, second-order 0.067; the **ST − S1 gap is the
primary interaction evidence**, S2 direction only (D.6).

**Pre-registration: CONFIRMED.**

> We pre-registered `rt60 × snr_db` as a genuine two-way interaction (SPEC §5, committed
> **`d8ddd4f`, 2026-07-27**, before any audio existed), under a rule fixed in advance:
> confirmed **iff** both factors' ST − S1 gaps exceed 0.020 with the 95 % bootstrap CI
> entirely above it, **and** the pair ranks first in S2. Measured: **0.128 [0.091, 0.164]**
> for `rt60`, **0.112 [0.072, 0.152]** for `snr_db`, S2(`rt60`, `snr_db`) = **0.034 ± 0.006,
> rank 1/6** (direction only) — **CONFIRMED**, ~5× clear. Reverb and noise compound.

**The best mechanistic finding: the damage is monotone in DRR, not RT60.** The `rt60` marginal
is **non-monotonic** — 0.2026 → 0.6359 → 0.4495 → 0.7581, a dip at 0.7 of depth **0.1864
[0.1574, 0.2142]** — because each level is delivered by the **nearest measured RIR, a
different real room**, and RT60 says nothing about how much direct sound reaches the mic.

| requested | room | measured RT60 | DRR dB | C50 dB | marginal WER |
|---|---|---|---|---|---|
| 0.2 | Restaurant | 0.193 | **16.90** | 28.10 | 0.2026 |
| 0.45 | Bar | 0.474 | −2.05 | 10.22 | 0.6359 |
| 0.7 | Campground Dining | 0.680 | 4.26 | 10.03 | 0.4495 |
| 1.0 | Shower | 1.011 | **−10.02** | 2.12 | 0.7581 |

`spearman(DRR, WER) = −1.000` against `spearman(RT60, WER) = +0.800`: **reverb benchmarks
parameterised by RT60 alone will mis-rank conditions.** The conclusion is sharper still —
**non-monotonicity along `rt60` is not a property of a response surface at all: each request
indexes an unrelated real room, so whether a dip exists is a property of which RIRs were
curated, not of reverberation time.** Re-sample the axis and it moves, which is why **0 of 6**
surrogate-proposed cells reproduced under real oracle calls and why my own explanation for
that was itself **falsified** (D.7; the six are nowhere presented as measured surprises). A GP
given `rt60` as a *continuous* coordinate assumes a smoothness the instrument lacks; the
defensible coordinate is **DRR**.

### 6.4 L2 — learned confidence calibration

Grouped by **condition** and never by word — a random word split leaks, and the symptom is a
*better* ECE — **a feature-conditioned calibrator cuts ECE from 0.0507 raw to 0.0077**,
against 0.0346 for a global temperature. **Above rt60 = 0.7 reported confidence must be
discounted by ~0.07** to become a calibrated probability (0.81 reported vs 0.75 observed,
8144 held-out words), above `mic_rolloff` 0.5 by ~0.06 (D.10). The layer inherits §6.1's blind
spot: a calibrated confidence can only describe words the model emitted, so this one is fit on
the 169 conditions that emitted any and is **silent about the worst 7**.

### 6.5 D3b — active learning: a NULL, and the null is robust

Does straddle (boundary-seeking) acquisition map the failure boundary in fewer oracle calls
than random at equal budget? **No savings claim is supported: the `boundary_rmse` target
0.162 was reached by 2 of 8 active seeds and 4 of 8 random seeds inside the 45-evaluation
budget**, so median evals-to-target is `inf` for both arms and the budget is reported rather
than a ratio. Random matches or beats straddle throughout and the winner **flips between
splits** (median paired difference **+0.003**). **The whole target curve is published (D.8) so
the headline row cannot be cherry-picked**, the threshold was not massaged, and **all 8 seeds
ran against the surrogate oracle: NO seed was confirmed end-to-end against the live API.**

**This is a method meeting a surface it has no purchase on, not a broken implementation, and
the control that establishes that is `tests/test_active_learning.py`, which passes on *planted
synthetic structure* with the banner "active sampling reaches target fidelity in far fewer
oracle calls than random."** The same machinery beats random when the boundary is sharp and
fails here — a reader will assume a bug unless this is said (D.8).

### 6.6 L1 — multi-model comparison

**Scope first, because it explains an apparent contradiction.** The Whisper arm ran on the
10-clip AL subset, so L1 uses the **n = 1757 rows per model** both arms ran — which is why
nova-3's dead-zone rate reads **1.14 % (2/176)** here but **3.41 % (6/176)** in §6.1. The arms
also disagree about number *orthography*, not acoustics, an offset indistinguishable from an
acoustic effect, so **both** were re-scored once through one published normalizer (C).

**The finding is not that Whisper is worse — it is that Whisper is worse *at knowing* it is
worse.** Within-model confidence-vs-WER shape (percentiles only; scales are not comparable
across families): **nova-3 ρ = −0.929 vs whisper-base ρ = −0.566**, dead-zone rate **1.14 % vs
39.20 %**, and **the two models do not fail silently in the same places: shared dead zones 0,
Jaccard 0.000.** With §7's sim2real Jaccard of 0.00 that is two independent senses in which a
dead-zone map fails to transfer — across RIR provenance and across model family. **You cannot
borrow someone else's dead-zone map.**

**The mechanism differs, not just the magnitude:** whisper-base runs insertions **9.4×**
nova-3's rate per reference word and is worse in all eight divergence regions (D.9). Nova-3
goes quiet; **Whisper invents**:

```
[u02 @ rt60-1_snr-5_babble_opus-lowrate_roll-1]   3 ref words -> 49 hyp words
REF: call maria at
HYP: I'm gonna go here and do a little bit of the work. You call her, you have a
     passport, you have a file. You have a file, you have a file, you have a file,
     you have a file, you have a file. You have a file.
```

**A WER of 0.996 understates this**: WER caps damage per reference word, whereas an invented
sentence fed to a downstream agent is unbounded harm — which is why Whisper's WER exceeds 1.0
in two regions (D.9).

### 6.7 L3 — paralinguistic vs lexical: the monitor alarms *early*

An agent monitoring its own audio health with cheap features is assuming they track lexical
accuracy. Six single-factor sweeps say they do not. **Two return a DECOUPLED
verdict and both point the same way:** under increasing reverb **`f0` collapses at
rt60 ≈ 0.62 while WER only halves at ≈ 0.85**, and under falling SNR **`rms` collapses at
≈ 4.46 dB while WER halves at ≈ 6.61 dB**. **The paralinguistic stream LEADS, so a
feature-based monitor would alarm *before* the transcript measurably degrades — the opposite
of the failure mode this layer was designed to find**; such a monitor is conservative, not
blind. The other four hit a lexical floor or yield no supportable threshold, **a power
limitation and not a finding of stability**. The layer also produced **this project's second
worked example of its signature failure mode** — a degeneracy that let
`compare_degradation_rates` do real arithmetic on a meaningless curve and report it as a
finding, now refused rather than invented (E).

### 6.8 L4 — voice-agent layer: NOT BUILT

**The live voice agent was scoped out and not built.** `deadzone/agent_eval.py` is a
**synthetic-validated scaffold only** — no live STT → LLM → TTS loop, no turn-taking
measurement on real audio — and §6.2's entity error rate is computed offline from the master
table.

---

## 7. The sim-vs-real gap

**n_pairs = 176**, nova-3: the simulated arm re-runs the identical condition list on a common
clip subset with 16 pyroomacoustics RIRs (failures excluded — real 0/7040, sim 6/1760). Two
matchings are load-bearing and getting either wrong manufactures a result — **both arms are
scored on the same 10 clips** (the 40-clip real arm against the 10-clip sim arm reads a
19.9-point gap, a corpus difference masquerading as a simulation gap) and **pairs are matched
on the measured Schroeder T20 RT60 of *both* files, never on the Sabine target** (max |Δ|
**0.017 s**). Both are enforced in code (Appendix F).

| aspect | result |
|---|---|
| **LEVEL** | sim **underestimates WER by 12.1 points**, 95 % CI [−15.0, −9.6] |
| **ORDER** | Spearman **ρ = 0.873** (p = 3.2 × 10⁻⁵⁶), Kendall τ = 0.698 |
| **DEAD ZONES** | real 2, sim 1, both 0 → **Jaccard 0.00, recall 0.00** |

**The sharper finding is the dead-zone recall of zero.** Ranking is the easy part; what this
project actually delivers — *which specific conditions are silently dangerous* — is what the
simulation gets wrong, missing both real dead zones and inventing a different one. For a
synthetic-RIR-only benchmark: trust it to say condition A is worse than B; do not trust it to
locate your danger zone.

---

## 8. Limitations and honest boundaries

*One line each; elaborations in H.3.*

1. **Behavioural factors are bracketed out** — accent, code-switching, disfluency, rate, head
   orientation, and above all **the Lombard effect, which cannot be simulated: in noise people
   involuntarily change pitch, loudness, spectral tilt and timing, so noise does not merely
   mask the signal, it changes how the signal is produced.** Every SNR result here describes a
   talker who does not react to the noise, which no real talker is.
2. **One speaker, one accent, 40 utterances** — a limit on external validity, not a caveat.
3. **References are human-verified but human** — clean floor **WER 1.65 %** (6/363 words),
   adjudicated by ear, never edited to match the model.
4. **Elevated capture noise floor** — room tone **−52.9 dBFS** against a −60 dBFS target,
   constant across takes: an offset, not a confound.
5. **Commercial model literals move** — Deepgram **`nova-3`**, run **2026-08-05**: a re-run in
   three months is not the same experiment.
6. **The `rt60` axis is realized by nearest-match snapping**, so a continuous sweep sees a
   step function — the other face of §6.3's DRR finding.
7. **Deletions carry no confidence** — 35.1 % of reference words and 69.3 % of all errors,
   invisible to every confidence-based analysis here including the headline (§6.1).
8. **The narrowband codec is G.726, not AMR-NB** (§4) — stated, not silent.
9. **WER is not the deployment metric** — entity error diverges from it (§6.2) and it
   *understates* a hallucinating model (§6.6).
10. **Composition order is fixed and order effects are unstudied** (§4, A.4).
11. **`snr_db` is capped at 20 dB, a measurement not a preference** — a 25 dB request is
    delivered at ~22.5 dB (§5).
12. **Vendor confidence is not a calibrated probability by construction** — the premise of
    §6.4, not a defect; Whisper's is *derived* differently, hence within-model percentiles.
13. **The live voice agent was not built** (§6.8) — a scaffold, presented as exactly that.
14. **Cross-model absolute WER is a comparison aid** — three residuals survive by design,
    pinned by tests, and the L1 arm runs on 10 clips (Appendix C).
15. **The active-learning null is a surrogate-oracle result** (§6.5), and the ST − S1 gaps
    carry a small upward finite-clip bias (D.6).
16. **The L3 sweeps are n = 5 clips**, and four of six gave no quotable threshold.

---

## 9. What I'd do next, and what field data would earn

**First, fix the survivor bias in the headline (limitation 7):** emission rate × calibrated
per-word confidence, an "expected words recovered" score, robust to the deletion blind spot
and needing no vendor change. **Second, re-parameterise the reverb axis on DRR or C50 and
re-run** — §6.3 shows RT60 mislabels the delivered acoustics badly enough to invert the
ordering, so a DRR-indexed grid would give the surrogate a coordinate it can interpolate and a
fairer second test of §6.5's null. **Third, the agent layer as designed but not built:**
replay-mode injection of degraded WAV bytes into a streaming socket, never speaker-to-mic
playback, scored against the same `task_specs.json` slots.

**And what field data would earn.** Everything in limitation 1, the Lombard effect above all:
this rig can tell you what a room does to a fixed signal, not what a room does to a *person*.
The right follow-up is not "replace the sim with field data" but "use the sim to find candidate
dead zones cheaply, then spend field recordings confirming *only those*" — the active-learning
argument, one level up.

---

## 10. Reproducibility

**The experiment freeze is `results/MANIFEST.json`** and this section defers to it (Appendix
F). Headline: **10,906 Deepgram calls ≈ 745 minutes of audio ≈ $3.20**, the main grid being
7040 calls in 394 s. Whisper runs locally at zero cost; every row caches append-only, so an
interrupted run resumes without re-billing.

```
./.venv/bin/python tests/test_pipeline.py            # the three trap functions, offline
./.venv/bin/python scripts/check_recordings.py       # 40/40 corpus gate
./.venv/bin/python scripts/smoke_join1.py            # JOIN-1 validation gate
./.venv/bin/python scripts/run_experiment.py --dry-run   # call plan + cost, no calls
./.venv/bin/python scripts/run_experiment.py --clips all --models nova-3
./.venv/bin/python scripts/run_experiment.py --clips al --models whisper-base --workers 1
./.venv/bin/python scripts/run_experiment.py --clips al --rir-subdir rirs_sim --results results_sim
```

One gap, stated rather than glossed: the working tree was **dirty** when the manifest was
written, so the recorded SHA identifies the commit the run started from, not an exact tree.
`requirements.lock.txt` is committed alongside; `grid-v1` tags that SHA.

---

*What surprised me:* the confidence–WER correlation came back at **−0.957** — far better
self-knowledge than the premise assumed — and the right response was to lead with that rather
than bury it and sell the 3.4 %.

*What I got wrong first:* `apply_rir` renormalized over whole-file RMS instead of the input's
active-speech region, so the reverb tail quietly de-calibrated every downstream SNR by an
amount that **grew with RT60** — clean-looking audio, plausible numbers, no error message,
caught only by a test that reconstructed the mix and checked delivered SNR against the
request (§4).

---
---

# Appendices

*Reference material. Not part of the main read.*

## Appendix A — The trap functions in full

Source: `deadzone/audio_pipeline.py`; tests in `tests/test_pipeline.py`.

**A.1 `mix_at_snr(speech, noise, snr_db, fs, seed)`.** Adds noise at a target SNR
measured over active speech. `active_speech_mask` is a relative-energy VAD: a 20 ms
frame counts as speech if its energy is within 30 dB of the loudest frame. Speech
power is `mean(speech[mask]**2)`, the noise is scaled to
`speech_power / 10**(snr_db/10)`, and the result is `speech + scale * noise`.

*The trap:* computing speech power over the whole file includes the silent pad and
the inter-word gaps, which deflates it, so the delivered SNR is quieter than
requested — and the error size depends on each clip's silence fraction, making it a
per-clip confound rather than a constant offset. Noise shorter than the clip is
tiled; longer noise is random-cropped from a seed derived from the condition name,
so different conditions do not all see the same excerpt while the same condition
reproduces exactly. `measured_snr_db` is the diagnostic inverse used by the tests
and by the JOIN-1 numeric cross-check (agreement to 0.01 dB on real audio).

**A.2 `apply_rir(speech, rir, fs, preserve_level=True)`.** Convolves, trims, then
renormalizes; returns a signal the same length as the input and aligned to its
onset.

*Trap 1, delay.* `wet = fftconvolve(speech, rir)`, then `direct = argmax(|rir|)`
and `wet = wet[direct : direct + len(speech)]`. Without the trim the output is
shifted later by the RIR's pre-delay, WER inherits a pure alignment artifact, and —
because pre-delay correlates with room size — it reads as a reverb effect.

*Trap 2, level.* Renormalize to the RMS of the **input's active-speech region**:

```python
mask = active_speech_mask(speech, fs)
ref, cur = rms(speech[mask]), rms(wet[mask])
wet = wet * (ref / cur)
```

Using whole-file RMS instead is the bug described in §4: the reverb tail leaks
energy into the formerly-silent pad, inflating the whole-file figure, so the
renormalizer scales the signal down, the active speech ends up quieter than
intended, and the downstream `mix_at_snr` — which calibrates against active-speech
power — delivers a worse SNR than requested, by an amount that grows with RT60. The
wet signal is onset-aligned to the input, which is why the input's mask is the
correct one to apply to both.

**A.3 `classify_errors(reference, hypothesis)`.** Word-level Levenshtein DP plus
backtrace, returning WER, `n_ref`, counts of `match/sub/del/ins`, and `edits` as an
ordered list of `(op, ref_word|None, hyp_word|None)`. The edit list is what makes
the fingerprint layer possible.

`normalize_text` lowercases, **deletes** apostrophes (rather than replacing them
with a space), maps remaining punctuation to spaces, collapses whitespace, and
merges a tiny orthographic compound map. Replacing an apostrophe with a space turns
`o'brien` into two tokens, which against a one-token reference scores as an
insertion plus a substitution — two errors for a perfect transcription. Deleting the
mark is also the standard WER-scoring convention. Fixing normalization moved the
clean corpus from WER 3.03 % to 1.65 %.

The compound map (currently one entry, `wi fi → wifi`) is for orthographic
conventions only: cases where both spellings render identical speech. It is
explicitly **not** a place to absorb recognition errors — `nair`/`nayar` and
`gate`/`gait` are different words and are regression-tested to stay errors, so the
normalizer cannot become a laundering mechanism.

**A.4 Composition order** (moved from §4). `apply_condition` composes in the fixed
order `("rir", "noise", "mic_rolloff", "codec")`. This is physically motivated and
therefore a constant rather than a factor: reverberation is imprinted in the room;
ambient noise sums **at the microphone** and is not reverberated by the talker's
room; the microphone then colours whatever arrives at it; the transmission codec
degrades last, on the signal the microphone actually produced. Mixing noise *before*
convolution would both reverberate the noise — physically wrong for an ambient field
— and calibrate SNR against the dry signal rather than the one the microphone hears,
so the delivered SNR would drift with RT60. Order effects are not studied here
(limitation 10); the order is stated so that it can be challenged.

**A.5 The third recurrence: a guard whose pass and fail were indistinguishable.**
Found by audit inside this project's *own* analysis code, and the cleanest specimen
of the family. The exact-partition check in the sensitivity layer computed

```python
worst = np.max(err)
if worst > tol:
    raise ...
```

With a NaN anywhere in the response vector, `worst` is `nan`, and `nan > tol`
evaluates to `False` — so the guard **passes**, every Sobol index returns `nan`, and
the report prints `sum(S_u) = nan` in the slot where a measurement belongs. The
guard's success condition and its failure condition were the same branch. Same shape
as the `apply_rir` renormalization bug (§4) and the L3 degeneracy guard (Appendix E):
a computation that succeeds on meaningless input and emits a clean-looking number.
The fix is an explicit non-finite check before the comparison, and a test that plants
a NaN and asserts the raise.

## Appendix B — `align_confidences`, and why it raises

Per-word confidences come back aligned 1:1 with the **raw** transcript's whitespace
tokens. `classify_errors` scores **normalized** tokens. Those lists are the same
length only while normalization preserves token count, and it does not:

```
"follow-up"  -> ["follow", "up"]   1 raw -> 2 normalized   (SPLIT)
"wi" "fi"    -> ["wifi"]           2 raw -> 1 normalized   (MERGE)
"--"         -> []                 1 raw -> 0 normalized   (DROP)
```

On the real 7040-row grid this hit **123 rows (1.75 %)** — 122 merges, 1 split. A
`zip()` would have bound every confidence after the first mismatch to the wrong
word, which still trains, still scores, and is invisible downstream: the calibration
layer would have learned from mislabelled data and reported a clean-looking ECE.

The fix carries confidences through the *same* transformation as the text. Stage 1
normalizes each raw token in isolation and replicates its confidence onto every
piece it produces (a split duplicates — each piece was heard with that confidence).
Stage 2 applies the compound merge across the flattened stream, averaging the
confidences it collapses. Tokens that normalize away contribute nothing. Finally the
function asserts that its own token output equals `normalize_text(transcript)` and
**raises `ConfidenceAlignmentError` rather than returning a mismatched list**.

*Audit of the fix against the reported numbers:* hypothesis words fit on went 41,692
(old skip path) → **42,732** (+1040, +2.43 %); 0 rows remain misaligned. Effect on
the headline ECE: raw −0.0005, temperature −0.0000, feature +0.0001 — **negligible**.
The defect was real, but the old path *skipped* rather than zipped, so the
previously reported number was already safe.

## Appendix C — Cross-model normalization for the L1 arm

The two arms disagree about **orthography**, not acoustics:

```
reference     call maria at four zero five nine one two seven seven
nova-3        call maria at four zero five nine one two seven seven   WER 0.00
whisper-base  Call Maria 405-912-7777.                                WER 0.82
```

Deepgram's formatting is switched off at the adapter (`smart_format`, `punctuate`,
`numerals` all False), so its output is already word-form; Whisper has no equivalent
switch. On a corpus loaded with phone numbers, spelled codes, addresses and amounts,
that inflates Whisper's WER by a **condition-independent constant of roughly 0.20 to
0.60** depending on entity density — and a constant offset is mathematically
indistinguishable from an acoustic effect once it is in the table. Same failure shape
as the `o'brien` bug, an order of magnitude larger.

**Why not add digit expansion to `normalize_text`:** the corpus itself uses both
conventions, so no single rule is correct.

```
u02  "four zero five"      should become 405,   NOT "four hundred five"
u05  "fourteen hundred"    should become 1400,  NOT "one four zero zero"
u11  "eighty eight"        should become 88,    NOT "eight eight"
```

Any rule that gets `u02` right gets `u05` wrong. Guessing inside a trap function to
make a number look better is the exact failure this project studies, so
`normalize_text` was left alone.

**What `deadzone/cross_model_norm.py` does instead**, symmetrically to reference and
hypothesis, and only for the L1 arm: (1) the Whisper authors' published
`EnglishTextNormalizer` — a citable community standard, the same normalizer used to
report Whisper's own published numbers; (2) digit-run and alphanumeric-boundary
splitting, so digit *grouping* becomes irrelevant. A symmetric transform can move
the absolute WER but cannot systematically favour one arm — that is the safety
argument, and §6.6's audit is its empirical check (nova-3 shift −0.014, i.e. ~0 as
predicted; Whisper +0.090). If `openai-whisper` is not importable the module
**raises** rather than falling back, since a silent fallback would leave the offset
in place while looking like a corrected number.

**Residuals, pinned by tests rather than patched:** letter runs inside spelled codes
(`AW` vs `a w`) still differ, because splitting letter runs unconditionally would
shatter ordinary words; the normalizer is asymmetric about leading zeros, so `Q9J05`
costs one spurious deletion; it is inconsistent about leading digits in mixed codes
(`1Z99AW5`). The transform is deliberately **not idempotent** —
`EnglishTextNormalizer` rewrites a standalone `1` as `one` while digit-run splitting
manufactures standalone digits — so it must be applied exactly once, to raw text. A
test pins the non-idempotence so the constraint cannot rot silently, and another
pins that genuine recognition errors are still counted as errors.

## Appendix D — Full result tables

**D.1 The six dead-zone conditions** (nova-3, n = 40 clips, 363 reference words
each; gap = `mean_conf − (1 − WER)`). Source: `results/dead_zones.csv`.

| # | rt60 | SNR | noise | codec | rolloff | mean conf | WER | gap |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.7 | 20 | babble | opus-lowrate | 1.0 | 0.843 | 0.387 | 0.230 |
| 2 | 0.45 | 10 | road | none | 1.0 | 0.822 | 0.377 | 0.199 |
| 3 | 0.45 | 10 | road | g726 | 0.0 | 0.806 | 0.337 | 0.143 |
| 4 | 0.2 | 0 | babble | opus-lowrate | 0.5 | 0.807 | 0.336 | 0.143 |
| 5 | 0.45 | 0 | engine | g726 | 0.0 | 0.829 | 0.306 | 0.136 |
| 6 | 0.2 | 0 | babble | g726 | 0.5 | 0.811 | 0.314 | 0.125 |

**D.2 Failure fingerprints by factor family**, Δ as a fraction of reference words,
`d` = Cohen's d. Source: `./.venv/bin/python -m deadzone.analysis.fingerprints`.

| family | dominant edit | Δ | d | implied fix |
|---|---|---|---|---|
| `snr_db` | del | +0.344 | +0.82 | front-end recovery / enhancement |
| `mic_rolloff` | del | +0.264 | +0.60 | front-end recovery, bandwidth extension |
| `rt60` | del | +0.212 | +0.48 | dereverberation (WPE) or closer/beamformed capture |
| `codec = opus-lowrate` | del | +0.111 | +0.25 | front-end recovery |
| `noise = babble` | del | +0.063 | +0.14 | front-end recovery |
| `codec = g726` | **sub** | +0.061 | +0.29 | entity-aware / constrained decoding + boosting |
| `noise = road` | **sub** | +0.059 | +0.28 | entity boosting + matched augmentation |
| `noise = engine` | del | **−0.127** | −0.28 | **NO FIX** — relative improvement |
| `codec = none` | del | **−0.104** | −0.23 | **NO FIX** — relative improvement |

The deletion/substitution split is the actionable part: deletions mean whole words
never reached the decoder, so a language-model or boosting fix cannot recover them,
whereas substitutions mean the acoustic evidence arrived degraded and a decoding-side
prior can.

**D.3 Destroyed-reference-word rate by word class:**

| class | n | share | **rate** | examples |
|---|---|---|---|---|
| proper_noun | 4660 | 14.98 % | **0.646** | okafor, zhang, wei, kowalski, nair |
| spelled_letter | 1187 | 3.82 % | **0.613** | b, d, x, z, h |
| content_word | 11,296 | 36.32 % | 0.530 | gate, file, before, send, number |
| function_word | 8615 | 27.70 % | 0.462 | the, to, is, for, a |
| digit_word | 5341 | 17.17 % | **0.361** | two, seven, one, four, nine |

Entity error rate **0.633** against overall WER **0.511**. Digit words are the
*least* destroyed class, below even function words — spoken digits are a small,
acoustically distinctive closed set, so the entity damage is carried by proper nouns
and spelled letters specifically, not by numbers.

**D.3b Insertions by noise type** (foreign = token absent from the reference):

| group | n_ins | foreign | frac | top foreign tokens |
|---|---|---|---|---|
| babble | 941 | 865 | **0.92** | you, reference, please, a, and |
| engine | 157 | 147 | 0.94 | we, please, a, reference, i |
| road | 162 | 144 | 0.89 | please, three, to, a, we |

Babble carries ~3× the insertions of the other arms; the foreign fraction is high
everywhere, so competing-speech capture is the dominant insertion mechanism across
noise types rather than a babble-only artifact. Worked example
(`rt60-0.2_snr-10_babble_g726_roll-1`, clip u03) — inserted `['can', 'we', 'side',
'of']` yielding *"can we get a move to room four b on the side of the floor"*.

**D.4 Grid-wide edit composition**, as a fraction of the 63,888 reference words
scored: deletions 0.351, substitutions 0.136, insertions 0.020; micro-averaged WER
0.507, mean per-condition WER 0.511.

**D.5 Exact main-effect screen** (marginal contrasts from the complete factorial; 0
API calls), with the genuine 8-run PB design computed on 8 of the 144 measured cells
as a check on how much Resolution-III aliasing would have bitten:

| factor | exact effect | 95 % CI | PB(8-run) |
|---|---|---|---|
| `snr_db` | −0.5986 | [−0.6210, −0.5747] | −0.4484 |
| `rt60` | +0.5554 | [0.5282, 0.5835] | +0.5047 |
| `mic_rolloff` | +0.2294 | [0.2096, 0.2489] | +0.2967 |
| `codec` | +0.1196 | [0.1106, 0.1286] | +0.0689 |

**`SCREEN_CAVEAT`, quoted verbatim as SPEC R4.3 requires:**

> *"Plackett-Burman is a RESOLUTION-III design: two-factor interactions are ALIASED
> onto main effects. A factor with a weak main effect but a strong interaction can
> therefore look dead here and be wrongly dropped. So we drop CONSERVATIVELY — only
> factors whose |effect| is near-zero relative to the largest, and never below
> `min_survivors`. This is a documented limitation of screening, not a bug: Stage-2
> Sobol is what actually resolves interactions."*

It does not bind on a complete factorial (§5): every combination is measured, so no
effect is aliased onto any other. The 8-run PB column above is run anyway on 8 of the
144 measured cells — where it agrees with the exact answer the aliasing did not bite;
where it disagrees, the difference *is* the aliasing, measured rather than assumed.

Survivors: all four; dropped: none. Marginal mean WER by level —
`snr_db` 0 = 0.823 / 5 = 0.595 / 10 = 0.405 / 20 = 0.224;
`rt60` 0.2 = 0.203 / 0.45 = 0.636 / 0.7 = 0.449 / 1.0 = 0.758;
`mic_rolloff` 0.0 = 0.440 / 0.5 = 0.424 / 1.0 = 0.670;
`codec` none = 0.435 / g726 = 0.545 / opus-lowrate = 0.555.

**D.6 Four disclosures about the exact decomposition.**

*(i) The estimand.* The response decomposed is the **unweighted mean of per-clip WER**
within a condition, not the word-weighted pooled corpus WER — the right partner for a
clip-level bootstrap, and named here because SPEC justifies the 40-clip corpus size
in reference *words*, which implies the pooled quantity. Both were computed: max
per-cell difference 0.0194, total variance 0.12658 vs 0.12586, any S1/ST shifts by at
most 0.0033, rankings identical, verdict unchanged.

*(ii) Upward finite-clip bias in the ST − S1 gap.* Clip-sampling noise in the cell
means lands disproportionately in the high-order ANOVA terms (weight 0.25 for the
4-way term against 0.021 for a main effect), inflating ST and deflating S1. Two
independent estimates agree: the bootstrap bias estimate (mean of replicates minus
point estimate) is **+0.0041** for both `rt60` and `snr_db`, and a 20-clip split-half
gives gaps 0.1312 / 0.1156 against the full-sample 0.1275 / 0.1120 — consistent with
a ~1/n bias of that size. Bias-corrected gaps are ≈ **0.124** (`rt60`) and ≈ **0.108**
(`snr_db`), and the order-4 variance share (0.0112) is mostly noise floor rather than
real four-way structure. The pre-registration verdict is unaffected: the margin over
the pre-set 0.020 threshold is ~5×, and the conservative CI lower bound 0.0915 is
4.5× it.

*(iii) The published gap CI is conservative by ~2.5×.* It is built by adding the S1
and ST confidence half-widths in quadrature, which assumes independence. They are not
independent — bootstrap correlation **+0.86 (`rt60`), +0.88 (`snr_db`), +0.86
(`mic_rolloff`), +0.68 (`codec`)** — so their difference is far better determined than
quadrature implies: the direct CI on the `rt60` gap has half-width **0.0143** against
quadrature's **0.0360**. The code computes the tighter interval and deliberately does
not use it for the verdict, which is conservative in the only direction that matters
for a pre-registered test.

*(iv) Sobol indices are relative to a distribution over the inputs.* These are with
respect to the **uniform distribution over the realized design levels**, not uniform
over the continuous ranges. `snr_db`'s levels {0, 5, 10, 20} are unequally spaced, so
the implied prior places 3/4 of its mass at or below 10 dB; `rt60`'s are near-uniform
on [0.2, 1.0]. The indices are "share of variance across the conditions we actually
ran" — the honest quantity for this grid, but they would shift under a different
level allocation.

**D.7 The six unreproduced surrogate proposals** (`results/interaction_report.txt`).
All six share factor = `rt60`, proposed dip at 0.7, probes [0.6, 0.7, 0.8]:

| # | snr_db | codec | rolloff | surrogate | measured WER (0.6 / 0.7 / 0.8) | result |
|---|---|---|---|---|---|---|
| 1 | 7.50 | none | 0.750 | 0.3404 | 0.0659 / 0.0679 / 0.0725 | no dip |
| 2 | 17.50 | opus-lowrate | 0.750 | 0.2520 | 0.0479 / 0.0254 / 0.0243 | no dip |
| 3 | 10.00 | none | 0.750 | 0.2414 | 0.0368 / 0.0479 / 0.0443 | no dip |
| 4 | 7.50 | none | 0.625 | 0.2215 | 0.0225 / 0.0368 / 0.0400 | no dip |
| 5 | 7.50 | g726 | 0.500 | 0.2214 | 0.1325 / 0.0674 / 0.0645 | no dip |
| 6 | 7.50 | none | 0.875 | 0.4754 | 0.0679 / 0.0879 / 0.0659 | no dip |

Every probe sits in a **benign corner** — measured WER spans 0.0225 – 0.1325 (range
0.1100) versus 0.3086 for the in-grid dip. The surrogate placed its proposals where
the response is nearly flat and clip noise dominates. These six are **not presented
as measured surprises** anywhere in this document; the in-grid dips (§6.3) are real
measurements with CIs and stand on their own.

*The two scans walked different room triplets* — sharing only one room, which is why the
mechanism predicts opposite signs and why the probe was never a test of the in-grid dip.
The falsified hypothesis was that all three probe requests snapped to one RIR; they did
not.

| scan | triplet, by DRR (dB) | outcome |
|---|---|---|
| GRID | Bar −2.05 · **Campground 4.26** · Shower −10.02 | **DIP** — middle room has the *best* DRR |
| PROBE | Office 7.76 · **Campground 4.26** · Classroom 9.42 | **PEAK** — middle room has the *worst* DRR |

**D.8 Active-learning target curve in full** (`results/al_savings.txt`), so the
headline row cannot be cherry-picked. Median evals-to-target across 8 seeds:

| target | active | random | reached (act/rand of 8) |
|---|---|---|---|
| 0.144 | inf | inf | 0 / 1 |
| 0.159 | inf | inf | 2 / 2 |
| 0.167 | inf | 32 | 3 / 5 |
| 0.182 | inf | 21 | 3 / 7 |
| 0.198 | inf | 16 | 4 / 8 |
| 0.205 | 36 | 15 | 6 / 8 |
| 0.221 | 34 | 15 | 7 / 8 |
| 0.228 | 20 | 15 | 8 / 8 |

Per-seed evals to the 0.162 target — active `[inf, inf, inf, inf, 45, 15, inf, inf]`,
random `[15, inf, 15, inf, inf, 21, inf, 15]`; median `inf` for both. **Split-seed
robustness:** across 4 train/test splits the per-split winner is random, active,
active, random with paired differences +0.041, +0.005, −0.003, +0.005 — the sign
straddles zero. **Fidelity floor** (surrogate fitted to all 106 training conditions
and scored on the held-out set — the best any arm could reach with unlimited calls):
`boundary_rmse` 0.185, `boundary_error` 0.157, `global_rmse` 0.202 against a test-set
sd of 0.333. **Leakage check:** 106 training vs 70 held-out conditions, 13 near the
contour, shared factor vectors = 0 → disjoint, verified on the actual factor matrices
rather than only by construction.

**D.9 L1 divergence regions, full ranking** (`results/model_arms.txt`; whisper-base
is the worse model in every region):

| factor | span | WER gap | nova-3 | whisper | dead-zone rate |
|---|---|---|---|---|---|
| `snr_db` | 10–15 dB | 0.638 | 0.315 | 0.953 | 0.0 % / 38.5 % |
| `noise_type` | engine | 0.627 | 0.335 | 0.962 | 0.0 % / 37.5 % |
| `snr_db` | 15–20 dB | 0.624 | 0.150 | 0.774 | 0.0 % / 58.3 % |
| `rt60` | 0.6–0.8 | 0.610 | 0.372 | 0.981 | 2.8 % / 47.2 % |
| `codec` | g726 | 0.593 | 0.468 | **1.060** | 0.0 % / 28.1 % |
| `rt60` | 0.4–0.6 | 0.585 | 0.481 | **1.066** | 0.0 % / 21.2 % |
| `rt60` | 0.2–0.4 | 0.581 | 0.157 | 0.738 | 0.0 % / 72.2 % |
| `mic_rolloff` | 0.0–0.25 | 0.576 | 0.336 | 0.912 | 0.0 % / 50.0 % |

**Hallucination metrics** (whisper-base): median hyp/ref length ratio 1.00, **p95
2.75**, **9.9 %** of rows exceed twice the reference length, mean foreign-token fraction
**0.528**. The §6.6 example in full, plus two more on the same 3-word reference:

```
[u02 @ rt60-1_snr-5_babble_opus-lowrate_roll-1]   3 ref words -> 49 hyp words
REF: call maria at
HYP: I'm gonna go here and do a little bit of the work. You call her, you have a
     passport, you have a file. You have a file, you have a file, you have a file,
     you have a file, you have a file. You have a file.
```


```
[u02 @ rt60-0.7_snr-0_babble_g726_roll-1]   3 ref words -> 38 hyp words
REF: call maria at
HYP: I don't know if you can do it. I'm calling you. I have a voice there. I'm not
     going to put it in the comments. I'm not going to put it in the comments.

[u02 @ rt60-1_snr-5_babble_opus-lowrate_roll-0.5]   3 ref words -> 34 hyp words
REF: call maria at
HYP: The car is going to be in the air for a minute. The car is going to be in the
     air at 4.05. The car is going to be in the air for a minute.
```

**D.10 L2 calibration, full numbers** (`results/calibration.{json,txt}`). 7040 rows,
42,732 hypothesis words, grouped by **condition** (169 groups), 5 seeds. ECE median
[min, max] on held-out conditions:

| | ECE |
|---|---|
| raw confidence | **0.0507** [0.0496, 0.0586] |
| + temperature scaling (T = 1.385 [1.354, 1.435]) | **0.0346** [0.0312, 0.0391] |
| + feature-conditioned | **0.0077** [0.0045, 0.0106] |

Held-out **clips** as an independent robustness check reproduce the ordering with a
smaller margin: 0.0487 → 0.0396 → 0.0196. A random word-level split is **not offered by
the code**: words from one clip+condition are highly correlated, so such a split leaks —
and the symptom of the leak is a *better* ECE, which is exactly the kind of
improvement-shaped defect this project is about. Word-level correctness labels come from
the `classify_errors` edit list filtered to edits with a hypothesis word (`match`, `sub`,
`ins`), which is the hypothesis sequence in order and therefore aligns 1:1 with
`word_confidences`; `align_confidences` (Appendix B) guarantees that alignment or raises.

**D.11 Deletion blindness, the arithmetic** (the survivor-bias objection in §6.1). Deletions
are **22,416 words = 35.1 % of the 63,888 reference words and 69.3 % of all errors**, and
they carry no hypothesis token, therefore no confidence. Two different denominators follow:

| quantity | value | what it means |
|---|---|---|
| emitted-word accuracy | **0.767** | share of *emitted* words that are correct — what a perfectly calibrated confidence converges on |
| reference recovery | **0.513** | share of *reference* words recovered — what a reader assumes confidence describes |
| overstatement | **0.254** | the gap between the two |

At the limit, **7 of the 176 conditions returned an empty transcript on every one of the 40
clips** (WER 1.00, 100 % deletions). They contribute **zero** hypothesis words, so they are
absent from every confidence statistic in this report: the D1 correlation is computed over
169 conditions and the L2 calibrator (§6.4) is fit on those same 169 and is **silent about
the worst 7**. Nothing in the confidence-based layers can see the conditions where the model
fails hardest, which is why §9's first proposal is a joint emission-rate × confidence
statistic rather than a better calibrator.

## Appendix E — L3 sweep detail and the degeneracy guard

Six controlled single-factor sweeps, 8 levels each, 5 clips, with two methodological
points that are easy to get wrong: the clean baseline is the **raw capture resampled
to the sweep's 16 kHz** — not a composed near-clean condition (which already carries
an RIR and mixed noise) and not the 48 kHz original (whose `centroid` / `rolloff` /
`flatness` integrate to a different Nyquist and would inject a large constant offset
unrelated to degradation) — and WER is measured on the **same wavs** the features come
from, never read off a nearby master-grid cell.

| sweep | verdict |
|---|---|
| `rt60` (snr 20) | LEXICAL FLOOR — features move, WER does not (range 0.047) |
| `snr_db` (rt60 0.2) | LEXICAL FLOOR — WER 0.000 → 0.054, 3/5 clips flat; `rms` spearman 1.00 vs severity, 0 sign flips |
| `rt60` @ snr 0 | NO SUPPORTABLE THRESHOLD — WER 0.025 → 1.000, headline feature does not trend (ρ 0.62, 5 flips) |
| **`rt60` @ opus-lowrate, roll 1.0** | **DECOUPLED** — `f0` collapses at rt60 ≈ 0.62, WER halves at ≈ 0.85 (ρ 0.54, max gap 0.78) |
| `rt60` @ g726, roll 0.5 | NO SUPPORTABLE THRESHOLD — WER 0.082 → 1.000, feature ρ 0.19, 2 flips |
| **`snr_db` @ g726, roll 1.0** | **DECOUPLED** — `rms` collapses at ≈ 4.46 dB, WER halves at ≈ 6.61 dB (ρ 0.95, max gap 0.32) |

Benign-edge lexical curves (mean WER over 5 clips, severity order):

```
rt60    0.2  0.31  0.43  0.54  0.66  0.77  0.89  1.00
WER    0.000 0.000 0.000 0.000 0.029 0.000 0.029 0.047

snr_db  20   17.1  14.3  11.4   8.6   5.7   2.9   0.0
WER    0.000 0.000 0.000 0.029 0.054 0.025 0.025 0.050
```

**The degeneracy guard.** `compare_degradation_rates` min–max normalizes both curves
before locating the 0.5 crossing, so a 0.0 → 1.0 collapse and a 0.000 → 0.054 wander
are indistinguishable to it. The first run therefore reported *"under falling SNR,
rolloff holds to 15.29 dB while WER halves at 11.58 dB: lexical accuracy leads"* —
arithmetically correct, semantically empty, and a sentence that would have entered
this write-up as a finding. Two guards now refuse rather than invent:
`_curve_degeneracy` with `MIN_LEXICAL_RANGE = 0.10` (a lexical curve with no dynamic
range yields no threshold) and a feature-side trend guard with
`MIN_TREND_RHO = 0.70` (spearman of the drift curve against severity rank).
Non-quotable half-levels print in brackets in `results/l3_decoupling.txt`; clips whose
own WER is flat get no vote on which curve leads; a non-trending feature is labelled a
**power limitation, not a finding of stability**. Same failure shape as the
`apply_rir` bug (§4) and the confidence-alignment bug (Appendix B): a computation that
succeeds on meaningless input.

## Appendix F — Environment and artifacts

Python 3.11.9 · numpy 2.4.6 · scipy 1.17.1 · soundfile 0.14.0 · librosa 0.11.0 ·
pyroomacoustics 0.10.1 · SALib 1.5.2 · scikit-learn 1.9.0 · pandas 3.0.5 ·
deepgram-sdk 7.6.0 · openai-whisper 20250625 · **ffmpeg 8.1.2** (`adpcm_g726` at
16 kbit/s / 8 kHz; `libopus` at 8 kbit/s / 16 kHz). Platform Darwin 24.6.0 arm64.

Model literals: Deepgram **`nova-3`**, pre-recorded API
(`listen.v1.media.transcribe_file`), with `smart_format` / `punctuate` / `numerals`
all **False**. Baseline arm: `openai-whisper` **`base`**, local, whose "confidence" is
*derived*, not native — see `_parse_whisper_result`. The Whisper arm must be run
`--workers 1`: it pulls in Numba, whose default workqueue threading layer is not
threadsafe and aborts under concurrent entry.

Assets: 16 measured RIRs (MIT Acoustical Reverberation Survey), measured T20 RT60
0.193 – 1.011 s. 16 synthetic RIRs (pyroomacoustics, 4 shoebox geometries × 2
source/mic distances), paired 1:1 on measured RT60, max |Δ| 0.019 s at generation
(`results/rir_pairs.json`) and 0.017 s as realized in the D4 matching. 12 DEMAND noise
clips, 4 per type, 300 s each, two environments per type; SHA-256 per file in
`results/asset_manifest.json`.

| artifact | contents |
|---|---|
| `results/MANIFEST.json` | the experiment freeze (§10); generated 2026-08-05T17:24:08Z at git `a6ece0c` |
| `results/master.csv` | 176 conditions × 40 clips × nova-3 = 7040 rows, 0 failures; plus the 10-clip whisper-base arm |
| `results/clean_baseline.csv` | 40 raw clips, WER 1.65 % |
| `results/dead_zones.csv` | the 6 dead-zone conditions, ranked by gap (§6.1, D.1) |
| `results/sensitivity_report.txt`, `results/sobol.json` | exact functional-ANOVA decomposition + clip bootstrap (§6.3, D.5–D.6) |
| `results/interaction_report.txt`, `results/interactions.json` | in-grid dips, the DRR mechanism, the reconciliation, the pre-registration verdict (§6.3, D.7) |
| `results/calibration.{json,txt}` | L2 ECE, temperature and feature calibrators, deletion-blindness audit (§6.4) |
| `results/al_savings.{json,txt}`, `al_curve.json`, `al_trajectory.json` | the active-learning null, full target curve, leakage check, fidelity floor (§6.5, D.8) |
| `results/model_arms.{json,txt}` | L1 normalization audit, per-model dead zones, divergence regions, hallucination examples (§6.6, D.9) |
| `results/sim2real.{json,txt}` | the paired measured-vs-simulated arm (§7); n_pairs 176 on the 10 clips common to both arms (`"clips_matched": false`, `"n_clips": 10` recorded explicitly), 5280 real rows dropped by the restriction and 0 sim, failures excluded (real 0/7040, sim 6/1760). The clip restriction is **enforced in code and pinned by a test** whose planted structure makes the unmatched arithmetic flip the *sign* of the gap (matched +0.10 vs unmatched −0.0714) |
| `results/l3_decoupling.{json,txt}` | the six paralinguistic sweeps and the degeneracy guard (§6.7, Appendix E) |
| `results/smoke_join1.csv` | JOIN-1 gate rows (3 clips × gates A/B/C) |
| `results_sim/master_sim.csv` | the simulated-RIR arm, in a **separate** cache dir — the cache key is `(clip_id, condition_name, model)` and does not encode which RIR library produced the row, so a shared cache would report a sim2real gap of exactly zero |
| `report/measurements.md` | capture-chain measurements recorded at record time |

## Appendix G — Grid construction and the JOIN-1 gate

**The 13-call probe (§5).** Run before the grid to decide where to spend calls. At
`rt60 = 0.5`, no codec, rolloff 0.3, WER stayed ≈ 0.00 across the whole SNR range 0 to
25 dB — **SNR alone barely moves the model**. At `rt60 = 1.0` + `g726` + rolloff 1.0,
WER ran **0.18 – 0.46 at every SNR** — the damage is an *interaction*. That result
reallocated the design from a wide SNR sweep to a complete factorial crossing reverb,
codec and rolloff with SNR, and it recurs independently in the L3 benign-edge sweeps
(§6.7). It changed where we *sample*, never what we *predicted*: the `rt60 × snr_db`
pre-registration (§6.3) predates it by weeks, at commit `d8ddd4f`.

**The JOIN-1 validation gate (§5)**, passed **9/9** on clips `u02`, `u17`, `u36` before
any grid row was trusted:

| gate | condition | criterion | result |
|---|---|---|---|
| A | raw clip, composer bypassed | WER 0.00, non-empty word confidences | pass ×3 |
| B | `rt60 0.2 · SNR 25 · babble · no codec · rolloff 0` | WER < 0.15 | pass ×3 |
| C | `rt60 1.0 · SNR 0 · babble · g726 · rolloff 1.0` | WER > 0.5, non-degenerate edit mix | pass ×3 |

Gate A is the only true null in the design: `apply_condition` *always* applies an RIR
and *always* mixes noise, so there is no "clean" `Condition` — the control is the raw
file. Gate B isolates the composer from the model: a raw clip at 0.00 with a benign
condition spiking would locate the bug in the composition chain rather than the ASR. The
numeric cross-check reconstructed the mix and compared `measured_snr_db` against the
request at `snr_db ∈ {0, 10, 25}`, agreeing to **0.01 dB**.

## Appendix H — Corpus, and the limitation notes

**H.1 The three subsets, fixed once and never chosen ad hoc.** The 40 clips `u01`–`u40` are
used whole for the main grid and the screen. The **10-clip subset** `u02, u05, u06, u11, u17,
u22, u24, u33, u36, u39` — spanning names, digit strings, spelled codes and addresses — is the
oracle set for active learning (§6.5), the Whisper arm (§6.6) and both sim2real arms (§7); a
single-clip oracle is far too noisy for a GP, and a fixed subset keeps the arms comparable.
`u02` is the smoke clip used at the JOIN-1 gate (Appendix G).

**H.2 Entity stress cases (moved from §5).** The manifest was written, not sampled, so that
the fingerprint layer would have something to classify: personal names across several
orthographic families (Nguyen, Okafor, Yamamoto, Kowalski, Nair), spelled alphanumeric codes
(`u06`, `u17`, `u33`, `u39`), phone / PO-box / card digit strings, street addresses, currency
amounts, drug dosages and clock times, at varied utterance lengths. Average 9.1 words per
utterance, **363 reference words per condition**. Digits are read in *spoken* form and the
reference stores them the same way — `u02` is "four zero five, nine one two, seven seven", not
"four oh five" — because `normalize_text` deliberately does not map digits to words in either
direction (Appendix A.3, and the cross-model consequence in Appendix C).

**H.3 Notes on the limitations.**

*On 1 (behavioural factors).* The Lombard effect is the load-bearing one because it is not a
missing *input* to the simulator but a missing *feedback loop*: the talker hears the noise and
changes production. A rig that convolves and mixes a fixed recording cannot represent that at
any fidelity, so the SNR axis here should be read as "signal degraded by X dB of masking",
never as "a person speaking in a room that loud". Accent and code-switching are absent for a
simpler reason — one speaker (limitation 2).

*On 4 (noise floor).* Room tone measured **−52.9 dBFS** against the −60 dBFS target, constant
across takes because all 40 were recorded in one sitting. It is therefore an offset rather
than a per-clip confound, but it does bound the usable SNR axis, which is the mechanism behind
limitation 11: the corpus's own inherent SNR is ~25–28 dB, so a 25 dB request is delivered at
~22.5 dB and the grid stops at 20 dB.

*On 6 (nearest-match snapping).* `AssetLibrary.resolve()` selects the RIR whose **measured**
T20 RT60 is closest to the request. Across the four grid levels the realized values land
within **0.024 s** of the request, so the grid itself is well-served; the limitation bites on
any *continuous* sweep, which sees a step function rather than a curve. Always regress on
`rir_rt60_measured`, never on the requested `rt60` — and see §6.3 for why even the measured
RT60 is the wrong coordinate.

## Appendix I — Prior work, annotated

Expanded from §3. Nothing in the method here is novel; this is the genre it sits in.

- **WildASR / "Back to Basics: Revisiting ASR in the Age of Voice Agents" (2026)** —
  controlled acoustic shifts applied to fixed linguistic content, image-source room
  simulation, RT60 sweeps. The closest neighbour in both method and framing, and the reason
  this write-up claims a lens rather than a method.
- **Speech Robustness Bench** — a named benchmark placing models under additive noise,
  reverberation, time transforms and adversarial perturbations.
- **"When Denoising Hinders" (2026)** — perceptually cleaner audio is not necessarily more
  ASR-robust, because enhancement is itself a distribution shift. The perception/recognition
  mismatch hypothesis is already published; this project does not re-claim it.
- **Ko et al. 2017** — RIR augmentation for far-field ASR, the lineage the measured-RIR arm
  belongs to. **Kim et al. 2017** — far-field ASR at deployment scale. **REVERB** and
  **CHiME** — the challenge series that established reverberation and multi-condition
  evaluation.
- **Scheibler et al. 2018 (pyroomacoustics)** — the image-source simulator used for the
  synthetic arm in §7.
- **Carlini & Wagner** — audio adversarial examples: the extreme case of a confidently wrong
  transcript, and the reason "confident and wrong" is a studied failure mode rather than a
  novel framing.
- Speech-enhancement evaluation work showing that ASR rankings and perceptual-quality
  rankings disagree, and ASR-based intelligibility prediction showing low human/machine
  correlation — the same gap §6.7 measures between paralinguistic features and lexical
  accuracy.
