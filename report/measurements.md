# Measured facts about the capture chain

Running log of things measured at record time that feed the write-up's
limitations section (SPEC appendix R8.2 §8). Recorded when fresh, because none
of this is recoverable after the fact.

---

## Room-tone noise floor — 2026-08-04

**Measured:** `-52.9 dBFS` RMS over 9.94 s of silence
(`data/recordings/_roomtone.wav`, 48 kHz / mono / PCM_16, built-in MacBook Pro
microphone). SPEC R1.3 targets `<= -60 dBFS`, so the capture chain is **~7 dB
over target**.

**Diagnosis.** Not broadband mic self-noise — the mic itself is quiet
(-36 to -46 dB relative above 2 kHz). The floor is almost entirely
low-frequency, tonal, and stationary:

| band | share of energy |
|---|---|
| 20–63 Hz | 24.0 % |
| 63–125 Hz | 66.9 % |
| 125–250 Hz | 6.8 % |
| > 250 Hz | 2.3 % |

Dominant tone at **120 Hz** (2nd harmonic of 60 Hz mains), plus 63 and 94 Hz.
Per-100 ms level spread is only 5.4 dB → continuous source, not intermittent.
This is mains hum coupling in through the power chain, not room noise.

**Why it was not fixed.** The hum tracks the mains/monitor connection, and the
external monitor is needed to read the recording manifest while recording.
Filtering was rejected: a high-pass at 80 Hz recovers only 2.4 dB, and reaching
-60 dBFS requires cutting at 120 Hz, which sits on top of the voice fundamental
— and SPEC R1.2 forbids EQ in the capture chain regardless.

**The number that actually matters — measured in-clip, 2026-08-04.** The
standalone room-tone file is pessimistic; the authoritative floor is the one
measured inside the recordings themselves, through the same gain path as the
speech. On the two checkpoint clips:

| clip | floor | active speech | inherent SNR |
|---|---|---|---|
| u01 | -55.5 dBFS | -27.3 dBFS | **+28.2 dB** |
| u02 | -56.4 dBFS | -28.7 dBFS | **+27.7 dB** |

This is active-speech **RMS** vs floor **RMS**, not the peak-vs-RMS form R1.3's
DoD is phrased in. RMS-vs-RMS is the correct comparison because `mix_at_snr()`
calibrates added noise against active-speech *energy*; the peak-vs-RMS figure
runs ~18 dB optimistic and should not be used to judge calibration headroom.

**Quantified impact on the SNR axis.** Inherent SNR of ~28 dB bounds what the
`snr_db` factor can physically deliver, and the factor space asks for up to 25:

| requested | delivered | error |
|---|---|---|
| 25 dB | ~22.5 dB | -2.5 dB |
| 20 dB | ~19.0 dB | -1.0 dB |
| 15 dB | ~14.7 dB | -0.3 dB |
| <= 10 dB | — | negligible |

The axis compresses at the benign end and is clean everywhere the dead zones
live — i.e. the bias is largest exactly where the model is *not* failing.

**Open decision (park until R4, do not change mid-session):** cap `snr_db` high
at **20** instead of 25 in `DEFAULT_FACTOR_SPACE`. That holds the error to <= 1 dB
across the whole axis and is more honest than claiming a 25 dB condition the
capture chain cannot deliver. One-line change, but it touches `design.py`, which
`conditions.py` asserts against, so it needs a full test pass.

**Note:** raising the recorder's Input volume does NOT improve this. That is
preamp gain — it scales speech and room noise identically. Only speaking louder
or sitting closer moves the ratio, and either must then be held constant across
all 40 takes.

---

## Clean-condition WER floor — 2026-08-04

Corpus of 40 utterances, raw (undegraded) clips through Nova-3.
Artifacts: `results/clean_baseline.csv`, `results/clean_transcripts.jsonl`.

**Corpus WER 1.65 %** (6 errors / 363 reference words), **35/40 clips exactly
correct**. This is the floor every degraded condition is measured against and it
belongs in the write-up as a stated number.

### Normalization fixes applied first (not model errors)

The first pass scored 3.03 % (11 errors). Four of those eleven were orthographic,
not acoustic, and were removed by fixing `normalize_text()`:

| clip | reference | hypothesis | cause |
|---|---|---|---|
| u24 | `obrien` | `o'brien` | punctuation was replaced by a *space*, splitting one token into two → 1 ins + 1 sub for a perfect transcription |
| u31 | `wifi` | `wi fi` | orthographic compound; identical spoken content |

Both were **condition-independent constants**, i.e. they would have landed in
every grid cell identically — and a constant clean-condition error is
indistinguishable from a dead zone, because the model is confident and "wrong".
Fix: apostrophes are now deleted rather than split on (standard WER-scoring
convention), plus a small explicit compound map. Applied symmetrically to
reference and hypothesis, so cross-model comparability is preserved.
Regression-tested in `test_pipeline.py::test_normalization_orthographic_variants`,
which also asserts that genuine errors (`nair`/`nayar`, `gate`/`gait`) still
count — the compound map must never become a laundering mechanism.

A seventh error (u12, `toward` → `towards`) was a **speaker deviation, not a
model error**: adjudicated by ear, the speaker did say "towards". Per R1.6 Rule 2
the manifest row was corrected — both `ground_truth` *and* `say_this`, so the
prompt stays consistent with what was actually said and a future re-record
reproduces it. The reference was NOT changed to match anything the model heard;
it was changed to match the audio, verified by listening.

### The residual 5 errors are real, and three are proper nouns

| clip | ref → hyp | class |
|---|---|---|
| u04 | `nair` → `nayar` | proper noun |
| u07 | `okafor` → `okafar` | proper noun |
| u26 | `kowalski` → `koalski` | proper noun |
| u22 | `gate` → `gait` | true homophone, acoustically unresolvable |
| u15 | `one four` → `fourteen` | number aggregation |

**Three of five clean-condition errors are proper-noun spelling** — an early,
pre-degradation signal for the D2 fingerprint layer, and a reminder that the
entity failure mode exists even at 0 dB of added degradation.

**u15 is a ready-made illustration for the agent layer (R7.8):** WER counts two
errors, but an entity extractor asked for the unit number would score `fourteen`
as *correct*. That is precisely the WER-vs-task-accuracy divergence `agent_eval.py`
was built to expose.

**Constraint this imposes on recording.** The monitor/power configuration must
stay **identical across all 40 takes**. A constant floor is a fixed offset and
harmless; a floor that changes mid-corpus would be a genuine between-clip
confound.

**Goes in the write-up as:** a stated capture-chain limitation with the measured
number, the ~1 dB high-SNR bias, and the note that it is constant across the
corpus rather than varying with condition.

---

## R4 grid — real results (2026-08-05)

`results/master.csv`, run_id `run-20260805T070146Z-6a77c4`.
176 conditions x 40 clips x nova-3 = **7040 rows, 0 failures**, 394 s wall clock,
~$2.52. WER spans **0.006 to 1.000** across conditions (median 0.535), so the
rebalanced grid resolves the full dynamic range rather than a flat surface.

### Why the grid was rebalanced (pre-grid probe, 13 calls)

Probing one clip before spending the budget produced two measurements that
changed the allocation:

1. **SNR alone barely moves the model.** At rt60 = 0.5, codec = none,
   rolloff = 0.3, WER stayed ~0.00 from 0 dB to 25 dB SNR. Even 0 dB babble
   transcribed perfectly.
2. **The damage is an interaction.** At rt60 = 1.0 + g726 + rolloff = 1.0, WER ran
   0.18-0.46 at *every* SNR, with confidence still 0.59-0.92 — and non-monotonic
   in SNR (0.455 at 25 dB vs 0.273 at 5 dB).

The original 60-cell grid put 42 cells at `codec="none"` and only 2 in the
harsh-channel region, so ~70 % of the budget would have measured a flat surface.
`interaction_grid()` crosses reverb x SNR x codec x rolloff fully (144 cells) plus
a noise-character arm (32). This reallocates **where we sample**; it does not
touch the §5 pre-registration of `rt60 x snr_db`, which stands as written.

### D1 — the headline is more nuanced than the premise assumed

> ⚠️ **CORRECTED 2026-08-05 — the numbers first published in this section were an
> estimand mismatch and are RETRACTED.** Per-condition `mean_conf` was averaged
> over only the clips that produced words, while `wer` was averaged over **all**
> clips — including the ones that returned an *empty transcript*, which
> contribute WER 1.0 and no confidence at all. Subtracting two averages taken
> over different populations inflated every gap (mean **+0.109**, max **+0.524**)
> and manufactured four of the six published dead zones. Neither average was
> wrong on its own; the defect lived entirely in the subtraction, which is why
> nothing looked wrong. It was found by the **listening pass** (SPEC A.R3.5) and
> by no test: the dead-zone exemplar clips sounded intelligible. Fixed in
> `d7afd32`; full account in SPEC Appendix G and write-up §6.1. Corrected figures
> below; the retracted ones are kept folded underneath, because the correction is
> itself a finding.

- Global **spearman(confidence, WER) = −0.9795** on the same-subset (paired)
  pairing and **−0.9523** on the all-clips pairing, **n = 169** either way — the
  7 conditions that emitted no words on any clip have no confidence to correlate
  and are excluded from both. Nova-3 largely *does* know when it is failing. This
  is reported first, on purpose: the self-aware regions are the majority and
  burying them would oversell the result.
- It is **overconfident in 91 % of conditions (154/169)**, mean gap **+0.147**.
- **1.14 % of conditions (2/176) are genuine dead zones**, and the taxonomy is
  now three-way: **2 `dead_zone`** (confidently wrong on the clips it spoke on),
  **4 `silence_driven`** (the apparent gap was the mismatch, not confident
  error), **7 `mute_zone`** (empty transcript on *every* clip). Mute zones are
  deliberately *not* dead zones: they are the worst conditions measured, and a
  confidence-based monitor is structurally blind to them because there is no
  confidence to be low.
- Ranked #1: `rt60 = 0.45 s (measured 0.474), SNR = 0 dB, engine, g726,
  rolloff = 0.0` -> mean word confidence **0.8294** while WER is **0.3061**
  (n = 40 clips, 363 ref words, gap **+0.1355**). **0 of its 40 clips were
  silent**, so its confidence and its WER are averaged over the same 40 clips and
  the claim needs no asterisk.
- **31.4 % of clip-rows (2210/7040) produced no words at all**, spanning 123 of
  the 176 conditions. That is the population the mismatch was silently mixing in.

The honest framing for the write-up is unchanged in shape and smaller in
magnitude: the danger is not that the model is blind, it is that it is *mostly*
self-aware — which makes the 1.1 % of conditions where it is not far more
dangerous, because a downstream system calibrated on the average behaviour will
trust it there.

<details><summary>the retracted pre-correction numbers (kept visible on purpose)</summary>

Published here before the fix, computed on the mismatched pairing:

- *"Global **spearman(confidence, WER) = -0.957**."* — an **artifact**. All 176
  conditions were passed to the correlation while n = 169 was reported, so the 7
  mute conditions entered as fabricated points sitting exactly at the ideal
  corner of a negative correlation. −0.957 sat *between* the two honest numbers
  (−0.9795 and −0.9523), which is precisely why it never looked wrong. An earlier
  investigation of this same number concluded "n = 169, not 176" and stopped
  there; the count was right and the computation was still mixing populations. A
  partial explanation that reconciles the arithmetic is the most dangerous kind,
  because it closes the question.
- *"**overconfident in 92 % of conditions** (mean gap 0.256)"* — those are the
  all-clips pairing's figures. That pairing alone would also have called
  **6/176 (3.41 %)** conditions dead zones.
- *"Ranked #1: `rt60 = 0.7 s, SNR = 20 dB, babble, opus-lowrate, rolloff = 1.0`
  -> mean word confidence **0.843** while WER is **0.387**."* — this condition is
  now classified **`silence_driven`, not a dead zone**. **10 of its 40 clips were
  silent**; on the 30 it spoke on the model was **81.8 % accurate at 0.843 mean
  confidence**, i.e. well calibrated. Its gap fell **+0.230 → +0.025**. The
  published headline dead zone was a condition where the model was behaving
  correctly.

</details>

### D2 — deletions dominate; entities are hit hardest

| family | dominant edit | delta (of ref words) | implied fix |
|---|---|---|---|
| snr_db | del | +0.344 | front-end recovery |
| mic_rolloff | del | +0.264 | front-end recovery |
| rt60 | del | +0.212 | dereverberation (WPE) / closer capture |
| codec = opus-lowrate | del | +0.111 | front-end recovery |
| codec = g726 | **sub** | +0.061 | entity-aware / constrained decoding |
| noise = road | **sub** | +0.059 | entity boosting + matched augmentation |
| noise = engine | del | **-0.127** | **NO FIX** — relative improvement |
| codec = none | del | **-0.104** | **NO FIX** — relative improvement |

Destroyed-reference-word rate by class: **proper_noun 0.646**, **spelled_letter
0.613**, content 0.530, function 0.462. Entity error rate **0.633 vs WER 0.511
(gap +0.122)** — entities degrade *faster* than WER, which is exactly the
divergence the agent layer exists to exploit.

Insertions under babble are **92 % tokens foreign to the reference** — the model
transcribing background speakers rather than confusing the target. A different
mechanism from acoustic confusion, and reported separately so the fingerprint
isn't wrong.

### L2 — calibration

On held-out **conditions** (grouped split, never random over words): ECE
**0.051 raw -> 0.032 temperature (T=1.39) -> 0.006 feature-conditioned**.
Conditioning on the acoustic parameters cuts calibration error to an eighth,
~2.4x better than a global temperature. Above rt60 = 0.7, reported confidence
must be discounted by ~0.07 (0.81 reported vs 0.74 observed, n = 7980 words).

**Stated blind spot:** 22 411 deleted reference words (35.6 % of the reference)
carry no hypothesis word and therefore no confidence, so they are invisible to
any confidence-calibration analysis. Given D2 shows deletions are the *dominant*
failure mode, this is a substantive limitation, not a footnote.

### A real alignment defect the guard caught

`analysis.layers.word_records` raised `AlignmentError` on **123 of 7040 rows
(1.75 %)** rather than zipping. Root cause: `edits` are built from *normalized*
tokens while `word_confidences` come from *raw* transcript tokens, so any
normalization that changes token count breaks the 1:1 assumption — here tokens
like `follow-up`, which `normalize_text` splits on the hyphen (1 raw -> 2
normalized). A `zip()` would have silently bound every subsequent confidence in
that row to the wrong word, quietly corrupting the calibration fit.

Handled by the counted-skip path (123 rows dropped and reported). The proper fix
is to carry confidences through the same token transformation as the text —
duplicating on a split, averaging on a merge — and is worth doing before the
calibration numbers go in the write-up as final.

---

## D4 sim-vs-real gap — real result (2026-08-05)

176 paired conditions, nova-3, 10-clip subset. Real arm from `results/master.csv`,
simulated arm from `results_sim/master_sim.csv` (pyroomacoustics RIRs).
Paired on **`rir_rt60_measured`**, never the requested rt60; max |delta| 0.017 s.

- **LEVEL:** sim **underestimates WER by 12.1 points** [95 % CI −15.0, −9.6].
- **ORDER:** Spearman **rho = 0.873** (p = 3e−56), Kendall tau 0.698 — ordering
  is preserved.
- **DEAD ZONES: Jaccard 0.00, recall 0.00.** The simulated arm finds *none* of the
  real dead zones. It misses both (`rt60-0.7_snr-5_babble_opus-lowrate_roll-1`,
  `rt60-1_snr-0_road_none_roll-1`) and invents a different one
  (`rt60-0.45_snr-0_babble_opus-lowrate_roll-1`).

**Verdict: order preserved, level offset.** A pyroomacoustics-only testbed ranks
conditions much like the measured one but reads ~12 points optimistic in absolute
WER — *use it to rank, not to quote numbers*.

**The sharper finding is the dead-zone recall of zero.** Ranking is the easy part;
the thing this project actually delivers — *which* conditions are silently
dangerous — is exactly what the simulation gets wrong. Anyone building a
synthetic-RIR-only robustness benchmark can rank their conditions with it and
still be pointed at the wrong danger zone. That is a stronger and more useful
claim than the level offset, and it is only visible because the dead-zone set was
computed on both arms rather than just comparing mean WER.

---

## NEW BLOCKER — Whisper's output formatting breaks cross-model WER (2026-08-05)

The L1 second arm now runs (see below), but its WER is **not comparable to
nova-3's on this corpus**, and the reason is formatting rather than acoustics:

| model | output on u02 |
|---|---|
| nova-3 | `call maria four zero five nine one two seven seven` |
| whisper-base | `Call Maria 405-912-717.` |

`normalize_text` strips case and punctuation but deliberately does **not** map
digits to words (`audio_pipeline.py`, normalization-parity note). Deepgram's
formatting is turned off at the adapter (`smart_format` / `punctuate` /
`numerals = False`) precisely so its raw output is word-form; Whisper has no
equivalent switch and emits `405`. On a corpus deliberately loaded with phone
numbers, codes, addresses and amounts, that inflates Whisper's WER by a large,
condition-independent constant — measured at WER 0.82 on a near-clean cell.

**Consequence for L1:** cross-model *absolute WER* is invalid on this corpus. The
within-model comparisons remain valid — `within_model_conf_percentile` is
scale-free and each model's dead-zone map is computed against its own WER
distribution — so the "does the confidence-vs-WER *shape* differ" question can
still be answered.

**Not fixed, deliberately.** The tempting fix is a digit→word expansion in
`normalize_text`, but it is genuinely ambiguous: this corpus spells digits
individually ("four zero five", "one four"), so `405 -> four zero five` is right
here while `14 -> one four` would be wrong for a corpus saying "fourteen", and
currency like `$47.50` has no single correct expansion. Guessing inside a trap
function to make a number look better is exactly the failure this project is
about. Options, in preference order:
1. Report L1 as a within-model shape comparison only, and state this limitation.
2. Add a corpus-specific digit expansion applied symmetrically to both sides,
   documented as corpus-specific and covered by tests.
3. Restrict the L1 arm to the non-numeric utterances in the manifest.
