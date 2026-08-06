# Deadzone — executive summary

> **This is a one-page precis.** The full document is `report/writeup.md` (§1–10 ≈ 15 min,
> plus appendices). Every number here is read from `results/`, not restated from prose.

**What it is.** A controlled-degradation rig for Deepgram Nova-3: **176 acoustic conditions ×
40 utterances = 7,040 transcriptions, 0 failures** (11,086 Deepgram calls total, ≈ $3.26). Real
ingredients — 16 measured RIRs, DEMAND noise, real ffmpeg codecs — controlled assembly. It asks
not *how much* the model breaks but **whether it knows it is breaking**. **The genre is
well-trodden** (WildASR, Speech Robustness Bench, "When Denoising Hinders") and nothing in the
method is novel; the delta is the lens — the confidence–accuracy gap *per condition* instead of
WER per condition. Caveat up front: every row ran through the **batch** endpoint, so this maps
acoustic robustness, not streaming behaviour.

### The headline — and it is a correction

Nova-3 mostly *does* know: **Spearman(confidence, WER) = −0.980** across the 169 of 176
conditions that returned any words. It is nonetheless **overconfident in 154/169 (91 %)**, and
**2 of 176 (1.14 %)** are genuine dead zones. Worst: `rt60 0.45 s / SNR 0 dB / engine / G.726`
→ **mean word confidence 0.829 at WER 0.306**, with **0 of 40** clips coming back empty.

An earlier version reported **6** dead zones at mean gap 0.256. It was wrong: confidence was
averaged over the clips that produced words, WER over all 40 — two estimands subtracted. Right
row count, no NaN, **no failing test among 21 suites**. What found it was a **listening pass**:
the exemplars sounded intelligible. Corrected mean gap **+0.147**, and the old set splits into
**2 dead zones / 4 silence-driven / 7 mute zones** (empty transcript on *every* clip; 31.4 % of
all rows are silent). Mute zones are the worst conditions measured and are **invisible to any
confidence-based monitor** — absent is not wrong.

### Mechanism: DRR, not RT60

The `rt60` marginal is non-monotonic — 0.203 → 0.636 → 0.449 → 0.758 — because each level is
delivered by a different *measured* room. **Spearman(DRR, WER) = −1.000** against
**Spearman(RT60, WER) = +0.800**. Reverb benchmarks parameterised by RT60 alone will mis-rank
conditions.

### Pre-registration: CONFIRMED

`rt60 × snr_db` was registered as a genuine two-way interaction (`d8ddd4f`, 2026-07-27, before
any audio existed) under a rule fixed in advance. The grid is a complete 4×4×3×3 factorial, so
Sobol indices are an **exact** functional-ANOVA decomposition, not a Saltelli estimate
(`sum(S_u) = 1.000000000000`): **ST − S1 = 0.128 [0.091, 0.164]** for `rt60` and
**0.112 [0.072, 0.152]** for `snr_db`, S2 rank 1/6, against a 0.020 threshold.

Also measured: entity error rate **0.633** vs WER 0.511, with proper nouns destroyed at
**0.646** but digit words at only **0.361**; a feature-conditioned calibrator cuts ECE
**0.051 → 0.008**.

### Three honest negatives

1. **Active learning is a null.** Straddle acquisition hit the boundary-RMSE target in **2 of 8
   seeds** against random's **4 of 8** at a 45-evaluation budget. Reported as a null. The
   synthetic control still passes, so this is a method meeting a surface it has no purchase on,
   not a broken implementation.
2. **Dead-zone maps do not transfer.** Synthetic RIRs rank conditions well (**ρ = 0.873**) but
   read **12.1 points optimistic** [−15.0, −9.6] and recover **none** of the real dead zones
   (**Jaccard 0.00**). Neither does another model family (nova-3 vs whisper-base, **Jaccard
   0.000**). You cannot borrow someone else's.
3. **Listening is not QA.** Drenched-but-quiet (DRR −10 dB, SNR 20 dB) and dry-but-buried
   (DRR +17 dB, SNR 0 dB) are statistically indistinguishable to the model: WER **0.112** vs
   **0.130**, paired difference **−0.018 [−0.065, +0.031]**, **18 of 40** clips scoring
   identically. No human is close to indifferent between them.

**Go deeper:** `report/writeup.md` §6.1 (headline and its correction), §6.3 (DRR +
pre-registration), §7 (sim-vs-real), §8 (16 limitations).
