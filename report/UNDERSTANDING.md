# UNDERSTANDING.md — what this project is, and where it is exposed

*Written for the author, before a technical interview. Plain English, every term
defined at first use. The point of this document is the weaknesses. Where
something is genuinely strong it is said once and left alone.*

*Every number here was re-read from `results/` or recomputed from
`results/master.csv` on 2026-08-06. Where I could not reproduce a number in
`report/writeup.md`, I say so and give what I got instead.*

---

## If you only remember five things

1. **The headline finding is real, small, and was published wrong once.** Nova-3
   is confidently wrong in **2 of 176 conditions (1.14 %)**. The earlier version
   said 6. A defect — two averages taken over different sets of clips and then
   subtracted — inflated it, and **no test caught it; a human listening to the
   audio did.**

2. **The count sits on a cliff, and the count is not the strongest form of the
   claim.** "Dead zone" is defined by two hardcoded numbers (`wer_hi = 0.3`,
   `conf_pct_hi = 0.6`) with **no sensitivity analysis anywhere in the repo**
   *(stale as of 2026-08-06 — there is one now, `results/dead_zone_sensitivity.{json,txt}`,
   and it reproduces the hand-run table below exactly; verdict FRAGILE)*. I
   ran one: nova-3 gives **13** dead zones at (0.30, 0.50), **2** at (0.30, 0.60),
   **0** at (0.35, 0.60), and **0** at any WER threshold ≥ 0.40. Lead with the
   *continuous* result instead — mean overconfidence **+0.147**, overconfident in
   **154 of 169** conditions — which needs no threshold at all.

3. **Two things were tried and failed, and both are in the deliverable.** The
   active-learning surrogate lost to random sampling (target reached by 2 of 8
   active seeds vs 4 of 8 random). Its obvious fix — re-labelling the reverb axis
   by a better acoustic coordinate — also lost, with a permutation control showing
   the physically correct labelling ranks **18th of 24**. Separately, a
   pre-registered prediction about a human listener **failed**, and its own scoring
   rubric was written so that it could not fail.

4. **The best mechanistic claim rests on four rooms and carries no interval.**
   "Damage is monotone in direct-to-reverberant ratio, not RT60" is stated as fact
   in the three-minute summary. n = 4. The exact one-sided p is **0.042**; Kendall
   two-sided is **0.083**. And the same table prints a competing measure (C50) at
   ρ = −0.800 whose *entire* disadvantage is **one pair of rooms whose C50 values
   differ by 0.19 dB**. This is the fastest way to undercut the project, using a
   column it publishes itself.

5. **The title says "streaming"; nothing streaming was measured.** Every row on
   every arm went through a batch (pre-recorded) endpoint. Behind that: one
   speaker, 40 utterances, **four rooms** (a Restaurant, a Bar, a Campground
   Dining Hall and a **Shower**) — 16 impulse responses sit on disk and the grid
   uses exactly 4.

---

## 1. The goal, in one sentence

Find the acoustic conditions where a speech-recognition model is **wrong while
still reporting high confidence** — because a model that knows it is struggling
can ask the caller to repeat, and a model that does not just commits to the wrong
words and passes them downstream.

---

## 2. What the project actually does, in human terms

### The vocabulary you need

| term | plain meaning |
|---|---|
| **ASR** | automatic speech recognition — audio in, text out |
| **WER** (word error rate) | errors ÷ words in the true transcript. 0 = perfect. Can exceed 1.0 if the model *invents* words |
| **substitution / deletion / insertion** | the three error types: wrong word, missing word, made-up word |
| **confidence** | a number the recognizer attaches to each word it emits. Deepgram and ElevenLabs return one; Whisper's is derived, not native |
| **SNR** (signal-to-noise ratio, dB) | how loud the speech is versus the background. 20 dB = quiet room. 0 dB = noise as loud as speech |
| **RIR** (room impulse response) | a recording of how one real room smears sound. Convolving speech with it makes the speech sound like it was said in that room |
| **RT60** | how long a room takes to decay by 60 dB. The standard "how reverberant" number |
| **DRR** (direct-to-reverberant ratio, dB) | how much sound reaches the mic straight from the mouth versus bouncing off walls. High = crisp, low = drenched |
| **C50** | a close cousin of DRR: energy in the first 50 ms versus everything after |
| **codec** | audio compression on a phone/VoIP link. Here: `none`, `g726` (narrowband telephony), `opus-lowrate` |
| **mic_rolloff** | a filter simulating a cheap microphone that loses high frequencies |
| **dead zone** | a condition where the model is *confidently wrong* — the thing this project hunts |
| **mute zone** | a condition where the model returns an **empty transcript** on every clip. Worse than a dead zone, and structurally invisible to a confidence monitor |
| **estimand** | the exact quantity a number is supposed to describe (which rows, which clips). Two correct averages over different populations, subtracted, produce a wrong answer with no error message. This is the project's signature bug |
| **Spearman ρ** | rank correlation, −1 to +1. −1 means "when confidence goes down, error goes up, perfectly" |
| **ECE** (expected calibration error) | how far a confidence score is from being a real probability. 0 = perfect |
| **Sobol index** | share of the variation in WER attributable to one factor. S1 = the factor alone; ST = the factor including everything it does in combination with others; **ST − S1 = interaction** |
| **Jaccard** | overlap between two sets: shared ÷ combined. 0 = no overlap |
| **GP surrogate / active learning** | fit a cheap statistical model to what you have measured, then let it pick which expensive measurement to make next |

### The machine

1. **40 short utterances** were recorded by one person in one sitting, deliberately
   loaded with the things that break recognizers: names (Nguyen, Okafor,
   Kowalski), phone numbers, spelled codes, addresses, amounts. Clean-condition
   error rate is **1.65 %** (6 errors in 363 reference words), every error checked
   by ear.

2. Each clip is pushed through a **degradation pipeline** that applies, in a fixed
   physically-motivated order: a real measured room impulse response → real
   recorded background noise at a chosen SNR → a cheap-microphone filter → a real
   ffmpeg codec round-trip. **Every ingredient is real; only the assembly is
   controlled.** That is the whole justification for using a simulator: in a real
   field recording every factor moves at once, so you cannot say which one caused
   the failure.

3. **176 conditions** are formed by crossing those knobs. The core is a complete
   4 × 4 × 3 × 3 grid on babble noise (144 cells), plus a 32-cell engine/road arm.

4. Every (clip, condition) pair is transcribed and scored, producing
   `results/master.csv` — **10,560 rows** across three recognizers:
   - **nova-3** (Deepgram) — 7,040 rows, all 40 clips, 0 failures. The spine.
   - **whisper-base** (open, runs locally) — 1,760 rows, 10 clips, 3 failures.
   - **elevenlabs-scribe** (`scribe_v2`) — 1,760 rows, 10 clips, 0 failures.

5. Analysis layers read that one table and ask different questions of it. Total
   spend: **14,606 API calls ≈ $3.70**.

### The three "trap functions"

Three pieces of DSP are correctness-critical in the same way: **if subtly wrong
they produce clean-looking garbage** — plausible audio, plausible numbers, no
exception. They are the strongest engineering signal in the repo and worth being
able to explain cold:

- `mix_at_snr` computes SNR over **active speech only**, not whole-file power. Use
  whole-file power and the silence deflates it, so your "10 dB" isn't 10 dB — and
  the error size varies per clip, making it a confound rather than an offset.
- `apply_rir` trims the room's leading delay (else every reverb condition inherits
  a pure timing artifact that reads as a reverb effect) **and** renormalizes level
  over the input's active region. The first version used whole-file RMS; the reverb
  tail inflates that, so every downstream SNR was de-calibrated **by an amount that
  grew with RT60** — i.e. the bug looked exactly like a reverb finding.
- `classify_errors` returns WER **plus typed edits**, and normalizes reference and
  hypothesis identically. An orthography mismatch lands in every cell equally, and
  a constant error is mathematically indistinguishable from a dead zone.

---

## 3. The eight panels

**There are eight, not six.** The two you are probably not counting are **panel 7
(multi-model comparison)** and **panel 8 (paralinguistic decoupling)**. They sit
*outside* the model toggle — panel 7 *is* the comparison between arms, and panel 8
reads audio files rather than the results table — so they render once for the whole
page instead of once per arm. Verified from `dashboard/build.py` (`per_model` dict
holds panels 1–6 keys; `cross` dict holds `model_arms` and `decoupling`) and
`dashboard/shell.html`.

### Panel 1 — "The silent-failure map" (the hero)

**Question:** for each acoustic condition, does the model's own confidence match
how wrong it actually is?

**Result (nova-3, 40 clips, 176 conditions, 169 of which produced any words):**
- Confidence tracks error almost perfectly: **Spearman ρ = −0.980**.
- But it is **overconfident in 91 % of conditions (154/169)**, mean gap **+0.147**.
- **2 of 176 (1.14 %)** qualify as dead zones. Worst: `rt60 0.45 s / SNR 0 dB /
  engine noise / G.726` → mean word confidence **0.829** at WER **0.306**, with
  **0 of 40 clips silent** (so confidence and WER cover the same clips — the claim
  needs no asterisk). Verified by identity: 0.829418 − (1 − 0.306109) = 0.135528 =
  the stored `gap_spoke`.
- Three categories, not one: **2 dead zones**, **4 silence-driven**, **7 mute
  zones**.

**Strength: MODERATE, and honest about it.** The finding that survives is
inverted from the project's premise — the model is *mostly* self-aware, and that
is the interesting part, because a system tuned on average behaviour will trust it
in the 1 % where it shouldn't. But see §4.1 and §4.2: the count is threshold-
fragile, and the "high" confidence in the #1 dead zone is 0.829 against **0.962**
on the mildest cell — a level many production systems would already treat as low.

### Panel 2 — "Is it a region, or three unlucky cells?"

**Question:** are the dangerous conditions a contiguous region of factor space, or
scattered noise?

**Result:** the RT60 × SNR heatmap, faceted by noise type and codec, with cell fill
= WER and border weight = the confidence gap. It renders; the honest answer given
2 dead zones is that **it cannot show a region, because there are two points.**

**Strength: WEAK as evidence, useful as an exhibit.** With n = 2 the panel is
showing you the WER surface with two markers on it. Do not let it imply a region.

### Panel 3 — "Failure fingerprints: not how many errors, which kind"

**Question:** does each degradation produce a *characteristic* error type, so each
implies a different fix?

**Result (nova-3, 7,040 rows, 63,888 reference words — verified by summing the
table myself):**
- **Deletions 0.351**, substitutions 0.136, insertions 0.020. Deletion is not one
  mechanism among several; it is *the* failure mode.
- Low SNR (+0.344), mic rolloff (+0.264), reverb (+0.212) and `opus-lowrate`
  (+0.111) all produce **deletions** → only front-end fixes help; keyword boosting
  cannot recover a word the acoustic model never emitted.
- `g726` (+0.061) and road noise (+0.059) produce **substitutions** → entity-aware
  decoding and boosting *can* help.
- Engine noise and `codec = none` correctly emit **NO FIX** (they are relative
  improvements).
- Destroyed-word rate by class: proper nouns **0.646**, spelled letters 0.613,
  content 0.530, function 0.462, **digit words 0.361 — the lowest.** So "entities
  degrade fastest" is carried by *names and spelled codes*, not by numbers. The
  naive read of that table gets it backwards.
- Entity error rate **0.633** vs WER **0.511**.
- Insertions under babble are **92 % words absent from the reference** — the model
  transcribing the *background talkers*, a different mechanism from confusion.
  (Note: this is not babble-specific — engine 0.94, road 0.89. Babble just has ~3×
  the insertions.)

**Strength: STRONG.** This is the most defensible layer in the project. Typed
edits, large n, each signature mapped to a concrete engineering action.

### Panel 4 — "What moves WER, and what only moves it in company"

**Question:** which factors matter, and do reverb and noise *compound*?

**Result (nova-3, babble-only 144-cell block, 40 clips = 5,760 transcriptions):**
- Because the grid is a **complete factorial with equal cell counts**, the variance
  decomposition is **exact**, not sampled: `sum(S_u) = 1.000000000000`, max
  absolute partition error 0.00e+00. This is genuinely stronger than the planned
  Saltelli-on-a-surrogate approach.
- ST − S1 (the interaction evidence): `rt60` **0.128**, `snr_db` **0.112**,
  `mic_rolloff` 0.084, `codec` 0.042 — all significant.
- **Pre-registration CONFIRMED.** `rt60 × snr_db` was registered as a genuine
  interaction on 2026-07-27 (commit `d8ddd4f`), *before any audio existed*, under a
  decision rule fixed in advance (gap > 0.020, CI entirely above it, pair ranks
  first in S2). It cleared, with the weakest of four factor × interval combinations
  clearing by 3.58×.
- Two CI forms are published side by side, and the **wider (conservative) one** is
  the one the verdict uses — the code computes the tighter, correct interval and
  deliberately declines to use it for the registered test.

**Strength: STRONG.** A pre-registration that was committed before data, had a
numeric decision rule, and was tested against the conservative interval. This is
the layer most likely to impress.

*Two things the panel header does not say:* the decomposition is **babble-only**
(`fixed_factors: {"noise_type": "babble"}` in `results/sobol.json` — the engine and
road arm is excluded), and `results/sobol.json` contains **bare `NaN` literals**,
so it is not valid strict JSON; only Python's `json` module will parse it. Both are
minor, but the second is the kind of thing an engineer notices in ten seconds.

### Panel 5 — "Can a surrogate find the failure boundary for less?"

**Question:** can boundary-seeking active learning map the failure boundary in
fewer expensive evaluations than picking conditions at random?

**Result: NO. This is a null, and it is the honest kind.**
- Target `boundary_rmse` 0.162 reached by **2 of 8 active seeds** and **4 of 8
  random seeds** within a 45-evaluation budget. Median evals-to-target is `inf` for
  both arms, so no ratio is reportable and none is claimed.
- The winner **flips between train/test splits** (active wins 2 of 4 splits, 13 of
  32 paired runs, median paired difference **+0.003** — positive means active is
  worse).
- **The acquisition function demonstrably worked**: it placed **58.3 %** of its
  chosen evaluations near the decision contour versus 30 % for random. It did its
  job and the job did not pay.
- The whole target curve is published so the headline row cannot be cherry-picked.
- **All 8 seeds ran against a surrogate oracle. No seed was confirmed end-to-end
  against the live API.**
- The synthetic control (`tests/test_active_learning.py`) still passes with the
  banner "active sampling reaches target fidelity in far fewer oracle calls than
  random" — so this is a method meeting a surface it has no purchase on, not a
  broken implementation.

**And the obvious fix was tested and also failed.** Since panel 4's mechanism says
RT60 mislabels the delivered acoustics and DRR orders them perfectly, the natural
rescue is to re-run in DRR coordinates. It changes nothing (14/32 paired runs,
median **+0.000**, wins **0 of 4** splits). **The negative control is the result:**
across all 24 permutations of the same four DRR values — spacing fixed, only which
room gets which label varying — the **physically correct assignment ranks 18th of
24 (permutation p = 0.75)**. Across 44 parameterisations the median paired
difference is −0.0001 and 23/44 favour active: a coin flip.

**Strength: NULL, robustly established.** Two nulls, a working acquisition
function, a permutation control and a stated ceiling (the reverb axis is four
discrete rooms, so no relabelling can add information). This is the most
intellectually honest thing in the repo.

### Panel 6 — "The simulator, audited against the rooms it stands in for"

**Question:** if you built this testbed with *synthetic* room simulation instead of
measured impulse responses, would you get the same answers?

**Result (nova-3, 176 paired conditions, both arms restricted to the 10 clips they
share):**
- **LEVEL:** simulation **underestimates WER by 12.1 points** [95 % CI −15.0, −9.6].
- **ORDER:** Spearman **ρ = 0.873**, Kendall τ = 0.698 — it ranks conditions well.
- **DEAD ZONES: real 1, sim 0, Jaccard 0.00, recall 0.00.** The simulation finds
  *none* of the real dead zones.

**Strength: STRONG, with one caveat.** The clip-matching is load-bearing and was a
real defect once: comparing the 40-clip real arm against the 10-clip sim arm reads
a **19.9-point** gap — 7.8 points of which is pure clip-difficulty confound. The
correct number is 12.1. The 19.9 figure must never be quoted as a result. Caveat:
the dead-zone sets here are computed *within* 10 clips, so "real 1" is not panel
1's "2 of 176", and saying so is mandatory.

### Panel 7 — "Do the models have the same dead zones?" **(cross-cutting)**

**Question:** is knowing-when-you-are-wrong a property of *this* model, of
commercial models generally, or of nothing in particular?

**Result — read the population carefully: all figures are on the 10 clips every arm
ran, n = 1,757 rows per arm.**

| arm | WER (all clips) | dead-zone rate | ρ(confidence, WER) | silent rows |
|---|---|---|---|---|
| nova-3 | 0.433 | **0.57 %** (1/176) | **−0.970** (n = 164) | 24.5 % |
| elevenlabs-scribe | 0.410 ‡ | 3.98 % (7/176) | −0.820 (n = 174) | 4.4 % |
| whisper-base | 0.996 | **39.20 %** (69/176) | **−0.590** (n = 171) | 19.1 % |

‡ = excluded from cross-model WER comparison (see §4.6).

- **The finding is not that Whisper is worse — it is that Whisper is worse *at
  knowing* it is worse.**
- **The models fail in opposite ways.** Substitutions 0.149 (nova-3) vs 0.413
  (whisper); insertions 0.021 vs 0.197 — **9.4×**. Under stress **nova-3 goes
  quiet and Whisper invents.** The worst example (verified in `master.csv`): **11
  reference words → 47 hypothesis words**, a degenerate repetition loop.
  *(Corrected 2026-08-06. Every document in this repo published this as "3 → 49",
  which is `hallucination_report` normalizing "four zero five" → "405" and then
  tokenizing with `[a-z']+` — manufacturing eight digit tokens and discarding
  them. A ratio between two differently-tokenized quantities: 16.3× reported
  against 4.3× actual. The mechanism stands; the magnitude was part tokenizer.)*
  At the
  extreme, one cell returned a row of decorative Unicode glyphs — not language at
  all — at confidence **0.926** (`u11`, `rt60-1_snr-0_babble_opus-lowrate_roll-1`).
- **nova-3 vs whisper dead-zone Jaccard = 0.000.** Combined with panel 6's Jaccard
  0.00, two independent senses in which a dead-zone map does not transfer.

**Strength: MIXED.** The nova-3-vs-Whisper contrast is strong. Everything
involving Scribe is fragile (§4.6), and the "commercial vs open" claim is
confounded (§4.9). **And there is a counter-example to the project's own headline
slogan — see §4.7.**

### Panel 8 — "Would a prosody monitor notice the transcript had failed?" **(cross-cutting)**

**Question:** if an agent watched cheap audio features (pitch, loudness, spectral
shape) as a health signal, would it notice the transcript had already failed?

**Result (5 clips, 6 single-factor sweeps):**
- **2 of 6 sweeps return a DECOUPLED verdict**, and both point the *same* way:
  under reverb `f0` (pitch) collapses at rt60 ≈ 0.62 while WER only halves at
  ≈ 0.85; under falling SNR `rms` (loudness) collapses at ≈ 4.46 dB while WER
  halves at ≈ 6.61 dB.
- **The paralinguistic stream leads** — a feature monitor would alarm *before* the
  transcript measurably degrades. Conservative, not blind. **The opposite of the
  failure mode the layer was built to look for.**
- The other 4 sweeps return **no supportable threshold**, and the code *refuses to
  quote one* rather than inventing it. That refusal is itself a good story: an
  earlier run reported "rolloff holds to 15.29 dB while WER halves at 11.58 dB" —
  real arithmetic on a curve that had barely moved, because the analysis min-max
  normalizes both curves and cannot tell a 0→1 collapse from a 0.000→0.054 wander.

**Strength: WEAK.** n = 5 clips, 2 of 6 sweeps produced a verdict, and the finding
is the reverse of the hypothesis. It is honestly reported, but do not present it as
a result on par with panels 3, 4 or 6.

---

## 4. Every place this is weak, unconfirmed, or didn't work

### 4.1 The headline was wrong, and got corrected — by ears, not by tests

The first published version reported **6 dead zones** at a mean gap of 0.256. It
was wrong. Per-condition confidence was averaged over only the clips that produced
words, while WER was averaged over **all** clips including ones that returned an
empty transcript (which score WER 1.0 and carry *no* confidence). Subtracting two
averages taken over different populations inflated every gap: mean **+0.109**, max
**+0.524**, and it manufactured **4 of the 6** headline conditions.

Scale: **2,210 of 7,040 nova-3 rows (31.4 %) are silent**, spanning 123 conditions.

Neither average was wrong on its own. The defect lived entirely in the
subtraction. Clean arithmetic, correct row count, no NaN, no exception, **no
failing test**. What found it: someone listened to the dead-zone exemplar clips and
noticed they sounded intelligible.

The old #1 (`rt60 0.7 / SNR 20 / babble / opus-lowrate / rolloff 1`, confidence
0.843 at WER 0.387) is now classified `silence_driven`: 10 of its 40 clips were
silent, and on the 30 it spoke on it was **81.8 % accurate at 0.843 confidence —
well calibrated.** The published headline dead zone was a condition where the model
was behaving correctly. **The pair 0.843 / 0.387 must never be quoted as a finding
again.**

A second instance in the same file: the global correlation was reported as
**−0.957 over n = 169**, but all **176** conditions were passed to the correlation
— so the 7 mute conditions entered as fabricated points sitting exactly at the
ideal corner of a negative relationship. Corrected: **−0.980 paired / −0.952
all-clips**, both at n = 169. The published −0.957 sat *between* the two honest
numbers, which is why nothing ever looked wrong. Worse: an earlier investigation of
that exact number concluded "n = 169, not 176" and stopped there. **The count was
right and the computation was still mixing populations. A partial explanation that
reconciles the arithmetic is the most dangerous kind, because it closes the
question.**

### 4.2 Only 2 of 176 conditions are dead zones — and the count is threshold-fragile

A "dead zone" is `WER ≥ wer_hi` **and** confidence in the top `(1 − conf_pct_hi)`
of that model's own conditions. Both thresholds are **hardcoded defaults** in
`deadzone/model_compare.py` (`wer_hi = 0.3`, `conf_pct_hi = 0.6`) and **there is no
sensitivity analysis anywhere in the repo.** I ran one (recomputed from
`results/master.csv` via `classify_conditions`):

| nova-3, dead zones | conf pct ≥ 0.5 | ≥ 0.6 | ≥ 0.7 | ≥ 0.8 |
|---|---|---|---|---|
| WER ≥ 0.20 | 31 | 14 | 2 | 0 |
| WER ≥ 0.25 | 22 | 6 | 0 | 0 |
| **WER ≥ 0.30 (published)** | 13 | **2** | 0 | 0 |
| WER ≥ 0.35 | 5 | 0 | 0 | 0 |
| WER ≥ 0.40 | 0 | 0 | 0 | 0 |

The published number is one cell in a table that runs from 31 to 0. Move either
threshold one notch and the headline count changes by 6×, or vanishes.

Two honest readings, and you should have both ready:

- **The defensive one:** nova-3 has **zero** conditions where it is confidently
  wrong at WER ≥ 0.40. That is a real, threshold-free statement: *this model is
  never confidently, badly wrong on this corpus.*
- **The strong one:** stop quoting a count. The **continuous** claim needs no
  threshold — mean overconfidence **+0.147**, overconfident in **154 of 169**
  conditions, ρ = −0.980. That is the finding. The count is a presentation choice.

A related problem: the #1 dead zone's "high" confidence is **0.829**, which is the
**64th percentile** of nova-3's own conditions. The mildest cell in the whole grid
reads **0.962**, and the highest is 0.981. So the dead zone's confidence is not
high in absolute terms — a production system with an 0.85 threshold would already
be flagging it. **The clean-condition confidence distribution is never quoted
anywhere in the write-up**, and it is the number that decides whether 0.829 is
alarming or reassuring.

Is "the model is mostly self-aware" the more accurate framing? **Yes.** The
write-up already leads with it, correctly. The residual is worth reporting because
of *where* it sits, not how big it is.

### 4.3 The pre-registered listening prediction FAILED — and its rubric could not fail

A sealed prediction was written before anyone listened: a human would rank the
babble-dominated clip clearly harder than the reverb-dominated one in two named
pairs, while the model scores them **exactly equal**.

**Outcome:** the predicted direction held in **1 of 3 pairs**. The listener found
the *reverb* arm harder in two of three. **Scored strictly against the
prediction's own sentence** ("a confident, immediate, non-marginal call in both
pairs") it fails in **both** named pairs — pair 1 on direction, pair 2 on magnitude
("a little gap" is not "clearly harder"). The 1-of-3 figure is the generous reading.

**The worse problem is the rubric.** The sealed "what each outcome means" section
listed exactly two outcomes: *unequal → prediction holds*, *equal → prediction
fails*. It never considered **"unequal, but backwards."** Under the rubric as
written, this backwards result **scores as a PASS**. A pre-registration whose
outcomes do not span what can be observed is decoration.

To the project's credit this is written up as the deliverable rather than buried,
with three rules adopted from it. But the honest summary is: **the project ran two
pre-registrations. The one with a numeric decision rule (panel 4) confirmed. The
one with a prose rule failed and the rule was unfalsifiable.**

What *does* survive, and is worth keeping: the listener had a stated preference in
**3 of 3** pairs while the model scored each pair **exactly equal** (0.333/0.333,
0.222/0.222, 0.250/0.250). Across all 40 clips the paired difference is **−0.0178,
95 % CI [−0.065, +0.031]**, with **18 of 40 clips scoring identically** (I verified
all of this from `master.csv`). So **"a human and the model disagree about which
clip is worse"** stands. The *mechanism* originally offered for it does not.

> **A stale number to know about:** `SPEC.md` Appendix G.10 says the two conditions
> are "exactly equal per-clip on u40, u26, u21, u10" — implying 4 clips. The
> artifact says **18**. The write-up says 18 and is correct; the SPEC is stale.

### 4.4 The DRR claim rests on four rooms and carries no interval

The write-up calls "damage is monotone in DRR, not RT60" **the best mechanistic
finding in the project** and states it as fact in the three-minute summary. Here is
the entire evidence base:

| requested | room | RT60 | DRR dB | C50 dB | marginal WER |
|---|---|---|---|---|---|
| 0.2 | Restaurant | 0.193 | +16.90 | 28.10 | 0.2026 |
| 0.45 | Bar | 0.474 | −2.05 | 10.22 | 0.6359 |
| 0.7 | Campground Dining Hall | 0.680 | +4.26 | 10.03 | 0.4495 |
| 1.0 | **Shower** | 1.011 | −10.02 | 2.12 | 0.7581 |

**n = 4.** I computed the significance myself: ρ(DRR, WER) = −1.000 with an exact
one-sided permutation p of **1/24 = 0.042**; Kendall's τ = −1.000, two-sided p =
**0.083**. **Every other claim in the write-up carries an interval. This one does
not.**

**And the refutation is printed in the project's own table.** ρ(C50, WER) =
**−0.800** — the same magnitude as RT60's +0.800. The entire DRR-beats-C50
separation is **one discordant pair**: Bar vs Campground, whose C50 values differ
by **0.19 dB**. An interviewer can undercut the project's best mechanistic claim in
thirty seconds using a column it publishes itself.

Two further things to say before someone says them to you:

1. **"C50/DRR predicts intelligibility better than T60" is a known result** in
   room acoustics and far-field ASR. The project claims no methodological novelty
   anywhere else — but presents *this* as a discovery with no lineage cited. That
   is the one place the no-novelty positioning leaks.
2. **The reverb axis is four rooms.** `data/rirs/` holds 16 measured impulse
   responses; `results/master.csv` contains exactly **4 distinct `rir_key` values**
   (I checked). Three of the four are exotic: a **Bar**, a **Campground Dining
   Hall**, and a **Shower**. Not one is a car cabin, an office, a kitchen, or a
   phone held 5 cm from the mouth — i.e. none is a place a voice agent actually
   runs. That is the sharpest version of the sim-vs-reality exposure, sharper than
   "it's a simulator."

**The claim that genuinely survives** is the weaker, better one the project also
makes: **non-monotonicity along `rt60` is not a property of a response surface at
all.** Each `rt60` request snaps to the nearest measured room, so whether a dip
exists is a property of *which four rooms were curated*. Re-sample the axis and it
moves — which is exactly what happened between two scans, and is why 0 of 6
surrogate-proposed cells reproduced. **Lead with that.** It is a methodological
warning about benchmark construction and it does not need n > 4.

### 4.5 Whisper's calibration is BLOCKED, not computed

`results/calibration.txt`: "ARM NOT CALIBRATED — whisper-base (reason: alignment).
69 of 1757 rows (3.93 %) still have a hypothesis-word count that disagrees with the
confidence-list length after re-alignment. word_records refuses to zip."

The refusal is correct — zipping would bind confidences to the wrong words and
train the calibrator on mislabelled data. But the consequence is that the
calibration layer is a **two-arm** result, not three:

| arm | words | ECE raw | + temperature | + feature-conditioned |
|---|---|---|---|---|
| nova-3 | 42,732 | 0.0507 | 0.0346 (T = 1.39) | **0.0077** |
| elevenlabs-scribe | 14,668 | 0.1646 ^ | 0.0755 (T = 4.11) | 0.0340 |
| whisper-base | — | **BLOCKED** | — | — |

^ = upper bound; the correctness labels come from the same alignment as WER, so
orthography disagreements label correct words incorrect.

And the layer has a structural blind spot it states clearly: **deletions carry no
hypothesis word and therefore no confidence.** For nova-3 they are 35.1 % of
reference words and **69.3 % of all errors.** A perfectly calibrated confidence
converges on **emitted-word accuracy 0.767**, not **reference recovery 0.513** — an
overstatement of **0.254** if read as the latter. Seven conditions contribute zero
words: **the calibrator is fit on 169 conditions and is silent about the worst 7.**

> **A stale number:** `report/measurements.md` still says "22 411 deleted reference
> words (35.6 %)" and "0.74 observed, n = 7980". The artifact says **22,416
> (35.1 %)** and **0.75 observed on 8,144 words**. `measurements.md` is explicitly
> exempt from the number-pinning test, so it drifts silently. It is a working log,
> not a deliverable — but do not quote from it.

### 4.6 ElevenLabs Scribe is excluded from cross-model WER — principled, or convenient?

**Principled, and I checked the enforcement.** The reason is that Scribe's
orthography is **non-deterministic across identical calls**: four repeat calls on
byte-identical audio returned different transcripts on **5 of 6** probe clips
(`A7X42` vs `A seven X four two`; `Q9J05` vs `Q nine J zero five`; and `u33` flips
the *other* way). Worth up to **0.727 strict WER on identical input**.

The distinction that makes the exclusion principled rather than convenient:
Whisper's orthography offset is a **constant** (+0.090 — measured, and nova-3's is
−0.014 ≈ 0 as predicted, which is the audit that validates the normalizer). A
constant can be characterised once and subtracted. **A per-call draw is variance,
not bias, and cannot be subtracted.** The exclusion is enforced by a raise in code
(`find_divergence_regions` / `compare_models` refuse an incomparable arm unless the
caller explicitly passes `exclude_incomparable=True`; there is no flag that
includes them), and the arm's own WER columns are printed with a dagger.

**Where it is convenient anyway, and you should concede this:**
- The exclusion is **also the outcome that avoids an inconvenient ranking.** Scribe
  scores 0.410 strict / 0.346 normalized against nova-3's 0.433 / 0.447 on the same
  cells — i.e. **on the raw numbers Scribe reads *better* than the spine**, and the
  reason those numbers are not comparable is real but is discovered on the arm
  that would have won.
- **The evidence for it is not persisted.** The 6-clip × 4-call probe
  (`scripts/probe_scribe_orthography.py`) explicitly **"Writes NOTHING"** — I read
  the source. There is no artifact in `results/` containing the repeat-call
  transcripts, the counts, or the 0.727 figure. It exists only as prose in
  `report/writeup.md` and narrative text inside `results/model_arms.txt`. The
  number-pinning test **deliberately does not pin it** for that reason. So the
  single fact that justifies excluding an entire arm is **unreproducible from the
  repo**. That is a real gap, the write-up flags it as one, and it is one command
  and one `json.dump` away from being fixed.
- **The grid was run once per cell**, so every Scribe number in the project carries
  this variance **unquantified**.
- Whether the same non-determinism affects Deepgram or Whisper was **never tested**.
  The honest scope is "measured on one arm, unmeasured on the others" — not "unique
  to this one."

### 4.7 A counter-example to the project's own headline slogan

The write-up says twice — in the three-minute summary and in §6.6 — **"you cannot
borrow someone else's dead-zone map,"** supported by two Jaccard-0.00 results
(nova-3 vs whisper, and real vs simulated RIRs).

But the same artifact (`results/model_arms.json`, `dead_zone_overlap.pairwise`)
contains:

```
elevenlabs-scribe | nova-3         jaccard 0.000   shared 0/8
elevenlabs-scribe | whisper-base   jaccard 0.101   shared 7/69
nova-3            | whisper-base   jaccard 0.000   shared 0/70
```

**Scribe and Whisper share 7 dead zones — Scribe's 7 are a strict subset of
Whisper's 69.** That is a real, non-zero overlap between two unrelated model
families. The write-up mentions the subset relation in §6.7 but never prints the
0.101, and the slogan is stated in §1 without qualification.

The defensible restatement: *dead-zone maps transfer poorly and unpredictably —
sometimes not at all (nova-3 shares nothing with either other arm), sometimes
partially (Scribe's set is contained in Whisper's).* Say that instead. If someone
finds the 0.101 while you are asserting a categorical "cannot," it costs you more
than the softer claim ever would.

### 4.8 The project never says what Deepgram's confidence score actually *is*

> ⚠️ **PARTLY SUPERSEDED 2026-08-06 — and consequence 3 below REVERSES.** The gap
> named here was real when written and has since been closed by
> `deadzone/analysis/confidence_char.py` → `results/confidence_char.{json,txt}`.
> What changed, and it matters because the answer you would give in a room is now
> the opposite one:
> - **Consequence 1 is CONFIRMED and now persisted.** Clean reference **0.9622**
>   at WER 0.0084; raw recordings give 0.9619, so the mildest grid cell is an
>   exact proxy. Dynamic range down to **0.422**.
> - **Consequence 2 is CLOSED.** `utterance_conf` was measured: distinct from
>   `mean_conf` (pearson 0.926) but **not better** at ranking bad transcripts, so
>   leaving it unread cost nothing measurable.
> - **Consequence 3 is REFUTED — do not say "the minimum is the operationally
>   relevant number" in an interview.** It was computed, and **the mean WINS**:
>   aggregate AUROC **0.944** for `mean` against **0.877** for `min`, a paired
>   difference of **[−0.080, −0.055]** — min is *significantly worse*. The
>   hypothesis in the paragraph below was tested and lost. The right reading is
>   that nova-3 spreads its information across words rather than concentrating it
>   in the weakest one, which is what you would expect of a per-frame acoustic
>   score rather than a lattice-derived one.
>
> The paragraph is left unedited below because it is the record of what was
> believed before it was measured, and "I predicted min would win, tested it, and
> it lost" is a stronger answer than either belief on its own. See
> `report/INTERVIEW_INTERNAL.md` §A/Q3 for the delivery.

Limitation 12 is purely negative: "vendor confidence is not a calibrated
probability by construction." Nothing anywhere characterises what it *is* — no
discussion of whether it is a decoder posterior, a lattice-derived score, or an
acoustic-model output; no reference distribution on clean audio; no per-word
distribution shape. For a project whose entire thesis is built on this scalar, that
is a hole.

Three concrete consequences:

1. **The clean-condition confidence is never quoted.** It is **0.962** at the
   mildest cell and 0.981 at the most confident (I computed both). Without that
   anchor there is no way to judge whether the dead zone's 0.829 is "confident."
2. **`utterance_conf` is captured and never used.** I grepped: within `deadzone/`
   it appears only in the adapters that *produce* it, the schema comment, and the
   type-coercion lists. **No analysis layer reads it.** A second confidence signal
   is stored in every one of 10,560 rows and never examined.
3. **`mean_conf` is the arithmetic mean** of the per-word confidences
   (`float(np.mean(confs))`). For a commit-or-ask-again decision that is the wrong
   statistic — one catastrophically wrong entity in an otherwise confident sentence
   is exactly the failure mode this project cares about, and averaging hides it.
   **The minimum, or a low percentile, is the operationally relevant number**, and
   the per-word confidences are stored in the table so it could be computed for
   free. It never was.

This is the sharpest form of the "domain depth" criticism in §6: the instrument is
excellent, and the model's own signal was taken as a given rather than
interrogated.

### 4.9 n = 3 models with only ONE open model — "commercial vs open" is confounded

Three arms: two commercial (nova-3, Scribe), one open (whisper-base). Any
difference between the commercial pair and the open one is equally explainable as
"commercial models are better calibrated" or "whisper-base specifically is not."
The write-up says this. It is worth internalising how thin the comparison actually
is:

The three-arm comparison (§6.7 Finding 2) is on the **159 conditions all three arms
spoke on**, with nova-3 restricted to the shared 10 clips. Under **strict** scoring
nova-3 is separable from Scribe (+0.203 [0.115, 0.312]) and Scribe is *not*
separable from Whisper (+0.074 [−0.112, +0.267]). Under **normalized** scoring the
verdict **reverses**: Scribe collapses onto nova-3 (+0.035 [−0.002, +0.077], the
interval clearing zero only just) and separates from Whisper (+0.227 [0.097,
0.376]).

**Two scorings, opposite verdicts.** What survives under both is only: **nova-3
beats the open baseline.** "Both commercial arms beat the open baseline" does
**not** survive. "Commercial models know when they are wrong" is not supported as a
class claim.

> **A provenance gap I verified:** I reproduced the **strict** row of that table
> exactly from `results/master.csv` (n = 159; ρ = −0.971 nova-3, −0.768 Scribe,
> −0.694 Whisper) using the arm-matched 10-clip intersection *without* excluding
> the 3 conditions containing a failed row. But **no script in the repo produces
> that table, no artifact in `results/` contains it, and the number-pinning suite
> explicitly does not pin it.** It exists only in the prose. It is reproducible;
> it is not *guarded*, so it can drift silently — which is the exact failure mode
> the pinning suite was built for.
>
> Two figures in the same paragraph I could **not** reproduce under any pairing I
> tried: the write-up's Scribe mean gap **+0.276** and nova-3's **+0.121** "on the
> same conditions." I get Scribe **+0.2719** (same-subset) or **+0.2776**
> (all-clips), and nova-3 **+0.1036** (same-subset) or **+0.1884** (all-clips) over
> Scribe's 174 non-mute conditions. `results/confidence_gap.txt` prints Scribe's as
> **+0.272**. Also: "positive in 174 of 174 conditions ... against nova-3's +0.121
> **on the same conditions**" cannot be literally true, because nova-3 is mute on 12
> of the shared cells and so has only 164 — a population slip of exactly the shape
> §4.1 is about, in the corrected deliverable.

### 4.10 One speaker, 40 utterances, simulated degradation, batch not streaming

- **One speaker, one accent, one sitting.** Not a caveat — a hard limit on external
  validity. Nothing here generalises across talkers.
- **The capture chain is imperfect and measured**: room-tone floor **−52.9 dBFS**
  against a −60 dBFS target, dominated by 120 Hz mains hum. Constant across takes,
  so an offset rather than a confound — but it caps the deliverable SNR at ~28 dB,
  which is why the SNR axis stops at 20.
- **All degradation is synthetic.** That is the deliberate trade (§5.1), and the
  sim-vs-real leg measures the cost of it.
- **Every row on every arm is BATCH, not streaming.** Deepgram via
  `listen.v1.media.transcribe_file`, never `listen.live`. ElevenLabs via
  `POST /v1/speech-to-text`, not `scribe_v2_realtime`. Whisper locally with full-file
  lookahead. A streaming decoder commits under a latency budget with truncated
  right context — a genuinely different problem. **The dead-zone map is not a
  live-agent map.**
- Worse for reproducibility: **the stored rows carry no per-row endpoint
  provenance.** `master.csv` records the registry key `"nova-3"`, not the API
  method. That every row was batch is established by *inference* (one `run_id`,
  the manifest's `api` field, `git log -S` on the call line). The write-up says
  this. An arm that mixed endpoints would be indistinguishable in the table.

### 4.11 Smaller things that will still cost you if someone finds them first

- **`results/sobol.json` is invalid strict JSON** — it contains bare `NaN`
  literals, so only Python's `json` module parses it. It is read by the write-up,
  the dashboard and `analysis/interactions.py`.
- **The Sobol / pre-registration result is babble-only.** `fixed_factors:
  {"noise_type": "babble"}`. The section header does not say so.
- **The dashboard's panel-7 caption says "Two commercial streaming models against
  an open baseline."** The write-up's limitation 17 says every arm ran in batch.
  The dashboard is being edited concurrently; check whether that word survived.
- **The dead-zone *ranking* clips delivered accuracy at zero** —
  `gap = mean_conf − clip(1 − wer, 0, 1)`. This is documented and load-bearing
  (Whisper's insertions push WER past 1.0, and unclipped an insertion storm would
  outrank a genuine dead zone). But it means **every Whisper dead zone's `gap`
  equals its `mean_conf` exactly**, so Whisper's dead-zone ranking is just "sorted
  by confidence." Fine; know it before it is pointed out.
- **The freeze artifact and the tag disagree.** `results/MANIFEST.json` records
  commit `0d7d8f5`; the `grid-v1` tag points at `bfd2d78`. The manifest's SHA is
  now several commits behind `HEAD` and the working tree is dirty.
- **Three README claims contradicted the write-up — all three were repaired during
  this session, and I re-verified the repairs.** Know the *shape* of each anyway,
  because these are the errors this project is prone to and the same three will
  recur in the next summary anyone writes:
  1. It advertised the active-learning surrogate as mapping the boundary "in far
     fewer oracle calls than a grid." **Panel 5 is a null**, and the comparison
     actually run was against *random*, which won. (Now reads "On this surface the
     answer was no.")
  2. It concluded **"Whisper is the outlier, not Nova-3."** The write-up measured
     that on the common 159 conditions and **declined** it: strict
     Scribe-over-Whisper is +0.074 [−0.112, +0.267], not separable; it separates
     only after normalization. (Now stated as a claim *about a scoring choice*.)
  3. It reassured that four correlations were "on the 10-clip subset the three arms
     share." True of the *clips* — but each ρ is computed over that arm's **own
     condition population** (n = 164 / 174 / 171, straight out of
     `results/model_arms.txt`). **That is the estimand bug's own shape, one level
     up, in the summary document** — the deliverable was correct and the précis of
     it reintroduced the bug. Say that out loud if asked: it shows the error is a
     *habit* the project has to actively guard against, not a one-off. (Now carries
     an explicit scope paragraph with all three n's.)

### 4.12 Which population is which — the table to keep in your head

Getting this wrong is the project's single most repeated error, committed at least
three times **by people who already knew about it.** Every number needs its
population attached.

| layer | model(s) | clips | conditions used | note |
|---|---|---|---|---|
| **D1 / panel 1** | nova-3 | **40** | 176 (169 with words) | dead-zone rate **1.14 %** |
| **L1 / panel 7** | all three | **10** (shared) | 176; per-arm non-mute 174 / 164 / 171 | nova-3 rate reads **0.57 %** here |
| **§6.7 three-arm table** | all three | **10** | **159** (all three spoke) | not persisted anywhere |
| **D4 / panel 6** | nova-3 | **10** (intersection) | 176 paired | "real 1 dead zone" ≠ D1's 2 |
| **Sobol / panel 4** | nova-3 | **40** | **144** babble cells | engine/road excluded |
| **L2 calibration** | nova-3; scribe | 40; 10 | 169; 174 groups | whisper **blocked** |
| **L3 / panel 8** | nova-3 | **5** | 6 sweeps | 2 gave a verdict |

**nova-3's dead-zone rate is 1.14 % on 40 clips and 0.57 % on 10. Both are
correct. Quoting one without its clip count is the bug.**

---

## 5. The five decisions a skeptical engineer will question

Each: the objection in the sharpest form a critic would put it, the honest defense,
and **where the defense runs out**. A defense with no limit is marketing.

### 5.1 "You simulated the acoustics instead of recording them."

**The objection, sharp:** *"You convolved clean studio speech with 4 impulse
responses and called it a robustness study. Real degradation is a person in a car
with the window down, talking too fast because they're annoyed, over a Bluetooth
headset that's already noise-suppressing. Your Shower RIR isn't a deployment. You
measured your own pipeline."*

**The defense:** counterfactual isolation is the one thing field data cannot give.
In the wild the mic, the placement, the noise and the codec all move at once, so
you can measure that WER went up and you can never say which factor did it. Here
one knob moves and everything else is held exactly fixed, which is what makes the
exact variance decomposition (panel 4) possible at all. And the ingredients are
real — measured impulse responses, recorded DEMAND noise, actual ffmpeg codec
round-trips. **Crucially, the cost of the simulation is measured rather than
asserted**: panel 6 runs the identical grid through synthetic room simulation and
reports it reads 12.1 points optimistic and recovers **zero** of the real dead
zones.

**Where it runs out:** the reverb axis is **four rooms**, three of them exotic (a
Bar, a Campground Dining Hall, a Shower), and *none* resembles a car cabin, an
office, or a phone at 5 cm. So the reverb factor is not a sweep — it is four
arbitrary points, and any non-monotonicity along it is a property of the curation
(§4.4). More fundamentally, the **Lombard effect** is bracketed out by
construction: in noise people involuntarily raise pitch, loudness and change
timing, so noise does not merely mask the signal, **it changes how the signal is
produced.** No room simulator captures a behaviour. Every SNR result here describes
a talker who does not react to the noise, which no real talker is. The right
follow-up is not "replace the sim with field data" but "use the sim to find
candidate dead zones cheaply, then spend field recordings confirming *only those*."

### 5.2 "You used the pre-recorded endpoint for a study about streaming."

**The objection, sharp:** *"The document is called a streaming-ASR silent-failure
map. You never called `listen.live`. A streaming decoder commits words under a
latency budget with almost no right context — that's a completely different
failure surface. You mapped batch robustness and put 'streaming' on the cover."*

**The defense:** the write-up concedes this as limitation 17 and repeats it in the
opening paragraph, the summary and the "what I'd do next" section — it is not
buried. The arm is included for the per-word confidence that makes the silent-
failure question *askable at all*; the literature's usual subjects (Whisper,
Conformer, wav2vec) don't expose one. And the arms were audited for a mode
*difference between them*, which matters because a mismatch would have been
self-serving rather than obvious — batch-vs-streaming would push WER up on
utterance-final words and on exactly the proper-noun / spelled-letter classes the
codec fingerprint attributes damage to, flatten the confidence-WER relation and
raise the dead-zone rate. In other words it would have **manufactured this
project's own thesis.** It was checked rather than assumed; all three arms are
batch.

**Where it runs out:** the concession does not make the results streaming results.
A streaming decoder commits under truncated right context, and the *interaction*
between that and reverb (where the acoustic evidence for a word arrives late) is
plausibly the single most important thing a Deepgram engineer would want to know —
and it is exactly what is missing. Also: **the table cannot prove its own claim.**
There is no per-row endpoint column; batch-ness is established by inference from a
single `run_id`, a manifest field and `git log -S`. That is convincing and it is
not a field. The fix is one column.

### 5.3 "You excluded the arm whose numbers you didn't like."

**The objection, sharp:** *"Scribe scored 0.410 strict against nova-3's 0.433 on
the same cells — it beat your spine. Then you found a reason to exclude it from
every WER comparison, and the evidence for that reason is a six-clip probe you
didn't save. That's convenient."*

**The defense:** the reason is a real and measurable property, not a preference.
Four identical calls on byte-identical audio returned different transcripts on 5 of
6 entity-bearing clips, and the two forms are worth up to **0.727 strict WER** on
the same input. That is *variance*, and variance cannot be subtracted the way
Whisper's *constant* +0.090 offset can. It is enforced in code — the cross-model
WER paths raise by default on an incomparable arm, and there is no flag that
includes them — and the arm is **not** silently dropped: it stays in every
within-model analysis (dead-zone rate, confidence shape, ECE), where the
contamination cannot cross arms. There is also an independent measured check: the
arm's rank correlation *moves* under normalization (−0.820 → −0.948) while nova-3's
does not (−0.970 → −0.970), and a rank correlation is invariant to a constant
offset and attenuated by a per-call one. **That is the signature of noise, not
bias, and it is measured rather than asserted.**

**Where it runs out:** the justification is **not persisted**.
`scripts/probe_scribe_orthography.py` writes nothing; the repeat-call transcripts,
the 5-of-6 count and the 0.727 figure exist only in prose. So the single fact
justifying the exclusion of an entire arm is not reproducible from the repo, and
the pinning suite deliberately does not pin it. Two further limits: the grid itself
was run **once per cell**, so every Scribe number carries this variance
unquantified; and **the same test was never run on Deepgram or Whisper.** "Scribe
is non-deterministic" may well be "single-call ASR benchmarking is unsound, and I
only tested one vendor." One evening of API calls and a `json.dump` closes all of
this.

### 5.4 "You built a headline on a vendor's confidence scalar you never characterised."

**The objection, sharp:** *"You're treating `mean_conf` as if it means something.
What is it — a decoder posterior, a lattice score, an acoustic-model output? What's
its distribution on clean audio? Why the arithmetic mean, when the operational
question is whether ANY word in this utterance is unsafe? And you stored
`utterance_conf` on all 10,560 rows and never looked at it."*

**The defense:** the layer that exists to address this is the honest one. L2 states
up front that vendor confidence is *not* a calibrated probability by construction —
that is the **premise** of the layer, not a defect it found — and then asks whether
a thin learned layer can turn it into one. It can: ECE **0.0507 → 0.0346**
(temperature) **→ 0.0077** (feature-conditioned), on **held-out conditions** with
a grouped split, never a random word-level split (which leaks, and whose symptom is
a *better* ECE). The result is actionable: above rt60 = 0.7 discount reported
confidence by ~0.07; above mic_rolloff 0.5 by ~0.06. And the structural blind spot
is quantified rather than hedged: deletions are 69.3 % of all errors and carry no
confidence at all.

**Where it runs out:** nothing in the project says what the number *is*. The
clean-condition reference distribution — **0.962** at the mildest cell — is never
quoted, so a reader has no anchor for judging whether the dead zone's 0.829 is
alarming. `utterance_conf` is captured on every row and **read by no analysis
module** (I grepped `deadzone/`). And `mean_conf` is a plain arithmetic mean, which
is the wrong statistic for a commit-or-ask-again decision — the minimum or a low
percentile is, and the per-word confidences are already stored, so it was free and
was never computed. **This is the "domain depth" gap in one paragraph: the
instrument is excellent and the signal it is built on was taken as given.**

### 5.5 "Forty utterances from one speaker, and a headline of two conditions."

**The objection, sharp:** *"You're generalising about a production ASR system from
one person reading 40 sentences in one sitting, and the entire deliverable is 2
conditions out of 176 selected by two thresholds you hardcoded and never varied.
That's an anecdote with error bars."*

**The defense:** 40 was chosen against a stated precision target, not by
convenience. WER precision is governed by *reference words*, not clip count: 40
clips ≈ 363 reference words per condition, putting the binomial standard error on a
per-condition WER of 0.5 at **~2.6 points** — tight enough to separate adjacent
grid cells, which is the entire point of a controlled rig. Fifteen clips gives ~3.5
points, at which neighbouring cells overlap and the sensitivity analysis is
measuring noise; 100 clips buys ~1.4 points while multiplying every API call count
by 2.5×. And 40 buys **entity coverage**: names, spelled codes, phone strings,
addresses, currency, dosages — cutting clips cuts *categories*. All confidence
intervals resample the **40 clips**, not the cells, because the same clips appear
in every cell and resampling cells would treat correlated numbers as independent
and understate every interval (this was verified against a deliberately-wrong
bootstrap, which came out 35–45 % too narrow).

**Where it runs out:** none of that touches **external validity**. One speaker, one
accent — the corpus is a *precision* argument, not a *generality* argument, and the
two are being conflated whenever someone reads "363 reference words" as
reassurance. And the "2 of 176" headline is genuinely fragile: `wer_hi = 0.3` and
`conf_pct_hi = 0.6` are hardcoded defaults with **no sensitivity analysis in the
repo**, and the count runs 13 → 2 → 0 across one notch of either threshold (§4.2).
**The right response is to stop leading with the count.** The continuous claim —
mean overconfidence +0.147, overconfident in 154 of 169 conditions, ρ = −0.980 —
needs no threshold and is not fragile at all.

---

## 6. The counterweight — what an adversarial reviewer actually concluded

One short section, because the rest of this document is deliberately negative and
you should not walk in believing the project is weak.

**The verdict was that this clears the bar**, and specifically on three things:

1. **The estimand correction shipped with its guard, not just its fix.** Both
   pairings are published; the mismatched quantity is reachable *only* under an
   explicit name (`gap_all_clips`); the estimand is named at the call site
   (`wer_key=`); and `find_dead_zones` was demoted to a thin view over
   `classify_conditions`, so **you cannot obtain dead zones without also being
   handed the mute zones.** Fixing a bug is ordinary. Making the bug unreachable by
   construction is not.
2. **The active-learning null was tested against its own obvious fix, and killed
   with a permutation control** in which the physically correct labelling ranks
   **18th of 24**. Most people stop at "it didn't work." Very few build the control
   that proves the coordinate was not the problem.
3. **WER comparability is enforced as a raise-by-default code gate with a measured
   justification** — nova-3's shift −0.014 (≈ 0, as predicted), Whisper's +0.090,
   Scribe's a per-call draw — rather than a convention in a docstring.

**The reservation, stated plainly because softening it defeats the purpose:**

> **Domain depth trails methods depth. The candidate can build the instrument;
> they haven't yet spent enough time inside the model.**

That is the single most useful sentence in this document. §4.8 is its concrete
form: a beautifully guarded pipeline built on a vendor scalar nobody opened up.
Read it as a *pointer*, not a verdict — it says where to spend the next two weeks.

---

## 7. The three questions most likely to expose you, and what to say

### Q1. "Your best mechanistic claim is n = 4. And your own table has C50 at −0.800. Why is DRR the story?"

**This is the sharpest attack available and it uses a column you published.** Do
not defend the strong form.

**Say:** *"You're right, and I should state it that way in the document. n = 4 —
exact one-sided p is 0.042, Kendall two-sided is 0.083, and it's the only claim in
the write-up without an interval. C50 sits at −0.800 and the entire separation from
DRR is one discordant pair, Bar versus Campground, whose C50 values differ by
0.19 dB. So the DRR-versus-C50 ordering is not established by this data, and I'd
also note that 'early-energy ratios beat T60 for intelligibility' is a known
room-acoustics result — I shouldn't present it as a discovery.*

*What actually survives, and is the more useful finding, is one level up: my rt60
axis snaps each request to the nearest measured room, so `rt60 = 0.45` is a Bar and
`rt60 = 0.7` is a Campground Dining Hall — unrelated rooms with different
direct-to-reverberant ratios. The non-monotonicity along that axis isn't a property
of reverberation; it's a property of which four rooms I curated. Re-sample the axis
and the dip moves, which is exactly what happened between two of my own scans — 0
of 6 surrogate-proposed cells reproduced against the real oracle. **That's a
warning about how reverb benchmarks are constructed, and it doesn't need n > 4.**
The fix is more rooms, not a better coordinate — and I know that because I tested
the better-coordinate hypothesis and it failed a 24-permutation control."*

### Q2. "So your headline is two conditions out of 176, at thresholds you picked. Is there a finding here?"

**Concede the count immediately; pivot to the continuous form; then give the
methodological finding, which is the real one.**

**Say:** *"The count is fragile and I'd rather not lead with it. `wer_hi = 0.3` and
`conf_pct_hi = 0.6` are defaults I never varied, and the count runs 13 at (0.30,
0.50), 2 at (0.30, 0.60), and 0 at (0.35, 0.60). I should have shipped that
sensitivity table.*

*The threshold-free version is the finding: nova-3's confidence tracks its own
error at Spearman −0.980 across the 169 conditions that produced words, and it's
still overconfident in 154 of 169, mean gap +0.147. So the model is mostly
self-aware — which is the dangerous part, because a system tuned on average
behaviour trusts it in the residual. I'd also flag that my #1 dead zone's
confidence is 0.829 against 0.962 on the mildest cell, so it's the 64th percentile
of that model's own distribution — many production thresholds would already catch
it.*

*The finding I'd actually defend is the one the correction produced. The published
headline was 6 dead zones. It was wrong: confidence averaged over the clips that
spoke, WER over all 40 including empty transcripts. Right row count, no NaN, no
exception, no failing test — the defect was entirely in subtracting two averages
over different populations. **A human listening to the exemplar clips found it; no
test could.** And it forced a taxonomy that matters operationally: 2 dead zones, 4
silence-driven, and 7 mute zones where the model emits nothing at all — and a
confidence-based monitor is structurally blind to those, because absent is not
wrong. That's the deliverable: the early-warning signal I proposed cannot see its
own worst failure mode, and I can tell you how large that hole is — deletions are
69.3 % of all errors."*

### Q3. "What is Deepgram's confidence score, actually?"

**This is the question you are least prepared for, and it is the one a Deepgram
Labs engineer is most likely to ask.** Do not bluff.

**Say:** *"I treated it as an ordinal signal and never characterised it, and that's
the biggest gap in the work. I don't know whether it's a decoder posterior, a
lattice-derived score or an acoustic-model output, and I didn't ask — I only
established what it is *not*, which is a calibrated probability. Three things I'd
fix first:*

*One, I never quoted its clean-condition distribution. It's 0.962 at my mildest
cell and 0.981 at the most confident, and without that anchor nobody can judge
whether 0.829 is 'confident.'*

*Two, I aggregate per word with the arithmetic mean, which is the wrong statistic
for the decision I care about. Committing or asking again is a question about the
*worst* word in the utterance, especially when it's the phone number. The minimum
or a low percentile is the operational number, and I have the per-word
confidences stored — it was free and I didn't compute it.*

*Three, I capture `utterance_conf` on all 10,560 rows and no analysis module reads
it. A second confidence signal, stored and unused.*

*I can tell you exactly what a thin learned layer does to it — ECE 0.051 raw, 0.035
under temperature scaling, 0.008 feature-conditioned on held-out conditions — and I
can tell you the discount schedule it learned. What I can't yet tell you is why it
behaves that way, and that's the part I'd want to spend time on with someone who
knows the decoder."*

**Then offer the honest self-assessment before it is offered to you:** *"The fair
summary of this project is that the methods depth is ahead of the domain depth. I
can build the instrument. I haven't yet spent enough time inside the model."*

---

## Appendix — numbers where an artifact disagrees with a document

Checked 2026-08-06 against `results/`. **`report/writeup.md` is clean** — its
number-pinning suite (`tests/test_report_numbers.py`) re-reads 157 figures across
234 prose sites from 13 artifacts and passes, and every check is proven able to
fail. These disagreements are all in documents *outside* that gate.

| where | says | artifact says |
|---|---|---|
| `report/measurements.md` | 22,411 deleted words, 35.6 % | **22,416**, **35.1 %** (`calibration.txt`) |
| `report/measurements.md` | discount 0.81 vs **0.74**, n = **7,980** | 0.81 vs **0.75**, n = **8,144** |
| `report/measurements.md` | ECE 0.051 → 0.032 → 0.006 | 0.0507 → **0.0346** → **0.0077** |
| `report/measurements.md` | sim misses **both** real dead zones, invents one | real **1**, sim **0** post-correction (`sim2real.txt`) |
| `SPEC.md` G.10 | pair equal on 4 clips (u40, u26, u21, u10) | **18 of 40** clips exactly equal |
| `SPEC.md` B.3 | rt60 = 0.45 → WER 0.6359, rt60 = 1.0 → 0.7581 | those are the **babble-only 144-cell marginals**; `al_drr.txt` reports **0.5559 / 0.7217** over all 176 conditions. Different populations, both correct — but neither says which |
| `README.md` — **repaired this session** | AL surrogate maps the boundary "in far fewer oracle calls" | **null**: 2/8 active vs 4/8 random |
| `README.md` — **repaired this session** | "Whisper is the outlier, not Nova-3" | not separable under strict scoring: +0.074 [−0.112, +0.267] |
| `README.md` — **repaired this session** | four ρ's "on the 10-clip subset the three arms share" | clips shared; **condition populations differ** (164 / 174 / 171) |
| `dashboard/shell.html` (being repaired) | "Two commercial **streaming** models" | every arm ran **batch** (limitation 17) |
| `results/MANIFEST.json` | freeze at `0d7d8f5` | `grid-v1` tag points at `bfd2d78`; both are behind `HEAD` |

**Two figures in `report/writeup.md` §6.7 I could not reproduce** (both are
deliberately outside the pinning suite): Scribe's mean gap **+0.276** and nova-3's
**+0.121** "on the same conditions." Over Scribe's 174 non-mute conditions I get
Scribe **+0.2719** same-subset / **+0.2776** all-clips, and nova-3 **+0.1036** /
**+0.1884**. `results/confidence_gap.txt` prints Scribe's as **+0.272**. It is
possible there is a subset definition I did not find; nothing in the repo defines
one.

---

## One operational note

**Adding this file breaks a currently-green test.**
`tests/test_report_numbers.py::test_the_pin_covers_every_report_document` asserts
that no `.md` in `report/` quotes figures without being pinned, with only
`measurements.md` and `INTERVIEW_RUNBOOK.md` exempt by name. `UNDERSTANDING.md` is
neither. The one-line fix is to add `"report/UNDERSTANDING.md"` to that test's
`exempt` set — it is an explainer over already-pinned artifacts, the same category
as the runbook. The alternative (adding it to `DOCS`) would require writing checks
for every figure quoted above, which duplicates the write-up's own gate.

**That the test fires at all is the right behaviour**, and it is the project's
thesis working on itself: a new prose surface appeared beside the deliverable, and
the repo refused to let it drift silently.
