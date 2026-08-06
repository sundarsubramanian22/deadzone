# INTERVIEW_INTERNAL.md — the private script

**The interviewer never sees this file.** It is not in the README, not linked from
anything, and not on screen. It sits in a second window or on paper.

> ## Which presenter document wins
>
> There is now exactly **one**, and it is this file. `report/INTERVIEW_RUNBOOK.md`
> was **deleted** on 2026-08-06 rather than reconciled — recoverable from git
> history if you ever want it. The reason is mechanical, not editorial: this
> document is in `DOCS` in `tests/test_report_numbers.py`, so 23 of its figures are
> re-read from `results/` on every test run and every check is proven able to fail.
> **The runbook was in that test's `exempt` set** — nothing checked its numbers,
> and the artifact wins over this file too. SPEC J.7 is the record of
> what happens to an unpinned presenter surface: it narrated a verdict the dashboard
> contradicted and an SNR level the grid never ran.
>
> Two known live conflicts, both of which the runbook loses (SPEC K.5 names it):
> it recommends saying *"streaming-capable, measured in batch"* — **superseded, see
> §0.2(a)** — and it asserts *"the project says 'streaming ASR' throughout"*, which
> is now false of every forward-facing surface. **Do not rehearse off both.**

**Who you are talking to.** ~1 year at Deepgram on DG Labs. Before that ~6 years at
Ford in data/analytics as a backend/full-stack SWE — Spring Boot, Angular, SQL,
Hadoop. Before that, a QA engineer writing automated test suites. So: **strong
general software-engineering and QA instincts, not a speech-ML researcher.**

What that means for every answer below:

- **Lead with the engineering, not the statistics.** "Here is the bug it prevents"
  beats "here is the estimator." The three trap functions are your best material
  with this person, not the Sobol decomposition.
- **"How do you know your numbers are right"** is his native question. He has spent
  a career on it. Every section below has an answer to it.
- **Reproducibility, provenance, caching, idempotency, failure handling** — he will
  notice these and value them. Volunteer them.
- **Do not perform depth you do not have.** He will spot it faster than a researcher
  would, because that is what QA is. The self-assessment line (§B) is a strength
  with this person, not a concession.
- **Do not explain WER, SNR, RT60, or what an RIR is.** He knows. Explaining basics
  is the fastest way to lose an expert's attention. **The subject under test is his
  product** — the framing is *"I built an instrument that finds where an ASR fails
  silently, and here is what it found,"* never *"your model is bad."*

**Every figure in this file was re-read from `results/` on 2026-08-06.**
`tests/test_report_numbers.py` pins the load-bearing ones — if an artifact moves and
this doc doesn't, the suite fails. Do not copy numbers forward from a summary; that
habit has already cost this project five figures (SPEC C.7).

---

## 0.1 THE POPULATION CARD — read this first, keep it open

**This is the project's signature bug and it has been committed at least three times
by people who already knew about it.** Two averages over different populations,
compared. Every multi-arm number needs its clip count spoken out loud.

| layer | model(s) | clips | conditions | the number that lives here |
|---|---|---|---|---|
| **D1 headline** (README §6) | nova-3 | **40** | 176 (169 spoke) | dead-zone rate **1.14 %** (2/176) |
| **L1 comparison** (README §8) | all three | **10** shared | 176; per-arm non-mute 174 / 164 / 171 | nova-3 reads **0.57 %** (1 of 176, 10 clips) here |
| three-arm ρ table | all three | **10** | **159** (all three spoke) | not persisted anywhere |
| **sim2real** | nova-3 | **10** (intersection) | 176 paired | "real 1 dead zone" ≠ D1's 2 |
| **Sobol / pre-registration** | nova-3 | **40** | **144** babble cells | engine/road excluded |
| **calibration** | nova-3; scribe | 40; 10 | 169; 174 groups | whisper **blocked** |
| **paralinguistic** | nova-3 | **5** | 6 sweeps | 2 gave a verdict |

> **nova-3's dead-zone rate is 1.14 % on 40 clips and 0.57 % on 10. Both are correct.
> Quoting either without its clip count is the bug.**

If he catches you mixing them, do **not** hand-wave. Say: *"That's the exact defect
this project is about, and I just did it. 1.14 % is the 40-clip corpus number; 0.57 %
is the 10-clip intersection every arm ran. Different populations, both real."* Owning
it in one sentence is worth more than never making the slip.

## 0.2 The three lines to have loaded before you start

**(a) The streaming boundary — PRE-EMPTED, not defended.** He works at a company
whose product *is* streaming, so this is the first thing he will reach for.

**The scope was dropped and the word is gone from the cover (SPEC Appendix K).** The
title is now **"Deadzone: Silent Failures in Speech Recognition."** Both *"streaming"*
and *"streaming-capable"* are barred as claims from every forward-facing surface —
`streaming-capable` was the previous compromise wording and it still put the word on
the cover of a document with no streaming row in it. **Do not say either.** The
sentences whose *point* is the limitation are kept and made more prominent, not less.

> *"One thing up front, because it's the first thing I'd ask. This project was
> originally scoped as a streaming study and **that scope was dropped — nothing here
> was streamed.** Every row of every arm went through a whole-file endpoint:
> Deepgram's pre-recorded `listen.v1.media.transcribe_file`, never `listen.live`;
> ElevenLabs' batch REST endpoint, not `scribe_v2_realtime`; Whisper locally with
> full-file lookahead. All three arms are batch, so the comparison isn't
> mode-confounded, but what I mapped is **acoustic robustness, not streaming
> behaviour, and the dead-zone map should not be read as a live-agent map.***
>
> *Nova-3 is in here for one reason and it was never the word 'streaming': **it
> exposes per-word confidence.** The literature's usual subjects — Whisper,
> Conformer, wav2vec — mostly don't, so the silent-failure question is unanswerable
> on them. That's the arm's justification.*
>
> *The failure mode still matters for a live voice agent. Everything here was
> measured batch."*

Then the payoff, which is the part he'll care about:

> *"And I did check the one thing that would have been self-serving: if one arm had
> been streaming and the others batch, that mismatch would have pushed WER up on
> utterance-final words and on exactly the proper-noun and spelled-letter classes I
> attribute to the codec, flattened the confidence-vs-WER relation, and raised the
> dead-zone rate. It would have **manufactured my own thesis.** So I checked rather
> than assumed."*

**If he greps the repo and finds the word — have the answer, don't be surprised by
it.** `SPEC.md` §0–§13 and Appendices A–J still say "streaming" verbatim (40 hits),
**deliberately.** SPEC is a dated log and the project's rule is **supersede forward,
never edit backward** — Appendix K records the decision rather than rewriting the
history that preceded it, and says so in its own first line ("Appendices A–J stay
verbatim, including every 'streaming' they contain"). That is a real answer.
*"I missed it"* is not, and neither is pretending the word was never there. The
existing bar covers `MANIFEST.json`'s arm key (renamed to `batch_commercial`); it
does **not** cover the SPEC log, and it should not.

**Where it runs out — have this ready, don't wait to be pushed:** the concession
doesn't make these streaming results. A streaming decoder commits under truncated
right context, and the *interaction* between that and reverb — where the acoustic
evidence for a word arrives late — is plausibly the single most interesting thing to
a DG Labs engineer, and it is exactly what's missing. Also: **the table cannot prove
its own claim.** There is no per-row endpoint column; batch-ness is established by
inference from one `run_id`, a manifest field and `git log -S`. That is convincing
and it is not a field. **The fix is one column.** Say that — a QA person will respect
"my evidence is inference and here's the schema change that makes it a fact" far more
than a confident assertion.

**(b) The self-assessment line.** Deploy it *before* he gets there — the natural
moment is right after the confidence-comparability section (README §10) or in answer
to Q3. Full text and timing in **§B**.

**(c) The framing for a null.** You will present two failed experiments on purpose.
Say once, early: *"There are two things in here that didn't work, and they're in the
deliverable rather than deleted. I'll flag them as I go."* Then he is not discovering
them, he is watching you show them.

## 0.3 Things you must NOT say

| Don't say | Because |
|---|---|
| "the dead zone is a quiet room / good SNR" | Both nova-3 dead zones are at **SNR 0 dB**, the harshest level in the grid. There is no 25 dB level — the SNR levels are 0, 5, 10, 20. |
| "confidence 0.843 at WER 0.387" | That was the **old** #1. It is now classified `silence_driven` — on the 30 clips it spoke on it was 81.8 % accurate at 0.843, i.e. **well calibrated**. Retracted. |
| "ρ = −0.957" | The artifact of mixing 176 conditions into an n = 169 statistic. Corrected to −0.980 paired / −0.952 all-clips. |
| "the sim2real gap is 19.9 points" | That was a corpus difference masquerading as a simulation gap. The clip-matched number is **12.1**. |
| "active learning saved N calls" | It is a **null**. Report the budget, never a ratio. |
| "Scribe has the best WER of the three" | Excluded from cross-arm WER by design and by a raise in code. Rank it on dead-zone rate and confidence shape, never on WER. |
| "babble causes substitutions" | Babble's dominant edit is **deletions**, like almost every family here. **G.726** is the substitution one. |
| "it says NOT CONFIRMED" | The pre-registration verdict is **CONFIRMED**. |
| "streaming" / "streaming-capable" | **Both barred as claims** (SPEC K.2). Say "a commercial ASR model that exposes per-word confidence." |
| "you cannot QA a voice agent by listening to it" | **Barred as a verdict.** It was the old closing of the listening demo and it is gone from the script. The pre-registration failed and the human half is n = 1 on clips chosen *because* the model tied. Say the narrow surviving claim instead — §5. |
| "Whisper is the outlier, not Nova-3" | Only true under *normalized* scoring. Strict scoring does not separate Scribe from Whisper (+0.074 [−0.112, +0.267]). It is a claim about a scoring choice. |

---

# The walkthrough — lockstep with the README

---

## §1 — Title + one-line what-it-is + the motivation hook

### (1) VERBATIM OPENING CUE

> *"Short version of what this is: I take forty clean recordings, damage them in
> exactly one controlled way at a time, and ask the recognizer not 'how wrong were
> you' but **'did you know you were wrong.'** The reason that's the question and not
> the other one is a coffee chat I had with **Pranav Bachu** on your side — Applied
> AI Verticals. Two things stuck. One, that in non-production voice-AI work with high
> WER the loop is often move sliders, re-measure, keep what went down — **without a
> mechanism.** Two, the open questions it left me with: how do **human and model
> perception** differ on the same audio, and in a drive-thru what actually drives WER
> — the physical setup, the speaker's distance from the mic, the type of mic feeding
> the agent? That plus a couple of papers it sent me to made me want to build an
> instrument for it."*

> ⚠️ **THE NAME IS DELIBERATELY OFF THE SCREEN.** The README reads *"a coffee chat
> with someone at Deepgram who works on deployed voice agents"* — the name was
> removed in commit `ecffe40` for a specific reason: **said out loud by you in the
> room, that is a recollection of a conversation; written on a shared screen in front
> of that person's colleague, next to a characterization of how the field works, it
> is something else.** The demo script does the same thing (`_demo_internal_notes.md`
> records it as a judgement call left for a human).
>
> **So: naming him aloud is your call, made in the room, where tone is available.** If
> you do, own the characterization as *yours* — "what I took away", not "what he
> said the field does". **You are talking to his colleague.**

**Say the disclaimer the README carries, out loud, in the same breath:** *"That's
what motivated me. It is not a claim about what Deepgram has or hasn't studied."*
Frame it as what made you want to build this, never as a gap you are pointing out.

Then, immediately, the streaming pre-empt from §0.2(a). Get it out before anything
else can be built on top of it.

### (2) THE WHYs HE WILL GRILL ON

**"Why does confidence matter more than accuracy?"**
> *"Because confidence is what decides the control flow. A model that knows it's
> struggling can ask the caller to repeat. A model that's wrong and confident commits
> the wrong transcript and hands it downstream. Same WER, completely different
> product outcome. That's not a research distinction, it's a branch in the agent's
> code."*

**"Isn't this a solved / well-trodden area?"** — Concede immediately and specifically.
> *"Yes, and the README says so before it says anything about my results. Controlled
> acoustic-degradation testbeds already exist and are good — WildASR, the Speech
> Robustness Bench, 'When Denoising Hinders'. RIR augmentation goes back to Ko et al.
> 2017 and the REVERB/CHiME challenges; my simulator is pyroomacoustics, Scheibler
> 2018. **The method is theirs.** What's different is the lens — the
> confidence–accuracy gap per condition instead of WER per condition — and that I'm
> testing commercial models that expose per-word confidence, where the literature
> mostly uses Whisper, Conformer, wav2vec. I don't claim novelty anywhere."*

### (3) LIMITATION TO PRE-EMPT HERE

**One speaker, one accent, forty utterances, one sitting.** Say it in the first
minute, not the last: *"Nothing in here generalizes across speakers. That's a hard
limit on external validity, not a caveat."*

---

## §2 — Core idea

### (1) VERBATIM OPENING CUE

> *"The reframe is one sentence. An aggregate WER hides three things a deployment
> needs: **which** conditions break the model, **what kind** of error each one
> produces, and **whether the model knows.** So instead of one number over a corpus,
> I get one number per condition — and next to it, the model's own confidence in
> that same condition. Where those two disagree is the whole project. I call the
> disagreement region a dead zone: confidently wrong."*

### (2) THE WHYs

**"Why is a controlled simulation better than just recording in the real world?"**
This is the single most important 'why' in the project. Answer with the mechanism,
not the philosophy:
> *"Counterfactual isolation. In a real field recording the mic, the placement, the
> noise and the codec all move at once, so you can measure that WER went up and you
> can never say which factor did it. Here one knob moves and everything else is held
> exactly fixed. That's not a nicety — it's what makes an **exact** variance
> decomposition possible at all, which is the sensitivity result later. And the
> ingredients are all real: measured impulse responses from real rooms, real recorded
> DEMAND noise, real ffmpeg codec round-trips. Only the assembly is synthetic."*

Then the part that closes it, because it converts an assumption into a measurement:
> *"And I don't just assert the simulation is fine — I measured what it costs. I ran
> the identical grid through synthetic pyroomacoustics rooms instead of measured
> ones. It reads **12.1 points optimistic** and recovers **zero** of the real dead
> zones."*

**"Isn't 'dead zone' just a threshold you picked?"** — Yes, and this is Q2. Don't
defend it here; flag that you'll get to it, or pivot straight to §A/Q2.

### (3) LIMITATION TO PRE-EMPT HERE

**The Lombard effect.** Name it before he can:
> *"The thing this rig structurally cannot do is behaviour. In noise people
> involuntarily raise pitch and loudness and change timing — noise doesn't just mask
> the signal, it changes how the signal is **produced**. No room simulator captures a
> behaviour. So every SNR result here describes a talker who doesn't react to the
> noise, which is no real talker. That's limitation 1, and it's the strongest single
> argument for what field recordings would actually buy me."*

---

## §3 — Pipeline diagram

**This is his home turf. Spend time here. It is the section where the project reads
as engineering rather than as a paper.**

### (1) VERBATIM OPENING CUE

> *"This is the whole machine. Recording in, four degradations applied in a fixed
> order, transcribe, score, one row in a table. Every arrow is a function with a
> test, and three of those functions are the ones that could have quietly ruined
> everything."*

### (2) THE WHYs

**"Why these degradations, and why in that order?"**
> *"It's the physical signal chain, not an arbitrary sequence. The **room** acts on
> the voice first — that's convolution with a measured impulse response. Then
> **background noise adds at the microphone**, because the noise is in the room too,
> not in the talker's throat. Then the **mic's own frequency response** — a cheap
> mic loses the top end. Then the **codec**, last, because compression happens on the
> wire after capture. It's a constant called `COMPOSITION_ORDER` in `conditions.py`,
> it's `("rir", "noise", "mic_rolloff", "codec")`, and it never varies — which is
> also a limitation, because order effects are unstudied."*

**"How do you know your numbers are right?"** — **THIS IS THE AMMUNITION. Lead with
it whenever he gives you an opening.** Three functions, each with a real bug that
produced plausible-looking garbage with **no error message**, each caught.

> *"There are three functions I call trap functions, and the thing they share is
> that if they're subtly wrong they don't throw — they hand you clean audio and
> plausible numbers, and nothing anywhere tells you.*
>
> *One — **`mix_at_snr`**. SNR has to be computed over **active speech energy only**,
> not whole-file power. If you use whole-file power, the silence in the clip deflates
> it, so your '10 dB' mix isn't 10 dB — and the error size varies per clip, so it
> isn't even a constant offset, it's a confound. I verified the delivered SNR against
> the requested one on real audio at 0, 10 and 25 dB and they agreed to **0.01 dB**.*
>
> *Two — **`apply_rir`**, and this is the good one. Convolving with a room impulse
> response adds the room's leading direct-path delay, so if you don't trim it, every
> reverb condition is time-shifted relative to its reference and you inherit a pure
> alignment artifact that reads as a reverb effect. That part I got right first time.
> What I got **wrong** was the level renormalization: I normalized on whole-file RMS,
> and the reverb tail inflates whole-file RMS. So the output level was wrong, which
> de-calibrated every downstream SNR — **by an amount that grew with RT60.** In other
> words the bug looked **exactly like a reverb finding.** A test caught it. No
> exception ever fired.*
>
> *Three — **`classify_errors`**. It returns WER **plus the typed edits** —
> substitutions, deletions, insertions — because a scalar can't tell you what kind of
> error you got, and the whole fingerprint layer needs the type. And it normalizes
> the reference and the hypothesis **identically**, because an orthography mismatch
> lands in every cell equally, and a constant error is mathematically
> indistinguishable from a real acoustic effect once it's in the table."*

**"Is the degraded audio reproducible?"** — Yes, and he'll like the mechanism:
> *"`apply_condition` is seeded from the condition's own name, so the noise crop and
> the noise selection are deterministic. Re-running produces byte-identical degraded
> audio. That's why the cache stores the **transcript** and not the wav — regenerating
> audio is cheap, disk isn't."*

**"What happens when a call fails?"**
> *"Nothing is dropped silently. A failed row is written with `failed=True` and the
> error string, and the run prints the failure rate at the end with a 2 % gate. The
> real grid came in at 3 failures out of 10,560 — 0.03 %, all Whisper.*
>
> *And there's a defect I found in that path worth telling you about, because it's
> the same shape as everything else here. The cache is keyed on
> `(clip, condition, model)`. I once ran an arm without its API key exported, so all
> twenty cells failed with 'key not set' — and the cache stored those failures. The
> retry then printed `20 cached, 0 to run` and replayed the identical error. **A
> cached failure and a reproducible failure are indistinguishable to a `key in
> dict`.** The fix is that a cached failure no longer satisfies a lookup unless it's
> explicitly classified terminal, and the only terminal category is 'the vendor
> rejected the audio payload' — which is safe precisely because the degraded audio is
> byte-reproducible, so the rejected payload provably is the payload that would be
> re-sent."*

### (3) LIMITATION TO PRE-EMPT HERE

**Composition order is fixed and order effects are unstudied** (limitation 10). One
line, then move on.

---

## §4 — Knobs table + key numbers

### (1) VERBATIM OPENING CUE

> *"Five knobs. Reverb, delivered by real measured impulse responses. Signal-to-noise
> ratio, with three kinds of real recorded noise. A cheap-mic frequency rolloff. And
> a transmission codec. Cross those and you get **176 conditions**; times **40 clips**
> times **3 recognizers** is **10,560 transcriptions** in one table — three failures,
> 0.03 %. Total spend was about **$3.70** across 14,606 API calls."*

### (2) THE WHYs — this section carries the heaviest ones

**"Why 40 clips?"** — Answer with precision, not convenience. He will respect the
arithmetic.
> *"It's a precision target, not a budget. WER precision is governed by **reference
> words**, not clip count. Forty clips at about nine words each is **363 reference
> words in every condition.** Treating word errors as roughly binomial, the standard
> error on a per-condition WER of 0.20 is about **2.2 percentage points** — tight
> enough to separate adjacent grid cells, which is the entire point of a controlled
> rig. Drop to 15 clips and you're at ~3.5 points, at which neighbouring cells
> overlap and the sensitivity analysis is measuring noise. Go to 100 and you only get
> to ~1.4 points — square-root returns — while **every API call count multiplies by
> 2.5×.***
>
> *And 40 buys **coverage**, not just samples. The manifest deliberately spans
> personal names, spelled alphanumeric codes, phone and PO strings, addresses,
> currency, dosages, times. **Cutting clips cuts categories, not just n** — and the
> categories are the failure modes the fingerprint layer classifies."*

**"Why the same room, mic and distance for every recording?"**
> *"Consistency, so the only thing that changes between conditions is the synthetic
> degradation. Distance especially: the impulse responses **already** simulate
> room-and-distance. If I'd varied capture distance at record time I'd have injected
> a second, uncontrolled copy of a factor the RIRs are there to control. Same
> argument for the mic and the room — a corpus recorded half in the morning and half
> with the AC on has a hidden two-level factor in it that nothing downstream can see."*

Two supporting facts he'll appreciate, if there's room:
- **The capture chain is measured, not assumed:** room-tone floor **−52.9 dBFS**
  against a −60 target, dominated by 120 Hz mains hum. Constant across takes, so an
  offset rather than a confound — but it caps deliverable SNR at ~28 dB, **which is
  why the SNR axis stops at 20.** That number is a measurement, not a preference.
- **The clean floor is measured too:** clean-condition WER **1.65 %** — 6 errors
  across 40 clips, 363 reference words, 5 clips non-zero, **every one adjudicated by
  ear.** Never "fixed" the manifest to match what the model heard; that would be
  training the ground truth on the system under test.

**"How were the 176 conditions formed?"**
> *"By crossing the knobs. The core is a **complete 4 × 4 × 3 × 3 factorial** on
> babble noise — reverb × SNR × codec × mic rolloff, 144 cells, **40 clips in every
> single cell** — plus a 32-cell engine and road arm.*
>
> *The payoff of it being **complete** is the part I'd point at. A complete factorial
> with equal cell counts admits a **finite, exact variance partition.** So my Sobol
> sensitivity indices aren't Monte-Carlo estimates from a Saltelli sampler — they're
> computed exactly by functional ANOVA of the measured grid. The partition check
> reads `sum(S_u) = 1.000000000000` with max absolute error `0.00e+00`. That's
> strictly stronger than what I originally planned, which was Saltelli on a fitted
> surrogate — that would have inherited the surrogate's bias **and** the sampler's
> variance."*

**"Why these three models?"**
> *"**nova-3** is the spine because it exposes native per-word confidence — without
> that there is no silent-failure question to ask. **Whisper-base** is the open
> baseline: it runs locally, costs nothing, and it's there to show I'm not
> API-dependent and that the instrument works without a vendor. **ElevenLabs Scribe**
> is a second commercial arm, added specifically so that 'commercial models know when
> they're wrong' wouldn't be a single-model claim.*
>
> *And the honest limit on that: **n = 3, with only one open model.** Any difference
> between the commercial pair and the open one is equally explainable as 'commercial
> models are better calibrated' or 'whisper-base specifically isn't.' Commercial-vs-open
> is **confounded** and I say so."*

**"Why does the bootstrap resample clips and not cells?"** — a genuinely good QA
question if he asks it.
> *"Because the same 40 clips appear in every one of the 144 cells, so the cell means
> are correlated **through the clips.** A cell-level bootstrap treats 144 correlated
> numbers as 144 independent draws and understates every interval. So I resample the
> 40 clips with replacement, recompute all 144 cell means from the resampled corpus,
> and redo the whole decomposition — every replicate is a complete, internally
> consistent alternate experiment.*
>
> *I checked what the wrong one would have bought me. Building the cell-wise
> bootstrap deliberately and comparing: on the two pre-registered factors the
> interaction interval comes out **32.5 % narrower for reverb and 44.1 % narrower for
> SNR.** So the wrong bootstrap would have manufactured significance on exactly the
> test I pre-registered."*

*(Provenance note for you, not for him: I re-derived those two figures today —
2,000 replicates, seed 0, `deadzone/analysis/sensitivity.py` primitives. SPEC C.4
records the original audit as "~35–45 % too narrow", which is the right order and
slightly loose on reverb. **This comparison is not persisted to any artifact.** If he
asks where it lives, say "I re-ran it, it's not in `results/` — it should be."
Note also `mic_rolloff` comes out 13.0 % narrower and `codec` goes the **other** way,
8.7 % wider; the effect is real on the big factors and not universal.)*

### (3) LIMITATION TO PRE-EMPT HERE

**The reverb axis is four rooms, not a sweep.** This is the sharpest version of the
sim-vs-reality exposure and it is much sharper than "it's a simulator". Say it here
so it isn't discovered later:
> *"One thing about that factor table that the table doesn't show you. `rt60` reads
> like a continuous axis from 0.2 to 1.0 seconds. It isn't. Each request snaps to the
> nearest **measured** impulse response, and the grid only ever resolves **four
> distinct rooms.** Sixteen are curated on disk; four are used. And three of the four
> are exotic — a Bar, a Campground Dining Hall, and a **Shower.** Not one of them is
> a car cabin, an office, a kitchen, or a phone held five centimetres from a mouth,
> which is to say **none of them is a place a voice agent actually runs.***
>
> *The fix is cheap and I priced it: the other twelve rooms are already on disk, so
> it's 12 rooms × 40 clips = **480 more calls, about 33 minutes of audio, roughly
> fourteen cents.** That's the top of my what-I'd-do-next list."*

---

## §5 — ▶ LIVE DEMO: `demo-listen` (THE HOOK)

**This is the strongest three minutes you have, and it works on this interviewer
specifically, because it makes him do QA on the model with his own ears and then
shows him the ears and the metric disagree.**

> ⚠️ **VOICE CHANGE — this segment is THE MOTIVATING HOOK, not a finding.** The
> script's closing heading is now *"THE QUESTION THAT STARTED THIS"* and it says in
> so many words that this segment "is the question it was built for, not one of its
> results." The old closing — *"you cannot QA a voice agent by listening to it"* —
> stated a verdict and **has been removed.** It is barred, because the
> pre-registration failed and the human half is n = 1 on clips selected *because the
> model tied on them.* Match that voice: you are manufacturing the question live, not
> demonstrating a conclusion.
>
> **The narrow claim that survives, and it is the only one to make:** a human had a
> **stated preference in 3 of 3** pairs while the model scored each pair **exactly
> equal** (0.333/0.333, 0.222/0.222, 0.250/0.250), and across all 40 clips the paired
> difference is **−0.0178 WER, 95 % CI [−0.065, +0.031]**, with **18 of 40** clips
> scoring identically.
>
> **Pair count:** the script now plays **2 pairs by default** (`DEFAULT_N_PAIRS = 2`,
> ~3 minutes including talk); the recorded session and the README's diagram are
> **3 of 3**. Either run `--pairs 3` to match the README, or say *"I'll play you two
> of the three"* — do not let the screen say 2 while your mouth says 3.

### (4) THE MARKER

```
[▶ now run demo 1]     make demo-listen
```

**Which pairs, and why that order.** The script's default order is **pair 2 (`u21`)
→ pair 3 (`u26`)**, with **pair 1 (`u40`) held in reserve**. That is not arbitrary:
pairs 2 and 3 are the two the single prior listener called **confidently**; on pair 1
he said in so many words that he was "not 100 % everyone would agree", so it is the
weakest place to open. The pair *labels* are the join key to `KEY.md`,
`DEMO_SCRIPT.md` and the sealed prediction — `BLIND_SHEET.md` lists its rows in
**play** order while keeping the labels unchanged, so nothing needs renumbering.

**Show that `PREREGISTERED_PREDICTION.md` exists and LEAVE IT CLOSED.** Say only
*"I've written down what I think you'll say."* Stating a prediction aloud before
someone judges is a demand characteristic, and this project's whole subject is not
fooling yourself with a number you wanted. It gets opened **after** he commits.

**Setup patter, said while it loads (it loads instantly, so this is really "before
you press enter"):**

> *"Before I show you a single number out of this project I want one judgement from
> you, made before you've seen anything that could steer it. Two pairs of short
> clips. Same voice, same sentence within a pair, damaged two different ways. For
> each pair, one question: which one is harder to understand? Replay as much as you
> like. 'About the same' is a real answer, and it's not a cop-out."*

**What to point at on screen:**
- Nothing, during the listening. Let him listen. **Do not narrate.** The script
  routes every pre-commit line through a redactor so nothing on screen names a
  condition, an SNR, an RT60 or a WER — that's structural, enforced by
  `tests/test_demo_listen.py` with a negative control, not by you remembering.
- After he commits, point at the **`WER 0.222 == 0.222`** line. Say: *"Identical. Not
  close — equal."*
- Then point at the two condition lines: **clip 1 is a good room with speech nearly
  buried in babble; clip 2 is a bad room with an almost silent background.** Opposite
  extremes of two different axes.
- Then the measured half: **paired difference −0.0178 WER, 95 % CI [−0.065, +0.031],
  over all 40 clips, and 18 of the 40 clips score exactly equal.**

**The condition card — have these two rows cold, they are the whole contrast:**

| | condition | room / mechanism | mean WER, 40 clips |
|---|---|---|---|
| **A** | `rt60-1_snr-20_babble_none_roll-0` | Shower IR, measured RT60 **1.011 s**, DRR **−10.02 dB** — but *quiet* | **0.1123** |
| **B** | `rt60-0.2_snr-0_babble_none_roll-0` | Restaurant IR, measured RT60 **0.193 s**, DRR **+16.90 dB** — speech buried | **0.1301** |

**Neither has a codec or a mic rolloff on it, so nothing else moves between them** —
this is a two-point counterfactual, not a comparison of two messy conditions. Paired
over the same 40 clips: **−0.0178, 95 % CI [−0.0654, +0.0310]**, a **10,000-resample
paired bootstrap over clips**, spanning zero. Say "paired bootstrap over clips" out
loud — the clips are the resampling unit everywhere in this project and he will
check that it is consistent.

**"18 of 40 tie" — say the next sentence too, or it sounds like 18 easy clips.**
**Four of the eighteen ties are non-zero:** `u40` 0.333, `u26` 0.250, `u21` 0.222,
`u10` 0.125. On those the model did not get both right — it got both **wrong, by the
same amount, in different places.** The transcripts make that better than the scalar
does, and **these are two of the clips he will have just heard**:

| clip | reference | A (reverb) | B (babble) |
|---|---|---|---|
| `u21` | *"…and legal **by monday**"* | `…and legal filing` — 1 sub + 1 del | `…and legal` — 2 deletions |
| `u26` | *"…for the **kowalski wedding**"* | `…for the cool seaway` — 2 subs, the name mangled into words | `…for the` — 2 deletions, the name simply gone |

Both rows: **WER identical** (0.222 and 0.250 respectively). **Same WER, different
failure *shape*** — which is the argument for typed edits over a scalar, made
audible on clips he has in his ear. It is also the cleanest bridge into §9.

**The line that lands it — the question, not a verdict:**
> *"You had a preference. The model doesn't — not a small difference, none. Your ears
> and that number aren't measuring the same thing, and I couldn't check which one was
> right. **That gap is the question this whole thing was built to answer**, and
> everything after this segment is the instrument, not one of its results."*

**If he ranks them equal, nothing breaks — and say that rather than looking
deflated.** One listener agreeing with the model is worth exactly as much as one
disagreeing with it, which is precisely why the interval underneath is the measured
half and this half is only the question. Then go to Beat 2, which needs no ranking
from anyone.

### (2) THE WHYs

**"Is this a finding?"** — **NO. Say so before he asks.** This is the single most
important discipline point in the demo.
> *"This half is n = 1, unblinded to the hypothesis, not counterbalanced, not
> level-matched, and the three clips were **selected precisely because the model tied
> on them.** It is an intuition pump and it is not data. The measured half — the
> paired 40-clip interval — is the result, and it would read the same if you'd ranked
> them the other way round, which is what actually happened when I ran it."*

**"You sealed a prediction about this — how did it go?"** — It **failed**, and the
failure is better material than the success would have been:
> *"It failed, and the rubric was worse than the miss. I sealed a prediction that a
> listener would find the babble clip clearly harder. The listener went the other way
> in two of three pairs. But the sealed 'what each outcome means' section listed
> exactly two outcomes — they rank a pair unequal, or they rank it equal. It never
> considered **'unequal, but backwards.'** So under the rubric as written, a miss
> scores as a **pass.** **A pre-registration that can't fail is decoration,** and I
> wrote one, in the project about not fooling yourself with a number. It's in the
> deliverable with the three rules I took from it."*

**If he asks for the mechanism — DECLINE IT.** The script declines it out loud and so
should you:
> *"There's a tidy mechanism I could hand you. I'm not going to. I thought of it
> after seeing the result, one listener can't settle it, and reaching for it is
> exactly the move this failure should make you distrust."*

**Two presenter notes that used to print on screen and now live in
`report/_demo_internal_notes.md`. Know both — this doc is the only place it is safe
to name them.**

1. **Do not repair the failed pre-registration with the DRR result on stage.** The
   tempting rescue is: the reverberant clip sounds harder *because* it has a DRR of
   −10 dB against the babble clip's +17 dB, and my own grid says damage is monotone
   in DRR. It fits the result now in view, and it is **post-hoc** — thought of after
   seeing the outcome, on a result one listener cannot adjudicate. **The model-side
   DRR result is separately real and measured, and it has its own exposure (§A/Q1,
   n = 4, no interval, C50 one swap behind). It is not a rescue for an n = 1 human
   result.** The note itself was removed from the terminal for a second reason worth
   knowing: it *named the mechanism it was telling you not to reach for*, on screen,
   which handed the room the post-hoc story anyway. **A stage direction that leaks
   its own subject is worse than no stage direction.**
2. **Do not read an edit-type signature off these three clips.** They happen to show
   **substitutions**, while the grid-level fingerprint for `rt60 ≥ 0.7` is
   **deletions** — the opposite. If he asks "so reverb causes substitutions?", the
   answer is *no, and these three clips are not the evidence either way*; the
   per-factor signatures are in `results/fingerprints.json` and README §8, computed
   over the full grid rather than three hand-picked pairs. **Say it if it comes up;
   do not pre-empt it.**

**Two on-screen lines that are NOT stage directions — if either appears, stop.**
`(these two are not equal — this demo set has drifted from the grid table)` and
`⚠ manifest and master.csv disagree on the paired difference …` both mean the demo
set and the grid have drifted apart, i.e. **the segment's premise is gone.** Say so
out loud and move on; do not improvise around it.

### (3) LIMITATION TO PRE-EMPT HERE

Covered above — say "n = 1, intuition pump, not data" *inside* the demo, not after.

### FAILURE MODE

- **No audio device / no sound on the call:** run `make demo-listen` with
  `--replay` — it plays back the recorded 2026-08-05 session's answers and never
  waits for input. Rehearse in that mode. If screen-sharing audio is unreliable, do
  the whole beat in `--replay` and say so: *"I'll play you the recorded session
  rather than fight the audio share."*
- **Preflight:** `./.venv/bin/python demos/demo_listen.py --check` prints `READY` and
  names any missing file. Run it before the call.
- **It waits for a human, so it is deliberately NOT in the `make demo` chain.** Don't
  chain it.

---

## §6 — The dead-zone finding, nova-3

**Lead with the CONTINUOUS result. Do not open on "2 of 176."**

### (1) VERBATIM OPENING CUE

> *"Here's the headline, and I want to give you the robust form of it first because
> the eye-catching form is fragile.*
>
> *Across the **169 of 176** conditions where nova-3 emitted any words at all, its
> own mean word confidence tracks its actual error rate at Spearman
> paired ρ = **−0.980**. So globally the model **does** know when it's failing. And
> it is nonetheless **overconfident in 154 of 169** conditions — **91 %** — with
> mean gap **+0.147**. Those three numbers need no threshold and they're the
> finding.*
>
> *The danger isn't that the model is blind. It's that it's **mostly self-aware** —
> which means any system that tunes its 'ask the caller to repeat' threshold on the
> model's average behaviour will trust it precisely in the residual where it
> shouldn't."*

Then, and only then, the count:

> *"If you put a threshold on it — WER at least 0.30 and confidence in the top 40 %
> of that model's own distribution — you get **2 of 176 conditions** as genuine dead
> zones, a rate of **1.14 %** (2 of 176, 40 clips). The worst is reverb 0.45 seconds,
> **SNR zero decibels**, engine noise, a G.726 phone codec:
> mean word confidence **0.829** at WER **0.306**, and **zero of forty clips came
> back empty**, so the confidence and the WER are averaged over the same clips and
> the claim needs no asterisk."*

### (2) THE WHYs

**"Two out of 176 at thresholds you picked. Is there a finding here?"** → **This is
Q2. Full verbatim answer in §A. Concede the count, pivot to the continuous form,
then give the methodological finding, which is the real one.**

**"How do I know 0.829 is high?"** — an excellent question and the project used to
have no answer. It does now.
> *"Fair, and until last week I couldn't have told you. I've since characterised the
> signal. The clean corner of the grid reads **0.962** and the raw undegraded
> recordings read the same, 0.962 per clip. The most confident condition is 0.981 and the
> least is 0.422, so the measured span is about 0.54. That puts 0.829 at the **64th
> percentile** of nova-3's own distribution — which means a production system with an
> 0.85 threshold would **already** be catching it. That's an honest deflation of my
> own headline and I'd rather say it than have you find it."*

**"Your headline changed. Why?"** — **This is the best story in the project. Tell it
in full.** It is engineering, it is a bug, and it was caught by a human, not a test.
> *"The first published version said **6** dead zones at a mean gap of 0.256. It was
> wrong, and the defect is one every engineer recognises.*
>
> *Per-condition confidence was averaged over only the clips that **produced words.**
> Per-condition WER was averaged over **all** clips — including ones that came back
> with an empty transcript, which score WER 1.0 and carry **no confidence at all.**
> Then I subtracted them. Neither average was wrong on its own. The defect lived
> entirely in the **subtraction.** Right row count, no NaN, no exception, and **no
> failing test.***
>
> *Scale: **2,210 of 7,040 nova-3 rows are silent — 31 %** — spanning 123 conditions.
> Mean gap inflation +0.109, max +0.524. It manufactured **four of the six** headline
> conditions.*
>
> *What found it: I finally sat down and **listened** to the dead-zone exemplar clips,
> and they sounded intelligible. That's what prompted the check. The last unchecked
> item on my own list was the one that found the headline error."*

**The ρ correction has its own mechanism, and it is the better half of the story —
have it ready, it takes fifteen seconds.**
> *"The published correlation was **−0.957**, computed over all 176 conditions while
> reporting n = 169. The 7 mute conditions have no confidence at all, so they enter
> at percentile 0 with WER 1.0 — **seven fabricated points parked at the ideal corner
> of a negative correlation.** Removing them makes the correlation **stronger**,
> −0.980, and that is the tell: a point whose removal *improves* the fit was never a
> measurement. −0.957 sat between the two honest numbers, which is exactly why
> nothing looked wrong."*

**The payoff exhibit for the mute category — it is the same condition as the old
#1 dead zone, so it pays off twice.** `rt60-0.7_snr-20_babble_opus-lowrate_roll-1`:
**SNR is 20 dB — it is quiet**, the damage is reverb + codec + rolloff, and you can
hear a person speaking clearly. **10 of the 40 clips returned nothing at all.** On
`u03` the model returned an empty string: WER 1.000, all **11** reference words
deleted. On the 30 clips it did speak on it was **81.8 % accurate at 0.843
confidence — well calibrated.** That is the condition the defect was hiding behind.

**"So how did you fix it so it can't come back?"** — this is the part he'll grade you
on, and it's strong.
> *"Not by picking the right one — by making the wrong one impossible to produce by
> accident. Both pairings are now stored and published side by side: `wer_spoke` and
> `wer_all_clips`, `gap_spoke` and `gap_all_clips`, plus `gap_inflation` as a
> first-class field so the size of the mismatch travels with the number. `gap` is an
> alias of the **same-population** one; the mismatched quantity exists only under an
> explicit name. The estimand is named **at the call site** — the functions take a
> `wer_key=` argument, so a reader sees which population a number describes without
> reading the producer. And `find_dead_zones` was demoted to a thin view over
> `classify_conditions`, so **you cannot obtain dead zones without also being handed
> the mute zones.***
>
> *The taxonomy that fell out of it is the operationally useful part: **2 dead
> zones**, **4 silence-driven**, and **7 mute zones** where the model returns nothing
> on any clip. A mute zone is not a dead zone — confidently wrong and entirely absent
> are different failures with different fixes — and critically, **a confidence-based
> monitor is structurally blind to a mute zone**, because there's no confidence to be
> low. So the early-warning signal I'm proposing cannot see its own worst failure
> mode, and I can tell you how big that hole is: deletions are 69.3 % of all errors."*

### (3) LIMITATION TO PRE-EMPT HERE

**The count is threshold-fragile, and there is now an artifact that says so.** Get
there before he does — this is the single most likely place to be caught.

> *"And I should hand you the sensitivity analysis I didn't originally ship. Over a
> defensible threshold box — WER from 0.10 to 0.50, confidence percentile 0.30 to
> 0.90 — the nova-3 count runs from **0 to 86**, median 2, and only about **5 % of
> the 63 grid points** return the published 2. One step either way in either
> direction gives you 0 to 22. The published members survive only **32 %** of the box
> and the robust core — conditions flagged everywhere — is **empty.***
>
> *One thing that partly rescues it: both tests are **monotone**, so the flagged sets
> are **nested.** Loosening a threshold can only **add** conditions, never swap one
> for another. So membership accretes rather than churning, and the honest statistic
> is **persistence** across the box, not the scalar count. Which is another way of
> saying the count is an operating point, not a measurement."*

**Also worth pre-empting:** the two headline dead zones are both at **SNR 0 dB**, the
harshest level in the grid, and both at the **low** end of the reverb axis — not
where you'd guess.

---

## §7 — ▶ LIVE DEMO: `make demo` (THE HERO)

### (4) THE MARKER

```
[▶ now run demo 2]     make demo
```

> ⚠️ **`make demo` CHANGED. It is now the LIVE beat, not the offline chain.** It runs
> `demos/demo_hero.py` — the merge of the old `demo_break` (audio + cached numbers)
> and `demo_live` (two real calls, no audio). **The interviewer picks the clip from a
> menu of measured dead zones**, then hears the raw recording and the degraded one,
> and **every number on screen in the live path comes from the two responses that
> just arrived.** ~2 min.
>
> - **`make demo-replay`** — the identical beat entirely from cache, **no network at
>   all.** This is rehearsal mode *and* the instant fallback.
> - **`make demo-all`** — the old chain: `test-core` → hero → `demo-al` → dashboard.
> - **`make demo-break` / `make demo-live`** — the two halves separately, kept as
>   fallbacks.
>
> **The offline guarantee was not dropped, it was moved.** No key, no network, a
> vendor error, a timeout or a missing SDK each print **one** explanatory line, fall
> through to the archived measurements **clearly labelled CACHED**, and **exit 0**. A
> wifi outage cannot take the demo down. Also: stdin is never required — EOF, Ctrl-C,
> a pipe or a non-tty each finish the beat and exit 0 rather than blocking in front of
> an audience.

**Decide before the call which one you run.** Live is a materially better demo —
*"the one thing a cached demo cannot show is the payload arriving"* — but it is the
only beat that touches the network. If the call is on flaky wifi, run
`make demo-replay` and **say so**: *"I'm running this from cache so we're not
watching a progress bar; `make demo` makes the two real calls."*

**Setup patter, before you press enter:**
> *"One recorded sentence, played twice. First the raw recording, then the same
> sentence pushed through a measured dead zone — a real room, real recorded noise at
> a chosen level, a real telephony codec. **You pick the clip.** Watch the two numbers
> move at different rates: the transcript is going to collapse and the model's own
> confidence is barely going to notice."*

**Hand him the menu.** Six clips, each measured in a dead zone, each labelled with
what the condition destroys — a gate number, a name, a licence plate. **Let him
choose.** It is a small thing and it converts him from audience into operator.

Point at, in order:
1. **`[1] RAW RECORDING`** — *"The control. And note there is no 'clean' condition in
   this design: `apply_condition` **always** applies a room and always mixes noise,
   so the only true null is the untouched file."*
2. **The per-word confidence bars.** *"Those are exactly as returned — nothing
   averaged, nothing rounded for the slide."*
3. **`[2] DEAD ZONE`** — let the audio play, then read the two-row delta table:
   *"WER and confidence moved at completely different rates. If confidence tracked
   accuracy it would read about 0.22 on this transcript."*
4. **The punchline.** On the u08 exemplar: *"It emitted 'depart' where the speaker
   said 'b', and reported **0.975 confidence** in it — **more confident in the word
   it invented than in any word it got right.**"*
   **Know how that sentence is built**: it is a ladder, not a string. It prints only
   when the payload that just arrived supports it — the most confident wrong word
   must be absent from the reference **and** more confident than every correct word.
   When the live call doesn't support it, the script prints the strongest sentence it
   *does* support and says so. **A demo about silent failure is not allowed to state
   a claim its own data contradicts.** If it downgrades live, say that out loud — it
   is a better moment than the punchline.
5. **`AND IT IS NOT ONE UNLUCKY CLIP`** — and **read the population line**, do not
   skip it: *"averaged over the 39 of 40 clips this condition produced words on — one
   came back empty and carries no confidence, so it sits out of both averages."*
   That line is not decoration; **it is the estimand, and this project published a
   wrong headline once for want of it.**
6. **`DOES IT REPRODUCE?`** — the live figures against the archived grid row.
   *"A commercial model literal is updated server-side, so a live call is not
   automatically the same experiment as the grid."* In replay mode they agree by
   construction and the panel says so — don't oversell it there.

**Then `make demo-al` (~11 s)** if you're running the full path. Frame it as a null
*before* it finishes:
> *"This one is a negative result and I'm showing it anyway. A surrogate predicts WER
> from the condition parameters and picks what to measure next, trying to walk onto
> the failure boundary. Watch the picks concentrate — the mechanism demonstrably
> works, straddle acquisition puts far more of its budget near the threshold than
> uniform random does. **And it doesn't pay.**"*

**Then the dashboard (4.5 min; the scripted path is in `dashboard/DEMO.md`).**

### ⚠ TWO ON-SCREEN LANDMINES — verified on disk 2026-08-06, both silent

Neither of these throws, neither looks wrong, and both put the **wrong arm's
headline** in front of the room.

1. **The dashboard's model buttons render ALPHABETICALLY, so the leftmost button is
   NOT the selected arm.** `default_model` is correctly `nova-3`, but the buttons
   come from the payload's key order — **`elevenlabs-scribe` · `nova-3` ·
   `whisper-base`** — so **nova-3 is the MIDDLE button** and the highlighted one.
   **Do not click blind**; read the `aria-pressed` highlight, or click `nova-3` by
   name. Clicking leftmost silently switches you to Scribe, whose sensitivity and
   sim2real panels are (correctly) empty.
2. **`results/confidence_gap.txt` opens on ELEVENLABS-SCRIBE, not nova-3.** The file
   has no default and no ordering guarantee — Scribe is simply the first block, and
   **nova-3's block does not start until line 47.** Scribe's headline reads *"mean
   word confidence **0.976** while WER is **0.428**"*, which looks spectacular, is a
   real number, and **is not nova-3's.** If you open that artifact on screen — or
   read it yourself on the morning — **scroll to the `model 'nova-3'` header first.**
   Worse, Scribe's dead-zone rate is the one figure in this project that is
   explicitly **not quotable** (§10), so the first block of that file is a number you
   are barred from saying.

### WHAT THE HERO DELIBERATELY DOES NOT PRINT — and what to say if asked

Four things were on screen in the old `demo_live.py` and are now off it. They are
real and they matter; they belong in the written record, not on a projector where
they crowd out the two numbers the beat is about. `tests/test_demo_hero.py` asserts
none of them can come back.

| stripped | if he asks, say |
|---|---|
| the cost line and the `$/min` vendor rate quoted with a date | *"It's in the freeze artifact — `results/MANIFEST.json` records the call counts, audio minutes and spend per arm, next to the model literal and the date it was priced at. That's a better answer than a number on a slide because it comes with its provenance attached."* Two calls cost a fraction of a cent; the whole three-arm grid was **14,606 calls ≈ $3.70**. |
| `run_id` and the row timestamp | *"`master.csv` carries `run_id` and `ts` per row and the manifest names the run. Every number the demo printed is re-readable from the table for that clip-condition-model triple — there's a test that asserts exactly that."* |
| the MANIFEST provenance paragraph | The panel keeps the **argument** (server-side literals move, so a live call is not automatically the same experiment) and drops the file path. |
| — | **What stayed on purpose:** the credential provenance line (`credential DEEPGRAM_API_KEY, from .env` — names the *variable*, never a value), the one-paragraph fallback notice, and the population line under the aggregate. |

**Never quote a cost from a demo screen or from a notes file. Quote the manifest.**

### FAILURE MODES

| Symptom | Do this |
|---|---|
| No wifi / no key / vendor error / timeout | **Nothing.** One line prints, it falls back to the archived measurements labelled CACHED, and exits 0. Read the yellow fallback paragraph aloud — it names the cause. |
| You want zero network risk | `make demo-replay`. Say you're running from cache. |
| `results/demo/hero/hero_cache.json` missing | `make demo-prep` — the `demo` target depends on it and rebuilds it automatically. |
| Any artifact missing | `make demo-check` names the missing file **and** the target that rebuilds it. Run it before the call, with wifi still on. |
| A dashboard panel shows a dashed box with text | **Read the text out loud.** It says exactly which payload was missing. It is a build-state fact, not a crash. |
| Panel 4 (sensitivity) blanks when you switch arms | **Expected and correct.** The exact decomposition needs the complete 4×4×3×3 factorial with 40 clips per cell — 5,760 transcriptions — and only nova-3 ran that. Say *"the decomposition is nova-3 only, by design"* and switch back. Serving nova-3's indices under another arm's label is precisely the mistake this project is about. |
| Panel 6 (sim2real) blanks when you switch arms | Also expected. The simulated-RIR arm was only ever run for nova-3, because that comparison is about **RIR provenance**, not model family. |
| Panels 7–8 don't move with the toggle | Correct — they're cross-model by construction. |
| Someone asks "is this real data?" | **Answer from the badge, immediately.** It reads `real grid · 12320 rows · 176 conditions`. Say it out loud in the first fifteen seconds; do not let anyone discover it at minute two. |

### ▶ OPTIONAL: `make demo-live` — and the `u11` exhibit, the best confidence material you have

`make demo-live` is the older, narrower half of the hero: **two real nova-3 calls on
one clip**, raw then the measured #1 dead zone, per-word confidences printed as they
arrive. **Operating data, so you can schedule it without guessing: 12.1 s of audio,
2 calls, ≈ $0.0009, ~20 s wall clock.** Rehearse it as
`./.venv/bin/python demos/demo_live.py --offline` — byte-identical presentation, no
key read at all — and preflight the real thing with `--check`.

**Placement rule: after §5's payoff, or as the answer to *"can I see the
confidences?"* — never at the top.** The spine has to establish itself offline
before anything touches the network.

**The exemplar is `u11`, and the PER-WORD numbers are the point, not the utterance
mean.** Reference: *"deliver it to sofia martinez at eighty eight elm street"*. In
the dead zone (`rt60-0.45_snr-0_engine_g726_roll-0`) it returns *"deliver it to sofia
martinez at three eight l three"* — **a delivery address destroyed** — with:

| word | outcome | confidence |
|---|---|---|
| `street` → `three` | **wrong** | **0.933** |
| `elm` → `l` | **wrong** | **0.926** |
| `eighty` → `three` | **wrong** | 0.826 |
| `martinez` | **right** | **0.336** ← the lowest in the utterance |

Utterance mean **0.849**; clean control **WER 0.000 at confidence 0.961**
(`results/clean_baseline.csv`), so the control is exact.

**Deliver the HONEST framing — it is better than the triumphant one.** At the
*utterance* level confidence is informative in direction and useless in magnitude:
WER goes 0.000 → 0.300 while confidence moves only 0.961 → 0.849. At the *word*
level it is **worse than uninformative** — the **lowest** confidence in the utterance
(0.336) is on the one word the model got **right**, while **two of the three
substitutions score above the utterance mean.**

> *"On this clip the signal is wrong in both directions at once — a **false alarm**
> on the one word it nailed, and **silence** on the three it invented. A threshold
> tuned to catch this fires on `martinez` and lets `three eight l three` through.
> That's the argument for the calibrator, and it's also why I won't tell you
> confidence thresholding is sufficient."*

**This is the only worked example in the project where the signal fails as a false
alarm AND as a miss on the same utterance**, so it is the concrete answer to *"is
your confidence signal actually usable as a gate?"* — which is the question §A/Q3's
decision rule (precision 0.994, recall 0.249) answers only in aggregate.

**Why nova-3 only in this demo, if he asks:** not a preference. Scribe's orthography
is non-deterministic call to call, so a live call could return a *different
transcript for the same bytes* on stage — which would be a fascinating thing to
discuss and a terrible thing to have happen mid-punchline. Whisper is local and has
no wire to watch.

---

## §8 — Model comparison (nova-3 / Scribe / Whisper)

**Every number in this section is on the 10-clip intersection. Say "on the ten clips
all three arms ran" out loud at the start and you will not have to say it again.**

### (1) VERBATIM OPENING CUE

> *"Three arms. Everything here is on the **ten clips all three ran** — nova-3 also
> ran forty, so its numbers move between this section and the last one, and that's a
> population difference, not a contradiction. On this subset nova-3's dead-zone rate
> reads **0.57 %** (1 of 176, 10 clips) where the 40-clip corpus number was 1.14 %.*
>
> *The comparison that matters is **not** that Whisper is worse. It's that
> **Whisper is worse at knowing it's worse.** Its confidence-vs-WER shape is
> whisper-base ρ = **−0.590** (n = 171) against nova-3 ρ = **−0.970** (n = 164) —
> and its dead-zone rate is **39.20 %** (69 of 176, 10 clips) against nova-3's
> 0.57 %."*

### (2) THE WHYs

**"How do you compare confidence across models at all?"** → **This is README §10 and
it deserves its own beat. Don't answer it fully here; say "that's the next section"
and move.** If he pushes, the short form: *"I don't pool them. Comparison is strictly
within each model's own distribution, and it's enforced by a raise in code."*

**"Whisper turned 3 words into 49?"** — the exhibit. Show it if it's on screen.
> *"That's the mechanism difference, and it's more interesting than the magnitude.
> **Under stress nova-3 goes quiet and Whisper invents.** Deletions are comparable —
> 0.270 against 0.289 of reference words — but substitutions are 2.8× and insertions
> are **9.4×**. At the extreme, the 11-word reference 'call maria at four zero five
> nine one two seven seven' comes back as a **47-word** degenerate repetition loop
> about having a file. And one cell came back as a row of
> decorative Unicode glyphs — not language at all — at **0.926 confidence.***
>
**The half of the arm comparison that is immune to ALL of the orthography argument —
and it inverts the safety story. Volunteer it.** An empty transcript is empty under
any normalizer.
> *"On the ten clips all three ran, nova-3 returns **nothing at all on 24.5 %** of
> clip-rows (431 of 1,757) and goes fully mute on **12** conditions; Scribe on
> **4.4 %** (78 of 1,757) and **2**. That's **5.5×.** Under stress **nova-3 goes
> quiet and Scribe keeps talking** — and a deletion carries no hypothesis token, so
> **63.2 % of nova-3's errors carry no confidence at all**, against
> **33.7 % for Scribe.** So the best-calibrated arm in the study has the **least
> monitorable** failure mode. That isn't a knock on the model — it's why I split mute zones out as
> their own category, and it's why the product recommendation is confidence **plus a
> 'did I get anything' check**, never confidence alone."*

> ⚠️ **POPULATION.** 63.2 % is the **10-clip matched** figure. On nova-3's own
> **40-clip** corpus the same quantity is **69.3 %** — the number §A/Q3 and §C quote.
> Both are correct; they are different populations and they are 6 points apart.
> Scribe's 33.7 % has only one population, because Scribe only ever ran the 10.

**And it qualifies the correlation table mechanically — say this before he finds
it:** *"nova-3's ρ is computed over **164** conditions after its **12** hardest were
dropped for emitting nothing; Scribe's over **174** after **2**. A model that goes
silent on its worst conditions is scored on an easier set. That's exactly why the
three-arm table is restricted to the 159 conditions all three spoke on."*

> *The point for a deployment: **WER structurally understates that.** WER caps damage
> at one error per reference word. A 47-word hallucination handed to a downstream LLM
> is unbounded harm. That's an independent argument for why WER isn't the deployment
> metric — separate from the entity argument in the fingerprints."*

**"Why is Scribe excluded from the WER comparison?"** — Answer with the property,
then concede the convenience.
> *"Because its orthography is **non-deterministic across identical calls.** Four
> repeat calls on byte-identical audio returned different transcripts on **5 of 6**
> entity-bearing probe clips — `A7X42` versus 'A seven X four two', `Q9J05` versus
> 'Q nine J zero five', and one clip flips the other way. That's worth up to **0.727
> strict WER on identical input.***
>
> *The distinction that makes the exclusion principled rather than convenient:
> Whisper's orthography offset is a **constant**, +0.090, and nova-3's is −0.014,
> essentially zero as predicted — which is the audit that validates my normalizer. A
> constant can be characterised once and subtracted. **A per-call draw is variance,
> not bias, and cannot be subtracted.** It's enforced in code — the cross-model WER
> paths raise on an incomparable arm, and there is no flag that includes them — and
> the arm is not silently dropped: it stays in every within-model analysis.*
>
> *There's an independent measured check too. Scribe's rank correlation **moves**
> under normalization, −0.820 to −0.948, while nova-3's does not move at all,
> −0.970 to −0.970. A rank correlation is invariant to a constant offset and
> attenuated by a per-call one. **That's the signature of noise rather than bias, and
> it's measured rather than asserted.**"*

**"Why ElevenLabs at all?" — answer with the DAY-ONE GATE, not the results.** It is
about method, and it is the better answer to that question.
> *"Before a single row entered the table I ran a one-clip, one-call probe, because
> the answer changed what the arm **is**: with per-word confidence it's a full arm and
> joins the headline; without one it's WER-only. It returns a per-word **`logprob`**,
> so it's the second arm in the study that can be asked the silent-failure question
> at all — Whisper can't, its per-word number is a derived proxy.*
>
> *Three things that gate caught, and each one is why it was a live probe and not a
> documentation read:*
>
> *One — **the vendor's own docs disagreed with each other.** The capabilities page
> shows a response with **no** `logprob`; the API reference lists it. One live call
> settles what no amount of reading would have.*
>
> *Two — **the trap I nearly walked into.** `language_probability` sits right there
> in the response on a friendly 0–1 scale. It is a **document-level language-detection
> score.** Mistaking it for confidence would assign **every word in a clip the same
> value** — a perfectly smooth, entirely fake confidence signal that would have
> correlated with nothing and looked completely plausible, and **no test in this repo
> would have caught it.** That's this project's signature failure mode, offered to me
> by a vendor's response schema on day one.*
>
> *Three — a word came back with **`logprob` exactly 0.0**, i.e. `exp()` exactly 1.0.
> Observed live, not hypothesised: `u02`, the word "at". Anything that then takes a
> **logit** of that confidence divides by zero, so the adapter clips — same class of
> bug as the calibration layer's `_logit` guard. You can see the clip in the shipped
> data: the arm's maximum confidence is **exactly 0.999999**, and that word is it.*
>
> *And cost was never the constraint — $0.22/hr batch, so the 10-clip arm that ran
> cost **$0.44** of the project's $3.70, and the full 40-clip grid would have been
> about **$1.76**. **The gate was the constraint, and it was the right one:** the
> orthography audit is what turned this from a third data point into a reversal."*

**Now concede, unprompted:**
> *"Two things I'd flag against myself. First, the exclusion is **also** the outcome
> that avoids an inconvenient ranking — Scribe scores 0.410 strict against nova-3's
> 0.433 on the same cells, so on the raw numbers it reads **better** than my spine,
> and the reason those numbers aren't comparable is real but was discovered on the
> arm that would have won. Second, **the evidence isn't persisted.** The repeat-call
> probe writes nothing to disk; the 5-of-6 count and the 0.727 exist only in prose.
> So the single fact justifying the exclusion of an entire arm is not reproducible
> from the repo. It's one command and a `json.dump` away and it should have been
> done. Also: I never ran the same test on Deepgram or Whisper, so the honest scope
> is 'measured on one arm, unmeasured on the others', not 'unique to this vendor.'"*

### (3) LIMITATION TO PRE-EMPT HERE — THE SPECULATION CALLOUT

**"Commercial models know when they're wrong" is NOT supported as a class claim. Say
so before he can construct it.**
> *"There's a tempting sentence here that I have to decline. On the 159 conditions
> all three arms spoke on, under **strict** scoring nova-3 separates from Scribe
> (+0.203 [0.115, 0.312]) and Scribe does **not** separate from Whisper (+0.074
> [−0.112, +0.267]). Apply the normalizer and the verdict **reverses**: Scribe
> collapses onto nova-3 (+0.035, interval barely clearing zero) and separates cleanly
> from Whisper (+0.227 [0.097, 0.376]).*
>
> ***Two scorings, opposite verdicts.** What survives under both is narrower than
> anyone would want: **nova-3 beats the open baseline.** 'Both commercial arms beat
> the open baseline' does not survive. 'Commercial models know when they're wrong' is
> not supported. And n = 3 with one open model means commercial-vs-open is confounded
> anyway."*

**The counter-example to your own slogan is now printed in the README's appendix, so
he will see it — get there first rather than being caught by it:**
> *"One against myself. The line I like is 'you cannot borrow someone else's
> dead-zone map,' supported by two Jaccard-zero results. But the same artifact shows
> **Scribe and Whisper share 7 dead zones — Scribe's seven are a strict subset of
> Whisper's sixty-nine, Jaccard 0.101.** So the defensible statement is that
> dead-zone maps transfer **poorly and unpredictably** — sometimes not at all,
> sometimes partially. I kept the slogan as a summary but printed the 0.101 next to
> it, because a categorical claim that costs you the room when someone finds the
> exception is a worse deal than the softer one."*

---

## §9 — Failure fingerprints → implied-fix table

**This is the most defensible layer in the project. Say so.** Typed edits, large n,
each signature mapped to a concrete engineering action.

### (1) VERBATIM OPENING CUE

> *"This is the layer I'd defend hardest, and it's the one that turns a benchmark
> into something actionable. **Stop counting errors and classify them.** Every
> condition gets an error signature from the aligned edits, and each signature
> implies a **different fix.** This is nova-3, 7,040 rows, 63,888 reference words.*
>
> *Headline: **deletions dominate.** Deletions are **0.351** of reference words
> against substitutions 0.136 and insertions 0.020. Deletion isn't one mechanism
> among several — it's **the** failure mode."*

### (2) THE WHYs

**"So what? What do I do differently?"** — the whole point. Have the table in your
head:

| factor | dominant edit | effect | implied fix |
|---|---|---|---|
| low SNR | **deletions** +0.344 | biggest single mover | front-end: gain / VAD thresholds |
| mic rolloff | **deletions** +0.264 | | front-end recovery |
| reverb | **deletions** +0.212 | | **dereverberation (WPE)** or closer / beamformed capture |
| `opus-lowrate` | **deletions** +0.111 | | front-end recovery |
| `g726` | **substitutions** +0.061 | | **entity-aware decoding + keyword boosting** |
| road noise | **substitutions** +0.059 | | entity boosting + condition-matched augmentation |
| engine noise, `codec=none` | fewer deletions | relative improvement | **NO FIX** — correctly emits none |

> *"The reason the split matters: **a deleted word never reached the decoder, so
> keyword boosting cannot recover it.** You need a front end. Whereas a substitution
> means the acoustic evidence arrived and got resolved wrong, which is exactly what a
> decoding-side prior can fix. So 'reverb hurts' and 'G.726 hurts' are the same
> statement in WER and **opposite engineering decisions** in the typed edits. That's
> the argument for the layer."*

> *"And keeping **two** codecs paid off: `g726` produces substitutions and
> `opus-lowrate` produces deletions. Two different mechanisms out of one factor — a
> single codec level would have hidden that."*

**"Do entities really degrade faster?"** — Yes, but **not the way people assume.**
> *"Entity error rate **0.633** against overall WER **0.511**, so yes, entities
> degrade faster than the transcript. But the destroyed-word table is the
> counterintuitive part: **proper nouns 0.646** and **spelled letters 0.613** are the
> most destroyed classes — and **digit words are the LEAST destroyed at 0.361**,
> below even function words at 0.462. So 'entities degrade faster' is carried by
> **names and spelled codes, not by numbers.** The naive read of that table gets it
> exactly backwards, and commercial models have clearly spent effort on digits."*

**"What about insertions under babble?"** — a nice mechanism catch:
> *"Insertions under babble are **92 % words that are absent from the reference** —
> the model is transcribing the **background talkers**, which is a completely
> different mechanism from acoustic confusion and implies a different fix,
> endpointing and speaker gating rather than denoising. Worth noting it's **not**
> babble-specific: engine is 0.94 foreign and road 0.89. Babble just carries about
> three times the insertions. So the claim is about insertions under *any* competing
> source."*

### (3) LIMITATION TO PRE-EMPT HERE

**Entity annotation is hand-authored, and the fingerprint layer inherits the
one-speaker limit.** Also: the implied fixes are **implied**, not tested —
> *"To be clear about what this layer is: it maps a signature to a fix I have **not
> run.** I'm not claiming dereverberation recovers X points; I'm claiming the error
> type rules out one whole family of fixes and points at another. Testing the fixes
> is a different project."*

---

## §10 — Confidence-comparability rigor callout

**Give this its own beat. It is a pure engineering-discipline section and it is
tailored to this interviewer.**

### (1) VERBATIM OPENING CUE

> *"This is a small section but it's the one I'd most want a QA person to look at.
> The obvious thing to do with three models that all return confidence is to put them
> on one axis. **I don't, and it's enforced in code rather than by convention.***
>
> *The three vendors' confidence numbers are on different internal scales — on this
> grid the per-word medians differ by roughly a **factor of three** across arms. So a
> shared absolute threshold is meaningless: pooling them compares **units**, not
> models. Every cross-model confidence statement goes through a **within-model
> percentile**, and the cross-model WER paths **raise by default** on an arm marked
> incomparable. There is no flag that includes them."*

### (2) THE WHYs

**"How do you know the normalizer is doing what you think?"** — the audit is the
answer, and he will like that it's a gate:
> *"There's a gate on every new arm. Cross-model scoring re-scores **every** arm,
> including the ones I expect not to move. nova-3 is already word-form — its adapter
> disables smart formatting, punctuation and numeral conversion — so its shift
> **should** be about zero, and it is: **−0.014.** Whisper's is **+0.090**, which is
> the normalizer recovering its digit orthography, not inventing accuracy. **A large
> nova-3 shift would have meant the normalizer was changing more than spelling** —
> that's the check, and it's what makes the +0.090 quotable."*

**"Why not just add digit mapping to your main scorer?"** — a good trap to have
already thought about:
> *"Because the corpus itself uses contradictory conventions. `u02` is 'four zero
> five' → 405, but `u05` is 'fourteen hundred' → 1400 and `u11` is 'eighty eight' →
> 88. **No single rule is correct**, and guessing inside a trap function is exactly
> the failure this project is about. So the cross-model normalization is a separate
> module used **only** for the L1 comparison, the trap function is left alone, and
> the three residuals that survive are **pinned by tests rather than patched.**"*

**"Is a within-model statistic automatically safe?"** — **No, and this is the sharp
bit. Volunteer it.**
> *"No — and I got this wrong once. I assumed 'within-model' was sufficient for
> Scribe. It isn't. **The dead-zone flag is a within-model statistic that thresholds
> an ABSOLUTE WER**, so it is not scale-free. Scribe reads 7 dead zones under strict
> scoring and **0** under the normalizer — all seven fall from WER 0.30–0.43 down to
> 0.08–0.14. They were orthography, not confident error. So Scribe's dead-zone rate
> is **not quotable.** The rule I took from it: admit a contaminated arm to **rank**
> statistics, never to **level** statistics — and check whether a statistic you called
> 'within-model' is secretly a level statistic."*

**And take it one step further than he will — it indicts your own headline metric,
which is the strongest version of the point.**
> *"It isn't only Scribe. **No arm's dead-zone rate is scale-free**, because
> `dead_zone_flags` thresholds an absolute WER. Re-scored through the normalizer,
> Scribe goes **7 → 0**, nova-3 **1 → 0**, and **Whisper 69 → 44.** So the count I
> put in the headline moves under a change that isn't acoustic at all. It survives
> for nova-3 specifically because that arm's own normalization shift is −0.014,
> essentially zero by construction — which is the audit, not a coincidence. But the
> honest statement is that this is a **limitation of my own headline metric**, and
> it's the same class of finding as the estimand mismatch — caught before publication
> this time instead of after."*

**One framing caution, and it matters with this audience:** *"vendor confidence is
not a calibrated probability"* is the **premise** of the calibration layer, **not a
criticism of their product.** Nothing in the docs claims calibration and nothing
should. Say it that way — the layer asks *can a thin learned wrapper make it one*,
and the answer is yes.

### (3) LIMITATION TO PRE-EMPT HERE

**This is the natural place for the self-assessment line (§B).** The section is about
how carefully the signal was *handled*; the honest next sentence is that it was never
*opened up*. Segue:
> *"And the flip side of all that care is worth saying out loud, because it's the
> fair criticism of this project…"* → deploy §B.

---

## §10b — "Run it" + the repo map  (README §10 — a 30-second beat, don't linger)

Between the comparability callout and the appendix the README has a setup block and a
repo map. **It is the one place where a backend engineer gets to see how the thing is
laid out, so give it thirty seconds and make one point:**

> *"One structural thing worth calling out. Everything under `deadzone/` is
> importable library code that is **free to re-run** — nothing in there spends money
> or writes an artifact. Everything that **spends money or writes an artifact** lives
> in `scripts/`. That boundary is deliberate and it's load-bearing: the grid runner
> and the budget-capped hunt live in `scripts/` and never under `analysis/`, because
> an API-spending path must not hide behind a module whose whole contract is that
> it's cheap to call in a loop. The budget ceiling is a checked invariant, not a
> comment."*

If he asks about setup: `python3 -m venv .venv`, install `requirements.txt`,
`make test` — every suite offline, no key, no network, no audio. `requirements.lock.txt`
pins the exact demo-machine environment.

---

## §11 — Appendix: things that didn't work

**Two nulls, both in the deliverable. Present them as a discipline exhibit, not an
apology.** Note the README puts this in a **collapsed `<details>` block** — so it is
there and honest, and he has to click. **If he doesn't click, open it yourself.**
Volunteering the failures is worth more than having them found.

### (1) VERBATIM OPENING CUE

> *"Two things in here failed and they're in the deliverable rather than deleted,
> because I think how you handle a null is more diagnostic than how you handle a
> result."*

### (2) THE WHYs

**Null 1 — active learning.**
> *"The idea: fit a cheap surrogate to what you've measured, let it choose the next
> expensive measurement, and map the failure boundary in fewer oracle calls than
> random sampling. **It lost.** Inside a 45-evaluation budget, the boundary-RMSE
> target was reached by **2 of 8 active seeds** and **4 of 8 random seeds.** Median
> evals-to-target is infinite for both arms, so **no ratio is reportable and none is
> claimed** — I report the budget. The winner flips between train/test splits: active
> wins 2 of 4 splits, 13 of 32 paired runs, median paired difference +0.003, and
> positive means active is worse.*
>
> *Two things that make it a null rather than a broken implementation. One, **the
> acquisition function demonstrably worked** — it put 58.3 % of its chosen
> evaluations near the decision contour against random's 30 %. It did its job and the
> job didn't pay. Two, **the synthetic control still passes**: on planted structure
> where the boundary is sharp, active reaches random's full-budget fidelity in 27
> calls instead of 45. So this is a method meeting a surface it has no purchase on —
> smooth, low-dimensional, wide boundary region.*
>
> *Provenance caveat I say out loud: **all 8 seeds ran against a surrogate oracle. No
> seed was confirmed end to end against the live API.**"*

**Null 2 — the obvious fix for null 1, which also failed. This is the best-engineered
thing in the appendix.**
> *"The null had exactly one obvious objection: **you gave the GP the wrong axis.**
> My own sensitivity work says RT60 mislabels the delivered acoustics and that
> direct-to-reverberant ratio orders the rooms perfectly. So the natural rescue is to
> re-run in DRR coordinates. **An objection with an obvious answer that you never run
> is the cheapest way for a null to be quietly wrong**, so I ran it. It changes
> nothing — 14 of 32 paired runs, median +0.000, wins **zero of four** splits.*
>
> *And **the negative control is the actual result.** I ran all 24 permutations of the
> same four DRR values — spacing held fixed, only which room gets which label varying.
> The **physically correct assignment ranks 18th of 24**, permutation p = 0.75.
> Seventeen arbitrary relabellings beat the right one. Across 44 parameterisations
> the median paired difference is −0.0001 and 23 of 44 favour active: a coin flip.*
>
> *The ceiling on that experiment, stated as a limit rather than a hedge: **the
> reverb axis is four discrete rooms**, so any reparameterisation is a relabelling of
> four points on a line. DRR cannot add information the grid never measured. Which is
> why the fix is **more rooms, not a better coordinate** — and I know that because I
> tested the better-coordinate hypothesis and it failed a 24-permutation control."*

**A near-miss worth telling if he engages** — it shows you check your own wins:
> *"On the headline split, DRR looked like a materially better surrogate coordinate —
> boundary RMSE 0.139 against RT60's 0.162 — and I nearly wrote that up as a
> secondary finding. It's false. Across all four splits the ordering **reverses** and
> RT60 has the lower mean. Only ~13 held-out conditions sit near the contour, so one
> split's fidelity is a 13-point statistic. **No absolute-fidelity claim for DRR
> survives the split check**, and the four-split table is now printed in the artifact
> specifically so the next reader can't repeat the mistake."*

**Null 3 (already told at §5, don't repeat it here) — the pre-registered listening
prediction failed and its rubric was unfalsifiable.** If you didn't get to it in the
demo, this is where it goes.

**The contrast to draw, and it's a good one:**
> *"So the project ran **two** pre-registrations. The one with a **numeric decision
> rule fixed in advance** — reverb × noise as a genuine interaction, committed
> `d8ddd4f` on 2026-07-27 before any audio existed — **confirmed**, with
> **ST − S1 = 0.128** for reverb and **0.112** for SNR against a threshold of 0.020,
> and the verdict deliberately uses the **wider, conservative** interval when the code
> computes a tighter correct one. The one with a **prose rule** failed, and the rule
> couldn't fail. That's the lesson, and it cost me nothing to learn it in public."*

**If he asks what "deliberately wider" means — the mechanism, per factor:** the
published interval adds the S1 and ST half-widths **in quadrature**, which assumes
they're independent. They are not: the bootstrap correlation is **+0.843** for `rt60`
and **+0.871** for `snr_db`. So the quoted interval is **2.49× wider than the direct
one for `rt60` and 2.70× for `snr_db`** — the code computes the tighter, correct
interval and **declines to use it**, conservative in the only direction that matters
for a pre-registered test. Both are persisted in `results/sobol.json`
(`s1_st_bootstrap_corr`, `gap_conf_ratio_quadrature_over_direct`).

> ⚠️ **Do not say "~2.5× wider" as a single figure** — that was a summary-log
> approximation and it does **not** hold across the factor table: `mic_rolloff` is
> 2.10× and `codec` only **1.27×**. Same trap as the 4.5×-vs-3.58× clearance in §C:
> an `rt60` statement quoted without its factor name.

### (3) LIMITATION TO PRE-EMPT HERE

**The AL null is a surrogate-oracle result** (say it every time) and **the DRR
mechanism claim itself is n = 4** — which is Q1. If he opens on the DRR table here,
go straight to §A/Q1.

---

# §A — The three killer questions, verbatim

## Q1. "Your best mechanistic claim is n = 4. And your own table has C50 at −0.800. Why is DRR the story?"

**This is the sharpest attack available and it uses a column you published. Do not
defend the strong form.** Concede in the first sentence.

> *"You're right, and I should state it that way in the document. **n = 4.** The exact
> one-sided permutation p is **0.042** and Kendall two-sided is **0.083**, and it is
> the only claim in the write-up **without an interval.** C50 sits at **−0.800**, the
> same magnitude as RT60's +0.800 in the other direction, and the entire separation
> between DRR and C50 is **one discordant pair** — Bar versus Campground Dining Hall,
> whose C50 values differ by **0.19 dB.** So the DRR-versus-C50 ordering is **not
> established by this data.***
>
> *I'd also flag something the write-up should say more loudly: 'early-to-late energy
> ratios beat T60 as an intelligibility predictor' is **decades-old room acoustics.**
> It is not my discovery. This project claims no methodological novelty anywhere else
> and that's the one place the positioning leaks.*
>
> ***What actually survives is one level up, and it's the more useful finding.** My
> rt60 axis snaps each request to the nearest measured room, so `rt60 = 0.45` is a
> Bar and `rt60 = 0.7` is a Campground Dining Hall — unrelated rooms with different
> direct-to-reverberant ratios. So the **non-monotonicity along that axis isn't a
> property of reverberation at all — it's a property of which four rooms I curated.**
> Re-sample the axis and the dip moves, which is exactly what happened between two of
> my own scans: **zero of six** surrogate-proposed counterintuitive cells reproduced
> against the real oracle, and the reason is that the two scans walked
> almost-non-overlapping room triplets whose DRR ordering predicted **opposite signs**
> — which is what was measured.*
>
> ***That's a warning about how reverb benchmarks are constructed, and it doesn't
> need n > 4.** Anyone parameterising a reverb axis by RT60 alone will mis-rank
> conditions for exactly this reason.*
>
> *And the fix is **more rooms, not a better coordinate** — I know that because I
> tested the better-coordinate hypothesis and it failed a 24-permutation control. Cost
> to settle it properly: the other 12 rooms are already curated on disk, so it's 480
> more calls, about 33 minutes of audio, fourteen cents. **That's the single largest
> piece of unearned confidence in this document and it's the cheapest thing on my
> list.**"*

**The four rooms, if he wants them on screen (population: all 176 nova-3 conditions):**

| requested | room | RT60 | DRR dB | C50 dB | marginal WER |
|---|---|---|---|---|---|
| 0.2 | Restaurant | 0.193 | **+16.90** | 28.10 | 0.2026 |
| 0.45 | Bar | 0.474 | −2.05 | 10.22 | 0.5559 |
| 0.7 | Campground Dining Hall | 0.680 | +4.26 | 10.03 | 0.4495 |
| 1.0 | **Shower** | 1.011 | **−10.02** | 2.12 | 0.7217 |

**The three rank correlations, which are the claim:** **ρ(DRR, WER) = −1.000** ·
**ρ(RT60, WER) = +0.800** · ρ(C50, WER) = −0.800. *"Damage is monotone in
direct-to-reverberant ratio and NOT in reverberation time — so a reverb benchmark
parameterised by RT60 alone will mis-rank its own conditions."* That is the sentence,
and it is useful to anyone building an ASR eval set, which is them. n = 4 stands.

*(Population trap in your own table: over the **babble-only 144-cell block** the same
marginals read 0.203 / 0.636 / 0.449 / 0.758. Both are correct. Say which one you're
quoting. The ordering — and therefore every ρ — is identical under both.)*

## Q2. "So your headline is two conditions out of 176, at thresholds you picked. Is there a finding here?"

**Concede the count immediately. Pivot to the continuous form. Then give the
methodological finding, which is the real one.**

> *"The count is fragile and I'd rather not lead with it. `wer_hi = 0.3` and
> `conf_pct_hi = 0.6` were defaults I never varied. I've since run the sweep: over a
> defensible box the nova-3 count runs **0 to 86**, median 2, and only about **5 % of
> 63 grid points** return the published 2. One step either way gives 0 to 22. The
> published members survive **32 %** of the box and the robust core is **empty.** I
> should have shipped that table with the finding.*
>
> *One structural thing partly rescues it: both tests are monotone, so the flagged
> sets are **nested** — loosening a threshold can only add conditions, never swap
> one. Membership accretes rather than churns, so the honest statistic is
> **persistence across the box**, not the scalar. Which is another way of saying the
> count is an **operating point**, not a measurement.*
>
> ***The threshold-free version is the finding.** nova-3's confidence tracks its own
> error at paired ρ = **−0.980** across the 169 conditions that produced words, and
> it is still **overconfident in 154 of 169** conditions — **91 %** — mean gap
> **+0.147**. No threshold anywhere in that sentence. So the model is **mostly
> self-aware**, which is the dangerous part: a system tuned on average behaviour
> trusts it in the residual. I'd also flag that my #1 dead zone's confidence is 0.829
> against 0.962 on the clean corner, so it sits at the **64th percentile** of that
> model's own distribution — many production thresholds would already catch it.*
>
> *There's a threshold-free defensive statement too, if you want the conservative
> read: **nova-3 has zero conditions where it is confidently wrong at WER ≥ 0.40.**
> That's real and needs no operating point.*
>
> ***But the finding I'd actually defend is the one the correction produced.** The
> published headline was **6** dead zones. It was wrong: confidence averaged over the
> clips that spoke, WER over all 40 including empty transcripts. Right row count, no
> NaN, no exception, **no failing test** — the defect was entirely in **subtracting
> two averages taken over different populations.** **A human listening to the exemplar
> clips found it; no test could.** And it forced a taxonomy that matters
> operationally: 2 dead zones, 4 silence-driven, and **7 mute zones** where the model
> emits nothing at all — and a confidence-based monitor is **structurally blind** to
> those, because absent is not wrong. **That's the deliverable: the early-warning
> signal I proposed cannot see its own worst failure mode**, and I can tell you how
> large that hole is — deletions are **69.3 % of all errors** and carry no confidence
> at all."*

## Q3. "What is Deepgram's confidence score, actually?"

**This is the question a DG Labs engineer is most likely to ask and the one you are
least able to answer from the inside. Do not bluff. The shape of the answer is: here
is what I MEASURED about how it BEHAVES, and here is what I still don't know about
what it IS.**

> *"Two halves to that, and I'll separate them: **what it is documented to be**, and
> **what I measured about how it behaves.** I can't tell you what it is internally,
> and neither can Deepgram's public docs.*
>
> ***What's published.** `confidence` appears at word and alternative level, defined
> identically for both: 'a floating point value between 0 and 1 that indicates
> overall transcript reliability.' That's the whole published definition — **product
> language, not decoder language.** No statement of what it's computed from, and **no
> claim that it's calibrated.** The launch material gives an audio embedding
> framework and a trained contextual mechanism for in-context learning at inference;
> no family, loss, decoder type or parameter count. Deepgram is assignee on an
> end-to-end-ASR-with-transformer patent with an autoregressive decoder and a softmax
> over vocabulary — **but a patent records what was filed, not what ships as
> `nova-3`**, so I'd treat it as house style, not a spec.*
>
> ***So the honest sentence is: the arm that performs best is the one I can say the
> least about.** By contrast, Whisper's is fully checkable — paper and code are
> public — and I did check it. Its per-word number is, from `whisper/timing.py`, the
> **mean over a word's subword tokens of the decoder's next-token softmax
> probability, conditioned on the tokens it already generated.** That's a joint
> acoustic-plus-own-context score, not an acoustic posterior. Which gives me a real
> hypothesis for why Whisper stays confident while hallucinating: **inside a
> repetition loop the preceding context is maximally predictive, so the score can
> stay high while the audio contributes almost nothing.** And the tell that OpenAI
> knows this: `transcribe.py` ships a **gzip compression-ratio threshold** alongside
> the log-probability threshold. **A gzip check exists because the log-probability
> doesn't reliably catch loops** — a loop is both highly compressible and highly
> probable under its own context. Their own model card documents both the
> hallucination and the repetition tendency, so what I measured is the documented
> failure mode, not a surprise.*
>
> *ElevenLabs documents its field as a per-word `logprob` — the same **kind** of
> quantity as Whisper's, on its own scale.*
>
> ***All of that is speculation about mechanism and I'd label it that way.** What I
> actually measured is behaviour:*
>
> ***One — the reference distribution.** The clean corner of the grid reads **0.962**,
> and the 40 raw undegraded recordings read the same, 0.962 per clip. The
> most confident condition in the grid is 0.981 and the least — among conditions that
> still emit words — is 0.422. So the measured dynamic range is about 0.54. Without
> that anchor nobody can judge whether my dead zone's 0.829 is 'confident.' It's the
> 64th percentile.*
>
> ***Two — it saturates.** 8.2 % of words come back at exactly 1.0 and 15.3 % within a
> thousandth of it. Those ceiling words are 99.9 % correct against an arm-wide
> emitted-word accuracy of 0.767 — so the saturation is **earned**, not degenerate.
> But tied words cannot be ordered by any threshold or percentile, which removes
> resolution exactly where a commit-or-reprompt rule needs it.*
>
> ***Three — and this one corrected me.** I aggregate per-word confidence with the
> arithmetic mean, and I assumed that was wrong: for a commit-or-ask-again decision
> the operational question is about the **worst** word in the utterance, especially
> when it's the phone number, so I expected the minimum or a low percentile to win. I
> tested it. **It doesn't.** Separating bad transcripts from good at WER ≥ 0.3, the
> mean scores AUROC **0.944**; median 0.938, p10 0.917, and min **0.877** —
> **significantly worse**, with the paired interval clear of zero. Nothing beats the
> mean, decided by the paired CI rather than by the ordering. So the default I never
> examined turns out to be right, and it's now a **measured result rather than an
> unexamined default.** I'll take being wrong about that.*
>
> *(Worth knowing: the low percentiles fall with utterance length for purely
> combinatorial reasons, so I printed a **confidence-free control** — `n_words` — in
> the same table, at AUROC 0.560. If the control had ranked high, part of min's
> apparent advantage would have been length rather than acoustics.)*
>
> ***Four — `utterance_conf`.** I captured a second, utterance-level score on all
> 10,560 rows and for a long time no analysis module read it. I've now checked it:
> for nova-3 it's genuinely **distinct** from the word mean — Pearson 0.926, mean
> absolute difference 0.071 — but it carries **no measurable advantage**, AUROC 0.936
> against the mean's 0.944 with the paired interval straddling zero. So leaving it
> unread cost nothing measurable. On the ElevenLabs arm it's redundant **by
> construction** — that adapter reuses the word mean because the vendor exposes no
> separate utterance score.*
>
> ***And the ceiling on all four of those:** every one describes only the words the
> model **emitted.** Deletions carry no hypothesis token and therefore no confidence
> — they're 35.1 % of reference words and **69.3 % of all errors.** A perfectly
> calibrated confidence converges on **emitted-word accuracy 0.767**, not on
> **reference recovery 0.513** — reading it as the latter overstates the system by
> 0.254.*
>
> ***What a thin learned layer does to it**, since that's the actionable part:
> ECE **0.0507** raw, 0.0346 under temperature scaling with T = 1.39,
> and **0.0077** feature-conditioned — on held-out **conditions**, with a grouped split, never a
> random word-level split, which leaks and whose symptom is a *better* ECE. And it
> learned a discount schedule you could ship: above rt60 = 0.7, discount reported
> confidence by about 0.07; above mic rolloff 0.5, by about 0.06.*

**He WILL notice Whisper is missing from that table. It is blocked, not computed —
own it, the answer is better than the number would have been:**
> *"**69 of 1,757** Whisper rows still have a hypothesis-word count that disagrees
> with the confidence-list length after re-alignment — `align_confidences` recovered
> **33** and could not recover those. `word_records` **refuses to zip**, because
> zipping binds confidences to the wrong words. The library does offer
> `on_misalign='skip'`, which drops them and hands me a number — and that number
> would print in the same column as the other two arms while being fit on a silently
> smaller and non-random set. **That's a protocol change disguised as a value.** So
> the row is blank and the reason is printed next to it. I'd rather show you a hole
> than a number I'd have to caveat."*
>
> ***What I can't tell you is why it behaves that way** — and that's the part I'd want
> to spend time on with someone who knows the decoder."*

**→ Then go straight into §B.**

### The confound to volunteer here — it cuts against your own headline

**This is the strongest thing you can say in this whole answer, because it argues
against the result that flatters you.** If he asks why nova-3 ranks first on every
confidence statistic, do not answer "because it's better calibrated."

> *"There's a confound I'd flag before I take credit for that ordering, and it isn't
> architectural. **A deleted word emits no token and therefore carries no
> confidence.** Deletions dominate nova-3's errors — 69 % of them — so **most of its
> failures are structurally excluded from the statistic that ranks it first.**
> Whisper's errors are insertion-heavy, so nearly every one of its failures has to
> carry a score that can be wrong. Part of that ordering is which failure mode the
> metric can see. That's testable: re-run the ordering against reference recovery
> instead of emitted-word accuracy — the repo reports both — and see if it holds.*
>
> *Second confound: `whisper-base` is **74 million parameters**; both commercial arms
> are undisclosed and near-certainly larger. So **'commercial beats open' isn't
> separated from 'large beats small'** either. `whisper-large-v3` on the same grid
> would settle it."*

**And the ordering is not as clean as it looks — know this before he constructs it.**
nova-3 ranks first on every confidence statistic, but **Scribe and Whisper swap
places depending on which statistic and which scoring**: Scribe is above on ρ and
ECE, Whisper is above on utterance-level AUROC. So *"commercial above open"* is
**not a robust ordering**, and an explanation that only produces it is explaining
something the data doesn't say.

*(Full sourcing, with DOCUMENTED / INFERRED / SPECULATION tags on every line, is in
`report/model_architecture_notes.md`. It quotes no figure from `results/` on purpose,
so there is nothing in it to drift.)*

---

# §B — The self-assessment line

**Deploy it BEFORE he offers it.** Best moments, in order of preference:
1. Immediately after the Q3 answer (it is the natural closing sentence).
2. At the end of README §10, as the segue out of the comparability section.
3. If he asks any variant of "what would you do differently" or "where are you weak."

> *"The fair summary of this project is that **the methods depth is ahead of the
> domain depth. I can build the instrument. I haven't yet spent enough time inside the
> model.***
>
> *The concrete form of that is §4.8 of my own notes: a carefully guarded pipeline
> built on a vendor scalar I never opened up. I'd read it as a pointer rather than a
> verdict — it says exactly where I'd spend the next two weeks."*

**Do not soften it, do not add "but".** It is the single most useful sentence in the
prep, it is true, and with an ex-QA engineer it reads as calibration rather than
weakness. If he agrees with it, that is the answer working, not the answer failing.

**If he asks "so what would those two weeks be?"** — have three concrete items:
1. **The survivor-bias fix.** Emission rate × calibrated per-word confidence — a
   single "expected words recovered" score that degrades toward zero exactly where a
   confidence monitor goes blind. Needs no vendor change.
2. **More rooms.** 12 rooms already on disk, 480 calls, ~$0.14, retires the n = 4
   DRR question and the nearest-match-snapping limitation together.
3. **A genuinely streaming arm.** `scribe_v2_realtime` exposes the **same per-word
   logprob over a websocket**, so the comparison would be batch-versus-streaming
   **within one vendor** — the only form of it not confounded by model family.

---

# §C — Numbers card (every figure with its population)

**D1 / headline — nova-3, 40 clips, 176 conditions**
- paired ρ = **−0.980** · all-clips pairing −0.952 · both at **n = 169**
- overconfident in **154 of 169** conditions — **91 %**; mean gap **+0.147**
- dead zones **2 of 176 conditions** = **1.14 %** (2 of 176, 40 clips)
- #1: `rt60-0.45_snr-0_engine_g726_roll-0` — mean word confidence **0.829** at WER **0.306**,
  **0/40 silent**. Identity check: 0.829 − (1 − 0.306) = 0.136 =
  stored `gap_spoke`. **Always verify a quoted dead-zone row with that identity, never
  by counting columns** — `rt60_measured` sits immediately left of `mean_conf` in
  `dead_zones.csv` and 0.680 reads exactly like a plausible confidence.
- #2: `rt60-0.2_snr-0_babble_opus-lowrate_roll-0.5` — conf 0.807, WER 0.319, 39/40 spoke
- categories: **2 dead zone · 4 silence-driven · 7 mute**
- silent rows **2,210 / 7,040 = 31.4 %** across 123 conditions
- threshold box: count **0–86**, median 2, published point returned at **5 %** of 63
  grid points, one-step range **0–22**, members persist **32 %**, robust core **0**

**Confidence characterisation — nova-3**
- clean corner **0.962** (WER 0.008) · raw undegraded recordings 0.962 per clip
- highest condition 0.981 · lowest scored condition 0.422 · drop 0.540
- saturation: 8.21 % exactly 1.0, 15.35 % within 0.001; ceiling words 99.9 % correct
- best aggregate = **mean**, AUROC **0.944**; median 0.938 · p10 0.917 · min **0.877**
  (significantly worse) · `utterance_conf` 0.936 · `n_words` control 0.560
- decision rule: flag lowest-confidence 10 % by mean → threshold 0.564, precision
  0.994, recall 0.249. **The worked counter-example is `u11` in §7** — one utterance
  on which the signal is a false alarm *and* a miss at once.

**§7's `u11` exhibit — nova-3, 1 clip, dead zone #1 (`rt60-0.45_snr-0_engine_g726_roll-0`)**
- wrong at **0.933** (`street`→`three`) · wrong at **0.926** (`elm`→`l`) · wrong at
  0.826 (`eighty`→`three`) · **right at 0.336** (`martinez`, the lowest in the
  utterance)
- utterance mean **0.849**; clean control WER 0.000 at **0.961**; degraded WER 0.300

**D2 fingerprints — nova-3, 7,040 rows, 63,888 reference words**
- del **0.351** · sub 0.136 · ins 0.020
- entity error **0.633** vs WER **0.511**
- destroyed: proper noun **0.646** · spelled letter 0.613 · content 0.530 · function
  0.462 · **digit word 0.361 (lowest)**
- babble insertions 92 % foreign; engine 0.94, road 0.89

**Sensitivity — nova-3, babble-only 144 cells × 40 clips = 5,760 transcriptions**
- exact partition: `sum(S_u) = 1.000000000000`, max abs error 0.00e+00
- **ST − S1 = 0.128** for `rt60` [0.091, 0.164] · **0.112** for `snr_db` [0.072, 0.152]
- also mic_rolloff 0.084 · codec 0.042; S2 rank 1/6 is `rt60 × snr_db`
- threshold fixed in advance 0.020; binding clearance is **snr_db at 3.58×** under the
  conservative quadrature interval (**not** 4.5× — that's rt60's, and quoting it
  without the factor name overstates by 28 %)
- total variance 0.126580

**Calibration (L2)**
- nova-3, 42,732 words, 169 condition groups, 5 seeds: ECE **0.0507** raw →
  0.0346 (T = 1.39) → **0.0077** feature-conditioned
- held-out clips robustness check: 0.0487 → 0.0396 → 0.0196
- Scribe 14,668 words: 0.1646 (upper bound) → 0.0755 → 0.0340
- **whisper BLOCKED**: 69 of 1,757 rows still have a hypothesis-word count that
  disagrees with the confidence-list length after re-alignment. `word_records`
  **refuses to zip** — zipping would bind confidences to the wrong words

**Sim2real — nova-3, 10-clip intersection, 176 paired conditions**
- level: sim **12.1 points optimistic** [−15.0, −9.6]
- order: rank correlation **0.873** (Kendall τ 0.698)
- dead zones real 1, sim 0, **Jaccard 0.00**, recall 0.00
- matched on **measured** Schroeder RT60, max |Δ| 0.017 s
- **19.9 is dead.** That was the unmatched-clip artifact — 7.8 points of pure clip
  difficulty.

**L1 — all three arms, 10 clips, 1,757 rows per arm**
- dead-zone rate: nova-3 **0.57 %** (1 of 176, 10 clips) · scribe 3.98 % (not
  quotable) · whisper **39.20 %** (69 of 176, 10 clips)
- shape ρ: nova-3 ρ = **−0.970** (n = 164) · scribe −0.820 (n = 174) ·
  whisper-base ρ = **−0.590** (n = 171)
- WER: nova-3 0.433 all-clips / 0.307 spoke · whisper 0.996 / **1.128** · scribe 0.410‡
- normalization shift: nova-3 **−0.014** · whisper **+0.090** · scribe +0.064 (a draw)
- edits (cross-model): nova-3 sub 0.149 / del 0.270 / ins 0.021; whisper 0.413 /
  0.289 / 0.197 → insertions **9.4×**
- hallucination: whisper p95 length ratio 2.75, 9.9 % of rows over 2× reference,
  mean foreign fraction 0.528; worst **11 ref words → 47 hyp words** (NOT the 3 → 49
  that `results/model_arms.{json,txt}` still prints — see the warning box below)

> ### ⛔ IF HE OPENS `results/model_arms.txt`, GET THERE FIRST
>
> Lines 171/175/179 still read `3 ref words -> 49 hyp words` (and `-> 38`, `-> 34`).
> **Those are wrong and the documents are right** — the inverse of the usual rule in
> this repo, so say it before he reads it.
>
> `hallucination_report` cross-model-normalizes the reference (`"four zero five"` →
> `"405"`) and *then* tokenizes with `[a-z']+` — letters only. It manufactures eight
> digit tokens and immediately discards them. The reference is **11** spoken words.
> The hypothesis contains no digits, so it loses nothing, and the ratio inflates to
> **16.3× against a true 4.3×**. Strict alignment: 11 → 47, WER 4.18, 1 match / 10
> sub / 36 ins.
>
> **The line to say:** *"That number is wrong and I know why — it's a ratio between
> two differently-tokenized quantities. I found it while regenerating the figure,
> because the chart drew the strict alignment and disagreed with the text above it.
> The docs are corrected; regenerating that artifact is a code change I haven't made
> yet."*
>
> Note the shape: it is this project's signature bug — **two quantities computed
> differently, then divided** — and it survived into four documents because everyone
> quoted the stored field instead of the alignment that produced the stored WER.
> The mechanism is untouched: Whisper did invent a 47-word repetition loop where
> nova-3 went silent. Only the magnitude moved.
- Jaccard: nova|whisper **0.000** · nova|scribe 0.000 · **scribe|whisper 0.101 (7
  shared)**
- silent rows: nova-3 24.5 % · whisper 19.1 % · scribe 4.4 %; mute conditions 12 / 5 / 2

**Active learning (D3b) — surrogate oracle, no live-API seed**
- target 0.162 reached by **2 of 8 active** vs **4 of 8 random** in 45 evals
- winner flips: active wins 2/4 splits, 13/32 paired, median paired diff **+0.003**
- acquisition worked: **58.3 %** of picks near the contour vs random's 30 %
- DRR re-run: 14/32, median +0.000, **0 of 4** splits; permutation control **rank
  18/24, p = 0.75**; across 44 parameterisations median −0.0001, 23/44 favour active

**Corpus / provenance**
- clean WER **1.65 %** — 6 errors, 363 reference words, 5 non-zero clips, all
  adjudicated by ear
- room tone **−52.9 dBFS** (target −60), 120 Hz mains hum; caps SNR at ~28 dB → axis
  stops at 20
- JOIN-1 gate **9/9** on u02/u17/u36; delivered vs requested SNR agreed to **0.01 dB**
- **10,560 rows**, 3 failures (0.03 %), **14,606 calls ≈ $3.70**
- codec is **G.726**, not AMR-NB — stock ffmpeg ships AMR decode-only. Say it; a
  silent substitution would not be defensible.

---

# §D — Where the documents disagree (and what to say if he finds one)

`report/writeup.md`, `README.md`, `report/UNDERSTANDING.md` and `report/STATUS.md`
are all pinned by `tests/test_report_numbers.py`, which re-reads their load-bearing
figures from the artifacts and **proves every check can fail** by mutating the prose.
So the pinned surfaces agree by construction. The gaps are outside the gate:

| where | says | artifact / reality says |
|---|---|---|
| **`UNDERSTANDING.md` §4.2 and §5.5** | "there is **no sensitivity analysis anywhere in the repo**" for the dead-zone thresholds | **Stale — one now exists**: `results/dead_zone_sensitivity.{json,txt}`, generated 2026-08-06. Its nova-3 surface reproduces UNDERSTANDING's hand-run table exactly (13 / 2 / 0 down the published column), and adds the persistence statistics. **If he reads UNDERSTANDING, correct it yourself: "that sentence is a day stale, the sweep is now an artifact."** |
| **`UNDERSTANDING.md` §4.8 / §5.4** | "`mean_conf` is the wrong statistic — the minimum or a low percentile is the operationally relevant number … it was never computed" | **Now computed, and the conclusion reverses.** Mean AUROC **0.944**, min **0.877 significantly worse**; nothing beats the mean. Also "`utterance_conf` … read by no analysis module" is now measured: distinct (Pearson 0.926) but no measurable advantage. **Q3 in §A is written against the NEW state.** |
| **`UNDERSTANDING.md` §4.4** | ρ(DRR, WER) table quotes marginal WER 0.6359 / 0.7581 for Bar / Shower | Those are the **babble-only 144-cell** marginals. `results/al_drr.txt` reports **0.5559 / 0.7217** over all 176 conditions. Both correct, different populations, and neither table says which. Ordering and every ρ are identical either way. |
| **`report/measurements.md`** (exempt from the pin, and it has drifted) | 22,411 deleted words / 35.6 % · discount 0.74 on n = 7,980 · ECE 0.051→0.032→0.006 · "sim misses **both** dead zones" | **22,416 / 35.1 %** · 0.75 on **8,144** · **0.0507 → 0.0346 → 0.0077** · real **1**, sim **0**. **Do not quote from `measurements.md`.** It is a working log, not a deliverable, and the exemption list is where numbers go to rot. |
| **`SPEC.md` G.10** | the indistinguishable pair is "exactly equal per-clip on u40, u26, u21, u10" — implying 4 clips | **18 of 40.** The write-up and the demo say 18 and are correct; the SPEC is stale. |
| **`SPEC.md` C.4** | the wrong (cell-wise) bootstrap comes out "~35–45 % too narrow" | Re-derived today: **rt60 32.5 %, snr_db 44.1 %**, mic_rolloff 13.0 %, and **codec goes the other way at 8.7 % wider.** Right order, loose on reverb, not universal. **Not persisted to any artifact.** |
| **`results/MANIFEST.json` vs the tag** | freeze recorded at `0d7d8f5` | `grid-v1` sits at `c321715`; both are behind `HEAD`. Regenerate the manifest on the committed tree before tagging — **and the tag is deliberately held until the write-up, dashboard and artifacts all agree.** |
| ~~`report/INTERVIEW_RUNBOOK.md`~~ | recommended saying *"streaming-capable, measured in batch"* | **RESOLVED by deletion, 2026-08-06.** Both claims were superseded by SPEC K.2. Rather than reconcile a second presenter document whose numbers nothing pinned, it was deleted. Recoverable from git history. |
| **`report/STATUS.md`** — "No arm is streaming" | *"The framing says 'streaming-capable model'"* | Accurate on the fact, **stale on the framing**: it no longer does (SPEC K.5). |
| ~~`README.md` §4 vs `demos/demo_listen.py`~~ | README's diagram said **3 blind pairs** while the script plays 2 | **RESOLVED 2026-08-06.** Decision: keep the default at **2 played live, the third held in reserve**, and the README now says exactly that. Three pairs were *measured*; two are *played*. |
| ~~`results/audio/demo/DEMO_SCRIPT.md`~~ (generated) | stated *"you cannot QA a voice agent by listening to it"* as a conclusion | **RESOLVED 2026-08-06 (`be4aed2`)** — fixed in the GENERATOR TEMPLATE, not the file, so it survives a rebuild. Was SPEC J.7's exact failure: a generated narration surface disagreeing with the script it narrates. |
| ~~`scripts/make_manifest.py`~~ | the arm taxonomy key was `"streaming_commercial"` while its own `api` field read `"pre-recorded"` | **RESOLVED 2026-08-06 (`8b42455`)** — renamed to `batch_commercial`. Nothing consumed the key by name (verified by grep first). Regenerate `results/MANIFEST.json` to pick it up. |
| **In-flight, unverified at the time of writing** | `README.md`, `Makefile`, `dashboard/DEMO.md`, `demos/demo_listen.py`, `tests/test_report_numbers.py` and several test suites all have uncommitted edits from concurrent work | **Check before you demo.** Two known historical traps in that set: the dashboard panel-7 caption once said "Two commercial **streaming** models" (now repaired), and `DEMO.md`'s script once narrated "NOT CONFIRMED" while the panel says **CONFIRMED**, plus an "SNR twenty-five decibels" level the grid never ran. |

**If he finds a disagreement you didn't flag**, the recovery line is the same every
time and it is a strong one:

> *"That's a real one — thank you. The pattern I'd point at is that everything inside
> `tests/test_report_numbers.py` can't drift, because it re-reads 157 figures across
> 234 prose sites from 13 artifacts and every check is proven able to fail by
> mutating the prose. **Everything the pin doesn't cover has drifted at least once.**
> That's not a coincidence, it's the argument for the pin — and the surfaces most
> likely to drift are the ones nobody thinks of as deliverables: a demo script, a
> caption, a working log."*

---

# §E — If he goes somewhere the README doesn't

**"Show me the code for one of the trap functions."** → `deadzone/audio_pipeline.py`.
Have `apply_rir` ready; the renormalize-over-active-region line is the anecdote.

**"How big is the test suite?"** → 27 offline suites, all runnable with no network and
no API key. `make test`. The number-pinning suite is the one to show him: it is
literally a QA suite for prose.

**"What would break if Deepgram changed the model behind `nova-3`?"** →
> *"Everything, silently — and that's why `results/MANIFEST.json` exists. It records
> the git SHA, the exact model literal, the ffmpeg build and codec decision, asset
> SHA-256s, realized call counts and cost. Commercial model literals are updated
> server-side, so **a re-run in three months is not the same experiment**, and without
> that file I couldn't say what I measured."*

**"How long did this take?"** → Don't inflate. The build plan budgeted ~40 hours;
the pipeline and corpus were the long poles, and the analysis layers were each a
mini-project. Say the honest thing.

**"Did you build the voice agent?"** → **No, and say it flatly.**
> *"No. `agent_eval.py` is a synthetic-validated scaffold — task/slot accuracy,
> entity error rate, turn-taking analysis over a timestamped event log — and it is
> presented as exactly that. There is no live STT→LLM→TTS loop in here. I scoped it
> out deliberately rather than half-building it, because a real-time three-vendor
> system is the highest demo risk in the project and it would have failed live for
> reasons that have nothing to do with the quality of the work."*

**"Does degradation break endpointing before it breaks transcription?"** → **The one
finding the project doesn't have and the one he will care about most.** Do not
speculate — hand him the design, because having the design is the signal.
> *"I don't have it, and it's the thing I'd build next. The design, though, I'm
> confident about:*
>
> ***Inject the degraded WAV bytes straight into the live socket, chunked and paced
> to realtime.** No audio hardware, fully reproducible, and it still exercises the
> real streaming and endpointing path.*
>
> *And the thing I would **not** do, which is the part that matters: **never play the
> audio through speakers into a microphone.** That re-introduces uncontrolled room
> acoustics **on top of** the simulated ones, which destroys the counterfactual-
> isolation premise the entire rig rests on. It's a party trick, not a measurement —
> and it would quietly undo the one property that makes the exact variance
> decomposition possible.*
>
> *One parameter I'd want on the record before running it: **`utterance_end_ms` is a
> parameter you CHOOSE, not a property you discover.** My clips carry ~0.5 s of
> trailing room tone by design — the VAD and `apply_rir`'s onset trim both need it —
> and that tail may itself trip the endpointer. So the value has to be set
> deliberately and **reported in the methods**, not tuned until the result looks
> good."*

**"What's the one thing you'd change about how you worked?"** →
> *"Pin the prose earlier. Four of the five worst number errors in this project
> survived multiple reviews because they were being copied forward from a progress
> log instead of re-read from the artifact. **The log is a summary, not a source.**
> The test that fixed it is 1,500 lines and I wrote it near the end; it should have
> been the third file in the repo."*

---

# §F — Run-of-show, and pre-flight the morning of

## The hour budget — and what to cut if you are running long

| beat | min |
|---|---|
| the reframe (§1–§2) | 2 |
| the audio demos (§5 + §7) | 6 |
| the instrument (§3–§4) | 4 |
| the dashboard | 9 |
| judgment — the corrections and the nulls (§6, §11) | 8 |
| product framing (§8–§10) | 4 |
| **spine total** | **33** |

**That leaves ~25 minutes for conversation, which is where the hiring signal actually
lives.** Do not spend it. If you are running long, cut in this order:

1. **§3 down to two bullets** (the pipeline diagram — he can read it).
2. **§4's third "why"** (why the same room/mic/distance).
3. **Listening pair 1 (`u40`)** — two pairs carry the beat and pair 1 is the marginal
   one (§5 explains why it is already third in the order).

> ### 🚫 NEVER CUT — these three are the interview
> - **The headline correction** (§6: 6 → 2 dead zones, found by ears not by a test).
> - **The failed pre-registration** (§5: it failed, and its rubric couldn't fail).
>   A demo that only reports the predictions that worked is the exact failure this
>   project is about.
> - **The Scribe reversal** (§8/§10: two scorings, opposite verdicts). It is the only
>   place the hour demonstrates the thesis on the **benchmark** rather than on a model.

## Pre-flight, the morning of

```bash
make demo-check                                     # wifi still ON
make test                                           # 29 offline suites
./.venv/bin/python demos/demo_listen.py --check     # prints READY
./.venv/bin/python dashboard/build.py --master results/master.csv   # NO extra flags
```

Then, in this order:
- **`make demo-replay`** — the hero from cache, no network. Confirms the whole beat
  works before you let it near wifi.
- **`make demo`** — the live one, wifi **on**, on the actual machine, at least once.
  It is the only beat that touches the network and you need to have seen the real
  latency.
- **`make demo-listen --pairs 3`** if you want to match the README's 3-of-3 diagram;
  otherwise say "two of the three."
- Read the dashboard badge out loud to yourself: it must say **`real grid`**, not
  `synthetic data`.
- Read `build.py`'s per-panel `ok` / `EMPTY` line. Four `EMPTY`s are expected and
  explained on screen (sensitivity and sim2real on the two non-spine arms). **Anything
  else that says EMPTY is news, and you want it to be news before you're standing up.**
- Have this file open in a second window, with **§0.1 (the population card)** and
  **§A (the three killer questions)** reachable without scrolling.

**Read these two before the call:**
- **`report/_demo_internal_notes.md`** — the stage directions moved off the demo
  screens, plus the four operational lines stripped from the hero and where they went
  instead. Folded into §5 and §7 above, but read the original.
- **`report/model_architecture_notes.md`** — what each arm's confidence field is,
  sourced to vendor docs, model cards and shipped source, with DOCUMENTED /
  INFERRED / SPECULATION tags. It is the backing for §A/Q3 and it is the difference
  between "I don't know" and "here is exactly how far the public record goes."

**There is no second presenter document to rehearse off.** `INTERVIEW_RUNBOOK.md`
was deleted rather than reconciled — see the box at the top. Two presenter
documents that disagree is SPEC J.7's failure, and it was the single most likely
thing to bite in the room.
