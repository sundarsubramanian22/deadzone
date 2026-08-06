# Deep-dive runbook — 60 min with a Deepgram engineer

Audience: **one software engineer on Deepgram Labs** — prototypes, agents,
cutting-edge voice AI — and the model under test is their product.

- **Do not explain WER, SNR, RT60, or what a RIR is.** They know. Explaining
  basics is the fastest way to lose an expert's attention.
- **The subject under test is their product.** The framing is *"I built an
  instrument that finds where an ASR fails silently, and here is what it found"* —
  never *"your model is bad."* The instrument is the deliverable; Nova-3 is the
  first thing measured with it. Say **"streaming-capable, measured in batch"**
  rather than "streaming" from the very first sentence — the concession later in
  this document only works if you have not already overclaimed it in the opener.
- **The data is favourable to them on the measured arms — but do not build the
  segment on that.** Nova-3's confidence tracks its own error rate at spearman
  **−0.980** (D1, 40 clips), and on the matched subset it beats Whisper by a
  distance (§4.4). That is true and it is theirs. It is *not* the load-bearing
  claim, and the third arm is now the proof of why: **a commercial peer landed,
  and Nova-3's apparent lead over it collapsed from 0.20 to 0.03 the moment the
  orthography was normalised.** See the box below and §4.4.
- **They asked about product decisions.** Budget real time for what a deployment
  should *do* with these findings.

> **Three dead-zone rates now — say which population, every time.** D1 runs the
> full 40-clip corpus for nova-3: **2 of 176 (1.14 %)**. L1 restricts *every* arm
> to the 10 clips all three ran: nova-3 **1 of 176 (0.57 %)**, Scribe **7 (3.98 %)**,
> Whisper **69 (39.20 %)**. Quote the matched ones whenever a second model is on
> the page, and never put 1.14 % beside 39.20 % as if they were matched.
> **And read the next box before you quote Scribe's 3.98 % at all — you should
> not quote it.**

> ✅ **ELEVENLABS SCRIBE: THE ARM HAS RUN.** **1,760 rows** in
> `results/master.csv` (176 conditions × the same 10 clips, **0 failures**), three
> arms in the dashboard toggle, and Scribe is the **first block** in
> `results/confidence_gap.txt`. The day-one gate held: per-word `logprob` (≤ 0;
> `exp()` for a probability), `transcribe_elevenlabs` in `audio_pipeline.py`,
> `elevenlabs-scribe` → `scribe_v2` in `MODEL_REGISTRY`.
>
> **What the arm bought is not a ranking — it is a demonstration that the ranking
> is a function of the scoring.** §4.4 is now the strongest segment in the hour.
> Two things must never be said, and they are both the tempting things:
>
> - ❌ **Never quote Scribe's dead-zone rate, in either direction.**
>   `dead_zone_flags` thresholds an **absolute** WER, so an arm carrying an
>   orthography offset crosses it for non-acoustic reasons. Under the normalizer
>   Scribe's count goes **7/176 → 0/176** — all seven fall from WER 0.30–0.43 to
>   0.08–0.12. **3.98 % does not go on a slide.**
> - ❌ **Never claim Nova-3 is meaningfully ahead of its commercial peer.**
>   Strict −0.970 vs −0.820 looks like a 0.15 gap; re-scored under the
>   orthography normalizer it is **−0.970 vs −0.948**, ~0.02 — and since
>   measurement error only attenuates a rank correlation, that is an **upper
>   bound** and may be zero.

> 🔁 **The model-comparison beat was written result-conditional, and the result
> came back better than any of its three branches.** Branch (b) technically fired
> — Scribe reads worse on dead-zone rate — but the honest reading of it is that
> **the metric that would have said so is not scale-free**, which is a finding
> about benchmarks rather than about either vendor. §4.4 is rewritten around that.
> Read it before the interview, not during.

---

## The spine (~33 min), leaving ~25 for the conversation

An hour with one engineer is a conversation that gets interrupted, not a talk.
Build the spine with deep branches, and let them steer.

### 1. The reframe (2 min, no slides, no screen)

One sentence: *aggregate WER hides where and how a model fails, so I stopped
asking "how much does it break" and asked "does it know it's breaking" —
because for a streaming voice agent, a confidently wrong transcript is far more
dangerous than a visibly uncertain one. Confidence is what decides whether the
system commits or asks the user to repeat.*

Then the honest positioning immediately, before they wonder: **this genre is
well-trodden** — WildASR, Speech Robustness Bench, "When Denoising Hinders".
Nothing in the method is novel. The contribution is the lens, the typed
fingerprints, and testing a commercial model that exposes per-word confidence
where the literature uses Whisper/Conformer/wav2vec.

Saying this *first, unprompted* is worth more than any result. An expert places
the work in its literature within five minutes whether you do it or not.

### 2. The visceral demo (6 min) — DO THIS EARLY

Audio first, while attention is highest. Everything is in `results/audio/demo/`;
hand over **`blind/` only** (8 neutral filenames + `BLIND_SHEET.md`). The working
filenames in the parent directory say `reverb` and `babble` — a listener who sees
them has been told the answer.

> ✅ **`DEMO_SCRIPT.md` HAS BEEN FIXED — read it, it is now the better document.**
> The earlier warning here (that it led with the wrong pair, asserted a refuted
> precedence-effect mechanism, and had no fallback for a backwards result) was
> accurate when written and **all three are repaired**, verified 2026-08-06: it
> now opens on **pair 2 then pair 3** with pair 1 explicitly third; §4 carries the
> failed prediction and labels the DRR story *"a hypothesis I formed after seeing
> the result, not a finding"*; and its Fallbacks list handles **"they rank the
> reverb clip harder"** by name. The beats below and `DEMO_SCRIPT.md` now agree —
> use whichever is in front of you.
>
> 🛑 **One live hazard in that file:** it is **generated** by
> `scripts/make_demo_audio.py`, which rewrites it unconditionally, and the
> revisions are hand-written. Do not run the generator before the interview. See
> `REGENERATION_HAZARD.md` in the same directory.

**Beat 1 — the ranking. Ask, do not predict.** Show that
`PREREGISTERED_PREDICTION.md` exists and **leave it closed** — stating a
prediction aloud before someone judges is a demand characteristic, and this
project's whole subject is not fooling yourself with a number you wanted. Say
only *"I've written down what I think you'll say."*

Play, in this order:

- **Pair 2 — `blind_06` / `blind_01` (`u21`)** ← lead here
- **Pair 3 — `blind_08` / `blind_04` (`u26`)**
- Pair 1 — `blind_03` / `blind_07` (`u40`) — only if he wants a third.

`BLIND_SHEET.md` lists the pairs as 1-2-3 and that numbering is fine to leave
alone — you are choosing the *play order*, not renumbering the sheet. The sheet
already asks him to say **how confident** he is on each pair; keep that, because
it is the thing that makes his answers comparable to the listener already run.

Pairs 2 and 3 are the ones the single listener so far called **confidently**;
pair 1 he called with low confidence by his own account ("not 100 % everyone
would agree"), so it is the weakest place to open. Ask one question — *"which of
these is harder to understand?"* — and let him finish every ranking before you
say anything.

**The reveal is that the model has no preference.** This is direction-agnostic:
it lands identically whichever way he ranks, which is exactly why the beat is
built on it.

- **A** = `rt60 1.0 / SNR 20 dB` — Shower IR, measured RT60 1.011 s, DRR
  **−10.02 dB**, but *quiet* → mean WER **0.1123**
- **B** = `rt60 0.2 / SNR 0 dB` — Restaurant IR, measured RT60 0.193 s, DRR
  **+16.90 dB**, speech buried → mean WER **0.1301**
- Neither has a codec or mic rolloff on it, so nothing else moves between them.
- Paired over the same 40 clips: **−0.0178, 95 % CI [−0.0654, +0.0310]** —
  10,000-resample paired bootstrap over clips, spans zero.

Per clip, inside the pairs he just heard:

| pair | clip | A (reverb) | B (babble) |
|---|---|---|---|
| 2 | `u21` | WER **0.222**, conf 0.879 | WER **0.222**, conf 0.854 |
| 3 | `u26` | WER **0.250**, conf 0.864 | WER **0.250**, conf 0.883 |
| 1 | `u40` | WER **0.333**, conf 0.805 | WER **0.333**, conf 0.807 |

**18 of the 40 clips tie exactly.** Four of those ties are non-zero — `u40`
0.333, `u26` 0.250, `u21` 0.222, `u10` 0.125 — so on those it did **not** get
both right; it got both **wrong, by the same amount, in different places.** The
transcripts make that better than the scalar does:

- `u21` ref `forward the file to accounting and legal by monday` →
  reverb `…and legal filing` (substitution) vs babble `…and legal` (deletion)
- `u26` ref `text me the address for the kowalski wedding` →
  reverb `…for the cool seaway` (the name mangled into words) vs babble
  `…for the` (the name simply gone)

Same WER, different failure *shape* — which is the whole argument for typed edits
over a scalar, made audible.

**Beat 1b — NOW open `PREREGISTERED_PREDICTION.md`. The prediction FAILED, and
you say so first.** Report it as a result. Do not soften it and do not skip it:
this is the project's own thesis applied to itself, in front of someone who will
notice if you only volunteer the predictions that worked.

> "I sealed a prediction about how you'd rank these before anyone listened, and I
> got it wrong. I predicted the loud-babble clip would be clearly harder in both
> of the pairs the document names — informational masking is brutal for a human,
> and the precedence effect makes reverb close to free. It held in **one of those
> two**, and weakly. In the third pair, the backup one, it went the other way as
> well. So: **one of three**, and the listener called the *reverb* clip harder
> twice. The mechanism I proposed does not hold at this DRR.
>
> And I'll say something worse about my own document: it listed exactly two
> outcomes, *unequal* → holds and *equal* → fails. It never considered **unequal
> but backwards**, which is what actually happened. A pre-registration that can
> only be confirmed or shrugged at is a badly written one, and that is a flaw in
> the registration, not in the data.
>
> What survives is the half I would have kept either way: **a human had a
> preference in 3 of 3 pairs, and the model scores every pair exactly equal.** The
> disagreement is the result. *'And here's why'* is not — I had a mechanism, I
> wrote it down, and the data went the other way."

Have the tally ready if he asks: preference expressed in **3 of 3** pairs,
confident in **2 of 3** (pairs 2 and 3), predicted direction held in **1 of 3**
(and in **1 of the 2** the sealed document actually named), reverb judged harder
in **2 of 3**.

**Do not repair the story on stage.** The tempting move is to reach for a new
mechanism — "well, at −10 dB DRR the reflections stop fusing" — and it is
available and it might even be right. It is also a post-hoc explanation for an
n = 1 observation that just refuted a pre-registered one, which is the exact move
this project spends 176 conditions refusing to make. If he proposes it, agree
that it is testable and say what would test it (a real listening study,
DRR-swept, many listeners). Do not adopt it as a finding.

**Label it honestly, out loud:** the human half is **n = 1, unblinded, one
listener, one speaker, one accent, clips not level-matched, order not
counterbalanced** — an intuition pump, not a measurement. The *measured* half is
the paired model-side result and its CI, and it reads the same whichever way he
ranks. Blurring those is the only way this beat can hurt you here. Doing the
human half properly is a listening study with many listeners, randomised order
and level-matched stimuli — the experiment this project does not have, and the
limitations section says so.

**The takeaway is a product claim, and it survives the failed prediction intact:**
*you cannot QA a voice agent by listening to it.* "Sounds fine to me" is not
evidence the ASR works, and "sounds terrible" is not evidence it fails — a human
ranks these unequal, in a direction I could not predict, and the model scores them
the same.

**Beat 2 — the payoff.** Play `blind_02` (clean control) then `blind_05`: `u03`
under `rt60-0.7 / snr-20 / babble / opus-lowrate / roll-1`. The SNR is **20 dB**
— it is *quiet*; the damage is reverb + codec + mic rolloff. He can hear a
person speaking clearly. The model returned an **empty string**: WER 1.000, all
11 reference words deleted, and **10 of 40 clips returned nothing at all** here.
Hold this clip — it comes back in §5, because this exact condition is what the
defect was hiding behind.

### 3. The instrument (4 min)

Where an ASR engineer actually engages, because they have been bitten by these.

- **SNR on active-speech energy only**, not whole-file power — otherwise silence
  deflates the denominator and your "10 dB" mix is not 10 dB.
- **After RIR convolution: trim the direct-path delay** (or every WER inherits a
  pure alignment artifact) **and renormalise over the input's active region**.
  The anecdote: the reverb tail leaks energy into the silent regions and
  de-calibrates every downstream SNR. Clean-looking garbage, no error message,
  caught by the test suite and by nothing else.
- **Typed edits, not scalar WER** — sub/del/ins is what makes fingerprints
  possible at all.
- **Real ingredients, controlled assembly**: measured RIRs (MIT survey), real
  noise (DEMAND). Only the combination is synthetic.

Have the composition order and its physical justification ready as a branch.

### 4. Findings, on the dashboard (9 min)

Offline, `file://`, wifi off. Order matters:

1. **The silent-failure map, in three categories.** Not one bucket: **dead zone**
   (spoke, confident, wrong) — **2 of 176**; **silence-driven** (looks like one
   only if you mis-pair the estimands) — **4**; **mute zone** (empty transcript
   on *every* clip) — **7**. A mute zone is not a dead zone: confidently wrong
   and entirely absent are different mechanisms with different fixes, and **a
   confidence monitor cannot see a mute zone at all**. The taxonomy came out of
   §5's correction — introduce it here, pay it off there.
   Headline, no asterisk: *at `rt60 0.45 s, SNR 0 dB, engine, g726, roll 0`,
   nova-3 returns mean word confidence **0.829** at WER **0.306** — **0 of 40
   clips came back empty**, so both averages are over the same clips.* It sits in
   **engine** noise, where confidence otherwise tracks WER at ρ = −0.99: a point
   failure, not a bad region.
2. **DRR, not RT60** — the best mechanistic result. The `rt60` axis is
   non-monotonic (0.2026 → 0.6359 → 0.4495 → 0.7581, marginals over the babble
   factorial core) because each level is delivered by the *nearest measured RIR*,
   i.e. a different real room. `spearman(DRR, WER) = −1.000` against
   `spearman(RT60, WER) = +0.800`.
   **Reverb benchmarks parameterised by RT60 alone will mis-rank conditions.**
   Directly useful to anyone building an ASR eval set, which is them.
   *(If he checks a different table and gets 0.2026 → 0.5559 → 0.4495 → 0.7217:
   that is the same four rooms averaged over all 176 conditions instead of the
   babble core. Both orderings are identical and both Spearmans are unchanged —
   say which population you are quoting and it is a non-event.)*
3. **Fingerprints → fixes.** Reverb and low SNR → *deletions*; g726 and road
   noise → *substitutions*; babble → insertions that are 92 % foreign tokens,
   the model transcribing background speakers, a different mechanism from
   acoustic confusion. **Do not call that babble-specific** — the foreign fraction
   is 0.94 under engine and 0.89 under road too; what babble has is ~3× the
   insertion *count*. The mechanism claim is "competing-source capture", not
   "babble". Proper nouns destroyed at 0.646 vs 0.462 for function words —
   carried by proper nouns and spelled letters (0.613), **not** by digits, which
   are the *least* destroyed class. Each signature implies a fix: keyword
   boosting, dereverberation, entity-aware decoding.
4. **Model comparison — the framing must not depend on who wins.** See the
   `[[PENDING SCRIBE]]` box at the top: **a third arm is a commercial peer, and
   its results do not exist yet.**

   **Primary framing, and it is direction-independent — open with this:** the
   interesting result is not which model is better, it is that the models have
   **different** dead zones. Measured, twice, in two unrelated senses:
   - **across model families:** nova-3 vs whisper-base, shared dead zones **0**,
     union 70, **Jaccard 0.000** (`results/model_arms.json`);
   - **across RIR provenance:** measured vs simulated RIRs, real 1, sim 0, shared
     0, **Jaccard 0.00, recall 0.00** (`results/sim2real.txt`).

   > "**You cannot borrow someone else's dead-zone map.** Not from another model,
   > not from a simulator. The map is the deliverable and it does not transfer —
   > which is an argument for the instrument, not for any one model's ranking."

   A third arm makes that claim *stronger whichever way it ranks*, and that is
   the whole reason to lead with it.

   **The Whisper leg — measured, quote freely, matched 10-clip subset:**
   dead-zone rate **0.57 % (1/176) vs 39.20 % (69/176)**, confidence-vs-WER
   **−0.970 (n = 164) vs −0.590 (n = 171)**, insertions **0.021 vs 0.197** of
   reference words. **State the asymmetry yourself:** Whisper is a *weak open
   baseline* whose "confidence" is a derived segment proxy — `exp(avg_logprob)`
   spread over the words — not a per-word posterior. He will know that. Beating it
   tests whether the confidence-vs-WER *shape* differs across families; it is not
   a peer comparison and you should not sell it as one.
   Then Whisper's hallucination mode as the interesting part: 3 reference words
   became 49, and WER exceeds 1.0 in two whole factor regions because insertions
   are unbounded. *WER understates that failure* — it caps damage at one error per
   reference word, while a 49-word hallucination handed to a downstream LLM is
   unbounded harm.
   **Best detail: §5's correction moves the two arms in *opposite* directions.**
   Restricting to the clips each model actually spoke on **lowers** nova-3's WER
   (0.433 → **0.307**) but **raises** Whisper's (0.996 → **1.128**) — its silent
   clips score exactly 1.0, so dropping them removes its *cheapest* rows and
   leaves the ones that hallucinate past 1.0. One model goes quiet under stress;
   the other invents.

   **The Scribe leg — three branches, decided now, not on stage.**
   `[[PENDING SCRIBE — do not speak any of these until rows exist]]`

   - **(a) Nova-3 ahead on dead-zone rate and calibration.** Then §4.4 reads as it
     always did, *but the ranking is still the second sentence.* First sentence
     stays the disjoint map: *"two commercial models, both exposing per-word
     confidence, and their dead zones do not overlap — one of them happens to have
     fewer."*
   - **(b) Scribe ahead on dead-zone rate or on ECE.** Deliver it as a finding, in
     one flat sentence, and then keep going:
     > "In region X the commercial peer was better calibrated — its dead-zone rate
     > there is A against Nova-3's B, and its post-calibration ECE is C against D.
     > The mechanism is visible in the edit signature: [substitutions vs deletions
     > vs silence]. That is the instrument doing its job — it was built to find
     > where *a* model fails silently, and it found a region where this one does.
     > What I'd want from you is whether that region matches anything you already
     > know about the acoustic front end."
     Said flatly, that is a result. Said apologetically, it is a concession — and
     hedging it in front of the people who build the model is the worse failure.
     **Never** editorialise beyond the measurement, and never generalise from a
     region to the product.
   - **(c) Mixed / within noise.** Most likely outcome. The primary framing above
     already covers it verbatim; add *"neither arm dominates, and the regions where
     each is blind are disjoint,"* which is the strongest version of the claim.

   **Comparability rules that must survive contact with an expert:**
   - Scribe reports a **log-probability**; `exp(logprob)` is its own scale and is
     **not** commensurate with a Deepgram acoustic confidence or with Whisper's
     segment proxy. Every cross-model claim goes through
     `within_model_conf_percentile`. Say this before he asks.
   - **The orthography audit is a gate, not a nicety** —
     `scripts/probe_scribe_orthography.py`, run and read before a single Scribe row
     enters `master.csv`. Benchmarks to quote: nova-3 shifts **−0.014** (already
     word-form, so ~0 is the *expected* result), whisper **+0.090** (the normalizer
     recovering digit orthography, not accuracy). A large unexplained Scribe shift
     means a condition-independent formatting offset — indistinguishable from an
     acoustic effect once it is in the table, which is exactly the failure this
     project is about.
   - **A third arm narrows the matched comparison.** L1 matches to the cells
     *every* arm ran; the current intersection is 10 clips × 176 conditions = 1757
     rows per arm. An arm on a different clip subset shrinks that further, and the
     arm census in `results/model_arms.txt` prints it. Quote the census, never a
     bare WER.
   - **With three arms, quote the PAIRWISE Jaccard, not the all-arm one.** The
     headline 0.000 today is a two-arm number, where all-arm and pairwise
     coincide. At three arms the all-arm figure becomes *"flagged by every arm"* —
     a stricter and different claim that will trend to zero for trivial reasons.
     `dead_zone_overlap.pairwise["nova-3|elevenlabs-scribe"]` is the one that
     carries the "you cannot borrow someone else's map" argument.
   - **Every dead-zone rate is flagged on `wer_spoke`, on the matched cells.**
     That is the §5 correction, and it applies to a new arm on day one rather than
     being discovered in it later. If a Scribe comparison ever gets quoted off the
     all-clips pairing, it is the same defect this whole runbook opens on.

### 5. Judgment — the section that decides the interview (8 min)

Everything above is competence. This is the part that is hard to fake, and the
first item is the strongest thing you say all hour.

#### I found a defect in my own headline, corrected it before showing you, and the fix produced a better taxonomy than the number it replaced

**The bug.** Per-condition `mean_conf` was averaged only over the clips that
produced words. Per-condition `wer` was averaged over *all* clips — including
ones that returned an **empty transcript**, contributing WER 1.0 and no
confidence at all. Subtracting them **mixed two estimands**. Clean arithmetic,
right row count, no NaN, no exception. The only thing that catches it is
asserting *which population each average is over*.

**How it was found — tell this part.** It was the listening pass, the last
unchecked item in the spec. The dead-zone example clips **sounded intelligible**,
which did not fit a story about being confidently wrong, which led to opening
the per-clip rows. **No test caught it. The audio did.** In a project whose
thesis is that a scalar can look fine while being wrong, that is the thesis
landing on its own author.

| | before (mixed) | after (same subset) |
|---|---|---|
| dead zones | 6 of 176 (3.41 %) | **2 of 176 (1.14 %)** |
| spearman(conf, WER) | −0.957 | **−0.980** paired / **−0.952** all-clips, both n = 169 |
| mean confidence gap | +0.256 | **+0.147** |
| overconfident conditions | 92 % | **91 % (154/169)** |

The old **−0.957** was computed over all 176 conditions while reporting n = 169.
The 7 mute conditions sit at percentile 0 with WER 1.0 — **seven fabricated
points parked at the ideal corner of a negative correlation.** Removing them
makes the correlation *stronger*, which is the tell that they were never
measurements.

**The old #1 dead zone is the payoff clip from §2** — `rt60 0.7 / snr 20 /
babble / opus-lowrate / roll 1`. Now classified `silence_driven`: 10 of 40 clips
silent, gap collapsing from **+0.230 to +0.025**. On the 30 clips it *did* speak
on it was **81.8 % accurate at 0.843 confidence — well calibrated.** Still
dangerous — a quarter of utterances vanishing is severe — but it is a **silence
failure, not a confidently-wrong one**, and the fix is different. 2,210 of 7,040
rows (**31.4 %**) are silent; 116 of the 169 conditions the model spoke in were
affected by the mis-pairing.

**The payoff:** the correction did not just shrink a number, it produced §4's
taxonomy — and `mute_zone`, the category a confidence monitor is structurally
blind to, exists only because of it. **Lead with this if the conversation is
going well.** Volunteering a correction to your own headline is the strongest
signal available, and much stronger from you than found by them.

#### The rest

- **The active-learning result is a null.** Straddle acquisition did not beat
  random: the target was reached by 2/8 active seeds and 4/8 random within a
  45-evaluation budget, and across 4 train/test splits the winner *flips*
  (active won 2/4 splits, 13/32 paired runs). No seed was confirmed end-to-end
  against the live API. The control matters: the same machinery *does* beat
  random on planted synthetic structure, so this is a method meeting a surface
  with no exploitable boundary, not a broken implementation. **Report the
  budget, not a ratio.**
- **"Did you re-run it under DRR?" — yes, and it is still a null.** He will ask
  this the moment you say DRR orders the conditions and RT60 does not. Under DRR,
  straddle beats random in **14/32** paired runs, median paired difference
  **+0.000** (RT60: 13/32, +0.003; negative would mean active is better). Across
  **44 coordinate systems** the median paired difference spans −0.0053 to +0.0106
  and **0/44** have a winner stable across 4 splits. Two negative controls:
  all 24 permutations of the same four DRR values put the true assignment at rank
  **18/24** (permutation p = 0.750), and random monotone relabellings of RT60 beat
  it 7/16. The ceiling is the honest part: **the grid contains exactly four
  distinct RIRs**, so any reparameterisation of the reverb axis is a relabelling
  of four points on a line, and the GP normalises each axis by its bounds — DRR
  cannot add information the grid never measured. Straddle *did* concentrate
  58.3 % of acquired evaluations near the contour vs 21.1 % for random, so the
  acquisition function worked and the job did not pay. `results/al_drr.txt`.
- **The pre-registration was CONFIRMED**, registered before any real audio
  existed (`d8ddd4f`, 2026-07-27) with a decision rule fixed in advance:
  ST−S1 = 0.128 (`rt60`), 0.112 (`snr_db`), `rt60 × snr_db` ranks 1/6 in S2. The
  quoted CI is ~2.5× wider than necessary — it adds S1 and ST in quadrature while
  their bootstrap correlation is +0.86, conservative in the only direction that
  matters for a pre-registered test.
  **Pair it with the one that failed** (§2 beat 1b) — *"I ran two
  pre-registrations. The one about the model confirmed. The one about the
  listener failed, in a direction my own document hadn't allowed for, and it's
  still in the repo."* Two registrations with one verdict each is a process; one
  registration with one confirmation is a lucky guess, and he can't tell the
  difference unless you show him both.
- **The sim-vs-real gap survived the correction untouched, and say so.** Level
  and order carry **no confidence term**, so both are bit-identical before and
  after: synthetic RIRs underestimate WER by **12.1 points [−15.0, −9.6]** but
  rank conditions well (**Spearman 0.873**). Dead-zone counts did move — real
  2 → 1, sim 1 → 0 — and **Jaccard is still 0.00**. Cross-model Jaccard is *also*
  0.00. Two independent senses in which **you cannot borrow someone else's
  dead-zone map**.

### 6. Product implications (4 min) — they asked for this explicitly

- **Confidence needs recalibration, and it is cheap.** A feature-conditioned
  calibrator cuts ECE from **0.051 to 0.008** on held-out conditions
  (temperature scaling alone gets 0.035). Concretely: above `rt60 = 0.7`,
  discount reported confidence by ~0.07; above `mic_rolloff = 0.5`, by ~0.06.
- **Confidence has a structural blind spot, and it now has a name.** Deletions
  are 35.1 % of reference words and 69.3 % of all errors, and carry no hypothesis
  token, so no confidence. A perfectly calibrated confidence converges on
  **emitted-word accuracy 0.767**, not **reference recovery 0.513** — anything
  thresholding on mean confidence is reading the wrong quantity by 0.254. The
  limit case is the **mute zone**: the calibrator is fit on 169 conditions and
  is **silent about the worst 7**. **Pair confidence with an utterance-level
  "did I get anything" check.**
- **Entity error rate diverges from WER** (0.633 vs 0.511), proper nouns worst
  hit. WER is the wrong acceptance metric for a slot-filling agent.
- **Turn-taking is unmeasured and probably matters more** — see the Labs section
  below; that is where this one goes if he takes it.

---

## Branches to have loaded, not presented

- **Why no live voice agent?** The measurement rig was the priority; a live
  three-vendor realtime system is the highest demo risk in the project.
  `agent_eval.py` is built and synthetic-validated — task/entity metrics and a
  turn-taking analyzer over a typed event stream — a drop-in once there is an
  agent to score. Have the file open.
- **"Why ElevenLabs Scribe, and where is it?"** `[[PENDING SCRIBE]]` The honest
  status: **the adapter and the confidence gate are done, the grid has not run,
  and there is no Scribe number to quote.** What the day-one gate actually
  established, which is the interesting part:
  - Scribe returns a **per-word `logprob`** (≤ 0), so it is the second arm in the
    study that can be asked the silent-failure question at all. Whisper cannot —
    its confidence is a derived segment proxy.
  - **The vendor's own docs disagreed with each other:** the capabilities page
    shows a response with no `logprob`; the API reference lists it. That is
    precisely why the gate is a live probe and not a documentation read.
  - **The trap that was avoided:** `language_probability` is a *document-level
    language-detection* score on 0–1. Mistaking it for confidence would assign
    every word in a clip the same value — a perfectly smooth, entirely fake
    confidence signal, and **no test in this repo would catch it.**
  - Observed live: a word (`u02`, "at") with `logprob` exactly **0.0**, i.e.
    `exp()` = exactly 1.0. Anything that then takes a logit of that confidence
    must clip — the same class of bug as the calibration layer's `_logit` guard.
  - Cost is not the constraint: $0.22/hr batch ⇒ the 10-clip subset ≈ **$0.43**,
    the full 40-clip grid ≈ **$1.72**. **The gate is the constraint** (the
    orthography audit, §4.4).
  - The forward-looking half, and it belongs in the *next steps* answer rather
    than the results: `scribe_v2_realtime` exposes **the same per-word `logprob`
    over a websocket**, so it is the cheapest route to this project's first
    genuinely *streaming* arm — which is the gap the streaming-framing section
    below already concedes.
- **Why G.726 and not AMR-NB?** Stock ffmpeg is AMR decode-only. The split paid
  off anyway: g726 produces substitutions, opus-lowrate deletions — two
  mechanisms from one factor.
- **Why 40 clips?** Precision is governed by reference word count, not clip
  count. ~340 ref words per condition puts the SE on a 0.20 WER at ~2.2 points;
  15 clips gives 3.5 and neighbouring cells overlap.
- **`results/audio/demo/isolation/`** — the factor ladder, `00_RAW_original` to
  `10_destroyed`, one degradation at a time.
- **The scratch notebook** — master table loaded, for "does that hold for engine
  noise too?"

## Demo hygiene

- **Wifi off, no API key, one command per demo.** Everything cached. Rehearse
  the full path once on the actual machine, projector attached, and
  screen-record a successful run as a fallback.
- Timings: reframe 2 · audio 6 · instrument 4 · dashboard 9 · judgment 8 ·
  product 4 = **33 min**, leaving ~25 for the conversation, which is where the
  hiring signal actually lives.
- Running long? Cut §3 to two bullets and §4 item 3. Inside §2, cut **pair 1**
  (`u40`) — two pairs carry the beat and pair 1 is the marginal one. **Never cut
  §5's first item, and never cut §2's beat 1b** (the failed pre-registration):
  a demo that only reports the predictions that worked is the exact failure this
  project is about.
- **Nothing in the run-of-show depends on a Scribe result.** If the arm has not
  run by the day, every `[[PENDING SCRIBE]]` block is skipped and the hour is
  unchanged. Rehearse it that way at least once.

---

## Calibrating for a Labs SWE specifically

### Get ahead of the streaming framing. He will catch it.

The project says "streaming ASR" throughout, but **the grid ran on the
pre-recorded endpoint, not `listen.live`.** Say so before he asks:

> "Everything in the grid is pre-recorded. That is the honest label. The
> confidence signal and the acoustic failure modes carry over, but anything
> about endpointing or turn-taking genuinely requires the live socket — which is
> exactly why that is the named next step rather than something I'm claiming."

Volunteered, this is a scoping decision. Discovered by him, it looks like you
did not know the difference. Same fact, opposite read.

### Turn-taking is his team's problem, and the most Labs-relevant thing here

The one finding the project *doesn't* have is the one he will care about most:
does degradation break endpointing before it breaks transcription? Have the
design ready — it shows you would do it correctly:

- **Inject the degraded WAV bytes into the live socket, chunked and paced to
  realtime.** Reproducible, no audio hardware, still exercises the real
  streaming and endpointing path.
- **Never play audio through speakers into a mic for measurement.** That
  re-introduces uncontrolled room acoustics *on top of* the simulated ones and
  destroys the counterfactual-isolation premise the whole rig rests on. A party
  trick, not a measurement.
- `utterance_end_ms` is **a parameter you chose**, not a property you
  discovered — and the ~0.5 s of trailing room tone in every clip may itself
  trip the endpointer, so it has to be set deliberately and reported.

### Connect empty transcripts to an agent failure — the mute zone is the evidence

The strongest *product* observation available, and after the correction it is no
longer an anecdote: it is a **measured category** with 7 conditions in it, plus
the §2 payoff clip where 10 of 40 clips vanished at **20 dB SNR**.

> "An agent receiving an empty transcript cannot distinguish *'the user said
> nothing'* from *'the user spoke and we failed.'* Those require opposite
> behaviours — keep waiting versus ask them to repeat. And confidence cannot
> help you at all here, because there are no words to attach a confidence to.
> That's why I split mute zones out rather than counting them as dead zones: a
> confidence-based monitor isn't merely bad at them, it's structurally blind to
> them. It's a hole a voice agent falls into silently."

### Be ready for "do you know what our confidence actually is?"

He may know precisely what the value represents. Do not bluff. The honest answer
is also the strongest one:

> "From outside it is a black box. I treat it as an *ordinal* signal within a
> single model — never comparable across vendors, which is why every cross-model
> claim goes through within-model percentiles. L2 asks whether a thin learned
> layer can turn it into a calibrated probability, and it can: ECE 0.051 to
> 0.008. If you can tell me what it actually represents, that changes what the
> calibrator should be conditioning on."

Ending on a question is good here. Labs engineers like being asked.

**One caution:** "vendor confidence is not a calibrated probability" is the
*premise* of that layer, not a criticism of their product. Say it that way.

**If the third arm comes up here** — and it will, because "what is your
confidence" and "how do you compare it to someone else's" are the same
conversation: ElevenLabs documents theirs as a **log-probability**, Deepgram
documents a confidence without saying what it is a posterior over. That
difference is the reason nothing in this study ever pools confidences across
vendors. It is a methods statement, not a scoreboard.

### On novelty

Labs values ambition, and the positioning rule says claim none. Not in tension:
claim no novelty of **method**, be ambitious about **what you would build next**
— the live agent leg, DRR-parameterised reverb axes, and what field data would
earn that simulation cannot.

---

## The four things to be sure you say

1. *"This genre is well-trodden and nothing in the method is novel."* — first
   five minutes, unprompted.
2. *"The active-learning leg is a null, and here is the control proving the
   method works on a surface that has structure."*
3. *"My headline was inflated because I paired two different populations. I
   found it by listening to the audio, not by running a test, and the fix
   produced a category — the mute zone — that a confidence monitor cannot see at
   all."*
4. *"I pre-registered how you'd rank those clips and I was wrong — and my
   pre-registration was badly written, because it never allowed for 'unequal but
   backwards.' The measured half is untouched: a human has a preference, the
   model scores them equal."*

**And the one thing to be sure you do NOT say:** any Scribe number, until the arm
has run. `[[PENDING SCRIBE]]`
