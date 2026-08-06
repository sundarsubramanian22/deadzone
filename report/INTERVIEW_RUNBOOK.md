# Deep-dive runbook — 60 min with a Deepgram engineer

Audience: **one software engineer on Deepgram Labs** — prototypes, agents,
cutting-edge voice AI — and the model under test is their product.

- **Do not explain WER, SNR, RT60, or what a RIR is.** They know. Explaining
  basics is the fastest way to lose an expert's attention.
- **The subject under test is their product.** The framing is *"I built an
  instrument that finds where an ASR fails silently, and here is what it found"* —
  never *"your model is bad."* The instrument is the deliverable; Nova-3 is the
  first thing measured with it.
- **Do not say "streaming" — and "streaming-capable" is barred too.** SPEC
  Appendix K (`bcd685a`) dropped the streaming scope outright: **nothing was ever
  streamed**, so the word is a claim rather than a description, and
  "streaming-capable" was the previous compromise that still put it on the cover.
  The project's external title is now **"Deadzone: Silent Failures in Speech
  Recognition"**, and every arm is **batch** — Deepgram via the pre-recorded
  endpoint, ElevenLabs via batch REST, Whisper locally with full-file lookahead.
  Say **"a commercial ASR model that exposes per-word confidence"**, which is the
  real reason the arm is in the study at all: Whisper, Conformer and wav2vec do
  not expose one, so the silent-failure question is unanswerable on them. Where
  deployment relevance is the point, say **"the failure mode matters for a live
  voice agent, though everything here was measured batch."** Keep the word only
  where the *point of the sentence is the limitation* — there it is a competence
  signal, and it is strengthened rather than softened.
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
because for a live voice agent, a confidently wrong transcript is far more
dangerous than a visibly uncertain one. Confidence is what decides whether the
system commits or asks the user to repeat.*

Then the scope, in the same breath and before any result: **everything here was
measured batch — no arm streamed.** Said in the opener it is a boundary you
drew; extracted later it is a concession. Same fact, opposite read (see *The
batch scope* below).

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
> ⚠️ **Its section 7 changed on 2026-08-06 and this runbook changed with it.**
> The script used to close on a verdict about what listening can establish about
> an ASR; it now closes on **the question**, because one listener on three pairs
> chosen *because the model tied on them* cannot carry a conclusion. If you have
> a printout older than that, throw it away — see *the close* below.
>
> ✅ **The regeneration hazard is CLOSED.** Every hand-written block in that file
> now lives in the generator template, so `scripts/make_demo_audio.py` reproduces
> it rather than reverting it, and it refuses to overwrite anything a human has
> edited since. You still have no reason to run the generator before the
> interview — but it is no longer a landmine. `REGENERATION_HAZARD.md` in the same
> directory carries the detail.

**Beat 1 — the ranking. Ask, do not predict.** Show that
`PREREGISTERED_PREDICTION.md` exists and **leave it closed** — stating a
prediction aloud before someone judges is a demand characteristic, and this
project's whole subject is not fooling yourself with a number you wanted. Say
only *"I've written down what I think you'll say."*

Play, in this order:

- **Pair 2 — `blind_06` / `blind_01` (`u21`)** ← lead here
- **Pair 3 — `blind_08` / `blind_04` (`u26`)**
- Pair 1 — `blind_03` / `blind_07` (`u40`) — only if he wants a third.

`BLIND_SHEET.md` now lists its rows **in play order — pair 2, pair 3, pair 1** —
while keeping the pair *labels* unchanged, so the sheet and the order above agree
without renumbering anything. (The labels are the join key to `KEY.md`,
`DEMO_SCRIPT.md` and the sealed prediction; renumbering them would break every
cross-reference.) The sheet already asks him to say **how confident** he is on
each pair; keep that, because it is what makes his answers comparable to the
listener already run — and *"about the same"* is a real answer, not a failed
demo.

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
> preference in 3 of 3 pairs, and the model scores every pair exactly equal.**
> That disagreement is the *question* this project came out of — not one of its
> results; one listener can't make it one. *'And here's why'* is the part I don't
> have: I had a mechanism, I wrote it down, and the data went the other way."

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

**The close is a QUESTION, not a verdict — and this CHANGED on 2026-08-06.**
This beat used to land on a product claim: ~~*you cannot QA a voice agent by
listening to it*~~. **Retracted — do not say it.** It is a conclusion, and this
beat cannot carry one: **n = 1, three pairs, selected *because* the model tied on
them**, with the pre-registered direction failing in 2 of 3. Asserting a
conclusion off that is the exact error this project exists to study, committed in
the project's own demo, in front of the one person guaranteed to notice. **It is
the motivating hook, and as the hook it is stronger** — the question is the reason
there is an instrument downstream of it. `demos/demo_listen.py` closes this way
(`ff1eb28`) and so does `DEMO_SCRIPT.md` §7; land in the same place:

> "Whichever way you called those — and 'about the same' is a real answer — the
> model reports no difference at all. Not a small one: **none**. Every pair you
> heard it scored identically, and across all **40** clips the two kinds of damage
> differ by **−0.0178 WER**, an interval that spans zero. Your ears and that
> number are not measuring the same thing, and I had no way to settle which of
> them to believe by listening harder. 'Sounds fine to me' is an opinion and I
> couldn't check it — so I built something that could measure it instead.
> Everything after this is that instrument. This beat is the question it was
> built for, not one of its results."

**If he ranks them equal, nothing breaks.** One listener agreeing with the model
is worth no more than one disagreeing with it — which is precisely why the
interval underneath is the measured half and this half is only the question. Say
that, then go to Beat 2, which needs no ranking from anyone.

**What is measured, and is safe to state as a result:** the paired model-side
interval and the 18-of-40 exact ties above, and Beat 2's empty transcript at
20 dB SNR. Those are grid measurements; they do not depend on anyone's ears and
they read the same whichever way he ranked.

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

> **The model toggle has THREE arms now** — `elevenlabs-scribe`, `nova-3`,
> `whisper-base`. Checked: the dashboard's `default_model` is **`nova-3`**, so it
> loads correctly — but the buttons render alphabetically, so **the leftmost
> button is not the selected arm.** Do not click blind.
> The file has no such default: **Scribe is the FIRST block in
> `results/confidence_gap.txt`**, so anyone reading that artifact from the top —
> including you, on the day — reads the wrong arm's headline first. It opens with
> a mean confidence of 0.976 at WER 0.428, which looks spectacular and is not
> nova-3's number.

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
4. **Model comparison — the framing must not depend on who wins, and now it
   provably cannot.** Three arms: nova-3, elevenlabs-scribe, whisper-base.

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

   **The third arm sharpened this rather than just repeating it, and the sharp
   part is the one non-zero number.** With three arms, quote the **pairwise**
   Jaccard (the runbook flagged this before the arm ran; it now has teeth):
   `nova-3|whisper-base` **0.000** (0/70) · `nova-3|elevenlabs-scribe` **0.000**
   (0/8) · `elevenlabs-scribe|whisper-base` **0.101** (7/69). So **nova-3's dead
   zone is shared with no other arm**, and the only overlap in the study is
   between Scribe and the open baseline — and *those seven conditions evaporate
   entirely under the orthography normalizer.* The one apparent counter-example
   to "you cannot borrow the map" turns out to be a scoring artifact, which is a
   better story than a clean sweep would have been.

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

   **The Scribe leg — the arm ran, and the result is a reversal, not a ranking.
   This is the strongest single beat in §4. Give it the time.**

   The lead sentence, and do not bury it behind a table:

   > "**Which commercial model appears to know when it is wrong depends on a
   > scoring choice most benchmarks make silently.** Not a little — it *reverses*."

   On the **159 conditions all three arms spoke on** — the only population in
   which three arms can be ranked at all, which is §5's lesson applied to
   correlations instead of gaps — within-model confidence-vs-WER Spearman, CIs
   from a 4000-replicate bootstrap over conditions:

   | scoring | nova-3 | Scribe | Whisper | nova-3 over Scribe | Scribe over Whisper |
   |---|---|---|---|---|---|
   | **strict** (spine scorer) | −0.971 | −0.768 | −0.694 | **+0.203 [0.115, 0.312]** — separable | +0.074 [−0.112, +0.267] — **not** separable |
   | **normalized** | −0.971 | −0.936 | −0.709 | +0.035 [−0.002, +0.077] — **not** separable | **+0.227 [0.097, 0.376]** — separable |

   Strictly scored, Scribe is separable from nova-3 and indistinguishable from the
   open baseline. Normalized, it is indistinguishable from nova-3 and separable
   from the baseline. **Same rows, same confidences, opposite verdict.** A
   benchmark that skipped the normalization audit would have published a confident
   and backwards answer — which is this project's entire thesis, arriving one level
   up from where it was aimed.

   **Then the mechanism, because the reversal is not magic — Scribe's orthography
   is non-deterministic.** Four identical calls, byte-identical audio, same model
   literal: **different transcripts on 5 of 6 probe clips.** `A7X42` three times
   and `A seven X four two` once; `Q9J05` vs `Q nine J zero five`; and `u33` flips
   the *other* way, so it is not one consistent policy. Worth up to **0.727 strict
   WER on identical input**, with zero recognition difference between the forms.

   > "Whisper's formatting offset is a **constant**, +0.090 — characterise it once
   > and subtract it, which is what `cross_model_norm.py` does. Scribe's is a
   > **per-call draw**. That is variance, not bias, and you cannot subtract
   > variance. It also turns the two residuals my normalizer documents as fixed —
   > the leading zero in `Q9J05`, the letter run `AW` — into run-to-run noise. So
   > Scribe is scoped **within-model and rank-only**, and that is enforced in code
   > and pinned by a test, not left to convention: the cross-model WER paths
   > **raise** on it unless the caller explicitly passes `exclude_incomparable`.
   > There is no flag that lets it in."

   The generalisable claim, and it is the one worth leaving in the room: **a
   benchmark that makes one call per clip is measuring a coin flip on
   entity-bearing utterances.** Repeat-call variance belongs in the harness
   alongside the acoustic conditions. *(Honest provenance: the repeat-call probe
   is 6 clips × 4 calls and is not persisted to an artifact; the grid itself ran
   once per cell, like every other arm, so it carries this variance unquantified.
   Say that — he will ask how many calls.)*

   **What IS claimable, and say it in exactly this shape:** nova-3's
   confidence-shape edge **over Whisper** is a real cross-vendor pattern, not a
   one-model quirk. A second independent commercial vendor lands at −0.82 strict /
   −0.95 normalized while Whisper sits at −0.59 / −0.60. **Whisper is the
   outlier.** The caveat travels with the claim: **n = 3 with only one open
   model**, so *"commercial vs open"* and *"vendor-specific"* remain confounded.

   **The half that survives all of it — and it inverts the safety story.** An
   empty transcript is empty under any normalizer, so this one is immune to
   everything above. On the matched subset nova-3 returns **nothing at all on
   24.5 %** of clip-rows (431/1757) and goes fully mute on 12 conditions; Scribe on
   **4.4 %** (78/1757) and 2. **5.5×.** Under stress **nova-3 goes quiet and Scribe
   keeps talking** — and a deletion carries no hypothesis token, so **63.2 % of
   nova-3's errors carry no confidence at all** against **33.7 % for Scribe**.

   > "So the best-calibrated arm in the study has the *least* monitorable failure
   > mode. Its dominant failure is invisible to exactly the confidence-based early
   > warning this project proposes. That is not a knock on the model — it is the
   > reason I split mute zones out as their own category, and it is why the product
   > recommendation is confidence **plus** a 'did I get anything' check, not
   > confidence alone."

   It also qualifies the correlation table mechanically, and volunteer this rather
   than let him find it: **nova-3's ρ is computed over 164 conditions after its 12
   hardest were dropped for emitting nothing; Scribe's over 174 after 2.** A model
   that goes silent on its worst conditions is scored on an easier set. That is
   precisely why the table above is restricted to the common 159.

   **Comparability rules that must survive contact with an expert:**
   - Scribe reports a **log-probability**; `exp(logprob)` is its own scale and is
     **not** commensurate with a Deepgram acoustic confidence or with Whisper's
     segment proxy. Every cross-model claim goes through
     `within_model_conf_percentile`. Say this before he asks.
   - **The orthography audit was a gate, and it is the reason the arm is scoped the
     way it is** — `scripts/probe_scribe_orthography.py`, run and read before the
     rows were trusted. The measured shifts: nova-3 **−0.014** (already word-form,
     so ~0 is the *expected* result and it is the control on the normalizer
     itself), whisper **+0.090** (recovering digit orthography, not accuracy),
     Scribe **+0.064** — but Scribe's is the only one that is *not repeatable*,
     which is the whole verdict.
   - **A third arm did NOT narrow the matched comparison, and say so if it comes
     up.** Scribe ran the same 10-clip AL subset, so the intersection is unchanged
     at 10 clips × 176 conditions = **1757 rows per arm** (3 dropped for Whisper's
     3 failures). An arm on a *different* subset would have shrunk it; the arm
     census in `results/model_arms.txt` prints the number either way. Quote the
     census, never a bare WER.
   - **With three arms, quote the PAIRWISE Jaccard, not the all-arm one.** The
     all-arm figure now means *"flagged by every arm"* — a stricter and different
     claim that trends to zero for trivial reasons. It reads 0.000 here, and it
     would read 0.000 for uninteresting reasons too. The pairwise numbers above
     are the ones that carry the argument.
   - **Every dead-zone rate is flagged on `wer_spoke`, on the matched cells.**
     That is the §5 correction, and it applied to the new arm on day one rather
     than being discovered in it later — Scribe's `n_silence_driven` is **0**
     precisely because the two pairings were checked against each other from the
     first row.
   - **And the rate itself is not scale-free.** `dead_zone_flags` thresholds an
     absolute WER (`wer_hi = 0.3`), so *no* arm's dead-zone rate survives an
     orthography change: Scribe's 7 → 0 under the normalizer, Whisper's 69 → 44.
     **This is a limitation of my own headline metric and you should volunteer it**
     — it is the same class of finding as the estimand mismatch in §5, caught
     before publication instead of after.

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
  **It replicates on the second commercial arm, which is what turns it from a
  nova-3 observation into a recommendation:** Scribe **0.165 → 0.076 → 0.034**,
  and its temperature is **4.11** against nova-3's **1.39** — i.e. a much larger
  correction, on a raw ECE that is itself an upper bound because its orthography
  mismatches are labelled as errors. Two vendors, same direction, same cheap fix.
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
- **"Why is Whisper's calibration row blank?"** He will notice — the L2 table has
  Scribe and nova-3 in it and Whisper missing. **It is blocked, not computed, and
  own that:**
  > "69 of 1757 Whisper rows still have a hypothesis-word count that disagrees
  > with the confidence-list length after re-alignment — `align_confidences`
  > recovered 33 and could not recover those. `word_records` refuses to zip,
  > because zipping binds confidences to the wrong words. The library offers
  > `on_misalign='skip'`, which drops them and gives me a number — and that number
  > would print in the same column as the other two arms while being fit on a
  > silently smaller and non-random set. That is a protocol change disguised as a
  > value. So the row is blank and the reason is printed next to it. I'd rather
  > show you a hole than a number I'd have to caveat."

  The two arms that *did* fit: nova-3 ECE **0.051 → 0.035 (temp) → 0.008
  (feature)**, T = 1.39; Scribe **0.165 → 0.076 → 0.034**, T = 4.11. Scribe's raw
  ECE is marked as an **upper bound** — its correctness labels come from the same
  alignment as WER, so its orthography mismatches get counted as errors. The
  raw→calibrated *improvement* still reads, because within one arm the labels are
  wrong in a fixed direction.

- **"Why ElevenLabs Scribe?"** The arm has run (§4.4). But the *day-one gate* is
  the better answer to this question, because it is about method rather than
  results:
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
  - Cost was never the constraint: $0.22/hr batch ⇒ the 10-clip subset ≈ **$0.43**
    (what actually ran), the full 40-clip grid ≈ **$1.72**. **The gate was the
    constraint**, and it was the right one — the orthography audit is what turned
    this from a third data point into §4.4's reversal.
  - The forward-looking half, and it belongs in the *next steps* answer rather
    than the results: `scribe_v2_realtime` exposes **the same per-word `logprob`
    over a websocket**, so it is the cheapest route to this project's first
    genuinely *streaming* arm — the boundary *The batch scope* below states
    outright, and the one place the word belongs, because there the point of the
    sentence IS the limitation.
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
  project is about. **§4.4's Scribe reversal is now in the never-cut set too** —
  it is the only place the hour demonstrates the thesis on the *benchmark* rather
  than on the model.
- **Nothing in the run-of-show requires the network.** The spine is offline by
  design and `make demo-live` is deliberately not a prerequisite of `make demo`.

### OPTIONAL: `make demo-live` (~20 s) — the only beat that touches the network

Two real nova-3 calls on one clip: raw, then the same clip in the **measured #1
dead zone**, with the per-word confidences printed as they come back. **12.1 s of
audio, 2 calls, ~$0.0009.** It is safe to schedule because it cannot fail loudly:
no key, no network, a vendor error or a timeout each print **one** explanatory
line, fall back to the cached replay, and **exit 0**.

- Rehearse it as `./.venv/bin/python demos/demo_live.py --offline` (byte-identical
  presentation, no key read at all). Preflight the real thing with `--check`.
- **Where it goes:** after §2's payoff clip, or as the answer to *"can I see the
  confidences?"* in §4. Do **not** put it at the top — the spine must establish
  itself offline first.
- **The exemplar is `u11`**, and the per-word numbers are the point, not the
  utterance mean. Ref: *"deliver it to sofia martinez at eighty eight elm street"*.
  In the dead zone it returns *"deliver it to sofia martinez at three eight l
  three"* — a delivery address destroyed — with:

  | word | outcome | confidence |
  |---|---|---|
  | `street` → `three` | **wrong** | **0.933** |
  | `elm` → `l` | **wrong** | **0.926** |
  | `eighty` → `three` | **wrong** | 0.826 |
  | `martinez` | **right** | **0.336** |

  Utterance mean **0.849**; clean control WER 0.000 at conf 0.961.

- **The honest framing, and deliver it as the honest one — it is better than the
  triumphant version.** At the *utterance* level confidence is informative in
  direction and useless in magnitude: WER goes 0.000 → 0.300 while confidence
  moves only 0.961 → 0.849. At the *word* level it is worse than uninformative —
  **the lowest confidence in the utterance (0.336) is on a word the model got
  right**, while two of the three substitutions score **above** the utterance
  mean. A threshold tuned to catch this failure fires on `martinez` and lets
  `three eight l three` through.
  > "So on this clip the signal is wrong in both directions at once — a false
  > alarm on the one word it nailed, and silence on the three it invented. That is
  > the argument for the calibrator in §6, and it is also why I won't tell you
  > confidence thresholding is sufficient."
- **It is nova-3 only, and that is deliberate, not an oversight.** Say so if
  asked: a live Scribe call could return either orthography (§4.4), so a live
  demo of it would be a coin flip on stage. The one arm whose output is
  reproducible is the one that goes live.
- One caution: a commercial model literal is updated server-side, so a live call
  months after the grid is **not** the same experiment. The demo prints the live
  result against the stored grid row every time, so a divergence is a talking
  point rather than a surprise. Today they agree exactly (WER 0.300, conf 0.849).

---

## Calibrating for a Labs SWE specifically

### The batch scope. Say it in the opener, and say it as a decision.

**The forward-facing framing no longer claims streaming.** SPEC Appendix K
(`bcd685a`, `8b42455`) dropped the scope; the write-up title and abstract, the
dashboard, `README.md` and `CLAUDE.md` all now state the batch scope outright
rather than implying otherwise. So this is not a concession you are bracing for;
it is a boundary you drew, and you should still volunteer it, because he builds
the thing it is a boundary on.

**All three arms are batch.** Deepgram via `listen.v1.media.transcribe_file`,
never `listen.live`; ElevenLabs via batch REST, never `scribe_v2_realtime`;
Whisper locally with full-file lookahead.

> "Everything in the grid is pre-recorded, and I stopped calling this a streaming
> study — nothing streamed, so the word was a claim rather than a description.
> Nova-3 is in here for the **per-word confidence**, which is what makes the
> silent-failure question askable at all; Whisper and Conformer don't expose one.
> The confidence signal and the acoustic failure modes carry over, but anything
> about endpointing or turn-taking genuinely requires the live socket — which is
> exactly why that is the named next step rather than something I'm claiming."

Volunteered, this is a scoping decision. Discovered by him, it looks like you
did not know the difference. Same fact, opposite read.

**If he greps the repo and finds the word:** `SPEC.md` §0–§13 and Appendices A–J
still say "streaming" verbatim, **deliberately**. SPEC is a dated log and the
project's rule is supersede forward, never edit backward — Appendix K records the
decision rather than rewriting the history that preceded it. That is a real
answer. *"I missed it"* is not, and neither is pretending the word was never
there.

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

**And the natural follow-up — "so is our confidence better than theirs?" — has a
real answer now, which is "I can't tell you, and here is exactly why."** Strict
scoring says nova-3 by +0.203 [0.115, 0.312]; normalized scoring says +0.035
[−0.002, +0.077], barely clear of zero, and since orthography noise only
attenuates a rank correlation that is an **upper bound**. The gap between those
two numbers is not measurement precision, it is a scoring convention. Give him the
question rather than a verdict — *"if you wanted to settle that properly, the
missing piece is repeat-call variance in the harness, not more conditions."*

### On novelty

Labs values ambition, and the positioning rule says claim none. Not in tension:
claim no novelty of **method**, be ambitious about **what you would build next**
— the live agent leg, DRR-parameterised reverb axes, and what field data would
earn that simulation cannot.

---

## The five things to be sure you say

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
5. *"Which commercial model appears to know when it is wrong reverses depending
   on a scoring choice most benchmarks make silently — and the reason is that one
   vendor's orthography is non-deterministic across identical calls, so a
   benchmark making one call per clip is measuring a coin flip."*

**And the five things to be sure you do NOT say:**

- **"You cannot QA a voice agent by listening to it."** Retracted 2026-08-06. It
  is a conclusion drawn from **one listener on three pairs chosen because the
  model tied on them**, with the pre-registered direction failing in 2 of 3. The
  listening beat is the **motivating hook**, and it closes on the question — see
  §2's close and `DEMO_SCRIPT.md` §7, which say the same thing in the same words.
- **"Streaming", or "streaming-capable".** Both are barred (SPEC Appendix K).
  Every arm is batch; say *"a commercial ASR model that exposes per-word
  confidence"*, and where deployment relevance is the point, *"the failure mode
  matters for a live voice agent, though everything here was measured batch."*
  The word survives only where the sentence's point IS the limitation.
- **Any Scribe dead-zone rate.** 3.98 % is real arithmetic on a metric that is not
  scale-free; it goes to **0/176** under the normalizer. The rate is not a
  property of the model.
- **That nova-3 is meaningfully better calibrated than its commercial peer.** The
  normalized gap is +0.035 [−0.002, +0.077] and that is an upper bound. Say
  "indistinguishable once orthography is controlled," which is both true and the
  more interesting sentence.
- **That commercial models know when they are wrong, as a class claim.** n = 3
  with exactly one open model. *"Commercial vs open"* and *"vendor-specific"* are
  confounded, and the one thing that IS supported — **Whisper is the outlier** —
  is narrower and survives both scorings.
