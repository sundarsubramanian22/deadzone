# INTERVIEW_INTERNAL.md — the decision doc

*Private. Never on screen.*

**What this is.** Not talking points — the prose is already known. This is for a
**decision** deep-dive: the question is almost always *"why did you choose this?"*,
rarely *"what was the math?"* Open it at the section being shown and read the
bullet for the choice being questioned.

**Per README section:**

- a **decision block** — each bullet is `Decision:` what was chosen · `Why:` the
  reasoning · **`If pushed:`** the concession, said before it is extracted;
- a **thin glossary** — only the terms a strong backend/QA engineer who is not a
  speech person would stop on.

Populations travel with the numbers as parentheticals rather than in a table,
because *which clips a figure is over* is this project's signature bug and the one
number-fact that has to survive being skimmed.

`<!-- SUNDAR: fill -->` marks a rationale that is **not recorded anywhere in the
repo**. Those are genuine gaps, never guesses — a motive invented here would get
repeated out loud.

---
## Read this first — the three rules the numbers obey

**1. Name the population, always.** nova-3 ran **40 clips**; whisper-base and
elevenlabs-scribe ran a **10-clip subset**. nova-3's dead-zone rate is **1.14 %
(2/176) on 40 clips** and **0.57 % (1/176) on 10** — both correct. Quoting either
without its clip count is this project's signature bug, committed at least four
times by people who already knew about it. `results/dead_zones.csv` holds **96 rows
across all three arms**; row 0 is not nova-3's.

**2. Verify a dead-zone row by identity, not by column position.**
`gap == mean_conf − (1 − wer_spoke)`. The schema puts `rt60_measured` immediately
before `mean_conf`, and 0.680 reads exactly like a plausible confidence — that
off-by-one was made once and caught only by arithmetic.

**3. Where a document and an artifact disagree, the artifact usually wins — with
one live exception.** `results/model_arms.{json,txt}` still prints the hallucination
exhibit as **3 → 49**. It is **11 → 47**. There the documents are right and the
artifact is stale; say so *before* anyone opens the file.

---

## Known disagreements between the README and its artifacts

Found while building these tables. The README is frozen; these are the answers, not
corrections to it.

| § | what the README says | what the artifact says |
|---|---|---|
| masthead | `40 × 176 × 3 = 10,560` | 40 × 176 × 3 = **21,120**. **10,560 is the correct row count** — only nova-3 ran 40 clips (7,040 + 1,760 + 1,760). The equation is shorthand that assumes all three arms ran 40. **Answer "no, it doesn't multiply — here's why", never defend it as arithmetic.** |
| §4 | "two confident, one 'a little gap'" | "a little gap" is **u21**, a *confident* call. The marginal pair is **u40** — *"both pretty bad … not 100 % eveyeron would agree"*. Only **u26** is `non_marginal: true`, which is why the sealed prediction fails on both *named* pairs (u40, u21). |
| §7 | "nova-3 leads on **every** statistic" | True of every *confidence* statistic. In the same table nova-3 is **worst on silent rows** (24.5 % vs Scribe's 4.4 %) and has the **most mute conditions** (12 vs 2). That is not a slip to hide — it is the sting made two paragraphs later: its dominant failure carries no confidence, so its failures are invisible to the signal that ranks it best. |
| §8 | "7 of 9 families are deletion-dominant (**front-end work**)" | 7 are deletion-dominant. **Only 5 earn a fix** — `engine` (Δ −0.127) and `codec=none` (Δ −0.104) are deletion-dominant in the *improving* direction and the code's own sign gate marks them `NO_FIX_NEEDED`. |
| §9 | "**clean-condition** median differs ~3×" | 3.28× is `word_conf_median_all` — the median over the **whole grid** (0.998 / 0.963 / 0.304). Clean-corner medians are 0.99999 / 0.99805 / 0.86278 → **1.16×**. The argument survives and is stronger: a 3× spread on the quantity you would threshold, across the grid you would deploy in. |

---

## Claims with no artifact behind them — know before offering

| claim | status |
|---|---|
| Scribe's orthography is non-deterministic: "4 identical calls, 5 of 6 clips differed, up to 0.727 WER (word error rate)" | **Measured once, never persisted.** `scripts/probe_scribe_orthography.py` makes **one** call per clip and writes nothing; there is no `--repeat` flag and no `results/scribe_repeat.json`. The figures trace to a prose reason string in `deadzone/model_compare.py`. The *consequence* — the exclusion — is enforced in code and pinned. The *evidence* cannot be opened. |
| The implied-fix column in §8 | `IMPLIED_FIXES` is a hardcoded lookup keyed on `(family, dominant_edit)`. The edit composition is **measured** over 7,040 transcriptions; the fix is a **mechanistic inference**. No denoising, boosting or gating experiment was run. There is no A/B. |
| "insertions under babble are competing-speech capture" | The 92 % foreign-token evidence is real, but it is **not babble-specific**: engine 0.936 is *higher*, road 0.889. Per reference word babble has the **lowest** insertion rate of the three (0.018 vs 0.027 / 0.028) — its larger raw count is 9× the rows. Scope the claim to insertions under **any competing source**. |

---
## §1 · The core idea

- **Decision:** Make the unit of finding "wrong *and* still confident" — the confidence–accuracy gap per condition — instead of WER per condition.
  **Why:** Confidence is the lever a voice agent actually acts on: it gates *use this transcript* vs *ask the caller to repeat*. Wrong-but-unsure costs a re-prompt; wrong-and-sure commits and propagates the error downstream. One aggregate WER cannot tell those two apart, and they are not the same product failure.
  **If pushed:** The genre is not novel — controlled-degradation grids are well-trodden and credited in the README (Ko 2017, Scheibler 2018, Shah 2025). The delta is the lens and the typed fingerprints, not the method.

## §2 · The pipeline

- **Decision:** Degrade clean recordings synthetically instead of recording in real noisy rooms.
  **Why:** In a field recording the room, the mic, the noise and the codec all move at once, so when WER moves you cannot say what moved it. Turning one knob with everything else frozen is the one thing field data cannot give you. Every *ingredient* is still real — measured impulse responses, real DEMAND noise, real ffmpeg passes — only the assembly is controlled.
  **If pushed:** Fidelity to any one deployment was never the goal, and the cost of the choice is measured rather than asserted: the sim-vs-real leg shows simulated rooms rank conditions correctly but get the level wrong and recover none of the dead zones.
- **Decision:** Fix one composition order — room → noise → mic → codec — in code, not as a knob.
  **Why:** That is the physical path an utterance actually takes: mouth → room → air → mic → wire. Any other ordering is a chain that cannot exist in the world, so measuring it would be measuring an artifact of my own pipeline.
  **If pushed:** Order effects are therefore unstudied. I fixed the order rather than crossing it; that is a stated limitation, not a claim that order does not matter.
- **Decision:** Build and test three "trap" functions (SNR — signal-to-noise ratio — mixing, room application, error classification) before running any of the grid.
  **Why:** Each of the three produces plausible audio and plausible numbers when subtly wrong, with no exception raised. Compute SNR over the whole file instead of over speech and it is deflated by an amount that varies per clip — a confound, not an offset. Leave the reverb tail in the level normalization and downstream SNR de-calibrates by an amount that *grows with RT60 (reverberation time — how long a room takes to decay 60 dB)*, so the bug looks exactly like a reverb finding.
  **If pushed:** These are the only stages where a silent bug poisons every downstream number, so they got the test budget first; the analysis layers got the same discipline only later, and that is where the remaining silent defects turned up.
- **Glossary.** **WER (word error rate)** — (substitutions + deletions + insertions) ÷ reference words. 0 is perfect; can exceed 1.0 if the model emits more words than were said.
- **RIR (room impulse response)** — a recording of one room's echo signature from one speaker position to one mic position; convolve clean speech with it and the speech is "in" that room. One RIR = one room + one distance.
- **direct-path delay** — the leading samples of an RIR before its loudest peak, i.e. mouth-to-mic flight time. Left in, it shifts the whole clip later in time and the scorer reads that pure timing offset as recognition error.
- **active-speech energy** — average power measured only over the frames a voice detector flags as speech, never over the whole file (which is mostly silence).

## §3 · The knobs

- **Decision:** 40 utterances, one speaker, one accent, one sitting.
  **Why:** WER precision is governed by total reference *words*, not clip count — 40 utterances ≈ 363 reference words per condition, about 2 points of standard error, tight enough to separate adjacent grid cells. 100 clips would only reach ~1.4 points while multiplying every API call by 2.5×. Holding speaker, room, distance and session identical stops room tone and mic distance becoming hidden extra factors on top of the ones I am turning. (Populations: nova-3 ran all 40; whisper-base and elevenlabs-scribe ran a 10-clip subset, so every cross-arm number is a 10-clip number.)
  **If pushed:** Nothing here generalizes across talkers or accents. This is an instrument for isolating acoustic factors, not a claim about ASR (automatic speech recognition) in general.
- **Decision:** Cap the SNR axis at 20 dB rather than 25.
  **Why:** Measured, not chosen. The corpus's own inherent SNR is ~25–28 dB, so a 25 dB request under-delivers by about 2.5 dB while 20 dB lands within 1 dB. A level I cannot actually deliver would put a per-clip error straight into the axis I am attributing effects to.
  **If pushed:** So the grid has no "nearly clean" rung. The clean reference arm — the raw recordings, no room and no noise, which is the only true null since the composer always applies both — covers that end instead.
- **Decision:** Spend the complete 4×4×3×3 factorial on babble; sample engine and road at 32 corners only.
  **Why:** A complete factorial — every combination, 40 clips in every cell, no holes — is what licenses an *exact* variance decomposition instead of a sampled estimate, so it was worth spending the calls densely on one noise type rather than thinning three. Babble is competing *speech*, not just energy — the model can transcribe the wrong talker instead of the target, a failure mode engine and road noise don't have (those just mask). It's also the realistic voice-agent background: a drive-thru, a call center, a cafeteria. So the full grid went on the noise type that breaks ASR in the most interesting and most deployment-relevant way, and engine and road were sampled at the corners just to check the story generalizes. With unlimited budget I'd have run all three fully crossed; babble was the one worth the full grid.
  **If pushed:** Engine and road are 16 conditions each, so any claim about them is corner-sampled, not a surface; the exact decomposition is a babble-core result and is quoted as one.
- **Decision:** Keep two codec levels — g726 and opus-lowrate — rather than one.
  **Why:** They break differently: g726 drives substitutions (a wrong word comes out), opus-lowrate drives deletions (the word never comes out). One codec level would have hidden a mechanism, not just a magnitude. g726 over AMR-NB because stock ffmpeg ships AMR decode-only, and depending on a source-built ffmpeg would make the grid unreproducible on the next machine.
  **If pushed:** Which narrowband codec was used is a methods fact and it is stated — a silent substitution would not have been defensible.
- **Glossary.** **RT60** — how long a room's reverberation takes to decay 60 dB. Bathroom ≈ 1 s, treated studio ≈ 0.2 s. The four levels snap to the nearest of 16 real measured rooms, so the axis is 4 discrete rooms, not a continuous sweep.
- **SNR (dB)** — how much louder the speech is than the noise: 0 dB = equal power, +20 dB = speech 100× the noise.
- **babble** — many people talking at once (cafeteria, station). The hardest noise class because it is *competing speech*: the model can transcribe the background instead of the target.
- **g726 / opus-lowrate** — g726 is ITU-T narrowband telephony ADPCM (adaptive differential pulse-code modulation) at 16 kbit/s (the "phone line" level); opus-lowrate is modern VoIP Opus starved to 8 kbit/s (the "bad VoIP" level). Both applied as real ffmpeg encode-decode round-trips, not modelled.
## §4 · ▶ Demo 1 — the disagreement

> ### 🗣 SAY THIS ON SCREEN
> **Spoken script — read aloud, not reference.** Every number below is an **ILLUSTRATIVE PLACEHOLDER**: read the real values off the screen from whatever clip loads. They are not measurements, nothing pins them, and they must never be quoted as results.
>
> *The moment:* two clips get identical WER (e.g. 0.222 vs 0.222) but the transcripts differ — one dropped 2 words, the other substituted 1 and dropped 1.
>
> *Why the WER is identical (say this):*
> "Both score 0.222 — two errors out of nine words. But look what broke: clip 1 just dropped two words, clip 2 swapped one and dropped one. Different rooms, different failures — one model went quiet, the other guessed wrong. WER is a headcount: it adds up substitutions, deletions, and insertions and divides by length. It counts how many errors, never which words or what kind. So two acoustically opposite conditions collapse to the same number."
>
> *Why it matters (the takeaway):*
> "That's the whole argument for not deploying on WER alone. Same score, but one needs denoising, the other needs dereverberation — WER can't tell you which. That's what the fingerprints section fixes: classify the error type, because deletion means a front-end fix and substitution means a decoding-side fix."
>
> *If pushed "so what would you use?":* → the §8 fingerprints answer (edit-type classification).

- **Decision:** Open on a listening test, and call it a **hook, not a finding**.
  **Why:** The one thing I need him to feel before any chart is that a human ear and this model's score are not the same axis. Fastest way to make someone believe that is to let them get it wrong themselves. I couldn't settle by ear which condition was actually harder — which is the whole reason I built an instrument instead of trusting QA-by-listening. *(Populations: the corpus check behind it is nova-3 on **all 40 clips**; whisper and Scribe only ever ran a 10-clip subset.)*
  **If pushed:** n=1, unblinded — the listener knew what the experiment was looking for, and went the opposite way to my sealed prediction in 2 of 3. Concede it as theatre with a real measurement behind it: the evidence is the model side, never the ear.
- **Decision:** Build the pairs from clips where nova-3 scores the two conditions **exactly equal and non-zero**.
  **Why:** I wanted the model equally *wrong* on both, not equally right — a tie at 0.00 proves nothing, a tie at 0.333 is the disagreement in its sharpest form. Both conditions also move exactly one knob (`codec=none`, `rolloff=0`), so the pair is a counterfactual rather than two arbitrary cells.
  **If pushed:** Yes, the on-stage tie is constructed — I selected for it. The unselected version is the corpus check: on **18 of 40** clips these conditions tie with no cherry-picking, and the paired difference is −0.018.
- **Decision:** Bootstrap the A-vs-B difference over **clips**, and report that it spans zero.
  **Why:** The same 40 sentences go through both conditions, so clip difficulty is a blocking factor — take per-clip differences first, then resample clips. Resampling words or cells throws the pairing away and returns a falsely narrow interval, which would have let me claim a separation I don't have.
  **If pushed:** 10,000 resamples of the 40-clip difference vector, seed 0. The CI (confidence interval) [−0.065, +0.031] is **failure to separate, not proof of equality** — say it before he does.

**Glossary** · **paired bootstrap** → resample the per-clip *differences* (not the raw scores) thousands of times and read the spread; keeps each clip's two measurements tied together. · **"spans zero"** → the interval contains 0, so the data is consistent with either direction — you failed to detect a difference, you did not demonstrate sameness.

## §5 · The finding: the confidence–accuracy gap
*Populations: nova-3, **40 clips** × 176 conditions = 7,040 rows, 0 failed. 169 conditions scored; the other 7 are mute — no words, so no confidence exists to score. Never quote a nova-3 rate without "40 clips": the other arms ran 10.*
- **Decision:** Lead with the **continuous** result — ρ = −0.980, mean gap +0.147, overconfident in 154 of 169 — and demote the dead-zone count to a subsection titled "why it is not the headline".
  **Why:** The correlation and the gap need no threshold, so nobody can move them by arguing with my cut-offs. The count is a presentation choice sitting on two hardcoded numbers, and I can show it moving. Leading with a number I can watch slide would be leading with my weakest claim.
  **If pushed:** ρ is over the same 169 conditions either way — the honest pairing scores confidence and accuracy on the identical clips (`wer_spoke`, −0.980); the mismatched one scores accuracy over all 40 (−0.952). Same rows in, different estimand.
- **Decision:** Set the operating point at `WER ≥ 0.30` **and** confidence in the model's own top 40%, then publish a full sweep of both.
  **Why:** I did not derive either threshold and I say so in the module's own docstring — they are round numbers. Given that, the defensible move wasn't to justify them post-hoc, it was to measure what they were worth and publish the answer against myself. As for 0.30 rather than 0.10: I wanted "dead zone" to mean the transcript is *substantially* broken — roughly a third of the words wrong — not merely imperfect. 0.10 is where task success starts to erode; 0.30 is where the transcript is clearly failing, so the bar is unambiguous. It is a defensible default, not something I derived — the module says so — and that is exactly why I don't lead with the count: I ran the full threshold sweep, the count is fragile, and the finding I actually stand on is the continuous ρ, which needs no threshold at all.
  **If pushed:** Concede immediately, then land the sweep — across the 63-point box the count runs **0 → 86** (fold-range 87×), the two published members are flagged in only **32%** of it, verdict **FRAGILE**, fixed in the module before any arm was swept. Then the counter-punch: **zero** conditions are confidently wrong at WER ≥ 0.40 — at my published confidence cut and every stricter one. That one only breaks if you also loosen "confident" to the model's top half or wider.
- **Decision:** Publish the estimand defect as a headline callout, including that v1 said 6 dead zones.
  **Why:** The correction moved everything in the *flattering* direction — ρ −0.952 → −0.980, dead zones 6 → 2 — which is exactly why nobody was motivated to look and exactly why it's worth showing. A bug that makes your result look better is the one your incentives will never find. And the fix is a guard, not a patch: `find_dead_zones` is a view over the classifier, so you cannot obtain dead zones without being handed the mute zones too.
  **If pushed:** What caught it was **listening**, not a test — the v1 dead-zone clips sounded intelligible. Right row count, no NaN, no exception, nothing red. Two individually correct averages over different clip sets, subtracted.
- **Decision:** Define "confident" as a **within-model percentile**, not an absolute confidence value.
  **Why:** The three arms return confidence on unrelated scales — Scribe a log-probability, Deepgram a [0,1] scalar — so a shared absolute cut-off would be measuring vendor conventions, not model behaviour. Percentile is the only fair form and it's enforced in code, not by convention.
  **If pushed:** It only fixes half of it. The WER side of the same test is still **absolute**, so "dead-zone rate" is a level statistic wearing a within-model label — which is precisely why Scribe's 7 dead zones collapse to 0 under the normalizer. They were spelling, not confident error.

**Glossary**
- **confidence** → a number the ASR returns alongside each word claiming how sure it is; a vendor scalar, not a calibrated probability. **within-model percentile** → that confidence rewritten as its rank inside *this model's own* range, 0–1, so two vendors' scales can be compared at all.
- **Spearman ρ** → correlation on ranks: does higher confidence go with lower error? −1 = the model always knows when it's failing; 0 = confidence carries no warning.
- **estimand** → the precise quantity a number estimates, *including which rows it was computed over*. The word this project's central bug is named after.
- **operating point** → the pair of thresholds that turn a continuous score into a yes/no flag. A presentation choice, not a measurement.
- **calibration** → whether a stated confidence matches observed accuracy: of everything you called 0.9, were 90% right?
- **dead zone / silence-driven / mute zone** → confidently wrong on the clips it spoke on (2) · looked wrong only because clips vanished (4) · returned nothing at all, on every clip (7). Three mechanisms, three different fixes; a confidence alarm is blind to the third.
## §6 · ▶ Demo 2 — the hero

> ### 🗣 SAY THIS ON SCREEN
> **Spoken script — read aloud, not reference.** Every number below is an **ILLUSTRATIVE PLACEHOLDER**: read the real values off the screen from whatever clip loads. They are not measurements, nothing pins them, and they must never be quoted as results.
>
> *The moment:* the degraded transcript collapses (WER 0→0.6) but confidence barely moves (0.997→0.687), and the model reports high confidence (e.g. 0.955) on an inserted word the speaker never said — higher than its average on words it got right. Meanwhile the hardest correct word (a surname) scores the lowest confidence in the utterance.
>
> *Why confidence barely moved (say this):*
> "Accuracy collapsed — WER went from zero to 0.6 — but the model's self-report only dropped from 0.997 to 0.687. It stayed confident while it fell apart. That gap is the dead zone: a system trusting this confidence would commit to a broken transcript."
>
> *Why it's confident on a wrong word (say this):*
> "It reported 0.955 on a word the speaker never said — higher than its average on the words it got right. A modern decoder's confidence isn't purely 'did I hear this' — it's partly 'does this word fit what I've already committed to.' The invented word makes a fluent English sentence, so it's linguistically confident even though there was no audio behind it. The confidence rides the fluency, not the acoustics."
>
> *Why that's the dangerous part (the takeaway):*
> "And it's inverted where it matters. The hardest correct word — a surname — is its least confident, because a name has no language pattern to lean on; it can only hear it. So the signal is backwards exactly where deployment needs it: sure about the filler it invented, unsure about the entity it nailed. That's why confidence thresholding alone isn't enough, and why I built the calibrator."
>
> *If pushed "is that how Deepgram's model actually works?":*
> "I can show the behavior from my data — confident insertions, unconfident entities. The mechanism I'm inferring from how autoregressive decoders generally work; I verified it from Whisper's published source, but Deepgram's internals aren't public, so for nova-3 it's a well-motivated hypothesis, not something I confirmed in their decoder."

- **Decision:** the hero demo makes two *real* Deepgram calls on stage instead of replaying cached numbers.
  **Why:** the whole claim is that the confidence number is not a stored constant. You have to hear the degraded audio to believe the condition is real, and watch the confidences come back off the wire to believe the model isn't hedging — a cached demo proves neither.
  **If pushed:** it can fail live, so it is built to fail quietly — no key, no wifi, a vendor error or a timeout gives one explanatory line, falls through to archived measurements labelled `CACHED`, and exits 0; `make demo-replay` runs the identical beat offline. nova-3 only, because Scribe's transcript for byte-identical audio is a coin flip and Whisper is local, so there is no wire to watch.
- **Decision:** the interviewer picks the clip from a menu (or random) rather than me showing one exemplar.
  **Why:** it is the answer to "did you cherry-pick the demo clip." The menu is *derived* — read out of `results/dead_zones.csv` and the master table, with no hardcoded clip list anywhere in the file, so re-run the grid and the menu moves with it.
  **If pushed:** the punchline is a ladder, not a script. The full "the most confident wrong word was invented and outranked every correct word" sentence prints **only** when the payload that just arrived supports it; otherwise it prints the strongest claim the data does support and says which one it fell back to.

## §7 · Three models
**Populations — say it before quoting anything:** nova-3 ran **40 clips** (7,040 rows); whisper-base and elevenlabs-scribe ran the same **10-clip subset** (1,760 rows each). Chart and cross-arm figures are the **matched 1,757 cells per arm**; **ECE (expected calibration error) and AUROC (area under the ROC curve — how well confidence ranks good transcripts above bad) are each arm's own full run**, not the intersection (neither subtracts one arm from another, so a full-run figure is the right one — it is simply not the matched population). nova-3's dead-zone rate is **1.14 % (2/176) on 40 clips** and **0.57 % (1/176) on 10** — both correct, and quoting either without its clip count is this project's signature bug.

- **Decision:** three arms — a commercial spine (nova-3), an open baseline (whisper-base), a commercial peer (Scribe) — with the two non-spine arms on the 10-clip AL subset rather than all 40.
  **Why:** the spine had to be a commercial model that exposes per-word confidence, because that *is* the headline signal; the open arm proves the finding isn't hostage to one vendor's API; and Scribe is the only other arm returning per-word confidence, which turns the question from *commercial vs open* into **is nova-3's self-knowledge a property of commercial models, or of nova-3?** The 10 clips were already the fixed subset for the AL oracle and both sim2real arms, so reusing them keeps every arm joinable — and Scribe deliberately ran the *same* ten, so adding a third arm did not shrink the intersection. nova-3 is the *subject* of the study; Scribe and Whisper are *context* — they exist to show nova-3 is neither uniquely good nor uniquely broken. 10 clips establishes that comparison without re-running the entire 40-clip grid three times, which for Scribe is real money per call and for Whisper is real runtime. And `whisper-base` was a deliberate choice of a cheap, fast, unambiguous open-source **floor**, not a size-matched competitor: its job is "here is what an uncalibrated open model does", not a fair fight. Which is also why the confound below gets conceded in the same breath — and if I extended this, `whisper-large-v3` on the same grid is the obvious next run to separate size from vendor.
  **If pushed:** one open model, and it is also the only small one (74M) — commercial-vs-open, vendor and size all move together, and nothing in this design separates them.
- **Decision:** Scribe stays in the rank comparisons and is barred from cross-model WER.
  **Why:** its orthography changes between identical calls — four calls on the same bytes returned different transcripts on 5 of 6 entity clips — so the offset is **variance, not bias**: a constant you can subtract, a coin flip you can't. Rank statistics are only attenuated by that noise (its ρ −0.820 is therefore a lower bound), while anything thresholded on an absolute WER moves across the line.
  **If pushed:** its dead-zone rate exists — 7 of 176 strict — and is deliberately neither drawn nor quoted, because under the normalizer all seven fall from WER 0.30–0.43 to 0.08–0.14, i.e. **zero**. They were spelling, not confident error. "Within-model" does not imply "scale-free." And the evidence for the exclusion is prose, not an artifact: the probe writes nothing to disk.
- **Decision:** Whisper's ECE is *refused*, not computed.
  **Why:** on 69 of 1,757 rows the hypothesis-word count and the confidence-list length still disagree after alignment, so binding them would score confidence *k* against word *k+1* and train a calibrator on mislabelled data. A number I can't trust is worse than a blank cell.
  **If pushed:** a weaker protocol would fit (skip the misaligned rows — 12,584 words), and I publish that size so the gap is visible, but it is a **different estimand** from the other two arms', so printing it in the same column would be a silent protocol change, not a fallback. AUROC survives the same block because it needs only a ranking of one score per utterance against a bad/good label — never a word-to-confidence binding.
- **Decision:** I decline the "commercial beats open" reading the table appears to offer.
  **Why:** Scribe and Whisper swap depending on which column you read — Scribe ahead on ρ (−0.820 vs −0.590), Whisper ahead on utterance AUROC (0.888 vs 0.737) — and their AUROC intervals overlap almost entirely ([0.616, 0.845] vs [0.659, 0.935]), so that ordering is a point estimate, not a separation. With three arms and one open model, "commercial" is confounded with vendor and with size.
  **If pushed:** nova-3 leads every *confidence* statistic and is the **worst** arm on failure mode — 24.5 % silent rows and 12 mute conditions against Scribe's 4.4 % and 2 — and a deleted word emits no token, so its dominant failure is invisible to the exact signal that ranks it first.

**Glossary**

- **AUROC** → given one bad utterance and one good one, the chance the score ranks them the right way round; 1.0 perfect, 0.5 a coin flip. Rank-based, so it is safe across vendors whose confidence scales are unrelated.
- **calibration / ECE / temperature scaling** → calibration asks whether "0.8 confident" means right 80 % of the time; ECE is the average gap between claimed confidence and observed accuracy, in bins (0 = honest); temperature scaling is the standard one-parameter fix — divide the confidence's logit by a fitted `T`, where `T > 1` flattens everything toward 0.5, so Scribe's 4.11 against nova-3's 1.39 says "far more overconfident" in one number.
- **orthography / normalizer** → orthography is how a transcript is *spelled* (`405` vs `four zero five`), not what was heard; a normalizer is a transform applied identically to reference and hypothesis so spelling differences cancel out.
- **logprob** → the natural log of a probability (always ≤ 0; 0 means certain). ElevenLabs returns one per word and the adapter `exp()`s it, which is what makes Scribe a confidence-bearing arm rather than a WER-only one.
- **hallucination / repetition loop** → fluent text that was never spoken, often stuck repeating one phrase. The exhibit is **11 words → 47**; `results/model_arms.{json,txt}` still prints the stale **3 → 49** — here the documents are right and the artifact was never regenerated.
- **insertion / substitution / deletion** → the three typed edits from aligning reference against hypothesis; WER = (sub + del + ins) / reference words. Insertions are unbounded, which is how WER exceeds 1.0; deletions emit no token, so they carry no confidence at all.
- **rank vs level statistic** → a rank statistic depends only on ordering (Spearman, AUROC); a level statistic depends on absolute values crossing a fixed cut-off (a dead-zone rate at WER ≥ 0.30, ECE). Noise attenuates the first and shoves the second across its threshold.
- **saturation / ceiling** → a large share of scores pinned within a hair of the maximum and therefore *tied*: 47.4 % of Scribe's words sit within 0.001 of 1.0, and tied values cannot be ordered by any threshold or percentile.
- **mute condition** → a condition where no clip returned a scorable word. Deliberately *not* counted as a dead zone: confidently wrong and entirely absent are different failures, and a confidence monitor is blind to the second because there is no confidence to be low.
## §8 · Failure fingerprints
- **Decision:** Score *what kind* of error each condition produces, not just how many. (nova-3, 40 clips, 7,040 rows, 63,888 reference words — none of §8 is on the 10-clip subset.)
  **Why:** A WER of 0.30 made of deletions and a WER of 0.30 made of entity substitutions are two different products breaking. A scalar can't tell them apart, so the alignment keeps the typed edit — `(op, ref_word, hyp_word)` — instead of collapsing to a count.
  **If pushed:** Dominance is `argmax |Δ vs that family's own clean half|`, not the tallest bar — which is why road noise reads del 0.37 / sub 0.19 on the chart and is still classified substitution-dominant (Δsub +0.059 vs Δdel +0.014). The tall deletion bar is the background level every cell in this grid carries; what road *causes* is substitutions.
- **Decision:** Map each dominant edit type to a *different class of fix* — deletion → front-end, substitution → decoding-side prior.
  **Why:** A deletion means no token was ever emitted, so no amount of boosting or constrained decoding can recover it — only changing the audio before the model sees it can. A substitution means a wrong word *was* chosen from degraded evidence, which is exactly where a prior helps. Diagnose it backwards and you spend months on the wrong engineering.
  **If pushed:** **The edit composition is measured; the implied fix is an inference.** It is a hardcoded lookup keyed on (family, dominant edit) — a mechanistic argument from the edit type, not a measured intervention. No dereverberation, no keyword boosting, no speaker gating was ever run; there is no A/B anywhere in this project. Also: 7 of 9 families are deletion-dominant but only **5** earn front-end work — `engine` and `codec=none` are deletion-dominant in the *improving* direction and the code refuses to prescribe for them (the README still says 7 → front-end; this is the correction).
- **Decision:** Score entities on their own axis, separately from overall WER.
  **Why:** WER weights "the" and "Nguyen" identically; a voice agent does not. Slots are the fields the task actually needs to survive, and a `critical` slot fails the task on its own whatever WER says. Entity error rate 0.633 vs WER 0.511 on the same rows.
  **If pushed:** The gap is reported signed either way — if entities had degraded *slower*, that would have been the finding. And the naive read of it is backwards: it's carried by proper nouns (0.646) and spelled letters (0.613); digit words are the *least* destroyed class (0.361), below function words.
## §9 · Confidence compared *within* a model
- **Decision:** Never compare raw confidence across arms — every cross-arm claim goes through a within-model percentile, enforced at a chokepoint in code rather than by convention. (Audit population: the 1,757 cells all three arms ran — 10-clip subset × 176 conditions.)
  **Why:** The three arms' word-confidence medians differ ~3× over the whole grid, so a shared absolute threshold is meaningless. A convention holds right up until someone in a hurry writes `if conf > 0.8`; a chokepoint that raises does not.
  **If pushed:** "Within-model" is not automatically safe — a per-arm statistic can still be a *level* statistic. Scribe's dead-zone rate is computed per-arm but thresholds an absolute WER, so it isn't quotable at all (7/176 strict → 0/176 under the normalizer: spelling, not confident error).
- **Decision:** Exclude `elevenlabs-scribe` from cross-model WER, as a registry entry that raises — no flag includes it.
  **Why:** Whisper's orthography offset is a **constant** (+0.090) — characterize once, normalize symmetrically, keep comparing. Scribe's is a **per-call draw**: four identical calls on byte-identical audio returned different transcripts on 5 of 6 probe clips, worth up to 0.727 WER on the same input. A constant can be subtracted; a coin flip cannot. The reason lives in the registry value because an exclusion whose justification sits in a commit message gets "fixed" by the next person who reads the code.
  **If pushed:** It cost me the arm that would have won — Scribe beats nova-3 on raw WER (0.410 vs 0.433). And the determinism evidence is the weakest thing on the page: the probe writes no artifact, so it's the one claim not pinned by `test_report_numbers.py`, and because the grid ran one call per cell every Scribe number carries that variance unquantified.
## §10 · Run it
Almost no decisions here — it's the run surface, and every spine beat is rehearsable offline with no key and no network.
The one deliberate line: `deadzone/` is free to re-run in a loop, `scripts/` spends money or overwrites artifacts, and that boundary is a checked invariant (`--max-calls`), not a comment.
## Appendix (a)(b)(c) — things I tried that did not work
- **Decision:** Publish the active-learning null as a finding, and refuse to move the target until it won.
  **Why:** The target isn't hand-picked — the code sets it to the median final fidelity the *random* arm reaches on its own full budget, so neither arm can be handed a bar the other can't clear. At that bar, 2 of 8 active seeds reached it against random's 4 of 8, and the median evals-to-target is `inf` for **both** arms, so **no savings ratio exists and none is claimed**. Tuning the threshold until active won was the one genuinely dishonest move available.
  **If pushed:** The acquisition function demonstrably worked — 58.3% of its picks landed near the decision contour against random's 30.0% (that pairing is the published RT60 arm; the 21.1% figure is the DRR (direct-to-reverberant ratio) arm's random baseline, don't cross them) — it did its job and the job didn't pay. The synthetic control still passes, so this isn't a broken implementation; the null belongs to the surface. **Every seed ran against a GP surrogate oracle — no seed was confirmed end-to-end against the live API.**
- **Decision:** Test the obvious objection to the null — "you gave the GP (Gaussian process — the statistical model doing the predicting) the wrong reverb axis" — *with a permutation control*, not on its own.
  **Why:** Re-running in DRR coordinates and finding no gain only shows that one fix failed. Running all 24 permutations of the same four room values shows *why* it failed: the physically correct assignment ranks 18th of 24 (p = 0.75), so a random relabelling does as well as the right one and the coordinate was never what was holding it back.
  **If pushed:** The ceiling is honest — the grid requests 4 reverb levels, so the axis is 4 discrete rooms and any re-parameterisation is a relabelling of 4 points. This says nothing about a reverb axis with enough rooms to have a shape; the fix is more rooms, not a better coordinate. And no absolute-fidelity claim for DRR survives the split check — the statement is "no improvement", never "better".
- **Decision:** Publish the failed listening pre-registration *and* the flaw in its own rubric.
  **Why:** The predicted direction held in 1 of 3 pairs, and the sealed rubric enumerated only *unequal → holds* and *equal → fails* — never "unequal, but backwards", which is what actually happened. Under its own wording the failure scores as a PASS. A rubric that can't fail is decoration, and the contrast is the lesson: the project's other pre-registration had a *numeric* decision rule, fixed before any audio existed, and confirmed cleanly.
  **If pushed:** The listening half is n = 1 and unblinded — a hook, not evidence. The result is the model side: those two conditions differ by −0.018 WER with a 95% CI spanning zero, and 18 of 40 clips score exactly equal.
- **Decision:** Restrict both arms to the clips they share before comparing simulated rooms against measured ones.
  **Why:** The real arm ran 40 clips, the sim arm 10. Compared as shipped it reads a 19.9-point gap; clip-matched it reads 12.1 — 7.8 points was pure clip difficulty wearing a simulation gap's clothes. Counterfactual isolation is the whole premise of the rig, so it has to apply to the comparison itself.
  **If pushed:** 19.9 is retracted and must never appear as a result. The guard reports-and-proceeds rather than raising, because a 10-vs-40 split is the *designed* state (the sim arm was subset to save spend); it only raises below 3 common clips. And D4's dead-zone sets are scoped to those 10 clips, so they are deliberately not the §5 table's.
## Glossary
- **typed edits (substitution / deletion / insertion)** → the three kinds of word error kept individually rather than summed: a wrong word emitted, a word that produced no token at all, a word invented with nothing behind it.
- **front-end vs decoding-side fix** → change the audio before the model sees it (dereverberation, gain, mic placement) vs change how it turns acoustics into text — *keyword boosting* = telling the decoder in advance which words matter; *entity-aware / constrained decoding* = restricting output to known-valid forms.
- **orthography** → how a transcript is *spelled*, independent of what was heard: `405-912-77` vs "four zero five nine one two seven seven". Both are correct; naively scored, one costs 8 word errors.
- **text normalizer** → a deterministic canonicalization applied identically to reference and hypothesis before scoring, so a spelling convention can't masquerade as an acoustic effect.
- **within-model percentile** → where a confidence value sits inside its own arm's distribution (0–1 rank), instead of its raw value.
- **GP surrogate / active learning** → fit a model to what you've measured that predicts unmeasured points *with an uncertainty bar*, then let it pick the next condition to test instead of running a fixed grid; the uncertainty bar is what makes the picking possible.
- **permutation test** → shuffle the labels every possible way and see where the real arrangement ranks among them; with 4 rooms there are only 24 arrangements, so it's exhaustive and assumption-free.
- **pre-registration** → writing down the prediction *and* the decision rule that would falsify it, committed, before looking at the data.
- **Jaccard** → set overlap, shared ÷ union: 1.0 identical, 0.00 disjoint.
- **ASR (automatic speech recognition) / STT (speech to text)** → the same thing: audio in, text out. "STT" is the product word, "ASR" the research one.
- **WER (word error rate)** → (substitutions + deletions + insertions) ÷ reference words. 0 = perfect. Can exceed 1.0, because insertions are unbounded.
- **RIR (room impulse response)** → a recording of one room's echo signature; convolve clean speech with it and the speech sounds like it was said in that room.
- **RT60 (reverberation time)** → seconds for a room's echo to decay 60 dB. Bathroom ≈ 1 s, treated studio ≈ 0.2 s.
- **SNR (signal-to-noise ratio, dB)** → how far speech sits above the background. 20 dB = quiet room; 0 dB = noise as loud as the speech.
- **ECE (expected calibration error)** → how far a confidence score is from being a real probability. 0 = a reported 0.8 is right 80 % of the time.
- **AUROC (area under the receiver-operating-characteristic curve)** → how well a score *ranks* bad cases above good ones. 0.5 = coin flip, 1.0 = perfect separation. Needs only an ordering, which is why it survives where ECE cannot be computed.
- **CI (confidence interval)** → the range the estimate would plausibly fall in on a re-run. One that spans zero is a failure to separate, **not** proof of equality.
- **ADPCM (adaptive differential pulse-code modulation)** → the compression family behind G.726, the narrowband telephony codec used here.
- **VAD (voice activity detection)** → deciding which samples contain speech; it is what makes "SNR over active speech only" possible.
- **RMS (root mean square)** → the standard loudness measure of a waveform; the trap in `apply_rir` is that a reverb tail inflates it.
- **HF (high frequency)** → the top of the spectrum, which is what a cheap microphone loses — the `mic_rolloff` knob.
- **API (application programming interface)** → here, a vendor's hosted transcription endpoint; "API calls" is the unit both cost and wall-clock are counted in.
- **sim2real** → how far a simulated testbed's numbers sit from the same measurement made with real ingredients.
- **DRR / C50** → two measures of how much sound arrives straight from the mouth versus bounced off walls (C50 = energy in the first 50 ms vs everything after); both describe a room in a way RT60's decay time does not.
## Operational — read before you present

### 1. The `u11` per-word confidence table — **the answer to "is your confidence signal usable as a gate?"**

Recovered from `git show be4aed2:report/INTERVIEW_RUNBOOK.md` (that file has since been **deleted** — it was the presenter's most-read document and nothing checked its numbers) and **fully re-verified this session against `results/master.csv`**.

**Clip `u11`, condition `rt60-0.45_snr-0_engine_g726_roll-0`** (the measured #1 nova-3 dead zone), model `nova-3`:

| | |
|---|---|
| reference | `deliver it to sofia martinez at eighty eight elm street` |
| dead-zone hypothesis | `deliver it to sofia martinez at three eight l three` |
| WER | **0.300** (10 ref words · 3 sub · 0 del · 0 ins · 7 match) |
| utterance mean confidence | **0.8488** |
| clean control (raw recording, no condition) | WER **0.000**, mean confidence **0.9610** |

Per-word, in hypothesis order — confidences align 1:1 with the hypothesis tokens (10 words, 10 confidences), **verified**:

| # | hypothesis word | outcome | confidence |
|---|---|---|---|
| 1 | `deliver` | right | 0.8683 |
| 2 | `it` | right | 0.9921 |
| 3 | `to` | right | 0.9589 |
| 4 | `sofia` | right | 0.7359 |
| 5 | **`martinez`** | **RIGHT** | **0.3361** ← *lowest in the utterance* |
| 6 | `at` | right | 0.9687 |
| 7 | `three` ← `eighty` | **WRONG** | 0.8256 |
| 8 | `eight` | right | 0.9434 |
| 9 | **`l` ← `elm`** | **WRONG** | **0.9260** |
| 10 | **`three` ← `street`** | **WRONG** | **0.9334** |

**Say it as the honest version — it is stronger than the triumphant one:**

> *"At the **utterance** level confidence is informative in direction and useless in magnitude: WER goes 0.000 → 0.300 while confidence moves only 0.961 → 0.849. At the **word** level it is worse than uninformative. The **lowest** confidence in the utterance, 0.336, is on `martinez` — the one hard word it got **right**. Two of the three substitutions score **above** the utterance mean. So a threshold tuned to catch these errors fires on `martinez` and **still** lets `three eight l three` through. The signal is wrong in both directions at once — a false alarm on the word it nailed, and silence on the three it invented. That is the argument for the calibrator, and it is also why I won't tell you confidence thresholding is sufficient."*

**Supporting facts, all verified:**
- This is a **delivery address destroyed** — `eighty eight elm street` → `three eight l three` at ~0.93 confidence. It is the entity-destruction fingerprint (D2: proper nouns are the most-destroyed word class at **0.646**, spelled letters **0.613**) with the clearest downstream consequence.
- **The clip was chosen from the grid, not by taste, and the rejections are the argument.** `u20` (higher utterance conf 0.890, but two of three errors are insertions at 0.565/0.529 — *"the model DID flag those"*, and they'd be right); `u38` (headline substitution only 0.544 confident); `u34` (the money clip, `eight hundred`→`three hundred euros`, but 0.723/0.675); `u18` (WER 0.500 but **all four errors are deletions — no hypothesis token, no confidence, nothing to put on screen**); `u07`/`u26`/`u31` (their **clean** transcripts aren't clean — `okafar`, `koalski`, split `wifi` — a control that's already wrong destroys the contrast).
- **Live is nova-3 only, and that is deliberate.** A live Scribe call could return either orthography, so it would be a coin flip on stage. The one arm whose output is reproducible is the one that goes live.
- **The condition's identity check** (never count columns in `dead_zones.csv` — that mis-read happened once): `gap_spoke = mean_conf − (1 − wer_spoke)` → `0.829418 − (1 − 0.306109) = 0.135528`, which reproduces the stored `gap_spoke`. `0 of 40` clips silent, so confidence and WER are over the same 40 clips and the gap needs no asterisk.

### 2. Two on-screen landmines — **both confirmed on disk this session**

**(a) The dashboard's model-toggle buttons render ALPHABETICALLY. The leftmost button is NOT the selected arm.**
- `dashboard/app.js:1442` does `Object.keys(DATA.models)` and appends a button per key **in that order**, with no sort applied at render — but the payload's key order is already alphabetical.
- Built payload key order, read out of `dashboard/deadzone.html`: **`['elevenlabs-scribe', 'nova-3', 'whisper-base']`**.
- `default_model` is **`nova-3`** — pinned to `SPINE_MODEL` (imported, so the two cannot disagree), explicitly *"the spine, not whichever arm sorts first"* (`build.py:796–804`).
- **So the page opens on the MIDDLE button.** If you point at the leftmost button while saying "this is the arm we're looking at", you are pointing at `elevenlabs-scribe` — the arm that is **rank-only and excluded from cross-model WER**. Point at the `aria-pressed="true"` one, or just say the arm's name.

**(b) `elevenlabs-scribe` is the FIRST block in `results/confidence_gap.txt`. Reading that file top-down on screen shows the WRONG arm's headline.**
- Line **1**: `D1 confidence-accuracy gap — model 'elevenlabs-scribe'`
- Line **47**: `... model 'nova-3'` ← **this is the one you want**
- Line **96**: `... model 'whisper-base'`
- The first `Headline:` line a reader hits (line 32) is Scribe's: *"mean word confidence 0.976 while WER is 0.428 (n = 10 clips)"* — **10 clips, and a 3.98% dead-zone rate that is NOT quotable** (all 7 of Scribe's strict dead zones fall to WER 0.08–0.14 under the normalizer; they are orthography, not confident error).
- nova-3's headline is at line 73: *"rt60 = 0.45 s, SNR = 0 dB, engine, g726, rolloff 0 → mean word confidence **0.829** while WER is **0.306** (n = **40** clips, 363 ref words; 0 clips came back empty)."*
- **If you open this file live, jump to line 47.** `sed -n '47,95p' results/confidence_gap.txt` is the safe command.

**(c) Bonus landmine, found while verifying — there is NO 25 dB SNR level in this grid.** `snr_db` is `{0, 5, 10, 20}` and the MANIFEST records the ceiling and the reason (*"corpus inherent SNR measures ~25–28 dB, so a 25 dB request under-delivers by ~2.5 dB"*). If any narration says "SNR twenty-five decibels", it is describing a cell that was never run.

### 3. `results/model_arms.{json,txt}` still prints the stale `3 → 49` — **the docs are right and the artifact is wrong**

**Say this BEFORE anyone opens the file**, because it inverts this repo's usual rule (*quote the artifact, never the summary*).

- `results/model_arms.txt` **line 171** reads `[whisper-base u02 @ rt60-1_snr-5_babble_opus-lowrate_roll-1]  3 ref words -> 49 hyp words`. Lines **175** and **179** carry the same `3 ref words ->` prefix for the two companion examples (`-> 38`, `-> 34`).
- **The correct figures, recomputed from `results/master.csv` this session:**

| condition | n_ref | hyp words | WER | alignment |
|---|---|---|---|---|
| `rt60-1_snr-5_babble_opus-lowrate_roll-1` | **11** | **47** | **4.1818** | 1 match · 10 sub · 0 del · 36 ins |
| `rt60-0.7_snr-0_babble_g726_roll-1` | **11** | **34** | 3.0909 | 0 match · 11 sub · 0 del · 23 ins |
| `rt60-1_snr-5_babble_opus-lowrate_roll-0.5` | **11** | **36** | 3.1818 | 1 match · 10 sub · 0 del · 25 ins |

- **The cause, and it is worth 20 seconds because it is the project's own failure mode in the project's own reporting code:** `hallucination_report` normalizes spoken numbers to digits and then tokenizes `[a-z']+` — **letters only**. That builds 8 digit tokens from `four zero five nine one two seven seven` and then discards them, collapsing the 11-word reference to 3. The ratio was reported as **16.3× against a true 4.3×**. **The loop is real; the magnitude was part model, part tokenizer.**
- `u02` reference (`recording_manifest.csv`): `call maria at four zero five nine one two seven seven` — **11 words**, confirming the correction.
- The hypothesis is worth reading aloud once: *"I'm gonna go here and do a little bit of the work. You call her, you have a passport, you have a file. You have a file, you have a file, you have a file, you have a file, you have a file. You have a file."* — a degenerate repetition loop. **WER understates it**: WER caps damage at one error per reference word, while a 47-word hallucination handed to a downstream LLM is unbounded harm. Second independent argument for why WER is not the deployment metric.
- Every prose document now says **11 → 47** (`README.md:253`, `report/writeup.md:468`, `report/UNDERSTANDING.md:320`, `report/INTERVIEW_INTERNAL.md:1766`). **Only the generated artifact is stale.**

### 4. The three killer questions — verbatim-ready

Source `report/UNDERSTANDING.md` §7, with every figure re-verified against `results/` this session.

---

**Q1. "Your best mechanistic claim is n = 4. And your own table has C50 at −0.800. Why is DRR the story?"**

**This is the sharpest attack available and it uses a column you published. Do not defend the strong form.**

> *"You're right, and I should state it that way in the document. n = 4 — the exact one-sided permutation p is 0.042, Kendall two-sided is 0.083, and it's the only claim in the write-up without an interval. C50 sits at −0.800 and the entire separation from DRR is one discordant pair, Bar versus Campground, whose C50 values differ by 0.19 dB. So the DRR-versus-C50 ordering is not established by this data — and I'd also note that 'early-energy ratios beat T60 for intelligibility' is a known room-acoustics result. I shouldn't present it as a discovery.*
>
> *What actually survives, and is the more useful finding, is one level up: my rt60 axis snaps each request to the nearest measured room, so `rt60 = 0.45` is a Bar and `rt60 = 0.7` is a Campground Dining Hall — unrelated rooms with different direct-to-reverberant ratios. The non-monotonicity along that axis isn't a property of reverberation; it's a property of **which four rooms I curated**. Re-sample the axis and the dip moves — which is exactly what happened between two of my own scans: 0 of 6 surrogate-proposed cells reproduced against the real oracle. That's a warning about how reverb benchmarks are constructed, and it doesn't need n > 4. The fix is more rooms, not a better coordinate — and I know that because I tested the better-coordinate hypothesis and it failed a 24-permutation control at p = 0.75."*

**Backup facts if pushed:** three of the four rooms are exotic — a **Bar**, a **Campground Dining Hall**, a **Shower**. Not one is a car cabin, an office, or a phone at 5 cm. `data/rirs/` holds 16 curated RIRs; `master.csv` contains exactly **4** distinct `rir_key` values, so the binding constraint is the **grid**, not the library. That is the sharpest version of the sim-vs-reality exposure — sharper than "it's a simulator."

---

**Q2. "So your headline is two conditions out of 176, at thresholds you picked. Is there a finding here?"**

**Concede the count immediately; pivot to the continuous form, which needs no threshold; then give the methodological finding, which is the real one.**

> *"The count is fragile and I'd rather not lead with it. `wer_hi = 0.3` and `conf_pct_hi = 0.6` are defaults I never varied, and the count runs **13** at (0.30, 0.50), **2** at (0.30, 0.60), and **0** at (0.35, 0.60). I have shipped that sensitivity table now — across the defensible box the count spans 0 to 86, and the two published members are flagged in only 32% of it.*
>
> *The threshold-free version is the finding: nova-3's confidence tracks its own error at Spearman **−0.980** across the **169** conditions that produced words, and it's still overconfident in **154 of 169**, mean gap **+0.147**. So the model is mostly self-aware — which is the dangerous part, because a system tuned on average behaviour trusts it in the residual. I'd also flag that my #1 dead zone's confidence is **0.829** against **0.962** on the mildest cell — the **64th percentile** of that model's own distribution, so many production thresholds would already catch it.*
>
> *The finding I'd actually defend is the one the correction produced. The published headline was **6** dead zones. It was wrong: confidence averaged over the clips that spoke, WER over all 40 including empty transcripts. Right row count, no NaN, no exception, no failing test — the defect was entirely in **subtracting two averages over different populations**. A human listening to the exemplar clips found it; no test could. And it forced a taxonomy that matters operationally: **2 dead zones, 4 silence-driven, 7 mute zones** where the model emits nothing at all — and a confidence-based monitor is **structurally blind** to those, because absent is not wrong. That's the deliverable: the early-warning signal I proposed cannot see its own worst failure mode, and I can tell you how large that hole is — deletions are **69.3%** of all nova-3 errors."*

**Every figure verified:** count surface (0.30,0.50)=13 · (0.30,0.60)=2 · (0.35,0.60)=0 (`dead_zone_sensitivity.txt`); ρ −0.980 spoke / −0.952 all-clips, gap +0.147, 91% overconfident (`confidence_gap.txt`), and 154/169 recomputed directly from `master.csv` this session; conf 0.829 at `conf_pct` 0.64.

---

**Q3. "What IS Deepgram's confidence score, actually?"**

**The question you are least prepared for, and the one a Deepgram engineer is most likely to ask. Do not bluff — and note that the "I never computed the minimum" version of this answer is now STALE. The prediction was tested and REFUTED.**

> *"I treated it as an ordinal signal. I don't know whether it's a decoder posterior, a lattice-derived score or an acoustic-model output — I didn't ask, and I only established what it is **not**, which is a calibrated probability. But I did go and characterise its measured behaviour, and one of the three things I expected turned out backwards.*
>
> ***One — the anchor, which I'd never quoted.*** *Clean reference **0.9622** per word (0.9621 per clip) on the mildest grid cell at WER 0.0084; the 40 raw recordings with no condition at all give **0.9619**, so the mildest cell is an almost exact proxy for true clean. Best condition **0.981**, and it falls to **0.422** on the harshest condition that still emits words — a span of 0.56. Without that ruler nobody can judge whether 0.829 is 'confident'; it's the 64th percentile.*
>
> ***Two — `utterance_conf`.*** *I capture it on all 10,560 rows and no analysis module read it. I've now measured it: it is genuinely **distinct** from the word mean (pearson 0.926, mean absolute difference 0.071), but **not better** at ranking bad transcripts — AUROC 0.936 against the word mean's 0.944, paired delta −0.007 [−0.015, +0.000]. So leaving it unread cost nothing measurable, and I can say that with an interval instead of a shrug.*
>
> ***Three — and this is the one I got wrong.*** *I predicted the **minimum** would be the operationally relevant aggregate: commit-or-ask-again is a question about the worst word in the utterance, especially when it's a phone number. I tested it. **The mean wins.** AUROC **0.944** for the arithmetic mean against **0.877** for min — a paired difference of **[−0.080, −0.055]**, so min is significantly **worse**, and so are p10 and p25. Nothing beats the mean on this arm, and that's now a measured result rather than an unexamined default. The reading I'd offer is that nova-3 spreads its information across words rather than concentrating it in the weakest one — which is what you'd expect of a per-frame acoustic score rather than a lattice-derived one. I'd want to know from you whether that's right.*
>
> *There's a control in that table on purpose: `n_words`, which is confidence-free, scores 0.606. Min and the low percentiles fall with utterance length for combinatorial reasons, so the control is there to show how much of their standing is length rather than acoustics.*
>
> *What a thin learned layer does to it I can tell you exactly: ECE **0.051** raw, **0.035** under temperature scaling, **0.008** feature-conditioned, on held-out **conditions** — and I can give you the discount schedule it learned. What I can't yet tell you is **why** it behaves that way, and that's the part I'd want to spend time on with someone who knows the decoder."*

⚠️ **`report/UNDERSTANDING.md` §4.8's consequence 3 — "the minimum is the operationally relevant number and it never was computed" — is SUPERSEDED.** The paragraph is left unedited in that file *as the record of what was believed before it was measured*, under a banner that says so. **Do not reproduce the stale version.** "I predicted min would win, tested it, and it lost" is a stronger answer than either belief on its own.

**Every Q3 figure verified in `results/confidence_char.{txt,json}`:** corner per-word 0.9621669, per-clip 0.9620988, WER 0.0084 · raw-capture per-clip 0.9619329, per-word 0.9613957, n = 40 clips / 363 words · highest condition 0.98077 (`rt60-0.7_snr-20_babble_none_roll-0.5`) · lowest 0.42197 (`rt60-0.45_snr-0_babble_none_roll-0`) · mean AUROC 0.944 [0.928, 0.956], min 0.877 [0.857, 0.896], ΔAUROC −0.067 [−0.080, −0.055], p10 0.917, n_words control 0.606 · utterance_conf 0.936, paired −0.007 [−0.015, +0.000], pearson 0.926 · 1,000 bootstrap replicates **over `clip_id`** · "bad" defined as row WER ≥ 0.3. **Saturation caveat if it comes up:** 15.3% of nova-3's words are within 0.001 of 1.0 and those words are 0.999 correct — tied words cannot be ordered by any threshold, so saturation removes resolution exactly where a commit/re-prompt rule needs it.

### 5. The self-assessment line — deploy it **before** he does

> *"The fair summary of this project is that **the methods depth is ahead of the domain depth. I can build the instrument. I haven't yet spent enough time inside the model.**"*

Land it at the end of Q3, not as a concession under pressure. It is true, it is the thing an adversarial reviewer concluded independently (`UNDERSTANDING.md` §6), and saying it first converts the biggest criticism of the work into evidence of calibration.

### 6. From `report/_demo_internal_notes.md` — stage directions that are **not for the screen**

That file quotes **no figures on purpose** — where a note depends on a number it names the artifact instead, so there is no second copy to drift. `tests/test_demo_listen.py::NoPresenterNotesOnScreen` asserts none of these phrasings reach stdout **and proves its matcher is real by finding them in that file**. If you move one back into a script, that test fails — intended behaviour, not a break.

**Stripped from the listening demo (`demos/demo_listen.py`, 2026-08-06):**
1. *"do not read an edit-type signature off these three clips"* — the three demo pairs happen to show **substitutions**, while the grid-level fingerprint for `rt60 ≥ 0.7` is **deletions**. If asked *"so reverberation causes substitutions?"*, the answer is **no, and these three clips are not the evidence either way** — the per-factor signatures are in `results/fingerprints.json` / D2, computed over the full grid, not three hand-picked pairs. Say it aloud if it comes up; do not pre-empt it on screen.
2. *"do NOT repair the failed pre-registration on stage with the DRR number"* — it fits the result now in view and it is **post-hoc**. The note was removed partly because **it named the mechanism it was telling you not to reach for**, on screen, handing the room the post-hoc story anyway. A stage direction that leaks its own subject is worse than none. The script now says, in the first person: *there is a tidy mechanism I could hand you, and I am not going to.*

**Stripped from the hero (`demos/demo_hero.py`, when `make demo` became the hero) — four things, none of them wrong, all of them crowding out the two numbers the beat exists to compare:**
1. **The cost line and the per-minute rate with its date.** Quoting a vendor rate *with the date it was true* is the right discipline — but two calls cost a fraction of a cent, and a dollar figure on the projector invites a conversation about pennies at the exact moment the room should be looking at a confidence bar. **Quote `results/MANIFEST.json`, never a demo's screen.** The constants still live in `demos/demo_live.py` (`USD_PER_MINUTE`, `RATE_AS_OF`, `GRID_CALLS`, `GRID_USD`) as a fallback target. If asked "what did this cost?": **$3.70 total, 14,606 calls, ~998 minutes of audio** — with the model literal and the rate date attached, from the freeze artifact.
2. **`run_id` and the row timestamp** in the reproduction check. An opaque hex id on a projector is noise; what the room needs is the delta and whether it cleared tolerance. Both live per-row in `master.csv`, and `tests/test_demo_hero.py` asserts every number the hero prints is re-readable from it.
3. **The MANIFEST paragraph.** The panel keeps the **argument** — a commercial model literal is re-pointed server-side, so a live call is not automatically the same experiment — and drops the file path. If the numbers moved it prints `MOVED` and why; if they agree it prints how tightly. *(Today they agree exactly: WER 0.300, conf 0.849.)*
4. Guarded by `tests/test_demo_hero.py::NothingOperationalReachesTheScreen`, which scans a replay run **and** a patched live run for a dollar amount, a per-minute rate, `run_id`, `MANIFEST` and `USD`, **pins each matcher with a positive control**, and asserts `demo_hero.py` does not import the pricing constants at all so they can't creep back through a helper.

**What deliberately STAYED on screen — do not "clean these up":**
- **The credential provenance line** (`credential DEEPGRAM_API_KEY, from .env`) — names the **variable**, never a value, never a prefix. It is there so a screen-share audience can see nothing secret is printed, and because *"which key did it use"* is the first question when a live call fails.
- **The fallback notice** — one yellow paragraph naming the cause, meant to be read aloud.
- **The population line** under every aggregate (`averaged over the N of M clips this condition produced words on`). **Not a stage direction — it is the estimand, and this project published a wrong headline once for want of it.**
- Four correctness signals that fire only in states that should not occur live: `(these two are not equal — this demo set has drifted from the grid table)` and `⚠ manifest and master.csv disagree on the paired difference…` — **if either appears on stage the segment's premise is gone; stop and say so.** Also `(results/master.csv not present — …)` (benign) and `(stdin is not a terminal: …)` (only in a piped run; on a live share stdin *is* a tty, so an interviewer never sees it).

**Two judgement calls left to a human:**
- **Naming the conversation that motivated the project.** The closing describes it without naming anyone, deliberately: framed as *what made me want to build this*, never as a claim about what any company has or hasn't studied. Naming the person is a call to make **out loud, in the room, where tone is available** — not in the terminal.
- **A live surface still carries the old finding-voice line.** `demos/demo_listen.py` no longer closes on *"you cannot QA a voice agent by listening to it"* — it closes on the **question**. The generated `results/audio/demo/DEMO_SCRIPT.md` still states it as a **conclusion**. Reconcile before rehearsing off both; SPEC J.7 is exactly this failure, found by a rehearsal rather than a test.

**`make demo-live` (the optional network beat)** — 2 real nova-3 calls, 12.1 s of audio, **~$0.0009**. Safe to schedule because it **cannot fail loudly**: no key, no network, a vendor error or timeout each print **one** explanatory line, fall back to the cached replay, and **exit 0**. Rehearse as `./.venv/bin/python demos/demo_live.py --offline` (byte-identical presentation, no key read at all); preflight with `--check`. **Where it goes:** after the payoff clip, or as the answer to *"can I see the confidences?"* — **never at the top**; the spine must establish itself offline first.
