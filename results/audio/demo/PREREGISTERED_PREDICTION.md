# Pre-registered prediction — SEALED · **TESTED · FAILED**

> ⚠️ **STATUS 2026-08-05 — THIS PREDICTION HAS BEEN TESTED.** Verdict:
> **FAILED on direction — the predicted direction held in 1 of 3 pairs.**
> The listener found the *reverb-dominated* arm harder in two of three
> pairs; the prediction said the *babble-dominated* arm would be harder.
>
> **The sealed text below is unchanged and is not to be edited** — it is the
> record of what was committed before anyone listened. The result is
> appended in **OUTCOME** at the end of this file. Superseding forward,
> never backward. Read OUTCOME before quoting anything above it.

Do not show this, or say it aloud, until the listener has ranked the clips.
Announcing a prediction before someone judges is a demand characteristic; this
project's whole subject is not fooling yourself with a number you wanted.

Written before any listener heard anything. The blind names are frozen in
`KEY.md`, so this is checkable after the fact rather than a claim.

## The prediction

> The listener will rank **blind_07.wav** clearly harder than
> **blind_03.wav**, and **blind_01.wav** clearly harder than
> **blind_06.wav** — a confident, immediate, non-marginal call in both
> pairs.
>
> The model scores the two clips within each of those pairs **EXACTLY equal**
> (pair 1: 0.333 vs 0.333; pair 2:
> 0.222 vs 0.222), and across all
> 40 clips the paired difference is -0.0178 WER,
> 95% CI [-0.0654, +0.0310] — spanning zero.

## What each outcome means

- **Listener ranks them clearly unequal** → prediction holds; the human and
  model axes are decoupled and the demo lands.
- **Listener ranks them equal** → prediction fails. Say so. n=1 was never going
  to settle it, and the MEASURED half (the CI above) is unaffected either way —
  which is exactly why the two halves are labelled separately.

The prediction is about a human. The confidence interval is about the model.
Only one of those two things is a measurement.

---

# OUTCOME — tested 2026-08-05

*The sealed record above — the prediction and its outcome rubric — is
**unaltered**. The only additions above this line are the status banner and the
suffix on the title, both purely additive, so that a reader cannot reach the
prediction without meeting its verdict. Everything below is the result, and
every cell of it is derived from the session recorded in `LISTENER_SESSIONS`
(`scripts/make_demo_audio.py`) joined to `results/master.csv` — nothing here is
typed, so it cannot drift out of agreement with the record it scores.*

## The listener response, verbatim

One listener, one sitting. They were given `blind/BLIND_SHEET.md` and nothing
else, and wrote their rankings before this file was opened. Quoted unedited,
typos included, because a cleaned-up quote is a paraphrase:

> "bro 3 and 7 are both pretty bad but honestly think that 7 is better, not
> 100% eveyeron would agree with my decision tho. 6 better than 1, but by a
> little gap that everyone would agree with. i think 4 better than 8, others
> would say the same p sure."

"Better" here means *easier to understand* — the listener is naming the easier
clip of each pair, so the **other** clip is the one they ranked harder.

## Per-pair scoring

`A` = reverb-dominated (`rt60-1_snr-20_babble_none_roll-0`, Shower, measured
RT60 1.011 s, **DRR -10.02 dB**, babble at 20.0 dB SNR — i.e. quiet). `B` =
babble-dominated (`rt60-0.2_snr-0_babble_none_roll-0`, Restaurant, measured
RT60 0.193 s, **DRR +16.90 dB**, babble at 0.0 dB SNR — i.e. buried).

| pair | clip | predicted harder | listener ranked harder | direction | listener's own words on strength | model WER (A / B) |
|---|---|---|---|---|---|---|
| 1 (named) | `u40` | `blind_07` = **B** babble-dominated | `blind_03` = **A** reverb-dominated | ❌ **OPPOSITE** | "both pretty bad ... not 100% eveyeron would agree" | 0.333 / 0.333 |
| 2 (named) | `u21` | `blind_01` = **B** babble-dominated | `blind_01` = **B** babble-dominated | ✅ as predicted | "a little gap that everyone would agree with" | 0.222 / 0.222 |
| 3 (backup, not named in the prediction) | `u26` | — (pattern implies **B** babble-dominated) | `blind_08` = **A** reverb-dominated | ❌ **OPPOSITE** | "p sure others would say the same" | 0.250 / 0.250 |

Blind-name mappings, from `KEY.md` / `manifest.json:blind_map`:
`blind_03` = `pair1_A_reverb_u40` · `blind_07` = `pair1_B_babble_u40` · `blind_06` = `pair2_A_reverb_u21` · `blind_01` = `pair2_B_babble_u21` · `blind_08` = `pair3_A_reverb_u26` · `blind_04` = `pair3_B_babble_u26`.

## Verdict: **FAILED on direction — the predicted direction held in 1 of 3 pairs**

- **The predicted direction held in 1 of 3 pairs.** The listener found the
  **reverb-dominated** arm harder in two of them.
- **Scored strictly against the prediction's own sentence** — "a confident,
  immediate, **non-marginal** call in **both** pairs" — it fails in two of the
  two named pairs: pair 1 on **direction**; pair 2 on **magnitude**. A
  marginal call is a miss when the sentence says "clearly harder". The 1-of-3
  figure is the generous reading.
- **The predicted mechanism is refuted at these settings.** The prediction
  rested on informational masking from competing speech overwhelming the
  precedence effect. At **DRR -10.02 dB** it did not: heavy reverberation was
  judged harder than 0.0 dB babble in two of three pairs. The mechanism may
  still hold at milder DRR — this says nothing about that — but it does not
  hold where it was predicted to.

## What survives — stated precisely, with nothing added

**The listener had a stated preference in three of three pairs, and the model
scores each of those pairs exactly equal** — 0.333/0.333, 0.222/0.222,
0.250/0.250 (read from `results/master.csv`, not from this file's history).

So this claim stands:

> **A human and the model disagree about which clip is worse.**

And this one does **not**:

> ~~And here is why: the precedence effect makes reverb cheap for humans and
> informational masking makes competing speech expensive.~~

The demo was rebuilt around the surviving claim. `DEMO_SCRIPT.md` is
**direction-agnostic**: the interviewer ranks first, then learns the model has
no preference. That works whichever way anyone hears it, and it is stronger for
not depending on a direction chosen in advance.

**The measured half is untouched, as designed.** Recomputed from
`results/master.csv` (40 clips, nova-3): mean WER A **0.112266**, mean WER B
**0.130051**, paired difference **-0.0177841**, 95 % CI **[-0.0653509,
+0.0309692]** — 10,000-resample paired bootstrap over clips, seed 0, and it
spans zero. **Nothing the listener said moves this number**, which is the
whole reason the two halves were labelled separately in the sealed text above.

## The flaw in this document — worth more than the prediction was

The sealed **"What each outcome means"** section listed exactly two outcomes:

- listener ranks them **unequal** → prediction holds;
- listener ranks them **equal** → prediction fails.

**It never considered "unequal, but backwards."** That is a real design flaw,
not a technicality, and it has a concrete consequence:

> **Under the rubric as written, this result scores as a PASS.** The
> listener did rank every pair unequal. Meanwhile the prediction *sentence*
> — which named a direction — was wrong in two pairs out of three.

A rubric whose outcomes do not span what can actually be observed is a rubric
that cannot fail, and a pre-registration that cannot fail is decoration. The gap
between the sentence and the rubric is precisely the gap that lets someone score
a miss as a hit — including in perfectly good faith, months later, from a file
that looks rigorous.

**Rule adopted from this, for any future pre-registration in this repo:**

1. **Enumerate outcomes over the full observable space**, not over the two the
   author has in mind. Here that is at minimum: *predicted direction* /
   *opposite direction* / *no preference* / *inconsistent across pairs* — and
   the last is what actually happened.
2. **The decision rule must be fixed in advance and must be able to fail.**
   Compare SPEC section 5's `rt60 x snr_db` registration, which fixed a numeric
   threshold, a CI condition and a rank check before any data existed — and was
   confirmed by clearing them, not by being re-read leniently.
3. **Score the prediction's sentence, not a looser paraphrase of it.** If the
   sentence says "non-marginal", a marginal call is a miss.

This entry is the deliverable. A pre-registration that failed and is written
down is worth more than one that "held" because nobody wrote down what holding
meant.

## POST-HOC — a hypothesis, explicitly not a finding

⚠️ **Generated after seeing the result. It is not evidence. Do not present it
as an explanation.**

In pair 1, the clip the listener preferred is the one whose transcript KEPT
the proper noun, while the clip they found harder mangled it — verified
against `results/master.csv`:

- reference: `ask yamamoto to sign page twelve before we file`
- `blind_03` (reverb-dominated, ranked HARDER):
  `ask yavamoto to sign page twelve before v five`
- `blind_07` (babble-dominated, ranked easier):
  `ask yamamoto to sign page twelve in world five`

Both score WER 0.333. **The listener was judging audio, not transcripts, and
never saw either transcript**, so any link between their preference and the
entity outcome is speculation on n=1. It is recorded because it is checkable,
and entity survival is a real axis elsewhere in this project (D2: proper nouns
are the most-destroyed word class, 0.646) — not because it explains anything
here.

## Provenance of this section

Listener response: transcribed verbatim from the session and stored in
`LISTENER_SESSIONS` in `scripts/make_demo_audio.py`, which is also where the
per-pair calls live — so this section is regenerated from the record rather than
retyped beside it, and a regeneration reproduces the outcome instead of erasing
it. Blind mappings: `manifest.json:blind_map`. WERs, transcripts, confidences
and the paired bootstrap: recomputed from `results/master.csv` at build time.
No number in this section was copied from a progress log — SPEC C.7 records what
that costs.
