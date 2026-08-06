# DEMO_SCRIPT — the listening exercise (~3 minutes)

Presenter's run-of-show. Everything here is offline: play wavs from
`blind/`, read from this file. No network, no API key.

**Hand over:** `blind/` only (8 wavs + `BLIND_SHEET.md`).
**Keep back:** this file, `KEY.md`, `PREREGISTERED_PREDICTION.md`,
`REGENERATION_HAZARD.md` (it names the conditions too).
The working filenames in the parent directory say `reverb` and `babble`; a
listener who sees them has been told the answer.

> ⚠️ **REVISED after the prediction in `PREREGISTERED_PREDICTION.md` was
> tested on 2026-08-05 and FAILED (direction held in 1 of 3 pairs).** This
> segment is **direction-agnostic**: the listener ranks, then learns the
> model has no preference. That works whichever way anyone hears it. The
> earlier version led with the pair that drew the least confident call and
> asserted a precedence-effect / informational-masking mechanism that the
> listening pass **refuted** — both are gone.
>
> **Section 7 changed again on 2026-08-06: the close is a QUESTION, not a
> verdict.** It used to land on a conclusion about what listening can and
> cannot establish about an ASR. One listener, on three pairs selected
> BECAUSE the model tied on them, cannot carry a conclusion — this segment
> is the MOTIVATING HOOK, and it is stronger as one. `demos/demo_listen.py`
> closes the same way; the two must not diverge.
>
> The revision lives in the generator template
> (`scripts/make_demo_audio.py`), not only in this file, so a regeneration
> reproduces it instead of reverting it. The play order and each pair's role
> are derived from the recorded listener calls; they are not hardcoded.

---

## 0. Before they listen (10 s) — show the sealed file, do not say what is in it

Open `PREREGISTERED_PREDICTION.md`, show that it exists, and **leave it closed**.
Saying a prediction out loud before someone judges is a demand characteristic.
This mirrors how the study handles its own hypotheses: `rt60 x snr_db` was
pre-registered in SPEC section 5 before any real audio existed, and confirmed on
the real grid at ST-S1 = 0.128 [0.091, 0.164]. Same discipline, three minutes
instead of three weeks.

**Do not make a new directional prediction on stage.** The file already
carries a tested outcome: it predicted *which* clip a listener would find
harder, and was wrong in 2 of 3 pairs. That failure is recorded under the
sealed text and you are going to show it in section 4. The claim this segment
rests on does not need a direction, so do not stake one.

The only thing worth predicting out loud is the part that replicated:

> "I've written down what I expect to happen — that you'll have a preference
> in each pair. You'll see the file in a minute, including the part of it I
> got wrong."

---

## 1. The task (20 s)

Give them `blind/BLIND_SHEET.md`. Two rules only:

> "Same speaker, same kind of sentence. For each pair: which is harder to
> understand? And use the volume knob however you like — the clips aren't
> level-matched and loudness isn't what I'm asking about."

---

## 2. They listen and rank (60 s)

**Play pair 2 (`u21`) first, then pair 3 (`u26`). Pair 1 (`u40`) is the third if they want one.**

All three carry identical evidential weight — the model scores every pair
**exactly** equal (pair 1: 0.333 / 0.333 · pair 2: 0.222 / 0.222 · pair 3:
0.250 / 0.250). The ordering is about the *listener*, not the data: in the one
session run so far, pair 2 and pair 3 drew confident calls, while pair 1 drew
the least confident one ("both pretty bad ... not 100% eveyeron would agree").
Opening on the pair most likely to produce a hedge wastes the strongest moment
of the segment.

**Do not steer, and do not react.** Let them finish every ranking before you
say anything at all. It does not matter which way they go — section 4 is
written so that any confident ranking lands, and so is the one they might not
make (see Fallbacks).

---

## 3. The reveal (45 s)

Leave `PREREGISTERED_PREDICTION.md` closed for one more minute — it comes out
in section 4. Reveal the conditions first:

> "Those two clips are not the same degradation. One is a bad ROOM but
> quiet: requested RT60 1.0 s at 20.0 dB SNR. That is the Shower impulse
> response — a real measured room — with a measured RT60 of 1.011 s and a
> direct-to-reverberant ratio of -10.02 dB.
>
> The other is a GOOD room with the speech nearly buried: RT60 0.2 s at 0.0
> dB SNR, the Restaurant response, DRR +16.90 dB. Neither one has a codec or
> mic rolloff on it, so nothing else is moving between them."

Then the numbers — read them exactly:

> "On the pair you just heard, the model scored them **identically**. Not
> close — equal. And not equal because it got both right: it got both
> **wrong**, by the same amount, in different places.
>
> Across all 40 clips: the reverb condition means WER **0.1123**, the babble
> condition **0.1301**. The paired difference is **-0.0178**, 95% CI
> **[-0.0654, +0.0310]** — spans zero. That is a 10,000-resample paired
> bootstrap over clips, seed 0. Statistically indistinguishable."

Per-pair WER, whichever ones you played, in play order (reverb arm / babble arm):

| pair | clip | reverb arm | babble arm |
|---|---|---|---|
| 2 | `u21` | **0.222** | **0.222** |
| 3 | `u26` | **0.250** | **0.250** |
| 1 | `u40` | **0.333** | **0.333** |

Show the transcripts if there's a screen — they make the point better than the
scalar does, because equal WER is arrived at by damaging different words:

- `u21` reference: `forward the file to accounting and legal by monday`
  - reverb: `forward the file to accounting and legal filing`
  - babble: `forward the file to accounting and legal`
- `u26` reference: `text me the address for the kowalski wedding`
  - reverb: `text me the address for the cool seaway`
  - babble: `text me the address for the`
- `u40` reference: `ask yamamoto to sign page twelve before we file`
  - reverb: `ask yavamoto to sign page twelve before v five`
  - babble: `ask yamamoto to sign page twelve in world five`

⚠️ **Do not generalize an edit-type signature from these three clips.** Here
the reverb arm happens to substitute and the babble arm to delete, but that is
three clips and it runs **opposite** to the grid-level fingerprint, where
`rt60 >= 0.7` drives **deletions** — see `results/fingerprints.txt`, which is
the measured statement and this is not. Use the transcripts to show *that* the
damage differs, not to claim *how* it differs.

---

## 4. The disagreement, and the prediction I got wrong (40 s)

**This section is deliberately direction-agnostic. It does not matter which
clip they picked.**

> "You had a preference. The model does not — it scores that pair equal, and
> across all the clips the difference is indistinguishable from zero.
>
> Notice what that argument does *not* rest on: which of the two you picked.
> Any confident human ranking of that pair is a ranking the model does not
> share."

Now open `PREREGISTERED_PREDICTION.md` and show the OUTCOME section. Deliver
this rather than skipping it — it is the strongest 20 seconds in the segment:

> "I did pre-register which way I thought a listener would go: that the
> babble-dominated clip would be the harder one. The listener I ran this on
> before you went the *other* way in 2 of the 3 pairs. So the direction is
> not established, and rather than quietly re-score it, the miss is written
> under the sealed text with the verdict on it.
>
> The rubric I wrote had exactly two outcomes — 'they rank them unequal' and
> 'they rank them equal.' It never considered 'unequal, but backwards.'
> Under my own rubric this scores as a **pass**, because they did rank them
> unequal. That is a rubric that cannot fail, and this project is entirely
> about not fooling yourself with a number you wanted.
>
> What survived is the half that never depended on the direction: a stated
> human preference in 3 pairs out of 3 — one of them hedged — and a model
> with none."

### Do NOT repair the story on stage

The tempting move, the moment someone says they found the reverberant clip
harder, is to reach for the DRR number: *"of course — that room is at
-10.02 dB DRR."* **Do not deliver that as the explanation.** It is
post-hoc: the human-side prediction was made in the opposite direction and lost,
DRR is simply the first number to hand that fits the result now in view, and
n=1 cannot adjudicate between the two stories. Presenting it as the mechanism is
exactly the move the failed prediction should have made you distrust.

If you want to raise it at all, raise it labelled:

> "Something I'd want to test — and to be clear, this is a hypothesis I
> formed *after* seeing the result, not a finding — is whether a listener's
> ranking tracks direct-to-reverberant ratio rather than RT60, the way the
> model's errors do. That is a proper listening study, and I haven't run
> it."

### If they ask why you kept a failed prediction in the repo

> "Because a pre-registration whose result isn't recorded is worse than none
> — anyone who finds it later assumes it was never run, or assumes it held.
> It cost me a mechanism I liked and bought a better finding: the flaw was
> in my outcome table, not in the listener."

### The model-side reverb result, which IS measured

Keep this on the model side of the line and it stands on its own — it is a grid
result, not an inference about the listener:

> "The model is not tracking RT60 either. Across the four reverb levels in
> the grid, the Spearman correlation of measured RT60 with WER is **+0.800**
> — but with direct-to-reverberant ratio it is **-1.000**. The pair you just
> heard is the extreme case: Shower at DRR -10.02 dB against Restaurant at
> +16.90 dB. RT60 is the number every reverb benchmark is parameterised by,
> and it mislabels the acoustics that actually get delivered."

---

## 5. The payoff (25 s)

Play `blind_02.wav` (clean control) then `blind_05.wav`.

> "Same sentence, same speaker. The second one is condition
> `rt60-0.7_snr-20_babble_opus-lowrate_roll-1` — and look at the SNR: **20.0
> dB**. That is QUIET. The damage here is reverb plus a low-rate codec plus
> full mic rolloff. It is not noise.
>
> You can still hear that someone is talking. The model returned an **empty
> string**. WER 1.000, all 11 reference words deleted. **10 of the 40 clips
> came back completely empty in this condition.**
>
> And there is no low confidence to catch it with. Mean word confidence is
> not low — it is *null*, because there are no words to be confident about.
> A monitor watching confidence sees nothing at all."

---

## 6. The honest label (20 s) — do not skip this

> "Two halves here, and they are not the same kind of thing.
>
> **The human half is an intuition pump, not a measurement — and here is
> every reason it is not one.** It is **n=1**: one listener, one speaker,
> one accent, one sitting. They were blind to *which clip was which
> condition*, but **not naive to the hypothesis** — they knew what this
> project claims, which is the kind of thing that moves a judgement. The
> clips are **not level-matched** (they are byte-identical to what the model
> was scored on, and a cosmetic gain would break that). **Presentation order
> is not counterbalanced** — with three pairs it cannot be; one pair plays
> the reverb arm first, the rest play it second. And I **selected these
> three clips precisely because the model tied on them**, which is a
> defensible choice for a demonstration and an indefensible one for an
> estimate.
>
> It is also **not replicated in direction.** I pre-registered which way a
> listener would go and got it wrong in 2 of 3 pairs. What repeated was only
> that there *was* a preference, in three pairs out of three — and one of
> those was hedged.
>
> **The measured half is the model-side paired result and its interval:**
> -0.0178 WER, CI [-0.0654, +0.0310], over all 40 clips, 10,000 resamples,
> resampled over clips, seed 0. That half is not selected and not affected
> by anything you just said. It would read the same if you had ranked them
> the other way round — which is what actually happened last time — or
> refused to rank them at all.
>
> Doing the human side properly is a listening study: many listeners naive
> to the hypothesis, randomized and counterbalanced order, level-matched
> stimuli, a real intelligibility score. That is exactly the experiment this
> project does not have, and the limitations section says so."

---

## 7. The close (10 s) — a QUESTION, not a verdict

> "Whichever way you called those pairs — and 'about the same' is a real
> answer — the model reports no difference at all. Not a small one: none.
> Every pair you heard it scored identically, and across all 40 clips the
> two kinds of damage differ by **-0.0178 WER**, an interval that spans
> zero.
>
> Your ears and that number are not measuring the same thing, and I had no
> way to settle which of them to believe by listening harder. 'Sounds fine
> to me' is an opinion and I could not check it — so I built something that
> could measure it instead. Everything after this segment is that
> instrument. This segment is the question it was built for, not one of its
> results."

**Do not upgrade that into a verdict on stage.** This script used to close on
a conclusion about what listening can and cannot establish about an ASR. One
listener, on three pairs selected BECAUSE the model tied on them, cannot carry
a conclusion — and asserting one here would be this project's own signature
failure committed in its own demo. As the motivating hook it is honest and it
is stronger, because the question is the reason there is an instrument
downstream of it.

The measured half of that same disagreement is section 5, and it needs no
ranking from anyone: the payoff clip is a condition where a listener can
plainly hear a person speaking and the model returns an empty string on 10 of
40 clips. Land on that, not on a claim about listening.

`demos/demo_listen.py` closes the same way, and the two must not diverge —
SPEC J.7 is a rehearsal finding a demo script narrating a verdict its own
artifact contradicted. The retracted line and the reasoning behind dropping it
are in `report/_demo_internal_notes.md`.

---

## Fallbacks

- **No speakers / bad room:** skip to section 3 and show the transcripts. The
  measured half needs no audio at all.
- **They rank the reverb clip harder** (i.e. the opposite of the sealed
  prediction — this is what the one listener so far did, in 2 of 3 pairs):
  **nothing changes.** Section 4 does not depend on the direction. Say so,
  then show the recorded outcome: "that's the direction I got wrong, and it's
  written down." Do **not** improvise a DRR explanation on the spot.
- **They rank the pair EQUAL, or say "I can't call it":** that is a real answer,
  not a failed demo — say so. One listener agreeing with the model is worth no
  more than one listener disagreeing with it, which is exactly why the interval
  underneath is the measured half and this half is only the question. The
  measured half is untouched. Then pivot to section 5 — the payoff clip needs no
  ranking at all: a human can obviously still hear speech, and the model returned
  an empty string.
- **They rank confidently but inconsistently across pairs** (one each way): that
  is the honest state of the human evidence and it is fine to say so. The
  model-side claim is per-pair and holds in all three.
- **They ask what you predicted, before ranking:** don't tell them. Say "after
  you've called it" — announcing it first is a demand characteristic, which is
  the reason the file is sealed in the first place.
- **They want to hear the factors separately:** `isolation/` is the ladder,
  `00_RAW_original` to `10_destroyed`, one factor at a time. See
  `WHAT_TO_LISTEN_FOR.md`.

## Provenance

Every number above is read or recomputed from `results/master.csv` at build
time by `scripts/make_demo_audio.py` — none is typed into this file.

The one thing in this document that does **not** come from the master table is
the **play order**, and it cannot: it is a judgement about a listener. It is
derived from the calls recorded in `LISTENER_SESSIONS` in that same script
(session 2026-08-05), so it has an owner and a date rather than being an
ordering nobody can account for. Change the record and this file changes with
it.

The wavs are bit-identical to what nova-3 transcribed: `apply_condition` is
seeded from the condition name and the writer is the grid's own
`write_degraded_wav`. Regenerate with:

    ./.venv/bin/python scripts/make_demo_audio.py --force

**If you hand-edit this file, that edit is safe.** The generator hashes what it
writes and refuses to overwrite anything that no longer matches — it will print
a refusal naming this path and carry on. Only `--force-docs` overrides that, and
it copies the current file to `<name>.superseded-<UTC>.md` before it does.
