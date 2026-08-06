# The demo — rehearsed running order and timings

Four pieces, ~7 minutes total. One command each; nothing is typed live.

| # | Piece | Command | Rehearsed | Network? |
|---|---|---|---|---|
| 1 | Opener — the trap functions are green | `make test-core` | **30 s** | offline |
| 2 | **The hero** — one clip, played and transcribed LIVE, twice | `make demo` | **2 min** | live, falls back |
| 3 | The active-learning loop | `make demo-al` | **30 s** | offline |
| 4 | The dashboard, the path below | `make dashboard` | **4.5 min** | offline |

`make demo-all` runs all four in that order.

**Piece 2 is the only one that touches the network, and it is safe.** The
interviewer picks a clip from a menu of measured dead zones (or takes a random
one); the raw recording plays and is transcribed live; then the degraded version
plays and is transcribed live; the punchline is computed from the payload that
just arrived, and the archived grid row is shown afterwards as a reproduction
check. No key, no network, a vendor error or a timeout each print **one** line,
fall back to the archived measurements for that same clip and condition, and
**exit 0**.

**Rehearse it with wifi off using `make demo-replay`** — the identical beat from
cache. That is also the instant fallback if the live path misbehaves on the day.
`make demo-break` (audio + cached numbers, no calls) and `make demo-live` (two
calls, no audio) are the two halves on their own, kept working as fallbacks.

There is **no live voice agent** in this demo. SPEC R7 was scoped out of this push
deliberately; `agent_eval.py` exists as a synthetic-validated scaffold and that is
exactly what it is presented as. Do not promise a live agent in the room.

**Preflight, before you turn wifi off:**

```bash
make demo-check     # every artifact the demo needs, verified on disk
make test           # every offline suite
```

`make demo-check` prints `READY — you can turn wifi off.` or names the missing file
and the target that rebuilds it.

---

# The scripted path through the dashboard

**Open:** `make dashboard`, or double-click `dashboard/deadzone.html`. No server, no
terminal, no wifi. Turn wifi **off** before you start — it is a stronger demo than
saying it works offline.

**If the dashboard needs rebuilding first (not needed on stage):**

```bash
make dashboard-build     # ~35 s; regenerates from results/master.csv — NO extra flags
./.venv/bin/python tests/test_dashboard.py    # offline render checks
```

`build.py` prints a per-panel `ok` / `EMPTY` line. **Read it before you walk into the
room.** On the current grid the expected line is:

```
[elevenlabs-scribe] silent_failure=ok, fingerprints=ok, sensitivity=EMPTY, active_learning=ok, sim2real=EMPTY
[nova-3]            silent_failure=ok, fingerprints=ok, sensitivity=ok,    active_learning=ok, sim2real=ok
[whisper-base]      silent_failure=ok, fingerprints=ok, sensitivity=EMPTY, active_learning=ok, sim2real=EMPTY
[cross-model]       model_arms=ok, decoupling=ok
```

Those four `EMPTY`s are **correct and explained on screen** — see the troubleshooting
table. Anything else that says `EMPTY` is news, and you want it to be news before you
are standing up.

The badge top-right reads `real grid · 12320 rows · 176 conditions`, or `synthetic
data` if someone rebuilt against the planted table. Say which one out loud in the
first fifteen seconds; do not let anyone discover it themselves at minute two.

The page opens on the **nova-3** arm. Panels 1–6 follow the arm toggle; panel 4 is a
single-arm result and blanks on the others; panels 7 and 8 sit outside the toggle.

---

**Timing, measured rather than asserted.** Read **only the block-quoted lines aloud**:
608 words, **~4:03 at a normal 150 wpm**, ~4:33 with the pointing and the two clicks. The
headings below carry the measured splits, not aspirational ones. Everything that did not fit
in a quote sits under it as a *Note* — for questions, and for the days you have room.

**If the slot really is 3 minutes**, cut in this order and stop when you fit: **panel 2**
(−32 s; panel 1 already made the point) and then **panel 5, the AL null** (−20 s; it is the
one finding that costs nothing to state later). That leaves frame + hero + fingerprints +
pre-registration + sim2real ≈ **3:00**. Do not cut panel 4 to save time — the
pre-registration verdict is the discipline highlight, and it is 36 seconds.

## 0:00–0:33 — Frame it (header, no clicking)

> "ASR is reported as one aggregate WER. That hides what a deployment dies of: not
> that the model is wrong, but wrong **and confident** — confidence decides whether the agent
> commits or asks you to repeat. Every input here is real; only the assembly is synthetic.
> That buys one knob at a time, everything else held still. And say the limitation before
> anyone asks: **every arm ran batch, not streaming.** This is acoustic robustness, not a
> live-agent map."

Point at the badge and read it: real grid, 12,320 rows, 176 conditions, three arms.

## 0:33–1:38 — Panel 1, the hero. This is the whole project.

**The readout is already loaded on the worst dead zone — you do not need to hover to get the
headline.** Hover later, to show it is live.

> "One point per condition. X is the model's own mean word confidence; Y is WER on the clips
> it actually spoke on — same clips on both axes, so they subtract. The shaded box is the
> dead zone.
>
> It opens on the worst one: reverb 0.45 seconds, SNR **zero** decibels, engine noise, a
> G.726 phone codec. Nova-3 reports **0.83** confidence at WER **0.31** — and **zero of forty
> clips came back empty**, so none of that gap is silence.
>
> The diff: 'eighty eight **elm street**' came back as 'three eight **l three**'. The frame
> survives, the address is destroyed, at 0.83 confidence.
>
> Counterweight: rho is **−0.98**. Globally it *does* know when it is failing. Only **2 of
> 176** conditions are confidently wrong; **95** are loudly wrong. The risk isn't blindness —
> it's that a system calibrated on that average trusts it in exactly those two places."

*Notes.* The two tiles below the plot: **4** conditions look dangerous only because clips
vanished, and **7** returned nothing on any clip — a confidence-based monitor is structurally
blind to those, because there is no confidence to be low. Confidence is the **64th
percentile** within nova-3, over 40 clips and 363 reference words.

## 1:38–2:13 — Panel 2, where in factor space?   *(first to cut)*

> "The same conditions in factor space, one per cell — the facets split noise, codec and
> rolloff, so nothing is averaged away. Solid red is a dead zone; dashed red is a cell where
> nothing came back at all.
>
> They are not where you'd guess: **bottom edge, worst SNR, at the *low* end of the reverb
> axis**, in two of three noise types. Where reverb and noise are both extreme the model
> doesn't go quietly wrong — it goes **silent**."

*Notes.* Overconfidence is systematic (91% of conditions report more confidence than
accuracy); confidently-*wrong* is the rare part. The engine and road facets are smaller
because those arms ran a reduced design — two rt60 × two SNR levels. That is the experiment,
not a missing panel.

## 2:13–2:58 — Panel 3, the mechanism, and the "so what"

> "Stop counting errors and classify them. Every family here has the same dominant signature:
> **deletions**. Falling SNR takes them to 0.53 of reference words, rolloff 0.51, reverb 0.46
> — babble too. The exception is **G.726**, which substitutes: it doesn't drop the word, it
> hands you a different one.
>
> That picks the fix, and the fix is printed on each row — a deleted word never reached the
> decoder, so boosting can't recover it; you need a front end. The red tick is entity error
> rate: **0.63 against an overall WER of 0.51**. Entities degrade faster than the transcript,
> and entities are what the agent is for."

## 2:58–3:38 — Panel 4, and the discipline point

Read the grey **SINGLE-ARM RESULT** banner first, then the green verdict box.

> "We pre-registered reverb-by-noise as a genuine interaction before any audio existed.
> Verdict: **CONFIRMED** — ST-minus-S1 is 0.128 for reverb, 0.112 for noise, both intervals
> clear of a threshold fixed in advance.
>
> And the banner: this panel is **nova-3 only** and does not move with the arm toggle. It
> needs a complete factorial, 40 clips in every cell, and only the spine arm ran that. On the
> other arms it is blank and says why — serving these numbers under another arm's label is
> exactly the mistake this project is about."

*Notes.* At the bottom, six counterintuitive cells the surrogate proposed that real
transcription **did not reproduce** — shown as unconfirmed rather than quietly dropped. If
the verdict ever reads NOT CONFIRMED, read it with the *same* emphasis; the panel gives both
statuses equal prominence on purpose.

## 3:38–4:33 — Close on panels 5 through 8   *(panel 5 is second to cut)*

Click **step ▶** three or four times on panel 5, then deliver the null in one breath.

> "A surrogate picks the next condition to measure — watch the points walk onto the boundary.
> The result is a **null**: the target was hit by one of three active seeds and two of three
> random. I report the budget, not a ratio."

Then panel 6, and end on the honest note rather than a claim:

> "The simulator auditing itself: **12.1 points optimistic**, rank correlation **0.87** — use
> it to *order* conditions, never to quote a number. Dead-zone overlap is **zero**, and panel
> 7 says the same across model families: you cannot borrow someone else's dead-zone map.
> What's out of scope is in the footer — accent, disfluency, the Lombard effect. That is where
> real field recordings would earn their keep."

*Notes.* There are **eight** panels: hero, factor-space grid, fingerprints, sensitivity,
active learning, sim2real, multi-model comparison, paralinguistic decoupling. Panels 7 and 8
sit **outside** the arm toggle — panel 7 *is* the comparison, and panel 8 reads audio rather
than the results table. On the AL null: the machinery *does* beat random on planted synthetic
structure (that test is in the suite), so this is a method meeting a surface it has no
purchase on, not a broken implementation — and all three seeds ran against the surrogate
oracle, none against the live API.

**Spare material, in the order to reach for it:** panel 7 (three arms; Whisper turning 3
reference words into 49 is on screen with the transcript), then the AL provenance line, then
panel 8.

## Things you must NOT say

| Don't say | Because |
|---|---|
| "The dead zone is a quiet room / good SNR" | Both dead zones are at **SNR 0 dB**, the worst level in the grid. There is no 25 dB level; the SNR levels are 0, 5, 10, 20. |
| "The dead zones are in every noise facet" | Two of three (engine and babble). Two conditions total, of 176. |
| "Babble causes substitutions" | Babble's dominant edit is **deletions**, like every other family here. **G.726** is the substitution one. |
| "It says NOT CONFIRMED" | It says **CONFIRMED**, in green. |
| "Panel 4 is this arm's sensitivity" | It is **nova-3's**, always, and it blanks on the other arms. |
| "Scribe has the best WER of the three" | `elevenlabs-scribe` is marked ‡ and is **excluded from cross-arm WER**: its orthography is non-deterministic across identical calls, so its offset is a per-call draw, not a constant that can be subtracted. Its dead-zone rate and confidence shape *are* comparable — they are computed within the arm. Rank it on those, never on WER. |
| "Active learning saved us N calls" | It is a **null**. Report the budget. |
| "This is a streaming-ASR study" / "we measured a streaming model" | **Nothing was streamed.** All three arms are batch: Deepgram's pre-recorded endpoint (never `listen.live`), ElevenLabs' batch REST endpoint (never `scribe_v2_realtime`), Whisper locally with full-file lookahead. The commercial arms are here for their **per-word confidence**, not for their streaming mode. Say it first, in the frame — it is limitation 17 in the write-up and it is stronger volunteered than extracted. |
| "Panel 6's gap is 19.9 points" | That number was a corpus-difference artifact and is dead. The clip-matched gap is **12.1**. |

## If something goes wrong

| Symptom | What to do |
|---|---|
| A panel shows a dashed box with text | Read it out. It says exactly which payload was missing. Move on — it is a build-state fact, not a crash. |
| The page looks wrong on the projector | Zoom to 150%. Everything is relative units; wide charts scroll inside their own container, the page body never scrolls sideways. |
| Someone asks "is this real data?" | Answer from the badge, immediately. Synthetic means planted structure and the panels are demonstrating that the instrument reads its own plant back. |
| The model toggle is greyed | The table had one arm. Say so; do not click at it. |
| Switching arms blanks **panel 4, sensitivity** | Expected, and it explains itself on screen. The exact functional-ANOVA decomposition needs the complete 4×4×3×3 factorial with 40 clips in every cell — 5,760 transcriptions — and only `nova-3` ran it; the other arms ran a 10-clip subset, which is not a complete factorial and has no exact decomposition. Showing nova-3's indices there would be nova-3's numbers under another arm's label. Say "the decomposition is nova-3 only, by design" and switch back. |
| Switching arms blanks **panel 6, sim2real** | Also expected, also explained on screen. The simulated-RIR arm was only ever run for `nova-3`, because D4 compares **measured-vs-synthetic RIR provenance**, not model families; running it per arm would double the API spend to answer a question nobody asked. |
| Panels 7–8 don't change with the toggle | Correct — multi-model comparison and paralinguistic decoupling are cross-model by construction. |
| Someone asks why Whisper's WER is above 1.0 | Insertions are unbounded. It hallucinates fluent text under heavy degradation; the 3-words-to-49 example is on panel 7. WER caps damage at one error per reference word, so it *understates* that failure. |
| Numbers differ from last week | The footer's build line has the source table and timestamp. Read it rather than guessing. |

## Rebuilding against the real grid

```bash
make dashboard-build        # == ./.venv/bin/python dashboard/build.py --master results/master.csv
```

**Build the committed page with no extra flags.** `--no-al` skips the active-learning loop
and leaves panel 5 in its empty state, which fails `tests/test_dashboard.py`'s
committed-artifact check — it is for iterating, never for the artifact you demo.
`--sobol-n` controls the surrogate Sobol sample size, and only matters when
`results/sobol.json` is absent.
