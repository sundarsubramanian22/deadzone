# The 3-minute path through the dashboard

**Open:** `dashboard/deadzone.html`, double-clicked. No server, no terminal, no wifi.
Turn wifi **off** before you start — it is a stronger demo than saying it works offline.

**Before you present, once:**

```bash
python3 dashboard/make_synthetic.py          # only if results/ has no table yet
python3 dashboard/build.py                   # auto-picks results/master.csv if it exists
python3 test_dashboard.py                    # 22 offline checks, ~3 s
```

`build.py` prints a per-panel `ok` / `EMPTY` line. **Read it before you walk into the room.**
Any panel that says `EMPTY` will show a written explanation instead of a chart — that is a
survivable demo, but you want to know which one it is in advance.

The badge top-right says `SYNTHETIC DATA` or `REAL GRID`. Say which one out loud in the first
fifteen seconds; do not let anyone discover it themselves at minute two.

---

## 0:00–0:20 — Frame it (header, no clicking)

> "Streaming ASR gets reported as one aggregate WER. That number hides the thing a deployment
> actually dies of: not that the model is wrong, but that it is wrong **and confident** — because
> confidence is what decides whether the agent commits to the transcript or asks you to repeat.
> This is a controlled rig for finding those conditions. Every input is real — measured room
> impulse responses, recorded noise, recorded speech. Only the assembly is synthetic, which buys
> the one thing field recordings can't give you: you turn one knob and hold everything else still."

Point at the badge: how many rows, how many conditions, which table.

## 0:20–1:10 — Panel 1, the hero. This is the whole project.

> "One point per acoustic condition. X is the model's own mean word confidence. Y is the WER we
> measured. The shaded box is the dead zone: the model is above its own top-40% confidence
> threshold **and** above 30% WER. Those are conditions where it is confidently wrong."

Read the count off the first tile. Then **hover a red-ringed point** and let the right-hand card
do the talking:

> "Here is one of them. rt60 one second, SNR twenty-five decibels — a *quiet* room, just a
> reverberant one. And here is what the transcript actually looks like."

Point at the diff: struck-through words are deletions, red are substitutions.

> "Reverb deletes words. The model's confidence is driven by the noise floor, and the noise floor
> here is fine — so it never notices. That is the failure mode you cannot see in an aggregate."

Then the counter-example, one sentence, so nobody thinks the model is simply bad:

> "The rho at the bottom is negative — globally, confidence *does* track error. The finding isn't
> that the model is blind. It's that there is a specific region where it goes blind."

## 1:10–1:40 — Panel 2, is it a region or a fluke?

> "Same conditions, laid out in the factor space. Fill is WER. Border weight is the
> confidence–accuracy gap, red border means dead zone. The dead zones are not scattered — they are
> the **top-right corner of the reverb axis at good SNR**, in every noise facet. It reproduces
> across noise types, which is what makes it a region and not three unlucky cells."

## 1:40–2:15 — Panel 3, the mechanism, and the "so what"

> "Now stop counting errors and classify them. Each family gets an edit signature: reverb is
> **deletions**, babble is **substitutions** plus insertions, codec is substitutions on entities.
> That matters because the signatures imply different fixes — and they are printed right there.
> Deletions under reverb mean whole words never reached the decoder, so keyword boosting cannot
> help you; you need a dereverberation front-end. Substitutions under babble are exactly what
> boosting *does* fix. The red tick is the entity error rate: entities degrade faster than the
> transcript as a whole, which is the number a voice agent actually cares about."

## 2:15–2:40 — Panel 4, and the discipline point

Scroll to the verdict box and read the status word out loud, *especially* if it says NOT CONFIRMED.

> "We pre-registered reverb-by-noise as a genuine interaction before seeing any data. Here is the
> verdict. It says NOT CONFIRMED — the point estimates go the right way but the CI doesn't clear
> the threshold, which means underpowered, not additive. I'm showing you that with the same
> prominence I'd have shown a confirmation. That's the whole reason for pre-registering."

If the provenance line says the indices came from a GP surrogate, say so in the same breath.

## 2:40–3:00 — Close on panels 5 and 6

Click **step ▶** three or four times.

> "The grid is expensive, so the surrogate chooses the next condition to measure — watch the points
> walk onto the failure boundary instead of spreading out evenly. Below is active versus random
> against the same budget."

Then panel 6, and end on the honest note rather than a claim:

> "And this is the simulator auditing itself: measured RIRs on the x-axis, RT60-matched synthetic
> twins on the y. Level is offset — the simulator is optimistic — but the rank correlation is high,
> so it ranks conditions correctly and should not be quoted for absolute numbers. And what's *not*
> in scope is in the footer: accent, disfluency, and the Lombard effect, because in real noise
> people change how they speak and no room simulator reproduces that. That boundary is where real
> field recordings would earn their keep."

---

## If something goes wrong

| Symptom | What to do |
|---|---|
| A panel shows a dashed box with text | Read it out. It says exactly which payload was missing. Move on — it is a build-state fact, not a crash. |
| The page looks wrong on the projector | Zoom to 150%. Everything is relative units; wide charts scroll inside their own container, the page body never scrolls sideways. |
| Someone asks "is this real data?" | Answer from the badge, immediately. Synthetic means planted structure and the panels are demonstrating that the instrument reads its plant back. |
| The model toggle is greyed | The table had one model arm. Say so; do not click at it. |
| Numbers differ from last week | The footer's build line has the source table and timestamp. Read it rather than guessing. |

## Rebuilding against the real grid

One flag. Nothing else changes — the front end never knows which table it got:

```bash
python3 dashboard/build.py --master results/master.csv
```

`--no-al` skips the active-learning loop (saves ~15 s of build time; the panel then renders its
empty state saying so). `--sobol-n` controls the surrogate Sobol sample size when
`results/sobol.json` is absent.
