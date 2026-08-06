# Internal notes for the demo kit — NOT FOR THE SCREEN

Stage directions, hazards and things-not-to-say. Everything in this file used to
print into the terminal, or was about to. It does not any more: on a screen-share
the audience reads the instructions being given about them, which is both
distracting and slightly insulting.

**This file is a handoff, not a deliverable.** It quotes no figures on purpose —
where a note depends on a number, it names the artifact the number lives in
instead, so there is no second copy of it to drift (SPEC C.7: the log is a
summary, not a source). Do not quote it in the write-up; fold anything worth
keeping into the internal doc and cite the artifact.

`tests/test_demo_listen.py::NoPresenterNotesOnScreen` asserts that none of the
phrasings below reach stdout, and proves the matcher is real by finding them
**here**. So: if you move a note back into the script, that test fails, which is
the intended behaviour and not a broken test.

---

## From `demos/demo_listen.py` (moved out 2026-08-06)

### 1. `reveal_lines()` — printed once per pair, after every reveal

> (presenter note: three clips — do not read an edit-type signature off them.
> At grid level rt60 >= 0.7 drives DELETIONS, which runs opposite to what these
> three happen to show.)

**Why it existed.** The three demo pairs happen to show substitutions where the
grid-level fingerprint for high reverberation is deletions. Somebody reading a
mechanism off three clips on stage would be reading noise, and would be
contradicted by the project's own D2 layer.

**What to do with it instead of printing it.** If an interviewer asks "so
reverberation causes substitutions?", the answer is *no, and these three clips
are not the evidence either way* — the per-factor edit-type signatures are in
`results/fingerprints.json` / the D2 section of `report/writeup.md`, computed
over the full grid rather than three hand-picked pairs. Say it out loud if it
comes up; do not pre-empt it on screen.

### 2. `prediction_lines()` — printed after the failed-pre-registration section

> (presenter note: do NOT repair this on stage with the DRR number. It fits the
> result now in view and it is post-hoc. The model-side DRR result — [two rank
> correlations across the grid] — is measured and stands on its own.)

**Why it existed.** When a listener finds the reverberant clip harder, there is
an extremely tempting mechanism to reach for, and it fits. It was thought of
after the result was seen, one listener cannot adjudicate it, and reaching for
it is precisely the move the failed pre-registration should make you distrust.

**Two further reasons to leave it alone**, both worth knowing before somebody
raises them:

- The note *named the mechanism it was telling you not to reach for*, on screen,
  which handed the room the post-hoc story anyway. A stage direction that leaks
  its own subject is worse than no stage direction.
- The model-side version of that result is real and measured, but see
  `report/UNDERSTANDING.md` §4.4 before leaning on it: it rests on four rooms,
  carries no interval, and the project's own published table contains a column
  that undercuts the separation. It is not a rescue for an n=1 human result; it
  is a separate claim with its own exposure.

The script now says, in the first person and to the room: *there is a tidy
mechanism I could hand you, and I am not going to.* That is the same content
without the aside, and it is stronger said aloud than printed.

---

## Other things on screen that are not stage directions but are worth knowing

None of these were moved — they are correctness signals, they fire only in
states that should not occur live, and a presenter needs to see them. Listed so
nobody "cleans them up" thinking they are asides.

- **`(these two are not equal — this demo set has drifted from the grid table)`**
  — fires when a demo pair is no longer an exact tie in `results/master.csv`.
  If this appears on stage the segment's premise is gone; stop and say so.
- **`⚠ manifest and master.csv disagree on the paired difference …`** — the demo
  set and the grid have drifted apart. Same response.
- **`(results/master.csv not present — the tie count is the one number this run
  cannot recompute)`** — benign, but it means the exact-tie count is missing.
- **`(stdin is not a terminal: …)`** — only ever appears in a piped run. On a
  live screen-share stdin *is* a tty, so an interviewer never sees this line.

## Judgement calls left for a human

- **Naming the conversation that motivated the project.** The closing describes
  it without naming anyone, deliberately: it is framed as *what made me want to
  build this*, never as a claim about what any company has or has not studied.
  Naming the person is the presenter's call to make out loud, in the room, where
  tone is available — not the terminal's.
- **Other surfaces still carry the old finding-voice line.** `demos/demo_listen.py`
  no longer closes on "you cannot QA a voice agent by listening to it" — it
  closes on the question instead. `report/INTERVIEW_RUNBOOK.md` and the generated
  `results/audio/demo/DEMO_SCRIPT.md` (written by `scripts/make_demo_audio.py`)
  still state it as a conclusion. Reconcile before anyone rehearses off both:
  SPEC J.7 is exactly this failure, found by a rehearsal rather than a test.
