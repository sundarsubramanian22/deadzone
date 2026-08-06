# 🛑 REGENERATION HAZARD — read before running `scripts/make_demo_audio.py`

> ## ✅ CLOSED 2026-08-06 — step 2 of "What to do" is DONE, and the kit was rebuilt from the templates
>
> **Every hand-written block named in the table below now lives in the generator
> template**, not only in the file it was typed into — the OUTCOME section and
> its verbatim listener response are regenerated from `LISTENER_SESSIONS` in
> `scripts/make_demo_audio.py`, and the play order and each pair's role are
> derived from that same record rather than hardcoded. So the documents in this
> directory are once again the generator's own output, `generated_docs.json`
> records their hashes, and **a plain `--force` rebuild reproduces them instead
> of reverting them.** That is the "real fix" step 2 asks for; it is no longer
> outstanding.
>
> The rebuild that closed it was a `--force-docs` run whose five backups were
> each verified byte-identical to `git show HEAD:results/audio/demo/<name>.md`
> before being deleted. **The pre-rebuild text of every document in this
> directory is recoverable from git**, e.g.
> `git show 'ff1eb28:results/audio/demo/DEMO_SCRIPT.md'` — these files are
> force-added to the repo (`c082b04`) even though `results/` is gitignored,
> precisely so that a record with no other home has one.
>
> **What changed in the prose, and why:** `DEMO_SCRIPT.md` section 7 closed on
> *a conclusion about what listening can and cannot establish about an ASR*.
> One listener, judging a three-clip set selected **because** the model tied on
> it, cannot carry a conclusion — so the segment is now framed as the
> **motivating hook**, and section 7 closes on the question instead.
> `demos/demo_listen.py` was changed the same way in `ff1eb28` and the two must
> not diverge again.
>
> **Two pairs is the beat; the third is a reserve.** That number is now derived
> rather than written — `n_primary()` counts the pairs the recorded listener
> called confidently, sections 2 and 6 and `KEY.md` all say it out loud, and
> `tests/test_make_demo_audio.py` pins it to `demos/demo_listen.py`'s
> `DEFAULT_N_PAIRS`. The clip **set** is three and the beat **plays** two; those
> are different numbers and a reader will conflate them unless every sentence
> says which one it means.
> The retracted line and the reasoning are in `report/_demo_internal_notes.md`.
> The `blind/BLIND_SHEET.md` residual closed with it: its rows had been left in
> pair order (1, 2, 3) while the derived play order is 2, 3, 1, and they are now
> emitted in play order with the pair labels unchanged — the labels are the join
> key to `KEY.md`, `DEMO_SCRIPT.md` and the sealed prediction, so renumbering
> them would break every cross-reference.
>
> **What this does NOT change:** everything in the section titled *"The one-line
> fact that must never be lost again"* still stands and is still duplicated
> here on purpose, and the two post-regeneration checks at the end of *"What to
> do"* are still the checks — they are pinned by
> `tests/test_make_demo_audio.py::TheAuthoredRecord::test_the_outcome_record_is_on_disk`,
> which hardcodes those exact strings.
>
> One correction to the banner below: `scripts/make_audio_sets.py` **no longer**
> rewrites `results/audio/listen/WHAT_TO_LISTEN_FOR.md` unconditionally. It grew
> the same content-hash guard in `278c282` (SPEC J.2).

> ## ✅ FIXED 2026-08-06 (`aea58e2`) — the guard described below is now IMPLEMENTED
>
> `make_demo_audio.py` now **refuses** to overwrite an authored document whose
> content hash does not match what it last wrote, and the "what to do" checklist
> further down is enforced by `tests/test_make_demo_audio.py` (16 tests,
> mutation-checked twice) rather than left to whoever remembers to read this.
> `--force` no longer unlocks documents; `--force-docs` does, and it copies each
> file to `<name>.superseded-<UTC>.md` first.
>
> **Two things this file got right and one it could not have known.**
> Right: the hazard was real, and the record it protects derives from no
> artifact. Right: naming a file the generator does not know about is what let
> the warning survive to be acted on. Could not have known: **the trigger was
> `tests/test_demo.py:398`, which invoked the generator with `--force` against
> the live tree on every `make test`.** A passing test suite was the delivery
> mechanism — so "just don't run the generator" was never sufficient advice.
>
> Kept, not deleted. The hazard is fixed for *this* generator;
> `scripts/make_audio_sets.py` still rewrites
> `results/audio/listen/WHAT_TO_LISTEN_FOR.md` unconditionally, and that is
> listening instructions sitting in the directory where a listening pass
> happens — the likeliest file in the repo for a human to annotate. The reading
> below is the argument for fixing that one too.

**Written 2026-08-05. This file is NOT generated** — `make_demo_audio.py` does
not know it exists, which is the only reason it can be trusted to survive.

## The one-line fact that must never be lost again

> **The pre-registered listening prediction was TESTED on 2026-08-05 and
> FAILED: the predicted direction held in 1 of 3 pairs.** The listener found
> the *reverb* arm harder in two of three pairs; the prediction said the
> *babble* arm would be harder in both named pairs. What survives is only the
> direction-agnostic half — a confident human preference in 3 of 3 pairs
> against a model that scores every pair exactly equal.

The full record is the **OUTCOME** section of `PREREGISTERED_PREDICTION.md`.
The one-liner is duplicated here **on purpose**: it is a scalar verdict with no
moving parts, so it cannot drift, and if the generated file is ever wiped the
*result* still exists on disk. Nothing else is duplicated — every number lives
in exactly one place.

## The hazard

`scripts/make_demo_audio.py` **writes these four documents unconditionally** on
any build (`--force`, or any run with the manifest absent). There is no
merge, no skip-if-modified, no warning:

*(State as of 2026-08-05, kept as the record of what was at risk. Every "YES"
below is now reproduced by the template — see the CLOSED banner at the top — and
the line numbers have moved since.)*

| file | written at | contained hand-written content? |
|---|---|---|
| `PREREGISTERED_PREDICTION.md` | `write_prediction()`, writes at `:468` | **YES — the entire OUTCOME section + the status banner** |
| `DEMO_SCRIPT.md` | `write_demo_script()`, writes at `:824` | **YES — sections 0, 2, 3's per-pair table, 4, 6, Fallbacks, Provenance note** |
| `KEY.md` | `write_key()`, `:472` | no |
| `WHAT_TO_LISTEN_FOR.md` | `write_what_to_listen_for()`, `:828` | pointer banner in Part 2 only |
| `blind/BLIND_SHEET.md` | `write_blind_sheet()`, `:581` | no |

This already happened once: a regeneration at 23:10:59 on 2026-08-05 silently
erased a status banner written minutes earlier. It was noticed only because the
editing tool reported the file had changed underneath it.

**This is the project's signature failure mode aimed at its own demo kit** — a
plausible-looking, freshly-generated file, no error, no warning, and the one
thing that took a human listener to produce is the thing that disappears.
Compare SPEC C.4, C.8 and Appendix G: three prior instances, all silent.

## What to do

**Before regenerating:**

1. `git diff -- results/audio/demo/` (or, if `results/` is gitignored on your
   checkout, copy the two files aside) so the hand-written blocks are
   recoverable.
2. Better: **port the hand-written content into the generator templates** —
   `write_prediction()` (line 432, writes at 468) and `write_demo_script()`
   (line 614, writes at 824) — so a regeneration reproduces it. That is the
   real fix and it is in `scripts/`, which the demo directory does not own.
3. Best: make the generator **refuse to overwrite** a doc whose content differs
   from what it would write, unless `--force-docs` is passed — the same
   report-and-refuse discipline `write_master` uses in `run_experiment.py`.

**After regenerating,** check that both files still contain:

- `PREREGISTERED_PREDICTION.md` → the string `OUTCOME — tested 2026-08-05` and
  the verdict `FAILED on direction`.
- `DEMO_SCRIPT.md` → section 4 titled *"The disagreement, and the prediction I
  got wrong"*, and **no** occurrence of the refuted mechanism text
  (`precedence effect` / `informational masking` presented as the explanation).

If either is missing, the outcome record has been destroyed — restore it before
the file is shown to anyone, because a `PREREGISTERED_PREDICTION.md` with no
outcome reads as an *open* prediction, and the next reader will assume it was
never run or, worse, that it held.

## Why the demo script changed

The prior version had three defects, all downstream of the failed prediction:

1. It named pairs 1 and 2 as the primary evidence. **Pair 1 is the pair that
   drew the listener's least confident call** ("both pretty bad"), so the
   script opened on its weakest moment. Now: pair 2, then pair 3, then pair 1.
2. Its section 4 asserted a **precedence-effect / informational-masking**
   mechanism — the explanation the listening pass *refuted*. Now: a
   direction-agnostic reveal (they rank; the model has no preference), plus
   explicit instructions not to substitute a post-hoc DRR story.
3. Its fallback list handled only *"they ranked them equal."* It had no branch
   for *"they ranked them backwards"* — the same blind spot as the
   pre-registration's own two-outcome rubric. Now covered.

The script's **numbers were correct throughout** and are unchanged.
