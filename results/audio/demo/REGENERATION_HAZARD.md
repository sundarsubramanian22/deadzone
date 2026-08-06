# 🛑 REGENERATION HAZARD — read before running `scripts/make_demo_audio.py`

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

| file | written at | contains hand-written content? |
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
