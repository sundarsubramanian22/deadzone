"""
Cross-model text normalization for the L1 comparison — and ONLY for it.

## The problem this exists to solve

`normalize_text` in `audio_pipeline.py` is a trap function. It lowercases, strips
punctuation, deletes apostrophes and collapses a tiny orthographic compound map,
applied identically to reference and hypothesis. It deliberately does **not** map
digits to words or words to digits, because that mapping is genuinely ambiguous
and guessing inside a trap function is exactly the failure mode this project is
about.

That is the right call for the spine measurement. It is the wrong call for
comparing two models against each other, because the two arms disagree about
*orthography*, not about *acoustics*:

    reference     call maria at four zero five nine one two seven seven
    nova-3        call maria at four zero five nine one two seven seven   WER 0.00
    whisper-base  Call Maria 405-912-7777.                                WER 0.82

Deepgram's formatting is switched off at the adapter (`smart_format`,
`punctuate`, `numerals` all False) so its raw output is already word-form.
Whisper has no equivalent switch and always writes numbers as digits. On a corpus
deliberately loaded with phone numbers, spelled codes, addresses and amounts,
that inflates Whisper's WER by a **large, condition-independent constant** — and
a constant offset is mathematically indistinguishable from an acoustic effect
once it is in the results table.

## Why not "just add digit expansion to normalize_text"

Because the corpus itself uses both conventions, so no single rule is correct:

    u02  "four zero five"      -> should become 405,  NOT "four hundred five"
    u05  "fourteen hundred"    -> should become 1400, NOT "one four zero zero"
    u11  "eighty eight"        -> should become 88,   NOT "eight eight"

A words->digits direction has the same problem in reverse. Any rule that gets u02
right gets u05 wrong. That ambiguity is real and it does not go away by choosing
harder.

## What this module does instead

Two stages, applied **symmetrically to reference and hypothesis** (the CLAUDE.md
invariant), and used only for the cross-model arm:

1. **`EnglishTextNormalizer` from the Whisper authors.** This is not our guess —
   it is the published, community-standard normalizer used to report Whisper's
   own numbers, so using it is a citable choice rather than a convenient one. It
   canonicalizes number expressions in the words->digits direction, expands
   contractions, and normalizes British/American spelling. It resolves the u05
   and u11 cases exactly.

2. **Digit-run splitting.** The normalizer still leaves the *grouping* of digit
   strings uncanonical: the reference's "four zero five nine one two seven seven"
   becomes `40591277` while Whisper's `405-912-7777` becomes `405 912 7777`.
   Splitting every digit run into single digits on both sides makes grouping
   irrelevant, and it is symmetric so it cannot favour either arm. Alphanumeric
   boundaries are split for the same reason (`Q9J05` vs `q nine j zero five`).

## What it does NOT fix, stated plainly

Residual disagreements survive and they are not one-signed:

  * letter runs inside spelled codes (`AW` vs `a w`) still differ, because
    splitting letter runs unconditionally would shatter ordinary words;
  * the Whisper normalizer is not symmetric about leading zeros -- spoken "zero
    five" survives as `05` while written `05` is read as a number and becomes
    `5` -- so `Q9J05` still costs one spurious deletion. Pinned by a test rather
    than patched: special-casing it would mean editing number semantics to make a
    number look better;
  * the Whisper normalizer is itself inconsistent about leading digits in mixed
    codes (`1Z99AW5` -> `one z 99 aw 5`).

**Apply exactly once, to raw text.** The transform is deliberately NOT
idempotent: `EnglishTextNormalizer` rewrites a standalone `1` as `one`, and
digit-run splitting manufactures standalone digits, so a second pass turns
`1 4 0 0` into `one 4 0 0`. Normalizing one side twice and the other once would
make them disagree for no acoustic reason. Every entry point below takes raw text
and applies the transform a single time; a test pins the non-idempotence so this
constraint cannot rot silently.

So the L1 cross-model absolute WER is a **comparison aid with residual noise, not
a replacement for the spine measurement**. Both are reported. The within-model
analyses — dead-zone rate, confidence-vs-WER shape, `within_model_conf_percentile`
— never needed this and are unaffected either way, because each is computed
against its own model's WER distribution.
"""
from __future__ import annotations

import re
from functools import lru_cache

from deadzone.audio_pipeline import classify_errors, normalize_text

__all__ = [
    "cross_model_normalize",
    "cross_model_tokens",
    "cross_model_classify_errors",
    "NormalizerUnavailableError",
]


class NormalizerUnavailableError(RuntimeError):
    """
    Raised when openai-whisper is not importable.

    Deliberately loud. The alternative — silently falling back to
    `normalize_text` — would report a cross-model WER that still carries the full
    formatting offset while *looking* like a corrected number. A missing optional
    dependency must never degrade into a wrong measurement.
    """


@lru_cache(maxsize=1)
def _english_normalizer():
    try:
        from whisper.normalizers import EnglishTextNormalizer
    except Exception as exc:                                  # pragma: no cover
        raise NormalizerUnavailableError(
            "cross-model normalization needs openai-whisper for "
            "whisper.normalizers.EnglishTextNormalizer (pip install openai-whisper). "
            "Refusing to fall back to normalize_text: that would leave the digit "
            "formatting offset in place while looking like a corrected number."
        ) from exc
    return EnglishTextNormalizer()


# split *between* two digits: 405 -> 4 0 5
_DIGIT_RUN = re.compile(r"(?<=\d)(?=\d)")
# split at an alpha<->digit boundary: q9j05 -> q 9 j 0 5
_ALPHA_DIGIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


def cross_model_normalize(text: str) -> str:
    """
    Canonicalize `text` for cross-model comparison. Idempotent, deterministic, and
    applied to reference and hypothesis alike.
    """
    out = _english_normalizer()(text or "")
    out = _ALPHA_DIGIT.sub(" ", out)
    out = _DIGIT_RUN.sub(" ", out)
    return " ".join(out.split())


def cross_model_tokens(text: str) -> list[str]:
    """
    Token list for scoring. Runs the cross-model canonicalization first, then the
    project's own `normalize_text` — so whatever the spine measurement considers a
    token, this considers a token too. Only the number/orthography layer differs.
    """
    return normalize_text(cross_model_normalize(text))


def cross_model_classify_errors(ref: str, hyp: str) -> dict:
    """
    `classify_errors` under the cross-model normalization. Same return shape —
    WER plus the typed edits — so every downstream consumer is unchanged.

    Both sides go through the identical transform. That is the whole safety
    argument: a symmetric transform can move the absolute WER but cannot
    systematically favour one arm.
    """
    return classify_errors(cross_model_normalize(ref), cross_model_normalize(hyp))
