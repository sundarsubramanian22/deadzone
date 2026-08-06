"""
analysis/fingerprints.py — D2, MECHANISM: typed failure fingerprints (SPEC §5.2,
A.R5.3).

D1 says *where* the model fails silently. It cannot say *what kind* of failure it
is, and "what kind" is what determines the fix. A WER of 0.30 made of deletions and
a WER of 0.30 made of entity substitutions are two different products breaking:

    deletions               words are vanishing              -> dereverberation
    entity substitutions    the NAME comes back wrong        -> keyword boosting
    insertions under babble the model transcribed SOMEONE ELSE -> speaker gating

So we do not count errors, we CLASSIFY them, from the typed edits `classify_errors`
was built to preserve (audio_pipeline TRAP 3). Every signature is paired with the
intervention it implies — §5.2 promises actionable guidance and this is where it
lands.

TWO RULES THAT KEEP THE FINGERPRINTS HONEST.

1. A level that is BETTER than its siblings gets NO PRESCRIPTION. Contrasting each
   categorical level against the others means some levels come out cleaner (engine
   noise causing fewer substitutions than babble). That contrast is worth printing;
   prescribing "keyword boosting" for the cleanest cell in the grid is not. Only a
   positive delta on the dominant edit type earns a fix (`degrades`).

2. Insertions under babble are COMPETING-SPEECH CAPTURE, not acoustic confusion
   (SPEC A.R5.3's named gotcha): babble contains speech and the model transcribes
   the background speakers. Different mechanism, different fix (target-speaker
   extraction, not denoising). `classify_insertions` splits every insertion into
   endogenous (the token is a word of this clip's own reference -> confusion) and
   foreign (it is not -> competing speech). The rule is a crude heuristic, so
   example transcripts travel with the numbers.

Deps: numpy (+ agent_eval/audio_pipeline for normalization parity). No audio, no
API. `task_specs.json` and `recording_manifest.csv` are OPTIONAL — absence degrades
the affected section to a stated "unavailable", never to silent zeros.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import string
import sys
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

# Allow `python deadzone/analysis/fingerprints.py` as well as `import deadzone.analysis.fingerprints`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deadzone.analysis import (                       # noqa: E402  (after the path shim)
    as_float, check_unique_cells, coerce_row, failure_summary,
    load_master_table, parse_edits, split_by_model, split_failures,
)
from deadzone.audio_pipeline import normalize_text    # noqa: E402
from deadzone.design import DEFAULT_FACTOR_SPACE, Factor, FactorSpace  # noqa: E402

EDIT_TYPES = ("sub", "del", "ins")
DESTROY_OPS = ("sub", "del")                 # a wrong name and a lost name are the
                                             # same product failure
WORD_CLASSES = ("proper_noun", "digit_word", "spelled_letter", "function_word",
                "content_word")

FAMILY_LABEL = {
    "rt60": "reverb", "snr_db": "noise level", "noise_type": "noise character",
    "codec": "channel / codec", "mic_rolloff": "mic frequency response",
}


# ===========================================================================
# 1. WORD CLASSES — what kind of reference word is getting destroyed
# ===========================================================================
#
# SPEC A.R5.3 asks for {proper_noun, digit_word, spelled_letter, function_word};
# `content_word` is a residual bucket, because forcing every ordinary noun into one
# of the four would corrupt the counts. The residual is reported, not hidden.

# Spoken-form digits. The corpus says "four zero five", never "405" (Deepgram
# formatting is off so both sides stay in word form), so digits are LEXICAL here.
# "oh" is counted as a digit: in a corpus of numbers, addresses and codes that is
# overwhelmingly what it is, and "four oh five" vs "four zero five" is exactly the
# entity failure this layer is about.
DIGIT_WORDS = frozenset("""
zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty
sixty seventy eighty ninety hundred thousand million billion
first second third fourth fifth sixth seventh eighth ninth tenth
oh
""".split())

FUNCTION_WORDS = frozenset("""
a an the and or but if then than that this these those there here
i you he she it we they me him her us them my your his its our their
is are was were be been being am do does did done doing have has had having
will would can could shall should may might must
to of in on at by for with from into onto over under about as up down out off
not no yes so very just too also only please thank thanks
""".split())

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_STRIP = string.punctuation + "“”‘’"


def proper_nouns_in_say_this(say_this: str) -> set[str]:
    """
    Proper nouns inferred from MID-SENTENCE CAPITALIZATION in the manifest's
    `say_this` column (the human-written prompt keeps its casing; `ground_truth` is
    already lowercased for scoring).

    Two limitations, both conservative — they under-count, never over-count, the
    right direction for a claim like "codec destroys proper nouns":
      * a SENTENCE-INITIAL name is indistinguishable from a capitalized common word;
      * a single capital ("the code is A seven X four two") is a SPELLED LETTER, not
        a name, so it is left to the spelled_letter rule.
    """
    found: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(say_this or ""):
        for i, tok in enumerate(sentence.split()):
            core = tok.strip(_STRIP)
            if not core or i == 0:              # sentence-initial: unknowable
                continue
            if core == "I" or len(core) == 1:   # pronoun / spelled letter
                continue
            if core[0].isupper():
                found.update(normalize_text(core))
    return found


def load_manifest(path: str = "recording_manifest.csv") -> dict[str, dict]:
    """
    id -> {ref_tokens, proper_nouns}: the opportunity denominator and the
    proper-noun index. {} if the manifest is absent — every consumer here treats it
    as optional enrichment.
    """
    if not path or not os.path.exists(path):
        return {}
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = (row.get("id") or "").strip()
            if cid:
                out[cid] = {
                    "ref_tokens": normalize_text(row.get("ground_truth") or ""),
                    "proper_nouns": proper_nouns_in_say_this(row.get("say_this") or ""),
                }
    return out


def proper_noun_index(manifest: dict[str, dict] | None) -> dict:
    """
    {"by_clip": {clip_id: set}, "all": set, "source": "manifest"|"unavailable"}.

    Per-clip beats a corpus-wide union because the same token can be a name in one
    utterance and a common word in another ("May", "Mark").

    No manifest -> the proper-noun class is UNAVAILABLE. There is deliberately no
    guess-the-name heuristic: "codec destroys proper nouns" is a headline claim, and
    a claim resting on "long word not in a hand-written list" is not worth making.
    """
    if not manifest:
        return {"by_clip": {}, "all": set(), "source": "unavailable"}
    by_clip = {cid: set(m.get("proper_nouns", set())) for cid, m in manifest.items()}
    allpn: set[str] = set()
    for s in by_clip.values():
        allpn |= s
    return {"by_clip": by_clip, "all": allpn, "source": "manifest"}


def classify_ref_word(word: str, clip_id: str | None = None,
                      pn_index: dict | None = None) -> str:
    """
    One reference word -> a WORD_CLASSES member. ORDER MATTERS:
      1. single alphabetic character -> spelled_letter, EXCEPT "a"/"i" (article and
         pronoun). This is why the capitalization pass skips single capitals.
      2. digit words before names, so a code word like "seven" is never captured by
         a name list.
      3. proper nouns from the manifest (per-clip first, corpus-wide second).
      4. function words, then the residual content_word bucket.
    """
    w = (word or "").strip().lower()
    if not w:
        return "content_word"
    if len(w) == 1 and w.isalpha():
        return "function_word" if w in {"a", "i"} else "spelled_letter"
    if w in DIGIT_WORDS:
        return "digit_word"
    pn = pn_index or {}
    if pn.get("source") == "manifest":
        if clip_id is not None and w in pn["by_clip"].get(clip_id, set()):
            return "proper_noun"
        if w in pn["all"]:
            return "proper_noun"
    return "function_word" if w in FUNCTION_WORDS else "content_word"


# ===========================================================================
# 2. PER-CONDITION EDIT COMPOSITION
# ===========================================================================

def _rates(row: dict) -> dict[str, float]:
    """Per-clip edit rates as a fraction of n_ref (the WER denominator)."""
    n_ref = as_float(row.get("n_ref"), 0.0)
    if not np.isfinite(n_ref) or n_ref <= 0:
        return {e: float("nan") for e in EDIT_TYPES}
    return {e: as_float(row.get(f"n_{e}"), 0.0) / n_ref for e in EDIT_TYPES}


def edit_composition(rows: Sequence[dict]) -> list[dict]:
    """
    Per condition: sub / del / ins as fractions of n_ref — the stacked-bar payload.

    Rates are MICRO (sum of edits / sum of reference words), so a long clip
    contributes proportionally, which is the right weighting for "what fraction of
    spoken words got destroyed". The three rates sum to the micro WER by
    construction; that invariant is what makes the stacked bar readable.
    """
    # A duplicated (clip, condition) row is counted twice in BOTH the numerator
    # and the denominator here, so `wer_micro` barely moves — but that clip's
    # edit-type mix gets double weight, which is precisely what a fingerprint
    # IS. A condition's signature could flip from deletions to substitutions on
    # one repeated row with no other visible symptom.
    check_unique_cells(
        rows, ("clip_id", "condition_name"), where="edit_composition",
        cause=("Usual cause: rows for one model concatenated from two runs, or "
               "a master table rebuilt over a cache that logged a re-run cell "
               "under a second run_id."))

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("condition_name"))].append(r)

    out: list[dict] = []
    for name, grp in groups.items():
        n_ref = float(sum(as_float(r.get("n_ref"), 0.0) for r in grp))
        rec = {"condition_name": name, "n_clips": len(grp), "n_ref_total": int(n_ref),
               "model": grp[0].get("model")}
        for e in EDIT_TYPES:
            n_e = float(sum(as_float(r.get(f"n_{e}"), 0.0) for r in grp))
            rec[f"n_{e}"] = int(n_e)
            rec[f"{e}_rate"] = (n_e / n_ref) if n_ref > 0 else float("nan")
        rec["wer_micro"] = sum(rec[f"{e}_rate"] for e in EDIT_TYPES)
        rec["dominant_edit"] = max(EDIT_TYPES, key=lambda e: rec[f"{e}_rate"])
        tot = sum(rec[f"n_{e}"] for e in EDIT_TYPES)   # "reverb is 70% deletions"
        for e in EDIT_TYPES:
            rec[f"{e}_share"] = (rec[f"n_{e}"] / tot) if tot else float("nan")
        rec.update({k: grp[0].get(k) for k in DEFAULT_FACTOR_SPACE.names})
        out.append(rec)
    out.sort(key=lambda r: r["condition_name"])
    return out


# ===========================================================================
# 3. FACTOR-FAMILY SIGNATURES + THE IMPLIED FIX
# ===========================================================================

# The three signature/fix pairs SPEC §5.2 and A.R5.3 name, keyed by
# (family, dominant edit) where `family` is a factor name for continuous factors and
# "factor=level" for categorical levels. Anything else falls through to
# GENERIC_FIXES rather than inventing a prescription for a signature this instrument
# has not observed.
IMPLIED_FIXES: dict[tuple[str, str], str] = {
    ("rt60", "del"):
        "dereverberation front-end (WPE) or closer/beamformed capture. Deletions "
        "mean whole words are vanishing under the reverb tail, not being confused "
        "— a language-model or boosting fix cannot recover a word the acoustic "
        "model never emitted.",
    ("noise_type=babble", "sub"):
        "keyword / entity boosting on the domain lexicon: competing speech is "
        "pulling entity words toward common vocabulary, which boosting directly "
        "counteracts.",
    ("noise_type=babble", "ins"):
        "SPEAKER GATING, not denoising: target-speaker extraction, or diarization "
        "with a primary-speaker filter. These insertions are the model correctly "
        "transcribing the WRONG PERSON (see INSERTION_CAVEAT) — a denoiser tuned "
        "for stationary noise will not touch them.",
    ("codec", "sub"):
        "entity-aware / constrained decoding plus keyword boosting: the codec has "
        "removed the high-frequency cues that separate letters and digits, so the "
        "acoustic evidence is genuinely gone and only a decoding-side prior can "
        "recover the entity.",
}

GENERIC_FIXES = {
    "sub": "entity/keyword boosting + condition-matched training augmentation.",
    "del": "front-end recovery (dereverberation / gain / VAD thresholds): the "
           "words are not reaching the decoder at all.",
    "ins": "endpointing and speaker gating: spurious tokens usually come from "
           "non-target audio, not from confusion about the target.",
}

# What a level that is BETTER than its siblings gets INSTEAD of a prescription
# (rule 1 in the module docstring). Printing "keyword boosting" next to the cleanest
# cell in the grid reads as an observed failure that was never observed.
NO_FIX_NEEDED = ("none — this level produces FEWER of this edit type than the "
                 "others; it is a relative improvement, not a failure mode.")

INSERTION_CAVEAT = (
    "Insertions under babble are largely COMPETING-SPEECH CAPTURE, not acoustic "
    "confusion: babble contains intelligible speech and the model transcribes the "
    "background speakers. Different mechanism, different fix (target-speaker "
    "extraction / speaker gating, not denoising), so it is counted separately. "
    "Split rule: an inserted token that also occurs in the clip's own reference is "
    "ENDOGENOUS (a split or duplicated real word -> confusion); one that occurs "
    "nowhere in the reference is FOREIGN (a word from outside the utterance -> "
    "competing speech). The rule is a heuristic — a foreign insertion can also be a "
    "hallucinated function word, Whisper's characteristic failure under heavy noise "
    "— so example transcripts travel with the numbers.")


def implied_fix(family: str, factor: str, edit: str) -> str:
    """(family, edit) -> intervention, falling back to (factor, edit) then generic."""
    return (IMPLIED_FIXES.get((family, edit))
            or IMPLIED_FIXES.get((factor, edit))
            or GENERIC_FIXES.get(edit, "no standard intervention recorded"))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (a - b). NaN-safe; inf if both groups constant."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                     / (a.size + b.size - 2))
    diff = float(a.mean() - b.mean())
    if pooled == 0:
        return float("inf") if diff > 0 else (float("-inf") if diff < 0 else 0.0)
    return diff / float(pooled)


def _continuous_contrast(rows: Sequence[dict], f: Factor) -> tuple[list, list, str] | None:
    """
    Split rows into the DEGRADED third and the CLEAN third of a continuous factor,
    using `Factor.degradation` so the sign convention is uniform: a positive delta
    always means "more degradation -> more of this edit type". For snr_db
    (degradation='down') the degraded end is LOW dB — reading that backwards is how
    a fingerprint table ends up claiming noise fixes ASR.
    """
    vals = np.array([as_float(r.get(f.name)) for r in rows], dtype=float)
    ok = np.isfinite(vals)
    if ok.sum() < 4 or np.ptp(vals[ok]) == 0:
        return None
    lo_t, hi_t = np.quantile(vals[ok], [1 / 3, 2 / 3])
    low_idx = np.where(ok & (vals <= lo_t))[0]
    high_idx = np.where(ok & (vals >= hi_t))[0]
    if len(low_idx) == 0 or len(high_idx) == 0:
        return None
    if f.degradation == "down":                       # smaller value = worse audio
        worse, better = low_idx, high_idx
        label = f"{f.name} <= {lo_t:.3g} (degraded) vs >= {hi_t:.3g} (clean)"
    else:                                             # 'up' or unspecified
        worse, better = high_idx, low_idx
        label = f"{f.name} >= {hi_t:.3g} (degraded) vs <= {lo_t:.3g} (clean)"
    return ([rows[i] for i in worse], [rows[i] for i in better], label)


def factor_signatures(rows: Sequence[dict], space: FactorSpace = DEFAULT_FACTOR_SPACE,
                      min_group: int = 3) -> list[dict]:
    """
    One signature per factor family: which edit type this factor CAUSES, how big the
    effect is, and what it implies you should do.

    Continuous factors contrast the degraded third against the clean third;
    categorical factors contrast each level against all other levels. Effect size is
    reported twice on purpose — `delta` (absolute change in the edit rate, the
    number to quote: "+0.28 of all spoken words deleted") and `cohens_d` (whether
    that is large relative to clip-to-clip scatter).

    The dominant edit type is the one with the largest |delta|. It selects the
    implied fix ONLY when `degrades` is true — see rule 1 in the module docstring;
    a level that produces FEWER of that edit than its siblings gets NO_FIX_NEEDED.
    """
    rows = list(rows)
    out: list[dict] = []
    contrasts: list[tuple[str, str, list, list, str]] = []

    for f in space.factors:
        if not rows or f.name not in rows[0]:
            continue
        if f.kind == "continuous":
            got = _continuous_contrast(rows, f)
            if got:
                worse, better, label = got
                contrasts.append((f.name, f.name, worse, better, label))
        else:
            for level in (f.levels or ()):
                worse = [r for r in rows if r.get(f.name) == level]
                better = [r for r in rows if r.get(f.name) not in (None, level)]
                if len(worse) >= min_group and len(better) >= min_group:
                    contrasts.append((f"{f.name}={level}", f.name, worse, better,
                                      f"{f.name} == {level} vs all other levels"))

    for family, factor, worse, better, label in contrasts:
        deltas, ds, rates_w, rates_b = {}, {}, {}, {}
        for e in EDIT_TYPES:
            a = np.array([_rates(r)[e] for r in worse], dtype=float)
            b = np.array([_rates(r)[e] for r in better], dtype=float)
            rates_w[e] = float(np.nanmean(a)) if a.size else float("nan")
            rates_b[e] = float(np.nanmean(b)) if b.size else float("nan")
            deltas[e] = rates_w[e] - rates_b[e]
            ds[e] = _cohens_d(a, b)
        dominant = max(EDIT_TYPES,
                       key=lambda e: abs(deltas[e]) if np.isfinite(deltas[e]) else -1)
        wer_w = float(np.nanmean([as_float(r.get("wer")) for r in worse]))
        wer_b = float(np.nanmean([as_float(r.get("wer")) for r in better]))

        # THE SIGN GATE (module rule 1). Only MORE of this edit type on the degraded
        # side is a failure signature, and only a failure signature earns a fix.
        degrades = bool(np.isfinite(deltas[dominant]) and deltas[dominant] > 0)
        arrow = f"{dominant.upper()}S" if degrades else f"FEWER {dominant.upper()}S"

        entry = {
            "family": family, "factor": factor, "label": FAMILY_LABEL.get(factor, factor),
            "contrast": label, "n_degraded": len(worse), "n_reference": len(better),
            "rates_degraded": rates_w, "rates_baseline": rates_b,
            "deltas": deltas, "cohens_d": ds,
            "dominant_edit": dominant, "degrades": degrades,
            "delta": deltas[dominant], "effect_size_d": ds[dominant],
            "wer_degraded": wer_w, "wer_baseline": wer_b, "wer_delta": wer_w - wer_b,
            "signature": (f"{FAMILY_LABEL.get(factor, factor)} [{label}] -> {arrow} "
                          f"({dominant} rate {rates_b[dominant]:.3f} -> "
                          f"{rates_w[dominant]:.3f}, delta {deltas[dominant]:+.3f} of "
                          f"ref words, d={ds[dominant]:+.2f})"),
            "implied_fix": (implied_fix(family, factor, dominant) if degrades
                            else NO_FIX_NEEDED),
        }
        if family.startswith("noise_type=") and dominant == "ins" and degrades:
            entry["caveat"] = INSERTION_CAVEAT
        out.append(entry)

    out.sort(key=lambda d: abs(d["delta"]) if np.isfinite(d["delta"]) else -1,
             reverse=True)
    return out


# ===========================================================================
# 4. SUBSTITUTED-WORD INVENTORY — which reference words get destroyed
# ===========================================================================

def substituted_word_inventory(rows: Sequence[dict], manifest: dict | None = None,
                               top_k: int = 40) -> dict:
    """
    Which reference words are destroyed (substituted OR deleted) and what CLASS they
    belong to.

    OPPORTUNITY DENOMINATOR. A raw count is confounded by how often a word is spoken
    ("the" is destroyed most because "the" is said most). With the manifest we count
    opportunities exactly (occurrences in that clip's reference x rows for that clip)
    and report a destruction RATE — the number that supports "proper nouns are
    destroyed 4x more often than function words". Without it, rates are NaN and only
    counts are reported: the edits column stores errors only (matches are not kept),
    so the denominator is genuinely unavailable and is stated, not faked.
    """
    pn = proper_noun_index(manifest)
    dest: Counter[str] = Counter()
    per_op: dict[str, Counter[str]] = {e: Counter() for e in DESTROY_OPS}
    confusions: dict[str, Counter[str]] = defaultdict(Counter)
    word_clip: dict[str, str] = {}          # a clip id per word, for per-clip classing

    for r in rows:
        cid = str(r.get("clip_id"))
        for op, ref_w, hyp_w in parse_edits(r):
            if ref_w is None or op not in DESTROY_OPS:
                continue                    # insertions carry no reference word
            per_op[op][ref_w] += 1
            dest[ref_w] += 1
            word_clip.setdefault(ref_w, cid)
            if op == "sub" and hyp_w:
                confusions[ref_w][hyp_w] += 1

    opportunities: Counter[str] = Counter()
    if manifest:
        rows_per_clip: Counter[str] = Counter(str(r.get("clip_id")) for r in rows)
        for cid, n_rows in rows_per_clip.items():
            for w in (manifest.get(cid, {}).get("ref_tokens") or []):
                opportunities[w] += n_rows

    by_word = []
    for w, n in dest.most_common():
        opp = opportunities.get(w, 0)
        by_word.append({
            "word": w, "word_class": classify_ref_word(w, word_clip.get(w), pn),
            "n_destroyed": n, "n_sub": per_op["sub"][w], "n_del": per_op["del"][w],
            "n_opportunities": opp if manifest else None,
            "destruction_rate": (n / opp) if opp else float("nan"),
            "top_confusions": confusions[w].most_common(3),
        })

    by_class: dict[str, dict] = {}
    total = sum(dest.values())
    for cls in WORD_CLASSES:
        words = [d for d in by_word if d["word_class"] == cls]
        n_d = sum(d["n_destroyed"] for d in words)
        n_opp = sum(d["n_opportunities"] or 0 for d in words) if manifest else 0
        by_class[cls] = {
            "n_words": len(words), "n_destroyed": n_d,
            "n_sub": sum(d["n_sub"] for d in words),
            "n_del": sum(d["n_del"] for d in words),
            "n_opportunities": n_opp if manifest else None,
            "destruction_rate": (n_d / n_opp) if n_opp else float("nan"),
            "share_of_destroyed": (n_d / total) if total else float("nan"),
            "examples": [d["word"] for d in words[:5]],
        }
    return {
        "by_word": by_word[:top_k], "by_class": by_class,
        "n_destroyed_total": total,
        "proper_noun_source": pn["source"],
        "opportunity_denominator": "manifest" if manifest else "unavailable",
        "ops_counted": DESTROY_OPS,
    }


# ===========================================================================
# 5. THE BABBLE INSERTION SPLIT (module rule 2 / SPEC A.R5.3's named gotcha)
# ===========================================================================

def _clip_reference_vocab(rows: Sequence[dict],
                          manifest: dict | None = None) -> dict[str, set[str]]:
    """
    clip_id -> the words that clip's reference contains. From the manifest when
    available (exact); otherwise reconstructed as the union of every ref_word seen in
    that clip's edits, which is PARTIAL by construction (a word never mis-recognized
    anywhere never appears) and so over-calls insertions "foreign". Flagged in the
    result as `reference_source`.
    """
    vocab: dict[str, set[str]] = defaultdict(set)
    if manifest:
        for cid, m in manifest.items():
            vocab[cid] |= set(m.get("ref_tokens") or [])
    for r in rows:
        for _op, ref_w, _hyp in parse_edits(r):
            if ref_w:
                vocab[str(r.get("clip_id"))].add(ref_w)
    return vocab


def classify_insertions(rows: Sequence[dict], manifest: dict | None = None,
                        group_key: str = "noise_type", n_examples: int = 3) -> dict:
    """
    Split every insertion into foreign (competing-speech capture) vs endogenous
    (acoustic confusion), aggregated by noise type and by condition, with example
    transcripts attached. See INSERTION_CAVEAT for the rule and its limits.
    """
    vocab = _clip_reference_vocab(rows, manifest)
    blank = {"n_ins": 0, "n_foreign": 0, "n_endogenous": 0, "n_ref": 0.0}
    by_group: dict[str, dict] = defaultdict(lambda: dict(blank, foreign_words=Counter()))
    by_condition: dict[str, dict] = defaultdict(lambda: dict(blank))
    examples: list[dict] = []

    for r in rows:
        cid, g = str(r.get("clip_id")), str(r.get(group_key))
        cond = str(r.get("condition_name"))
        ref_vocab = vocab.get(cid, set())
        foreign_here = [hyp for op, _ref, hyp in parse_edits(r)
                        if op == "ins" and hyp and hyp not in ref_vocab]
        n_endo = sum(1 for op, _ref, hyp in parse_edits(r)
                     if op == "ins" and hyp and hyp in ref_vocab)
        n_foreign = len(foreign_here)
        n_ref = as_float(r.get("n_ref"), 0.0)
        for tgt, key in ((by_group, g), (by_condition, cond)):
            tgt[key]["n_ins"] += n_foreign + n_endo
            tgt[key]["n_foreign"] += n_foreign
            tgt[key]["n_endogenous"] += n_endo
            tgt[key]["n_ref"] += n_ref if np.isfinite(n_ref) else 0.0
        by_group[g]["foreign_words"].update(foreign_here)
        if n_foreign:
            examples.append({
                "clip_id": cid, "condition_name": cond, group_key: g,
                "n_foreign": n_foreign, "foreign_tokens": foreign_here,
                "transcript": r.get("transcript"),
            })

    def _finish(d: dict) -> dict:
        n_ins, n_ref = d["n_ins"], d["n_ref"]
        d["foreign_frac"] = (d["n_foreign"] / n_ins) if n_ins else float("nan")
        d["foreign_rate"] = (d["n_foreign"] / n_ref) if n_ref else float("nan")
        d["ins_rate"] = (n_ins / n_ref) if n_ref else float("nan")
        if "foreign_words" in d:
            d["top_foreign_words"] = d.pop("foreign_words").most_common(8)
        return d

    examples.sort(key=lambda e: e["n_foreign"], reverse=True)
    return {
        "by_group": {k: _finish(dict(v)) for k, v in by_group.items()},
        "group_key": group_key,
        "by_condition": {k: _finish(dict(v)) for k, v in by_condition.items()},
        "examples": examples[:n_examples],
        "reference_source": "manifest" if manifest else "edits-only (partial)",
        "caveat": INSERTION_CAVEAT,
    }


# ===========================================================================
# 6. ENTITY ERROR RATE — OPTIONAL (task_specs.json may not exist yet)
# ===========================================================================

def load_task_specs(path: str = "task_specs.json") -> tuple[dict, str | None]:
    """
    clip_id -> agent_eval.TaskSpec, or ({}, reason) if the file is not usable.
    Absence degrades the entity fingerprint to a stated "unavailable"; it never
    fails the run and never silently produces zeros (which read as "no errors").
    """
    if not path or not os.path.exists(path):
        return {}, (f"{path!r} not found — entity fingerprint skipped "
                    f"(author it per SPEC A.R5.2, then re-run)")
    from deadzone.agent_eval import TaskSpec         # local import: keeps this optional
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return {}, f"{path!r} could not be read ({exc}) — entity fingerprint skipped"
    specs: dict[str, "TaskSpec"] = {}
    for cid, d in (raw or {}).items():
        slots = d.get("slots") if isinstance(d, dict) else None
        if slots:
            specs[str(cid)] = TaskSpec(slots=dict(slots),
                                       critical=tuple(d.get("critical") or ()))
    if not specs:
        return {}, f"{path!r} contained no usable specs — entity fingerprint skipped"
    return specs, None


def entity_error_by_condition(rows: Sequence[dict],
                              task_specs_path: str = "task_specs.json") -> dict:
    """
    Entity-error rate per condition via `agent_eval.entity_error_rate`, next to WER.

    The interesting quantity is the GAP: entity error rate minus WER. The entity
    layer *should* degrade faster (entities are rare, information-dense and
    un-guessable — a language model cannot fill them back in). If it does not, that
    is the finding, so `entity_minus_wer` is signed and reported either way.
    """
    specs, reason = load_task_specs(task_specs_path)
    if not specs:
        return {"available": False, "reason": reason, "by_condition": [],
                "overall": {}, "n_rows_scored": 0, "n_rows_unscorable": 0}

    from deadzone.agent_eval import evaluate_task
    per_cond: dict[str, dict] = defaultdict(
        lambda: {"eer": [], "wer": [], "critical": [], "n_rows": 0})
    n_no_spec = 0
    for r in rows:
        spec = specs.get(str(r.get("clip_id")))
        transcript = r.get("transcript")
        # a null transcript is a missing measurement, not a zero-entity score
        if spec is None or transcript is None:
            n_no_spec += 1
            continue
        res = evaluate_task(str(transcript), spec)
        c = per_cond[str(r.get("condition_name"))]
        c["eer"].append(res["entity_error_rate"])
        c["wer"].append(as_float(r.get("wer")))
        c["critical"].append(1.0 if res["critical_failure"] else 0.0)
        c["n_rows"] += 1

    by_condition = []
    for name, c in per_cond.items():
        eer = float(np.nanmean(c["eer"])) if c["eer"] else float("nan")
        wer = float(np.nanmean(c["wer"])) if c["wer"] else float("nan")
        by_condition.append({
            "condition_name": name, "n_rows": c["n_rows"],
            "entity_error_rate": eer, "wer": wer, "entity_minus_wer": eer - wer,
            "critical_failure_rate": (float(np.nanmean(c["critical"]))
                                      if c["critical"] else float("nan")),
        })
    by_condition.sort(key=lambda d: d["entity_minus_wer"], reverse=True)
    gaps = np.array([d["entity_minus_wer"] for d in by_condition], dtype=float)
    return {
        "available": True, "reason": None, "by_condition": by_condition,
        "n_rows_scored": int(sum(d["n_rows"] for d in by_condition)),
        "n_rows_unscorable": n_no_spec,
        "overall": {
            "mean_entity_error_rate": float(np.nanmean(
                [d["entity_error_rate"] for d in by_condition])) if by_condition else float("nan"),
            "mean_wer": (float(np.nanmean([d["wer"] for d in by_condition]))
                         if by_condition else float("nan")),
            "mean_entity_minus_wer": float(np.nanmean(gaps)) if gaps.size else float("nan"),
            "entities_degrade_faster": bool(np.nanmean(gaps) > 0) if gaps.size else False,
        },
    }


# ===========================================================================
# 7. THE REPORT
# ===========================================================================

def fingerprint_report(rows: Sequence[dict], model: str | None = None,
                       space: FactorSpace = DEFAULT_FACTOR_SPACE,
                       manifest_path: str = "recording_manifest.csv",
                       task_specs_path: str = "task_specs.json") -> dict:
    """
    The whole D2 layer for ONE model. Failed rows are split out first: their edit
    lists are empty, so averaging them in would dilute every rate toward zero and
    make a broken condition look CLEAN — the exact opposite of the truth.
    """
    if model is not None:
        rows = [r for r in rows if r.get("model") == model]
    rows = [coerce_row(r) for r in rows]     # None -> NaN, never 0.0 (see analysis/)
    fails = failure_summary(rows)
    ok, _bad = split_failures(rows)
    manifest = load_manifest(manifest_path)
    return {
        "model": model or (ok[0].get("model") if ok else None),
        "n_clip_rows_used": len(ok),
        "failures": fails,
        "manifest_loaded": bool(manifest),
        "composition": edit_composition(ok),
        "signatures": factor_signatures(ok, space),
        "inventory": substituted_word_inventory(ok, manifest),
        "insertions": classify_insertions(ok, manifest),
        "entities": entity_error_by_condition(ok, task_specs_path),
    }


def report_by_model(rows: Sequence[dict], **kw) -> dict[str, dict]:
    """One D2 report per model (fingerprints differ by model — that is L1's point)."""
    return {m: fingerprint_report(sub, model=m, **kw)
            for m, sub in split_by_model(rows).items()}


def plot_payload(report: dict) -> dict:
    """
    Dashboard contract for the D2 panels — JSON-serializable, no re-analysis needed
    downstream. STABLE SHAPE:

        {"model": str|None,
         "composition": [{"condition_name", "sub_rate", "del_rate", "ins_rate",
                          "wer_micro", "dominant_edit", "n_clips", "n_ref_total",
                          <each factor in DEFAULT_FACTOR_SPACE.names>}, ...],
                        # stacked bar; the three rates sum to wer_micro
         "signatures": [{"family", "label", "contrast", "dominant_edit", "delta",
                         "effect_size_d", "wer_delta", "degrades", "implied_fix",
                         "signature", "caveat"|None}, ...],
                        # `degrades` False => this level is BETTER than its siblings
                        # and `implied_fix` is NO_FIX_NEEDED. Do not render it as a
                        # recommendation.
         "word_classes": [{"word_class", "n_destroyed", "n_sub", "n_del",
                           "share_of_destroyed", "destruction_rate", "n_words",
                           "examples"}, ...],
         "insertion_split": [{"group", "n_ins", "n_foreign", "n_endogenous",
                              "foreign_frac", "top_foreign_words"}, ...],
         "insertion_examples": [{"clip_id","condition_name","foreign_tokens",
                                 "transcript"}, ...],
         "entity_gap": [{"condition_name", "entity_error_rate", "wer",
                         "entity_minus_wer", "critical_failure_rate"}, ...],
         "notes": {"insertion_caveat", "proper_noun_source",
                   "opportunity_denominator", "entities_available",
                   "entities_reason"},
         "n_failed_rows_excluded": int}
    """
    inv = report.get("inventory", {})
    ins = report.get("insertions", {})
    ent = report.get("entities", {})
    comp_keys = ("condition_name", "sub_rate", "del_rate", "ins_rate", "wer_micro",
                 "dominant_edit", "n_clips", "n_ref_total",
                 *DEFAULT_FACTOR_SPACE.names)
    sig_keys = ("family", "label", "contrast", "dominant_edit", "delta",
                "effect_size_d", "wer_delta", "degrades", "implied_fix", "signature")
    return {
        "model": report.get("model"),
        "composition": [{k: c.get(k) for k in comp_keys}
                        for c in report.get("composition", [])],
        "signatures": [{**{k: s.get(k) for k in sig_keys},
                        "caveat": s.get("caveat")}
                       for s in report.get("signatures", [])],
        "word_classes": [{"word_class": cls, **{k: v for k, v in d.items()
                                                if k != "n_opportunities"}}
                         for cls, d in inv.get("by_class", {}).items()
                         if d["n_words"]],
        "insertion_split": [{"group": g, **{k: v for k, v in d.items()
                                            if k != "n_ref"}}
                            for g, d in ins.get("by_group", {}).items()],
        "insertion_examples": ins.get("examples", []),
        "entity_gap": ent.get("by_condition", []),
        "notes": {
            "insertion_caveat": INSERTION_CAVEAT,
            "proper_noun_source": inv.get("proper_noun_source"),
            "opportunity_denominator": inv.get("opportunity_denominator"),
            "entities_available": bool(ent.get("available")),
            "entities_reason": ent.get("reason"),
        },
        "n_failed_rows_excluded": report.get("failures", {}).get("n_failed", 0),
    }


def format_signature_table(signatures: Sequence[dict], top_k: int = 12) -> str:
    """
    SPEC A.R5.3's Definition-of-Done table: one row per factor family —
    signature · dominant edit type · effect size · implied fix.
    """
    lines = ["Failure fingerprints — one row per factor family:",
             f"  {'family':<22} {'dominant':<9} {'delta':>7} {'d':>7} {'WER d':>7}  "
             f"signature / fix"]
    for s in signatures[:top_k]:
        lines.append(
            f"  {s['family']:<22} {s['dominant_edit']:<9} {s['delta']:>+7.3f} "
            f"{s['effect_size_d']:>+7.2f} {s['wer_delta']:>+7.3f}  {s['signature']}")
        lines.append(f"  {'':<22} {'FIX ->' if s['degrades'] else 'NO FIX:':<9} "
                     f"{s['implied_fix']}")
        if s.get("caveat"):
            lines.append(f"  {'':<22} {'NOTE ->':<9} {s['caveat'][:96]}...")
    return "\n".join(lines)


def format_report(report: dict) -> str:
    f, inv, ins = report["failures"], report["inventory"], report["insertions"]
    lines = [
        f"D2 failure fingerprints — model {report['model']!r}",
        f"  clip-rows used: {report['n_clip_rows_used']}   failed EXCLUDED: "
        f"{f['n_failed']} ({f['failure_rate']:.2%})",
        "  manifest: " + ("loaded" if report["manifest_loaded"]
                          else "ABSENT (proper-noun class unavailable)"),
        "",
        format_signature_table(report["signatures"]),
        "",
        f"Destroyed reference words by class "
        f"(proper-noun source: {inv['proper_noun_source']}, "
        f"denominator: {inv['opportunity_denominator']}):",
        f"  {'class':<16} {'n_destroyed':>11} {'share':>7} {'rate':>7}  examples",
    ]
    for cls, d in inv["by_class"].items():
        if d["n_words"]:
            lines.append(f"  {cls:<16} {d['n_destroyed']:>11} "
                         f"{d['share_of_destroyed']:>7.2%} "
                         f"{d['destruction_rate']:>7.3f}  {', '.join(d['examples'])}")
    lines += ["", f"Insertions split by {ins['group_key']} "
                  f"(reference source: {ins['reference_source']}):",
              f"  {'group':<14} {'n_ins':>6} {'foreign':>8} {'frac':>7}  "
              f"top foreign tokens"]
    for g, d in ins["by_group"].items():
        lines.append(f"  {g:<14} {d['n_ins']:>6} {d['n_foreign']:>8} "
                     f"{d['foreign_frac']:>7.2f}  "
                     f"{', '.join(w for w, _ in d.get('top_foreign_words', [])[:5])}")
    if ins["examples"]:
        ex = ins["examples"][0]
        lines.append(f"  example [{ex['condition_name']}, clip {ex['clip_id']}]: "
                     f"inserted {ex['foreign_tokens']} -> {ex['transcript']!r}")
    ent = report["entities"]
    lines.append("")
    if ent["available"]:
        o = ent["overall"]
        lines.append(f"Entity error rate {o['mean_entity_error_rate']:.3f} vs WER "
                     f"{o['mean_wer']:.3f} (gap {o['mean_entity_minus_wer']:+.3f}) — "
                     f"entities degrade "
                     f"{'FASTER' if o['entities_degrade_faster'] else 'no faster'} "
                     f"than WER")
    else:
        lines.append(f"Entity error rate: UNAVAILABLE — {ent['reason']}")
    return "\n".join(lines)


# ===========================================================================
# THE PERSISTED ARTIFACT
# ===========================================================================
#
# WHY D2 GETS A FILE. Every other analysis layer freezes itself to
# `results/<layer>.{json,txt}` — calibration, model_arms, sim2real, l3_decoupling.
# D2 did not: it printed to stdout and the write-up cited the COMMAND instead of
# an artifact. Every D2 number reproduces exactly today, but "reproduces" and
# "frozen" are different properties. The report is a pure function of
# `results/master.csv` + `recording_manifest.csv` + `task_specs.json`, so if any
# of those three moves — a re-run appending rows, a manifest transcript
# correction, a new entity slot — every fingerprint silently becomes a different
# number under the same claim, and a `grid-v1` tag would not have captured what
# was actually published. The tag can only freeze what is on disk.

OUT_JSON = "results/fingerprints.json"
OUT_TXT = "results/fingerprints.txt"

ARTIFACT_NOTE = (
    "D2 is a pure function of the three sources listed above and spends no API "
    "calls, so it is cheap to re-run — but it is frozen here because the WRITE-UP "
    "quotes these numbers. If master.csv, recording_manifest.csv or "
    "task_specs.json changes, re-running produces different fingerprints under "
    "the same claims; this file is what a git tag actually captures."
)


def build_artifact(rows: Sequence[dict], models: Sequence[str] | None = None,
                   manifest_path: str = "recording_manifest.csv",
                   task_specs_path: str = "task_specs.json") -> dict:
    """The whole D2 layer, every model, in one JSON-serializable payload."""
    names = list(models) if models else sorted(split_by_model(rows))
    by_model = {m: fingerprint_report(rows, model=m, manifest_path=manifest_path,
                                      task_specs_path=task_specs_path)
                for m in names}
    return {
        "layer": "D2 — typed failure fingerprints (SPEC §5.2, A.R5.3)",
        "artifact": OUT_JSON,
        "sources": {"table": None,              # filled by write_artifact's caller
                    "manifest": manifest_path,
                    "task_specs": task_specs_path},
        "models": names,
        "note": ARTIFACT_NOTE,
        "insertion_caveat": INSERTION_CAVEAT,
        "by_model": by_model,
    }


def format_artifact(payload: dict) -> str:
    """The printed report for every model, verbatim — the `.txt` sibling."""
    blocks = [format_report(rep) for rep in payload["by_model"].values()]
    return "\n\n".join(blocks)


def write_artifact(payload: dict, out_json: str | None = OUT_JSON,
                   out_txt: str | None = OUT_TXT) -> list[str]:
    """Freeze the payload to disk. Returns the paths actually written."""
    written: list[str] = []
    for path, render in ((out_json, None), (out_txt, format_artifact)):
        if not path:
            continue
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            if render is None:
                # `default=float` mirrors analysis/calibration_report.py: a numpy
                # scalar that slips through becomes a plain float rather than
                # raising at dump time and losing the whole artifact.
                json.dump(payload, fh, indent=2, default=float)
            else:
                fh.write(render(payload) + "\n")
        written.append(path)
    return written


# ===========================================================================
# CLI
# ===========================================================================

COMPOSITION_CSV_COLUMNS = ("condition_name", "model", *DEFAULT_FACTOR_SPACE.names,
                           "n_clips", "n_ref_total", "sub_rate", "del_rate",
                           "ins_rate", "wer_micro", "dominant_edit")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D2: typed failure fingerprints")
    ap.add_argument("--table", default="results/master.csv")
    ap.add_argument("--model", default=None)
    ap.add_argument("--manifest", default="recording_manifest.csv")
    ap.add_argument("--task-specs", default="task_specs.json")
    ap.add_argument("--out", default=None,
                    help="optional CSV of the per-condition edit composition")
    ap.add_argument("--out-json", default=OUT_JSON,
                    help="frozen D2 payload (pass '' to skip)")
    ap.add_argument("--out-txt", default=OUT_TXT,
                    help="the printed report, verbatim (pass '' to skip)")
    args = ap.parse_args(argv)

    rows = load_master_table(args.table)
    payload = build_artifact(rows, [args.model] if args.model else None,
                             manifest_path=args.manifest,
                             task_specs_path=args.task_specs)
    payload["sources"]["table"] = args.table
    comp: list[dict] = []
    for rep in payload["by_model"].values():
        print(format_report(rep), "\n")
        comp.extend(rep["composition"])

    for path in write_artifact(payload, args.out_json or None,
                               args.out_txt or None):
        print(f"wrote {path}")

    if args.out and comp:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(COMPOSITION_CSV_COLUMNS),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows({k: r.get(k) for k in COMPOSITION_CSV_COLUMNS} for r in comp)
        print(f"wrote {len(comp)} composition rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
