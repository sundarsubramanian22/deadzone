# What the three arms' confidence numbers actually are

Research note behind the README's "potential explanation" callout. **Nothing here is a
measurement**, and by design **it quotes no figure from `results/`** — observations are named
ordinally with the artifact that carries the number (`results/model_arms.{txt,json}`,
`results/confidence_char.txt`), so this note has nothing in it to drift. Its only numerals are
vendor-published facts and shipped-source constants. Tags: **DOCUMENTED** (vendor doc, model
card, paper, shipped source) · **INFERRED** (from behaviour or a published product constraint)
· **SPECULATION** (nobody published it and this project did not measure it).

**The ordering being explained** (`results/model_arms.txt`, 10 shared clips × 176 conditions):

- **nova-3 ranks first on every confidence statistic** — condition-level ρ(confidence,
  WER_spoke), dead-zone rate, utterance-level AUROC, word-level ECE.
- **Scribe and Whisper swap places depending on statistic and scoring** — Scribe above on ρ
  and ECE, Whisper above on utterance AUROC. So *"commercial above open"* is **not** a robust
  ordering, and an explanation that only produces it is explaining something the data does
  not say.

---

## 1. Deepgram Nova-3

- **DOCUMENTED — what the field is *said* to be.** `confidence` appears at word and
  alternative level, defined identically for both: *"a floating point value between 0 and 1
  that indicates overall transcript reliability. Larger values indicate higher confidence."*
  That is the whole published definition — product language, not decoder language. No
  statement of what it is computed from, and **no claim that it is calibrated.**
  <https://developers.deepgram.com/docs/pre-recorded-audio>
- **DOCUMENTED — architecture: essentially nothing.** The launch post offers *"a sophisticated
  audio embedding framework that uses representation learning to project audio into a highly
  compressed and expressive latent space"* and *"rather than relying on conventional language
  model post-processing, the model incorporates a trained contextual mechanism that enables
  in-context learning at inference time."* No family, loss, decoder type, parameter count or
  training hours. <https://deepgram.com/learn/introducing-nova-3-speech-to-text-api> ·
  <https://developers.deepgram.com/docs/models-languages-overview>
- **DOCUMENTED (patent — weak evidence about the shipped model).** Deepgram Inc is assignee of
  **US 12,380,880 B2, "End-to-end automatic speech recognition with transformer"**: an
  end-to-end ASR model feeding learned embeddings directly into a transformer encoder-decoder
  (skipping tokenization), with an **autoregressive** decoder and *"a final linear
  transformation, and a softmax layer, to produce output probabilities over a set of
  vocabulary."* (Also US 11,367,433 B2 / US 10,720,151 B2.) **A patent records what was filed,
  not what ships as `nova-3`** — house style, not a spec. <https://deepgram.com/patents> ·
  <https://patents.google.com/patent/US12380880B2/en>
- **INFERRED.** One literal `nova-3` serves the pre-recorded endpoint *and* `listen.live` with
  interim results, so it must commit text left-to-right under a latency budget with bounded
  right context. That rules out a free-running full-utterance decode as its *only* mode; it
  does not identify the family.
- **NOT PUBLIC — say so plainly.** How the confidence number is produced (decoder posterior?
  lattice/alignment score? frame posterior? trained confidence head?) is not stated anywhere I
  could find. **The arm that performs best is the one we can say the least about.**

## 2. ElevenLabs Scribe v2

- **DOCUMENTED — the field.** Per-word `logprob`: *"The log of the probability with which this
  word was predicted. Logprobs are in range [-infinity, 0], higher logprobs indicate a higher
  confidence the model has in its predictions."* Separately `language_probability` is *"The
  confidence score of the language detection (0 to 1)"* — **not** per-word confidence;
  treating it as one would give every word in a clip the same value.
  <https://elevenlabs.io/docs/api-reference/speech-to-text/convert>
- **DOCUMENTED — architecture: nothing.** No family, parameter count, tokenizer or training
  data in the launch post or docs. Independent survey: *"There is no public statement of
  Scribe v2's model family, encoder/decoder design, parameter count, tokenizer, training-hours
  total, or training-data composition"* — *"well documented as a product, weakly documented as
  a model."* <https://elevenlabs.io/blog/introducing-scribe-v2> ·
  <https://opentranscription.io/blog/elevenlabs-scribe-v2.html>
- **INFERRED (weak, and about a *different* model).** The **realtime** variant advertises
  *"Negative latency: Next word and punctuation prediction"* at <150 ms; predicting a word
  before it is spoken requires a language-model-conditioned component. This project ran the
  **batch** `scribe_v2`, so this is a hint about house style at most.
  <https://elevenlabs.io/blog/introducing-scribe-v2-realtime>
- **Behaviour this project established** (`results/confidence_char.txt`; scope in SPEC I): the
  scale tops out at exactly `logprob = 0.0`, so `exp()` reaches exactly 1.0 — the adapter clips
  to avoid `+inf` at the logit — and **nearly half its emitted words sit at that ceiling**,
  where ties cannot be ordered by any threshold. Its **orthography is non-deterministic across
  byte-identical calls**, which is why the arm is rank-only.

## 3. OpenAI Whisper (`whisper-base`)

The only arm whose mechanism is fully checkable — paper *and* code are public. That is why it
should carry the weight of the callout.

- **DOCUMENTED — architecture.** *"Whisper is a Transformer based encoder-decoder model, also
  referred to as a sequence-to-sequence model."* `base` is **74 M parameters**; trained on
  **680,000 hours** of weakly-supervised web audio.
  <https://huggingface.co/openai/whisper-base> · <https://arxiv.org/abs/2212.04356>
- **DOCUMENTED — the failure mode, by the vendor, in its own model card.** *"the predictions
  may include texts that are not actually spoken in the audio input (i.e. hallucination). We
  hypothesise that this happens because … the models combine predicting the next word with
  transcribing the audio itself."* And *"the architecture makes it prone to generating
  repetitive texts, which can be mitigated to some degree by beam search and temperature
  scheduling but not perfectly."* **The repetition loop this project measured is the documented
  failure mode, not a surprise.**
- **DOCUMENTED — what the number *is*, read off the shipped source.** `whisper/timing.py`,
  `find_alignment`: a teacher-forced forward pass over the already-decoded token sequence —
  `token_probs = sampled_logits.softmax(dim=-1)`,
  `text_token_probs = token_probs[arange, text_tokens]`, then
  `word_probabilities = [np.mean(text_token_probs[i:j]) …]`. The per-word number is therefore
  **the arithmetic mean, over a word's subword tokens, of the decoder's next-token softmax
  probability conditioned on the tokens it already generated** — not an acoustic posterior, but
  a joint acoustic + own-context score. (`utterance_conf` is `mean(exp(avg_logprob))` per
  segment.) Read in `deadzone/audio_pipeline.py::_parse_whisper_result`, which documents it.
- **DOCUMENTED — OpenAI ships heuristics conceding the score is insufficient.**
  `transcribe.py` defaults include `logprob_threshold=-1.0`, `no_speech_threshold=0.6` and
  **`compression_ratio_threshold=2.4`** — a *gzip* ratio test. A gzip check exists because the
  log-probability does not reliably catch repetition loops: a loop is both highly compressible
  *and* highly probable under its own context.
- **Literature — documented hypotheses, not this project's.** *Careless Whisper* (Koenecke et
  al., FAccT 2024) finds hallucinations correlated with longer non-vocal share and hypothesises
  *"Whisper's over-reliance on … modern language modeling is what leads to hallucinations"*
  (<https://arxiv.org/abs/2402.08021>); on non-speech audio the rate is far higher, with a small
  recurring repertoire (<https://arxiv.org/abs/2501.11378>). Attention-based seq2seq softmax is
  separately documented as overconfident
  (<https://research.google/pubs/confidence-estimation-for-attention-based-sequence-to-sequence-models-for-speech-recognition/>),
  and AED/RNN-T models are known to learn an **implicit internal LM** from training transcripts
  (<https://arxiv.org/abs/2104.05544>).

---

## Candidate explanations for the ordering

**C1 — The confound, and it is not architectural. SPECULATION** (partly measurable now). A
deleted word emits no token and therefore **no confidence**. Deletions dominate nova-3's
errors, so most of its failures are *structurally excluded from the statistic that ranks it
first*; Whisper's errors are insertion-heavy, so nearly every failure must carry a score that
can be wrong. **Explains:** nova-3 topping the ρ ranking while carrying the highest silent-row
rate and the most mute conditions (`model_arms.txt`, silence accounting). **Falsified by:**
re-running the ordering against reference recovery rather than emitted-word accuracy — the repo
reports both — and finding it unchanged. **Lead with this: strongest candidate, and it argues
against our own headline.**

**C2 — Whisper scores what it generated, not what it heard. SPECULATION** (best-evidenced). The
number is `p(token | preceding tokens, audio)`, source-verified in §3; inside a repetition loop
the preceding context is maximally predictive, so the score can stay high while the audio
contributes almost nothing. **Explains:** the highest dead-zone rate and weakest ρ of the three
arms, the reference→hypothesis blow-up example in `model_arms.txt`, and the non-language glyph
row returned at high confidence. **Falsified by:** an acoustic-only score over the same
hypotheses (CTC head, or forced alignment) showing the same dead-zone rate; or high-confidence
hallucinations whose contexts are *not* self-predictive.

**C3 — Commit pressure limits how far a generation can run away. SPECULATION.** nova-3 is served
for batch *and* `listen.live` (INFERRED, §1); Whisper decodes a 30-second window with full
lookahead and no commit deadline. **Explains:** the hypothesis/reference length-ratio gap in the
per-arm hallucination table. **Falsified by:** a chunked streaming Whisper wrapper in which the
loops persist at the same rate; or Deepgram stating batch and streaming share one graph.

**C4 — nova-3's field may be a product score, not a raw softmax. SPECULATION** (weakest).
Deepgram calls it *"transcript reliability"*, never a posterior, and a fraction of its words
return exactly 1.0 with a floor well above zero. **Explains:** its markedly lower ECE. **Weak
because** a float32 softmax *can* round to exactly 1.0 at a large logit margin, so saturation is
not evidence of post-processing. **Falsified by:** Deepgram stating the field is the unmodified
decoder posterior.

**C5 — Scribe's position is partly an artifact of its own output. SPECULATION** (partly
measured). Two non-architectural mechanisms: (i) nearly half its words are tied at the ceiling,
so utterance-level ranking has no resolution — sufficient to explain its weak AUROC without any
claim about self-knowledge; (ii) its WER *label* is a per-call draw, and measurement error in
the y-variable attenuates |ρ|. The repo measures (ii): Scribe's ρ strengthens materially under
normalization while nova-3's does not move. **Falsified by:** a persisted repeat-call artifact
(SPEC I.9 — does not exist yet) showing the residual variance is negligible.

**C6 — Confounded by size, not only by vendor. INFERRED.** `whisper-base` is 74 M parameters;
`nova-3` and `scribe_v2` are undisclosed and near-certainly larger. "Commercial beats open" is
not separated from "large beats small." **Falsified by:** `whisper-large-v3` on the same grid.

---

## Draft README callout

> **POTENTIAL EXPLANATION — speculation, not measured.** Deepgram's internals are not public;
> what follows is a hypothesis about the ordering, not a finding.
>
> - **Whisper's confidence scores what it *generated*.** Its per-word number is (source:
>   `whisper/timing.py`) the mean softmax probability of each subword **conditioned on the
>   tokens it already emitted** — a joint acoustic + language-model score, not an acoustic
>   posterior. *SPECULATION: in a repetition loop the context alone makes the next token
>   near-certain, which is why it stays confident while inventing.* OpenAI's model card
>   documents both the hallucination and the repetition tendency, and `transcribe.py` ships a
>   **gzip compression-ratio** check — a tell that the log-probability alone does not catch
>   loops.
> - **Deepgram documents its field only as "overall transcript reliability, 0–1"** — not whether
>   it is a decoder posterior, an alignment score, or a trained confidence head, and with no
>   claim of calibration. *The best-performing arm is the one we can say the least about.*
> - **ElevenLabs documents `logprob` as a token-prediction log-probability** — the same *kind* of
>   quantity as Whisper's, on its own scale. Nearly half its words are tied at the ceiling, which
>   **removes ranking resolution**; *SPECULATION: that alone is sufficient to explain its weak
>   utterance-level separation, with no claim about self-knowledge needed.*
> - **The confound that cuts against our own headline:** nova-3 fails by **deleting**, and a
>   deleted word carries no confidence — so most of its failures are invisible to the statistic
>   that ranks it best. Whisper fails by **inserting**, so every failure must carry a score that
>   can be wrong. *SPECULATION: part of the ordering is which failure mode the metric can see.*
> - **Also confounded by size:** `whisper-base` is 74 M parameters; both commercial arms are
>   undisclosed and near-certainly larger. "Commercial vs open" is not separated from "large vs
>   small."
