# Measured facts about the capture chain

Running log of things measured at record time that feed the write-up's
limitations section (SPEC appendix R8.2 §8). Recorded when fresh, because none
of this is recoverable after the fact.

---

## Room-tone noise floor — 2026-08-04

**Measured:** `-52.9 dBFS` RMS over 9.94 s of silence
(`data/recordings/_roomtone.wav`, 48 kHz / mono / PCM_16, built-in MacBook Pro
microphone). SPEC R1.3 targets `<= -60 dBFS`, so the capture chain is **~7 dB
over target**.

**Diagnosis.** Not broadband mic self-noise — the mic itself is quiet
(-36 to -46 dB relative above 2 kHz). The floor is almost entirely
low-frequency, tonal, and stationary:

| band | share of energy |
|---|---|
| 20–63 Hz | 24.0 % |
| 63–125 Hz | 66.9 % |
| 125–250 Hz | 6.8 % |
| > 250 Hz | 2.3 % |

Dominant tone at **120 Hz** (2nd harmonic of 60 Hz mains), plus 63 and 94 Hz.
Per-100 ms level spread is only 5.4 dB → continuous source, not intermittent.
This is mains hum coupling in through the power chain, not room noise.

**Why it was not fixed.** The hum tracks the mains/monitor connection, and the
external monitor is needed to read the recording manifest while recording.
Filtering was rejected: a high-pass at 80 Hz recovers only 2.4 dB, and reaching
-60 dBFS requires cutting at 120 Hz, which sits on top of the voice fundamental
— and SPEC R1.2 forbids EQ in the capture chain regardless.

**The number that actually matters — measured in-clip, 2026-08-04.** The
standalone room-tone file is pessimistic; the authoritative floor is the one
measured inside the recordings themselves, through the same gain path as the
speech. On the two checkpoint clips:

| clip | floor | active speech | inherent SNR |
|---|---|---|---|
| u01 | -55.5 dBFS | -27.3 dBFS | **+28.2 dB** |
| u02 | -56.4 dBFS | -28.7 dBFS | **+27.7 dB** |

This is active-speech **RMS** vs floor **RMS**, not the peak-vs-RMS form R1.3's
DoD is phrased in. RMS-vs-RMS is the correct comparison because `mix_at_snr()`
calibrates added noise against active-speech *energy*; the peak-vs-RMS figure
runs ~18 dB optimistic and should not be used to judge calibration headroom.

**Quantified impact on the SNR axis.** Inherent SNR of ~28 dB bounds what the
`snr_db` factor can physically deliver, and the factor space asks for up to 25:

| requested | delivered | error |
|---|---|---|
| 25 dB | ~22.5 dB | -2.5 dB |
| 20 dB | ~19.0 dB | -1.0 dB |
| 15 dB | ~14.7 dB | -0.3 dB |
| <= 10 dB | — | negligible |

The axis compresses at the benign end and is clean everywhere the dead zones
live — i.e. the bias is largest exactly where the model is *not* failing.

**Open decision (park until R4, do not change mid-session):** cap `snr_db` high
at **20** instead of 25 in `DEFAULT_FACTOR_SPACE`. That holds the error to <= 1 dB
across the whole axis and is more honest than claiming a 25 dB condition the
capture chain cannot deliver. One-line change, but it touches `design.py`, which
`conditions.py` asserts against, so it needs a full test pass.

**Note:** raising the recorder's Input volume does NOT improve this. That is
preamp gain — it scales speech and room noise identically. Only speaking louder
or sitting closer moves the ratio, and either must then be held constant across
all 40 takes.

---

## Clean-condition WER floor — 2026-08-04

Corpus of 40 utterances, raw (undegraded) clips through Nova-3.
Artifacts: `results/clean_baseline.csv`, `results/clean_transcripts.jsonl`.

**Corpus WER 1.65 %** (6 errors / 363 reference words), **35/40 clips exactly
correct**. This is the floor every degraded condition is measured against and it
belongs in the write-up as a stated number.

### Normalization fixes applied first (not model errors)

The first pass scored 3.03 % (11 errors). Four of those eleven were orthographic,
not acoustic, and were removed by fixing `normalize_text()`:

| clip | reference | hypothesis | cause |
|---|---|---|---|
| u24 | `obrien` | `o'brien` | punctuation was replaced by a *space*, splitting one token into two → 1 ins + 1 sub for a perfect transcription |
| u31 | `wifi` | `wi fi` | orthographic compound; identical spoken content |

Both were **condition-independent constants**, i.e. they would have landed in
every grid cell identically — and a constant clean-condition error is
indistinguishable from a dead zone, because the model is confident and "wrong".
Fix: apostrophes are now deleted rather than split on (standard WER-scoring
convention), plus a small explicit compound map. Applied symmetrically to
reference and hypothesis, so cross-model comparability is preserved.
Regression-tested in `test_pipeline.py::test_normalization_orthographic_variants`,
which also asserts that genuine errors (`nair`/`nayar`, `gate`/`gait`) still
count — the compound map must never become a laundering mechanism.

A seventh error (u12, `toward` → `towards`) was a **speaker deviation, not a
model error**: adjudicated by ear, the speaker did say "towards". Per R1.6 Rule 2
the manifest row was corrected — both `ground_truth` *and* `say_this`, so the
prompt stays consistent with what was actually said and a future re-record
reproduces it. The reference was NOT changed to match anything the model heard;
it was changed to match the audio, verified by listening.

### The residual 5 errors are real, and three are proper nouns

| clip | ref → hyp | class |
|---|---|---|
| u04 | `nair` → `nayar` | proper noun |
| u07 | `okafor` → `okafar` | proper noun |
| u26 | `kowalski` → `koalski` | proper noun |
| u22 | `gate` → `gait` | true homophone, acoustically unresolvable |
| u15 | `one four` → `fourteen` | number aggregation |

**Three of five clean-condition errors are proper-noun spelling** — an early,
pre-degradation signal for the D2 fingerprint layer, and a reminder that the
entity failure mode exists even at 0 dB of added degradation.

**u15 is a ready-made illustration for the agent layer (R7.8):** WER counts two
errors, but an entity extractor asked for the unit number would score `fourteen`
as *correct*. That is precisely the WER-vs-task-accuracy divergence `agent_eval.py`
was built to expose.

**Constraint this imposes on recording.** The monitor/power configuration must
stay **identical across all 40 takes**. A constant floor is a fixed offset and
harmless; a floor that changes mid-corpus would be a genuine between-clip
confound.

**Goes in the write-up as:** a stated capture-chain limitation with the measured
number, the ~1 dB high-SNR bias, and the note that it is constant across the
corpus rather than varying with condition.
