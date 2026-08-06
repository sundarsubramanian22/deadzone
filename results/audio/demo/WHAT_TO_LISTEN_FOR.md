# What to listen for

Two different jobs live in this directory. Don't mix them up.

---

## PART 1 — `isolation/`: is the composer physically plausible? (SPEC A.R3.5)

All from clip `u02`, so what changes is the CONDITION, not the speaker
or the sentence. Start with `00_RAW_original.wav`. **WER is irrelevant here** —
this is a DSP check, and the only instrument is your ears. The unit tests prove
the arithmetic; nothing but listening proves the RESULT.

1. **Onset alignment.** Every file must start when the original starts. A late
   start means `apply_rir`'s direct-path trim is wrong and every WER in the
   study carries a pure alignment artifact.
2. **Reverb sounds like a room** (`02_reverb_only`) — not a delay, not a metallic
   comb, no audible repeat.
3. **SNR is believable.** `01_benign` (20 dB) should be barely noisy;
   `03_noise_only` (0 dB) should nearly bury the speech. If 0 dB sounds mild, the
   calibration is off.
4. **Codecs sound like a phone line** (`05_g726_only`, `06_opus_only`) —
   bandlimited and gritty, not just quieter.

This is the check that caught the `apply_rir` renormalization bug: reverb tail
energy leaking into the silent regions de-calibrated every downstream SNR, and
it produced clean-looking garbage with no error message anywhere.

---

## PART 2 — `blind/`: the human axis the study doesn't have

Run this on someone else, not on yourself. `DEMO_SCRIPT.md` is the run-of-show,
`KEY.md` is the answer key, `blind/BLIND_SHEET.md` is the only page the listener
sees. Do not let them see the working filenames in this directory — they say
`reverb` and `babble`.

What you are listening for yourself, before you run it on anyone:

- **A** (`rt60-1_snr-20_babble_none_roll-0`) should sound like a **bad room, but quiet**: the
  Shower impulse response, measured RT60
  1.011 s, DRR -10.02 dB, with the babble at
  20.0 dB SNR and barely there.
- **B** (`rt60-0.2_snr-0_babble_none_roll-0`) should sound like a **good room with the speech nearly
  buried**: the Restaurant response, measured RT60
  0.193 s, DRR +16.90 dB, babble at
  0.0 dB.
- If those two do NOT sound clearly different to you, stop: something is wrong
  with the composer, because the model says they are equally damaging and the
  whole exercise is that a human should disagree.
- **The payoff** (`blind_05.wav`, condition
  `rt60-0.7_snr-20_babble_opus-lowrate_roll-1`) should still sound like a person
  speaking. The model returned an empty string on it — and on
  10 of 40 clips in that condition.

The clips are deliberately **not level-matched**: they are byte-identical to the
audio nova-3 was scored on, and a cosmetic gain would break that identity. Use
the volume knob and say so out loud when you run the exercise.

---

## SUPERSEDED

`results/audio/listen/DEADZONE_*.wav` are superseded by this directory: all six were
clip `u02`, which nova-3 transcribes at WER 0.000 in four of those six
conditions — so they demonstrated nothing. The ladder in that directory is fine
and is regenerated here as `isolation/`.
