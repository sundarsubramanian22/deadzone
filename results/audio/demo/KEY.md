# KEY — presenter only. Do not put this in `blind/`.

`blind/` is the handover folder: neutral names, byte-identical copies,
and `blind/BLIND_SHEET.md`. Everything that names a condition lives here
in the parent directory.

## Blind name -> working file

| blind | working file | condition |
|---|---|---|
| `blind_01.wav` | `pair2_B_babble_u21.wav` | B · babble-dominated · `rt60-0.2_snr-0_babble_none_roll-0` |
| `blind_02.wav` | `payoff_u03_clean.wav` | control · raw recording, 16 kHz |
| `blind_03.wav` | `pair1_A_reverb_u40.wav` | A · reverb-dominated · `rt60-1_snr-20_babble_none_roll-0` |
| `blind_04.wav` | `pair3_B_babble_u26.wav` | B · babble-dominated · `rt60-0.2_snr-0_babble_none_roll-0` |
| `blind_05.wav` | `payoff_u03_deadzone.wav` | payoff · `rt60-0.7_snr-20_babble_opus-lowrate_roll-1` |
| `blind_06.wav` | `pair2_A_reverb_u21.wav` | A · reverb-dominated · `rt60-1_snr-20_babble_none_roll-0` |
| `blind_07.wav` | `pair1_B_babble_u40.wav` | B · babble-dominated · `rt60-0.2_snr-0_babble_none_roll-0` |
| `blind_08.wav` | `pair3_A_reverb_u26.wav` | A · reverb-dominated · `rt60-1_snr-20_babble_none_roll-0` |

## The two conditions

| | A — reverb-dominated | B — babble-dominated |
|---|---|---|
| condition | `rt60-1_snr-20_babble_none_roll-0` | `rt60-0.2_snr-0_babble_none_roll-0` |
| requested rt60 | 1.0 s | 0.2 s |
| SNR | 20.0 dB (quiet) | 0.0 dB (buried) |
| noise | babble | babble |
| codec / rolloff | none / 0 | none / 0 |
| delivered room | Shower | Restaurant |
| RIR file | `mit_rt60-0.99_h081_Shower_2txts.wav` | `mit_rt60-0.20_h114_Restaurant_txts.wav` |
| measured RT60 | 1.011 s | 0.193 s |
| DRR | -10.02 dB | +16.90 dB |
| C50 | +2.12 dB | +28.10 dB |
| mean WER over 40 clips | 0.1123 | 0.1301 |

Each isolates ONE degradation: codec `none`, mic rolloff 0. Nothing in
this comparison is confounded by the channel factors.

**Paired difference (A-B): -0.0178 WER, 95% CI [-0.0654, +0.0310]** — spans zero. 10,000-resample paired bootstrap, seed 0, resampled over clips.

## Per-clip facts (all from `results/master.csv`)

### Pair 1 — `u40` (primary)

- reference: `ask yamamoto to sign page twelve before we file`
- **A** blind_03.wav — WER **0.333**, mean conf 0.805
  - hyp: `ask yavamoto to sign page twelve before v five`
- **B** blind_07.wav — WER **0.333**, mean conf 0.807
  - hyp: `ask yamamoto to sign page twelve in world five`

### Pair 2 — `u21` (primary)

- reference: `forward the file to accounting and legal by monday`
- **A** blind_06.wav — WER **0.222**, mean conf 0.879
  - hyp: `forward the file to accounting and legal filing`
- **B** blind_01.wav — WER **0.222**, mean conf 0.854
  - hyp: `forward the file to accounting and legal`

### Pair 3 — `u26` (backup)

- reference: `text me the address for the kowalski wedding`
- **A** blind_08.wav — WER **0.250**, mean conf 0.864
  - hyp: `text me the address for the cool seaway`
- **B** blind_04.wav — WER **0.250**, mean conf 0.883
  - hyp: `text me the address for the`

## Payoff

- condition `rt60-0.7_snr-20_babble_opus-lowrate_roll-1` — SNR 20.0 dB, i.e. QUIET. The damage is reverb + codec + rolloff, not noise.
- clean: `blind_02.wav` · dead zone: `blind_05.wav`
- `u03` reference (11 words): `the meeting moved to room four b on the third floor`
- transcript: **(empty string)**
- WER 1.000, all 11 reference words deleted, utterance confidence 0.00
- **10 of 40 clips returned an empty transcript in this condition**: u03, u04, u16, u18, u20, u21, u24, u32, u36, u40

Mean word confidence is NULL here, not low: there are no words to be
confident about. That is the deletion blindness the calibration layer
reports — deletions carry no hypothesis token, so they are invisible to
any confidence-based monitor.

