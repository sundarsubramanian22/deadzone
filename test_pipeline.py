"""
Tests that the three trap functions actually do the thing. All synthetic — no
external data, no API — so you can run this the moment you clone the repo and
know the DSP core is trustworthy before wiring in real audio.
"""
import numpy as np
from audio_pipeline import (
    mix_at_snr, measured_snr_db, active_speech_mask,
    apply_rir, rms, classify_errors, normalize_text,
)

fs = 16000
rng = np.random.default_rng(0)


def make_speech_with_silence():
    """A tone burst padded with silence — mimics a real clip with dead air."""
    t = np.arange(fs) / fs                      # 1 s
    tone = 0.5 * np.sin(2 * np.pi * 200 * t)    # active speech proxy
    silence = np.zeros(fs)                       # 1 s of silence
    return np.concatenate([silence, tone, silence])  # silence dominates the file


# --- TRAP 1: SNR is calibrated on active regions, not the whole file ---------
def test_snr_calibrated_on_active_speech():
    speech = make_speech_with_silence()
    noise = rng.standard_normal(len(speech))
    for target in (0.0, 10.0, 20.0):
        mixed = mix_at_snr(speech, noise, target, fs, seed=1)
        added = mixed - speech
        got = measured_snr_db(speech, added, fs)
        assert abs(got - target) < 0.5, f"SNR off: asked {target}, got {got:.2f}"

    # the trap itself: whole-file SNR would be badly wrong because 2/3 of the
    # file is silence. Prove active-vs-wholefile power differ a lot here.
    mask = active_speech_mask(speech, fs)
    active_pow = np.mean(speech[mask] ** 2)
    whole_pow = np.mean(speech ** 2)
    assert active_pow > 2.5 * whole_pow, "silence should deflate whole-file power"
    print("TRAP 1 ok: active-region SNR within 0.5 dB across 0/10/20 dB")


# --- TRAP 2: RIR convolution preserves onset alignment and level -------------
def test_rir_alignment_and_level():
    speech = make_speech_with_silence()

    # synthetic RIR: 50 ms of leading delay, a direct-path spike, two reflections
    delay = int(0.05 * fs)
    rir = np.zeros(delay + 2000)
    rir[delay] = 1.0
    rir[delay + 400] = 0.5
    rir[delay + 900] = 0.25

    wet = apply_rir(speech, rir, fs)

    assert len(wet) == len(speech), "output length must equal input length"

    # onset preserved: the reverberant burst should start near the dry burst,
    # NOT shifted by the RIR's 50 ms leading delay.
    dry_onset = np.argmax(np.abs(speech) > 0.05)
    wet_onset = np.argmax(np.abs(wet) > 0.05 * np.max(np.abs(wet)))
    assert abs(int(dry_onset) - int(wet_onset)) < 0.02 * fs, "onset drifted >20 ms"

    # level preserved on active region (within ~1 dB)
    mask = active_speech_mask(speech, fs)
    ratio_db = 20 * np.log10(rms(wet[mask]) / rms(speech[mask]) + 1e-12)
    assert abs(ratio_db) < 1.0, f"level drifted {ratio_db:.2f} dB"
    print("TRAP 2 ok: onset within 20 ms, level within 1 dB after convolution")


# --- TRAP 3: aligned edit types, not just a WER scalar -----------------------
def test_error_classification():
    # ref vs hyp designed to contain exactly 1 sub, 1 del, 1 ins.
    # The del ('the') and ins ('next') sit far apart, so the shifted indel
    # alignment (cost 3) strictly beats the positional all-sub path (cost 6) —
    # the sub/del/ins breakdown is therefore unambiguous.
    ref = "email daniel the report before noon this friday"
    hyp = "email dan report before noon this next friday"
    #        match  SUB   DEL('the') match  match match match INS('next') match
    out = classify_errors(ref, hyp)
    c = out["counts"]
    assert c["sub"] == 1, c
    assert c["del"] == 1, c
    assert c["ins"] == 1, c
    assert out["n_ref"] == 8, out["n_ref"]
    assert abs(out["wer"] - 3 / 8) < 1e-9, out["wer"]

    # perfect match -> 0 WER, and normalization ignores case/punctuation
    perfect = classify_errors("Room 4B.", "room 4b")
    assert perfect["wer"] == 0.0
    print("TRAP 3 ok: 1 sub / 1 del / 1 ins recovered, WER=3/8, norm robust")


# --- TRAP 3b: orthographic variants must not read as acoustic errors ---------
def test_normalization_orthographic_variants():
    """
    Both cases below were measured on the REAL clean corpus, where they produced
    4 of 11 errors. They are condition-independent, so they land in every grid
    cell identically -- and a constant clean-condition error is indistinguishable
    from a dead zone (confident + "wrong"). They must score as zero.
    """
    # apostrophes are deleted, not split on: "o'brien" is ONE token
    assert normalize_text("O'Brien") == ["obrien"]
    assert normalize_text("o’brien") == ["obrien"]        # curly quote from ASR
    assert classify_errors("nguyen and obrien approved",
                           "nguyen and o'brien approved")["wer"] == 0.0

    # orthographic compounds collapse, in either direction and either spelling
    assert normalize_text("wi fi") == ["wifi"]
    assert normalize_text("Wi-Fi") == ["wifi"]
    assert classify_errors("the wifi network is falcon",
                           "the wi fi network is falcon")["wer"] == 0.0

    # ...but REAL recognition errors must survive. The compound map is not a
    # laundering mechanism: different words stay different.
    assert classify_errors("priya nair", "priya nayar")["counts"]["sub"] == 1
    assert classify_errors("the gate code", "the gait code")["counts"]["sub"] == 1
    assert classify_errors("bravo one four", "bravo fourteen")["wer"] > 0.0
    print("TRAP 3b ok: orthographic variants score 0, real errors still count")


if __name__ == "__main__":
    test_snr_calibrated_on_active_speech()
    test_rir_alignment_and_level()
    test_error_classification()
    test_normalization_orthographic_variants()
    print("\nAll three trap functions verified.")
