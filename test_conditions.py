"""
Offline tests for the degradation composer (Task B3). Synthetic audio, synthetic
RIRs/noise held in memory — no real assets, no ffmpeg, no API. Run:

    python3 test_conditions.py

Asserts:
  * Condition fields ARE the design.py factor names (no redefinition/drift),
  * the string name is stable and round-trips (incl. the hyphenated codec),
  * apply_condition composes in the FIXED order rir->noise->mic_rolloff->codec
    (proven by exact reconstruction; a swapped order differs),
  * a reverb tail is laid down (present in formerly-silent regions),
  * output length == input length,
  * a Condition resolves DETERMINISTICALLY to the same ingredients + same audio,
  * a missing asset RAISES (no silent substitution),
  * an unavailable codec fails LOUDLY rather than skipping.
"""
from unittest import mock

import numpy as np

from design import DEFAULT_FACTOR_SPACE
from audio_pipeline import apply_rir, mix_at_snr
import conditions as C
from conditions import (
    Condition, AssetLibrary, RirAsset, NoiseAsset,
    apply_condition, apply_mic_rolloff, apply_codec,
    MissingAssetError, CodecUnavailableError, _stable_seed,
)

FS = 16000


# --- synthetic ingredients ---------------------------------------------------

def make_clean():
    """1 s silence | 1 s tone | 1 s silence — trailing silence reveals reverb."""
    t = np.arange(FS) / FS
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    sil = np.zeros(FS)
    return np.concatenate([sil, tone, sil])


def make_rir(rt60, seed=0):
    rng = np.random.default_rng(seed)
    L = max(8, int(rt60 * FS))
    decay = np.exp(-6.9 * (np.arange(L) / FS) / rt60)     # ~-60 dB over rt60
    h = rng.standard_normal(L) * decay
    h[0] = 1.0                                            # direct path
    return h


def make_library():
    rirs = [RirAsset(key=f"rir_{rt:.1f}", rt60=rt, data=make_rir(rt, seed=i), fs=FS)
            for i, rt in enumerate((0.2, 0.5, 0.9))]
    rng = np.random.default_rng(7)
    noise = [
        NoiseAsset("babble_a", "babble", rng.standard_normal(FS * 2), FS),
        NoiseAsset("babble_b", "babble", rng.standard_normal(FS * 2), FS),
        NoiseAsset("engine_a", "engine", rng.standard_normal(FS * 2), FS),
        # note: NO 'road' noise on purpose -> exercises missing-asset path
    ]
    return AssetLibrary(rirs=rirs, noise=noise)


# --- 1: schema is bound to the factor space ---------------------------------

def test_schema_matches_factor_space():
    assert list(Condition.__dataclass_fields__) == DEFAULT_FACTOR_SPACE.names
    # from_sample/to_sample bridges design.py samples <-> Condition
    sample = {"rt60": 0.5, "snr_db": 10.0, "noise_type": "babble",
              "codec": "amr", "mic_rolloff": 0.3}
    cond = Condition.from_sample(sample)
    assert cond.to_sample() == sample
    print("OK 1: Condition fields == DEFAULT_FACTOR_SPACE.names (imported, not redefined)")


# --- 2: stable, round-tripping name -----------------------------------------

def test_name_stable_and_roundtrips():
    cond = Condition(0.5, 10.0, "babble", "amr", 0.3)
    assert cond.name == "rt60-0.5_snr-10_babble_amr_roll-0.3", cond.name
    assert Condition.from_name(cond.name) == cond
    assert cond.name == Condition(0.5, 10.0, "babble", "amr", 0.3).name    # stable
    # hyphen in the codec value must survive the round-trip
    hy = Condition(0.5, 10.0, "babble", "opus-lowrate", 0.3)
    assert hy.name == "rt60-0.5_snr-10_babble_opus-lowrate_roll-0.3", hy.name
    assert Condition.from_name(hy.name) == hy
    print(f"OK 2: name stable + round-trips (e.g. {cond.name!r})")


# --- 3: fixed composition order (rir -> noise -> rolloff -> codec) -----------

def test_composition_order():
    lib = make_library()
    clean = make_clean()
    cond = Condition(0.9, 20.0, "babble", "none", 0.4)

    out = apply_condition(clean, cond, lib, FS)

    # reconstruct the documented chain with the same resolved assets + seed
    res = lib.resolve(cond)
    rir, _ = lib.load_rir(res.rir)
    noise, _ = lib.load_noise(res.noise)
    seed = _stable_seed(cond.name) % (2**32)
    expected = apply_rir(clean, rir, FS)
    expected = mix_at_snr(expected, noise, cond.snr_db, FS, seed=seed)
    expected = apply_mic_rolloff(expected, FS, cond.mic_rolloff)   # codec 'none' == passthrough
    assert np.allclose(out, expected, atol=1e-9), "apply_condition != documented order"

    # a SWAPPED order (noise before reverb) must differ -> order genuinely matters
    wrong = mix_at_snr(clean, noise, cond.snr_db, FS, seed=seed)
    wrong = apply_rir(wrong, rir, FS)
    assert not np.allclose(out, apply_mic_rolloff(wrong, FS, cond.mic_rolloff), atol=1e-6)

    # reverb tail is laid down: formerly-silent tail now carries decaying energy
    tail = out[2 * FS:]                       # last second was silence in `clean`
    early = np.mean(tail[:FS // 4] ** 2)
    late = np.mean(tail[-FS // 4:] ** 2)
    assert early > late, "reverb tail should decay across the trailing silence"
    print("OK 3: fixed order rir->noise->rolloff->codec (reconstruction matches; swap differs; tail decays)")


# --- 4: length preserved -----------------------------------------------------

def test_length_preserved():
    lib = make_library()
    clean = make_clean()
    for cond in (Condition(0.2, 5.0, "engine", "none", 0.0),
                 Condition(0.5, 15.0, "babble", "none", 0.7)):
        out = apply_condition(clean, cond, lib, FS)
        assert len(out) == len(clean), (cond.name, len(out), len(clean))
    print("OK 4: apply_condition output length == input length")


# --- 5: deterministic ingredient + audio resolution -------------------------

def test_deterministic_resolution():
    lib = make_library()
    clean = make_clean()
    cond = Condition(0.55, 12.0, "babble", "none", 0.2)

    r1, r2 = lib.resolve(cond), lib.resolve(cond)
    assert (r1.rir.key, r1.noise.key) == (r2.rir.key, r2.noise.key)
    # closest-RT60 rule: 0.55 is nearest the 0.5 RIR
    assert r1.rir.key == "rir_0.5", r1.rir.key
    # same Condition -> byte-identical audio
    a1 = apply_condition(clean, cond, lib, FS)
    a2 = apply_condition(clean, cond, lib, FS)
    assert np.array_equal(a1, a2), "same Condition must yield identical audio"
    print(f"OK 5: deterministic — resolves to ({r1.rir.key}, {r1.noise.key}) and identical audio")


# --- 6: missing asset raises (never silently substitutes) -------------------

def test_missing_asset_raises():
    lib = make_library()
    clean = make_clean()

    # 'road' noise does not exist in the library
    road = Condition(0.5, 10.0, "road", "none", 0.1)
    try:
        apply_condition(clean, road, lib, FS)
        assert False, "expected MissingAssetError for absent noise type"
    except MissingAssetError as e:
        assert "road" in str(e)

    # no RIRs at all
    empty = AssetLibrary(rirs=[], noise=lib.noise)
    try:
        apply_condition(clean, Condition(0.5, 10.0, "babble", "none", 0.1), empty, FS)
        assert False, "expected MissingAssetError for absent RIR"
    except MissingAssetError:
        pass
    print("OK 6: missing noise/RIR raises MissingAssetError (no silent substitution)")


# --- 7: unavailable codec fails loudly --------------------------------------

def test_codec_fails_loudly_without_ffmpeg():
    x = np.random.default_rng(0).standard_normal(FS)
    # force the "no ffmpeg" branch regardless of the host environment
    with mock.patch("conditions.shutil.which", return_value=None):
        try:
            apply_codec(x, FS, "amr")
            assert False, "expected CodecUnavailableError when ffmpeg is missing"
        except CodecUnavailableError as e:
            assert "ffmpeg" in str(e).lower() and "install" in str(e).lower()
        # and the failure must surface through apply_condition, not be swallowed
        lib = make_library()
        cond = Condition(0.5, 10.0, "babble", "amr", 0.0)
        try:
            apply_condition(make_clean(), cond, lib, FS)
            assert False, "expected apply_condition to surface CodecUnavailableError"
        except CodecUnavailableError:
            pass
    assert apply_codec(x, FS, "none") is not None    # 'none' still works
    print("OK 7: AMR/Opus without ffmpeg -> loud CodecUnavailableError (tells you to install)")


if __name__ == "__main__":
    test_schema_matches_factor_space()
    test_name_stable_and_roundtrips()
    test_composition_order()
    test_length_preserved()
    test_deterministic_resolution()
    test_missing_asset_raises()
    test_codec_fails_loudly_without_ffmpeg()
    print("\nAll condition tests passed — named, reproducible degradations compose "
          "the trap functions in the one correct order.")
