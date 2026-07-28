"""
conditions.py — the degradation composer (Task B3 / SPEC §4-6).

The layer BETWEEN the factor space (design.py) and the DSP core
(audio_pipeline.py). It turns raw real ingredients (clean speech + measured RIRs
+ recorded noise) into named, reproducible degradation Conditions that the grid
runs over — and composes the pipeline's trap functions in the one acoustically
correct order.

  design.py  --sample dict-->  Condition  --apply_condition-->  degraded wav
                               (this file)   (reuses audio_pipeline traps)

Design rules honored here:
  * Condition fields ARE the DEFAULT_FACTOR_SPACE factor names, imported from
    design.py — never redefined (a drift assert fires at import if they diverge).
  * Real ingredients only (SPEC §4): a Condition resolves DETERMINISTICALLY to a
    real RIR (closest measured RT60) and a real noise clip of the requested type.
    Missing asset => loud failure, never a silent substitution.
  * SNR / RIR math is REUSED from audio_pipeline (mix_at_snr, apply_rir) — not
    reimplemented (those are the tested trap functions; see CLAUDE.md).

Deps: numpy, scipy, soundfile; ffmpeg (only for the AMR/Opus codec stage).
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from design import DEFAULT_FACTOR_SPACE
from audio_pipeline import apply_rir, mix_at_snr, active_speech_mask, rms


# ---------------------------------------------------------------------------
# Bind the Condition schema to the factor space (single source of truth)
# ---------------------------------------------------------------------------

# Field order == factor-space order. If design.py's factors ever change, the
# assert below fails loudly instead of letting the schema silently drift.
_CONT = {f.name: (float(f.low), float(f.high))
         for f in DEFAULT_FACTOR_SPACE.factors if f.kind == "continuous"}
_LEVELS = {f.name: tuple(f.levels)
           for f in DEFAULT_FACTOR_SPACE.factors if f.kind == "categorical"}

# cosmetic short keys for the string name (presentation only, not a redefinition)
_ABBR = {"rt60": "rt60", "snr_db": "snr", "mic_rolloff": "roll"}


class MissingAssetError(RuntimeError):
    """Raised when a Condition cannot resolve to a required real ingredient."""


class CodecUnavailableError(RuntimeError):
    """Raised when a requested codec cannot be applied (e.g. ffmpeg missing)."""


@dataclass(frozen=True)
class Condition:
    """
    A named, fully-specified degradation — one cell of the experiment grid.
    Fields mirror DEFAULT_FACTOR_SPACE exactly (imported, not redefined).
    """
    rt60: float
    snr_db: float
    noise_type: str
    codec: str
    mic_rolloff: float

    def __post_init__(self):
        # categorical membership (levels come from design.py, not hardcoded here)
        if self.noise_type not in _LEVELS["noise_type"]:
            raise ValueError(f"noise_type {self.noise_type!r} not in {_LEVELS['noise_type']}")
        if self.codec not in _LEVELS["codec"]:
            raise ValueError(f"codec {self.codec!r} not in {_LEVELS['codec']}")

    # --- interop with design.py's wer_fn(sample: dict) ---------------------
    @classmethod
    def from_sample(cls, sample: dict) -> "Condition":
        return cls(**{k: sample[k] for k in DEFAULT_FACTOR_SPACE.names})

    def to_sample(self) -> dict:
        return asdict(self)

    # --- stable string name (results-table key + cache filename) -----------
    @property
    def name(self) -> str:
        parts = []
        for f in DEFAULT_FACTOR_SPACE.factors:
            v = getattr(self, f.name)
            if f.kind == "continuous":
                parts.append(f"{_ABBR[f.name]}-{v:g}")
            else:
                parts.append(str(v))            # bare token, e.g. babble / opus-lowrate
        return "_".join(parts)

    @classmethod
    def from_name(cls, name: str) -> "Condition":
        """Inverse of .name for clean grid values (see caveat below)."""
        tokens = name.split("_")
        factors = DEFAULT_FACTOR_SPACE.factors
        if len(tokens) != len(factors):
            raise ValueError(f"name {name!r} has {len(tokens)} tokens, expected {len(factors)}")
        kwargs = {}
        for tok, f in zip(tokens, factors):
            if f.kind == "continuous":
                prefix = _ABBR[f.name] + "-"
                if not tok.startswith(prefix):
                    raise ValueError(f"token {tok!r} missing prefix {prefix!r}")
                kwargs[f.name] = float(tok[len(prefix):])
            else:
                kwargs[f.name] = tok
        return cls(**kwargs)


# NOTE: name() uses %g, so it round-trips EXACTLY for the clean grid values we
# generate (0.5, 10, 0.3, ...). Arbitrary high-precision floats would round to
# 6 sig figs in the string — fine for filenames, but from_name is only a true
# inverse for grid-generated values.

# Drift guard: the dataclass field names MUST equal the factor-space names.
assert [fld for fld in Condition.__dataclass_fields__] == DEFAULT_FACTOR_SPACE.names, (
    "Condition fields drifted from DEFAULT_FACTOR_SPACE — keep them in lockstep"
)


# ---------------------------------------------------------------------------
# Ingredient assets — deterministic real-file lookup
# ---------------------------------------------------------------------------

@dataclass
class RirAsset:
    key: str                 # stable id (path or synthetic id)
    rt60: float              # measured RT60 (s)
    data: np.ndarray | None = None   # in-memory samples (tests); else loaded from `key`
    fs: int | None = None


@dataclass
class NoiseAsset:
    key: str
    noise_type: str
    data: np.ndarray | None = None
    fs: int | None = None


@dataclass
class ResolvedAssets:
    rir: RirAsset
    noise: NoiseAsset


def _stable_seed(s: str) -> int:
    """Deterministic, process-independent seed from a string (unlike hash())."""
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


class AssetLibrary:
    """
    A registry of real ingredients + the DETERMINISTIC selection rules:
      * RIR  : the asset whose measured RT60 is CLOSEST to the requested value
               (ties broken by sorted key — fully reproducible).
      * noise: a clip of the requested type; if several exist, a seeded pick
               keyed on the Condition NAME, so a Condition always lands on the
               same clip.
    A real subclass populates from disk (data/rirs, data/noise); tests populate
    in-memory. Selection logic lives here so both paths share it.
    """

    def __init__(self, rirs=None, noise=None):
        self.rirs: list[RirAsset] = list(rirs or [])
        self.noise: list[NoiseAsset] = list(noise or [])

    def pick_rir(self, rt60: float) -> RirAsset:
        if not self.rirs:
            raise MissingAssetError(
                "no RIR assets registered — cannot realize any reverb condition "
                "(populate data/rirs with measured RIRs)"
            )
        # closest measured RT60; deterministic tie-break by key
        return min(self.rirs, key=lambda r: (abs(r.rt60 - rt60), r.key))

    def pick_noise(self, noise_type: str, seed_str: str) -> NoiseAsset:
        matches = sorted((n for n in self.noise if n.noise_type == noise_type),
                         key=lambda n: n.key)
        if not matches:
            have = sorted({n.noise_type for n in self.noise})
            raise MissingAssetError(
                f"no noise asset of type {noise_type!r} (have: {have}) — "
                f"refusing to substitute a different noise type"
            )
        rng = np.random.default_rng(_stable_seed(seed_str))
        return matches[int(rng.integers(0, len(matches)))]

    def resolve(self, condition: Condition) -> ResolvedAssets:
        """A Condition -> its concrete ingredients (deterministic)."""
        return ResolvedAssets(
            rir=self.pick_rir(condition.rt60),
            noise=self.pick_noise(condition.noise_type, condition.name),
        )

    # subclasses override to lazily load samples from `key`
    def load_rir(self, asset: RirAsset) -> tuple[np.ndarray, int]:
        if asset.data is None:
            raise MissingAssetError(f"RIR {asset.key!r} has no data and no loader")
        return asset.data, asset.fs

    def load_noise(self, asset: NoiseAsset) -> tuple[np.ndarray, int]:
        if asset.data is None:
            raise MissingAssetError(f"noise {asset.key!r} has no data and no loader")
        return asset.data, asset.fs


# ---------------------------------------------------------------------------
# Channel stages: mic frequency-response rolloff, then codec
# ---------------------------------------------------------------------------

# mic_rolloff maps 0..1 -> low-pass cutoff. 0 = essentially full-band (clean),
# 1 = a muffled cheap-mic response. A documented proxy, not a measured mic.
_ROLLOFF_CUTOFF_MAX = 0.95     # * Nyquist, at rolloff=0 (near passthrough)
_ROLLOFF_CUTOFF_MIN_HZ = 1500  # at rolloff=1 (aggressive high-frequency loss)


def apply_mic_rolloff(x: np.ndarray, fs: int, rolloff: float) -> np.ndarray:
    """
    Low-pass proxy for a cheap-mic frequency response. rolloff in [0,1]: higher
    = lower cutoff = more high-frequency energy removed. Zero-phase (filtfilt) so
    it adds no delay (keeps onset alignment the RIR stage established).
    """
    if rolloff <= 0:
        return np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    hi_cut = _ROLLOFF_CUTOFF_MAX * nyq
    cutoff = hi_cut + (rolloff * (_ROLLOFF_CUTOFF_MIN_HZ - hi_cut))   # interp down
    cutoff = float(np.clip(cutoff, 50.0, nyq * 0.99))
    sos = butter(4, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


# ffmpeg codec params: (needs resample to codec-native rate, then back)
_CODEC_SPECS = {
    "amr": {"rate": 8000, "args": ["-c:a", "amr_nb", "-b:a", "7.4k", "-f", "amr"]},
    "opus-lowrate": {"rate": 16000, "args": ["-c:a", "libopus", "-b:a", "8k", "-f", "opus"]},
}


def apply_codec(x: np.ndarray, fs: int, codec: str) -> np.ndarray:
    """
    Encode/decode through a real codec to imprint its artifacts. 'none' is a
    passthrough. AMR-NB / low-rate Opus go through ffmpeg (round-tripping the
    audio), returning samples at the ORIGINAL fs and (trimmed/padded) length.

    Fails LOUDLY if ffmpeg or the encoder is unavailable — never silently skips
    the degradation.
    """
    if codec == "none":
        return np.asarray(x, dtype=np.float64)
    if codec not in _CODEC_SPECS:
        raise ValueError(f"unknown codec {codec!r} (expected one of {list(_LEVELS['codec'])})")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise CodecUnavailableError(
            f"codec {codec!r} requires ffmpeg, which is not on PATH. Install it "
            f"(macOS: `brew install ffmpeg`) — refusing to skip the codec stage."
        )

    import soundfile as sf                    # local import; only needed for codec
    spec = _CODEC_SPECS[codec]
    x = np.asarray(x, dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, enc, dec = td / "in.wav", td / f"enc.{codec}", td / "out.wav"
        sf.write(src, x, fs)
        try:
            # encode (resample to codec-native rate)
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(src), "-ar", str(spec["rate"]), "-ac", "1",
                            *spec["args"], str(enc)],
                           check=True, capture_output=True)
            # decode back to wav at the original fs
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(enc), "-ar", str(fs), "-ac", "1", str(dec)],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode(errors="replace")[-400:]
            raise CodecUnavailableError(
                f"ffmpeg failed to apply codec {codec!r} (encoder may not be built "
                f"into your ffmpeg). Install an ffmpeg with the encoder. Detail:\n{msg}"
            ) from e
        y, _ = sf.read(dec, dtype="float64")
    if y.ndim > 1:
        y = y[:, 0]
    return y


# ---------------------------------------------------------------------------
# THE COMPOSER — fixed, documented order
# ---------------------------------------------------------------------------

# Composition order is fixed because it mirrors the physical signal chain:
#   1. RIR (apply_rir)        reverb is imprinted on the SOURCE in the room,
#                             before the wave reaches the mic.
#   2. noise  (mix_at_snr)    ambient engine/road/babble sums at the MIC.
#   3. mic rolloff            the microphone's frequency response colours it.
#   4. codec                  the intercom's codec/bitrate degrades it LAST,
#                             along the transmission chain.
# Reordering (e.g. noise before reverb) would reverberate the noise or calibrate
# SNR against the wrong signal — acoustically wrong, so the order is not a knob.
COMPOSITION_ORDER = ("rir", "noise", "mic_rolloff", "codec")


def apply_condition(clean_audio: np.ndarray, condition: Condition,
                    assets: AssetLibrary, fs: int) -> np.ndarray:
    """
    Realize one Condition on a clean clip. Returns degraded audio the SAME length
    as the input, composing the tested trap functions in COMPOSITION_ORDER.

    Deterministic: noise selection and the SNR crop are both seeded from the
    Condition name, so re-running yields identical audio (cache-safe).
    """
    clean = np.asarray(clean_audio, dtype=np.float64)
    n = len(clean)
    resolved = assets.resolve(condition)
    seed = _stable_seed(condition.name) % (2**32)

    # 1. reverb (room) — reuse the tested trap; onset-aligned + level-preserved
    rir, rir_fs = assets.load_rir(resolved.rir)
    if rir_fs is not None and rir_fs != fs:
        raise MissingAssetError(
            f"RIR {resolved.rir.key!r} is {rir_fs} Hz but audio is {fs} Hz; "
            f"resample the asset library to a single rate before composing"
        )
    y = apply_rir(clean, rir, fs)

    # 2. noise at the mic — reuse the tested SNR trap (active-speech calibrated)
    noise, noise_fs = assets.load_noise(resolved.noise)
    if noise_fs is not None and noise_fs != fs:
        raise MissingAssetError(
            f"noise {resolved.noise.key!r} is {noise_fs} Hz but audio is {fs} Hz"
        )
    y = mix_at_snr(y, noise, condition.snr_db, fs, seed=seed)

    # 3. mic frequency response
    y = apply_mic_rolloff(y, fs, condition.mic_rolloff)

    # 4. transmission codec (may resample internally); restore exact length
    y = apply_codec(y, fs, condition.codec)
    if len(y) != n:
        y = y[:n] if len(y) > n else np.pad(y, (0, n - len(y)))
    return y
