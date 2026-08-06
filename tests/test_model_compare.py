"""
Offline validation for Layer 1 (model_compare.py). No API, no audio — synthetic
per-model results tables with a KNOWN planted difference stand in for real grid
runs, and we assert the comparison surfaces it.

Planted structure (two models over the same rt60 x snr grid):
  * shared base WER rising with reverb and falling with SNR,
  * model "weak" gets a big WER BUMP for rt60 > 0.7 while staying CONFIDENT there
    (a dead zone: confidently wrong); "strong" has no bump and its confidence
    correctly DROPS as conditions worsen,
  * the two models live on DIFFERENT confidence scales — every "weak" confidence
    (<=0.70) is below every "strong" confidence (>=0.85). A naive raw-confidence
    comparison would call weak "never confident" and MISS its dead zone; only the
    within-model normalization catches it. That is the whole point of Layer 1.

Asserted: registry has three one-line arms; the comparison flags the rt60>0.7
region with weak as the worse model; weak has a dead zone there and strong does
not (despite weak's lower absolute confidence); strong's confidence tracks its
errors (negative shape) and weak's does not.

Deterministic (pure functions of the grid, no rng).  Run:
    python3 test_model_compare.py
"""
import numpy as np

from design import DEFAULT_FACTOR_SPACE
from model_compare import (
    MODEL_REGISTRY, get_model, compare_models, dead_zone_flags,
    within_model_conf_percentile, confidence_wer_shape,
)
from audio_pipeline import _parse_vosk_result

SPACE = DEFAULT_FACTOR_SPACE
RT60 = np.linspace(0.2, 1.0, 9)
SNR = np.linspace(0.0, 25.0, 5)


def _base_wer(rt60, snr):
    rn, qn = (rt60 - 0.2) / 0.8, snr / 25.0
    return 0.05 + 0.2 * rn + 0.3 * (1 - qn)


def _row(rt60, snr, wer, conf):
    return {"rt60": float(rt60), "snr_db": float(snr), "noise_type": "babble",
            "codec": "none", "mic_rolloff": 0.0,
            "wer": float(min(max(wer, 0.0), 1.0)), "mean_conf": float(conf)}


def build_tables():
    strong, weak = [], []
    for rt60 in RT60:
        rn = (rt60 - 0.2) / 0.8
        for snr in SNR:
            qn = snr / 25.0
            base = _base_wer(rt60, snr)
            # strong: no bump; confidence high-scale and tracks cleanliness
            strong.append(_row(rt60, snr, base,
                               conf=min(max(0.90 + 0.09 * qn - 0.05 * rn, 0), 1)))
            # weak: big WER bump past rt60=0.7 but STAYS confident there (its own
            # top). Elsewhere its confidence is low-scale (~0.45-0.55).
            bump = 0.5 if rt60 > 0.7 else 0.0
            wconf = 0.70 if rt60 > 0.7 else (0.45 + 0.10 * qn)
            weak.append(_row(rt60, snr, base + bump, conf=wconf))
    return {"strong": strong, "weak": weak}


# ---- 1: registry is three one-line arms, resolvable but not called ----------

def test_registry():
    assert set(MODEL_REGISTRY) == {"nova-3", "whisper-base", "vosk"}, MODEL_REGISTRY
    assert len(MODEL_REGISTRY) == 3
    for name in MODEL_REGISTRY:
        assert callable(get_model(name)), name        # resolvable, NOT invoked
    print(f"OK 1: MODEL_REGISTRY = {tuple(MODEL_REGISTRY)} (each a one-line arm)")


# ---- 2: cross-scale confidence handled WITHIN each model --------------------

def test_within_model_normalization():
    tables = build_tables()
    weak_max = max(r["mean_conf"] for r in tables["weak"])
    strong_min = min(r["mean_conf"] for r in tables["strong"])
    # the scales genuinely don't overlap -> raw cross-model comparison is invalid
    assert weak_max < strong_min, (weak_max, strong_min)
    # yet within-model percentile still puts weak's confident rows at the top
    # (region rows tie at percentile ~0.84; non-region tops out ~0.60)
    pct = within_model_conf_percentile(tables["weak"])
    top_rows = [tables["weak"][i] for i in np.where(pct >= 0.7)[0]]
    assert top_rows and all(r["rt60"] > 0.7 for r in top_rows), "weak's top-confidence rows"
    print(f"OK 2: weak conf<= {weak_max:.2f} < strong conf>= {strong_min:.2f}; "
          f"within-model percentile still isolates weak's confident region")


# ---- 3: the comparison surfaces the planted weak region --------------------

def test_comparison_surfaces_weak_region():
    tables = build_tables()
    res = compare_models(tables, SPACE, n_bins=4, wer_hi=0.3, conf_pct_hi=0.6)

    top = res["divergence_regions"][0]
    assert top["factor"] == "rt60", top
    assert top["worse_model"] == "weak", top
    assert top["better_model"] == "strong", top
    lo, hi = top["span"]
    assert lo >= 0.7, ("divergence should be at the high-reverb end", top["span"])
    assert top["wer_gap"] > 0.3, top["wer_gap"]

    dz = top["dead_zone_rate_by_model"]
    assert dz["weak"] > 0.8 and dz["strong"] == 0.0, dz    # weak silent-fails here
    print(f"OK 3: top divergence = {top['factor']} {top['span']}, worse=weak "
          f"(WER gap {top['wer_gap']:.2f}); dead-zone rate weak={dz['weak']:.2f} "
          f"strong={dz['strong']:.2f}")
    return res


# ---- 4: dead-zone rate + confidence-shape distinguish the models -----------

def test_dead_zone_and_shape():
    tables = build_tables()
    res = compare_models(tables, SPACE, n_bins=4)
    pm = res["per_model"]

    # weak silently fails overall; strong (confidence tracks error) does not
    assert pm["weak"]["dead_zone_rate"] > 0.0, pm["weak"]
    assert pm["strong"]["dead_zone_rate"] == 0.0, pm["strong"]

    # shape: strong's confidence tracks its WER (negative); weak's fails to
    s_strong = pm["strong"]["shape"]["spearman"]
    s_weak = pm["weak"]["shape"]["spearman"]
    assert s_strong < 0, s_strong
    assert s_weak > s_strong, (s_weak, s_strong)      # weak is less self-aware
    print(f"OK 4: dead-zone rate weak={pm['weak']['dead_zone_rate']:.2f} vs "
          f"strong={pm['strong']['dead_zone_rate']:.2f}; conf-vs-WER shape "
          f"strong rho={s_strong:.2f} < weak rho={s_weak:.2f}")


# ---- 5: the third adapter arm parses to the shared contract (offline) ------

def test_vosk_parses_to_contract():
    canned = {"text": "call maria at four zero five",
              "result": [{"word": "call",  "conf": 0.98, "start": 0.1, "end": 0.3},
                         {"word": "maria", "conf": 0.91, "start": 0.3, "end": 0.9},
                         {"word": "at",    "conf": 0.88, "start": 0.9, "end": 1.0},
                         {"word": "four",  "conf": 0.95, "start": 1.0, "end": 1.2},
                         {"word": "zero",  "conf": 0.93, "start": 1.2, "end": 1.4},
                         {"word": "five",  "conf": 0.90, "start": 1.4, "end": 1.7}]}
    out = _parse_vosk_result(canned)
    assert set(out) == {"transcript", "word_confidences", "mean_conf", "utterance_conf"}
    assert out["transcript"] == "call maria at four zero five"
    assert len(out["word_confidences"]) == 6
    assert abs(out["mean_conf"] - float(np.mean([0.98, 0.91, 0.88, 0.95, 0.93, 0.90]))) < 1e-9
    assert out["utterance_conf"] == out["mean_conf"]     # Vosk has no separate one
    # honest empty case: no words -> no fabricated confidences
    empty = _parse_vosk_result({"text": ""})
    assert empty["word_confidences"] == [] and np.isnan(empty["mean_conf"])
    print("OK 5: Vosk arm parses to the shared contract (per-word conf surfaced, "
          "no fabrication on empty)")


if __name__ == "__main__":
    test_registry()
    test_within_model_normalization()
    test_comparison_surfaces_weak_region()
    test_dead_zone_and_shape()
    test_vosk_parses_to_contract()
    print("\nAll model-compare tests passed — cross-model dead-zone comparison "
          "works within-model (scale-free) and surfaces the planted weak region.")
