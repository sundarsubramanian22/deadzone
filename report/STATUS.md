# STATUS — 2026-08-06

What is done, what is not, and what a next session should pick up. Written to be
read cold. Numbers here are pointers; the artifacts in `results/` are the source.

## Done

| piece | state |
|---|---|
| Corpus (40 clips, ground truth) | done — clean WER 1.65 %, every error adjudicated by ear |
| Grid, 3 arms | **10,560 rows**, 3 failures (0.03 %) — nova-3 (40 clips × 176 conditions), whisper-base and elevenlabs-scribe (10-clip subset × 176) |
| D1 silent-failure map | done, and **corrected** — the estimand mismatch (SPEC Appendix G) cut dead zones 6 → 2 and moved ρ to −0.980 paired |
| D2 fingerprints | done — deletions dominate; entity error 0.633 vs WER 0.511 |
| Sensitivity | done, **exact** functional ANOVA on the complete 4×4×3×3 factorial; pre-registration CONFIRMED |
| D3b active learning | done — a **null**, and it survived its own obvious fix (Appendix H) |
| D4 sim2real | done — 12.1 pts optimistic, ρ 0.873, dead-zone Jaccard 0.00 |
| L1 three-arm comparison | done — Scribe is **rank-only**, enforced in code |
| L2 calibration | done — ECE 0.0507 → 0.0346 → 0.0077 |
| L3 decoupling | done — two decoupled verdicts, the paralinguistic stream leads |
| Confidence characterization | done — clean baseline 0.962, mean beats min (AUROC 0.944 vs 0.877) |
| Dashboard | 8 panels, offline, `file://` |
| Demo kit | `make demo` (offline) + `make demo-listen` (interactive) + `make demo-live` (optional, needs key) |
| Write-up | `report/writeup.md`, ~14 min; `report/UNDERSTANDING.md` is the plain-English exposure map |
| Tests | 27 suites |

## Not done, and why

1. **`grid-v1` sits at `c321715`**, well behind HEAD. It was held because the
   write-up, dashboard and demo script had to agree first (SPEC F.2, J.7). They
   now largely do; moving the tag is a one-line job for a clean tree.
2. **`results/MANIFEST.json`** was last regenerated at `bfd2d78`. Re-run
   `scripts/make_manifest.py` on the committed tree before tagging.
3. **`report/measurements.md`** is the only prose in `report/` not covered by
   `tests/test_report_numbers.py`, and it has drifted on ~5 figures.
4. **§6.7's three-arm bootstrap CIs exist only as prose** — no generator, no
   artifact. They were computed once and typed in. Either write the generator or
   mark them as such.
5. **The Scribe repeat-call probe has no artifact on disk** (SPEC I.9). The
   non-determinism finding rests on a 6-clip × 4-call measurement recorded in a
   log. `--repeat N` writing `results/scribe_repeat.json` would fix it for ~$0.005.
6. **`dashboard/DEMO.md` and the narration strings in `demos/` are not pinned**
   by any test. `tests/test_report_numbers.py` covers `report/` only. This is the
   surface that drifted twice already.
7. **Bare NaN in three JSON artifacts** (`fingerprints`, `l3_decoupling`,
   `sim2real`) — invalid JSON to a strict parser. Python's `json` accepts it, so
   nothing in-repo notices.
8. **R7, the live voice agent, is deliberately out of scope.** `agent_eval.py` is
   a synthetic-validated scaffold and is presented as one.

## The honest weak spots (fuller version in `report/UNDERSTANDING.md`)

- **The DRR finding rests on 4 rooms.** ρ = −1.000 across four points has no
  interval, and C50 at −0.800 is one adjacent swap behind. It is a strong
  *hypothesis* and the write-up should not read as more.
- **The dead-zone count of 2 is one draw.** Over a defensible threshold box it
  ranges 0–86; only ~5 % of grid points return 2, and the robust core is empty.
  The *persistence* curve is the real statistic, not the scalar.
- **One speaker, one accent, 40 utterances.** Nothing here generalizes across
  speakers.
- **No arm is streaming.** All three are batch. The framing says "streaming-capable
  model", not "measured in streaming mode".
- **The active-learning null is a null**, reproduced across 44 parameterisations.
- **The human half of the listening beat is n = 1 and unblinded.** Intuition pump,
  never data.
