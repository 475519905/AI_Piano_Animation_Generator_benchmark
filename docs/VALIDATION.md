# v0.1.0 verification

Date: 2026-11-09. This is verification of the **evaluation implementation**, not a model benchmark.

## Local checks

- 57 unit/integration tests passed on Python 3.9.6 / NumPy 2.0.2.
- CLI smoke tests evaluated identical, floating, jitter and wrong-hand synthetic variants.
- The synthetic split-declaration manifest passed both piece/performance and optional performer separation checks.
- An installable wheel was built with an isolated build backend, installed into a separate virtual environment,
  and its `piano-eval` command ran from outside the source directory.
- Input/report schema, overwrite protection, non-finite inputs, pickle refusal, explicit missingness,
  quaternion sign equivalence, unchanged input arrays, per-second derivative units, reference-assigned contact
  and aggregation arithmetic are covered by tests.

## Expected negative-control behavior

| Synthetic condition | Expected observation |
|---|---|
| Identical arrays | Zero reference pose error, 100% reference-assigned contact in eligible samples; **no overall pass/quality claim** |
| Floating 10 mm | Target-key surface contact falls to 0 under the example's 2 mm tolerance |
| Alternating 4 mm jitter | Derivative jerk strongly increases relative to the same reference trajectory |
| Wrong-hand label | Fingering/hand label comparison exposes the error even if the unchanged geometric trajectory still matches |
| Static raised fingertip | Low jerk does not hide missing target-key contact |
| Missing collision audit | `not_measured`, never “0 collisions” |

The fixture is an invented arithmetic trajectory, not a song or a plausible pianist animation. Its numeric scores
must not be cited as AiPiano's real-model performance. No real recordings, model checkpoints, proprietary assets
or private reports were used in these tests.

## CI and reproducibility

The repository includes a read-only GitHub Actions workflow for Python 3.9 / 3.12 with NumPy 2.0.2.
Check the actual Actions run for its status; local success is not automatically claimed as CI success.
Action references are pinned to commit IDs. Tests and the scorer require no secrets or online model service.

Not validated here: Blender/C4D export adapters, real model accuracy, live rig/skin geometry, human ratings,
all future dependency versions, or end-to-end deployment performance.
