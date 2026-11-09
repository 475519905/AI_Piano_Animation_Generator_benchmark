# Data contract v1

The CLI reads **prediction.npz**, **reference.npz**, and **metadata.json**. NPZ loading uses
`allow_pickle=False`; object arrays are rejected. There is no model/checkpoint loading.
Unknown array or metadata keys are rejected to catch typos and accidental service-config input.

## Canonical conventions

- Positions: meters, keyboard-local **right-handed** coordinates: X toward higher notes/right,
  Y away from the seated pianist, Z upward. Freeze the instrument calibration before evaluation.
- Time: seconds in a shared clock. Uniform sampling at explicit `fps`. No automatic offset correction,
  interpolation, crop, root alignment, scale fit or dynamic time warping.
- Finger slots and labels: right thumb/index/middle/ring/little = 0..4, left thumb..little = 5..9.
- Each sample is one complete performance under the declared protocol, not a conveniently selected clip.
- T = frames, J = chosen matched points, B = matched rotations, N = aligned note events, K = keys.
  Key index order must be fixed by your adapter and reference generation; the scorer does not infer MIDI pitch mapping.

## Arrays

| Key | Prediction | Reference | Type / shape |
|---|---|---|---|
| `time_s` | required | required | finite numeric `(T,)`, nonempty, same values |
| `points_m` | optional | optional | finite numeric `(T,J,3)` |
| `point_ids` | required when point error is measured | same | unique nonempty Unicode strings `(J,)`, same order |
| `rotations_xyzw` | optional | optional | finite numeric `(T,B,4)`, nonzero quaternion |
| `rotation_ids` | required with paired rotations | same | Unicode strings `(B,)`, same order |
| `finger_labels` | optional | optional | integer `(N,)`; prediction -1..9, reference 0..9 |
| `event_start_s` | optional | optional | finite numeric `(N,)`, animation event times vs reference MIDI event times |
| `event_ids` | required for fingering/timing | same | unique Unicode strings `(N,)`, same order |
| `fingertips_m` | optional | optional, not used as a target trajectory by this key | finite numeric `(T,10,3)` |
| `contact_mask` | not allowed | optional | **boolean** `(T,10)`, independently derived expected contact/hold intervals |
| `contact_key` | not allowed | optional | integer `(T,10)`, -1 inactive, 0..K-1 active |
| `key_bounds_m` | not allowed | optional | numeric `(K,2,3)` or `(T,K,2,3)`, min corner then max corner |

Include fingertip positions in `points_m` with explicit IDs if you want their reference position errors.
`fingertips_m` is used for descriptive derivatives and reference-assigned geometric contact measurements.
Frames with no eligible reference contacts yield `not_measured`, not a perfect contact score.
Point, rotation and event modalities without a reference/prediction pair are explicitly unmeasured.

Static key boxes are only an approximation. If the key moves, export per-frame bounds.
A tilted or nonrectangular key/skin cannot be faithfully represented by these boxes; do not label the
result full mesh collision. Reference contact masks must not be copied from the tested model's own output.

## Metadata

Required fields are shown by `examples/generate_synthetic.py`:

```json
{
  "schema_version": 1,
  "sample_id": "anonymous-sample-001",
  "piece_group": "work-and-all-arrangements-001",
  "performance_id": "recording-001",
  "coordinate_calibration_id": "frozen-keyboard-calibration-v1",
  "postprocessing": "none",
  "position_unit": "m",
  "coordinate_frame": "keyboard_x_right_y_away_z_up",
  "finger_order": "right_thumb_to_little_then_left_thumb_to_little",
  "split": "test",
  "stage": "final_evaluated",
  "conditioning": "predicted_fingering",
  "sampling_unit": "full_performance",
  "fps": 30,
  "contact_tolerance_mm": 2.0
}
```

- `stage`: `raw_prediction` or `final_evaluated`.
- `conditioning`: `predicted_fingering`, `reference_fingering`, or `not_applicable`.
- `contact_tolerance_mm` is mandatory **when measuring contact**, chosen and frozen by the study.
  The example's 2 mm is arbitrary, not a recommended universal standard.
- `postprocessing` should identify a fixed version/config; use separate reports for different variants.
- Use anonymized IDs in public metadata. Never add service credentials or personal paths.

Reports include input SHA-256, evaluator-source SHA-256, Python/NumPy versions, counts and explicit missingness.
Input hashes are checked before and after evaluation; changing an input during evaluation is an error.
Hashes identify bytes; they do not establish data rights, absence of leakage or scientific validity.

## Batch aggregation

```sh
python -m piano_eval aggregate report-a.json report-b.json --output aggregate.json
```

Do not aggregate negative-control variants as if they were different performances from a benchmark.
The generator deliberately uses the same `performance_id` for variants, so pooling them is rejected.
Use one report per real full performance. Common fps/stage/conditioning/split/postprocessing/tolerance are required.
Reports must be from the same tool version. Pose pools also require matching point schemas.
Micro results use sums/counts; macro results weight performances equally; missing counts remain visible.

## Split declaration check

Create a list of `{sample_id,piece_group,performance_id,split}` records, optionally `performer_id`:

```sh
python -m piano_eval check-splits split-manifest.json --output split-check.json
# If claiming generalization to unseen performers:
python -m piano_eval check-splits split-manifest.json --hold-out-performers --output performer-check.json
```

Any piece group or performance reused across splits is rejected. Performer overlap is rejected only with the explicit flag.
This checks declarations, **not** file-level similarities, hidden training exposure or whether a “sealed” test set was viewed.
