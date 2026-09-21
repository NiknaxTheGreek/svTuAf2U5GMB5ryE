# Leakage-Safe Split Policy

## Status

Step 4 source-group audit and split preparation: COMPLETE.

The final development split is fixed at the canonical source-group level. Frames from one source clip may not appear in more than one partition.

## Split unit

Canonical group key:

`(label, VideoID)`

Stable group ID:

`<label>__<zero-padded VideoID>`

The original supplied `training/` and `testing/` directories are ignored as evaluation partitions because Step 2 established that they divide frames from the same source clips.

## Fixed split

Target ratio: 60% train / 20% validation / 20% test by frames, subject to whole-source-group assignment and class balance.

| Partition | Source groups | Flip groups | Notflip groups | Frames | Flip frames | Notflip frames |
|---|---:|---:|---:|---:|---:|---:|
| Train | 70 | 39 | 31 | 1,795 | 872 | 923 |
| Validation | 23 | 13 | 10 | 597 | 290 | 307 |
| Test | 24 | 13 | 11 | 597 | 290 | 307 |

Frame-level flip prevalence:

- Train: 48.5794%
- Validation: 48.5762%
- Test: 48.5762%

## Leakage checks

Source-group overlap counts:

- Train vs validation: 0
- Train vs test: 0
- Validation vs test: 0

All 117 source groups occur exactly once. All 2,989 frames inherit exactly one partition from their source group.

## Selection rule

The split optimizer used only:

- supplied label,
- canonical source-group ID,
- frame count.

No image pixels, image-derived features, model predictions, model scores, or evaluation outcomes were used to choose the partition assignment.

Exact class-specific group counts and frame totals were enforced, with a stable SHA-256-derived tie-break objective. The resulting assignment is now frozen in `splits/source_group_split.csv`.

## Holdout discipline

From this point forward:

- Train: model fitting and fit-derived preprocessing only.
- Validation: model/threshold/hyperparameter selection and development diagnostics.
- Test: untouched by model selection. Use only for the locked final evaluation after the modeling decision is complete.

If the model or methodology is changed after inspecting test performance, the test set can no longer be described as an untouched final holdout.

## Reproducibility

Clean GitHub Actions run #8 reproduced the split generated locally byte-for-byte.

- `source_group_split.csv` SHA-256:
  `5eaf917adc51db9a9bd10d9d4c6487050284df6e47c928d18ec5d1e94d64b551`
- `frame_split.csv` SHA-256:
  `a2b50b0a1826924cb84da0f17732457a8a23410afada3c0e3fce8a2693628dfe`

The authoritative workflow regenerates the frame-level mapping from the original Drive archive and fails if source-group or split invariants are violated.
