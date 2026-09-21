# Source-Group Reconstruction

## Status

Step 3 source-group reconstruction: COMPLETE.

The reconstruction was generated from the authoritative Step 2 image inventory and independently reproduced in GitHub Actions run #7 after downloading and auditing the original Google Drive archive.

## Canonical grouping rule

Each logical source clip is identified by:

`source_group = (label, VideoID)`

The supplied `training/` and `testing/` folders are treated only as original storage partitions. They are not independent source groups.

Stable IDs use:

- `flip__0001`
- `notflip__0001`

Target mapping is now locked as:

- `notflip -> 0`
- `flip -> 1`

## Reconstructed population

- Total frames: 2,989
- Canonical source groups: 117
- Flip groups: 65
- Notflip groups: 52
- Groups represented in both supplied folders: 115
- Training-only groups: 2
- Testing-only groups: 0
- Duplicate `(label, VideoID, FrameNumber)` keys: 0

The two training-only groups are:

- `flip__0023`
- `flip__0025`

## Frame-sequence gaps

Three source groups contain missing frame-number spans in the supplied data:

- `flip__0023`: missing 7-15
- `flip__0029`: missing 10-17
- `notflip__0021`: missing 9-21

These missing frame numbers are preserved as data gaps. No synthetic or inferred frames are created.

## Reproducibility

The local reconstruction and clean GitHub Actions reconstruction produced identical canonical files.

- `source_groups.csv` SHA-256:
  `904271e433d13e9b83bb8cf87fc9bb996add3a8bde11b885a156a45b86b334fb`
- `frames_with_source_group.csv` SHA-256:
  `6b34a5892f1efabe08ca70f6136cd02511c9e43e20db35f86ceec0319acdff6c`

The reconstruction script contains dataset-lock assertions for the authoritative 2,989-frame / 117-group population and fails if those invariants change unexpectedly.

## Split implication

All future model-development splits must be assigned at `source_group_id` level. Every frame from one source group must remain in the same train, validation, or test partition.

Random frame-level splitting remains prohibited.
