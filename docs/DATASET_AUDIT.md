# Authoritative Dataset Audit

## Status

Step 2 dataset inspection: COMPLETE.

The audit was executed on the authoritative Google Drive archive `images.zip` in GitHub Actions. The archive downloaded as 939,921,132 bytes and passed ZIP integrity testing before extraction.

Archive SHA-256:

`033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f`

## Dataset inventory

- Total images: 2,989
- Training / flip: 1,162
- Training / notflip: 1,230
- Testing / flip: 290
- Testing / notflip: 307
- Total flip: 1,452
- Total notflip: 1,537
- All 2,989 images: 1080 x 1920, RGB, JPEG
- Unreadable images: 0
- Filename violations: 0
- Unexpected label directories: 0
- Exact duplicate groups: 0
- Cross-split exact duplicate groups: 0

## Source-clip reconstruction

Filenames follow `<VideoID>_<FrameNumber>.jpg`.

Folder-defined clip counts:

- Training / flip: 65
- Training / notflip: 52
- Testing / flip: 63
- Testing / notflip: 52

When training and testing are compared by `(label, VideoID)`:

- Testing groups: 115
- Testing groups also represented in training: 115
- Flip shared groups: 63 / 63 testing groups
- Notflip shared groups: 52 / 52 testing groups
- All 597 testing frames belong to a `(label, VideoID)` group also represented in training.
- No identical frame numbers occur across the two folders for the same group.
- 113 / 115 shared groups have a contiguous combined frame-number sequence.
- 114 / 115 shared groups have overlapping train/test frame-number ranges.

This is consistent with the supplied folders being a frame-level partition of source clips rather than clip-disjoint training and testing datasets.

## Perceptual cross-split audit

Using 64-bit dHash with Hamming distance <= 4 as a screening threshold:

- Cross-split near-duplicate candidates: 517
- Distance 0 candidates: 201
- Same-label candidates: 514
- Cross-label candidates: 3
- Same-label + same-VideoID candidates: 475
- Same-label + same-VideoID candidates within +/-3 frame numbers: 403
- Shared source groups with at least one same-key near match: 109 / 115
- Shared source groups with at least one same-key dHash distance 0 match: 77 / 115

dHash similarity is screening evidence rather than proof of byte identity. The source-key and frame-number evidence provides the grouping evidence; perceptual similarity independently reinforces it.

## Locked split decision

The supplied `training/` and `testing/` folders MUST NOT be treated as an independent evaluation split for model selection or final F1 reporting.

For leakage-safe modeling, first recombine frames belonging to the same source clip using:

`source_group = (label, VideoID)`

Then create new train/validation/test partitions at the source-group level so every source group occurs in exactly one partition.

A random frame-level split is prohibited.

## Target provenance

The supplied labels remain treated as clip-derived labels inherited by frames unless independent frame-level annotation is later demonstrated. Reported model performance therefore measures agreement with the supplied labels.

## Additional audit flag

Three testing `notflip` frames had dHash-nearest training frames from the `flip` class at distance <= 1. These are not exact duplicates and should be retained as later error-analysis / possible label-boundary inspection candidates rather than automatically relabeled or removed.
