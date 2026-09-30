# MonReader V2 Split Audit

## Frozen split rules

All splits are derived from the verified 2,989-row `DATA-001` manifest.

For temporal regimes, each canonical video is sorted by `FrameNumber`. With `n` observed images:

`cut = floor(0.80 × n)`

Rows with temporal rank below `cut` are the earlier portion; rows at or above `cut` are the later portion. The rule operates on observed images and does not invent missing frame numbers.

Source-safe holdout: **ENV-03**.

## Accounting

| Regime | Training | Primary test | Context-only | Excluded | Total |
|---|---:|---:|---:|---:|---:|
| O | 2392 | 597 | 0 | 0 | 2989 |
| S | 2219 | 770 | 0 | 0 | 2989 |
| T | 2365 | 624 | 0 | 0 | 2989 |
| ST | 1756 | 161 | 1072 | 0 | 2989 |

ST context-only decomposition:
- earlier held-out ENV-03: 609
- later known-source environments: 463

## Scientific meaning

**O — Original:** supplied training vs supplied testing exactly as distributed. Its test is source-joint: all 597 test images are from videos represented in supplied training.

**S — Source-Safe:** all ENV-03 images are test; all ENV-01/02/04 images are training. Train/test canonical-video overlap is zero.

**T — Temporal:** every video contributes its earlier observed 80% to training and its later observed 20% to test. This measures future-frame generalization within known sources.

**ST — Source-Safe + Temporal:** ENV-03 is completely absent from training. For ENV-03, earlier frames are context-only and later frames form the primary test. For known environments, earlier frames train and later frames are context-only. Train/test canonical-video overlap is zero.

## Leakage/accounting gates

- Every regime accounts for exactly 2,989 unique sample IDs.
- Excluded samples: 0 in every regime.
- S and ST primary train/test video overlap: 0.
- T chronological boundaries are strict within every video.
- ST contains no ENV-03 frame in training.
- ST primary test contains only later ENV-03 frames.

No model training or model-performance evaluation is part of this phase.
