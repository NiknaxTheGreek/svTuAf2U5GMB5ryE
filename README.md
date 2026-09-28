# MonReader — leakage-safe experimental foundation

This repository contains the active, cleaned MonReader foundation for classifying a single page image as flip or notflip.

## Why the original split is not the final evaluation design

The authoritative archive contains 2,989 valid RGB JPEG frames. Reconstructing clips with (label, VideoID) produced 117 temporal clips. Of those, 115 cross the supplied training/testing boundary. Perceptual similarity and SSIM analysis confirmed highly similar neighbouring frames from the same temporal sequence on opposite sides of that boundary.

The supplied split is retained as D1 for comparison, not treated as an unseen-source evaluation.

## Frozen constructions

| Dataset | Construction | Train | Test | Source-safe? |
|---|---|---:|---:|---|
| D1 | supplied split, all images | 2,392 | 597 | No |
| D2 | frozen environment-safe split, all images | 2,392 | 597 | Yes |
| D3 | supplied split after frozen deduplication | 1,521 | 391 | No |
| D4 | D2 assignment after frozen deduplication | 1,504 | 408 | Yes |

D2 has 44 train and 11 test environment groups with zero environment-group, temporal-clip, or frame-path overlap. D4 reuses the exact D2 environment assignment and then applies the frozen deduplication mask.

## Deduplication

Perceptual candidates are screened with dHash <= 4 AND pHash <= 4 and confirmed with SSIM >= 0.95 at 256×455 grayscale. The frozen connected-component construction contains 189 groups and retains one deterministic representative per group, leaving 1,912 frames.

## Diagnostic shortcut evidence

Simple global visual properties carry meaningful class signal. Saturation and brightness are the strongest retained diagnostic associations. This motivates later controlled ablations, but hand-removal experiments and other exploratory preprocessing are intentionally not part of the active foundation.

## Current status

FOUNDATION_FROZEN_READY_FOR_MODELING

The active branch intentionally contains no canonical legacy CNN/final-test result. Earlier modelling, post-holdout analysis, hand-removal pilots, OCR experiments, and other exploratory material were backed up before cleanup and are recoverable from the recorded Drive backup in PROJECT_STATE.yaml.

The next step is to define and freeze a fresh modelling protocol using only the non-test portion of the source-safe construction, then compare fixed models across D1–D4 without using protected D2/D4 test pixels for selection or redesign.

## Key evidence

- audit/ — authoritative dataset and source-group receipts
- raw_temporal_audit/ — temporal-order evidence
- raw_similarity_audit/ and raw_ssim_audit/ — leakage/near-duplicate evidence
- raw_environment_groups/ — frozen conservative environment grouping
- raw_global_dedup/ — frozen deduplication evidence
- raw_d2_split/ — frozen D2 manifests and receipt
- raw_d3_d4_split/ — frozen D3/D4 manifests and receipt
- raw_feature_abc/ — retained shortcut diagnostic
- TARGET_PROVENANCE.yaml — label-provenance and claim boundary
