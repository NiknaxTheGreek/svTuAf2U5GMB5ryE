# MonReader V2 Data Audit

## Purpose

This phase establishes the identity, integrity and source/temporal structure of the MonReader dataset before any V2 model training or experimental split construction.

## Dataset identity

The official `images.zip` archive was downloaded from the project-supplied Google Drive source and independently checked by both `sha256sum` and the Python audit program.

| Check | Verified value |
|---|---:|
| Archive bytes | 939,921,132 |
| SHA-256 | `033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f` |
| Images accounted for | 2,989 / 2,989 |
| Decode failures | 0 |
| Unknown/skipped members | 0 |

The canonical row-level manifest is `manifests/datasets/DATA-001.csv` with SHA-256 `3fb8a29f1a51fc7930fe6c876d0fcb8d99ec59e147d4abbff503da4870f15b0b`.

## Composition

| Population | Flip | Notflip | Total |
|---|---:|---:|---:|
| Supplied training | 1,162 | 1,230 | 2,392 |
| Supplied testing | 290 | 307 | 597 |
| **All images** | **1,452** | **1,537** | **2,989** |

There are 117 canonical videos. Because four-digit numeric IDs are reused across class folders, the canonical source identity is `{label}/{raw_video_id}`.

Every image is RGB, three-channel, and 1080×1920 pixels (aspect ratio 0.5625). Video length ranges from 8 to 55 observed frames, with a median of 29.

## Exact-duplicate integrity

Exact duplicates were defined from decoded RGB pixel arrays rather than filenames or compressed JPEG bytes.

No exact decoded-pixel duplicate groups were found. Consequently there are no exact duplicate groups crossing supplied train/test boundaries, labels, or canonical videos.

This rules out exact-pixel duplication as an explanation for strong future Original-model scores, but it does not rule out strong near-frame correlation.

## Temporal structure

Frames were sorted by parsed `FrameNumber` within each canonical video. Three videos contain gaps in the observed frame-number sequence:

- `flip/0023`: missing 7–15 (9 frame numbers);
- `flip/0029`: missing 10–17 (8 frame numbers);
- `notflip/0021`: missing 9–21 (13 frame numbers).

These are gaps in numbering within observed ranges, not unaccounted archive members. All 2,989 supplied images remain accounted for.

## Supplied Original split: source and temporal overlap

The supplied test contains 597 images from 115 videos. Every one of those 115 test videos also appears in supplied training; therefore the Original split contains **no unseen-video test population**.

For each supplied-test frame, the audit measured the absolute `FrameNumber` distance to the nearest supplied-training frame from the same canonical video:

- 573 / 597 test images (**95.98%**) are one frame number from a training frame;
- 596 / 597 (**99.83%**) are within two frame numbers;
- median nearest distance = 1;
- maximum nearest distance = 3.

This is a major interpretation constraint. High performance on the supplied Original split may be scientifically valid for that benchmark, but it cannot by itself establish unseen-source generalization and may benefit from highly correlated neighboring frames.

The post-hoc Original temporal-future population is also structurally identified in advance of model evaluation: 20 supplied-test images across 18 videos occur later than every supplied-training frame from the same canonical video. This remains source-joint, not source-disjoint.

## Source/environment grouping

A four-environment video map was structurally checked against the fresh manifest: all 117 canonical videos are covered exactly once, with no missing, extra or duplicate video keys.

| Environment | Videos | Images | Flip images | Notflip images |
|---|---:|---:|---:|---:|
| ENV-01 | 21 | 584 | 249 | 335 |
| ENV-02 | 32 | 699 | 465 | 234 |
| ENV-03 | 29 | 770 | 348 | 422 |
| ENV-04 | 35 | 936 | 390 | 546 |

A fresh visual review used earliest, middle and latest frames from every canonical video (351 representative frames total). The four groups show distinct, internally consistent document/layout and capture contexts: a technical/workbook family, a small prose paperback, an illustrated Stephen Hawking paperback, and a separate technical/reference family.

The grouping is therefore accepted for V2 source-safe split construction. It should be described as a reviewed visual/source grouping, not as metadata-level proof that each environment is one physical recording session.

## Scientific status after Phase 1

The raw dataset identity and accounting gate is **PASS**. No V2 model has been trained, no model test predictions have been examined, and no O/S/T/ST split has yet been frozen.

The next phase is reproducible construction and accounting of the four V2 regimes:

- O — supplied Original train/test;
- S — source-safe held-out environment;
- T — within-video chronological 80/20;
- ST — held-out environment plus temporal restriction.

Every regime must account explicitly for all 2,989 images before modeling begins.
