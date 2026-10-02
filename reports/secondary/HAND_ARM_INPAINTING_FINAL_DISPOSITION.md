# Phase 11B — Naturalistic hand/arm removal PoC final disposition

Status: **FAILED PoC — stop full-scale naturalistic hand/arm cleaning.**

This is a valid negative result. No primary test image was used to select or tune any reconstruction method, and no classifier prediction was used for method selection.

## Development sequence

### Stage 1 — historical SCHP/MediaPipe + TELEA
Run `36983299845`, artifact `11216458307`.

The recovered arm/hand masking pipeline executed successfully, but large-mask TELEA reconstruction created obvious smooth/polygonal regions. Visual QA failed.

### Stage 2 — same-video real pixels + TELEA residual
Run `36984523373`, artifact `11217245472`.

- 46/48 targets had accepted same-video alignments.
- Mean real-pixel coverage: 0.2506.
- Median real-pixel coverage: 0.1748.
- 5/48 met the >=80% real-pixel / <=20% residual gate.

Quantitative and visual QA failed.

### Stage 3 — LaMa full and video + LaMa residual
Run `36985589884`, artifact `11217248702`.

LaMa preserved all outside-mask pixels exactly and improved visual quality relative to TELEA, but many large masked regions remained implausible, including blurred/flesh-toned smears, damaged text regions and hand-like residual structure. Visual QA failed.

### Stage 4 — broad temporal mosaic + LaMa residual
Run `37017176634`, artifact `11232014527`.

Artifact SHA256:
`04f760352a26c3e7ecf947f105b0180edab8566f39db39cc5a7707120cc7b1fd`.

All same-video ST non-test frames were eligible as reconstruction sources. Primary ST test frames were forbidden.

Results:

- mean source pool: 24.85 frames;
- median source pool: 25.5;
- mean accepted sources: 22.75;
- median accepted sources: 24;
- mean real-pixel fill: 0.5655;
- median real-pixel fill: 0.6253;
- p10 real-pixel fill: 0.0607;
- minimum real-pixel fill: 0.0424;
- mean residual: 0.4345;
- median residual: 0.3747;
- 15/48 targets met the >=80% real-pixel coverage gate;
- fraction meeting gate: 0.3125;
- required fraction: 0.90;
- outside-mask changed images: 0.

The frozen quantitative gate therefore failed.

Visual review of all four contact sheets also failed. Although many transferred page pixels are plausible, large residual regions remain visibly synthetic or mis-reconstructed, especially on frames with persistent hand/arm occlusion across the video.

## Final decision

The prospectively frozen stop rule is invoked.

Therefore:

- do not promote any naturalistic hand/arm reconstruction to the 100-image validation gate;
- do not construct a 2,989-image naturalistic hand/arm-cleaned corpus;
- do not run the planned cleaned-domain fixed-config retraining or cleaned-domain HPO as though a valid cleaned corpus existed;
- do not describe the existing occlusion/masking experiment as hand removal or inpainting.

The already-completed mask/occlusion sensitivity experiment remains valid as a controlled perturbation analysis.

A future project could revisit naturalistic removal if higher-frame-rate source video, depth/multi-view data, or a validated document-specific reconstruction model becomes available.
