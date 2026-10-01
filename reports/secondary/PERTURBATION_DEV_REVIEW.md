# Phase 11 perturbation development review

## Scope

This review freezes perturbations for post-hoc shortcut sensitivity only. It cannot alter Best O/S/T/ST.

Development used 48 fixed, stratified **ST context-only** images. No primary test row and no classifier metric was used to choose a transform.

## Grayscale

Rec.709 grayscale is frozen. Its mean prospectively defined luma MAE was 0.2515 versus 1.0814 for Pillow L on the development set, so it better preserved the specified luminance signal while removing chroma.

## Hand/arm perturbation

All four 12-image review sheets were visually inspected.

The selected perturbation is **`mp_full_seed_c50`**:

- detect MediaPipe landmarks on the original full-resolution RGB frame;
- map the hand hull into the model canvas;
- extend around the landmark seed with the local strict-skin neighborhood;
- fill masked pixels with the median colour of a local unmasked border ring.

Why this variant:

- the hull-only candidate often under-covers fingers/forearm;
- the seed+skin extension more consistently covers the visibly relevant hand region;
- the fallback version can introduce masks when MediaPipe has no detection, increasing the risk of masking skin-like page/background content;
- full-resolution detection is preferred to preprocessed detection because landmark localization is performed before model-canvas resizing.

Development detection was nonzero for 44/48 images; four misses remain. This matters: the transform is **not** a validated complete hand/arm remover.

## Control

A matched generic-occlusion control mirrors the exact hand mask horizontally and applies the same fill. The model response to hand masking will therefore be compared with response to an equal-area non-hand occlusion.

## Interpretation boundary

The next experiment is a **shortcut perturbation sensitivity analysis** on frozen Best-O C14. It can support statements about model dependence on pixels selected by this detector relative to matched occlusion. It cannot establish that all human pixels were removed, nor can it change any primary champion.
