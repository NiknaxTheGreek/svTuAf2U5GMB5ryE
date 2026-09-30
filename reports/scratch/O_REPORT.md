# MonReader V2 — Original (O) Scratch-CNN Selection Benchmark

## Question

How well does the frozen scratch-CNN candidate family perform under the **supplied MonReader train/test split**?

This is the conventional MonReader benchmark. Because the 597-image O test is used to select among 20 already-frozen candidates, the result is reported as a **selection-benchmark estimate**, not as an untouched post-selection test estimate.

## Execution integrity

The frozen bank contained C01–C20 and was unchanged from the pre-test audit.

For O:

- training rows: 2,392;
- test rows: 597;
- threshold: 0.5;
- epochs per candidate: exactly 20;
- validation rows loaded during candidate training: 0;
- test rows loaded during candidate training: 0;
- all 20 final epoch-20 checkpoints were round-trip loaded and hash-verified before the O test cache was built;
- all 20 candidate training jobs completed successfully;
- every candidate received exactly one O-test evaluation.

The closed-test checkpoint registry explicitly reported `PASS_ALL_20_FROZEN_BEFORE_O_TEST`.

A prior workflow attempt failed during **synthetic preflight** with a Python import-path error. It did not train MonReader candidates or open O test data. The successful run changed only import-path/module invocation handling.

## Best O

**C14** is Best O by the predeclared ordering.

Configuration:

- depth: 5 convolution blocks;
- start filters: 32;
- dropout: 0.273005;
- optimizer: Adam;
- learning rate: 0.000134508;
- weight decay: 9.61225e-06;
- batch size: 41;
- trainable parameters: 1,570,081;
- final epoch-20 training loss: 0.016775.

### O-test metrics

| Metric | Best O (C14) |
|---|---:|
| F1 | **0.998273** |
| Precision | 1.000000 |
| Recall | 0.996552 |
| Accuracy | 0.998325 |
| Balanced accuracy | 0.998276 |
| ROC-AUC | 0.999787 |
| PR-AUC | 0.999788 |
| TN / FP / FN / TP | 307 / 0 / 1 / 289 |

There was one error: `testing/flip/0041_000000010`, a false negative with (p(flip)=0.0420).

The top of the frozen bank was not limited to one unusually lucky candidate: C13 and C06 both achieved about 0.993 F1, and C10/C07 achieved about 0.991. Conversely, several smaller candidates had much poorer fixed-threshold F1. Architecture/training configuration therefore materially affected the result.

## Strong-score audit

The near-perfect score triggered an explicit audit.

### Exact duplicates

The V2 raw-data audit found:

- exact decoded-pixel duplicate groups: **0**;
- exact cross-split duplicate groups: **0**.

Therefore exact pixel duplication does not explain C14's score.

### Source overlap

The supplied Original split is completely source-joint at the canonical-video level:

- O test images: 597;
- O test images from training-seen videos: **597 / 597**;
- unseen O-test videos: **0**.

The supplied O split therefore cannot answer unseen-source generalization.

### Temporal proximity

For the 597 O-test frames:

- 573 / 597 (**95.98%**) have a same-video training frame only one frame number away;
- 596 / 597 (**99.83%**) are within two frame numbers of a same-video training frame.

This is the main interpretation constraint on the very high O score.

### Temporal-future subset

Twenty O-test frames across 18 videos occur later than every supplied-training frame from that same video.

Frozen Best O is perfect on those 20 images:

- F1 = 1.0;
- precision = 1.0;
- recall = 1.0;
- accuracy = 1.0.

This subset is **source-joint + temporally future**. It is not source-disjoint and is too small to carry the main generalization conclusion.

## Visual error analysis

The sole false negative, flip/0041 frame 10, was inspected next to the nearest O-training frames from the same video:

- frame 11: training, distance 1;
- frame 12: training, distance 2;
- frame 13: training, distance 3;
- frame 14: training, distance 4.

The target and neighboring training frames are visually extremely similar: a mostly blank page is partially lifted, hands are visible at the lower/right portions of the image, and the target contains visible motion blur near the right hand/page edge.

This demonstrates the strong frame correlation directly. It also shows that adjacency does not guarantee a correct prediction: C14 still assigned only 0.042 probability to the positive class on frame 10.

Borderline correct examples frequently contain visible hands, raised page edges, blur or page geometries that could resemble transition states. This observation **does not establish a hand shortcut**. It strengthens the rationale for the later controlled hand/arm removal experiment.

## Interpretation

Best O is a very strong result **for the supplied conventional benchmark**.

It should not be summarized as evidence that MonReader will achieve ~0.998 F1 for a new user, new acquisition environment or unseen video source. The split contains pervasive same-video and near-frame correlation, and no source-disjoint O diagnostic exists.

The next primary scientific question is therefore S:

> Does the same frozen candidate family retain performance when the entire held-out acquisition environment is absent from training?

No O candidate will be modified or retrained based on this result.
