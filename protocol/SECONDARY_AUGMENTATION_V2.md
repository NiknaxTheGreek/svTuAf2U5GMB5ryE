# Phase 11 fixed augmentation ablation

Status: **prospectively frozen secondary/exploratory ablation**.

Config SHA256: `5168b13d3ce4f286df71a1f823cb78889098fa843602154c34683ede80dce18b`.

The four already-frozen scratch champion configurations are retrained once each with the same mild augmentation recipe:

- O: C14
- S: C18
- T: C06
- ST: C07

No candidate search is reopened.

Training remains 20 epochs, no validation, no early stopping, same optimizer/hyperparameters/seed as the frozen candidate configuration, natural class balance, final epoch-20 checkpoint, threshold 0.5.

Training-only augmentation:

- no horizontal or vertical flip;
- with p=0.75: rotation uniform ±3 degrees, x/y translation uniform ±3% of canvas, scale 0.98–1.02;
- with p=0.75: brightness 0.90–1.10 and contrast 0.90–1.10;
- bilinear interpolation and black fill.

Before augmented training, the recipe is visually checked on the fixed 48-image ST-context development sample. No primary-test model behavior participates in this semantic gate.

After training, each augmented checkpoint is evaluated on the identical already-opened regime test and compared with the corresponding frozen non-augmented champion using paired frame and video/group bootstrap. Results are post-hoc and cannot replace Best O/S/T/ST.
