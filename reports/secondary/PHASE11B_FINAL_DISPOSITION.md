# Phase 11B — Final disposition

Status: **COMPLETE — secondary/exploratory, non-selection-bearing**.

The frozen O/S/T/ST champions are unchanged by every result in this phase.

## 1. Naturalistic hand/arm removal

Final disposition: **FAILED PoC — STOP RULE INVOKED**.

The broad temporal mosaic + LaMa residual stage was the final permitted reconstruction attempt.

- final PoC run: `37017176634`;
- artifact: `11232014527`;
- mean real-pixel fill: 0.5655;
- median real-pixel fill: 0.6253;
- targets meeting the >=80% real-pixel gate: 15/48 = 0.3125;
- required fraction: 0.90;
- outside-mask changed images: 0;
- visual QA: FAIL.

Therefore no 100-image naturalistic hand/arm validation, no 2,989-image cleaned corpus, and no cleaned-domain retraining/HPO are authorized.

The controlled hand-mask/occlusion sensitivity analysis remains valid and must not be described as naturalistic hand removal.

## 2. Chroma counterfactual development

The first 100-image context-only development QA preserved luminance and geometry but showed that the initial naturalistic palette counterfactual was too weak.

A later full-strength reproducibility rerun on the same frozen development sample was completed on 2026-10-03:

- workflow run: `37152168592`;
- artifact: `11284402455`;
- artifact digest: `sha256:457da43f6fcbe36eaef52429d2a6be869908f563b60c6feecb0d2c1f54315495`;
- images: 100;
- primary-test images used: 0;
- classifier predictions used: 0;
- matched-rotation mean Rec.709 luminance MAE: 0.170584;
- naturalistic mean Rec.709 luminance MAE: 0.189370;
- naturalistic p95 per-image luminance MAE: 0.235276;
- median naturalistic chroma displacement: **2.259161**;
- required minimum: 3.0.

The development-sample naturalistic transform therefore remains below the frozen non-triviality threshold.

This rerun reproduced the already committed development evidence. The GitHub Actions job was marked red only because the final publishing step called `git commit` when there were no changed files. The scientific run and artifact upload both succeeded. The publishing workflow was subsequently made idempotent in commit `5edeb3f30ed48ed2afed120e7d1e48a899bab304`.

## 3. Independent naturalistic chroma v2 validation

The corrected v2 procedure was frozen before primary-test evaluation and validated on a **new deterministic 100-image ST-context sample**.

- validation run: `37073967071`;
- artifact: `11255238172`;
- artifact SHA256: `953ae8bc50833f72c6bbb452cdee9ce2901d1318f2318a7ea3d4fcf4e2220af5`;
- development/validation overlap: 0;
- primary-test overlap: 0;
- classifier predictions used: 0;
- universal donor pool: 1,428 non-test images across O/S/T/ST;
- same-video donors: 0;
- same-environment donors: 0;
- same-label fraction after label-independent selection: 0.50;
- mean Rec.709 luminance MAE: 0.187563;
- p95 per-image luminance MAE: 0.239726;
- median chroma displacement: **3.751614**;
- quantitative gate: PASS;
- visual gate after review of all 100 validation examples: PASS.

Naturalistic chroma v2 is therefore frozen and valid for the planned secondary sensitivity evaluation. No further naturalistic-colour tuning is permitted.

## 4. Frozen-champion all-regime evaluation

All four frozen scratch champions were evaluated without retraining or threshold changes on:

1. original RGB;
2. Rec.709 grayscale;
3. matched +90-degree chroma rotation;
4. independently validated naturalistic chroma v2.

Workflow run `37142158225` completed successfully for O, S, T and ST and for the combined summarization/publish job.

Combined artifact:

- artifact: `11281857048`;
- digest: `sha256:8480288af706f2305472f7eb8455971ebc10364bb6607f98863d054d2a9c05be`.

F1 results:

| Regime | Original | Grayscale | Matched rotation | Naturalistic v2 | Naturalistic v2 ΔF1 |
|---|---:|---:|---:|---:|---:|
| O | 0.9983 | 0.6840 | 0.6539 | 0.9386 | -0.0596 |
| S | 0.8196 | 0.6225 | 0.6225 | 0.7310 | -0.0887 |
| T | 0.8800 | 0.6695 | 0.6695 | 0.8224 | -0.0576 |
| ST | 0.7040 | 0.7109 | 0.7246 | 0.5818 | -0.1222 |

The paired video/group 95% intervals for naturalistic-v2 ΔF1 exclude zero for O and T, narrowly include zero for S, and include zero for ST. These are post-hoc sensitivity findings and cannot redefine the frozen regime champions.

## Final Phase 11B decision

Phase 11B is scientifically closed:

- **naturalistic hand/arm removal:** failed PoC; stopped;
- **development chroma QA:** underpowered naturalistic variant documented;
- **independent naturalistic chroma v2:** validated and frozen;
- **all-regime chroma sensitivity evaluation:** complete;
- **selection impact:** none;
- **retraining from these results:** none;
- **further tuning on primary test images:** forbidden.

Any future continuation should treat the files and artifacts above as frozen Phase 11B evidence rather than reopening the failed reconstruction or chroma-development paths.
