# MonReader V2 — Cross-Regime Synthesis

## Scope

This phase synthesizes already-frozen scratch-CNN results. It performs no training, no threshold tuning, and no new model selection. All uncertainty is explicitly selection-conditioned.

## Champion envelope

| Regime | Champion | n | F1 | Frame 95% CI | Video/group 95% CI | Accuracy | ROC-AUC | PR-AUC | Question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| O | C14 | 597 | **0.9983** | 0.995–1.000 | 0.994–1.000 | 0.9983 | 0.9998 | 0.9998 | supplied source-joint benchmark |
| S | C18 | 770 | **0.8196** | 0.788–0.850 | 0.674–0.915 | 0.8377 | 0.9002 | 0.8995 | unseen ENV-03 source/environment |
| T | C06 | 624 | **0.8800** | 0.852–0.907 | 0.837–0.917 | 0.8894 | 0.9518 | 0.9637 | later frames within known videos/sources |
| ST | C07 | 161 | **0.7040** | 0.606–0.790 | 0.466–0.857 | 0.7702 | 0.9082 | 0.9150 | unseen ENV-03 plus later-frame stress test |

The regimes are not interchangeable evaluation populations, so the numerically highest F1 must not be called the globally best model. O is the conventional supplied benchmark; S isolates unseen-source difficulty; T isolates later-frame difficulty within known sources; ST combines unseen source and later-frame stress.

## Fixed O-winning configuration: C14

C14 is retrained from scratch inside each regime using the same frozen architecture/hyperparameters; these are not the same learned weights.

| Regime | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC | F1 retention vs O |
|---|---:|---:|---:|---:|---:|---:|---:|
| O | 0.9983 | 1.0000 | 0.9966 | 0.9983 | 0.9998 | 0.9998 | 1.000 |
| S | 0.6936 | 0.6977 | 0.6897 | 0.7247 | 0.7932 | 0.7908 | 0.695 |
| T | 0.2981 | 1.0000 | 0.1752 | 0.5849 | 0.9414 | 0.9567 | 0.299 |
| ST | 0.4646 | 1.0000 | 0.3026 | 0.6708 | 0.8889 | 0.9024 | 0.465 |

C14's fixed-threshold F1 degrades sharply outside O, especially in T and ST. In T, high ROC-AUC/PR-AUC coexist with low F1, consistent with a large operating-threshold/calibration shift rather than total loss of ranking signal.

## Uncertainty

Each champion uses 5,000 frame bootstrap resamples and 5,000 canonical-video/group bootstrap resamples with fixed seed 20261001. The intervals are not adjusted for selecting the champion from 20 candidates on the same test.

## Next gate

Proceed to the fixed ImageNet ResNet18 comparator. It must remain separate from scratch champion selection: no sweep, no validation, 5 fixed head-only epochs + 15 fixed full-network epochs, final epoch-20 checkpoint, and one test evaluation per selected regime.
