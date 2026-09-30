# MonReader V2 Baseline Results

The baseline protocol was frozen before these test results were generated.

| Regime | Baseline | F1 | Precision | Recall | Accuracy | Balanced acc. | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| O | Majority | 0.0000 | 0.0000 | 0.0000 | 0.5142 | 0.5000 | — | — |
| O | LogReg A | 0.7842 | 0.8195 | 0.7517 | 0.7990 | 0.7977 | 0.8770 | 0.8945 |
| O | LogReg B | 0.8571 | 0.8773 | 0.8379 | 0.8643 | 0.8636 | 0.9511 | 0.9540 |
| O | LogReg C | 0.9268 | 0.9366 | 0.9172 | 0.9296 | 0.9293 | 0.9844 | 0.9857 |
| S | Majority | 0.0000 | 0.0000 | 0.0000 | 0.5481 | 0.5000 | — | — |
| S | LogReg A | 0.4568 | 1.0000 | 0.2960 | 0.6818 | 0.6480 | 0.9173 | 0.9176 |
| S | LogReg B | 0.6583 | 0.5865 | 0.7500 | 0.6481 | 0.6570 | 0.7358 | 0.7047 |
| S | LogReg C | 0.5560 | 0.7926 | 0.4282 | 0.6909 | 0.6679 | 0.7829 | 0.7820 |
| T | Majority | 0.0000 | 0.0000 | 0.0000 | 0.4968 | 0.5000 | — | — |
| T | LogReg A | 0.7854 | 0.8223 | 0.7516 | 0.7933 | 0.7935 | 0.8423 | 0.8767 |
| T | LogReg B | 0.7450 | 0.8619 | 0.6561 | 0.7740 | 0.7748 | 0.8337 | 0.8720 |
| T | LogReg C | 0.8043 | 0.9113 | 0.7197 | 0.8237 | 0.8244 | 0.8729 | 0.9074 |
| ST | Majority | 0.0000 | 0.0000 | 0.0000 | 0.5280 | 0.5000 | — | — |
| ST | LogReg A | 0.3656 | 1.0000 | 0.2237 | 0.6335 | 0.6118 | 0.9356 | 0.9424 |
| ST | LogReg B | 0.5714 | 0.7200 | 0.4737 | 0.6646 | 0.6545 | 0.7627 | 0.7903 |
| ST | LogReg C | 0.1687 | 1.0000 | 0.0921 | 0.5714 | 0.5461 | 0.7980 | 0.7844 |

All logistic-regression scaling statistics were fitted on training rows only.
Feature sets A/B/C are all reported; no feature set was selected after viewing the tests.
These results do not open or rank the scratch-CNN candidate bank.
