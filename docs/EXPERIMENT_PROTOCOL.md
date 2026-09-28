# Frozen experiment protocol boundary

## Active scientific question

How much apparent MonReader classification performance changes when evaluation is made source-safe and when strong temporal near-duplicate redundancy is removed.

## Controlled constructions

- D1: supplied split, all frames.
- D2: frozen environment-group-disjoint split, all frames.
- D3: supplied split after frozen deduplication.
- D4: exact D2 environment assignment after frozen deduplication.

The intended comparisons are D1 vs D2 for split construction, D1 vs D3 for deduplication under the supplied split, and D2 vs D4 for deduplication under the source-safe split.

## Protected evaluation rule

D2 and D4 test pixels are protected. They must not be used for model-family selection, epoch selection, threshold selection, debugging, preprocessing redesign, or ablation choice.

Model selection must use only non-test data. After the modelling protocol is frozen, selected configurations may be refit on their allowed full training populations and evaluated once on their corresponding frozen test partitions.

## Legacy results

Earlier source-group-only modelling and its final-test/sequence results are historical provenance, not active canonical results under this revised evaluation design. They were removed from the active branch after a verified GitHub Actions and Google Drive backup.
