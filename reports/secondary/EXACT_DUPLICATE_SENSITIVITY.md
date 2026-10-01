# Phase 11 — Exact-duplicate sensitivity

Status: **not applicable / structural null result**.

The authoritative Phase 1 data audit defined exact duplicates by decoded RGB pixel-array identity and found:

- 2,989 / 2,989 images accounted for;
- zero exact decoded-pixel duplicate groups;
- therefore zero duplicate groups crossing supplied train/test boundaries, labels, or canonical videos.

Because the exact-duplicate population is empty, there is no scientifically meaningful "remove exact duplicates and re-evaluate" sensitivity experiment to run. Removing an empty set would reproduce the original datasets exactly.

This is a valid negative finding: exact-pixel duplication cannot explain the high Original benchmark performance. It does not address near-frame temporal correlation, which is analyzed separately and is known to be substantial in O.

No model result or frozen champion is changed by this secondary item.
