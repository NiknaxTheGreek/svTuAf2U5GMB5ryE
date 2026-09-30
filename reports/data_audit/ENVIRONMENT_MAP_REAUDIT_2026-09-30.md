# MonReader V2 Environment-Map Re-Audit — 2026-09-30

Status: **PASS for use as the reviewed source/environment grouping input to the V2 split-construction phase.**

This review does not itself freeze the S or ST split. Those splits must be generated and audited separately.

## Structural coverage

Fresh V2 raw-data manifest:
- canonical videos: 117
- environment-map rows: 117
- unique mapped video IDs: 117
- missing videos: 0
- extra videos: 0
- duplicate mapping keys: 0
- invalid environment IDs: 0
- reviewed-status exceptions: 0

Environment populations:
- ENV-01: 21 videos, 584 images
- ENV-02: 32 videos, 699 images
- ENV-03: 29 videos, 770 images
- ENV-04: 35 videos, 936 images

## Fresh visual evidence

The V2 audit regenerated earliest, middle, and latest frames for every canonical video (351 representative frames total) from the SHA-verified source archive.

Observed grouping evidence:
- ENV-01: consistent technical/workbook document family, repeated grayscale figures/screens, stable white tabletop and camera geometry.
- ENV-02: distinct small cream prose paperback, dense text, stable form factor, lighting, tabletop and hand/camera context.
- ENV-03: distinct illustrated Stephen Hawking paperback, including recognizable astronomy imagery/front matter and consistent page/camera context.
- ENV-04: distinct technical/reference document family dominated by code, tables and figures, with stable layout and capture geometry.

No obvious cross-environment misassignment was observed in the 351-frame overview. The ENV-01/ENV-04 groups are both technical in broad subject matter but visibly differ in document/layout family.

## Limit

This is a reviewed visual/source grouping. It is not metadata-level proof that every group corresponds to one physical recording session. The grouping is supported by document identity/layout, camera/table geometry, hands/lighting context and temporal/source numbering evidence.

## Evidence lineage

Fresh environment audit:
- workflow run: 36692724047
- source branch commit: 965002d7b5720bebc18d4ba79ecf63a8116a61e5
- evidence artifact ID: 11086442556
- artifact SHA-256: 7096c4eec3a080179ce4351c71d91f727587fa2418200f0d0033f6bc6e9459b1

The map entered this review only as a historical reference. This receipt records its fresh V2 re-audit.
