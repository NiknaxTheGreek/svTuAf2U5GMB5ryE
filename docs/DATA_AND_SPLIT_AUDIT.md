# Data and split audit

The authoritative archive SHA-256 is 033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f.

The archive contains 2,989 readable 1080×1920 RGB JPEG images: 1,452 flip and 1,537 notflip. No filename violations or exact duplicate groups were found.

Using (label, VideoID) as the temporal-clip key reconstructs 117 clips: 65 flip and 52 notflip. 115 clips appear in both supplied training and testing folders. Temporal ordering was empirically supported: all 117 ordered clips were smoother than 95% of 200 random shuffles under both grayscale MAE and dHash distance.

Cross-split perceptual screening and SSIM confirmation showed highly similar neighbouring frames from the same temporal sequence on opposite sides of the supplied split. The supplied split is therefore retained as D1 but is not treated as source-independent.

A conservative acquisition-environment review produced 55 frozen environment groups. These grouping units are used for leakage control and are not claimed to be known recording-session ground truth.

D2 chooses 11 whole environment groups for test while exactly matching the supplied D1 test frame/class totals: 597 frames, with 290 flip and 307 notflip. D2 has zero environment-group, temporal-clip, and frame-path overlap.

Deduplication uses perceptual-hash screening followed by SSIM confirmation. The frozen deduplication has 189 connected components, removes 1,077 frames, and retains 1,912 representatives. D3 applies it to the supplied split; D4 applies it within the exact frozen D2 assignment.
