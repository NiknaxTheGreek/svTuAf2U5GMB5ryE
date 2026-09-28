from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

required = [
    "PROJECT_STATE.yaml",
    "TARGET_PROVENANCE.yaml",
    "README.md",
    "audit/audit_summary.json",
    "audit/source_group_summary.json",
    "raw_environment_groups/environment_grouping_summary.json",
    "raw_d2_split/d2_frame_split_manifest.csv",
    "raw_d2_split/d2_split_summary.json",
    "raw_d2_split/d2_split_receipt.json",
    "raw_global_dedup/global_dedup_summary.json",
    "raw_d3_d4_split/d3_frame_manifest.csv",
    "raw_d3_d4_split/d4_frame_manifest.csv",
    "raw_d3_d4_split/d3_d4_split_summary.json",
    "raw_d3_d4_split/d3_d4_split_receipt.json",
]
missing = [p for p in required if not (ROOT / p).exists()]
assert not missing, f"Missing required files: {missing}"

with (ROOT / "audit/audit_summary.json").open() as f:
    audit = json.load(f)
assert audit["archive_sha256"] == "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"
assert audit["total_images"] == 2989
assert audit["merged_source_group_count"] == 117
assert audit["testing_source_groups_also_in_training"] == 115

with (ROOT / "raw_environment_groups/environment_grouping_summary.json").open() as f:
    env = json.load(f)
assert env["environment_group_count"] == 55
assert env["reviewed_candidate_links"] == 82

with (ROOT / "raw_d2_split/d2_split_summary.json").open() as f:
    d2 = json.load(f)
assert d2["d2"]["train"]["environment_groups"] == 44
assert d2["d2"]["test"]["environment_groups"] == 11
assert d2["d2"]["test"]["frames"] == 597
assert d2["d2"]["test"]["flip_frames"] == 290
assert d2["d2"]["test"]["notflip_frames"] == 307
assert d2["overlap_checks"] == {
    "environment_group_overlap": 0,
    "sequence_overlap": 0,
    "frame_path_overlap": 0,
}
assert sha256(ROOT / "raw_d2_split/d2_frame_split_manifest.csv") == "48eef3d49f6df755ac74cddd6e22f3a3ad9c593bfa72ce62b7d5e234d0f21ed3"

with (ROOT / "raw_global_dedup/global_dedup_summary.json").open() as f:
    dd = json.load(f)
assert dd["near_duplicate_group_count"] == 189
assert dd["images_retained_after_dedup"] == 1912

with (ROOT / "raw_d3_d4_split/d3_d4_split_summary.json").open() as f:
    d34 = json.load(f)
assert d34["d3"]["status"] == "FROZEN"
assert d34["d4"]["status"] == "FROZEN"
assert d34["d4"]["overlap_checks"] == {
    "environment_group_overlap": 0,
    "sequence_overlap": 0,
    "frame_path_overlap": 0,
}
assert sha256(ROOT / "raw_d3_d4_split/d3_frame_manifest.csv") == "7583b706109164bef2c2abdc77c96fabb71e53f976ef3a7c9030282c5abfdcf4"
assert sha256(ROOT / "raw_d3_d4_split/d4_frame_manifest.csv") == "68144c1ad150f0557e05e05d083e3a0199b8a43b13ed64b47ef77db2920a1ff5"

for forbidden in [
    "results",
    "splits",
    "raw_hand_arm_ablation",
    "raw_hand_removed_audit_final",
    "raw_page_number_audit",
    "raw_model_selection",
    "raw_epoch_selection",
    "raw_scratch_epoch_selection",
]:
    assert not (ROOT / forbidden).exists(), f"Obsolete path still present: {forbidden}"

print("PASS: frozen MonReader D1-D4 foundation verified")
