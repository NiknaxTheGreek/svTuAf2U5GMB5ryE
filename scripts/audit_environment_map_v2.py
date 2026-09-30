from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps

ALLOWED_ENVS = {"ENV-01", "ENV-02", "ENV-03", "ENV-04"}
POSITIONS = ("earliest", "middle", "latest")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_tile(encoded: bytes, caption: str, size=(120, 213), label_h=28) -> Image.Image:
    with Image.open(BytesIO(encoded)) as im:
        im = im.convert("RGB")
        thumb = ImageOps.contain(im, size)
    canvas = Image.new("RGB", (size[0], size[1] + label_h), "white")
    x = (size[0] - thumb.width) // 2
    y = label_h + (size[1] - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((3, 5), caption, fill="black", font=ImageFont.load_default())
    return canvas


def make_contact_sheet(env_id: str, groups: list[tuple[str, list[tuple[str, str]]]], zf: ZipFile, out: Path) -> None:
    per_row = 3
    tile_w, tile_h = 120, 241
    block_w = tile_w * 3
    block_h = tile_h
    header_h = 36
    rows_n = math.ceil(len(groups) / per_row)
    sheet = Image.new("RGB", (block_w * per_row, header_h + block_h * rows_n), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), f"{env_id} — earliest / middle / latest per canonical video", fill="black", font=ImageFont.load_default())
    for i, (video_id, selected) in enumerate(groups):
        block_x = (i % per_row) * block_w
        block_y = header_h + (i // per_row) * block_h
        for j, (position, member) in enumerate(selected):
            encoded = zf.read(member)
            short = video_id.replace("notflip", "NF").replace("flip", "F")
            cap = f"{short} {position[0].upper()}"
            tile = build_tile(encoded, cap)
            sheet.paste(tile, (block_x + j * tile_w, block_y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    archive = Path(args.archive)
    manifest_path = Path(args.manifest)
    mapping_path = Path(args.mapping)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = read_csv(manifest_path)
    mapping = read_csv(mapping_path)
    manifest_videos = sorted({r["video_id"] for r in manifest})
    map_ids = [r["video_id"] for r in mapping]
    duplicate_map_ids = sorted(v for v, n in Counter(map_ids).items() if n > 1)
    missing = sorted(set(manifest_videos) - set(map_ids))
    extras = sorted(set(map_ids) - set(manifest_videos))
    invalid_envs = sorted({r["environment_id"] for r in mapping} - ALLOWED_ENVS)
    nonreviewed = sorted({r.get("review_status", "") for r in mapping if r.get("review_status", "") != "reviewed"})

    if duplicate_map_ids or missing or extras or invalid_envs:
        raise RuntimeError(json.dumps({
            "duplicate_map_ids": duplicate_map_ids,
            "missing_manifest_videos": missing,
            "extra_mapped_videos": extras,
            "invalid_environment_ids": invalid_envs,
        }, indent=2))

    env_by_video = {r["video_id"]: r["environment_id"] for r in mapping}
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in manifest:
        by_video[r["video_id"]].append(r)

    group_counts = Counter(env_by_video.values())
    class_video_counts: dict[str, Counter] = defaultdict(Counter)
    image_counts: dict[str, Counter] = defaultdict(Counter)
    selected_rows: list[dict[str, str]] = []
    contact_groups: dict[str, list[tuple[str, list[tuple[str, str]]]]] = defaultdict(list)

    for video_id in manifest_videos:
        env = env_by_video[video_id]
        label = video_id.split("/", 1)[0]
        class_video_counts[env][label] += 1
        rows = sorted(by_video[video_id], key=lambda r: int(r["frame_number"]))
        for r in rows:
            image_counts[env]["all"] += 1
            image_counts[env][r["supplied_split"]] += 1
            image_counts[env][r["label"]] += 1
        inds = [0, len(rows) // 2, len(rows) - 1]
        selected: list[tuple[str, str]] = []
        for pos, idx in zip(POSITIONS, inds):
            r = rows[idx]
            selected.append((pos, r["archive_member"]))
            selected_rows.append({
                "environment_id": env,
                "video_id": video_id,
                "label": label,
                "position": pos,
                "frame_number": r["frame_number"],
                "archive_member": r["archive_member"],
            })
        contact_groups[env].append((video_id, selected))

    with ZipFile(archive) as zf:
        for env in sorted(contact_groups):
            make_contact_sheet(env, sorted(contact_groups[env]), zf, out / f"{env}_contact_sheet.jpg")

    with (out / "representative_frames.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["environment_id", "video_id", "label", "position", "frame_number", "archive_member"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected_rows)

    summary = {
        "mapping_status_before_reaudit": "historical_reviewed_reference",
        "visual_reaudit_status": "PENDING_FRESH_REVIEW",
        "manifest_video_count": len(manifest_videos),
        "mapping_row_count": len(mapping),
        "unique_mapping_video_count": len(set(map_ids)),
        "coverage": {
            "missing_manifest_videos": missing,
            "extra_mapped_videos": extras,
            "duplicate_mapping_video_ids": duplicate_map_ids,
            "invalid_environment_ids": invalid_envs,
            "nonreviewed_status_values": nonreviewed,
            "exact_video_coverage": not (missing or extras or duplicate_map_ids or invalid_envs),
        },
        "environment_video_counts": dict(sorted(group_counts.items())),
        "environment_class_video_counts": {
            env: dict(sorted(c.items())) for env, c in sorted(class_video_counts.items())
        },
        "environment_image_counts": {
            env: dict(sorted(c.items())) for env, c in sorted(image_counts.items())
        },
        "representative_frame_count": len(selected_rows),
        "representative_policy": "earliest, middle (index n//2), latest frame after sorting by FrameNumber within each canonical video",
        "contact_sheets": [f"{env}_contact_sheet.jpg" for env in sorted(contact_groups)],
    }
    (out / "environment_map_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
