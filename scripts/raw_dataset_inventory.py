#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NAME_RE = re.compile(r"^(?P<video_id>.+)_(?P<frame_num>\d+)$")
EXPECTED_SPLITS = {"training", "testing"}
EXPECTED_LABELS = {"flip", "notflip"}


def resolve_dataset_root(root: Path) -> Path:
    root = root.resolve()
    if (root / "training").is_dir() and (root / "testing").is_dir():
        return root
    if (root / "images" / "training").is_dir() and (root / "images" / "testing").is_dir():
        return root / "images"
    raise FileNotFoundError(f"Could not locate training/ and testing/ under {root}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_preview(dataset_root: Path, image_rows: list[dict], out_path: Path) -> list[str]:
    selected = []
    for split in ("training", "testing"):
        for label in ("flip", "notflip"):
            rows = sorted(
                [r for r in image_rows if r["split"] == split and r["label"] == label and not r["read_error"]],
                key=lambda r: r["path"],
            )
            if not rows:
                continue
            idxs = [0] if len(rows) == 1 else [0, len(rows) // 2]
            for i in idxs:
                selected.append(rows[i])

    tile_w, tile_h = 250, 455
    cols = 4
    rows_n = max(1, (len(selected) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * tile_w, rows_n * tile_h), "white")
    draw = ImageDraw.Draw(canvas)

    for i, row in enumerate(selected):
        src = dataset_root / row["path"]
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((220, 390))
            x0 = (i % cols) * tile_w + (tile_w - img.width) // 2
            y0 = (i // cols) * tile_h + 28
            canvas.paste(img, (x0, y0))
        label_text = f'{row["split"]}/{row["label"]} | {row["filename"]}'
        draw.text(((i % cols) * tile_w + 6, (i // cols) * tile_h + 6), label_text, fill="black")

    canvas.save(out_path, format="JPEG", quality=88)
    return [r["path"] for r in selected]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    dataset_root = resolve_dataset_root(args.root)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    all_file_rows = []
    image_rows = []
    unreadable = []
    filename_violations = []

    all_files = sorted(p for p in dataset_root.rglob("*") if p.is_file())
    for path in all_files:
        rel = path.relative_to(dataset_root).as_posix()
        all_file_rows.append({
            "path": rel,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "is_image_extension": path.suffix.lower() in IMG_EXTS,
        })

        if path.suffix.lower() not in IMG_EXTS:
            continue

        parts = Path(rel).parts
        split = parts[0] if len(parts) >= 1 else None
        label = parts[1] if len(parts) >= 2 else None

        match = NAME_RE.match(path.stem)
        if match:
            video_id = match.group("video_id")
            frame_num = int(match.group("frame_num"))
        else:
            video_id = None
            frame_num = None
            filename_violations.append(rel)

        width = height = mode = fmt = digest = err = None
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                fmt = img.format
            digest = sha256_file(path)
        except Exception as exc:
            err = repr(exc)
            unreadable.append({"path": rel, "error": err})

        image_rows.append({
            "split": split,
            "label": label,
            "path": rel,
            "filename": path.name,
            "video_id": video_id,
            "frame_num": frame_num,
            "width": width,
            "height": height,
            "mode": mode,
            "format": fmt,
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "read_error": err,
        })

    write_csv(out / "all_files_inventory.csv", all_file_rows)
    write_csv(out / "image_inventory.csv", image_rows)
    (out / "unreadable.json").write_text(json.dumps(unreadable, indent=2), encoding="utf-8")
    (out / "filename_violations.json").write_text(json.dumps(filename_violations, indent=2), encoding="utf-8")

    preview_path = out / "preview_grid.jpg"
    preview_sources = create_preview(dataset_root, image_rows, preview_path)
    (out / "preview_grid.b64.txt").write_text(
        base64.b64encode(preview_path.read_bytes()).decode("ascii"),
        encoding="utf-8",
    )
    preview_path.unlink()

    folder_counts = Counter((r["split"], r["label"]) for r in image_rows)
    property_counts = Counter(
        (r["width"], r["height"], r["mode"], r["format"])
        for r in image_rows
        if not r["read_error"]
    )
    extensions = Counter(r["extension"] for r in all_file_rows)

    summary = {
        "scan_complete": True,
        "dataset_root_name": dataset_root.name,
        "all_file_count": len(all_file_rows),
        "image_file_count": len(image_rows),
        "non_image_file_count": len(all_file_rows) - len(image_rows),
        "folder_image_counts": {
            f"{split}|{label}": count
            for (split, label), count in sorted(folder_counts.items(), key=lambda x: (str(x[0][0]), str(x[0][1])))
        },
        "file_extension_counts": dict(sorted(extensions.items())),
        "image_property_counts": {
            f"{w}x{h}|{mode}|{fmt}": count
            for (w, h, mode, fmt), count in sorted(property_counts.items(), key=lambda x: str(x[0]))
        },
        "unreadable_image_count": len(unreadable),
        "filename_pattern_violation_count": len(filename_violations),
        "total_image_bytes": sum(r["size_bytes"] for r in image_rows),
        "minimum_image_size_bytes": min((r["size_bytes"] for r in image_rows), default=None),
        "maximum_image_size_bytes": max((r["size_bytes"] for r in image_rows), default=None),
        "preview_source_paths": preview_sources,
        "scope_note": "Raw inventory/metadata only. No duplicate analysis, leakage analysis, class-balance interpretation, feature analysis, or modeling is performed here."
    }
    (out / "raw_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
