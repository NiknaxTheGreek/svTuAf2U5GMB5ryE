#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

VIDEO_IDS = ["0001", "0002", "0005", "0035", "0041"]

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def choose_endpoints(rows):
    rows = sorted(rows, key=lambda r: int(r["frame_num"]))
    if len(rows) == 1:
        return [rows[0], rows[0]]
    return [rows[0], rows[-1]]

def find_image(root: Path, rel: str) -> Path:
    p = root / rel
    if p.exists():
        return p
    p2 = root / "images" / rel
    if p2.exists():
        return p2
    raise FileNotFoundError(rel)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rows = read_rows(args.inventory)
    root = Path(args.image_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thumb_w, thumb_h = 240, 426
    header_h = 58
    row_label_w = 105
    gap = 10
    cols = [
        ("notflip", "early"),
        ("notflip", "late"),
        ("flip", "early"),
        ("flip", "late"),
    ]
    width = row_label_w + 4 * thumb_w + 5 * gap
    row_h = thumb_h + 62
    height = header_h + len(VIDEO_IDS) * row_h + gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = "Same VideoID across labels: raw-frame visual check"
    draw.text((10, 8), title, fill="black", font=font)
    for j, (label, pos) in enumerate(cols):
        x = row_label_w + gap + j * (thumb_w + gap)
        draw.text((x, 30), f"{label} / {pos}", fill="black", font=font)

    selected = []
    for i, vid in enumerate(VIDEO_IDS):
        y0 = header_h + i * row_h
        draw.text((10, y0 + 12), f"VideoID\n{vid}", fill="black", font=font)

        by_label = {}
        for label in ("notflip", "flip"):
            rr = [r for r in rows if r["video_id"] == vid and r["label"] == label]
            by_label[label] = choose_endpoints(rr) if rr else []

        slots = []
        for label in ("notflip", "flip"):
            chosen = by_label[label]
            if len(chosen) == 2:
                slots.extend([(label, "early", chosen[0]), (label, "late", chosen[1])])
            else:
                slots.extend([(label, "early", None), (label, "late", None)])

        for j, (label, pos, r) in enumerate(slots):
            x = row_label_w + gap + j * (thumb_w + gap)
            if r is None:
                draw.rectangle([x, y0, x+thumb_w, y0+thumb_h], outline="black")
                draw.text((x+10, y0+20), "missing", fill="black", font=font)
                continue
            p = find_image(root, r["path"])
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb_w, thumb_h))
                cell = Image.new("RGB", (thumb_w, thumb_h), "white")
                ox = (thumb_w - im.width)//2
                oy = (thumb_h - im.height)//2
                cell.paste(im, (ox, oy))
                canvas.paste(cell, (x, y0))
            cap = f'{r["split"]}/{label}\nframe {int(r["frame_num"])}'
            draw.text((x, y0 + thumb_h + 6), cap, fill="black", font=font)
            selected.append({
                "video_id": vid,
                "label": label,
                "position": pos,
                "split": r["split"],
                "frame_num": int(r["frame_num"]),
                "path": r["path"],
            })

    sheet = out / "same_video_cross_label_contact_sheet.jpg"
    canvas.save(sheet, quality=92)
    (out / "same_video_cross_label_contact_sheet.b64.txt").write_text(\n        base64.b64encode(sheet.read_bytes()).decode("ascii"), encoding="ascii"\n    )\n
    with open(out / "same_video_selected_frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id","label","position","split","frame_num","path"])
        w.writeheader()
        w.writerows(selected)

    print(f"Wrote {sheet}")
    print(f"Selected {len(selected)} raw frames")

if __name__ == "__main__":
    main()
