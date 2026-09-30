from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont

EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"
EXPECTED_MANIFEST_SHA256 = "3fb8a29f1a51fc7930fe6c876d0fcb8d99ec59e147d4abbff503da4870f15b0b"
EXPECTED_O_SPLIT_SHA256 = "d48e182da1b170365b39693317e1edf6dd3395c02ccabdd6a771d7be5e226680"
EXPECTED_FALSE_NEGATIVE = "testing/flip/0041_000000010"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--error-cases", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Archive SHA-256 mismatch")
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Manifest SHA-256 mismatch")
    if sha256_file(args.split) != EXPECTED_O_SPLIT_SHA256:
        raise RuntimeError("O split SHA-256 mismatch")

    manifest = read_csv(args.manifest)
    split = read_csv(args.split)
    errors = read_csv(args.error_cases)
    manifest_by_id = {r["sample_id"]: r for r in manifest}
    split_by_id = {r["sample_id"]: r for r in split}

    false_negatives = [r for r in errors if r["kind"] == "false_negative"]
    if len(false_negatives) != 1 or false_negatives[0]["sample_id"] != EXPECTED_FALSE_NEGATIVE:
        raise RuntimeError(f"Unexpected Best-O false-negative set: {false_negatives}")

    selected: list[dict[str, str]] = []
    fn = false_negatives[0]
    target = split_by_id[fn["sample_id"]]
    target_video = target["video_id"]
    target_frame = int(target["frame_number"])

    selected.append({
        "sample_id": fn["sample_id"],
        "category": "false_negative",
        "prob_flip": fn["prob_flip"],
        "y_pred": fn["y_pred"],
    })

    same_video_train = [
        r for r in split
        if r["video_id"] == target_video and r["role"] == "train"
    ]
    same_video_train.sort(
        key=lambda r: (abs(int(r["frame_number"]) - target_frame), int(r["frame_number"]))
    )
    for r in same_video_train[:4]:
        selected.append({
            "sample_id": r["sample_id"],
            "category": "nearest_train_neighbor",
            "prob_flip": "",
            "y_pred": "",
        })

    for r in [x for x in errors if x["kind"] == "low_confidence_correct"][:8]:
        selected.append({
            "sample_id": r["sample_id"],
            "category": "low_confidence_correct",
            "prob_flip": r["prob_flip"],
            "y_pred": r["y_pred"],
        })

    unique = []
    seen = set()
    for r in selected:
        if r["sample_id"] not in seen:
            seen.add(r["sample_id"])
            unique.append(r)
    selected = unique

    for item in selected:
        if item["sample_id"] not in manifest_by_id or item["sample_id"] not in split_by_id:
            raise RuntimeError(f"Selected sample missing from frozen manifests: {item['sample_id']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    font = ImageFont.load_default()
    with ZipFile(args.archive) as zf:
        for item in selected:
            m = manifest_by_id[item["sample_id"]]
            s = split_by_id[item["sample_id"]]
            with zf.open(m["archive_member"]) as f:
                image = Image.open(f).convert("RGB")
                image.thumbnail((240, 426))
            card = Image.new("RGB", (280, 520), "white")
            card.paste(image, ((280 - image.width) // 2, 8))
            draw = ImageDraw.Draw(card)
            lines = [
                item["category"],
                item["sample_id"],
                f"frame={m['frame_number']} role={s['role']}",
            ]
            if item["prob_flip"]:
                lines.append(f"p(flip)={float(item['prob_flip']):.4f} pred={item['y_pred']}")
            y = 448
            for line in lines:
                draw.text((5, y), line, fill="black", font=font)
                y += 14
            cards.append(card)

    cols = 3
    rows = math.ceil(len(cards) / cols)
    sheet = Image.new("RGB", (cols * 280, rows * 520), (225, 225, 225))
    for i, card in enumerate(cards):
        sheet.paste(card, ((i % cols) * 280, (i // cols) * 520))
    sheet_path = args.output_dir / "o_visual_diagnostics.jpg"
    sheet.save(sheet_path, quality=92, optimize=True)

    selection_path = args.output_dir / "selection.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "category", "prob_flip", "y_pred"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected)

    receipt = args.output_dir / "receipt.txt"
    receipt.write_text(
        "\n".join([
            f"archive_sha256={sha256_file(args.archive)}",
            f"manifest_sha256={sha256_file(args.manifest)}",
            f"o_split_sha256={sha256_file(args.split)}",
            f"selected_count={len(selected)}",
            f"false_negative={EXPECTED_FALSE_NEGATIVE}",
            f"sheet_sha256={sha256_file(sheet_path)}",
            f"selection_sha256={sha256_file(selection_path)}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(receipt.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
