from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.chroma_counterfactual_v2 import (
    PaletteTransferConfig,
    matched_chroma_rotation,
    naturalistic_palette_transfer,
    rec709_grayscale,
    stable_donor_order,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def decode_member(z: ZipFile, member: str) -> np.ndarray:
    data = z.read(member)
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def choose_donor(
    target_id: str,
    target_video: str,
    target_env: str,
    donor_ids: list[str],
    by_id: dict[str, dict[str, str]],
    env_by_id: dict[str, str],
) -> str:
    ordered = stable_donor_order(target_id, [x for x in donor_ids if x != target_id])

    preferred = [
        sid for sid in ordered
        if by_id[sid]["video_id"] != target_video and env_by_id.get(sid) != target_env
    ]
    if preferred:
        return preferred[0]

    different_video = [sid for sid in ordered if by_id[sid]["video_id"] != target_video]
    if different_video:
        return different_video[0]

    if ordered:
        return ordered[0]
    raise RuntimeError("No donor available")


def build_sheet(entries: list[dict], path: Path) -> None:
    cell = (150, 267)
    gap = 7
    label_h = 52
    rows = len(entries)
    canvas = Image.new(
        "RGB",
        (4 * cell[0] + 5 * gap, rows * (cell[1] + label_h + gap) + gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for i, e in enumerate(entries):
        y = gap + i * (cell[1] + label_h + gap)
        images = [
            e["original"],
            e["grayscale"],
            e["rotation"],
            e["naturalistic"],
        ]
        labels = [
            "ORIGINAL",
            "GRAYSCALE",
            "CHROMA +90",
            "NATURALISTIC",
        ]
        for col, (arr, label) in enumerate(zip(images, labels)):
            x = gap + col * (cell[0] + gap)
            thumb = ImageOps.fit(
                Image.fromarray(arr).convert("RGB"),
                cell,
                method=Image.Resampling.BILINEAR,
            )
            canvas.paste(thumb, (x, y))
            draw.text((x, y + cell[1] + 3), label, fill="black")
        draw.text(
            (gap, y + cell[1] + 23),
            f'{e["sample_id"]} donor={e["donor_id"]} '
            f'Ymae={e["naturalistic_metrics"]["luma_mae"]:.3f} '
            f'dC={e["naturalistic_metrics"]["median_chroma_displacement"]:.1f}',
            fill="black",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)
    buf = io.BytesIO()
    canvas.resize(
        (canvas.width * 3 // 4, canvas.height * 3 // 4),
        Image.Resampling.LANCZOS,
    ).save(buf, format="JPEG", quality=62, optimize=True, subsampling=2)
    path.with_suffix(".review.jpg.b64.txt").write_text(
        base64.b64encode(buf.getvalue()).decode("ascii") + "\n",
        encoding="ascii",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split-o", type=Path, required=True)
    ap.add_argument("--split-s", type=Path, required=True)
    ap.add_argument("--split-t", type=Path, required=True)
    ap.add_argument("--split-st", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = read_csv(args.manifest)
    by_id = {r["sample_id"]: r for r in manifest}
    splits = {
        "O": read_csv(args.split_o),
        "S": read_csv(args.split_s),
        "T": read_csv(args.split_t),
        "ST": read_csv(args.split_st),
    }
    role_by_regime = {
        regime: {r["sample_id"]: r["role"] for r in rows}
        for regime, rows in splits.items()
    }

    all_ids = set(by_id)
    donor_ids = sorted(
        sid for sid in all_ids
        if all(
            role_by_regime[regime].get(sid) != "test"
            for regime in ("O", "S", "T", "ST")
        )
    )
    if not donor_ids:
        raise RuntimeError("Universal non-test donor pool is empty")

    env_by_id: dict[str, str] = {}
    for rows in splits.values():
        for r in rows:
            env_by_id.setdefault(r["sample_id"], r["environment_id"])

    sample_rows = read_csv(args.sample)
    target_ids = [r["sample_id"] for r in sample_rows]
    if len(target_ids) != 100 or len(set(target_ids)) != 100:
        raise ValueError("Expected exact frozen 100-image PoC sample")

    st_role = role_by_regime["ST"]
    if any(st_role.get(sid) != "context" for sid in target_ids):
        raise ValueError("All chroma QA targets must be ST context-only")

    cfg = PaletteTransferConfig(
        strength=0.75,
        covariance_epsilon=1e-4,
        stats_stride=4,
        gamut_iterations=14,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_root = args.output_dir / "qa"
    records = []
    visuals = []

    with ZipFile(args.archive) as z:
        for i, sample_row in enumerate(sample_rows, 1):
            sid = sample_row["sample_id"]
            target_meta = by_id[sid]
            target_rgb = decode_member(z, target_meta["archive_member"])

            donor_id = choose_donor(
                sid,
                target_meta["video_id"],
                sample_row["environment_id"],
                donor_ids,
                by_id,
                env_by_id,
            )
            donor_meta = by_id[donor_id]
            donor_rgb = decode_member(z, donor_meta["archive_member"])

            gray = rec709_grayscale(target_rgb)
            rotation, rotation_metrics = matched_chroma_rotation(
                target_rgb, iterations=cfg.gamut_iterations
            )
            natural, natural_metrics = naturalistic_palette_transfer(
                target_rgb, donor_rgb, cfg=cfg
            )

            donor_label = donor_meta["label"]
            target_label = target_meta["label"]

            rec = {
                "sample_id": sid,
                "target_video_id": target_meta["video_id"],
                "target_environment_id": sample_row["environment_id"],
                "target_label_for_audit_only": target_label,
                "donor_id": donor_id,
                "donor_video_id": donor_meta["video_id"],
                "donor_environment_id": env_by_id.get(donor_id),
                "donor_label_for_audit_only": donor_label,
                "same_label_after_label_independent_selection": bool(donor_label == target_label),
                "same_video": bool(donor_meta["video_id"] == target_meta["video_id"]),
                "same_environment": bool(env_by_id.get(donor_id) == sample_row["environment_id"]),
                "rotation": rotation_metrics,
                "naturalistic": natural_metrics,
            }
            records.append(rec)
            visuals.append({
                "sample_id": sid,
                "donor_id": donor_id,
                "original": target_rgb,
                "grayscale": gray,
                "rotation": rotation,
                "naturalistic": natural,
                "naturalistic_metrics": natural_metrics,
            })

            print(
                f"{i}/100 {sid} donor={donor_id} "
                f"rotY={rotation_metrics['luma_mae']:.3f} "
                f"natY={natural_metrics['luma_mae']:.3f} "
                f"dC={natural_metrics['median_chroma_displacement']:.2f}",
                flush=True,
            )

    for start in range(0, 100, 10):
        build_sheet(
            visuals[start:start+10],
            qa_root / f"chroma_qa_{start//10+1:02d}.jpg",
        )

    rot_y = np.asarray([r["rotation"]["luma_mae"] for r in records], dtype=float)
    nat_y = np.asarray([r["naturalistic"]["luma_mae"] for r in records], dtype=float)
    nat_dc = np.asarray(
        [r["naturalistic"]["median_chroma_displacement"] for r in records],
        dtype=float,
    )
    rot_comp = np.asarray(
        [r["rotation"]["gamut_compressed_fraction"] for r in records],
        dtype=float,
    )
    nat_comp = np.asarray(
        [r["naturalistic"]["gamut_compressed_fraction"] for r in records],
        dtype=float,
    )

    summary = {
        "status": "CHROMA_100_READY_FOR_VISUAL_REVIEW",
        "images": 100,
        "primary_test_images_used": 0,
        "classifier_predictions_used": False,
        "universal_donor_pool_size": len(donor_ids),
        "target_label_used_for_donor_selection": False,
        "same_label_fraction_after_selection": float(
            np.mean([r["same_label_after_label_independent_selection"] for r in records])
        ),
        "same_video_donors": int(sum(r["same_video"] for r in records)),
        "same_environment_donors": int(sum(r["same_environment"] for r in records)),
        "matched_rotation": {
            "mean_luma_mae": float(rot_y.mean()),
            "p95_per_image_luma_mae": float(np.quantile(rot_y, 0.95)),
            "mean_gamut_compressed_fraction": float(rot_comp.mean()),
            "quantitative_luma_gate_pass": bool(rot_y.mean() <= 0.5),
        },
        "naturalistic": {
            "mean_luma_mae": float(nat_y.mean()),
            "p95_per_image_luma_mae": float(np.quantile(nat_y, 0.95)),
            "median_per_image_chroma_displacement": float(np.median(nat_dc)),
            "mean_gamut_compressed_fraction": float(nat_comp.mean()),
            "luma_gate_pass": bool(
                nat_y.mean() <= 0.5 and np.quantile(nat_y, 0.95) <= 1.0
            ),
            "nontrivial_chroma_gate_pass": bool(np.median(nat_dc) >= 3.0),
            "visual_gate": "PENDING",
        },
    }
    summary["quantitative_gate_pass"] = bool(
        summary["matched_rotation"]["quantitative_luma_gate_pass"]
        and summary["naturalistic"]["luma_gate_pass"]
        and summary["naturalistic"]["nontrivial_chroma_gate_pass"]
        and summary["same_video_donors"] == 0
    )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
