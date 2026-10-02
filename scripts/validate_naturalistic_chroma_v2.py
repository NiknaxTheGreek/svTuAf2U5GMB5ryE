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
    rgb_to_ycc709,
    stable_donor_order,
)

SELECTION_STATS_STRIDE = 16
SHORTLIST_SIZE = 16


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def decode_member(z: ZipFile, member: str) -> np.ndarray:
    with Image.open(io.BytesIO(z.read(member))) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def chroma_stats(rgb: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    _, cb, cr = rgb_to_ycc709(rgb)
    c = np.stack([cb[::stride, ::stride], cr[::stride, ::stride]], axis=-1)
    c = c.reshape(-1, 2) / 128.0
    return c.mean(axis=0), np.cov(c, rowvar=False)


def palette_distance(
    target_stats: tuple[np.ndarray, np.ndarray],
    donor_stats: tuple[np.ndarray, np.ndarray],
) -> float:
    mu_t, cov_t = target_stats
    mu_d, cov_d = donor_stats
    return float(
        np.linalg.norm(mu_d - mu_t)
        + np.linalg.norm(cov_d - cov_t, ord="fro")
    )


def choose_donor_v2(
    target_id: str,
    target_video: str,
    target_env: str,
    donor_ids: list[str],
    by_id: dict[str, dict[str, str]],
    env_by_id: dict[str, str],
    target_rgb: np.ndarray,
    z: ZipFile,
    stats_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[str, float, list[str]]:
    eligible = [
        sid for sid in donor_ids
        if sid != target_id
        and by_id[sid]["video_id"] != target_video
        and env_by_id.get(sid) != target_env
    ]
    if len(eligible) < SHORTLIST_SIZE:
        raise RuntimeError(
            f"Only {len(eligible)} eligible donors for {target_id}, "
            f"need {SHORTLIST_SIZE}"
        )

    ordered = stable_donor_order(target_id, eligible)
    shortlist = ordered[:SHORTLIST_SIZE]
    target_stats = chroma_stats(target_rgb, SELECTION_STATS_STRIDE)

    best_id = None
    best_distance = -1.0
    for donor_id in shortlist:
        if donor_id not in stats_cache:
            donor_rgb = decode_member(z, by_id[donor_id]["archive_member"])
            stats_cache[donor_id] = chroma_stats(
                donor_rgb, SELECTION_STATS_STRIDE
            )
        dist = palette_distance(target_stats, stats_cache[donor_id])
        if dist > best_distance:
            best_id = donor_id
            best_distance = dist

    if best_id is None:
        raise RuntimeError("Failed to select donor")
    return best_id, best_distance, shortlist


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
        imgs = [
            e["original"],
            e["grayscale"],
            e["rotation"],
            e["naturalistic"],
        ]
        labels = ["ORIGINAL", "GRAYSCALE", "CHROMA +90", "NATURALISTIC V2"]
        for col, (arr, label) in enumerate(zip(imgs, labels)):
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
    ap.add_argument("--development-sample", type=Path, required=True)
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
    env_by_id: dict[str, str] = {}
    for rows in splits.values():
        for r in rows:
            env_by_id.setdefault(r["sample_id"], r["environment_id"])

    donor_ids = sorted(
        sid for sid in by_id
        if all(
            role_by_regime[regime].get(sid) != "test"
            for regime in ("O", "S", "T", "ST")
        )
    )

    sample_rows = read_csv(args.sample)
    target_ids = [r["sample_id"] for r in sample_rows]
    development_ids = {
        r["sample_id"] for r in read_csv(args.development_sample)
    }
    if len(target_ids) != 100 or len(set(target_ids)) != 100:
        raise ValueError("Expected exact 100-image validation sample")
    if set(target_ids) & development_ids:
        raise ValueError("Development/validation overlap")
    if any(role_by_regime["ST"].get(sid) != "context" for sid in target_ids):
        raise ValueError("Validation targets are not all ST context-only")

    cfg = PaletteTransferConfig(
        strength=1.0,
        covariance_epsilon=1e-4,
        stats_stride=4,
        gamut_iterations=14,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_root = args.output_dir / "qa"
    records = []
    visuals = []
    stats_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    with ZipFile(args.archive) as z:
        for i, row in enumerate(sample_rows, 1):
            sid = row["sample_id"]
            target_meta = by_id[sid]
            target_rgb = decode_member(z, target_meta["archive_member"])

            donor_id, donor_distance, shortlist = choose_donor_v2(
                sid,
                target_meta["video_id"],
                row["environment_id"],
                donor_ids,
                by_id,
                env_by_id,
                target_rgb,
                z,
                stats_cache,
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

            rec = {
                "sample_id": sid,
                "target_video_id": target_meta["video_id"],
                "target_environment_id": row["environment_id"],
                "target_label_for_audit_only": target_meta["label"],
                "donor_id": donor_id,
                "donor_video_id": donor_meta["video_id"],
                "donor_environment_id": env_by_id.get(donor_id),
                "donor_label_for_audit_only": donor_meta["label"],
                "same_label_after_label_independent_selection": bool(
                    donor_meta["label"] == target_meta["label"]
                ),
                "same_video": bool(
                    donor_meta["video_id"] == target_meta["video_id"]
                ),
                "same_environment": bool(
                    env_by_id.get(donor_id) == row["environment_id"]
                ),
                "donor_palette_distance": donor_distance,
                "donor_shortlist_size": len(shortlist),
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
                f"dist={donor_distance:.4f} "
                f"Y={natural_metrics['luma_mae']:.3f} "
                f"dC={natural_metrics['median_chroma_displacement']:.2f}",
                flush=True,
            )

    for start in range(0, 100, 10):
        build_sheet(
            visuals[start:start+10],
            qa_root / f"naturalistic_v2_{start//10+1:02d}.jpg",
        )

    nat_y = np.asarray(
        [r["naturalistic"]["luma_mae"] for r in records], dtype=float
    )
    nat_dc = np.asarray(
        [
            r["naturalistic"]["median_chroma_displacement"]
            for r in records
        ],
        dtype=float,
    )
    nat_comp = np.asarray(
        [
            r["naturalistic"]["gamut_compressed_fraction"]
            for r in records
        ],
        dtype=float,
    )
    distances = np.asarray(
        [r["donor_palette_distance"] for r in records], dtype=float
    )

    summary = {
        "status": "NATURALISTIC_CHROMA_V2_VALIDATION_READY_FOR_VISUAL_REVIEW",
        "images": 100,
        "development_overlap": 0,
        "primary_test_images_used": 0,
        "classifier_predictions_used": False,
        "target_label_used_for_donor_selection": False,
        "universal_donor_pool_size": len(donor_ids),
        "donor_shortlist_size": SHORTLIST_SIZE,
        "donor_selection_stats_stride": SELECTION_STATS_STRIDE,
        "same_video_donors": int(sum(r["same_video"] for r in records)),
        "same_environment_donors": int(
            sum(r["same_environment"] for r in records)
        ),
        "same_label_fraction_after_selection": float(
            np.mean(
                [
                    r["same_label_after_label_independent_selection"]
                    for r in records
                ]
            )
        ),
        "mean_selected_palette_distance": float(distances.mean()),
        "naturalistic": {
            "strength": 1.0,
            "mean_luma_mae": float(nat_y.mean()),
            "p95_per_image_luma_mae": float(np.quantile(nat_y, 0.95)),
            "median_per_image_chroma_displacement": float(np.median(nat_dc)),
            "mean_gamut_compressed_fraction": float(nat_comp.mean()),
            "luma_gate_pass": bool(
                nat_y.mean() <= 0.5 and np.quantile(nat_y, 0.95) <= 1.0
            ),
            "nontrivial_chroma_gate_pass": bool(
                np.median(nat_dc) >= 3.0
            ),
            "visual_gate": "PENDING",
        },
    }
    summary["quantitative_gate_pass"] = bool(
        summary["naturalistic"]["luma_gate_pass"]
        and summary["naturalistic"]["nontrivial_chroma_gate_pass"]
        and summary["same_video_donors"] == 0
        and summary["same_environment_donors"] == 0
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
