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
    naturalistic_palette_transfer,
    rgb_to_ycc709,
    stable_donor_order,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def decode_member(z: ZipFile, member: str) -> np.ndarray:
    data = z.read(member)
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def chroma_stats(rgb: np.ndarray, stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    _, cb, cr = rgb_to_ycc709(rgb)
    c = np.stack([cb[::stride, ::stride], cr[::stride, ::stride]], axis=-1)
    flat = c.reshape(-1, 2) / 128.0
    return flat.mean(axis=0), np.cov(flat, rowvar=False)


def psd_sqrt(mat: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) * 0.5)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def gaussian_w2_distance(
    mu_a: np.ndarray,
    cov_a: np.ndarray,
    mu_b: np.ndarray,
    cov_b: np.ndarray,
) -> float:
    dmu = mu_a - mu_b
    sqrt_a = psd_sqrt(cov_a)
    middle = sqrt_a @ cov_b @ sqrt_a
    sqrt_middle = psd_sqrt(middle)
    cov_term = np.trace(cov_a + cov_b - 2.0 * sqrt_middle)
    d2 = float(dmu @ dmu + max(float(cov_term), 0.0))
    return float(np.sqrt(max(d2, 0.0)))


def build_sheet(entries: list[dict], path: Path) -> None:
    cell = (190, 338)
    gap = 8
    label_h = 58
    rows = len(entries)
    canvas = Image.new(
        "RGB",
        (2 * cell[0] + 3 * gap, rows * (cell[1] + label_h + gap) + gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for i, e in enumerate(entries):
        y = gap + i * (cell[1] + label_h + gap)
        for col, (arr, label) in enumerate([
            (e["original"], "ORIGINAL"),
            (e["naturalistic"], "NATURALISTIC V2"),
        ]):
            x = gap + col * (cell[0] + gap)
            thumb = ImageOps.fit(
                Image.fromarray(arr).convert("RGB"),
                cell,
                method=Image.Resampling.BILINEAR,
            )
            canvas.paste(thumb, (x, y))
            draw.text((x, y + cell[1] + 3), label, fill="black")
        draw.text(
            (gap, y + cell[1] + 24),
            f'{e["sample_id"]} donor={e["donor_id"]} '
            f'd={e["donor_distance"]:.3f} '
            f'Y={e["metrics"]["luma_mae"]:.3f} '
            f'dC={e["metrics"]["median_chroma_displacement"]:.1f}',
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
    roles = {
        regime: {r["sample_id"]: r["role"] for r in rows}
        for regime, rows in splits.items()
    }

    env_by_id: dict[str, str] = {}
    for rows in splits.values():
        for r in rows:
            env_by_id.setdefault(r["sample_id"], r["environment_id"])

    donor_ids = sorted(
        sid for sid in by_id
        if all(roles[regime].get(sid) != "test" for regime in ("O", "S", "T", "ST"))
    )
    sample_rows = read_csv(args.sample)
    target_ids = [r["sample_id"] for r in sample_rows]

    if len(target_ids) != 100 or len(set(target_ids)) != 100:
        raise ValueError("Expected exact frozen 100-image sample")
    if any(roles["ST"].get(sid) != "context" for sid in target_ids):
        raise ValueError("Targets must all be ST context rows")

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

    with ZipFile(args.archive) as z:
        # Cache only donor stats actually requested by deterministic shortlists.
        stats_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        image_cache: dict[str, np.ndarray] = {}

        def get_image(sid: str) -> np.ndarray:
            if sid not in image_cache:
                image_cache[sid] = decode_member(z, by_id[sid]["archive_member"])
            return image_cache[sid]

        def get_stats(sid: str) -> tuple[np.ndarray, np.ndarray]:
            if sid not in stats_cache:
                stats_cache[sid] = chroma_stats(get_image(sid), stride=4)
            return stats_cache[sid]

        for i, sample_row in enumerate(sample_rows, 1):
            sid = sample_row["sample_id"]
            target_meta = by_id[sid]
            target_rgb = get_image(sid)
            mu_t, cov_t = get_stats(sid)

            eligible = [
                d for d in donor_ids
                if d != sid
                and by_id[d]["video_id"] != target_meta["video_id"]
                and env_by_id.get(d) != sample_row["environment_id"]
            ]
            if len(eligible) < 64:
                eligible = [
                    d for d in donor_ids
                    if d != sid and by_id[d]["video_id"] != target_meta["video_id"]
                ]
            ordered = stable_donor_order(sid, eligible)
            shortlist = ordered[:64]
            if not shortlist:
                raise RuntimeError(f"No donor shortlist for {sid}")

            scored = []
            for donor_id in shortlist:
                mu_d, cov_d = get_stats(donor_id)
                dist = gaussian_w2_distance(mu_t, cov_t, mu_d, cov_d)
                scored.append((dist, donor_id))
            scored.sort(key=lambda x: (-x[0], x[1]))
            donor_distance, donor_id = scored[0]

            donor_rgb = get_image(donor_id)
            natural, metrics = naturalistic_palette_transfer(
                target_rgb, donor_rgb, cfg=cfg
            )

            rec = {
                "sample_id": sid,
                "target_video_id": target_meta["video_id"],
                "target_environment_id": sample_row["environment_id"],
                "target_label_for_audit_only": target_meta["label"],
                "donor_id": donor_id,
                "donor_video_id": by_id[donor_id]["video_id"],
                "donor_environment_id": env_by_id.get(donor_id),
                "donor_label_for_audit_only": by_id[donor_id]["label"],
                "same_label_after_label_independent_selection": bool(
                    by_id[donor_id]["label"] == target_meta["label"]
                ),
                "same_video": bool(by_id[donor_id]["video_id"] == target_meta["video_id"]),
                "same_environment": bool(
                    env_by_id.get(donor_id) == sample_row["environment_id"]
                ),
                "shortlist_size": len(shortlist),
                "selected_gaussian_chroma_distance": donor_distance,
                "naturalistic": metrics,
            }
            records.append(rec)
            visuals.append({
                "sample_id": sid,
                "donor_id": donor_id,
                "donor_distance": donor_distance,
                "original": target_rgb,
                "naturalistic": natural,
                "metrics": metrics,
            })
            print(
                f"{i}/100 {sid} donor={donor_id} dist={donor_distance:.3f} "
                f"Y={metrics['luma_mae']:.3f} "
                f"dC={metrics['median_chroma_displacement']:.2f}",
                flush=True,
            )

    for start in range(0, 100, 10):
        build_sheet(
            visuals[start:start+10],
            qa_root / f"naturalistic_v2_{start//10+1:02d}.jpg",
        )

    y = np.asarray([r["naturalistic"]["luma_mae"] for r in records], dtype=float)
    dc = np.asarray(
        [r["naturalistic"]["median_chroma_displacement"] for r in records],
        dtype=float,
    )
    comp = np.asarray(
        [r["naturalistic"]["gamut_compressed_fraction"] for r in records],
        dtype=float,
    )
    dist = np.asarray(
        [r["selected_gaussian_chroma_distance"] for r in records],
        dtype=float,
    )

    summary = {
        "status": "NATURALISTIC_CHROMA_V2_100_READY_FOR_VISUAL_REVIEW",
        "images": 100,
        "primary_test_images_used": 0,
        "classifier_predictions_used": False,
        "target_label_used_for_donor_selection": False,
        "universal_donor_pool_size": len(donor_ids),
        "shortlist_size": 64,
        "strength": 1.0,
        "same_video_donors": int(sum(r["same_video"] for r in records)),
        "same_environment_donors": int(sum(r["same_environment"] for r in records)),
        "same_label_fraction_after_selection": float(
            np.mean([r["same_label_after_label_independent_selection"] for r in records])
        ),
        "selected_donor_distance": {
            "mean": float(dist.mean()),
            "median": float(np.median(dist)),
            "min": float(dist.min()),
        },
        "mean_luma_mae": float(y.mean()),
        "p95_per_image_luma_mae": float(np.quantile(y, 0.95)),
        "median_per_image_chroma_displacement": float(np.median(dc)),
        "p10_per_image_chroma_displacement": float(np.quantile(dc, 0.10)),
        "mean_gamut_compressed_fraction": float(comp.mean()),
        "luma_gate_pass": bool(y.mean() <= 0.5 and np.quantile(y, 0.95) <= 1.0),
        "nontrivial_chroma_gate_pass": bool(np.median(dc) >= 3.0),
        "quantitative_gate_pass": bool(
            y.mean() <= 0.5
            and np.quantile(y, 0.95) <= 1.0
            and np.median(dc) >= 3.0
            and sum(r["same_video"] for r in records) == 0
        ),
        "visual_gate": "PENDING",
    }

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
