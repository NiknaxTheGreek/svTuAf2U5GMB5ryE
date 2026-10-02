from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

KR = 0.2126
KG = 0.7152
KB = 0.0722
DONOR_SEED = "monreader-chroma-donor-v2"
EPS = 1e-7


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def decode_member(z: ZipFile, member: str) -> np.ndarray:
    with z.open(member) as f:
        return np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)


def rgb_to_ycbcr709(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb_u8.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = KR * r + KG * g + KB * b
    cb = 0.5 + (b - y) / (2.0 * (1.0 - KB))
    cr = 0.5 + (r - y) / (2.0 * (1.0 - KR))
    return y, cb, cr


def ycbcr709_to_rgb_unclipped(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + 2.0 * (1.0 - KR) * (cr - 0.5)
    b = y + 2.0 * (1.0 - KB) * (cb - 0.5)
    g = (y - KR * r - KB * b) / KG
    return np.stack([r, g, b], axis=-1)


def rec709_luma(rgb01: np.ndarray) -> np.ndarray:
    return KR * rgb01[..., 0] + KG * rgb01[..., 1] + KB * rgb01[..., 2]


def clip_with_luma_correction(
    rgb_unclipped: np.ndarray,
    target_y: np.ndarray,
    iterations: int = 4,
) -> tuple[np.ndarray, float]:
    clip_fraction = float(np.mean((rgb_unclipped < 0.0) | (rgb_unclipped > 1.0)))
    rgb = np.clip(rgb_unclipped, 0.0, 1.0)
    for _ in range(iterations):
        delta = target_y - rec709_luma(rgb)
        rgb = np.clip(rgb + delta[..., None], 0.0, 1.0)
    return rgb, clip_fraction


def to_u8(rgb01: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgb01 * 255.0), 0, 255).astype(np.uint8)


def grayscale(rgb_u8: np.ndarray) -> np.ndarray:
    y, _, _ = rgb_to_ycbcr709(rgb_u8)
    yy = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(yy[..., None], 3, axis=2)


def matched_rotation(rgb_u8: np.ndarray):
    y, cb, cr = rgb_to_ycbcr709(rgb_u8)
    cbc = cb - 0.5
    crc = cr - 0.5
    cb2 = 0.5 - crc
    cr2 = 0.5 + cbc
    rgb_pre = ycbcr709_to_rgb_unclipped(y, cb2, cr2)
    rgb01, clip_fraction = clip_with_luma_correction(rgb_pre, y)
    out = to_u8(rgb01)
    final_y, _, _ = rgb_to_ycbcr709(out)
    err = np.abs(final_y - y) * 255.0
    mag0 = np.sqrt(cbc * cbc + crc * crc)
    mag1 = np.sqrt((cb2 - 0.5) ** 2 + (cr2 - 0.5) ** 2)
    return out, {
        "clip_fraction": clip_fraction,
        "luma_abs_error_mean": float(err.mean()),
        "luma_abs_error_p95": float(np.quantile(err, 0.95)),
        "chroma_magnitude_mae_preclip": float(np.mean(np.abs(mag1 - mag0))),
    }, err


def symmetric_matrix_power(cov: np.ndarray, power: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, EPS)
    return vecs @ np.diag(vals ** power) @ vecs.T


def naturalistic_transfer(target_u8: np.ndarray, donor_u8: np.ndarray):
    y, cb, cr = rgb_to_ycbcr709(target_u8)
    _, dcb, dcr = rgb_to_ycbcr709(donor_u8)

    x = np.stack([cb.ravel(), cr.ravel()], axis=1)
    d = np.stack([dcb.ravel(), dcr.ravel()], axis=1)
    mu_x = x.mean(axis=0)
    mu_d = d.mean(axis=0)
    cov_x = np.cov(x, rowvar=False) + np.eye(2) * EPS
    cov_d = np.cov(d, rowvar=False) + np.eye(2) * EPS

    whitening = symmetric_matrix_power(cov_x, -0.5)
    colouring = symmetric_matrix_power(cov_d, 0.5)
    a = colouring @ whitening
    xt = (x - mu_x) @ a.T + mu_d

    cb2 = xt[:, 0].reshape(cb.shape)
    cr2 = xt[:, 1].reshape(cr.shape)
    rgb_pre = ycbcr709_to_rgb_unclipped(y, cb2, cr2)
    rgb01, clip_fraction = clip_with_luma_correction(rgb_pre, y)
    out = to_u8(rgb01)

    final_y, _, _ = rgb_to_ycbcr709(out)
    err = np.abs(final_y - y) * 255.0

    _, out_cb, out_cr = rgb_to_ycbcr709(out)
    out_x = np.stack([out_cb.ravel(), out_cr.ravel()], axis=1)
    return out, {
        "clip_fraction": clip_fraction,
        "luma_abs_error_mean": float(err.mean()),
        "luma_abs_error_p95": float(np.quantile(err, 0.95)),
        "target_cbcr_mean_before": mu_x.tolist(),
        "donor_cbcr_mean": mu_d.tolist(),
        "output_cbcr_mean_after_clip": out_x.mean(axis=0).tolist(),
        "target_cbcr_cov_before": cov_x.tolist(),
        "donor_cbcr_cov": cov_d.tolist(),
        "output_cbcr_cov_after_clip": (np.cov(out_x, rowvar=False) + np.eye(2) * EPS).tolist(),
    }, err


def stable_key(target_id: str, donor_id: str) -> str:
    return hashlib.sha256(
        f"{DONOR_SEED}|{target_id}|{donor_id}".encode("utf-8")
    ).hexdigest()


def choose_donor(
    target_id: str,
    target_env: str,
    context_rows: list[dict[str, str]],
    st_by_id: dict[str, dict[str, str]],
) -> dict[str, str]:
    candidates = [
        r for r in context_rows
        if r["sample_id"] != target_id
        and st_by_id[r["sample_id"]]["environment_id"] != target_env
    ]
    if not candidates:
        candidates = [r for r in context_rows if r["sample_id"] != target_id]
    if not candidates:
        raise ValueError("No context donor candidates")
    candidates.sort(key=lambda r: (stable_key(target_id, r["sample_id"]), r["sample_id"]))
    return candidates[0]


def build_sheet(entries: list[dict], path: Path) -> None:
    cell = (150, 267)
    gap = 7
    label_h = 58
    rows = len(entries)
    canvas = Image.new(
        "RGB",
        (4 * cell[0] + 5 * gap, rows * (cell[1] + label_h + gap) + gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for i, e in enumerate(entries):
        y0 = gap + i * (cell[1] + label_h + gap)
        variants = [
            ("ORIGINAL", e["original"]),
            ("GRAYSCALE", e["grayscale"]),
            ("MATCHED +90", e["matched"]),
            ("NATURALISTIC", e["naturalistic"]),
        ]
        for col, (label, arr) in enumerate(variants):
            x0 = gap + col * (cell[0] + gap)
            thumb = ImageOps.fit(
                Image.fromarray(arr).convert("RGB"),
                cell,
                method=Image.Resampling.BILINEAR,
            )
            canvas.paste(thumb, (x0, y0))
            draw.text((x0, y0 + cell[1] + 3), label, fill="black")
        draw.text(
            (gap, y0 + cell[1] + 23),
            f'{e["sample_id"]} donor={e["donor_id"]} {e["target_env"]}->{e["donor_env"]}',
            fill="black",
        )
        draw.text(
            (gap, y0 + cell[1] + 39),
            f'match clip={e["matched_metrics"]["clip_fraction"]:.3f} '
            f'nat clip={e["natural_metrics"]["clip_fraction"]:.3f}',
            fill="black",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)
    buf = io.BytesIO()
    canvas.resize(
        (canvas.width * 3 // 4, canvas.height * 3 // 4),
        Image.Resampling.LANCZOS,
    ).save(buf, format="JPEG", quality=64, optimize=True, subsampling=2)
    path.with_suffix(".review.jpg.b64.txt").write_text(
        base64.b64encode(buf.getvalue()).decode("ascii") + "\n",
        encoding="ascii",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--st-membership", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = read_csv(args.manifest)
    by_id = {r["sample_id"]: r for r in manifest}
    st = read_csv(args.st_membership)
    st_by_id = {r["sample_id"]: r for r in st}
    sample = read_csv(args.sample)

    context_rows = [r for r in st if r["role"] == "context"]
    context_ids = {r["sample_id"] for r in context_rows}
    test_ids = {r["sample_id"] for r in st if r["role"] == "test"}
    target_ids = [r["sample_id"] for r in sample]

    if len(target_ids) != 100 or len(set(target_ids)) != 100:
        raise ValueError("Expected exact frozen 100-image sample")
    if any(sid not in context_ids for sid in target_ids):
        raise ValueError("Chroma QA target outside ST context")
    if set(target_ids) & test_ids:
        raise ValueError("Primary-test contamination")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_root = args.output_dir / "qa"
    records = []
    visuals = []
    matched_errors = []
    natural_errors = []
    matched_clip_num = 0.0
    natural_clip_num = 0.0

    with ZipFile(args.archive) as z:
        for i, row in enumerate(sample, 1):
            sid = row["sample_id"]
            meta = by_id[sid]
            original = decode_member(z, meta["archive_member"])
            gray = grayscale(original)

            matched, mm, me = matched_rotation(original)

            donor_row = choose_donor(
                sid,
                st_by_id[sid]["environment_id"],
                context_rows,
                st_by_id,
            )
            donor_meta = by_id[donor_row["sample_id"]]
            donor_st = st_by_id[donor_row["sample_id"]]
            donor = decode_member(z, donor_meta["archive_member"])
            natural, nm, ne = naturalistic_transfer(original, donor)

            pixels = original.shape[0] * original.shape[1] * 3
            matched_clip_num += mm["clip_fraction"] * pixels
            natural_clip_num += nm["clip_fraction"] * pixels
            matched_errors.append(me.ravel())
            natural_errors.append(ne.ravel())

            rec = {
                "sample_id": sid,
                "target_environment_id": st_by_id[sid]["environment_id"],
                "target_label_recorded_but_not_used_for_donor": meta["label"],
                "donor_sample_id": donor_row["sample_id"],
                "donor_environment_id": donor_st_by_id[sid]["environment_id"],
                "donor_label_recorded_but_not_used_for_selection": donor_meta["label"],
                "different_environment": donor_st_by_id[sid]["environment_id"] != st_by_id[sid]["environment_id"],
                "matched": mm,
                "naturalistic": nm,
                "shape": list(original.shape),
            }
            records.append(rec)
            visuals.append({
                "sample_id": sid,
                "target_env": st_by_id[sid]["environment_id"],
                "donor_id": donor_row["sample_id"],
                "donor_env": donor_st_by_id[sid]["environment_id"],
                "original": original,
                "grayscale": gray,
                "matched": matched,
                "naturalistic": natural,
                "matched_metrics": mm,
                "natural_metrics": nm,
            })
            print(
                f"{i}/100 {sid} donor={donor_row['sample_id']} "
                f"match_luma={mm['luma_abs_error_mean']:.3f} "
                f"nat_luma={nm['luma_abs_error_mean']:.3f}",
                flush=True,
            )

    for start in range(0, 100, 10):
        build_sheet(
            visuals[start:start+10],
            qa_root / f"chroma_qa_{start//10+1:02d}.jpg",
        )

    matched_err = np.concatenate(matched_errors)
    natural_err = np.concatenate(natural_errors)
    total_channels = sum(np.prod(r["shape"]) for r in records)

    matched_summary = {
        "mean_absolute_luminance_error": float(matched_err.mean()),
        "p95_absolute_luminance_error": float(np.quantile(matched_err, 0.95)),
        "rgb_gamut_clip_fraction": float(matched_clip_num / total_channels),
    }
    natural_summary = {
        "mean_absolute_luminance_error": float(natural_err.mean()),
        "p95_absolute_luminance_error": float(np.quantile(natural_err, 0.95)),
        "rgb_gamut_clip_fraction": float(natural_clip_num / total_channels),
    }
    matched_gate = (
        matched_summary["mean_absolute_luminance_error"] <= 1.0
        and matched_summary["p95_absolute_luminance_error"] <= 3.0
        and matched_summary["rgb_gamut_clip_fraction"] <= 0.20
    )
    natural_quant_gate = (
        natural_summary["mean_absolute_luminance_error"] <= 1.0
        and natural_summary["p95_absolute_luminance_error"] <= 3.0
        and natural_summary["rgb_gamut_clip_fraction"] <= 0.15
    )

    summary = {
        "status": "CHROMA_100_READY_FOR_VISUAL_REVIEW",
        "images": 100,
        "primary_test_images_used": 0,
        "classifier_predictions_used": False,
        "donor_selection_uses_target_label": False,
        "different_environment_donors": int(
            sum(r["different_environment"] for r in records)
        ),
        "matched_chroma_rotation": matched_summary,
        "matched_quantitative_gate_pass": bool(matched_gate),
        "naturalistic_palette_counterfactual": natural_summary,
        "naturalistic_quantitative_gate_pass": bool(natural_quant_gate),
        "naturalistic_visual_gate": "PENDING",
        "all_geometry_unchanged_by_definition": True,
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
