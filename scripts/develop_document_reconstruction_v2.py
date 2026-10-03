from __future__ import annotations

import argparse
import base64
import io
import json
import os
import warnings
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoImageProcessor
from huggingface_hub import hf_hub_download
import onnxruntime as ort

from scripts.hand_arm_clean_dataset_onnx_tasks_v2 import (
    MODEL_ID,
    MODEL_REVISION,
    ONNX_FILE,
    TaskHandMasker,
)
from scripts.develop_video_assisted_inpainting_v2 import (
    decode_member,
    estimate_alignment,
    final_hand_arm_mask,
    read_csv,
    target_features,
)
from scripts.develop_lama_inpainting_v2 import run_lama_scaled


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    out = np.empty((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out


def detect_document_quad(rgb: np.ndarray, cfg: dict) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    max_side = int(cfg["document_normalization"]["max_detection_side"])
    scale = min(1.0, max_side / float(max(h, w)))
    sw, sh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    med = float(np.median(gray))
    lo = int(max(0, 0.55 * med))
    hi = int(min(255, max(lo + 1, 1.45 * med)))
    edges = cv2.Canny(gray, lo, hi)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ker, iterations=2)

    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, ker, iterations=2)

    candidates: list[tuple[float, np.ndarray]] = []
    image_area = float(sw * sh)
    min_frac = float(cfg["document_normalization"]["min_quad_area_fraction"])
    max_frac = float(cfg["document_normalization"]["max_quad_area_fraction"])

    for binary in (edges, bright):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            area = float(cv2.contourArea(cnt))
            frac = area / image_area
            if frac < min_frac or frac > max_frac:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.015, 0.02, 0.03, 0.04):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                q = order_quad(approx.reshape(4, 2))
                q_area = abs(float(cv2.contourArea(q.astype(np.float32))))
                if q_area <= 0:
                    continue
                rect = cv2.minAreaRect(q.astype(np.float32))
                rw, rh = rect[1]
                rect_area = max(float(rw * rh), 1.0)
                rectangularity = min(1.0, q_area / rect_area)
                score = frac * (0.65 + 0.35 * rectangularity)
                candidates.append((score, q))
                break

    if not candidates:
        return None
    q = max(candidates, key=lambda x: x[0])[1] / scale
    q[:, 0] = np.clip(q[:, 0], 0, w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    return q.astype(np.float32)


def page_mask_from_quad(shape: tuple[int, int], quad: np.ndarray | None) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if quad is None:
        mask[:] = 1
    else:
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
    return mask.astype(bool)


def canonical_document(rgb: np.ndarray, quad: np.ndarray | None, cfg: dict) -> np.ndarray:
    cw = int(cfg["document_normalization"]["canonical_width"])
    ch = int(cfg["document_normalization"]["canonical_height"])
    h, w = rgb.shape[:2]
    if quad is None:
        return cv2.resize(rgb, (cw, ch), interpolation=cv2.INTER_AREA)
    dst = np.asarray(
        [[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(order_quad(quad), dst)
    return cv2.warpPerspective(
        rgb, H, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def appearance_descriptor(rgb: np.ndarray, quad: np.ndarray | None, cfg: dict) -> np.ndarray:
    canon = canonical_document(rgb, quad, cfg)
    gray = cv2.cvtColor(canon, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    dw = int(cfg["retrieval"]["descriptor_width"])
    dh = int(cfg["retrieval"]["descriptor_height"])
    g = cv2.resize(gray.astype(np.float32), (dw, dh), interpolation=cv2.INTER_AREA)
    e = cv2.resize(grad, (dw, dh), interpolation=cv2.INTER_AREA)
    g = (g - g.mean()) / (g.std() + 1e-6)
    e = (e - e.mean()) / (e.std() + 1e-6)
    vec = np.concatenate([g.reshape(-1), e.reshape(-1)]).astype(np.float32)
    n = float(np.linalg.norm(vec))
    return vec / max(n, 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def detailed_document_alignment(
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    target_quad: np.ndarray,
    source_rgb: np.ndarray,
    source_mask: np.ndarray,
    source_quad: np.ndarray,
    similarity: float,
    cfg: dict,
) -> dict | None:
    h, w = target_rgb.shape[:2]
    acfg = cfg["alignment"]
    H0 = cv2.getPerspectiveTransform(order_quad(source_quad), order_quad(target_quad))

    source_warp0 = cv2.warpPerspective(
        source_rgb, H0, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    source_clean0 = cv2.warpPerspective(
        (source_mask == 0).astype(np.uint8), H0, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    valid0 = cv2.warpPerspective(
        np.ones(source_mask.shape, dtype=np.uint8), H0, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)

    page = page_mask_from_quad((h, w), target_quad)
    target_valid = ((target_mask == 0) & page).astype(np.uint8) * 255
    source_valid = (source_clean0 & valid0 & page).astype(np.uint8) * 255

    orb = cv2.ORB_create(
        nfeatures=int(acfg["orb_nfeatures"]),
        fastThreshold=5,
    )
    tg = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    sg = cv2.cvtColor(source_warp0, cv2.COLOR_RGB2GRAY)
    tkp, tdesc = orb.detectAndCompute(tg, target_valid)
    skp, sdesc = orb.detectAndCompute(sg, source_valid)
    min_good = int(acfg["minimum_good_matches"])
    if tdesc is None or sdesc is None or len(tkp) < min_good or len(skp) < min_good:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(sdesc, tdesc, k=2)
    ratio_thr = float(acfg["lowe_ratio"])
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < ratio_thr * n.distance]
    if len(good) < min_good:
        return None

    src_pts = np.float32([skp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([tkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hres, inlier_mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        float(acfg["ransac_reprojection_px"]),
    )
    if Hres is None or inlier_mask is None:
        return None
    inliers = int(inlier_mask.ravel().sum())
    inlier_ratio = inliers / max(len(good), 1)
    if inliers < int(acfg["minimum_inliers"]) or inlier_ratio < float(acfg["minimum_inlier_ratio"]):
        return None

    H = Hres @ H0
    warped = cv2.warpPerspective(
        source_rgb, H, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    source_clean = cv2.warpPerspective(
        (source_mask == 0).astype(np.uint8), H, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    valid = cv2.warpPerspective(
        np.ones(source_mask.shape, dtype=np.uint8), H, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)

    compare = (target_mask == 0) & source_clean & valid & page
    denom = max(int(((target_mask == 0) & page).sum()), 1)
    overlap = float(compare.sum() / denom)
    if overlap < float(acfg["minimum_clean_overlap_fraction"]):
        return None

    wg = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    errors = np.abs(tg.astype(np.int16) - wg.astype(np.int16))[compare]
    median_error = float(np.median(errors)) if errors.size else 999.0
    if median_error > float(acfg["maximum_median_gray_absolute_error"]):
        return None

    fill_valid = (target_mask > 0) & source_clean & valid & page
    if not np.any(fill_valid):
        return None

    quality = float(
        (inliers * inlier_ratio * max(overlap, 1e-6) * max(similarity, 0.05))
        / (1.0 + median_error)
    )
    return {
        "warped": warped,
        "fill_valid": fill_valid,
        "quality": quality,
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "overlap_fraction": overlap,
        "median_gray_abs_error": median_error,
        "document_similarity": similarity,
        "alignment_mode": "document_plane_orb_refined",
    }


def normalize_historical_alignment(a: dict, similarity: float) -> dict:
    return {
        "warped": a["warped"],
        "fill_valid": a["fill_valid"],
        "quality": float(a["quality"] * max(similarity, 0.05)),
        "good_matches": int(a["good_matches"]),
        "inliers": int(a["inliers"]),
        "inlier_ratio": float(a["inlier_ratio"]),
        "overlap_fraction": float(a["overlap_fraction"]),
        "median_gray_abs_error": float(a["median_gray_abs_error"]),
        "document_similarity": similarity,
        "alignment_mode": "same_video_historical_fallback",
    }


def robust_observed_medoid(
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    accepted: list[dict],
    max_donors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    h, w = target_mask.shape
    accepted = sorted(
        accepted,
        key=lambda x: (-x["alignment"]["quality"], x["sample_id"]),
    )[:max_donors]
    if not accepted:
        return target_rgb.copy(), np.zeros((h, w), dtype=np.uint16), np.zeros((h, w), bool), []

    colors = np.stack([x["alignment"]["warped"] for x in accepted], axis=0).astype(np.float32)
    valid = np.stack([x["alignment"]["fill_valid"] for x in accepted], axis=0)
    support = valid.sum(axis=0).astype(np.uint16)
    real = (target_mask > 0) & (support > 0)

    obs = colors.copy()
    obs[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(obs, axis=0)
    median = np.nan_to_num(median, nan=0.0)

    dist = np.sum((colors - median[None, ...]) ** 2, axis=-1)
    dist[~valid] = np.inf
    best = np.argmin(dist, axis=0)
    yy, xx = np.indices((h, w))
    out = target_rgb.copy()
    out[real] = colors[best[real], yy[real], xx[real]].astype(np.uint8)

    details = []
    for x in accepted:
        a = x["alignment"]
        details.append({
            "sample_id": x["sample_id"],
            "video_id": x["video_id"],
            "donor_type": x["donor_type"],
            "frame_distance": x.get("frame_distance"),
            "document_similarity": float(a["document_similarity"]),
            "alignment_mode": a["alignment_mode"],
            "good_matches": int(a["good_matches"]),
            "inliers": int(a["inliers"]),
            "inlier_ratio": float(a["inlier_ratio"]),
            "overlap_fraction": float(a["overlap_fraction"]),
            "median_gray_abs_error": float(a["median_gray_abs_error"]),
            "quality": float(a["quality"]),
            "fill_pixels": int(a["fill_valid"].sum()),
        })
    return out, support, real, details


def mask_preview(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    tint = np.zeros_like(out)
    tint[..., 0] = 255
    a = (mask > 0)[..., None].astype(np.float32) * 0.45
    return np.clip(out * (1 - a) + tint * a, 0, 255).astype(np.uint8)


def support_preview(rgb: np.ndarray, mask: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    residual = (mask > 0) & (support == 0)
    low = (mask > 0) & (support == 1)
    tint_res = np.zeros_like(out)
    tint_res[..., 2] = 255
    tint_low = np.zeros_like(out)
    tint_low[..., 0] = 255
    tint_low[..., 1] = 180
    a_res = residual[..., None].astype(np.float32) * 0.48
    a_low = low[..., None].astype(np.float32) * 0.25
    out = out * (1 - a_res) + tint_res * a_res
    out = out * (1 - a_low) + tint_low * a_low
    return np.clip(out, 0, 255).astype(np.uint8)


def build_sheet(entries: list[dict], path: Path) -> None:
    cell = (150, 267)
    gap = 7
    label_h = 54
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
            mask_preview(e["original"], e["mask"]),
            support_preview(e["mosaic"], e["mask"], e["support"]),
            e["final"],
        ]
        labels = [
            "BEFORE",
            "HAND/ARM MASK",
            f'OBSERVED real={e["real_fraction"]:.2f}',
            "FINAL + LAMA RESIDUAL",
        ]
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
            (gap, y + cell[1] + 25),
            f'{e["sample_id"]} donors={e["accepted"]} xvideo={e["cross_video"]} support2={e["support2"]:.2f}',
            fill="black",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)
    buf = io.BytesIO()
    canvas.resize(
        (canvas.width * 3 // 4, canvas.height * 3 // 4),
        Image.Resampling.LANCZOS,
    ).save(buf, format="JPEG", quality=65, optimize=True, subsampling=2)
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
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--hand-model", type=Path, required=True)
    ap.add_argument("--lama-model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    os.environ["LAMA_MODEL"] = str(args.lama_model)
    from simple_lama_inpainting import SimpleLama

    manifest = read_csv(args.manifest)
    by_id = {r["sample_id"]: r for r in manifest}
    st = read_csv(args.st_membership)
    targets = read_csv(args.sample)
    target_ids = [r["sample_id"] for r in targets]

    st_context = {r["sample_id"] for r in st if r["role"] == "context"}
    st_test = {r["sample_id"] for r in st if r["role"] == "test"}
    st_non_test = {r["sample_id"] for r in st if r["role"] != "test"}

    required = int(cfg["development_population"]["required_images"])
    if len(target_ids) != required or any(sid not in st_context for sid in target_ids):
        raise ValueError("Development targets do not match the frozen ST-context population")
    if set(target_ids) & st_test:
        raise ValueError("Primary-test target contamination")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    arm_model_path = hf_hub_download(MODEL_ID, ONNX_FILE, revision=MODEL_REVISION)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
    session = ort.InferenceSession(
        arm_model_path, opts, providers=["CPUExecutionProvider"]
    )
    hand_masker = TaskHandMasker(args.hand_model)
    lama = SimpleLama()

    source_info: dict[str, dict] = {}
    by_video: dict[str, list[str]] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_root = args.output_dir / "final"
    mosaic_root = args.output_dir / "observed_mosaic"
    confidence_root = args.output_dir / "confidence"
    qa_root = args.output_dir / "qa"

    records: list[dict] = []
    visuals: list[dict] = []

    try:
        with ZipFile(args.archive) as z:
            eligible_ids = sorted(st_non_test)
            print(f"Precomputing masks/document descriptors for {len(eligible_ids)} ST non-test rows", flush=True)
            for idx, sid in enumerate(eligible_ids, 1):
                meta = by_id[sid]
                rgb = decode_member(z, meta["archive_member"])
                mask, arm_px, hand_px, detected_hands = final_hand_arm_mask(
                    rgb, processor, session, hand_masker
                )
                quad = detect_document_quad(rgb, cfg)
                desc = appearance_descriptor(rgb, quad, cfg)
                source_info[sid] = {
                    "mask": mask,
                    "quad": quad,
                    "descriptor": desc,
                    "arm_pixels": arm_px,
                    "hand_pixels": hand_px,
                    "detected_hands": detected_hands,
                }
                by_video.setdefault(meta["video_id"], []).append(sid)
                if idx % 100 == 0 or idx == len(eligible_ids):
                    print(f"precompute {idx}/{len(eligible_ids)}", flush=True)

            cross_top_k = int(cfg["source_population"]["cross_video_retrieval_top_k"])
            min_sim = float(cfg["retrieval"]["minimum_cosine_similarity"])
            max_donors = int(cfg["alignment"]["max_accepted_donors"])

            for ti, sid in enumerate(target_ids, 1):
                meta = by_id[sid]
                target_rgb = decode_member(z, meta["archive_member"])
                target_info = source_info[sid]
                target_mask = target_info["mask"]
                target_quad = target_info["quad"]
                target_desc = target_info["descriptor"]
                _, target_kp, target_desc_orb = target_features(target_rgb, target_mask)

                same_ids = [
                    x for x in by_video.get(meta["video_id"], [])
                    if x != sid and x not in st_test
                ]
                same_ids.sort(
                    key=lambda x: (
                        abs(int(by_id[x]["frame_number"]) - int(meta["frame_number"])),
                        x,
                    )
                )

                scored_cross = []
                if target_quad is not None:
                    for x in eligible_ids:
                        if x == sid or by_id[x]["video_id"] == meta["video_id"]:
                            continue
                        sinfo = source_info[x]
                        if sinfo["quad"] is None:
                            continue
                        sim = cosine_similarity(target_desc, sinfo["descriptor"])
                        if sim >= min_sim:
                            scored_cross.append((sim, x))
                    scored_cross.sort(key=lambda t: (-t[0], t[1]))
                cross_ids = [x for _, x in scored_cross[:cross_top_k]]
                similarity_map = {x: s for s, x in scored_cross[:cross_top_k]}

                accepted: list[dict] = []
                attempts = {"same_video": 0, "cross_video": 0}
                for donor_type, ids in (("same_video", same_ids), ("cross_video", cross_ids)):
                    for src_id in ids:
                        if src_id in st_test:
                            raise RuntimeError("Forbidden ST test donor reached alignment")
                        attempts[donor_type] += 1
                        smeta = by_id[src_id]
                        sinfo = source_info[src_id]
                        source_rgb = decode_member(z, smeta["archive_member"])
                        sim = (
                            cosine_similarity(target_desc, sinfo["descriptor"])
                            if donor_type == "same_video"
                            else similarity_map[src_id]
                        )
                        a = None
                        if target_quad is not None and sinfo["quad"] is not None:
                            a = detailed_document_alignment(
                                target_rgb,
                                target_mask,
                                target_quad,
                                source_rgb,
                                sinfo["mask"],
                                sinfo["quad"],
                                sim,
                                cfg,
                            )
                        if a is None and donor_type == "same_video":
                            old = estimate_alignment(
                                target_rgb,
                                target_mask,
                                target_kp,
                                target_desc_orb,
                                source_rgb,
                                sinfo["mask"],
                            )
                            if old is not None:
                                a = normalize_historical_alignment(old, sim)
                        if a is not None:
                            accepted.append({
                                "sample_id": src_id,
                                "video_id": smeta["video_id"],
                                "donor_type": donor_type,
                                "frame_distance": (
                                    abs(int(smeta["frame_number"]) - int(meta["frame_number"]))
                                    if donor_type == "same_video" else None
                                ),
                                "alignment": a,
                            })

                mosaic, support, real_fill, donor_details = robust_observed_medoid(
                    target_rgb, target_mask, accepted, max_donors
                )
                mask_pixels = max(int((target_mask > 0).sum()), 1)
                real_fraction = float(real_fill.sum() / mask_pixels)
                residual = (target_mask > 0) & (~real_fill)
                residual_fraction = float(residual.sum() / mask_pixels)
                support2_fraction = float(((target_mask > 0) & (support >= 2)).sum() / mask_pixels)

                if np.any(residual):
                    generated = run_lama_scaled(
                        lama, mosaic, residual.astype(np.uint8) * 255
                    )
                    final = generated
                    final[~residual] = mosaic[~residual]
                else:
                    final = mosaic.copy()
                final[target_mask == 0] = target_rgb[target_mask == 0]

                for root, arr in ((mosaic_root, mosaic), (final_root, final)):
                    out = root / meta["archive_member"]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(arr).save(out, format="PNG")

                conf_path = confidence_root / (meta["archive_member"] + ".png")
                conf_path.parent.mkdir(parents=True, exist_ok=True)
                conf = np.clip(support.astype(np.float32) / max(max_donors, 1) * 255.0, 0, 255).astype(np.uint8)
                Image.fromarray(conf).save(conf_path)

                outside_changed = bool(
                    np.any(final[target_mask == 0] != target_rgb[target_mask == 0])
                )
                cross_accepted = sum(d["donor_type"] == "cross_video" for d in donor_details)
                same_accepted = sum(d["donor_type"] == "same_video" for d in donor_details)
                meets_gate = bool(
                    real_fraction >= float(cfg["reconstruction"]["real_pixel_gate"])
                    and residual_fraction <= float(cfg["reconstruction"]["residual_gate"])
                )

                rec = {
                    "sample_id": sid,
                    "video_id": meta["video_id"],
                    "frame_number": int(meta["frame_number"]),
                    "label": meta["label"],
                    "mask_fraction": float((target_mask > 0).mean()),
                    "arm_pixels": int(target_info["arm_pixels"]),
                    "hand_pixels": int(target_info["hand_pixels"]),
                    "detected_hands": int(target_info["detected_hands"]),
                    "document_quad_found": target_quad is not None,
                    "same_video_candidates": len(same_ids),
                    "cross_video_candidates": len(cross_ids),
                    "same_video_attempts": attempts["same_video"],
                    "cross_video_attempts": attempts["cross_video"],
                    "accepted_donors": len(donor_details),
                    "accepted_same_video": same_accepted,
                    "accepted_cross_video": cross_accepted,
                    "real_pixel_fill_fraction": real_fraction,
                    "multi_source_support_fraction": support2_fraction,
                    "residual_fraction": residual_fraction,
                    "meets_fill_gate": meets_gate,
                    "outside_mask_changed": outside_changed,
                    "accepted_donor_details": donor_details,
                }
                records.append(rec)
                visuals.append({
                    "sample_id": sid,
                    "original": target_rgb,
                    "mask": target_mask,
                    "mosaic": mosaic,
                    "support": support,
                    "final": final,
                    "real_fraction": real_fraction,
                    "support2": support2_fraction,
                    "accepted": len(donor_details),
                    "cross_video": cross_accepted,
                })
                print(
                    f"{ti}/{required} {sid} real={real_fraction:.3f} residual={residual_fraction:.3f} "
                    f"support2={support2_fraction:.3f} donors={len(donor_details)} xvideo={cross_accepted}",
                    flush=True,
                )
    finally:
        hand_masker.close()

    for start in range(0, len(visuals), 12):
        build_sheet(
            visuals[start:start + 12],
            qa_root / f"hand_reconstruction_v2_{start // 12 + 1}.jpg",
        )

    real = np.asarray([r["real_pixel_fill_fraction"] for r in records], dtype=float)
    residual = np.asarray([r["residual_fraction"] for r in records], dtype=float)
    support2 = np.asarray([r["multi_source_support_fraction"] for r in records], dtype=float)
    gate = np.asarray([r["meets_fill_gate"] for r in records], dtype=bool)
    accepted = np.asarray([r["accepted_donors"] for r in records], dtype=int)
    cross = np.asarray([r["accepted_cross_video"] for r in records], dtype=int)

    required_fraction = float(cfg["reconstruction"]["required_fraction_targets_passing"])
    summary = {
        "status": "HAND_ARM_RECONSTRUCTION_V2_48_READY_FOR_VISUAL_REVIEW",
        "images": len(records),
        "primary_test_images_used": 0,
        "classifier_predictions_used": False,
        "target_label_used_for_retrieval": False,
        "source_population": "ST non-test only",
        "method": "document-plane normalization + label-blind cross-video retrieval + ORB verification + observed-color medoid + LaMa residual",
        "document_quad_found_targets": int(sum(r["document_quad_found"] for r in records)),
        "real_pixel_fill_fraction": {
            "mean": float(real.mean()),
            "median": float(np.median(real)),
            "p10": float(np.quantile(real, 0.10)),
            "min": float(real.min()),
        },
        "multi_source_support_fraction": {
            "mean": float(support2.mean()),
            "median": float(np.median(support2)),
        },
        "residual_fraction": {
            "mean": float(residual.mean()),
            "median": float(np.median(residual)),
            "p90": float(np.quantile(residual, 0.90)),
            "max": float(residual.max()),
        },
        "accepted_donors": {
            "mean": float(accepted.mean()),
            "median": float(np.median(accepted)),
            "max": int(accepted.max()),
            "mean_cross_video": float(cross.mean()),
            "targets_with_cross_video_donor": int(np.sum(cross > 0)),
        },
        "images_meeting_fill_gate": int(gate.sum()),
        "fraction_images_meeting_fill_gate": float(gate.mean()),
        "required_fraction_images_meeting_fill_gate": required_fraction,
        "quantitative_gate_pass": bool(gate.mean() >= required_fraction),
        "outside_mask_changed_images": int(sum(r["outside_mask_changed"] for r in records)),
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
