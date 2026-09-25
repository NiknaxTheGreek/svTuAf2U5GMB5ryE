#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from pytesseract import Output

SAMPLE_POSITIONS = 5
MAX_PAGE_NUM = 1200

def sample_indices(n: int, k: int = SAMPLE_POSITIONS) -> list[int]:
    if n <= k:
        return list(range(n))
    vals = np.linspace(0, n - 1, k)
    return sorted(set(int(round(v)) for v in vals))

def prep(im: Image.Image) -> Image.Image:
    im = im.convert("L")
    # Keep enough resolution for small page numbers.
    target_w = 900
    if im.width != target_w:
        h = int(round(im.height * target_w / im.width))
        im = im.resize((target_w, h))
    im = ImageOps.autocontrast(im)
    im = im.filter(ImageFilter.SHARPEN)
    return im

def numeric_tokens(im: Image.Image):
    data = pytesseract.image_to_data(
        im,
        output_type=Output.DICT,
        config="--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789"
    )
    out = []
    W, H = im.size
    for i, raw in enumerate(data["text"]):
        s = re.sub(r"\D", "", raw or "")
        if not s:
            continue
        try:
            val = int(s)
        except ValueError:
            continue
        if val < 1 or val > MAX_PAGE_NUM:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        x, y = int(data["left"][i]), int(data["top"][i])
        w, h = int(data["width"][i]), int(data["height"][i])
        cx = (x + w/2) / W
        cy = (y + h/2) / H
        out.append({
            "value": val,
            "text": raw,
            "conf": conf,
            "x": x, "y": y, "w": w, "h": h,
            "cx": cx, "cy": cy,
        })
    return out

def candidate_score(t):
    # Favor high confidence, top/bottom margins, and outer halves.
    edge_y = min(t["cy"], 1 - t["cy"])
    edge_x = min(t["cx"], 1 - t["cx"])
    margin_bonus = 0.0
    if t["cy"] < 0.28 or t["cy"] > 0.72:
        margin_bonus += 1.2
    elif t["cy"] < 0.38 or t["cy"] > 0.62:
        margin_bonus += 0.4
    if t["cx"] < 0.45 or t["cx"] > 0.55:
        margin_bonus += 0.4
    return (t["conf"] / 100.0) + margin_bonus - 0.5 * edge_y - 0.15 * edge_x

def pick_page_candidates(tokens):
    ranked = sorted(tokens, key=candidate_score, reverse=True)
    # Keep a compact set for later sequence-level reasoning.
    keep = ranked[:8]
    left = [t for t in keep if t["cx"] < 0.5]
    right = [t for t in keep if t["cx"] >= 0.5]

    pair = None
    best_pair_score = -1e9
    for a in left:
        for b in right:
            diff = abs(a["value"] - b["value"])
            # Typical facing pages are consecutive or nearly consecutive.
            pair_bonus = 2.0 if diff == 1 else (0.8 if diff <= 3 else -0.8)
            parity_bonus = 0.25 if ((a["value"] % 2) != (b["value"] % 2)) else 0
            s = candidate_score(a) + candidate_score(b) + pair_bonus + parity_bonus
            if s > best_pair_score:
                best_pair_score = s
                pair = (a, b)

    return keep, pair, best_pair_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.inventory.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["video_id"])].append(r)

    tasks = []
    sequence_task_counts = {}
    for (label, vid), rr in sorted(groups.items()):
        rr = sorted(rr, key=lambda r: int(r["frame_num"]))
        chosen_idx = sample_indices(len(rr), SAMPLE_POSITIONS)
        sequence_task_counts[(label, vid)] = len(chosen_idx)
        for pos_idx, idx in enumerate(chosen_idx):
            tasks.append((label, vid, pos_idx, idx, rr[idx]))

    def process_task(task):
        label, vid, pos_idx, idx, r = task
        p = args.image_root / r["path"]
        if not p.exists():
            p = args.image_root / "images" / r["path"]
        with Image.open(p) as im0:
            im = prep(im0)
            toks = numeric_tokens(im)
        keep, pair, pair_score = pick_page_candidates(toks)
        left_val = pair[0]["value"] if pair else ""
        right_val = pair[1]["value"] if pair else ""
        return {
            "sequence_id": f"{label}__{vid}",
            "label": label,
            "video_id": vid,
            "sample_position_index": pos_idx,
            "sequence_frame_index": idx,
            "frame_num": int(r["frame_num"]),
            "path": r["path"],
            "numeric_token_count": len(toks),
            "top_candidates": json.dumps([
                {
                    "value": t["value"],
                    "conf": round(t["conf"], 1),
                    "cx": round(t["cx"], 3),
                    "cy": round(t["cy"], 3),
                    "score": round(candidate_score(t), 3),
                } for t in keep
            ]),
            "left_page_candidate": left_val,
            "right_page_candidate": right_val,
            "pair_score": round(pair_score, 4) if pair else "",
        }

    with ThreadPoolExecutor(max_workers=6) as ex:
        frame_rows = list(ex.map(process_task, tasks))
    frame_rows.sort(key=lambda r: (r["label"], r["video_id"], r["sample_position_index"]))

    seq_values = defaultdict(list)
    for r in frame_rows:
        if r["left_page_candidate"] != "":
            seq_values[(r["label"], r["video_id"])].append(int(r["left_page_candidate"]))
        if r["right_page_candidate"] != "":
            seq_values[(r["label"], r["video_id"])].append(int(r["right_page_candidate"]))

    seq_rows = []
    for (label, vid), rr in sorted(groups.items()):
        uniq = sorted(set(seq_values.get((label, vid), [])))
        seq_rows.append({
            "sequence_id": f"{label}__{vid}",
            "label": label,
            "video_id": vid,
            "sampled_frames": sequence_task_counts[(label, vid)],
            "detected_page_values": json.dumps(uniq),
            "detected_page_value_count": len(uniq),
            "min_detected_page": min(uniq) if uniq else "",
            "max_detected_page": max(uniq) if uniq else "",
        })

    with (args.out_dir / "page_number_frame_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        w.writeheader(); w.writerows(frame_rows)

    with (args.out_dir / "page_number_sequence_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
        w.writeheader(); w.writerows(seq_rows)

    summary = {
        "sequence_count": len(seq_rows),
        "sampled_frame_count": len(frame_rows),
        "sample_positions_per_sequence_target": SAMPLE_POSITIONS,
        "frames_with_any_numeric_token": sum(r["numeric_token_count"] > 0 for r in frame_rows),
        "frames_with_candidate_page_pair": sum(r["left_page_candidate"] != "" and r["right_page_candidate"] != "" for r in frame_rows),
        "sequences_with_at_least_one_page_pair": sum(r["detected_page_value_count"] > 0 for r in seq_rows),
        "scope_note": "Stage-1 OCR audit samples up to five frames per temporal clip. OCR candidates are noisy measurements, not verified page labels. Unreadable frames remain unresolved rather than forced."
    }
    (args.out_dir / "page_number_ocr_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
