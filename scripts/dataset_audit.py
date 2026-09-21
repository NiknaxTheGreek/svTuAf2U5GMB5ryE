#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMG_EXTS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
NAME_RE=re.compile(r"^(?P<video_id>.+)_(?P<frame_num>\\d+)$")

def resolve_root(root: Path) -> Path:
    root=root.resolve()
    if (root/"training").is_dir() and (root/"testing").is_dir():
        return root
    if (root/"images"/"training").is_dir() and (root/"images"/"testing").is_dir():
        return root/"images"
    raise FileNotFoundError(f"Could not find training/ and testing/ under {root} or {root/'images'}")

def sha256_file(path: Path, chunk_size: int=1024*1024) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def dhash64(path: Path) -> int:
    with Image.open(path) as img:
        img=img.convert("L").resize((9,8), Image.Resampling.LANCZOS)
        px=list(img.getdata())
    bits=0
    bitpos=0
    for row in range(8):
        offset=row*9
        for col in range(8):
            if px[offset+col] > px[offset+col+1]:
                bits |= 1 << bitpos
            bitpos += 1
    return bits

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def safe_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--near-threshold", type=int, default=4)
    args=p.parse_args()

    root=resolve_root(args.root)
    out=args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    records=[]
    unreadable=[]
    filename_violations=[]
    unexpected_labels=[]
    sha_to_rows=defaultdict(list)

    for split in ("training","testing"):
        split_dir=root/split
        if not split_dir.is_dir():
            raise AssertionError(f"Missing split directory: {split_dir}")
        for label_dir in sorted(x for x in split_dir.iterdir() if x.is_dir()):
            label=label_dir.name
            if label not in {"flip","notflip"}:
                unexpected_labels.append(str(label_dir))
            for path in sorted(x for x in label_dir.rglob("*") if x.is_file() and x.suffix.lower() in IMG_EXTS):
                m=NAME_RE.match(path.stem)
                if m:
                    video_id=m.group("video_id")
                    frame_num=int(m.group("frame_num"))
                else:
                    video_id=None
                    frame_num=None
                    filename_violations.append(safe_rel(path,root))

                width=height=mode=fmt=sha=dh=err=None
                try:
                    with Image.open(path) as img:
                        img.verify()
                    with Image.open(path) as img:
                        width,height=img.size
                        mode=img.mode
                        fmt=img.format
                    sha=sha256_file(path)
                    dh=dhash64(path)
                except Exception as e:
                    err=repr(e)
                    unreadable.append({"path":safe_rel(path,root),"error":err})

                row={
                    "split":split,
                    "label":label,
                    "path":safe_rel(path,root),
                    "filename":path.name,
                    "video_id":video_id,
                    "frame_num":frame_num,
                    "width":width,
                    "height":height,
                    "mode":mode,
                    "format":fmt,
                    "size_bytes":path.stat().st_size,
                    "sha256":sha,
                    "dhash64_hex":None if dh is None else f"{dh:016x}",
                    "read_error":err,
                }
                records.append(row)
                if sha:
                    sha_to_rows[sha].append(row)

    frame_counts=Counter((r["split"],r["label"]) for r in records)
    prop_counts=Counter(
        (r["width"],r["height"],r["mode"],r["format"])
        for r in records if r["read_error"] is None
    )

    clip_frames=defaultdict(list)
    for r in records:
        if r["video_id"] is not None:
            clip_frames[(r["split"],r["label"],r["video_id"])].append(r["frame_num"])

    clip_rows=[]
    for (split,label,vid),frames in sorted(clip_frames.items()):
        clip_rows.append({
            "split":split,
            "label":label,
            "video_id":vid,
            "frame_count":len(frames),
            "min_frame":min(frames),
            "max_frame":max(frames),
        })

    exact_groups=[]
    cross_split_exact=[]
    for sha,rows in sha_to_rows.items():
        if len(rows)<=1:
            continue
        g={
            "sha256":sha,
            "count":len(rows),
            "paths":[r["path"] for r in rows],
            "splits":sorted({r["split"] for r in rows}),
            "labels":sorted({r["label"] for r in rows}),
        }
        exact_groups.append(g)
        if len(g["splits"])>1:
            cross_split_exact.append(g)

    train=[(r,int(r["dhash64_hex"],16)) for r in records if r["split"]=="training" and r["dhash64_hex"]]
    test=[(r,int(r["dhash64_hex"],16)) for r in records if r["split"]=="testing" and r["dhash64_hex"]]

    nearest_rows=[]
    near_pairs=[]
    for test_row,test_hash in test:
        best_dist=65
        best_train=None
        for train_row,train_hash in train:
            d=hamming64(test_hash,train_hash)
            if d<best_dist:
                best_dist=d
                best_train=train_row
                if d==0:
                    break
        if best_train is not None:
            item={
                "test_path":test_row["path"],
                "test_label":test_row["label"],
                "test_video_id":test_row["video_id"],
                "test_frame_num":test_row["frame_num"],
                "nearest_train_path":best_train["path"],
                "nearest_train_label":best_train["label"],
                "nearest_train_video_id":best_train["video_id"],
                "nearest_train_frame_num":best_train["frame_num"],
                "dhash_hamming_distance":best_dist,
                "same_label":test_row["label"]==best_train["label"],
                "same_label_video_id":(
                    test_row["label"]==best_train["label"]
                    and test_row["video_id"]==best_train["video_id"]
                ),
                "frame_num_delta":(
                    None if test_row["frame_num"] is None or best_train["frame_num"] is None
                    else abs(test_row["frame_num"]-best_train["frame_num"])
                ),
            }
            nearest_rows.append(item)
            if best_dist<=args.near_threshold:
                near_pairs.append(item)

    write_csv(out/"image_inventory.csv",records)
    write_csv(out/"clip_inventory.csv",clip_rows)
    write_csv(out/"cross_split_nearest_dhash.csv",nearest_rows)
    write_csv(out/"cross_split_near_duplicate_candidates.csv",near_pairs)
    (out/"exact_duplicate_groups.json").write_text(json.dumps(exact_groups,indent=2),encoding="utf-8")
    (out/"cross_split_exact_duplicate_groups.json").write_text(json.dumps(cross_split_exact,indent=2),encoding="utf-8")
    (out/"unreadable.json").write_text(json.dumps(unreadable,indent=2),encoding="utf-8")

    training_clip_keys={(r["label"],r["video_id"]) for r in records if r["split"]=="training" and r["video_id"] is not None}
    testing_clip_keys={(r["label"],r["video_id"]) for r in records if r["split"]=="testing" and r["video_id"] is not None}
    shared_clip_keys=training_clip_keys & testing_clip_keys

    summary={
        "dataset_root":str(root),
        "total_images":len(records),
        "frame_counts":{f"{s}|{l}":n for (s,l),n in sorted(frame_counts.items())},
        "class_totals":{l:sum(n for (s,ll),n in frame_counts.items() if ll==l) for l in sorted({ll for _,ll in frame_counts})},
        "folder_defined_clip_count":len(clip_rows),
        "clip_counts":dict(Counter(f"{r['split']}|{r['label']}" for r in clip_rows)),
        "shared_label_video_id_count":len(shared_clip_keys),
        "shared_label_video_id_counts_by_label":{
            label:sum(1 for lab,_ in shared_clip_keys if lab==label)
            for label in sorted({lab for lab,_ in shared_clip_keys})
        },
        "image_property_counts":{f"{w}x{h}|{m}|{fmt}":n for (w,h,m,fmt),n in prop_counts.items()},
        "unreadable_count":len(unreadable),
        "filename_violation_count":len(filename_violations),
        "unexpected_label_directory_count":len(unexpected_labels),
        "exact_duplicate_group_count":len(exact_groups),
        "cross_split_exact_duplicate_group_count":len(cross_split_exact),
        "dhash_near_threshold":args.near_threshold,
        "cross_split_near_duplicate_candidate_count":len(near_pairs),
        "minimum_cross_split_dhash_distance":min((r["dhash_hamming_distance"] for r in nearest_rows),default=None),
        "near_candidate_same_label_count":sum(bool(r["same_label"]) for r in near_pairs),
        "near_candidate_cross_label_count":sum(not bool(r["same_label"]) for r in near_pairs),
        "near_candidate_same_label_video_id_count":sum(bool(r["same_label_video_id"]) for r in near_pairs),
        "near_candidate_same_label_video_id_adjacent_frame_le_3_count":sum(
            bool(r["same_label_video_id"]) and r["frame_num_delta"] is not None and r["frame_num_delta"]<=3
            for r in near_pairs
        ),
        "decision_note":"dHash candidates are screening evidence, not automatic proof of duplicate identity; inspect flagged pairs before declaring the supplied split leakage-safe."
    }
    (out/"audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()

# Workflow trigger: authoritative audit execution
