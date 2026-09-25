#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import numpy as np
import imagehash
from skimage.metrics import structural_similarity as ssim

SIZE=(256,455)
D_THR=4
P_THR=4
SSIM_THR=0.95

class DSU:
    def __init__(self,n):
        self.p=list(range(n)); self.r=[0]*n
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]: ra,rb=rb,ra
        self.p[rb]=ra
        if self.r[ra]==self.r[rb]: self.r[ra]+=1

def load_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize(SIZE),dtype=np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    with a.inventory.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    n=len(rows)

    dh=[]; ph=[]
    paths=[]
    seq=[]
    for r in rows:
        p=a.image_root/r["path"]
        if not p.exists(): p=a.image_root/"images"/r["path"]
        paths.append(p)
        with Image.open(p) as im:
            rgb=im.convert("RGB")
            dh.append(int(str(imagehash.dhash(rgb,hash_size=8)),16))
            ph.append(int(str(imagehash.phash(rgb,hash_size=8)),16))
        seq.append(f'{r["label"]}__{r["video_id"]}')

    screened=[]
    needed=set()
    for i in range(n-1):
        for j in range(i+1,n):
            d=(dh[i]^dh[j]).bit_count()
            if d>D_THR: continue
            p=(ph[i]^ph[j]).bit_count()
            if p>P_THR: continue
            screened.append((i,j,d,p))
            needed.add(i); needed.add(j)

    cache={i:load_gray(paths[i]) for i in sorted(needed)}
    confirmed=[]
    dsu=DSU(n)
    buckets=Counter()
    for i,j,d,p in screened:
        score=float(ssim(cache[i],cache[j],data_range=255))
        if score < SSIM_THR:
            continue
        same_seq=seq[i]==seq[j]
        same_split=rows[i]["split"]==rows[j]["split"]
        same_label=rows[i]["label"]==rows[j]["label"]
        if same_seq and same_split:
            bucket="same_sequence_same_split"
        elif same_seq:
            bucket="same_sequence_cross_split"
        elif same_split and same_label:
            bucket="different_sequence_same_split_same_label"
        elif same_split:
            bucket="different_sequence_same_split_cross_label"
        elif same_label:
            bucket="different_sequence_cross_split_same_label"
        else:
            bucket="different_sequence_cross_split_cross_label"
        buckets[bucket]+=1
        dsu.union(i,j)
        confirmed.append({
            "i":i,"j":j,"ssim":score,"dhash":d,"phash":p,"bucket":bucket,
            "path_a":rows[i]["path"],"path_b":rows[j]["path"],
            "sequence_a":seq[i],"sequence_b":seq[j],
            "split_a":rows[i]["split"],"split_b":rows[j]["split"],
            "label_a":rows[i]["label"],"label_b":rows[j]["label"],
            "frame_a":int(rows[i]["frame_num"]),"frame_b":int(rows[j]["frame_num"]),
        })

    comps=defaultdict(list)
    affected=set()
    for e in confirmed:
        affected.add(e["i"]); affected.add(e["j"])
    for i in sorted(affected):
        comps[dsu.find(i)].append(i)

    groups=[g for g in comps.values() if len(g)>=2]
    groups.sort(key=lambda g:(-len(g), min(g)))

    # Deterministic representative: earliest frame within a temporal sequence;
    # if a component spans sequences, tie-break by label, video_id, frame_num, path.
    def rep_key(i):
        r=rows[i]
        return (r["label"], r["video_id"], int(r["frame_num"]), r["path"])

    group_rows=[]
    membership=[]
    removed=set()
    for gid,g in enumerate(groups,1):
        rep=min(g,key=rep_key)
        for i in g:
            keep=(i==rep)
            if not keep: removed.add(i)
            membership.append({
                "dedup_group_id":gid,
                "keep":int(keep),
                "path":rows[i]["path"],
                "label":rows[i]["label"],
                "video_id":rows[i]["video_id"],
                "sequence_id":seq[i],
                "frame_num":int(rows[i]["frame_num"]),
                "split":rows[i]["split"],
                "sha256":rows[i]["sha256"],
            })
        group_rows.append({
            "dedup_group_id":gid,
            "group_size":len(g),
            "representative_path":rows[rep]["path"],
            "representative_sequence":seq[rep],
            "representative_split":rows[rep]["split"],
            "representative_label":rows[rep]["label"],
            "sequence_count":len(set(seq[i] for i in g)),
            "split_count":len(set(rows[i]["split"] for i in g)),
            "label_count":len(set(rows[i]["label"] for i in g)),
            "min_ssim_within_confirmed_edges":min(
                e["ssim"] for e in confirmed if e["i"] in g and e["j"] in g
            ),
            "max_ssim_within_confirmed_edges":max(
                e["ssim"] for e in confirmed if e["i"] in g and e["j"] in g
            ),
        })

    confirmed.sort(key=lambda x:x["ssim"],reverse=True)
    membership.sort(key=lambda x:(x["dedup_group_id"],-x["keep"],x["label"],x["video_id"],x["frame_num"]))

    def write_csv(path,data):
        if not data:
            path.write_text("",encoding="utf-8"); return
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)

    write_csv(a.out_dir/"all_confirmed_near_duplicate_pairs.csv",confirmed)
    write_csv(a.out_dir/"near_duplicate_groups.csv",group_rows)
    write_csv(a.out_dir/"near_duplicate_membership.csv",membership)

    kept=[i for i in range(n) if i not in removed]
    kept_by_label=Counter(rows[i]["label"] for i in kept)
    removed_by_label=Counter(rows[i]["label"] for i in removed)
    removed_by_split=Counter(rows[i]["split"] for i in removed)

    summary={
        "image_count":n,
        "screen_rule":f"dHash <= {D_THR} AND pHash <= {P_THR}",
        "screened_pair_count":len(screened),
        "confirm_rule":f"SSIM >= {SSIM_THR}",
        "ssim_resolution":f"{SIZE[0]}x{SIZE[1]} grayscale",
        "confirmed_pair_count":len(confirmed),
        "confirmed_pair_buckets":dict(sorted(buckets.items())),
        "near_duplicate_group_count":len(groups),
        "images_in_near_duplicate_groups":len(affected),
        "images_removed_if_one_rep_per_group":len(removed),
        "images_retained_after_dedup":len(kept),
        "retained_by_label":dict(kept_by_label),
        "removed_by_label":dict(removed_by_label),
        "removed_by_original_split":dict(removed_by_split),
        "group_size_min":min((len(g) for g in groups),default=0),
        "group_size_median":float(np.median([len(g) for g in groups])) if groups else 0,
        "group_size_max":max((len(g) for g in groups),default=0),
        "groups_spanning_multiple_sequences":sum(len(set(seq[i] for i in g))>1 for g in groups),
        "groups_spanning_both_original_splits":sum(len(set(rows[i]["split"] for i in g))>1 for g in groups),
        "groups_spanning_both_labels":sum(len(set(rows[i]["label"] for i in g))>1 for g in groups),
        "representative_rule":"deterministic earliest by (label, video_id, frame_num, path) within each connected component",
        "scope_note":"Confirmed near-duplicate groups are based on perceptual-hash screening plus SSIM confirmation. Connected components allow transitive grouping. Canonical raw data are not modified."
    }
    (a.out_dir/"global_dedup_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
