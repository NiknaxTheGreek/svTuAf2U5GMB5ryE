#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import imagehash

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
    for r in rows:
        p=a.image_root/r["path"]
        if not p.exists(): p=a.image_root/"images"/r["path"]
        with Image.open(p) as im:
            im=im.convert("RGB")
            dh.append(int(str(imagehash.dhash(im,hash_size=8)),16))
            ph.append(int(str(imagehash.phash(im,hash_size=8)),16))

    seq=[f'{r["label"]}__{r["video_id"]}' for r in rows]
    split=[r["split"] for r in rows]
    label=[r["label"] for r in rows]

    best_score=[999]*n; best_j=[-1]*n; best_d=[999]*n; best_p=[999]*n
    buckets=Counter()
    very=[]; moderate=[]

    def bucket(i,j):
        same_seq=seq[i]==seq[j]; cross_split=split[i]!=split[j]; same_label=label[i]==label[j]
        if same_seq and cross_split: return "same_sequence_cross_split"
        if same_seq: return "same_sequence_same_split"
        if cross_split and same_label: return "different_sequence_cross_split_same_label"
        if cross_split and not same_label: return "different_sequence_cross_split_cross_label"
        if same_label: return "different_sequence_same_split_same_label"
        return "different_sequence_same_split_cross_label"

    for i in range(n-1):
        for j in range(i+1,n):
            d=(dh[i]^dh[j]).bit_count()
            p=(ph[i]^ph[j]).bit_count()
            s=d+p
            if seq[i]!=seq[j]:
                if s<best_score[i]:
                    best_score[i]=s; best_j[i]=j; best_d[i]=d; best_p[i]=p
                if s<best_score[j]:
                    best_score[j]=s; best_j[j]=i; best_d[j]=d; best_p[j]=p
            if d<=8 and p<=8:
                b=bucket(i,j)
                buckets["moderate__"+b]+=1
                rec={"bucket":b,"dhash":d,"phash":p,
                     "path_a":rows[i]["path"],"path_b":rows[j]["path"],
                     "seq_a":seq[i],"seq_b":seq[j],
                     "split_a":split[i],"split_b":split[j],
                     "label_a":label[i],"label_b":label[j]}
                if len(moderate)<2000: moderate.append(rec)
                if d<=4 and p<=4:
                    buckets["very__"+b]+=1
                    if len(very)<1000: very.append(rec)

    nn=[]
    for i,j in enumerate(best_j):
        nn.append({
            "path":rows[i]["path"],"sequence_id":seq[i],"split":split[i],"label":label[i],
            "nearest_other_sequence_path":rows[j]["path"] if j>=0 else "",
            "nearest_other_sequence_id":seq[j] if j>=0 else "",
            "nearest_other_sequence_split":split[j] if j>=0 else "",
            "nearest_other_sequence_label":label[j] if j>=0 else "",
            "dhash_distance":best_d[i] if j>=0 else "",
            "phash_distance":best_p[i] if j>=0 else "",
            "combined_distance":best_score[i] if j>=0 else ""
        })

    def write_csv(path,data):
        if not data: path.write_text("",encoding="utf-8"); return
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)

    write_csv(a.out_dir/"very_high_similarity_pairs.csv",very)
    write_csv(a.out_dir/"moderate_similarity_pairs.csv",moderate)
    write_csv(a.out_dir/"nearest_neighbor_outside_sequence.csv",nn)

    scores=np.array([x["combined_distance"] for x in nn],dtype=float)
    summary={
      "image_count":n,
      "total_unordered_pairs":n*(n-1)//2,
      "exact_sha256_duplicate_groups_already_observed":0,
      "very_high_similarity_rule":"dHash <= 4 AND pHash <= 4",
      "moderate_similarity_rule":"dHash <= 8 AND pHash <= 8",
      "pair_counts":dict(sorted(buckets.items())),
      "nearest_neighbor_outside_sequence":{
        "median_combined_distance":float(np.median(scores)),
        "q1_combined_distance":float(np.quantile(scores,.25)),
        "q3_combined_distance":float(np.quantile(scores,.75)),
        "cross_split_count":sum(x["split"]!=x["nearest_other_sequence_split"] for x in nn),
        "cross_label_count":sum(x["label"]!=x["nearest_other_sequence_label"] for x in nn)
      },
      "scope_note":"Perceptual hashes screen for visual similarity; they do not prove duplication. Within-sequence similarity is expected. Cross-sequence/cross-split candidates require interpretation; no image is deleted."
    }
    (a.out_dir/"perceptual_similarity_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
