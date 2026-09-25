#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

THUMB=(96,171)
SAMPLES=5

def sample_idx(n,k=SAMPLES):
    if n<=k: return list(range(n))
    return sorted(set(int(round(x)) for x in np.linspace(0,n-1,k)))

def load_rgb(path,size=THUMB):
    with Image.open(path) as im:
        return np.asarray(ImageOps.fit(im.convert("RGB"), size, method=Image.Resampling.BILINEAR),dtype=np.float32)/255.0

def hist48(arr):
    feats=[]
    for c in range(3):
        h,_=np.histogram(arr[:,:,c],bins=16,range=(0,1),density=True)
        feats.extend(h.tolist())
    v=np.asarray(feats,dtype=np.float32)
    return v/(np.linalg.norm(v)+1e-8)

def border_mask(h,w,f=0.22):
    yy,xx=np.ogrid[:h,:w]
    return (yy < h*f) | (yy >= h*(1-f)) | (xx < w*f) | (xx >= w*(1-f))

def center_mask(h,w,f=0.22):
    yy,xx=np.ogrid[:h,:w]
    return ~((yy < h*f) | (yy >= h*(1-f)) | (xx < w*f) | (xx >= w*(1-f)))

def edge_map(gray):
    gx=np.zeros_like(gray); gy=np.zeros_like(gray)
    gx[:,1:-1]=(gray[:,2:]-gray[:,:-2])*0.5
    gy[1:-1,:]=(gray[2:,:]-gray[:-2,:])*0.5
    e=np.sqrt(gx*gx+gy*gy)
    # resize to 24x43 using PIL
    e=np.clip(e/(e.max()+1e-8),0,1)
    im=Image.fromarray((e*255).astype(np.uint8)).resize((24,43),Image.Resampling.BILINEAR)
    return np.asarray(im,dtype=np.float32).reshape(-1)/255.0

def cosine_distance(a,b):
    return 1.0-float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

def rank_percentiles(vals):
    order=np.argsort(vals)
    pct=np.empty(len(vals),dtype=np.float64)
    if len(vals)==1:
        pct[0]=0
        return pct
    for rank,idx in enumerate(order):
        pct[idx]=rank/(len(vals)-1)
    return pct

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)

    with args.inventory.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in rows:
        groups[(r["label"],r["video_id"])].append(r)

    clips=[]
    for (label,vid),rr in sorted(groups.items()):
        rr=sorted(rr,key=lambda r:int(r["frame_num"]))
        picks=sample_idx(len(rr))
        arrs=[]
        pick_paths=[]
        for idx in picks:
            r=rr[idx]
            p=args.image_root/r["path"]
            if not p.exists(): p=args.image_root/"images"/r["path"]
            arrs.append(load_rgb(p))
            pick_paths.append(str(p))
        stack=np.stack(arrs,axis=0)
        med=np.median(stack,axis=0)
        mean=np.mean(stack,axis=0)
        h,w,_=med.shape
        bm=border_mask(h,w); cm=center_mask(h,w)
        border=np.concatenate([med[bm].mean(axis=0),med[bm].std(axis=0)])
        center=np.concatenate([med[cm].mean(axis=0),med[cm].std(axis=0)])
        hist=hist48(med)
        gray=med.mean(axis=2)
        edge=edge_map(gray)
        # coarse geometry/color thumbnail
        coarse=np.asarray(Image.fromarray((med*255).astype(np.uint8)).resize((12,21),Image.Resampling.BILINEAR),dtype=np.float32).reshape(-1)/255.0

        clips.append({
            "label":label,"video_id":vid,"sequence_id":f"{label}__{vid}",
            "frame_count":len(rr),"rows":rr,"pick_paths":pick_paths,
            "median":med,"border":border,"center":center,"hist":hist,"edge":edge,"coarse":coarse
        })

    pairrecs=[]
    for i in range(len(clips)-1):
        a=clips[i]
        for j in range(i+1,len(clips)):
            b=clips[j]
            # Distances: all lower is more similar
            border=float(np.mean(np.abs(a["border"]-b["border"])))
            center=float(np.mean(np.abs(a["center"]-b["center"])))
            hist=cosine_distance(a["hist"],b["hist"])
            edge=float(np.mean(np.abs(a["edge"]-b["edge"])))
            coarse=float(np.mean(np.abs(a["coarse"]-b["coarse"])))
            pairrecs.append({
                "i":i,"j":j,
                "sequence_a":a["sequence_id"],"sequence_b":b["sequence_id"],
                "label_a":a["label"],"label_b":b["label"],
                "video_id_a":a["video_id"],"video_id_b":b["video_id"],
                "same_numeric_video_id":int(a["video_id"]==b["video_id"]),
                "border_l1":border,"center_l1":center,"hist_cosine":hist,
                "edge_l1":edge,"coarse_l1":coarse
            })

    for metric in ["border_l1","center_l1","hist_cosine","edge_l1","coarse_l1"]:
        vals=np.array([r[metric] for r in pairrecs],dtype=float)
        pct=rank_percentiles(vals)
        for r,p in zip(pairrecs,pct):
            r[metric+"_pct"]=float(p)

    for r in pairrecs:
        # Session-oriented composite: background/camera geometry weighted most.
        r["composite_pct"]=float(
            0.30*r["border_l1_pct"]+
            0.25*r["coarse_l1_pct"]+
            0.20*r["edge_l1_pct"]+
            0.15*r["hist_cosine_pct"]+
            0.10*r["center_l1_pct"]
        )

    # reciprocal nearest-neighbor ranks
    byclip=defaultdict(list)
    for idx,r in enumerate(pairrecs):
        byclip[r["i"]].append((r["composite_pct"],idx,r["j"]))
        byclip[r["j"]].append((r["composite_pct"],idx,r["i"]))
    ranks={}
    for ci,lst in byclip.items():
        for rank,(_,idx,other) in enumerate(sorted(lst),1):
            ranks[(ci,other)]=rank
    for r in pairrecs:
        r["rank_a_to_b"]=ranks[(r["i"],r["j"])]
        r["rank_b_to_a"]=ranks[(r["j"],r["i"])]
        r["reciprocal_top3"]=int(r["rank_a_to_b"]<=3 and r["rank_b_to_a"]<=3)

    pairrecs.sort(key=lambda r:r["composite_pct"])
    strong=[r for r in pairrecs if r["composite_pct"]<=0.05 and r["reciprocal_top3"]]
    inspect=pairrecs[:24]

    # contact sheets: 8 pairs/page; show first/middle/last frames for each clip
    font=ImageFont.load_default()
    cell_w,cell_h=170,302
    gap=8
    row_h=cell_h+36
    rows_per_sheet=8
    sheets=[]
    for page in range(math.ceil(len(inspect)/rows_per_sheet)):
        subset=inspect[page*rows_per_sheet:(page+1)*rows_per_sheet]
        canvas=Image.new("RGB",(6*cell_w+7*gap,len(subset)*row_h+(len(subset)+1)*gap),"white")
        draw=ImageDraw.Draw(canvas)
        for rr_idx,rec in enumerate(subset):
            y=gap+rr_idx*(row_h+gap)
            a=clips[rec["i"]]; b=clips[rec["j"]]
            for side,clip in enumerate([a,b]):
                seqrows=clip["rows"]
                inds=[0,len(seqrows)//2,len(seqrows)-1]
                for k,idx in enumerate(inds):
                    r=seqrows[idx]
                    p=args.image_root/r["path"]
                    if not p.exists(): p=args.image_root/"images"/r["path"]
                    with Image.open(p) as im:
                        thumb=ImageOps.fit(im.convert("RGB"),(cell_w,cell_h),method=Image.Resampling.BILINEAR)
                    x=gap+(side*3+k)*cell_w+(side*3+k)*gap
                    canvas.paste(thumb,(x,y))
                x0=gap+side*3*cell_w+side*3*gap
                draw.text((x0,y+cell_h+4),clip["sequence_id"],fill="black",font=font)
            draw.text((gap+3*cell_w+3*gap,y+cell_h+18),
                      f'composite={rec["composite_pct"]:.4f} ranks={rec["rank_a_to_b"]}/{rec["rank_b_to_a"]}',
                      fill="black",font=font)
        fp=args.out_dir/f"session_candidates_page_{page+1}.jpg"
        canvas.save(fp,quality=90)
        sheets.append(fp.name)

    fields=[
        "sequence_a","sequence_b","label_a","label_b","video_id_a","video_id_b",
        "same_numeric_video_id","border_l1","center_l1","hist_cosine","edge_l1","coarse_l1",
        "border_l1_pct","center_l1_pct","hist_cosine_pct","edge_l1_pct","coarse_l1_pct",
        "composite_pct","rank_a_to_b","rank_b_to_a","reciprocal_top3"
    ]
    with (args.out_dir/"all_clip_pair_similarity.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in pairrecs: w.writerow({k:r[k] for k in fields})
    with (args.out_dir/"strong_session_candidates.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in strong: w.writerow({k:r[k] for k in fields})

    summary={
        "clip_count":len(clips),
        "pair_count":len(pairrecs),
        "candidate_rule":"composite percentile <= 0.05 AND reciprocal top-3 nearest neighbors",
        "strong_candidate_pair_count":len(strong),
        "top_24_for_visual_inspection":[
            {k:r[k] for k in ["sequence_a","sequence_b","same_numeric_video_id","composite_pct","rank_a_to_b","rank_b_to_a"]}
            for r in inspect
        ],
        "contact_sheets":sheets,
        "method_note":"Clip signatures use five temporally spaced frames per clip. Median-frame border/background, coarse geometry/color, edge structure, color histogram and center statistics are combined. These are candidate session/environment similarities, not ground-truth session labels."
    }
    (args.out_dir/"session_cluster_screen_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
