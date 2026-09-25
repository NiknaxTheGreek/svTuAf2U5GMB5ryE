#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,math
from collections import defaultdict
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont

CELL_W,CELL_H=190,338
PAIRS_PER_PAGE=8

def resolve(root,rel):
    p=root/rel
    if p.exists(): return p
    return root/"images"/rel

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--candidates",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)

    with args.inventory.open(newline="",encoding="utf-8") as f:
        inv=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in inv:
        groups[f'{r["label"]}__{r["video_id"]}'].append(r)
    for k in groups:
        groups[k]=sorted(groups[k],key=lambda x:int(x["frame_num"]))

    with args.candidates.open(newline="",encoding="utf-8") as f:
        cand=list(csv.DictReader(f))
    cand=sorted(cand,key=lambda r:float(r["composite_pct"]))

    font=ImageFont.load_default()
    gap=10; row_h=CELL_H+42
    pages=[]
    for pi in range(math.ceil(len(cand)/PAIRS_PER_PAGE)):
        subset=cand[pi*PAIRS_PER_PAGE:(pi+1)*PAIRS_PER_PAGE]
        W=6*CELL_W+7*gap
        H=len(subset)*(row_h+gap)+gap
        canvas=Image.new("RGB",(W,H),"white")
        draw=ImageDraw.Draw(canvas)
        for ri,rec in enumerate(subset):
            y=gap+ri*(row_h+gap)
            pair_idx=pi*PAIRS_PER_PAGE+ri+1
            for side,seq in enumerate([rec["sequence_a"],rec["sequence_b"]]):
                rows=groups[seq]
                inds=[0,len(rows)//2,len(rows)-1]
                for k,idx in enumerate(inds):
                    p=resolve(args.image_root,rows[idx]["path"])
                    with Image.open(p) as im:
                        thumb=ImageOps.fit(im.convert("RGB"),(CELL_W,CELL_H),method=Image.Resampling.BILINEAR)
                    x=gap+(side*3+k)*(CELL_W+gap)
                    canvas.paste(thumb,(x,y))
                x0=gap+side*3*(CELL_W+gap)
                draw.text((x0,y+CELL_H+4),seq,fill="black",font=font)
            txt=(f'#{pair_idx:02d} comp={float(rec["composite_pct"]):.4f} '
                 f'ranks={rec["rank_a_to_b"]}/{rec["rank_b_to_a"]} '
                 f'labels={rec["label_a"]}/{rec["label_b"]}')
            draw.text((gap+3*(CELL_W+gap),y+CELL_H+20),txt,fill="black",font=font)
        fp=args.out_dir/f"all_candidates_page_{pi+1:02d}.jpg"
        canvas.save(fp,quality=92)
        pages.append(fp.name)

    # review template
    fields=["pair_index","sequence_a","sequence_b","label_a","label_b","composite_pct",
            "rank_a_to_b","rank_b_to_a","decision","confidence","rationale"]
    with (args.out_dir/"candidate_review_template.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i,r in enumerate(cand,1):
            w.writerow({
                "pair_index":i,"sequence_a":r["sequence_a"],"sequence_b":r["sequence_b"],
                "label_a":r["label_a"],"label_b":r["label_b"],
                "composite_pct":r["composite_pct"],"rank_a_to_b":r["rank_a_to_b"],
                "rank_b_to_a":r["rank_b_to_a"],"decision":"","confidence":"","rationale":""
            })
    print(f"rendered {len(cand)} pairs across {len(pages)} pages")

if __name__=="__main__":
    main()
