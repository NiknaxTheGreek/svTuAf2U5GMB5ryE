#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from collections import defaultdict,deque
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont

CELL_W,CELL_H=150,267
PAIRS_PER_PAGE=8

def resolve(root, rel):
    p=root/rel
    if p.exists(): return p
    return root/"images"/rel

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--candidates",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    with a.inventory.open(newline="",encoding="utf-8") as f:
        inv=list(csv.DictReader(f))
    seqs=defaultdict(list)
    for r in inv:
        seq=f'{r["label"]}__{str(r["video_id"]).zfill(4)}'
        seqs[seq].append(r)
    for k in seqs:
        seqs[k]=sorted(seqs[k],key=lambda r:int(r["frame_num"]))

    with a.candidates.open(newline="",encoding="utf-8") as f:
        cand=list(csv.DictReader(f))
    cand=sorted(cand,key=lambda r:float(r["composite_pct"]))

    # candidate graph components
    adj=defaultdict(set)
    nodes=set()
    for r in cand:
        x,y=r["sequence_a"],r["sequence_b"]
        adj[x].add(y); adj[y].add(x); nodes|={x,y}
    comp_id={}
    comps=[]
    for node in sorted(nodes):
        if node in comp_id: continue
        cid=f"C{len(comps)+1:02d}"
        q=[node]; comp=[]
        comp_id[node]=cid
        while q:
            u=q.pop(); comp.append(u)
            for v in sorted(adj[u]):
                if v not in comp_id:
                    comp_id[v]=cid; q.append(v)
        comps.append(sorted(comp))

    font=ImageFont.load_default()
    manifest=[]
    pages=[]
    row_h=CELL_H+34
    gap=7
    for page_idx in range(math.ceil(len(cand)/PAIRS_PER_PAGE)):
        sub=cand[page_idx*PAIRS_PER_PAGE:(page_idx+1)*PAIRS_PER_PAGE]
        canvas=Image.new("RGB",(6*CELL_W+7*gap,len(sub)*(row_h+gap)+gap),"white")
        draw=ImageDraw.Draw(canvas)
        for rr,rec in enumerate(sub):
            y=gap+rr*(row_h+gap)
            for side,seq in enumerate([rec["sequence_a"],rec["sequence_b"]]):
                rows=seqs[seq]
                inds=[0,len(rows)//2,len(rows)-1]
                for k,idx in enumerate(inds):
                    p=resolve(a.image_root,rows[idx]["path"])
                    with Image.open(p) as im:
                        t=ImageOps.fit(im.convert("RGB"),(CELL_W,CELL_H),method=Image.Resampling.BILINEAR)
                    col=side*3+k
                    x=gap+col*(CELL_W+gap)
                    canvas.paste(t,(x,y))
                x0=gap+(side*3)*(CELL_W+gap)
                draw.text((x0,y+CELL_H+3),seq,fill="black",font=font)
            txt=f'#{page_idx*PAIRS_PER_PAGE+rr+1:02d} {comp_id[rec["sequence_a"]]} comp={float(rec["composite_pct"]):.4f} ranks={rec["rank_a_to_b"]}/{rec["rank_b_to_a"]}'
            draw.text((gap+3*(CELL_W+gap),y+CELL_H+17),txt,fill="black",font=font)
            out=dict(rec)
            out["candidate_rank"]=page_idx*PAIRS_PER_PAGE+rr+1
            out["component_id"]=comp_id[rec["sequence_a"]]
            out["review_page"]=page_idx+1
            out["review_row"]=rr+1
            manifest.append(out)
        fp=a.out_dir/f"all_candidates_page_{page_idx+1:02d}.jpg"
        canvas.save(fp,quality=90)
        pages.append(fp.name)

    fields=list(manifest[0].keys())
    with (a.out_dir/"candidate_review_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(manifest)

    summary={
        "candidate_pair_count":len(cand),
        "candidate_clip_count":len(nodes),
        "connected_component_count":len(comps),
        "component_sizes":sorted([len(x) for x in comps],reverse=True),
        "components":[{"component_id":f"C{i+1:02d}","clips":c} for i,c in enumerate(comps)],
        "review_pages":pages,
        "review_rule":"Each candidate pair shows first/middle/last frame from both temporal clips. Visual review should favor precision: merge only when book/layout/background/camera/hand setup strongly agree; ambiguous pairs remain separate."
    }
    (a.out_dir/"candidate_review_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
