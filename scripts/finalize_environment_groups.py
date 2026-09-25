#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path

class UF:
    def __init__(self,items):
        self.p={x:x for x in items}
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra!=rb:self.p[rb]=ra

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--reviewed-links",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    with a.inventory.open(newline="",encoding="utf-8") as f:
        inv=list(csv.DictReader(f))
    seqrows=defaultdict(list)
    for r in inv:
        sid=f'{r["label"]}__{r["video_id"]}'
        seqrows[sid].append(r)
    sequences=sorted(seqrows)

    with a.reviewed_links.open(newline="",encoding="utf-8") as f:
        links=list(csv.DictReader(f))

    uf=UF(sequences)
    merge_links=[r for r in links if r["decision"]=="MERGE"]
    for r in merge_links:
        uf.union(r["sequence_a"],r["sequence_b"])

    comps=defaultdict(list)
    for s in sequences:
        comps[uf.find(s)].append(s)

    multi=[sorted(v) for v in comps.values() if len(v)>1]
    multi=sorted(multi,key=lambda x:(-len(x),x))
    single=[v[0] for v in comps.values() if len(v)==1]
    single=sorted(single)

    gid={}
    for i,c in enumerate(multi,1):
        g=f"E{i:02d}"
        for s in c: gid[s]=g
    for i,s in enumerate(single,1):
        gid[s]=f"S{i:02d}"

    group_to_seqs=defaultdict(list)
    for s,g in gid.items(): group_to_seqs[g].append(s)

    seq_manifest=[]
    for s in sequences:
        rows=seqrows[s]
        label=s.split("__",1)[0]
        vid=s.split("__",1)[1]
        tr=sum(r["split"]=="training" for r in rows)
        te=sum(r["split"]=="testing" for r in rows)
        gs=group_to_seqs[gid[s]]
        glabels=sorted(set(x.split("__",1)[0] for x in gs))
        seq_manifest.append({
            "sequence_id":s,"label":label,"video_id":vid,
            "environment_group_id":gid[s],
            "group_type":"multi_clip" if len(gs)>1 else "singleton",
            "group_clip_count":len(gs),
            "group_labels":"|".join(glabels),
            "frame_count":len(rows),
            "training_frames":tr,"testing_frames":te,
        })

    group_summary=[]
    for g,ss in sorted(group_to_seqs.items()):
        frame_count=0; flip=0; nf=0; tr=0; te=0
        for s in ss:
            for r in seqrows[s]:
                frame_count+=1
                if r["label"]=="flip": flip+=1
                else: nf+=1
                if r["split"]=="training": tr+=1
                else: te+=1
        labels=sorted(set(s.split("__",1)[0] for s in ss))
        group_summary.append({
            "environment_group_id":g,
            "group_type":"multi_clip" if len(ss)>1 else "singleton",
            "clip_count":len(ss),
            "frame_count":frame_count,
            "flip_frames":flip,
            "notflip_frames":nf,
            "training_frames":tr,
            "testing_frames":te,
            "mixed_label_group":int(len(labels)>1),
            "labels":"|".join(labels),
            "sequences":"|".join(sorted(ss)),
        })

    def write_csv(path,rows):
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            w.writeheader();w.writerows(rows)

    write_csv(a.out_dir/"environment_group_manifest.csv",seq_manifest)
    write_csv(a.out_dir/"environment_group_summary.csv",group_summary)

    summary={
        "sequence_count":len(sequences),
        "reviewed_candidate_links":len(links),
        "merge_links":sum(r["decision"]=="MERGE" for r in links),
        "ambiguous_links":sum(r["decision"]=="AMBIGUOUS" for r in links),
        "separate_links":sum(r["decision"]=="SEPARATE" for r in links),
        "environment_group_count":len(group_to_seqs),
        "multi_clip_group_count":len(multi),
        "singleton_group_count":len(single),
        "multi_clip_group_sizes":sorted([len(x) for x in multi],reverse=True),
        "mixed_label_group_count":sum(int(r["mixed_label_group"]) for r in group_summary),
        "largest_group_frame_count":max(r["frame_count"] for r in group_summary),
        "smallest_group_frame_count":min(r["frame_count"] for r in group_summary),
        "review_policy":"Candidate links were visually reviewed using first/middle/last frames from both clips. Only reviewed MERGE links are unioned. AMBIGUOUS would remain separate. Connected components are used only after link-level review.",
        "scope_note":"These are conservative acquisition-environment grouping units for source-safe splitting, not claims of known recording-session ground truth."
    }
    (a.out_dir/"environment_grouping_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
