from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.chroma_counterfactual_v2 import (
    PaletteTransferConfig,
    naturalistic_palette_transfer,
    stable_donor_order,
)


STRENGTHS = (1.0, 1.5)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def decode_member(z: ZipFile, member: str) -> np.ndarray:
    data = z.read(member)
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def choose_donor(target_id, target_video, target_env, donor_ids, by_id, env_by_id):
    ordered = stable_donor_order(target_id, [x for x in donor_ids if x != target_id])
    preferred = [
        sid for sid in ordered
        if by_id[sid]["video_id"] != target_video and env_by_id.get(sid) != target_env
    ]
    if preferred:
        return preferred[0]
    different_video = [sid for sid in ordered if by_id[sid]["video_id"] != target_video]
    if different_video:
        return different_video[0]
    if ordered:
        return ordered[0]
    raise RuntimeError("No donor available")


def build_sheet(entries, path: Path):
    cell=(150,267); gap=7; label_h=52
    canvas=Image.new("RGB",(3*cell[0]+4*gap,len(entries)*(cell[1]+label_h+gap)+gap),"white")
    draw=ImageDraw.Draw(canvas)
    for i,e in enumerate(entries):
        y=gap+i*(cell[1]+label_h+gap)
        imgs=[e["original"],e["s1"],e["s15"]]
        labels=["ORIGINAL","NATURAL 1.0","NATURAL 1.5"]
        for col,(arr,label) in enumerate(zip(imgs,labels)):
            x=gap+col*(cell[0]+gap)
            thumb=ImageOps.fit(Image.fromarray(arr).convert("RGB"),cell,method=Image.Resampling.BILINEAR)
            canvas.paste(thumb,(x,y))
            draw.text((x,y+cell[1]+3),label,fill="black")
        draw.text(
            (gap,y+cell[1]+23),
            f'{e["sample_id"]} dC1={e["m1"]["median_chroma_displacement"]:.1f} '
            f'dC15={e["m15"]["median_chroma_displacement"]:.1f}',
            fill="black"
        )
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,quality=94)
    buf=io.BytesIO()
    canvas.resize((canvas.width*3//4,canvas.height*3//4),Image.Resampling.LANCZOS).save(
        buf,format="JPEG",quality=62,optimize=True,subsampling=2
    )
    path.with_suffix(".review.jpg.b64.txt").write_text(
        base64.b64encode(buf.getvalue()).decode("ascii")+"\n",encoding="ascii"
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",required=True,type=Path)
    ap.add_argument("--manifest",required=True,type=Path)
    ap.add_argument("--split-o",required=True,type=Path)
    ap.add_argument("--split-s",required=True,type=Path)
    ap.add_argument("--split-t",required=True,type=Path)
    ap.add_argument("--split-st",required=True,type=Path)
    ap.add_argument("--sample",required=True,type=Path)
    ap.add_argument("--output-dir",required=True,type=Path)
    args=ap.parse_args()

    manifest=read_csv(args.manifest); by_id={r["sample_id"]:r for r in manifest}
    splits={
      "O":read_csv(args.split_o),"S":read_csv(args.split_s),
      "T":read_csv(args.split_t),"ST":read_csv(args.split_st)
    }
    role={k:{r["sample_id"]:r["role"] for r in rows} for k,rows in splits.items()}
    env_by_id={}
    for rows in splits.values():
        for r in rows: env_by_id.setdefault(r["sample_id"],r["environment_id"])

    donor_ids=sorted(
        sid for sid in by_id
        if all(role[k].get(sid)!="test" for k in ("O","S","T","ST"))
    )
    sample_rows=read_csv(args.sample)
    if len(sample_rows)!=100 or any(role["ST"].get(r["sample_id"])!="context" for r in sample_rows):
        raise ValueError("Expected frozen 100-image ST-context sample")

    args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[]; visuals=[]
    configs={
      s:PaletteTransferConfig(strength=s,covariance_epsilon=1e-4,stats_stride=4,gamut_iterations=14)
      for s in STRENGTHS
    }

    with ZipFile(args.archive) as z:
        for i,row in enumerate(sample_rows,1):
            sid=row["sample_id"]; meta=by_id[sid]
            target=decode_member(z,meta["archive_member"])
            donor_id=choose_donor(sid,meta["video_id"],row["environment_id"],donor_ids,by_id,env_by_id)
            donor=decode_member(z,by_id[donor_id]["archive_member"])

            outs={}; mets={}
            for s in STRENGTHS:
                out,m=naturalistic_palette_transfer(target,donor,cfg=configs[s])
                outs[s]=out; mets[s]=m

            records.append({
              "sample_id":sid,"donor_id":donor_id,
              "strength_1_0":mets[1.0],"strength_1_5":mets[1.5]
            })
            visuals.append({
              "sample_id":sid,"original":target,"s1":outs[1.0],"s15":outs[1.5],
              "m1":mets[1.0],"m15":mets[1.5]
            })
            print(f'{i}/100 {sid} dC1={mets[1.0]["median_chroma_displacement"]:.2f} dC15={mets[1.5]["median_chroma_displacement"]:.2f}',flush=True)

    for start in range(0,100,10):
        build_sheet(visuals[start:start+10],args.output_dir/"qa"/f"strength_qa_{start//10+1:02d}.jpg")

    summary={
      "status":"CHROMA_STRENGTH_100_READY_FOR_VISUAL_REVIEW",
      "images":100,"primary_test_images_used":0,"classifier_predictions_used":False,
      "candidates":{}
    }
    for s,key in [(1.0,"1.0"),(1.5,"1.5")]:
        ys=np.asarray([r[f"strength_{str(s).replace('.','_')}"]["luma_mae"] for r in records],dtype=float)
        dc=np.asarray([r[f"strength_{str(s).replace('.','_')}"]["median_chroma_displacement"] for r in records],dtype=float)
        comp=np.asarray([r[f"strength_{str(s).replace('.','_')}"]["gamut_compressed_fraction"] for r in records],dtype=float)
        summary["candidates"][key]={
          "mean_luma_mae":float(ys.mean()),
          "p95_per_image_luma_mae":float(np.quantile(ys,0.95)),
          "median_per_image_chroma_displacement":float(np.median(dc)),
          "mean_gamut_compressed_fraction":float(comp.mean()),
          "luma_gate_pass":bool(ys.mean()<=0.5 and np.quantile(ys,0.95)<=1.0),
          "nontrivial_chroma_gate_pass":bool(np.median(dc)>=3.0),
          "visual_gate":"PENDING"
        }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (args.output_dir/"records.json").write_text(json.dumps(records,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
