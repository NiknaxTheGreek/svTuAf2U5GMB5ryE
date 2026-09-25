#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import cv2, numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont

WORK_SCALE=4
FEATURE_SIZE=(270,480)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def mask_small(img):
    h,w=img.shape[:2]
    sm=cv2.resize(img,(w//WORK_SCALE,h//WORK_SCALE),interpolation=cv2.INTER_AREA)
    sh,sw=sm.shape[:2]
    ycc=cv2.cvtColor(sm,cv2.COLOR_BGR2YCrCb); hsv=cv2.cvtColor(sm,cv2.COLOR_BGR2HSV)
    Y,Cr,Cb=cv2.split(ycc); H,S,V=cv2.split(hsv)
    skin=((Y>35)&(Cr>=130)&(Cr<=185)&(Cb>=70)&(Cb<=140)&
          (((H<=28)|(H>=170))&(S>=18)&(S<=210)&(V>=45))).astype(np.uint8)*255
    skin=cv2.morphologyEx(skin,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    skin=cv2.morphologyEx(skin,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    n,lab,stats,_=cv2.connectedComponentsWithStats(skin,8)
    keep=np.zeros_like(skin)
    margin=max(6,int(.035*min(sh,sw))); min_area=max(45,int(.0015*sh*sw))
    for k in range(1,n):
        x,y,ww,hh,area=stats[k]
        touches=x<=margin or y<=margin or x+ww>=sw-margin or y+hh>=sh-margin
        plausible=y+hh>=int(.55*sh) or x<=margin or x+ww>=sw-margin
        if area>=min_area and touches and plausible: keep[lab==k]=255
    kd=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    keep=cv2.dilate(keep,kd,iterations=1); keep=cv2.morphologyEx(keep,cv2.MORPH_CLOSE,kd)
    return sm,keep

def clean(img):
    h,w=img.shape[:2]; sm,mask=mask_small(img)
    if not mask.any(): return img.copy(),mask
    fill=cv2.inpaint(sm,mask,3,cv2.INPAINT_TELEA)
    fill=cv2.resize(fill,(w,h),interpolation=cv2.INTER_CUBIC)
    fullmask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
    feather=cv2.GaussianBlur(fullmask,(0,0),4).astype(np.float32)/255.
    out=img.astype(np.float32)*(1-feather[...,None])+fill.astype(np.float32)*feather[...,None]
    return np.clip(out,0,255).astype(np.uint8),mask

def feats(img):
    sm=cv2.resize(img,FEATURE_SIZE,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY); hsv=cv2.cvtColor(sm,cv2.COLOR_BGR2HSV)
    return {
      "brightness_mean":float(gray.mean()),
      "contrast_std":float(gray.std()),
      "sharpness_laplacian_var":float(cv2.Laplacian(gray,cv2.CV_64F).var()),
      "saturation_mean":float(hsv[:,:,1].mean()),
      "edge_density":float((cv2.Canny(gray,80,160)>0).mean())
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--out-root",required=True,type=Path)
    ap.add_argument("--report-dir",required=True,type=Path)
    ap.add_argument("--shard-index",required=True,type=int)
    ap.add_argument("--num-shards",default=4,type=int)
    a=ap.parse_args(); a.out_root.mkdir(parents=True,exist_ok=True); a.report_dir.mkdir(parents=True,exist_ok=True)
    with a.inventory.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    rows=[r for i,r in enumerate(rows) if i%a.num_shards==a.shard_index]
    mani=[]; fs=[]; prev=[]
    for k,r in enumerate(rows):
        src=a.image_root/r["path"]
        if not src.exists(): src=a.image_root/"images"/r["path"]
        img=cv2.imread(str(src)); out,smask=clean(img)
        h,w=img.shape[:2]; cov=float((smask>0).mean())
        dst=a.out_root/r["path"]; dst.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(dst),out,[int(cv2.IMWRITE_JPEG_QUALITY),95])
        of,cf=feats(img),feats(out)
        rec={"path":r["path"],"label":r["label"],"video_id":r["video_id"],"frame_num":r["frame_num"],
             "split":r["split"],"width":w,"height":h,"mask_fraction":cov,"source_sha256":r["sha256"],
             "clean_sha256":sha256_file(dst),"source_bytes":src.stat().st_size,"clean_bytes":dst.stat().st_size}
        mani.append(rec)
        fs.append({**{x:rec[x] for x in ["path","label","video_id","frame_num","split","mask_fraction"]},
                   **{f"original_{x}":v for x,v in of.items()},
                   **{f"cleaned_{x}":v for x,v in cf.items()},
                   **{f"delta_{x}":cf[x]-of[x] for x in of},
                   "original_jpeg_bytes":rec["source_bytes"],"cleaned_jpeg_bytes":rec["clean_bytes"],
                   "delta_jpeg_bytes":rec["clean_bytes"]-rec["source_bytes"]})
        if len(prev)<2 and cov>.01:
            # small before/after side-by-side
            b=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); c=Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
            b=ImageOps.fit(b,(216,384),method=Image.Resampling.BILINEAR); c=ImageOps.fit(c,(216,384),method=Image.Resampling.BILINEAR)
            sheet=Image.new("RGB",(442,410),"white"); sheet.paste(b,(0,0)); sheet.paste(c,(226,0))
            ImageDraw.Draw(sheet).text((4,389),f'{r["path"]} mask={cov:.1%}',fill="black")
            sheet.save(a.report_dir/f"preview_{a.shard_index}_{len(prev)}.jpg",quality=90); prev.append(1)
        if (k+1)%100==0: print(f"shard {a.shard_index}: {k+1}/{len(rows)}",flush=True)
    for name,data in [("manifest",mani),("features",fs)]:
        p=a.report_dir/f"shard_{a.shard_index}_{name}.csv"
        with p.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    print(json.dumps({"shard":a.shard_index,"count":len(rows),"masked":sum(x["mask_fraction"]>0 for x in mani)},indent=2))
if __name__=="__main__": main()
