#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageOps,ImageDraw
import mediapipe as mp

DETECT_W=540
INPAINT_W=270

def resize_keep(img,w):
    h0,w0=img.shape[:2]
    h=max(1,round(h0*w/w0))
    return cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)

def hand_arm_mask(img_bgr,hands):
    h0,w0=img_bgr.shape[:2]
    work=resize_keep(img_bgr,DETECT_W)
    h,w=work.shape[:2]
    rgb=cv2.cvtColor(work,cv2.COLOR_BGR2RGB)
    res=hands.process(rgb)
    mask=np.zeros((h,w),np.uint8)
    detected=0
    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            detected+=1
            pts=np.array([[lm.x*w,lm.y*h] for lm in hand.landmark],dtype=np.float32)
            pts[:,0]=np.clip(pts[:,0],0,w-1); pts[:,1]=np.clip(pts[:,1],0,h-1)
            hull=cv2.convexHull(pts.astype(np.int32))
            cv2.fillConvexPoly(mask,hull,255)

            palm=(pts[5]+pts[9]+pts[13]+pts[17])/4
            wrist=pts[0]
            palm_w=max(12.0,float(np.linalg.norm(pts[5]-pts[17])))
            v=wrist-palm
            nv=float(np.linalg.norm(v))
            if nv<1e-6:
                v=np.array([0,1],dtype=np.float32); nv=1
            v=v/nv
            perp=np.array([-v[1],v[0]],dtype=np.float32)

            # Extend the forearm from the wrist outward far enough to reach a boundary.
            L=max(h,w)*0.65
            end=wrist+v*L
            hw0=0.62*palm_w
            hw1=0.88*palm_w
            poly=np.array([
                wrist+perp*hw0,
                wrist-perp*hw0,
                end-perp*hw1,
                end+perp*hw1
            ],dtype=np.int32)
            cv2.fillConvexPoly(mask,poly,255)

            # Expand around fingers/hand edge.
            rad=max(9,int(round(palm_w*0.28)))
            if rad%2==0: rad+=1
            k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(rad,rad))
            mask=cv2.dilate(mask,k,iterations=1)

    full=cv2.resize(mask,(w0,h0),interpolation=cv2.INTER_NEAREST)
    return full,detected

def inpaint(img,mask):
    if not mask.any(): return img.copy()
    h0,w0=img.shape[:2]
    sm=resize_keep(img,INPAINT_W)
    mh,mw=sm.shape[:2]
    mm=cv2.resize(mask,(mw,mh),interpolation=cv2.INTER_NEAREST)
    fill=cv2.inpaint(sm,mm,3,cv2.INPAINT_TELEA)
    fill=cv2.resize(fill,(w0,h0),interpolation=cv2.INTER_CUBIC)
    feather=cv2.GaussianBlur(mask,(0,0),sigmaX=4).astype(np.float32)/255.
    out=img.astype(np.float32)*(1-feather[...,None])+fill.astype(np.float32)*feather[...,None]
    return np.clip(out,0,255).astype(np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    ap.add_argument("--limit",type=int,default=60)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    with a.inventory.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    # Spread sample across corpus rather than taking one sequence.
    idxs=np.linspace(0,len(rows)-1,min(a.limit,len(rows))).round().astype(int)
    sample=[rows[i] for i in sorted(set(idxs.tolist()))]

    mp_hands=mp.solutions.hands
    recs=[]; previews=[]
    with mp_hands.Hands(static_image_mode=True,max_num_hands=2,model_complexity=1,
                        min_detection_confidence=0.35,min_tracking_confidence=0.35) as hands:
        for r in sample:
            p=a.image_root/r["path"]
            if not p.exists(): p=a.image_root/"images"/r["path"]
            img=cv2.imread(str(p))
            mask,n=hand_arm_mask(img,hands)
            out=inpaint(img,mask)
            cov=float((mask>0).mean())
            recs.append({"path":r["path"],"label":r["label"],"split":r["split"],
                         "hands_detected":n,"mask_fraction":cov})
            if len(previews)<12:
                b=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                c=Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
                b=ImageOps.fit(b,(162,288),method=Image.Resampling.BILINEAR)
                c=ImageOps.fit(c,(162,288),method=Image.Resampling.BILINEAR)
                pair=Image.new("RGB",(334,316),"white"); pair.paste(b,(0,0)); pair.paste(c,(172,0))
                ImageDraw.Draw(pair).text((2,292),f'{Path(r["path"]).name} hands={n} mask={cov:.1%}',fill="black")
                previews.append(pair)

    cols=2; pw,ph=334,316; gap=8; rowsn=math.ceil(len(previews)/cols)
    sheet=Image.new("RGB",(cols*pw+(cols+1)*gap,rowsn*ph+(rowsn+1)*gap),"white")
    for i,pair in enumerate(previews):
        rr,cc=divmod(i,cols); sheet.paste(pair,(gap+cc*(pw+gap),gap+rr*(ph+gap)))
    sheet.save(a.out_dir/"mediapipe_hand_arm_preview.jpg",quality=90)

    with (a.out_dir/"sample_hand_mask_stats.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    cov=np.array([r["mask_fraction"] for r in recs])
    summary={
      "sample_count":len(recs),
      "images_with_hands_detected":sum(r["hands_detected"]>0 for r in recs),
      "total_hands_detected":sum(r["hands_detected"] for r in recs),
      "mask_fraction_mean":float(cov.mean()),
      "mask_fraction_median":float(np.median(cov)),
      "mask_fraction_max":float(cov.max()),
      "zero_mask_count":int(np.sum(cov==0))
    }
    (a.out_dir/"sample_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
