#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from huggingface_hub import hf_hub_download
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception:
    mp = None

MODEL_ID = "pirocheto/schp-pascal-7"\nMODEL_REVISION = "e97480b846bf0f23a9f9b7ab673dc1c86af89467"\nONNX_FILE = "onnx/schp-pascal-7-int8-static.onnx"

def hand_mask_mediapipe(rgb: np.ndarray) -> np.ndarray:
    h,w,_=rgb.shape
    mask=np.zeros((h,w),np.uint8)
    if mp is None:
        return mask
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.35,
    )
    try:
        res=hands.process(rgb)
        if not res.multi_hand_landmarks:
            return mask
        for lmset in res.multi_hand_landmarks:
            pts=np.array(
                [[int(lm.x*w),int(lm.y*h)] for lm in lmset.landmark],
                dtype=np.int32
            )
            hull=cv2.convexHull(pts)
            cv2.fillConvexPoly(mask,hull,255)
            # expand to include fingers/hand boundary not perfectly covered by landmarks
            radius=max(10,int(round(0.025*min(h,w))))
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
            mask=cv2.dilate(mask,kernel,iterations=1)
    finally:
        hands.close()
    return mask

def schp_arm_mask(pil_img, processor, session, arm_ids):
    orig_w,orig_h=pil_img.size
    inputs=processor(images=pil_img,return_tensors="np")
    pixel_values=np.asarray(inputs["pixel_values"],dtype=np.float32)
    input_name=session.get_inputs()[0].name
    out_names=[o.name for o in session.get_outputs()]
    target="logits" if "logits" in out_names else out_names[0]
    logits=session.run([target],{input_name:pixel_values})[0]
    pred=logits.argmax(axis=1)[0].astype(np.uint8)
    mask512=np.isin(pred,np.asarray(arm_ids)).astype(np.uint8)*255
    mask=cv2.resize(mask512,(orig_w,orig_h),interpolation=cv2.INTER_NEAREST)
    return mask

def clean_one(src: Path, dst: Path, processor, session, arm_ids):
    pil=Image.open(src).convert("RGB")
    rgb=np.asarray(pil)
    h,w,_=rgb.shape

    arm=schp_arm_mask(pil,processor,session,arm_ids)
    hand=hand_mask_mediapipe(rgb)
    mask=np.maximum(arm,hand)

    # Conservative dilation + closing to cover borders between skin/arm and page.
    rad=max(3,int(round(0.008*min(h,w))))
    ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad+1,2*rad+1))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,ker)
    mask=cv2.dilate(mask,ker,iterations=1)

    frac=float((mask>0).mean())

    # Inpaint at half resolution for speed, then composite only inside mask.
    scale=0.5
    sw,sh=max(2,int(round(w*scale))),max(2,int(round(h*scale)))
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    small=cv2.resize(bgr,(sw,sh),interpolation=cv2.INTER_AREA)
    msmall=cv2.resize(mask,(sw,sh),interpolation=cv2.INTER_NEAREST)
    cleaned_small=cv2.inpaint(small,msmall,5,cv2.INPAINT_TELEA)
    cleaned_up=cv2.resize(cleaned_small,(w,h),interpolation=cv2.INTER_CUBIC)

    # Feather composite so untouched pixels remain original.
    feather=max(3,int(round(0.004*min(h,w))))
    alpha=cv2.GaussianBlur((mask.astype(np.float32)/255.0),(0,0),feather)
    alpha=np.clip(alpha[...,None],0,1)
    out=(bgr.astype(np.float32)*(1-alpha)+cleaned_up.astype(np.float32)*alpha)
    out=np.clip(out,0,255).astype(np.uint8)

    dst.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(dst),out,[int(cv2.IMWRITE_JPEG_QUALITY),95])
    return frac, int((arm>0).sum()), int((hand>0).sum())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",type=Path,required=True)
    ap.add_argument("--image-root",type=Path,required=True)
    ap.add_argument("--out-root",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--limit",type=int,default=0)
    args=ap.parse_args()

    with args.inventory.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    rows=sorted(rows,key=lambda r:(r["label"],r["video_id"],int(r["frame_num"]),r["split"]))
    if args.limit>0:
        # balanced-ish spread rather than just first N
        idx=np.linspace(0,len(rows)-1,args.limit,dtype=int)
        rows=[rows[i] for i in idx]

    processor=AutoImageProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,trust_remote_code=True)
    model_path=hf_hub_download(MODEL_ID,ONNX_FILE,revision=MODEL_REVISION)
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=max(1,min(8,__import__("os").cpu_count() or 1))
    session=ort.InferenceSession(model_path,opts,providers=["CPUExecutionProvider"])
    arm_ids=[3,4]

    report=[]
    for k,r in enumerate(rows,1):
        src=args.image_root/r["path"]
        if not src.exists():
            src=args.image_root/"images"/r["path"]
        dst=args.out_root/r["path"]
        frac,arm_px,hand_px=clean_one(src,dst,processor,session,arm_ids)
        report.append({
            "path":r["path"],"label":r["label"],"video_id":r["video_id"],
            "frame_num":r["frame_num"],"split":r["split"],
            "mask_fraction":frac,"arm_pixels":arm_px,"hand_pixels":hand_px,
            "cleaned_path":str(dst)
        })
        if k%25==0 or k==len(rows):
            print(f"{k}/{len(rows)}")

    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(report[0].keys()))
        w.writeheader(); w.writerows(report)
    summary={
        "processed":len(report),
        "model":MODEL_ID,\n        "model_revision":MODEL_REVISION,\n        "onnx_file":ONNX_FILE,
        "arm_label_ids":arm_ids,
        "mediapipe_enabled":mp is not None,
        "mean_mask_fraction":float(np.mean([x["mask_fraction"] for x in report])),
        "median_mask_fraction":float(np.median([x["mask_fraction"] for x in report])),
        "zero_mask_images":int(sum(x["mask_fraction"]==0 for x in report)),
        "method":"SCHP Pascal-7 arm segmentation + MediaPipe Hands union; conservative dilation; half-resolution Telea inpainting composited only within feathered mask.",
        "note":"Derived ablation dataset only. Canonical originals are never modified."
    }
    (args.report.parent/"cleaning_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
