from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from src.scratch_v2 import (
    EXPECTED_ARCHIVE_SHA256, EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_SPLIT_SHA256, ScratchCNN, ScratchCNNConfig, candidate_config,
    load_candidate_bank, preprocess_to_uint8, read_csv, sha256_file,
)
from src.secondary_perturbations_v2 import (
    apply_hand_mask, apply_matched_control_mask, rec709_grayscale,
)
from src.mediapipe_hand_mask_v2 import MediaPipeHandMasker

BEST_O_ID="C14"
BEST_O_SHA="57f70fb3ebecbade054a18b22c423182175f84625179d49c3625e520b215bc49"
HAND_MODEL_SHA="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
N_BOOT=5000
SEED=20261001


def metric(y,p):
    yp=(p>=0.5).astype(np.int64)
    tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
    return {
        "f1":float(f1_score(y,yp,zero_division=0)),
        "precision":float(precision_score(y,yp,zero_division=0)),
        "recall":float(recall_score(y,yp,zero_division=0)),
        "accuracy":float(accuracy_score(y,yp)),
        "balanced_accuracy":float(balanced_accuracy_score(y,yp)),
        "roc_auc":float(roc_auc_score(y,p)),
        "pr_auc":float(average_precision_score(y,p)),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"n":int(len(y)),
    }


def f1_from_pred(y,yp):
    return float(f1_score(y,yp,zero_division=0))


def ci(x):
    return [float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]


def boot_deltas(y,pred_orig,preds,groups):
    out={}
    uniq=np.asarray(sorted(set(groups.tolist())),dtype=object)
    gidx={g:np.where(groups==g)[0] for g in uniq}
    names=list(preds)
    frame={n:np.empty(N_BOOT) for n in names}
    group={n:np.empty(N_BOOT) for n in names}
    frame_contrast=np.empty(N_BOOT)
    group_contrast=np.empty(N_BOOT)

    rng=np.random.default_rng(SEED)
    for b in range(N_BOOT):
        idx=rng.integers(0,len(y),size=len(y))
        base=f1_from_pred(y[idx],pred_orig[idx])
        vals={}
        for n in names:
            vals[n]=f1_from_pred(y[idx],preds[n][idx])-base
            frame[n][b]=vals[n]
        frame_contrast[b]=vals["hand_mask"]-vals["matched_control"]

    rng=np.random.default_rng(SEED+1)
    for b in range(N_BOOT):
        sampled=rng.choice(uniq,size=len(uniq),replace=True)
        idx=np.concatenate([gidx[g] for g in sampled])
        base=f1_from_pred(y[idx],pred_orig[idx])
        vals={}
        for n in names:
            vals[n]=f1_from_pred(y[idx],preds[n][idx])-base
            group[n][b]=vals[n]
        group_contrast[b]=vals["hand_mask"]-vals["matched_control"]

    for n in names:
        out[n]={
            "frame_delta_f1_ci95":ci(frame[n]),
            "video_group_delta_f1_ci95":ci(group[n]),
        }
    out["hand_minus_control_delta_f1_contrast"]={
        "frame_ci95":ci(frame_contrast),
        "video_group_ci95":ci(group_contrast),
    }
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",required=True,type=Path)
    ap.add_argument("--manifest",required=True,type=Path)
    ap.add_argument("--membership",required=True,type=Path)
    ap.add_argument("--bank",required=True,type=Path)
    ap.add_argument("--checkpoint",required=True,type=Path)
    ap.add_argument("--hand-model",required=True,type=Path)
    ap.add_argument("--saved-predictions",required=True,type=Path)
    ap.add_argument("--output-dir",required=True,type=Path)
    args=ap.parse_args()

    if sha256_file(args.archive)!=EXPECTED_ARCHIVE_SHA256: raise ValueError("archive SHA mismatch")
    if sha256_file(args.manifest)!=EXPECTED_DATASET_MANIFEST_SHA256: raise ValueError("manifest SHA mismatch")
    if sha256_file(args.membership)!=EXPECTED_SPLIT_SHA256["O"]: raise ValueError("O split SHA mismatch")
    if sha256_file(args.checkpoint)!=BEST_O_SHA: raise ValueError("Best O checkpoint SHA mismatch")
    if sha256_file(args.hand_model)!=HAND_MODEL_SHA: raise ValueError("hand model SHA mismatch")

    bank=load_candidate_bank(args.bank)
    cfg=candidate_config(bank,BEST_O_ID)
    model=ScratchCNN(ScratchCNNConfig(depth=int(cfg["depth"]),start_filters=int(cfg["start_filters"]),dropout=float(cfg["dropout"])))
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if ckpt["candidate_id"]!=BEST_O_ID or ckpt["regime"]!="O" or ckpt["epoch"]!=20:
        raise ValueError("Best O checkpoint payload mismatch")
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    model.eval()

    manifest=read_csv(args.manifest); split=read_csv(args.membership)
    by_id={r["sample_id"]:r for r in manifest}
    test=[r for r in split if r["role"]=="test"]
    if len(test)!=597: raise ValueError("O test count changed")

    saved={}
    with args.saved_predictions.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["candidate_id"]==BEST_O_ID: saved[r["sample_id"]]=r
    if len(saved)!=597: raise ValueError("saved C14 O predictions count changed")

    rows=[]
    variants={"original":[],"grayscale":[],"hand_mask":[],"matched_control":[]}
    yy=[]
    with MediaPipeHandMasker(args.hand_model,0.5,0.5,2) as masker, ZipFile(args.archive) as z:
        for k,row in enumerate(test):
            src=by_id[row["sample_id"]]
            with z.open(src["archive_member"]) as fh:
                with Image.open(fh) as im:
                    source_rgb=np.asarray(im.convert("RGB"),dtype=np.uint8)
                    original=preprocess_to_uint8(im)
            masks,count=masker.candidate_masks_from_source_rgb(original,source_rgb)
            mask=masks["mp_seed_skin"]
            hand=apply_hand_mask(original,mask,fill="local_border_median")
            control,control_mask=apply_matched_control_mask(original,mask)
            gray=rec709_grayscale(original)
            stack=np.stack([original,gray,hand,control]).astype(np.float32)/255.0
            with torch.no_grad():
                probs=torch.sigmoid(model(torch.from_numpy(stack))).cpu().numpy()
            y=1 if row["label"]=="flip" else 0
            yy.append(y)
            for name,prob in zip(variants,probs,strict=True):
                variants[name].append(float(prob))
            s=saved[row["sample_id"]]
            if int(s["y_true"])!=y or int(s["y_pred"])!=int(probs[0]>=0.5):
                raise ValueError(f"saved prediction identity mismatch: {row['sample_id']}")
            if abs(float(s["prob_flip"])-float(probs[0]))>1e-5:
                raise ValueError(f"saved original probability drift: {row['sample_id']}")
            rows.append({
                "sample_id":row["sample_id"],"video_id":row["video_id"],"frame_number":int(row["frame_number"]),
                "y_true":y,"detected_hands":int(count),"mask_coverage":float(mask.mean()),
                "control_coverage":float(control_mask.mean()),
                "p_original":float(probs[0]),"p_grayscale":float(probs[1]),"p_hand_mask":float(probs[2]),"p_matched_control":float(probs[3]),
                "pred_original":int(probs[0]>=0.5),"pred_grayscale":int(probs[1]>=0.5),"pred_hand_mask":int(probs[2]>=0.5),"pred_matched_control":int(probs[3]>=0.5),
            })
            if (k+1)%100==0 or k+1==len(test): print(f"perturbation_progress={k+1}/597",flush=True)

    y=np.asarray(yy,dtype=np.int64)
    p={n:np.asarray(v,dtype=np.float64) for n,v in variants.items()}
    pred={n:(v>=0.5).astype(np.int64) for n,v in p.items()}
    metrics={n:metric(y,v) for n,v in p.items()}
    deltas={n:float(metrics[n]["f1"]-metrics["original"]["f1"]) for n in ("grayscale","hand_mask","matched_control")}
    groups=np.asarray([r["video_id"] for r in rows],dtype=object)
    boot=boot_deltas(y,pred["original"],{n:pred[n] for n in ("grayscale","hand_mask","matched_control")},groups)

    detected=np.asarray([r["mask_coverage"]>0 for r in rows])
    changed={
        n:int(np.sum(pred[n]!=pred["original"])) for n in ("grayscale","hand_mask","matched_control")
    }
    mean_abs_prob_delta={
        n:float(np.mean(np.abs(p[n]-p["original"]))) for n in ("grayscale","hand_mask","matched_control")
    }
    hand_specific_prob=float(np.mean(np.abs(p["hand_mask"]-p["original"]))-np.mean(np.abs(p["matched_control"]-p["original"])))

    summary={
        "status":"POSTHOC_SHORTCUT_PERTURBATION_COMPLETE",
        "selection_bearing":False,"retraining_performed":False,"threshold":0.5,
        "candidate_id":BEST_O_ID,"checkpoint_sha256":BEST_O_SHA,
        "test_images":597,"videos":len(set(groups.tolist())),
        "hand_detector_nonzero_images":int(detected.sum()),"hand_detector_zero_images":int((~detected).sum()),
        "metrics":metrics,"delta_f1_vs_original":deltas,"bootstrap":boot,
        "changed_prediction_count":changed,"mean_absolute_probability_change":mean_abs_prob_delta,
        "hand_minus_control_mean_absolute_probability_change":hand_specific_prob,
        "interpretation_rule":"Evidence for hand-specific shortcut dependence requires hand-mask degradation materially beyond matched equal-area control, preferably with video-group bootstrap support. This detector is not validated complete human segmentation."
    }

    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    changed_rows=[r for r in rows if r["pred_grayscale"]!=r["pred_original"] or r["pred_hand_mask"]!=r["pred_original"] or r["pred_matched_control"]!=r["pred_original"]]
    with (args.output_dir/"changed_cases.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(changed_rows)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=["# Phase 11 — Best-O perturbation sensitivity","","Post-hoc frozen-model analysis only; no retraining or champion change.","","| Variant | F1 | Precision | Recall | Accuracy | ΔF1 vs original |","|---|---:|---:|---:|---:|---:|"]
    for n in ("original","grayscale","hand_mask","matched_control"):
        m=metrics[n]; d=0 if n=="original" else deltas[n]
        lines.append(f"| {n} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['accuracy']:.4f} | {d:+.4f} |")
    lines+=["",f"Hand-mask minus matched-control ΔF1 contrast, video/group 95% CI: {boot['hand_minus_control_delta_f1_contrast']['video_group_ci95']}.","",summary["interpretation_rule"],""]
    (args.output_dir/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
