from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.scratch_v2 import (
    MemmapDataset, ScratchCNN, ScratchCNNConfig, candidate_config,
    load_candidate_bank, sha256_file,
)

BEST_O_ID="C14"
BEST_O_SHA="57f70fb3ebecbade054a18b22c423182175f84625179d49c3625e520b215bc49"
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


def ci(x):
    return [float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]


def resize_roundtrip(batch: torch.Tensor, hw: tuple[int,int]) -> torch.Tensor:
    small=TF.resize(batch,list(hw),interpolation=InterpolationMode.BILINEAR,antialias=True)
    return TF.resize(small,[398,224],interpolation=InterpolationMode.BILINEAR,antialias=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache-data",required=True,type=Path)
    ap.add_argument("--cache-index",required=True,type=Path)
    ap.add_argument("--bank",required=True,type=Path)
    ap.add_argument("--checkpoint",required=True,type=Path)
    ap.add_argument("--saved-predictions",required=True,type=Path)
    ap.add_argument("--output-dir",required=True,type=Path)
    args=ap.parse_args()

    if sha256_file(args.checkpoint)!=BEST_O_SHA:
        raise ValueError("Best O checkpoint SHA mismatch")
    bank=load_candidate_bank(args.bank)
    cfg=candidate_config(bank,BEST_O_ID)
    model=ScratchCNN(ScratchCNNConfig(depth=int(cfg["depth"]),start_filters=int(cfg["start_filters"]),dropout=float(cfg["dropout"])))
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if payload["candidate_id"]!=BEST_O_ID or payload["regime"]!="O" or payload["epoch"]!=20:
        raise ValueError("Best O checkpoint identity mismatch")
    model.load_state_dict(payload["model_state_dict"],strict=True)
    model.eval()

    ds=MemmapDataset(args.cache_data,args.cache_index)
    if len(ds)!=597: raise ValueError("O test count changed")
    loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0)
    probs={"original":[],"half_resolution":[],"quarter_resolution":[]}
    yy=[]
    with torch.no_grad():
        for batch in loader:
            x=batch["image"]
            variants=[
                x,
                resize_roundtrip(x,(199,112)),
                resize_roundtrip(x,(100,56)),
            ]
            for name,v in zip(probs,variants,strict=True):
                probs[name].extend(torch.sigmoid(model(v)).cpu().numpy().tolist())
            yy.extend(batch["label"].cpu().numpy().astype(int).tolist())

    y=np.asarray(yy,dtype=np.int64)
    p={k:np.asarray(v,dtype=np.float64) for k,v in probs.items()}
    pred={k:(v>=0.5).astype(np.int64) for k,v in p.items()}

    saved={}
    with args.saved_predictions.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["candidate_id"]==BEST_O_ID:saved[r["sample_id"]]=r
    ids=[r["sample_id"] for r in ds.rows]
    if len(saved)!=597:raise ValueError("Saved C14 prediction count changed")
    saved_p=np.asarray([float(saved[i]["prob_flip"]) for i in ids])
    saved_pred=np.asarray([int(saved[i]["y_pred"]) for i in ids],dtype=np.int64)
    if np.max(np.abs(saved_p-p["original"]))>1e-5 or not np.array_equal(saved_pred,pred["original"]):
        raise ValueError("Original model inference drift")

    metrics={k:metric(y,v) for k,v in p.items()}
    groups=np.asarray([r["video_id"] for r in ds.rows],dtype=object)
    uniq=np.asarray(sorted(set(groups.tolist())),dtype=object)
    gidx={g:np.where(groups==g)[0] for g in uniq}
    bootstrap={}
    for j,name in enumerate(("half_resolution","quarter_resolution")):
        frame=np.empty(N_BOOT); group=np.empty(N_BOOT)
        rng=np.random.default_rng(SEED+j*100)
        for b in range(N_BOOT):
            idx=rng.integers(0,len(y),size=len(y))
            frame[b]=f1_score(y[idx],pred[name][idx],zero_division=0)-f1_score(y[idx],pred["original"][idx],zero_division=0)
        rng=np.random.default_rng(SEED+j*100+1)
        for b in range(N_BOOT):
            sample=rng.choice(uniq,size=len(uniq),replace=True)
            idx=np.concatenate([gidx[g] for g in sample])
            group[b]=f1_score(y[idx],pred[name][idx],zero_division=0)-f1_score(y[idx],pred["original"][idx],zero_division=0)
        bootstrap[name]={"frame_delta_f1_ci95":ci(frame),"video_group_delta_f1_ci95":ci(group)}

    rows=[]
    for i,src in enumerate(ds.rows):
        rows.append({
            "sample_id":src["sample_id"],"video_id":src["video_id"],"frame_number":int(src["frame_number"]),"y_true":int(y[i]),
            "p_original":float(p["original"][i]),"p_half_resolution":float(p["half_resolution"][i]),"p_quarter_resolution":float(p["quarter_resolution"][i]),
            "pred_original":int(pred["original"][i]),"pred_half_resolution":int(pred["half_resolution"][i]),"pred_quarter_resolution":int(pred["quarter_resolution"][i]),
        })

    summary={
        "status":"POSTHOC_RESOLUTION_SENSITIVITY_COMPLETE",
        "selection_bearing":False,"retraining_performed":False,
        "candidate_id":BEST_O_ID,"checkpoint_sha256":BEST_O_SHA,
        "metrics":metrics,
        "delta_f1_vs_original":{
            "half_resolution":float(metrics["half_resolution"]["f1"]-metrics["original"]["f1"]),
            "quarter_resolution":float(metrics["quarter_resolution"]["f1"]-metrics["original"]["f1"]),
        },
        "bootstrap":bootstrap,
        "changed_prediction_count":{
            name:int(np.sum(pred[name]!=pred["original"])) for name in ("half_resolution","quarter_resolution")
        },
        "interpretation":"Post-hoc sensitivity to loss of spatial detail; cannot alter Best O."
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    with (args.output_dir/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
