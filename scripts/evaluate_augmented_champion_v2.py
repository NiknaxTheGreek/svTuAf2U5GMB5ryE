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

from src.scratch_v2 import (
    EXPECTED_SPLIT_SHA256, MemmapDataset, ScratchCNN, ScratchCNNConfig,
    candidate_config, load_candidate_bank, sha256_file,
)
from src.secondary_augmentation_v2 import EXPECTED_AUGMENTATION_CONFIG_SHA256

CHAMPIONS={"O":"C14","S":"C18","T":"C06","ST":"C07"}
TEST_ROWS={"O":597,"S":770,"T":624,"ST":161}
N_BOOT=5000
SEED=20261001


def metrics(y,p):
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


def ci(a):
    return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache-data",required=True,type=Path)
    ap.add_argument("--cache-index",required=True,type=Path)
    ap.add_argument("--cache-receipt",required=True,type=Path)
    ap.add_argument("--bank",required=True,type=Path)
    ap.add_argument("--checkpoint",required=True,type=Path)
    ap.add_argument("--metadata",required=True,type=Path)
    ap.add_argument("--registry",required=True,type=Path)
    ap.add_argument("--regime",required=True,choices=("O","S","T","ST"))
    ap.add_argument("--saved-original-predictions",required=True,type=Path)
    ap.add_argument("--output-dir",required=True,type=Path)
    args=ap.parse_args()

    candidate=CHAMPIONS[args.regime]
    reg=json.loads(args.registry.read_text(encoding="utf-8"))
    if reg["status"]!="PASS_ALL_4_AUGMENTED_CHECKPOINTS_FROZEN":
        raise ValueError("Augmented checkpoint gate not passed")
    meta=json.loads(args.metadata.read_text(encoding="utf-8"))
    if meta["regime"]!=args.regime or meta["candidate_id"]!=candidate:
        raise ValueError("Augmented metadata identity mismatch")
    if meta["augmentation_config_sha256"]!=EXPECTED_AUGMENTATION_CONFIG_SHA256:
        raise ValueError("Augmentation config mismatch")
    if meta["checkpoint_sha256"]!=sha256_file(args.checkpoint):
        raise ValueError("Augmented checkpoint SHA mismatch")

    receipt=json.loads(args.cache_receipt.read_text(encoding="utf-8"))
    if receipt["regime"]!=args.regime or receipt["role"]!="test" or receipt["rows"]!=TEST_ROWS[args.regime]:
        raise ValueError("Test cache population mismatch")
    if receipt["split_sha256"]!=EXPECTED_SPLIT_SHA256[args.regime]:
        raise ValueError("Test cache split mismatch")

    bank=load_candidate_bank(args.bank)
    cfg=candidate_config(bank,candidate)
    model=ScratchCNN(ScratchCNNConfig(depth=int(cfg["depth"]),start_filters=int(cfg["start_filters"]),dropout=float(cfg["dropout"])))
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(payload["model_state_dict"],strict=True)
    model.eval()

    ds=MemmapDataset(args.cache_data,args.cache_index)
    loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0)
    yy=[]; pp=[]
    with torch.no_grad():
        for batch in loader:
            pp.extend(torch.sigmoid(model(batch["image"])).cpu().numpy().tolist())
            yy.extend(batch["label"].cpu().numpy().astype(int).tolist())
    y=np.asarray(yy,dtype=np.int64); p=np.asarray(pp,dtype=np.float64); yp=(p>=0.5).astype(np.int64)
    aug_m=metrics(y,p)

    saved={}
    with args.saved_original_predictions.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["candidate_id"]==candidate: saved[r["sample_id"]]=r
    if len(saved)!=len(ds):
        raise ValueError("Original champion prediction count mismatch")
    ids=[r["sample_id"] for r in ds.rows]
    orig_y=np.asarray([int(saved[i]["y_true"]) for i in ids],dtype=np.int64)
    orig_p=np.asarray([float(saved[i]["prob_flip"]) for i in ids],dtype=np.float64)
    orig_pred=np.asarray([int(saved[i]["y_pred"]) for i in ids],dtype=np.int64)
    if not np.array_equal(y,orig_y):
        raise ValueError("Original/augmented label alignment mismatch")
    orig_m=metrics(y,orig_p)
    observed=aug_m["f1"]-orig_m["f1"]

    groups=np.asarray([r["video_id"] for r in ds.rows],dtype=object)
    frame=np.empty(N_BOOT); group=np.empty(N_BOOT)
    rng=np.random.default_rng(SEED)
    for b in range(N_BOOT):
        idx=rng.integers(0,len(y),size=len(y))
        frame[b]=f1_score(y[idx],yp[idx],zero_division=0)-f1_score(y[idx],orig_pred[idx],zero_division=0)
    uniq=np.asarray(sorted(set(groups.tolist())),dtype=object)
    gidx={g:np.where(groups==g)[0] for g in uniq}
    rng=np.random.default_rng(SEED+1)
    for b in range(N_BOOT):
        samp=rng.choice(uniq,size=len(uniq),replace=True)
        idx=np.concatenate([gidx[g] for g in samp])
        group[b]=f1_score(y[idx],yp[idx],zero_division=0)-f1_score(y[idx],orig_pred[idx],zero_division=0)

    rows=[]
    for src,yt,prob,pred in zip(ds.rows,y,p,yp,strict=True):
        rows.append({"sample_id":src["sample_id"],"video_id":src["video_id"],"frame_number":int(src["frame_number"]),"y_true":int(yt),"prob_flip":float(prob),"y_pred":int(pred)})

    summary={
        "status":"SECONDARY_AUGMENTED_CHAMPION_EVALUATED",
        "selection_bearing":False,"regime":args.regime,"candidate_id":candidate,
        "checkpoint_sha256":meta["checkpoint_sha256"],"augmentation_config_sha256":EXPECTED_AUGMENTATION_CONFIG_SHA256,
        "original_frozen_champion_metrics":orig_m,"augmented_retrain_metrics":aug_m,
        "observed_delta_f1_augmented_minus_original":float(observed),
        "frame_bootstrap_delta_f1_ci95":ci(frame),
        "video_group_bootstrap_delta_f1_ci95":ci(group),
        "bootstrap_resamples":N_BOOT,"bootstrap_seed":SEED,
        "interpretation":"Post-hoc fixed augmentation ablation; cannot replace the frozen champion."
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/f"{args.regime}_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    with (args.output_dir/f"{args.regime}_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
