#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit

BASE_FEATURES=[
    "brightness_mean","contrast_std","sharpness_laplacian_var",
    "saturation_mean","edge_density","jpeg_bytes"
]

def stats(a):
    a=np.asarray(a,float)
    q5,q25,q50,q75,q95=np.quantile(a,[.05,.25,.5,.75,.95])
    return {
        "mean":float(a.mean()),"std":float(a.std(ddof=0)),
        "p05":float(q5),"q1":float(q25),"median":float(q50),
        "q3":float(q75),"p95":float(q95),
        "min":float(a.min()),"max":float(a.max()),
        "iqr":float(q75-q25)
    }

def cohend(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    va=a.var(ddof=1); vb=b.var(ddof=1)
    sp=np.sqrt(((len(a)-1)*va+(len(b)-1)*vb)/(len(a)+len(b)-2))
    return float((b.mean()-a.mean())/sp) if sp>0 else 0.0

def eval_model(df,prefix,train_idx,test_idx,name):
    feats=[f"{prefix}_{x}" for x in BASE_FEATURES]
    X=df[feats].astype(float)
    y=(df["label"]=="flip").astype(int).values
    pipe=Pipeline([
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(max_iter=2000,solver="liblinear",random_state=42))
    ])
    pipe.fit(X.iloc[train_idx],y[train_idx])
    p=pipe.predict_proba(X.iloc[test_idx])[:,1]
    pred=(p>=.5).astype(int)
    return {
        "name":name,
        "accuracy":float(accuracy_score(y[test_idx],pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y[test_idx],pred)),
        "precision":float(precision_score(y[test_idx],pred,zero_division=0)),
        "recall":float(recall_score(y[test_idx],pred,zero_division=0)),
        "f1":float(f1_score(y[test_idx],pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y[test_idx],p)),
        "confusion_matrix":confusion_matrix(y[test_idx],pred).tolist(),
        "standardized_coefficients":{
            base:float(v) for base,v in zip(BASE_FEATURES,pipe.named_steps["clf"].coef_[0])
        }
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--paired-features",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(a.paired_features)
    df["video_id"]=df["video_id"].astype(str).str.zfill(4)
    df["sequence_id"]=df["label"].astype(str)+"__"+df["video_id"]

    y=(df["label"]=="flip").astype(int).values
    paired={}
    class_assoc={"original":{},"cleaned":{}}

    for base in BASE_FEATURES:
        o=df[f"original_{base}"].astype(float).values
        c=df[f"cleaned_{base}"].astype(float).values
        d=c-o
        corr=float(pearsonr(o,c).statistic)
        orig_sd=float(np.std(o,ddof=0))
        paired[base]={
            "original":stats(o),
            "cleaned":stats(c),
            "delta_cleaned_minus_original":stats(d),
            "mean_absolute_delta":float(np.mean(np.abs(d))),
            "mean_abs_delta_over_original_sd":float(np.mean(np.abs(d))/orig_sd) if orig_sd>0 else None,
            "paired_original_cleaned_r":corr,
            "fraction_abs_delta_gt_0_1_original_sd":float(np.mean(np.abs(d)>.1*orig_sd)) if orig_sd>0 else None,
            "fraction_abs_delta_gt_0_5_original_sd":float(np.mean(np.abs(d)>.5*orig_sd)) if orig_sd>0 else None,
        }
        for prefix,vals in [("original",o),("cleaned",c)]:
            nf=vals[df["label"].values=="notflip"]
            fl=vals[df["label"].values=="flip"]
            r,p=pointbiserialr(y,vals)
            class_assoc[prefix][base]={
                "notflip":stats(nf),"flip":stats(fl),
                "point_biserial_r":float(r),"p_value":float(p),
                "cohens_d_flip_vs_notflip":cohend(nf,fl)
            }

    tr=np.where(df["split"].astype(str).str.lower().eq("training").values)[0]
    te=np.where(df["split"].astype(str).str.lower().eq("testing").values)[0]

    groups=df["sequence_id"].values
    gss=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42)
    tr2,te2=next(gss.split(df,y,groups=groups))

    models={}
    for prefix in ["original","cleaned"]:
        models[prefix]={
            "supplied_original_split":eval_model(df,prefix,tr,te,f"{prefix}_supplied_original_split"),
            "provisional_clip_grouped":eval_model(df,prefix,tr2,te2,f"{prefix}_provisional_clip_grouped")
        }
    for split_name in ["supplied_original_split","provisional_clip_grouped"]:
        models["delta_"+split_name]={
            "roc_auc_cleaned_minus_original":
                models["cleaned"][split_name]["roc_auc"]-models["original"][split_name]["roc_auc"],
            "f1_cleaned_minus_original":
                models["cleaned"][split_name]["f1"]-models["original"][split_name]["f1"],
            "accuracy_cleaned_minus_original":
                models["cleaned"][split_name]["accuracy"]-models["original"][split_name]["accuracy"]
        }

    # Changes in label association caused by cleaning
    association_change={}
    for base in BASE_FEATURES:
        ro=class_assoc["original"][base]["point_biserial_r"]
        rc=class_assoc["cleaned"][base]["point_biserial_r"]
        do=class_assoc["original"][base]["cohens_d_flip_vs_notflip"]
        dc=class_assoc["cleaned"][base]["cohens_d_flip_vs_notflip"]
        association_change[base]={
            "original_r":ro,"cleaned_r":rc,"delta_r":rc-ro,
            "original_cohens_d":do,"cleaned_cohens_d":dc,"delta_cohens_d":dc-do
        }

    summary={
        "image_count":int(len(df)),
        "paired_feature_shift":paired,
        "class_association":{"original":class_assoc["original"],"cleaned":class_assoc["cleaned"]},
        "association_change":association_change,
        "feature_only_models":models,
        "mask_fraction":stats(df["mask_fraction"].astype(float).values),
        "notes":[
            "Cleaned images are a derived ablation dataset, not replacements for canonical originals.",
            "A feature shift after hand/arm removal may reflect removal of real hand pixels, inpainting artifacts, or both.",
            "The clip-grouped split is a diagnostic only until conservative environment/session groups are frozen."
        ]
    }
    (a.out_dir/"original_vs_cleaned_shortcut_analysis.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    rows=[]
    for base in BASE_FEATURES:
        rows.append({
            "feature":base,
            "mean_abs_delta_over_original_sd":paired[base]["mean_abs_delta_over_original_sd"],
            "paired_r":paired[base]["paired_original_cleaned_r"],
            **association_change[base]
        })
    pd.DataFrame(rows).to_csv(a.out_dir/"original_vs_cleaned_feature_summary.csv",index=False)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
