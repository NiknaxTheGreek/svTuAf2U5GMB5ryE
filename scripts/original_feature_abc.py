#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit

FEATURES=[
    "brightness_mean","contrast_std","sharpness_laplacian_var",
    "saturation_mean","edge_density","jpeg_bytes"
]

def qstats(s):
    a=np.asarray(s,float)
    q1,med,q3=np.quantile(a,[.25,.5,.75])
    return {
        "n":int(len(a)),"mean":float(a.mean()),"std":float(a.std(ddof=0)),
        "median":float(med),"q1":float(q1),"q3":float(q3),"iqr":float(q3-q1),
        "min":float(a.min()),"max":float(a.max())
    }

def cohend(x0,x1):
    x0=np.asarray(x0,float); x1=np.asarray(x1,float)
    n0,n1=len(x0),len(x1)
    v0=x0.var(ddof=1); v1=x1.var(ddof=1)
    sp=np.sqrt(((n0-1)*v0+(n1-1)*v1)/(n0+n1-2))
    return float((x1.mean()-x0.mean())/sp) if sp>0 else 0.0

def eval_split(df,train_idx,test_idx,split_name):
    X=df[FEATURES].copy()
    y=(df["label"]=="flip").astype(int).values
    pipe=Pipeline([
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(max_iter=2000,solver="liblinear",random_state=42))
    ])
    pipe.fit(X.iloc[train_idx],y[train_idx])
    p=pipe.predict_proba(X.iloc[test_idx])[:,1]
    pred=(p>=0.5).astype(int)
    metrics={
        "split_name":split_name,
        "train_n":int(len(train_idx)),"test_n":int(len(test_idx)),
        "train_flip_rate":float(y[train_idx].mean()),
        "test_flip_rate":float(y[test_idx].mean()),
        "accuracy":float(accuracy_score(y[test_idx],pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y[test_idx],pred)),
        "precision":float(precision_score(y[test_idx],pred,zero_division=0)),
        "recall":float(recall_score(y[test_idx],pred,zero_division=0)),
        "f1":float(f1_score(y[test_idx],pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y[test_idx],p)),
        "confusion_matrix":confusion_matrix(y[test_idx],pred).tolist(),
    }
    coefs=dict(zip(FEATURES,pipe.named_steps["clf"].coef_[0].tolist()))
    metrics["standardized_logit_coefficients"]=coefs
    return metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features-csv",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(a.features_csv)
    df["sequence_id"]=df["label"].astype(str)+"__"+df["video_id"].astype(str).str.zfill(4)

    # A: distributions by class
    distributions={}
    assoc_rows=[]
    y=(df["label"]=="flip").astype(int).values
    for feat in FEATURES:
        nf=df.loc[df.label=="notflip",feat].astype(float)
        fl=df.loc[df.label=="flip",feat].astype(float)
        r,p=pointbiserialr(y,df[feat].astype(float).values)
        d=cohend(nf.values,fl.values)
        distributions[feat]={
            "notflip":qstats(nf),
            "flip":qstats(fl),
            "flip_minus_notflip_mean":float(fl.mean()-nf.mean()),
            "flip_minus_notflip_median":float(fl.median()-nf.median()),
            "cohens_d_flip_vs_notflip":d,
            "point_biserial_r":float(r),
            "point_biserial_p":float(p),
        }
        assoc_rows.append({
            "feature":feat,
            "point_biserial_r":float(r),
            "p_value":float(p),
            "cohens_d_flip_vs_notflip":d,
            "abs_r":abs(float(r)),
            "abs_d":abs(d)
        })
    assoc_rows=sorted(assoc_rows,key=lambda x:x["abs_r"],reverse=True)

    # C1: supplied original split
    tr=np.where(df["split"].astype(str).str.lower().eq("training").values)[0]
    te=np.where(df["split"].astype(str).str.lower().eq("testing").values)[0]
    supplied=eval_split(df,tr,te,"supplied_original_train_test")

    # C2: provisional clip-grouped diagnostic
    groups=df["sequence_id"].values
    gss=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=42)
    tr2,te2=next(gss.split(df[FEATURES],y,groups=groups))
    grouped=eval_split(df,tr2,te2,"provisional_grouped_by_temporal_clip_80_20")
    grouped["train_sequence_count"]=int(df.iloc[tr2]["sequence_id"].nunique())
    grouped["test_sequence_count"]=int(df.iloc[te2]["sequence_id"].nunique())
    grouped["sequence_overlap"]=int(len(set(df.iloc[tr2]["sequence_id"]) & set(df.iloc[te2]["sequence_id"])))

    summary={
        "image_count":int(len(df)),
        "features":FEATURES,
        "class_counts":df["label"].value_counts().to_dict(),
        "A_distributions_by_label":distributions,
        "B_association_ranking":assoc_rows,
        "C_feature_only_baselines":{
            "supplied_original_split":supplied,
            "provisional_clip_grouped_diagnostic":grouped
        },
        "notes":[
            "Associations are descriptive, not causal.",
            "The supplied split is known to contain same-sequence cross-split leakage.",
            "The clip-grouped diagnostic prevents exact temporal-clip overlap but is not the final corrected D2 split because conservative higher-level session grouping is not yet frozen."
        ]
    }
    (a.out_dir/"abc_original_feature_analysis.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    pd.DataFrame(assoc_rows).to_csv(a.out_dir/"feature_associations.csv",index=False)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
