from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

N_BOOT = 5000
SEED = 20261001
CHAMPIONS = {"O":"C14","S":"C18","T":"C06","ST":"C07"}


def read_scratch(regime: str):
    rows=[]
    with Path(f"results/scratch/{regime}/predictions.csv").open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["candidate_id"]==CHAMPIONS[regime]:
                rows.append(r)
    return {r["sample_id"]:r for r in rows}


def read_resnet(regime: str):
    with Path(f"results/resnet/RESNET18_{regime}_predictions.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    return {r["sample_id"]:r for r in rows}


def f1(y, pred):
    return float(f1_score(y,pred,zero_division=0))


def ci(x):
    return [float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]


def main():
    out={"seed":SEED,"resamples":N_BOOT,"comparison":"ResNet18 minus frozen scratch champion","regimes":{}}
    for j,regime in enumerate(("O","S","T","ST")):
        s=read_scratch(regime); r=read_resnet(regime)
        if set(s)!=set(r):
            raise RuntimeError(f"{regime} sample identity mismatch")
        ids=sorted(s)
        y=np.asarray([int(s[i]["y_true"]) for i in ids],dtype=np.int64)
        if not np.array_equal(y,np.asarray([int(r[i]["y_true"]) for i in ids],dtype=np.int64)):
            raise RuntimeError(f"{regime} labels mismatch")
        sp=np.asarray([int(s[i]["y_pred"]) for i in ids],dtype=np.int64)
        rp=np.asarray([int(r[i]["y_pred"]) for i in ids],dtype=np.int64)
        groups=np.asarray([s[i]["video_id"] for i in ids],dtype=object)
        obs=f1(y,rp)-f1(y,sp)

        rng=np.random.default_rng(SEED+j*100+1)
        frame=np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx=rng.integers(0,len(ids),size=len(ids))
            frame[b]=f1(y[idx],rp[idx])-f1(y[idx],sp[idx])

        uniq=np.asarray(sorted(set(groups.tolist())),dtype=object)
        gidx={g:np.where(groups==g)[0] for g in uniq}
        rng=np.random.default_rng(SEED+j*100+2)
        group=np.empty(N_BOOT)
        for b in range(N_BOOT):
            sampled=rng.choice(uniq,size=len(uniq),replace=True)
            idx=np.concatenate([gidx[g] for g in sampled])
            group[b]=f1(y[idx],rp[idx])-f1(y[idx],sp[idx])

        out["regimes"][regime]={
            "scratch_champion":CHAMPIONS[regime],
            "n":len(ids),
            "video_groups":len(uniq),
            "scratch_f1":f1(y,sp),
            "resnet18_f1":f1(y,rp),
            "observed_f1_difference":obs,
            "frame_bootstrap_ci95":ci(frame),
            "video_group_bootstrap_ci95":ci(group),
            "group_ci_excludes_zero": bool(ci(group)[0] > 0 or ci(group)[1] < 0),
        }

    Path("results/resnet18/paired_bootstrap.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=["# Fixed ResNet18 vs frozen scratch champions","","Paired bootstrap comparison on identical saved test samples. Positive differences favor ResNet18. All analysis is post-hoc and non-selection-bearing.","","| Regime | Scratch | Scratch F1 | ResNet18 F1 | ΔF1 | Frame 95% CI | Video/group 95% CI |","|---|---:|---:|---:|---:|---:|---:|"]
    for regime in ("O","S","T","ST"):
        x=out["regimes"][regime]
        lines.append(f"| {regime} | {x['scratch_champion']} | {x['scratch_f1']:.4f} | {x['resnet18_f1']:.4f} | {x['observed_f1_difference']:+.4f} | {x['frame_bootstrap_ci95'][0]:+.3f}–{x['frame_bootstrap_ci95'][1]:+.3f} | {x['video_group_bootstrap_ci95'][0]:+.3f}–{x['video_group_bootstrap_ci95'][1]:+.3f} |")
    lines += ["","The fixed ResNet18 recipe is a comparator, not a replacement selection procedure. Interpret a difference cautiously when the video/group interval includes zero.",""]
    Path("reports/resnet18").mkdir(parents=True,exist_ok=True)
    Path("reports/resnet18/RESNET18_COMPARATOR_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
