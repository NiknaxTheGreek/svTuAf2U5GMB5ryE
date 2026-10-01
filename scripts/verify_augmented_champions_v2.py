from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.scratch_v2 import EXPECTED_BANK_SHA256, EXPECTED_SPLIT_SHA256, sha256_file
from src.secondary_augmentation_v2 import EXPECTED_AUGMENTATION_CONFIG_SHA256

CHAMPIONS = {"O": "C14", "S": "C18", "T": "C06", "ST": "C07"}


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args()
    records=[]
    for regime,candidate in CHAMPIONS.items():
        pt=args.root/f"{regime}_{candidate}_aug.pt"
        js=args.root/f"{regime}_{candidate}_aug.json"
        if not pt.is_file() or not js.is_file():
            raise FileNotFoundError(f"Missing augmented checkpoint for {regime}")
        m=json.loads(js.read_text(encoding="utf-8"))
        if m["status"]!="FROZEN_AUGMENTED_FINAL_EPOCH_20" or m["epochs_completed"]!=20:
            raise ValueError(f"{regime} augmented checkpoint status invalid")
        if m["candidate_id"]!=candidate or m["regime"]!=regime:
            raise ValueError(f"{regime} augmented identity mismatch")
        if m["test_rows_loaded"]!=0 or m["validation_rows_loaded"]!=0:
            raise ValueError(f"{regime} augmented training touched test/validation")
        if m["bank_sha256"]!=EXPECTED_BANK_SHA256:
            raise ValueError(f"{regime} bank mismatch")
        if m["augmentation_config_sha256"]!=EXPECTED_AUGMENTATION_CONFIG_SHA256:
            raise ValueError(f"{regime} augmentation config mismatch")
        if m["split_sha256"]!=EXPECTED_SPLIT_SHA256[regime]:
            raise ValueError(f"{regime} split mismatch")
        if m["checkpoint_sha256"]!=sha256_file(pt):
            raise ValueError(f"{regime} checkpoint hash mismatch")
        p=torch.load(pt,map_location="cpu",weights_only=False)
        if p["regime"]!=regime or p["candidate_id"]!=candidate or p["epoch"]!=20:
            raise ValueError(f"{regime} checkpoint payload mismatch")
        records.append({"regime":regime,"candidate_id":candidate,"checkpoint_sha256":m["checkpoint_sha256"],"final_train_loss":m["final_train_loss"]})
    payload={"status":"PASS_ALL_4_AUGMENTED_CHECKPOINTS_FROZEN","selection_bearing":False,"records":records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":payload["status"],"checkpoint_count":4},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
