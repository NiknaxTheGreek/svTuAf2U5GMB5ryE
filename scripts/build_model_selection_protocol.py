#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-environment-split", required=True, type=Path)
    ap.add_argument("--d2-frame-manifest", required=True, type=Path)
    ap.add_argument("--training-script", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups = read_csv(args.d2_environment_split)
    frames = read_csv(args.d2_frame_manifest)

    train_groups = [r for r in groups if r["d2_split"] == "train"]
    test_groups = [r for r in groups if r["d2_split"] == "test"]
    if len(train_groups) != 44 or len(test_groups) != 11:
        raise SystemExit("Frozen D2 environment counts changed")

    train_frames = [r for r in frames if r["d2_split"] == "train"]
    test_frames = [r for r in frames if r["d2_split"] == "test"]
    if len(train_frames) != 2392 or len(test_frames) != 597:
        raise SystemExit("Frozen D2 frame counts changed")

    # Temporary validation is for recipe/epoch selection only. Final D1-D4
    # models are refit on each dataset's full non-test training portion.
    target_val_frames = round(len(train_frames) * 0.20)
    target_val_groups = round(len(train_groups) * 0.20)
    max_group_frames = 100

    d2_train_flip = sum(r["label"] == "flip" for r in train_frames)
    d2_train_flip_rate = d2_train_flip / len(train_frames)

    eligible = sorted(
        [r for r in train_groups if int(r["frame_count"]) <= max_group_frames],
        key=lambda r: r["environment_group_id"],
    )

    # Exact DP on frame count and group count, retaining one lexicographically
    # smallest group tuple for each (frames, flip_frames, groups, has_mixed).
    states: dict[tuple[int, int, int, int], tuple[str, ...]] = {
        (0, 0, 0, 0): ()
    }
    by_id = {r["environment_group_id"]: r for r in train_groups}

    for r in eligible:
        f = int(r["frame_count"])
        p = int(r["flip_frames"])
        mixed = int(r["group_category"] == "mixed")
        for state, selected in list(states.items()):
            sf, sp, sg, sm = state
            new_state = (
                sf + f,
                sp + p,
                sg + 1,
                min(1, sm + mixed),
            )
            if new_state[0] > target_val_frames or new_state[2] > target_val_groups:
                continue
            candidate = selected + (r["environment_group_id"],)
            current = states.get(new_state)
            if current is None or candidate < current:
                states[new_state] = candidate

    feasible: list[tuple[float, tuple[str, ...], int]] = []
    for (f, p, g, has_mixed), selected in states.items():
        if f != target_val_frames or g != target_val_groups or not has_mixed:
            continue
        flip_rate = p / f
        feasible.append((abs(flip_rate - d2_train_flip_rate), selected, p))

    if not feasible:
        raise SystemExit(
            "No exact D2 development validation split satisfies the frozen constraints"
        )

    feasible.sort(key=lambda x: (x[0], x[1]))
    _, validation_ids, validation_flip = feasible[0]
    validation_ids_set = set(validation_ids)

    dev_train_ids = {
        r["environment_group_id"] for r in train_groups
    } - validation_ids_set
    protected_test_ids = {r["environment_group_id"] for r in test_groups}

    if validation_ids_set & protected_test_ids or dev_train_ids & protected_test_ids:
        raise SystemExit("Protected D2 test group leaked into development")

    env_rows: list[dict[str, object]] = []
    for r in sorted(groups, key=lambda x: x["environment_group_id"]):
        gid = r["environment_group_id"]
        if gid in protected_test_ids:
            role = "protected_test"
        elif gid in validation_ids_set:
            role = "dev_validation"
        else:
            role = "dev_train"
        env_rows.append({**r, "model_selection_role": role})

    env_fields = list(groups[0].keys()) + ["model_selection_role"]
    env_path = args.out_dir / "d2_model_selection_environment_split.csv"
    write_csv(env_path, env_rows, env_fields)

    frame_rows: list[dict[str, object]] = []
    for r in frames:
        gid = r["environment_group_id"]
        if gid in protected_test_ids:
            role = "protected_test"
        elif gid in validation_ids_set:
            role = "dev_validation"
        else:
            role = "dev_train"
        frame_rows.append({**r, "model_selection_role": role})

    frame_fields = list(frames[0].keys()) + ["model_selection_role"]
    frame_path = args.out_dir / "d2_model_selection_frame_manifest.csv"
    write_csv(frame_path, frame_rows, frame_fields)

    def frame_stats(role: str) -> dict[str, int]:
        subset = [r for r in frame_rows if r["model_selection_role"] == role]
        return {
            "frames": len(subset),
            "flip_frames": sum(r["label"] == "flip" for r in subset),
            "notflip_frames": sum(r["label"] == "notflip" for r in subset),
            "environment_groups": len({r["environment_group_id"] for r in subset}),
            "temporal_clips": len({r["sequence_id"] for r in subset}),
        }

    dev_train_stats = frame_stats("dev_train")
    val_stats = frame_stats("dev_validation")
    protected_stats = frame_stats("protected_test")

    if val_stats["frames"] != target_val_frames:
        raise SystemExit("Development validation frame target changed")
    if val_stats["environment_groups"] != target_val_groups:
        raise SystemExit("Development validation group target changed")
    if val_stats["flip_frames"] != validation_flip:
        raise SystemExit("Development validation class count changed")
    if protected_stats["frames"] != 597:
        raise SystemExit("Protected D2 test size changed")

    # The validation groups are disjoint from development training by construction.
    train_val_env_overlap = len(dev_train_ids & validation_ids_set)
    train_val_seq_overlap = len(
        {r["sequence_id"] for r in frame_rows if r["model_selection_role"] == "dev_train"}
        & {r["sequence_id"] for r in frame_rows if r["model_selection_role"] == "dev_validation"}
    )
    if train_val_env_overlap or train_val_seq_overlap:
        raise SystemExit("Development train/validation source overlap detected")

    protocol = {
        "protocol_id": "D1_D4_COMMON_MODEL_SELECTION_V1",
        "selection_population": "D2 training only",
        "selection_reason": (
            "Use source-safe non-test data to select the common training duration "
            "without inspecting D2/D4 protected test images or metrics."
        ),
        "development_split": {
            "target_validation_fraction_of_d2_training_frames": 0.20,
            "validation_environment_groups": list(validation_ids),
            "constraints": {
                "exact_validation_frames": target_val_frames,
                "exact_validation_group_count": target_val_groups,
                "max_single_validation_group_frames": max_group_frames,
                "requires_at_least_one_mixed_label_environment": True,
                "tie_break": (
                    "Among exact feasible sets, minimize absolute flip-rate deviation "
                    "from D2 training, then choose lexicographically smallest group tuple."
                ),
            },
            "dev_train": dev_train_stats,
            "dev_validation": val_stats,
            "protected_test": protected_stats,
            "dev_train_validation_environment_overlap": train_val_env_overlap,
            "dev_train_validation_sequence_overlap": train_val_seq_overlap,
        },
        "fixed_model_recipe": {
            "architecture": "MobileNetV3-Small",
            "pretrained_weights": "ImageNet IMAGENET1K_V1",
            "image_size": 160,
            "seed": 2026,
            "batch_size": 64,
            "head_epochs": 2,
            "maximum_finetune_epochs": 6,
            "maximum_total_epochs": 8,
            "head_learning_rate": 0.001,
            "finetune_learning_rate": 0.0002,
            "optimizer": "AdamW",
            "weight_decay": 0.0001,
            "unfrozen_feature_blocks_during_finetune": 4,
            "loss": "unweighted BCEWithLogitsLoss",
            "decision_threshold": 0.5,
            "threshold_tuned": False,
            "data_augmentation": (
                "RandomResizedCrop(160, scale=0.90-1.00, ratio=0.95-1.05), "
                "RandomHorizontalFlip(0.5), ColorJitter(brightness=0.10, contrast=0.10), "
                "ImageNet normalization"
            ),
        },
        "selection_rule": {
            "only_selected_quantity": "training epoch",
            "candidate_epochs": list(range(1, 9)),
            "metric": "validation F1",
            "choose": "highest validation F1",
            "tie_break": "earliest epoch",
            "architecture_search": False,
            "threshold_search": False,
            "augmentation_search": False,
            "optimizer_search": False,
        },
        "final_refit_rule": {
            "after_epoch_lock": (
                "Retrain one model for each D1-D4 dataset from the same pretrained "
                "initialization/seed for the selected number of epochs using that "
                "dataset's full non-test training portion."
            ),
            "validation_rows_returned_to_final_training": True,
            "same_recipe_across_D1_D4": True,
            "class_rebalancing": "none",
        },
        "final_evaluation_rule": {
            "D2_D4_test_access_before_all_models_frozen": False,
            "evaluate_all_D1_D4_final_models_in_one_locked_evaluation_workflow": True,
            "primary_metric": "F1",
            "secondary_metrics": [
                "precision", "recall", "accuracy", "balanced_accuracy", "confusion_matrix"
            ],
            "claim_scope": {
                "D1": "supplied-split performance with known source leakage",
                "D2": "unseen conservative environment-group performance on all images",
                "D3": "post-hoc deduplicated supplied-split sensitivity result; source leakage remains",
                "D4": "unseen conservative environment-group performance after frozen deduplication",
            },
        },
        "training_recipe_source_sha256": sha256(args.training_script),
    }

    protocol_path = args.out_dir / "model_selection_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "inputs": {
            str(args.d2_environment_split): sha256(args.d2_environment_split),
            str(args.d2_frame_manifest): sha256(args.d2_frame_manifest),
            str(args.training_script): sha256(args.training_script),
        },
        "outputs": {
            env_path.name: sha256(env_path),
            frame_path.name: sha256(frame_path),
            protocol_path.name: sha256(protocol_path),
        },
        "acceptance": {
            "selection_uses_only_D2_training_groups": True,
            "protected_test_groups_in_development": 0,
            "development_train_validation_environment_overlap": 0,
            "development_train_validation_sequence_overlap": 0,
            "threshold_fixed_at_0_5": True,
            "only_epoch_selected": True,
            "final_refit_uses_full_training_portion": True,
        },
        "status": "PASS",
    }
    receipt_path = args.out_dir / "model_selection_protocol_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(protocol, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
