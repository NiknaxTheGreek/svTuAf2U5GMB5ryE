from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = "a4bebfc6c21df01e9294dacccb26ba84602cb3800b20414ad1d51f6f343c3b12"
EXPECTED_IDS = [f"C{i:02d}" for i in range(1, 21)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--md-out", required=True)
    args = ap.parse_args()

    path = Path(args.bank)
    digest = sha256_file(path)
    bank = json.loads(path.read_text(encoding="utf-8"))
    candidates = bank.get("candidates", [])
    fixed = bank.get("fixed_training", {})
    search = bank.get("search_space", {})

    errors = []
    if digest != EXPECTED_SHA256:
        errors.append(f"SHA-256 mismatch: {digest} != {EXPECTED_SHA256}")
    if len(candidates) != 20:
        errors.append(f"Expected 20 candidates, found {len(candidates)}")

    ids = [c.get("candidate_id") for c in candidates]
    if ids != EXPECTED_IDS:
        errors.append(f"Candidate IDs/order differ from C01-C20: {ids}")
    if len(set(ids)) != len(ids):
        errors.append("Candidate IDs are not unique")

    def identity_without_id(c):
        return json.dumps({k: v for k, v in c.items() if k != "candidate_id"}, sort_keys=True)

    unique_configs = len({identity_without_id(c) for c in candidates})
    if unique_configs != 20:
        errors.append(f"Expected 20 unique hyperparameter configurations, found {unique_configs}")

    opt_allowed = set(search["optimizer"])
    violations = []
    for c in candidates:
        cid = c["candidate_id"]
        checks = [
            ("depth", search["depth"]["min"] <= c["depth"] <= search["depth"]["max"]),
            ("start_filters", search["start_filters"]["min"] <= c["start_filters"] <= search["start_filters"]["max"]),
            ("dropout", search["dropout"]["min"] <= c["dropout"] <= search["dropout"]["max"]),
            ("learning_rate", search["learning_rate"]["min"] <= c["learning_rate"] <= search["learning_rate"]["max"]),
            ("batch_size", search["batch_size"]["min"] <= c["batch_size"] <= search["batch_size"]["max"]),
            ("optimizer", c["optimizer"] in opt_allowed),
            ("seed", c["seed"] == 42),
        ]
        wd = c["weight_decay"]
        wd_ok = wd == 0 or (
            search["weight_decay"]["positive_min"] <= wd <= search["weight_decay"]["positive_max"]
        )
        checks.append(("weight_decay", wd_ok))
        for field, ok in checks:
            if not ok:
                violations.append({"candidate_id": cid, "field": field, "value": c[field]})

    if violations:
        errors.append(f"Range/allowed-value violations: {violations}")

    expected_fixed = {
        "augmentation": False,
        "canvas_hxw": [398, 224],
        "checkpoint": "final_epoch_20",
        "class_sampling": "natural",
        "early_stopping": False,
        "epochs": 20,
        "gradient_clipping": None,
        "loss": "BCEWithLogitsLoss",
        "model_seed": 42,
        "scaling": "[0,1]",
        "scheduler": None,
        "threshold": 0.5,
        "validation": False,
    }
    if fixed != expected_fixed:
        errors.append(f"fixed_training block differs from expected V2 policy: {fixed}")

    if bank.get("status") != "FROZEN_BEFORE_V2_TEST_EXPOSURE":
        errors.append(f"Unexpected bank status: {bank.get('status')}")
    if bank.get("trial_count") != 20:
        errors.append(f"Unexpected trial_count: {bank.get('trial_count')}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "bank_path": path.as_posix(),
        "sha256": digest,
        "expected_sha256": EXPECTED_SHA256,
        "candidate_count": len(candidates),
        "unique_candidate_ids": len(set(ids)),
        "unique_hyperparameter_configurations": unique_configs,
        "candidate_ids": ids,
        "range_violations": violations,
        "fixed_training_matches_v2": fixed == expected_fixed,
        "search_space": search,
        "fixed_training": fixed,
        "errors": errors,
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = f"""# Frozen 20-Candidate Bank Audit

Status: **{result['status']}**

- SHA-256: `{digest}`
- Expected SHA-256: `{EXPECTED_SHA256}`
- Candidate count: {len(candidates)}
- Unique candidate IDs: {len(set(ids))}
- Unique hyperparameter configurations: {unique_configs}
- IDs: C01–C20
- Fixed epochs: {fixed.get('epochs')}
- Validation: {fixed.get('validation')}
- Early stopping: {fixed.get('early_stopping')}
- Checkpoint rule: {fixed.get('checkpoint')}
- Threshold: {fixed.get('threshold')}
- Augmentation in primary comparison: {fixed.get('augmentation')}

The bank is acceptable only if this audit is PASS before scratch-CNN test evaluation.
"""
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text(md, encoding="utf-8")

    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
