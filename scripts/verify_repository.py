#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-12

REQUIRED_FILES = [
    "README.md",
    "PROJECT_STATE.yaml",
    "requirements.txt",
    "requirements-torch-cpu.txt",
    "docs/DATASET_AUDIT.md",
    "docs/SOURCE_GROUP_RECONSTRUCTION.md",
    "docs/SPLIT_POLICY.md",
    "docs/BASELINE.md",
    "docs/SINGLE_IMAGE_MODEL.md",
    "docs/FINAL_TEST_EVALUATION.md",
    "docs/POST_HOLDOUT_ANALYSIS.md",
    "docs/SEQUENCE_EXTENSION.md",
    "docs/FINAL_REPORT.md",
    "audit/audit_summary.json",
    "audit/source_group_summary.json",
    "audit/split_summary.json",
    "results/baseline_metrics.json",
    "results/step6_model_metrics.json",
    "results/final_test_metrics.json",
    "results/final_test_errors_by_group.csv",
    "results/post_holdout_analysis.json",
    "results/post_holdout_source_groups.csv",
    "results/post_holdout_frame_position.csv",
    "results/post_holdout_robustness.csv",
    "results/sequence_metrics.json",
    "splits/source_group_split.csv",
    "scripts/dataset_audit.py",
    "scripts/reconstruct_source_groups.py",
    "scripts/create_group_split.py",
    "scripts/run_baseline.py",
    "scripts/train_mobilenet_v3_small.py",
    "scripts/evaluate_final_test.py",
    "scripts/post_holdout_analysis.py",
    "scripts/sequence_extension.py",
    ".github/workflows/dataset-audit.yml",
    ".github/workflows/model-step6.yml",
    ".github/workflows/final-test-preflight.yml",
    ".github/workflows/final-test-once.yml",
    ".github/workflows/post-holdout-step8.yml",
    ".github/workflows/sequence-step9.yml",
]

FORBIDDEN_LEGACY_FILES = [
    "docs/POST_HOLDOUT_DIAGNOSTICS.md",
    "results/step8_frame_position_diagnostics.csv",
    "results/step8_perturbation_robustness.csv",
    "results/step8_post_holdout_summary.json",
    "results/step8_source_group_diagnostics.csv",
]

EXPECTED_SPLIT_SHA256 = "5eaf917adc51db9a9bd10d9d4c6487050284df6e47c928d18ec5d1e94d64b551"


class VerificationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise VerificationError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def get(obj: dict, path: str):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            fail(f"Missing JSON path {path}")
        cur = cur[part]
    return cur


def expect_close(name: str, actual, expected: float) -> None:
    try:
        value = float(actual)
    except Exception as exc:
        fail(f"{name} is not numeric: {actual!r} ({exc})")
    if abs(value - expected) > TOL:
        fail(f"{name}: expected {expected}, observed {value}")


def verify_files() -> None:
    missing = [p for p in REQUIRED_FILES if not (ROOT / p).is_file()]
    if missing:
        fail(f"Missing required files: {missing}")
    legacy = [p for p in FORBIDDEN_LEGACY_FILES if (ROOT / p).exists()]
    if legacy:
        fail(f"Redundant legacy files still present: {legacy}")


def verify_python_syntax() -> None:
    for path in sorted((ROOT / "scripts").glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            fail(f"Python syntax failure in {path.relative_to(ROOT)}: {exc}")


def verify_environment_pins() -> None:
    core = set(
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    expected_core = {
        "gdown==6.4.0",
        "numpy==2.5.3",
        "pandas==3.0.6",
        "Pillow==12.3.0",
        "scikit-learn==1.9.1",
        "scipy==1.18.1",
    }
    if core != expected_core:
        fail(f"Core environment pins changed: {sorted(core)}")

    torch_text = (ROOT / "requirements-torch-cpu.txt").read_text(encoding="utf-8")
    for token in (
        "https://download.pytorch.org/whl/cpu",
        "torch==2.8.0",
        "torchvision==0.23.0",
    ):
        if token not in torch_text:
            fail(f"Missing CPU torch environment pin: {token}")


def verify_split() -> None:
    path = ROOT / "splits/source_group_split.csv"
    observed_sha = sha256(path)
    if observed_sha != EXPECTED_SPLIT_SHA256:
        fail(f"Frozen split SHA changed: {observed_sha}")

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 117:
        fail(f"Expected 117 source groups, observed {len(rows)}")

    ids = [r["source_group_id"] for r in rows]
    if len(set(ids)) != 117:
        fail("Duplicate source_group_id in frozen split")

    group_counts = Counter(r["partition"] for r in rows)
    if group_counts != Counter({"train": 70, "validation": 23, "test": 24}):
        fail(f"Unexpected partition group counts: {dict(group_counts)}")

    frame_counts = Counter()
    label_counts = Counter()
    for row in rows:
        frame_counts[row["partition"]] += int(row["frame_count"])
        label_counts[row["label"]] += 1

    if frame_counts != Counter({"train": 1795, "validation": 597, "test": 597}):
        fail(f"Unexpected partition frame counts: {dict(frame_counts)}")
    if label_counts != Counter({"flip": 65, "notflip": 52}):
        fail(f"Unexpected source-group label counts: {dict(label_counts)}")


def verify_metrics() -> None:
    baseline = load_json("results/baseline_metrics.json")
    expect_close(
        "baseline.validation.f1",
        get(baseline, "validation.f1"),
        0.8369905956112853,
    )

    model = load_json("results/step6_model_metrics.json")
    expect_close(
        "step6.validation.f1",
        get(model, "validation.f1"),
        0.9295302013422819,
    )
    expect_close(
        "step6.f1_delta_vs_baseline",
        get(model, "f1_delta_vs_baseline"),
        0.09253960573099662,
    )
    if get(model, "selection.best_epoch") != 4:
        fail("Selected model epoch changed")
    if get(model, "selection.decision_threshold") != 0.5:
        fail("Selected model threshold changed")

    final = load_json("results/final_test_metrics.json")
    expect_close("final_test.f1", get(final, "test.f1"), 0.7514450867052023)
    expect_close("final_test.precision", get(final, "test.precision"), 0.851528384279476)
    expect_close("final_test.recall", get(final, "test.recall"), 0.6724137931034483)
    expect_close("final_test.accuracy", get(final, "test.accuracy"), 0.7839195979899497)
    if get(final, "test.tp") != 195 or get(final, "test.fn") != 95:
        fail("Final-test positive confusion counts changed")
    if get(final, "test.tn") != 273 or get(final, "test.fp") != 34:
        fail("Final-test negative confusion counts changed")
    if get(final, "threshold_tuned_on_test") is not False:
        fail("Final-test threshold governance changed")

    post = load_json("results/post_holdout_analysis.json")
    if get(post, "governance.analysis_only") is not True:
        fail("Post-holdout analysis is no longer marked analysis-only")
    if get(post, "governance.post_test_tuning_performed") is not False:
        fail("Post-holdout tuning governance changed")

    sequence = load_json("results/sequence_metrics.json")
    if get(sequence, "selected_method") != "mean_prob":
        fail("Sequence aggregation selection changed")
    expect_close(
        "sequence.validation.f1",
        get(sequence, "validation_sequence.f1"),
        0.9629629629629629,
    )
    expect_close(
        "sequence.exploratory_test.f1",
        get(sequence, "test_sequence_exploratory.f1"),
        0.8,
    )
    if get(sequence, "governance.test_used_for_aggregation_selection") is not False:
        fail("Sequence test-selection governance changed")


def verify_state_and_docs() -> None:
    state = (ROOT / "PROJECT_STATE.yaml").read_text(encoding="utf-8")
    if "current_phase: STEP_10_REPRODUCIBILITY_AND_PROJECT_REPORT" not in state:
        fail("PROJECT_STATE current phase is stale")
    if "post_holdout_diagnostics:" in state:
        fail("Duplicate legacy post_holdout_diagnostics state section remains")
    if "holdout_status: CONSUMED" not in state:
        fail("Final holdout status is not recorded as consumed")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "docs/FINAL_REPORT.md").read_text(encoding="utf-8")
    for name, text in (("README", readme), ("FINAL_REPORT", report)):
        if "0.75145" not in text:
            fail(f"{name} does not report the locked final F1")
        if "0.92953" not in text:
            fail(f"{name} does not report the locked validation F1")
        if "post-holdout" not in text.lower():
            fail(f"{name} does not preserve post-holdout framing")


def main() -> int:
    checks = [
        ("required files", verify_files),
        ("Python syntax", verify_python_syntax),
        ("environment pins", verify_environment_pins),
        ("frozen split", verify_split),
        ("locked metrics", verify_metrics),
        ("state and documentation", verify_state_and_docs),
    ]

    for name, func in checks:
        func()
        print(f"PASS: {name}")

    print("PASS: repository reproducibility and governance verification")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
