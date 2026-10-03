from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    summaries = {}
    for regime in ("O", "S", "T", "ST"):
        path = args.input_root / regime / f"{regime}_summary.json"
        summaries[regime] = json.loads(path.read_text(encoding="utf-8"))

    lines = [
        "# Phase 11B — Frozen-champion chroma counterfactual results",
        "",
        "Post-hoc, non-selection-bearing sensitivity analysis. "
        "The frozen O/S/T/ST champions remain unchanged.",
        "",
        "| Regime | Variant | F1 | ΔF1 vs original | Video/group 95% CI | Changed predictions |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for regime in ("O", "S", "T", "ST"):
        s = summaries[regime]
        base = s["metrics"]["original"]["f1"]
        lines.append(
            f"| {regime} | original | {base:.4f} | +0.0000 | — | 0 |"
        )
        for variant in ("grayscale", "matched_rotation", "naturalistic_v2"):
            f1 = s["metrics"][variant]["f1"]
            delta = s["delta_f1_vs_original"][variant]
            ci = s["bootstrap"][variant]["video_group_delta_f1_ci95"]
            changed = s["changed_prediction_count"][variant]
            lines.append(
                f"| {regime} | {variant} | {f1:.4f} | {delta:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {changed} |"
            )

    lines += [
        "",
        "## Interpretation framework",
        "",
        "- Grayscale is the crude chroma-removal ablation.",
        "- Matched +90° chroma rotation is a strong colour-distribution control "
        "that preserves luminance/spatial structure.",
        "- Naturalistic v2 is the independently validated plausible RGB "
        "counterfactual with target-label-independent donor selection.",
        "- A naturalistic-v2 effect provides stronger evidence that chromatic "
        "cues influence the frozen model than grayscale alone.",
        "",
        "No result in this report can redefine Best O/S/T/ST.",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
