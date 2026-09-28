from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch

from src.config import apply_overrides, load_config, to_runtime
from src.splits import validate_dataset_split_compatibility
from src.training import build_optimizer
from src.utils import configure_logging, resolve_device, seed_everything

LOGGER = logging.getLogger("monreader")
ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MonReader experiment configuration.")
    parser.add_argument("--config", required=True, help="Path to a validated YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Safe override for the random seed.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Safe override for runtime device selection.",
    )
    return parser.parse_args(argv)


def run_synthetic_smoke(runtime) -> dict[str, object]:
    seed_everything(runtime.seed)
    device = resolve_device(runtime.device)

    model = torch.nn.Linear(4, 1).to(device)
    optimizer = build_optimizer(
        model.parameters(),
        name=runtime.optimizer_name,
        learning_rate=runtime.learning_rate,
        weight_decay=runtime.weight_decay,
        momentum=runtime.momentum,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    inputs = torch.tensor(
        [[0.0, 1.0, 0.5, -0.5], [1.0, 0.0, -0.5, 0.5]],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    value = float(loss.detach().cpu().item())
    if not math.isfinite(value):
        raise RuntimeError("Synthetic smoke loss is not finite.")

    return {
        "status": "ok",
        "mode": runtime.mode,
        "device": device.type,
        "dataset_id": runtime.dataset_id,
        "split_id": runtime.split_id,
        "seed": runtime.seed,
        "optimizer": runtime.optimizer_name,
        "loss": round(value, 6),
    }


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        config = apply_overrides(config, seed=args.seed, device=args.device)
        runtime = to_runtime(config)
        validate_dataset_split_compatibility(
            ROOT / "manifests" / "dataset_split_compatibility.yaml",
            dataset_id=runtime.dataset_id,
            split_id=runtime.split_id,
        )
        if runtime.mode != "synthetic_smoke":
            raise ValueError(f"Unsupported run mode: {runtime.mode}")
        result = run_synthetic_smoke(runtime)
    except Exception as exc:
        LOGGER.error("Run failed: %s", exc)
        return 1

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
