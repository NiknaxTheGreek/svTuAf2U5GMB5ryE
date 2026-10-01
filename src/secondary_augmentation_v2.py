from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

EXPECTED_AUGMENTATION_CONFIG_SHA256 = "5168b13d3ce4f286df71a1f823cb78889098fa843602154c34683ede80dce18b"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_augmentation_config(path: str | Path) -> dict:
    path = Path(path)
    digest = sha256_file(path)
    if digest != EXPECTED_AUGMENTATION_CONFIG_SHA256:
        raise ValueError(f"Augmentation config SHA mismatch: {digest}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["selection_bearing"] is not False:
        raise ValueError("Augmentation ablation must remain non-selection-bearing")
    if cfg["training"] != {
        "early_stopping": False,
        "epochs": 20,
        "seed": 42,
        "threshold": 0.5,
        "validation": False,
    }:
        raise ValueError("Frozen augmentation training block changed")
    return cfg


def _uniform(low: float, high: float, generator: torch.Generator | None) -> float:
    x = torch.rand((), generator=generator).item()
    return low + (high - low) * x


def sample_params(
    height: int,
    width: int,
    cfg: dict,
    generator: torch.Generator | None = None,
) -> dict:
    aug = cfg["augmentation"]
    affine_cfg = aug["affine"]
    photo_cfg = aug["photometric"]

    apply_affine = torch.rand((), generator=generator).item() < float(affine_cfg["probability"])
    if apply_affine:
        angle = _uniform(*map(float, affine_cfg["rotation_degrees_uniform"]), generator)
        tx_frac = _uniform(*map(float, affine_cfg["translation_fraction_x_uniform"]), generator)
        ty_frac = _uniform(*map(float, affine_cfg["translation_fraction_y_uniform"]), generator)
        scale = _uniform(*map(float, affine_cfg["scale_uniform"]), generator)
        translate = [int(round(tx_frac * width)), int(round(ty_frac * height))]
    else:
        angle = 0.0
        scale = 1.0
        translate = [0, 0]

    apply_photo = torch.rand((), generator=generator).item() < float(photo_cfg["probability"])
    if apply_photo:
        brightness = _uniform(*map(float, photo_cfg["brightness_factor_uniform"]), generator)
        contrast = _uniform(*map(float, photo_cfg["contrast_factor_uniform"]), generator)
    else:
        brightness = 1.0
        contrast = 1.0

    return {
        "apply_affine": bool(apply_affine),
        "angle_degrees": float(angle),
        "translate_x_pixels": int(translate[0]),
        "translate_y_pixels": int(translate[1]),
        "scale": float(scale),
        "apply_photometric": bool(apply_photo),
        "brightness_factor": float(brightness),
        "contrast_factor": float(contrast),
    }


def apply_params(image: torch.Tensor, params: dict) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW image, got {tuple(image.shape)}")
    out = image
    if params["apply_affine"]:
        out = TF.affine(
            out,
            angle=float(params["angle_degrees"]),
            translate=[int(params["translate_x_pixels"]), int(params["translate_y_pixels"])],
            scale=float(params["scale"]),
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
            center=None,
        )
    if params["apply_photometric"]:
        out = TF.adjust_brightness(out, float(params["brightness_factor"]))
        out = TF.adjust_contrast(out, float(params["contrast_factor"]))
    return out.clamp_(0.0, 1.0)


def augment_batch(batch: torch.Tensor, cfg: dict) -> torch.Tensor:
    if batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError(f"Expected NCHW batch, got {tuple(batch.shape)}")
    out = []
    h, w = int(batch.shape[2]), int(batch.shape[3])
    for image in batch:
        params = sample_params(h, w, cfg, generator=None)
        out.append(apply_params(image, params))
    return torch.stack(out, dim=0)


def stable_sample_generator(sample_id: str, seed: int = 20261001) -> torch.Generator:
    digest = hashlib.sha256(f"{seed}|{sample_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    g = torch.Generator()
    g.manual_seed(value)
    return g
