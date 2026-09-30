from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

CANVAS_WIDTH = 224
CANVAS_HEIGHT = 398
EXPECTED_BANK_SHA256 = "a4bebfc6c21df01e9294dacccb26ba84602cb3800b20414ad1d51f6f343c3b12"
EXPECTED_ARCHIVE_BYTES = 939_921_132
EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"
EXPECTED_DATASET_MANIFEST_SHA256 = "3fb8a29f1a51fc7930fe6c876d0fcb8d99ec59e147d4abbff503da4870f15b0b"
EXPECTED_SPLIT_SHA256 = {
    "O": "d48e182da1b170365b39693317e1edf6dd3395c02ccabdd6a771d7be5e226680",
    "S": "a54cce72ad1f275096f06b98bbc9ec32393e9f112cc0536abd132571e1919619",
    "T": "0a1f39ebb01f14989aad37b87c39fbb874307f81bdc4770d15e8df7c7b9aa625",
    "ST": "ed62ad946187ba33154e84e6732fdcf6110e8309c402b568259bcf2c0e177a85",
}


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_generator(seed: int = 42) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_candidate_bank(path: str | Path) -> dict:
    bank_path = Path(path)
    digest = sha256_file(bank_path)
    if digest != EXPECTED_BANK_SHA256:
        raise ValueError(f"Candidate-bank SHA-256 mismatch: {digest}")
    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    ids = [row["candidate_id"] for row in payload["candidates"]]
    if ids != [f"C{i:02d}" for i in range(1, 21)]:
        raise ValueError("Candidate IDs/order differ from frozen C01-C20 bank")
    fixed = payload["fixed_training"]
    expected = {
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
    if fixed != expected:
        raise ValueError("Candidate-bank fixed_training block differs from V2")
    return payload


def candidate_config(bank: dict, candidate_id: str) -> dict:
    matches = [dict(row) for row in bank["candidates"] if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one candidate config for {candidate_id}")
    return matches[0]


@dataclass(frozen=True)
class ScratchCNNConfig:
    depth: int
    start_filters: int
    dropout: float

    def __post_init__(self) -> None:
        if not 1 <= self.depth <= 7:
            raise ValueError("depth must be in [1,7]")
        if not 8 <= self.start_filters <= 64:
            raise ValueError("start_filters must be in [8,64]")
        if not 0.0 <= self.dropout <= 0.5:
            raise ValueError("dropout must be in [0,0.5]")


class ScratchCNN(nn.Module):
    def __init__(self, config: ScratchCNNConfig) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 3
        out_channels = config.start_filters
        for _ in range(config.depth):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
            out_channels *= 2
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(in_channels, 1)
        self.apply(_kaiming_initialize)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.features(inputs)
        values = self.pool(values).flatten(1)
        values = self.dropout(values)
        return self.classifier(values).squeeze(1)


def _kaiming_initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_optimizer(model: nn.Module, config: dict):
    name = str(config["optimizer"]).lower()
    kwargs = {
        "lr": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
    }
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def resolved_optimizer_settings(config: dict) -> dict:
    name = str(config["optimizer"]).lower()
    common = {
        "name": name,
        "lr": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
    }
    if name in {"adam", "adamw"}:
        return {**common, "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False}
    if name == "rmsprop":
        return {**common, "alpha": 0.99, "eps": 1e-8, "momentum": 0.0, "centered": False}
    raise ValueError(name)


def preprocess_to_uint8(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    source_width, source_height = rgb.size
    scale = min(CANVAS_WIDTH / source_width, CANVAS_HEIGHT / source_height)
    resized_width = max(1, min(CANVAS_WIDTH, round(source_width * scale)))
    resized_height = max(1, min(CANVAS_HEIGHT, round(source_height * scale)))
    resized = TF.resize(
        rgb,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    pad_left = (CANVAS_WIDTH - resized_width) // 2
    pad_right = CANVAS_WIDTH - resized_width - pad_left
    pad_top = (CANVAS_HEIGHT - resized_height) // 2
    pad_bottom = CANVAS_HEIGHT - resized_height - pad_top
    canvas = TF.pad(
        resized,
        [pad_left, pad_top, pad_right, pad_bottom],
        fill=(0, 0, 0),
        padding_mode="constant",
    )
    tensor = TF.pil_to_tensor(canvas)
    if tensor.shape != (3, CANVAS_HEIGHT, CANVAS_WIDTH):
        raise RuntimeError(f"Unexpected transformed shape: {tuple(tensor.shape)}")
    return tensor.numpy()


class MemmapDataset(Dataset):
    def __init__(self, cache_data: str | Path, cache_index: str | Path):
        self.cache_data = str(Path(cache_data))
        self.cache_index = Path(cache_index)
        self.rows = read_csv(self.cache_index)
        if not self.rows:
            raise ValueError("Cache index is empty")
        self._array = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        return state

    def _memmap(self):
        if self._array is None:
            self._array = np.load(self.cache_data, mmap_mode="r")
            if self._array.shape != (len(self.rows), 3, CANVAS_HEIGHT, CANVAS_WIDTH):
                raise ValueError(f"Unexpected cache shape: {self._array.shape}")
            if self._array.dtype != np.uint8:
                raise ValueError(f"Unexpected cache dtype: {self._array.dtype}")
        return self._array

    def __getitem__(self, index: int):
        row = self.rows[index]
        array = np.array(self._memmap()[index], copy=True)
        image = torch.from_numpy(array).to(dtype=torch.float32) / 255.0
        label = 1.0 if row["label"] == "flip" else 0.0
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "frame_number": int(row["frame_number"]),
        }
