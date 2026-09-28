"""MonReader visualization utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_image_grid(
    image_paths: Sequence[str | Path],
    *,
    titles: Sequence[str] | None = None,
    columns: int = 4,
    figsize: tuple[float, float] | None = None,
):
    """Return a simple reviewer-facing grid of image files."""
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("image_paths must not be empty")
    if columns < 1:
        raise ValueError("columns must be at least 1")
    if titles is not None and len(titles) != len(paths):
        raise ValueError("titles must match image_paths length")

    rows = (len(paths) + columns - 1) // columns
    if figsize is None:
        figsize = (3.0 * columns, 4.0 * rows)

    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    flat_axes = np.asarray(axes).reshape(-1)

    for index, (axis, path) in enumerate(zip(flat_axes, paths, strict=False)):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        if titles is not None:
            axis.set_title(titles[index], fontsize=8)
        axis.axis("off")

    for axis in flat_axes[len(paths):]:
        axis.axis("off")

    fig.tight_layout()
    return fig
