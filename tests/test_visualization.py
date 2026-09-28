from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from src.visualization import plot_image_grid


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (12, 8), color=color)
    image.save(path, format="JPEG")


def test_plot_image_grid_returns_figure(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    _write_image(first, (10, 20, 30))
    _write_image(second, (40, 50, 60))
    figure = plot_image_grid([first, second], titles=["a", "b"], columns=2)
    assert len(figure.axes) == 2
    plt.close(figure)


def test_plot_image_grid_rejects_title_mismatch(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    _write_image(image, (10, 20, 30))
    with pytest.raises(ValueError, match="titles"):
        plot_image_grid([image], titles=[])
