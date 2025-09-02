from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .utils import save_image


def save_pyramid_images(gaussians: List[np.ndarray], out_dir: Path) -> None:
    for i, g in enumerate(gaussians):
        save_image(g, out_dir / f"pyramid_level{i}.png")
