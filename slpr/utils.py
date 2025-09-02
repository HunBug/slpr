from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image
import cv2


def to_numpy_gray(img: Image.Image) -> np.ndarray[Any, Any]:
    if img.mode != "L":
        img = img.convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def to_numpy_rgb(img: Image.Image) -> np.ndarray[Any, Any]:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def from_numpy_image(arr: np.ndarray[Any, Any]) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        return Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    else:
        raise ValueError("from_numpy_image expects 2D or 3D (H,W,3) array")


def load_image(path: Path, mode: str = "gray") -> np.ndarray[Any, Any]:
    img = Image.open(path)
    if mode == "gray":
        return to_numpy_gray(img)
    elif mode in ("rgb", "luma"):
        return to_numpy_rgb(img)
    else:
        raise ValueError(f"Unknown color mode: {mode}")


def save_image(arr: np.ndarray[Any, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from_numpy_image(arr).save(path)


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def make_output_dir(
    output_root: Path,
    stamp: str,
    asset_name: str,
    config_name: str,
) -> Path:
    out = output_root / stamp / asset_name / config_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


@dataclass
class CasePaths:
    output_dir: Path
    pyramid_dir: Path

    @classmethod
    def create(
        cls,
        output_root: Path,
        stamp: str,
        asset_name: str,
        config_name: str,
    ) -> "CasePaths":
        base = make_output_dir(
            output_root,
            stamp,
            asset_name,
            config_name,
        )
        pyramid = base / "pyramid"
        pyramid.mkdir(parents=True, exist_ok=True)
        return cls(output_dir=base, pyramid_dir=pyramid)


# Color conversions
def rgb_to_gray(img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("rgb_to_gray expects (H,W,3)")
    # BT.601 luma
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.astype(np.float32)


def rgb_to_ycbcr(
    img: np.ndarray[Any, Any],
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("rgb_to_ycbcr expects (H,W,3)")
    # Use cv2 for correctness; cv2 expects 0..255 uint8
    u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    return y, cb, cr


def ycbcr_to_rgb(
    y: np.ndarray[Any, Any],
    cb: np.ndarray[Any, Any],
    cr: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    if y.shape != cb.shape or y.shape != cr.shape:
        raise ValueError("Y, Cb, Cr shapes must match")
    ycrcb = np.stack(
        [
            np.clip(y, 0.0, 1.0),
            np.clip(cr, 0.0, 1.0),
            np.clip(cb, 0.0, 1.0),
        ],
        axis=2,
    )
    u8 = (ycrcb * 255.0).astype(np.uint8)
    rgb = cv2.cvtColor(u8, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
    return rgb
