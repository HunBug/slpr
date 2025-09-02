from __future__ import annotations

import math
from typing import Dict

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(m))


def try_ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None
    s, _ = ssim(a, b, full=True, data_range=1.0)
    return float(s)


def compare_to_original(
    original: np.ndarray, recon: np.ndarray
) -> Dict[str, float | None]:
    return {
        "mse": mse(original, recon),
        "psnr": psnr(original, recon),
        "ssim": try_ssim(original, recon),
    }
