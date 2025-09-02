from __future__ import annotations

from typing import List

import cv2
import numpy as np


def build_gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    assert image.ndim == 2, "Only grayscale supported for now"
    g = [image]
    for _ in range(1, levels):
        blurred = cv2.GaussianBlur(g[-1], (0, 0), sigmaX=1.0, sigmaY=1.0)
        down = cv2.resize(
            blurred,
            (blurred.shape[1] // 2, blurred.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
        if down.shape[0] < 8 or down.shape[1] < 8:
            break
        g.append(down)
    return g


def _upsample_to_shape(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def build_laplacian_pyramid(
    image: np.ndarray, levels: int
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """
    Returns: (laplacians, base_low_res, gaussians)
    laplacians: list from level0 (finest) to levelN-1 (coarser)
    base_low_res: smallest gaussian image (GN)
    gaussians: the gaussian pyramid (G0..GN)
    """
    gaussians = build_gaussian_pyramid(image, levels)
    laplacians: list[np.ndarray] = []
    for i in range(len(gaussians) - 1):
        up = _upsample_to_shape(gaussians[i + 1], gaussians[i].shape[:2])
        lap = gaussians[i] - up
        laplacians.append(lap)
    base = gaussians[-1]
    return laplacians, base, gaussians


def visualize_pyramid(gaussians: list[np.ndarray]) -> list[np.ndarray]:
    # Normalize each level for viewing
    vis = []
    for g in gaussians:
        m, M = float(np.min(g)), float(np.max(g))
        if M - m < 1e-8:
            v = np.zeros_like(g)
        else:
            v = (g - m) / (M - m)
        vis.append(v)
    return vis
