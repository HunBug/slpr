from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Any

import numpy as np
from tqdm import tqdm


def _rand_int(low: int, high: int, rng: np.random.Generator) -> int:
    return int(rng.integers(low, high + 1))


def _sample_patch(
    img: np.ndarray,
    y: int,
    x: int,
    k: int,
    jitter: float,
    rng: np.random.Generator,
) -> float:
    h, w = img.shape
    r = k // 2
    # jitter in continuous coords, then round
    jy = int(round(rng.normal(0.0, jitter)))
    jx = int(round(rng.normal(0.0, jitter)))
    yy0 = max(0, min(h - 1, y + jy - r))
    xx0 = max(0, min(w - 1, x + jx - r))
    yy1 = max(0, min(h - 1, y + jy + r))
    xx1 = max(0, min(w - 1, x + jx + r))
    # random coordinate in the window
    sy = _rand_int(yy0, yy1, rng)
    sx = _rand_int(xx0, xx1, rng)
    return float(img[sy, sx])


def _map_coord(
    level_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    y: int,
    x: int,
) -> Tuple[int, int]:
    th, tw = target_shape
    lh, lw = level_shape
    yy = int(round(y * (lh - 1) / max(1, th - 1)))
    xx = int(round(x * (lw - 1) / max(1, tw - 1)))
    return yy, xx


def stochastic_reconstruct(
    laplacians: List[np.ndarray[Any, Any]],
    base: np.ndarray[Any, Any],
    target_shape: Tuple[int, int],
    patch_size: int,
    jitter: float,
    noise_strength: float,
    seed: int | None = None,
    show_progress: bool = True,
    update_cb: Optional[Callable[[int], None]] = None,
) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(seed)
    th, tw = target_shape
    out = np.zeros((th, tw), dtype=np.float32)

    # Upsample base to target (prefer OpenCV, fallback to Pillow)
    try:
        import cv2  # type: ignore

        base_u = cv2.resize(base, (tw, th), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image
        # Use numeric constant for broad compatibility
        # 2 == BILINEAR in PIL.Image
        resample = 2
        img = Image.fromarray((np.clip(base, 0.0, 1.0) * 255).astype(np.uint8))
        img = img.resize((tw, th), resample)
        base_u = (np.asarray(img).astype(np.float32) / 255.0)

    out += base_u

    # For each level, sample per-pixel from local neighborhoods
    for lvl, lap in enumerate(laplacians):
        lh, lw = lap.shape
        y_iter = range(th)
        if show_progress and th * tw > 256 * 256:
            y_iter = tqdm(y_iter, desc=f"lvl {lvl} rows", leave=False)
        for y in y_iter:
            for x in range(tw):
                yy, xx = _map_coord((lh, lw), (th, tw), y, x)
                v = _sample_patch(lap, yy, xx, patch_size, jitter, rng)
                out[y, x] += v
            if update_cb is not None:
                update_cb(1)

    if noise_strength > 0:
        noise = rng.normal(0.0, noise_strength, size=out.shape).astype(
            np.float32
        )
        out += noise

    out = np.clip(out, 0.0, 1.0)
    return out


def stochastic_average(
    laplacians: List[np.ndarray[Any, Any]],
    base: np.ndarray[Any, Any],
    target_shape: Tuple[int, int],
    patch_size: int,
    jitter: float,
    noise_strength: float,
    samples: int,
    seed: int | None = None,
    show_progress: bool = False,
    update_cb: Optional[Callable[[int], None]] = None,
) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(seed)
    acc: np.ndarray[Any, Any] | None = None
    it = range(samples)
    if show_progress and samples > 1:
        from tqdm import trange

        it = trange(samples, desc="averaging", leave=False)
    for _ in it:
        s = stochastic_reconstruct(
            laplacians,
            base,
            target_shape,
            patch_size,
            jitter,
            noise_strength,
            seed=int(rng.integers(0, 2**31 - 1)),
            show_progress=False,
            update_cb=update_cb,
        )
        if acc is None:
            acc = s
        else:
            acc += s
    if samples <= 0:
        raise ValueError("samples must be > 0")
    assert acc is not None
    return acc / float(samples)
