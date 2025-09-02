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


def _map_row_cols(
    level_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    y: int,
) -> Tuple[int, np.ndarray[Any, Any]]:
    """Map row y to level-row and target columns to level columns.

    Returns (yy_scalar, xx_vector) for the given row y. This matches
    applying the previous _map_coord for all x in the row.
    """
    th, tw = target_shape
    lh, lw = level_shape
    yy = int(round(y * (lh - 1) / max(1, th - 1)))
    # Vectorized mapping for all columns
    if tw <= 1:
        xx = np.zeros((tw,), dtype=np.int32)
    else:
        x = np.arange(tw, dtype=np.float32)
        xx = np.round(x * (lw - 1) / float(tw - 1)).astype(np.int32)
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

    r = max(0, int(patch_size) // 2)

    # For each level, sample in row blocks to reduce Python overhead
    block_rows = 32
    for lvl, lap in enumerate(laplacians):
        lh, lw = lap.shape
        y_blocks = range(0, th, block_rows)
        if show_progress and th * tw > 256 * 256:
            y_blocks = tqdm(y_blocks, desc=f"lvl {lvl} rows", leave=False)
        # Precompute xx mapping for the whole target width once per level
        _, xx_all = _map_row_cols((lh, lw), (th, tw), 0)
        xx_all = xx_all[None, :]  # (1, tw)
        for y0 in y_blocks:
            bh = min(block_rows, th - y0)
            ys = np.arange(y0, y0 + bh, dtype=np.int32)
            # Map all rows to level rows
            if th <= 1:
                yy = np.zeros((bh, 1), dtype=np.int32)
            else:
                yy = np.round(
                    ys.astype(np.float32) * (lh - 1) / float(max(1, th - 1))
                ).astype(np.int32)[:, None]  # (bh, 1)

            # Jitter per-pixel in the block
            if jitter != 0.0:
                jy = np.rint(
                    rng.normal(0.0, jitter, size=(bh, tw))
                ).astype(np.int32)
                jx = np.rint(
                    rng.normal(0.0, jitter, size=(bh, tw))
                ).astype(np.int32)
            else:
                jy = np.zeros((bh, tw), dtype=np.int32)
                jx = np.zeros((bh, tw), dtype=np.int32)

            # Compute window centers and bounds
            cy = np.clip(yy + jy, 0, lh - 1)
            cx = np.clip(xx_all + jx, 0, lw - 1)
            yy0 = np.clip(cy - r, 0, lh - 1)
            yy1 = np.clip(cy + r, 0, lh - 1)
            xx0 = np.clip(cx - r, 0, lw - 1)
            xx1 = np.clip(cx + r, 0, lw - 1)

            # Draw random coords within inclusive bounds
            hy = (yy1 - yy0 + 1).astype(np.int32)
            hx = (xx1 - xx0 + 1).astype(np.int32)
            ry = rng.random(size=(bh, tw))
            rx = rng.random(size=(bh, tw))
            sy = (yy0 + np.floor(ry * hy)).astype(np.int32)
            sx = (xx0 + np.floor(rx * hx)).astype(np.int32)

            # Gather and accumulate
            out[y0: y0 + bh, :] += lap[sy, sx].astype(np.float32)
            if update_cb is not None:
                update_cb(bh)

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
