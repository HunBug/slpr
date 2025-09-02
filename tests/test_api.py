from __future__ import annotations

import numpy as np

from slpr import SLPRParams, SLPRSession, reconstruct_image


def _img_gray(h: int, w: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, num=h*w, dtype=np.float32).reshape(h, w)
    return y


def _img_rgb(h: int, w: int) -> np.ndarray:
    y = _img_gray(h, w)
    return np.stack([y, 1 - y, 0.5 * np.ones_like(y)], axis=2)


def test_stateless_gray_shapes():
    img = _img_gray(64, 96)
    out = reconstruct_image(
        img, "gray", (128, 160), levels=4, samples=1, seed=123
    )
    assert out.shape == (128, 160)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_session_rgb_determinism():
    img = _img_rgb(32, 40)
    params = SLPRParams(levels=4, samples=1)
    s1 = SLPRSession(img, "rgb", params)
    s2 = SLPRSession(img, "rgb", params)
    out1 = s1.reconstruct((64, 80), seed=42)
    out2 = s2.reconstruct((64, 80), seed=42)
    assert out1.shape == (64, 80, 3)
    assert np.allclose(out1, out2, atol=1e-6)


def test_luma_reconstruct():
    img = _img_rgb(48, 48)
    params = SLPRParams(levels=4, samples=2)
    sess = SLPRSession(img, "luma", params)
    out = sess.reconstruct((96, 96), seed=7)
    assert out.shape == (96, 96, 3)
