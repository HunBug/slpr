import numpy as np

from slpr.utils import rgb_to_gray, rgb_to_ycbcr, ycbcr_to_rgb


def test_rgb_to_gray_shape_and_range():
    img = np.random.rand(16, 16, 3).astype(np.float32)
    y = rgb_to_gray(img)
    assert y.shape == (16, 16)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_ycbcr_roundtrip_identity_on_solid():
    solid = np.ones((8, 8, 3), dtype=np.float32) * 0.25
    y, cb, cr = rgb_to_ycbcr(solid)
    back = ycbcr_to_rgb(y, cb, cr)
    assert back.shape == solid.shape
    # Allow small rounding differences due to uint8 conversions
    assert np.allclose(solid, back, atol=2/255.0)
