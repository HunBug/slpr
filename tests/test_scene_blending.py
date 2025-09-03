from typing import Literal
import numpy as np
from pathlib import Path

from slpr.api import SLPRParams
from slpr.scene import Scene, Source, Phase, Keyframe, render_scene_frames


def make_const_img(h, w, val, channels=3):
    if channels == 1:
        return np.full((h, w), val, dtype=np.float32)
    return np.full((h, w, channels), val, dtype=np.float32)


def simple_scene(blend_mode: Literal["weighted", "random"] = "weighted"):
    sources = [
        Source(name="a", path=Path("/dev/null")),
        Source(name="b", path=Path("/dev/null")),
    ]
    ph = Phase(
        duration_sec=1.0,
        start=Keyframe(
            mode="center", u=0.5, v=0.5, zoom=1.0,
            weights={"a": 1.0, "b": 0.0},
        ),
        end=Keyframe(
            mode="center", u=0.5, v=0.5, zoom=1.0,
            weights={"a": 0.0, "b": 1.0},
        ),
    )
    scene = Scene(
        sources=sources,
        color_mode="rgb",
        output_width=8,
        output_height=8,
        fps=2,
        phases=[ph],
        border_mode="black",
        algorithm="slpr",
        blend_mode=blend_mode,
        roi_min_zoom=4.0,
        roi_pad_scale=2.0,
    )
    return scene


def test_weighted_blend_deterministic_average():
    # Two constant images (0.2 and 0.8).
    a = make_const_img(8, 8, 0.2)
    b = make_const_img(8, 8, 0.8)
    scene = simple_scene(blend_mode="weighted")
    params = SLPRParams(
        levels=1, patch_size=1, jitter=0.0, noise_strength=0.0, samples=1
    )

    frames = render_scene_frames([a, b], scene, params, base_seed=123)
    # fps=2, duration=1.0 => 2 frames. With boundary dedup, expect 1 frame.
    assert len(frames) >= 1
    f = frames[0]
    # With weights at start {1,0}, expect image ~0.2.
    assert np.allclose(f.mean(), 0.2, atol=1e-4)


def test_weighted_blend_midpoint():
    a = make_const_img(8, 8, 0.0)
    b = make_const_img(8, 8, 1.0)
    scene = simple_scene(blend_mode="weighted")
    # Force mid-frame by using fps=3 (indices 0..2, middle has t~0.5)
    scene = scene.__class__(
        sources=scene.sources,
        color_mode=scene.color_mode,
        output_width=scene.output_width,
        output_height=scene.output_height,
        fps=3,
        phases=scene.phases,
        border_mode=scene.border_mode,
        algorithm=scene.algorithm,
        blend_mode=scene.blend_mode,
        roi_min_zoom=scene.roi_min_zoom,
        roi_pad_scale=scene.roi_pad_scale,
    )
    params = SLPRParams(
        levels=1, patch_size=1, jitter=0.0, noise_strength=0.0, samples=1
    )
    frames = render_scene_frames([a, b], scene, params, base_seed=42)
    # mid frame exists when nframes=3 and no dedup within one phase
    assert len(frames) == 3
    mid = frames[1]
    # weights interpolated to ~0.5, expect mean ~0.5
    assert np.allclose(mid.mean(), 0.5, atol=1e-3)


def test_random_blend_is_stochastic_but_seeded():
    a = make_const_img(8, 8, 0.0)
    b = make_const_img(8, 8, 1.0)
    scene = simple_scene(blend_mode="random")
    # Use fps=3 to ensure a middle frame (t~0.5) with non-degenerate weights
    scene = scene.__class__(
        sources=scene.sources,
        color_mode=scene.color_mode,
        output_width=scene.output_width,
        output_height=scene.output_height,
        fps=3,
        phases=scene.phases,
        border_mode=scene.border_mode,
        algorithm=scene.algorithm,
        blend_mode=scene.blend_mode,
        roi_min_zoom=scene.roi_min_zoom,
        roi_pad_scale=scene.roi_pad_scale,
    )
    params = SLPRParams(
        levels=1, patch_size=1, jitter=0.0, noise_strength=0.0, samples=1
    )
    f1 = render_scene_frames([a, b], scene, params, base_seed=100)[1]
    f2 = render_scene_frames([a, b], scene, params, base_seed=100)[1]
    f3 = render_scene_frames([a, b], scene, params, base_seed=101)[1]
    # Same seeds -> identical frames. Different seed -> different frame.
    assert np.allclose(f1, f2)
    assert not np.allclose(f1, f3)


def test_default_weights_fallback_to_first_source():
    # When weights are None or sum to zero, first source should dominate.
    a = make_const_img(8, 8, 0.25)
    b = make_const_img(8, 8, 0.75)
    # Scene with no weights specified on keyframes
    sources = [
        Source(name="a", path=Path("/dev/null")),
        Source(name="b", path=Path("/dev/null")),
    ]
    ph = Phase(
        duration_sec=1.0,
        start=Keyframe(mode="center", u=0.5, v=0.5, zoom=1.0, weights=None),
        end=Keyframe(mode="center", u=0.5, v=0.5, zoom=1.0, weights=None),
    )
    scene = Scene(
        sources=sources,
        color_mode="rgb",
        output_width=8,
        output_height=8,
        fps=1,
        phases=[ph],
        border_mode="black",
        algorithm="slpr",
        blend_mode="weighted",
        roi_min_zoom=4.0,
        roi_pad_scale=2.0,
    )
    params = SLPRParams(
        levels=1, patch_size=1, jitter=0.0, noise_strength=0.0, samples=1
    )
    f = render_scene_frames([a, b], scene, params, base_seed=7)[0]
    # Expect frame equals first image value everywhere.
    assert np.allclose(f, a, atol=1e-6)


def test_weights_are_normalized_for_weighted_blend():
    a = make_const_img(8, 8, 0.0)
    b = make_const_img(8, 8, 1.0)
    sources = [
        Source(name="a", path=Path("/dev/null")),
        Source(name="b", path=Path("/dev/null")),
    ]
    # Use explicit non-normalized weights 2:1 at both ends
    ph = Phase(
        duration_sec=1.0,
        start=Keyframe(
            mode="center", u=0.5, v=0.5, zoom=1.0, weights={"a": 2.0, "b": 1.0}
        ),
        end=Keyframe(
            mode="center", u=0.5, v=0.5, zoom=1.0, weights={"a": 2.0, "b": 1.0}
        ),
    )
    scene = Scene(
        sources=sources,
        color_mode="rgb",
        output_width=8,
        output_height=8,
        fps=1,
        phases=[ph],
        border_mode="black",
        algorithm="slpr",
        blend_mode="weighted",
        roi_min_zoom=4.0,
        roi_pad_scale=2.0,
    )
    params = SLPRParams(
        levels=1, patch_size=1, jitter=0.0, noise_strength=0.0, samples=1
    )
    f = render_scene_frames([a, b], scene, params, base_seed=0)[0]
    # Expect 2/3 mean (0*2/3 + 1*1/3 = 1/3).
    # With a=0 and b=1, weights {2,1} normalized to {2/3,1/3} -> mean 1/3.
    assert np.allclose(f.mean(), 1.0/3.0, atol=1e-4)


def test_border_black_fill():
    img = make_const_img(8, 8, 0.5)
    # Place crop outside the reconstructed image to trigger border fill.
    sources = [Source(name="a", path=Path("/dev/null"))]
    ph = Phase(
        duration_sec=1.0,
        start=Keyframe(mode="topleft", u=2.0, v=2.0, zoom=1.0, weights=None),
        end=Keyframe(mode="topleft", u=2.0, v=2.0, zoom=1.0, weights=None),
    )
    scene = Scene(
        sources=sources,
        color_mode="rgb",
        output_width=8,
        output_height=8,
        fps=1,
        phases=[ph],
        border_mode="black",
        algorithm="cv2_nearest",
        blend_mode="weighted",
        roi_min_zoom=4.0,
        roi_pad_scale=2.0,
    )
    params = SLPRParams(
        levels=1, patch_size=1, jitter=0.0, noise_strength=0.0, samples=1
    )
    f = render_scene_frames([img], scene, params, base_seed=0)[0]
    assert f.shape == (8, 8, 3)
    assert np.allclose(f, 0.0, atol=1e-6)


def test_border_white_fill():
    img = make_const_img(8, 8, 0.5)
    sources = [Source(name="a", path=Path("/dev/null"))]
    ph = Phase(
        duration_sec=1.0,
        start=Keyframe(mode="topleft", u=2.0, v=2.0, zoom=1.0, weights=None),
        end=Keyframe(mode="topleft", u=2.0, v=2.0, zoom=1.0, weights=None),
    )
    scene = Scene(
        sources=sources,
        color_mode="rgb",
        output_width=8,
        output_height=8,
        fps=1,
        phases=[ph],
        border_mode="white",
        algorithm="cv2_nearest",
        blend_mode="weighted",
        roi_min_zoom=4.0,
        roi_pad_scale=2.0,
    )
    params = SLPRParams(
        levels=1, patch_size=1, jitter=0.0, noise_strength=0.0, samples=1
    )
    f = render_scene_frames([img], scene, params, base_seed=0)[0]
    assert f.shape == (8, 8, 3)
    assert np.allclose(f, 1.0, atol=1e-6)


def test_roi_vs_full_equivalence_nearest():
    # ROI path (render_scene_to_pngs with zoom >= roi_min_zoom) should match
    # full-frame path (render_scene_frames) for cv2_nearest.
    import tempfile
    from PIL import Image  # type: ignore

    # Use a solid color so any minor mapping differences don't affect result.
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[:, :, 0] = 0.2
    img[:, :, 1] = 0.5
    img[:, :, 2] = 0.8
    sources = [Source(name="a", path=Path("/dev/null"))]
    # Large zoom to trigger ROI (default roi_min_zoom=4.0)
    ph = Phase(
        duration_sec=1.0,
        start=Keyframe(mode="center", u=0.5, v=0.5, zoom=8.0, weights=None),
        end=Keyframe(mode="center", u=0.5, v=0.5, zoom=8.0, weights=None),
    )
    scene = Scene(
        sources=sources,
        color_mode="rgb",
        output_width=16,
        output_height=16,
        fps=1,
        phases=[ph],
        border_mode="black",
        algorithm="cv2_nearest",
        blend_mode="weighted",
        roi_min_zoom=4.0,
        roi_pad_scale=2.0,
    )
    params = SLPRParams(
        levels=1, patch_size=1, jitter=0.0, noise_strength=0.0, samples=1
    )
    # Full path
    f_full = render_scene_frames([img], scene, params, base_seed=0)[0]
    # ROI path writes PNGs; read back the PNG and compare.
    with tempfile.TemporaryDirectory() as td:
        out_paths = __import__("slpr.scene", fromlist=["render_scene_to_pngs"]).render_scene_to_pngs(  # type: ignore  # noqa: E501
            [img], scene, params, Path(td), base_seed=0, workers=1
        )
        assert len(out_paths) == 1
        # Wait briefly for file to be visible on FS (should be immediate).
        import time
        p = out_paths[0]
        for _ in range(50):
            if p.exists():
                break
            time.sleep(0.01)
        assert p.exists()
        img_pil = Image.open(p).convert("RGB")
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    assert f_full.shape == arr.shape
    assert np.allclose(f_full, arr, atol=2/255.0)


def test_phase_boundary_deduplication():
    # Two 1s phases at fps=2 should yield 3 total frames (skip boundary dup).
    a = make_const_img(8, 8, 0.0)
    sources = [Source(name="a", path=Path("/dev/null"))]
    ph1 = Phase(
        duration_sec=1.0,
        start=Keyframe(mode="center", u=0.5, v=0.5, zoom=1.0),
        end=Keyframe(mode="center", u=0.5, v=0.5, zoom=1.0),
    )
    ph2 = Phase(
        duration_sec=1.0,
        start=Keyframe(mode="center", u=0.5, v=0.5, zoom=1.0),
        end=Keyframe(mode="center", u=0.5, v=0.5, zoom=1.0),
    )
    scene = Scene(
        sources=sources,
        color_mode="rgb",
        output_width=8,
        output_height=8,
        fps=2,
        phases=[ph1, ph2],
        border_mode="black",
        algorithm="cv2_nearest",
        blend_mode="weighted",
        roi_min_zoom=4.0,
        roi_pad_scale=2.0,
    )
    params = SLPRParams(
        levels=1, patch_size=1, jitter=0.0, noise_strength=0.0, samples=1
    )
    frames = render_scene_frames([a], scene, params, base_seed=0)
    assert len(frames) == 3
