from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import yaml
import warnings

from .api import SLPRParams, SLPRSession, ColorMode


@dataclass
class Keyframe:
    # Position in source image as normalized coords in [0,1]
    # mode "center": (u,v) is the crop center
    # mode "topleft": (u,v) is the crop's top-left
    mode: Literal["center", "topleft"] = "center"
    u: float = 0.5
    v: float = 0.5
    zoom: float = 1.0


@dataclass
class Phase:
    # Duration of this phase in seconds. Number of frames rendered for this
    # phase is round(duration_sec * fps).
    duration_sec: float
    start: Keyframe
    end: Keyframe
    easing: Literal["linear"] = "linear"


@dataclass
class Scene:
    input_path: Path
    color_mode: ColorMode
    output_width: int
    output_height: int
    fps: int
    phases: List[Phase]
    border_mode: Literal["black", "white", "mirror", "repeat", "edge"] = (
        "black"
    )
    algorithm: Literal[
        "slpr",
        "cv2_nearest",
        "cv2_linear",
        "cv2_area",
        "cv2_cubic",
        "cv2_lanczos4",
    ] = "slpr"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interp_kf(a: Keyframe, b: Keyframe, t: float) -> Keyframe:
    # Interpolate in the same mode; mode mismatch falls back to end's mode
    mode = b.mode if a.mode != b.mode else a.mode
    return Keyframe(
        mode=mode,
        u=float(_lerp(a.u, b.u, t)),
        v=float(_lerp(a.v, b.v, t)),
        zoom=float(_lerp(a.zoom, b.zoom, t)),
    )


def load_scene(path: Path) -> Scene:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    fps = int(data.get("fps", 30))
    border_mode = data.get("border_mode", "black")
    algorithm = data.get("algorithm", "slpr")
    phases: List[Phase] = []
    for ph in data["phases"]:
        # New schema: { duration_sec, start: {..}, end: {..} }
        if "duration_sec" in ph:
            a = ph.get("start", {})
            b = ph.get("end", {})
            phases.append(
                Phase(
                    duration_sec=float(ph["duration_sec"]),
                    start=Keyframe(
                        mode=a.get("mode", "center"),
                        u=float(a.get("u", 0.5)),
                        v=float(a.get("v", 0.5)),
                        zoom=float(a.get("zoom", 1.0)),
                    ),
                    end=Keyframe(
                        mode=b.get("mode", "center"),
                        u=float(b.get("u", 0.5)),
                        v=float(b.get("v", 0.5)),
                        zoom=float(b.get("zoom", 1.0)),
                    ),
                    easing=ph.get("easing", "linear"),
                )
            )
        else:
            # Backward compatibility: legacy schema with start_frame/end_frame
            # and keyframes: [start, end]. Compute duration from fps.
            kfs = ph["keyframes"]
            a = kfs[0]
            b = kfs[1]
            start_frame = int(ph["start_frame"])  # inclusive
            end_frame = int(ph["end_frame"])  # inclusive
            duration_frames = max(1, end_frame - start_frame + 1)
            duration_sec = duration_frames / float(max(1, fps))
            phases.append(
                Phase(
                    duration_sec=float(duration_sec),
                    start=Keyframe(
                        mode=a.get("mode", "center"),
                        u=float(a.get("u", 0.5)),
                        v=float(a.get("v", 0.5)),
                        zoom=float(a.get("zoom", 1.0)),
                    ),
                    end=Keyframe(
                        mode=b.get("mode", "center"),
                        u=float(b.get("u", 0.5)),
                        v=float(b.get("v", 0.5)),
                        zoom=float(b.get("zoom", 1.0)),
                    ),
                    easing=ph.get("easing", "linear"),
                )
            )
    return Scene(
        input_path=Path(data["input_path"]),
        color_mode=data.get("color_mode", "luma"),
        output_width=int(data["output_width"]),
        output_height=int(data["output_height"]),
        fps=fps,
        border_mode=border_mode,
    algorithm=algorithm,
        phases=phases,
    )


def _compute_crop(
    src_h: int,
    src_w: int,
    out_h: int,
    out_w: int,
    kf: Keyframe,
) -> Tuple[int, int, int, int]:
    """Compute crop rectangle (y0, x0, h, w) for a keyframe.

    We never resize the final output; instead, we choose the reconstruction
    target shape as (out_h * zoom, out_w * zoom), and then center/topleft
    crop back to (out_h, out_w) within the reconstructed image space.

    Here we compute the corresponding crop reference in source coordinates;
    the reconstruct call uses target_shape=(out_h*zoom, out_w*zoom), and
    cropping is applied after reconstruction.
    """
    # Clamp zoom
    z = max(1e-6, float(kf.zoom))
    # Determine the region size in the reconstructed (zoomed) image
    recon_h = int(round(out_h * z))
    recon_w = int(round(out_w * z))
    # Center/topleft selection translated to source coords later by caller
    return 0, 0, recon_h, recon_w


def render_scene_frames(
    image: np.ndarray[Any, Any],
    scene: Scene,
    params: SLPRParams,
    *,
    base_seed: Optional[int] = None,
) -> List[np.ndarray[Any, Any]]:
    """Render all frames described by the scene.

    - No disk I/O here (frames returned as a list of arrays).
    - Always reconstruct at the exact target shape required per frame without
      post-resize of the final (out_h,out_w) crop.
        - For luma, chroma is kept constant and resized to the target shape
            internally.
        - Seeds vary per frame: seed_i = None if base_seed is None else
            base_seed + i
    """
    h, w = image.shape[:2]
    out_h, out_w = scene.output_height, scene.output_width

    # Aspect ratio: crop input to match output aspect (no final resize)
    target_ar = out_w / float(max(1, out_h))
    input_ar = w / float(max(1, h))
    if abs(target_ar - input_ar) > 1e-6:
        warnings.warn(
            f"Input AR {input_ar:.4f} != output AR {target_ar:.4f};"
            " center-cropping input to match output aspect."
        )
        if input_ar > target_ar:
            # Crop width
            new_w = int(round(h * target_ar))
            x0c = max(0, (w - new_w) // 2)
            if image.ndim == 2:
                image = image[:, x0c:x0c + new_w]
            else:
                image = image[:, x0c:x0c + new_w, :]
            w = new_w
        else:
            # Crop height
            new_h = int(round(w / target_ar))
            y0c = max(0, (h - new_h) // 2)
            if image.ndim == 2:
                image = image[y0c:y0c + new_h, :]
            else:
                image = image[y0c:y0c + new_h, :, :]
            h = new_h

    use_slpr = scene.algorithm == "slpr"
    sess = SLPRSession(image, scene.color_mode, params) if use_slpr else None

    # Build timeline frames
    frames: List[np.ndarray[Any, Any]] = []

    # Render sequentially by phase using fps * duration
    fps = max(1, scene.fps)
    frame_idx = 0
    first_phase = True
    for ph in scene.phases:
        nframes = max(1, int(round(ph.duration_sec * fps)))
        # Determine index range; skip first frame for subsequent phases to
        # avoid duplicate boundary frame.
        start_i = 0 if (first_phase or nframes == 1) else 1
        first_phase = False
        for i in range(start_i, nframes):
            if nframes <= 1:
                t = 0.0
            else:
                t = i / float(nframes - 1)
            kf = _interp_kf(ph.start, ph.end, t)

            # Target recon size = out size scaled by zoom
            recon_h = max(1, int(round(scene.output_height * kf.zoom)))
            recon_w = max(1, int(round(scene.output_width * kf.zoom)))
            if use_slpr:
                seed_i = (
                    None if base_seed is None else int(base_seed + frame_idx)
                )
                assert sess is not None
                recon = sess.reconstruct((recon_h, recon_w), seed=seed_i)
            else:
                # OpenCV interpolation path
                import cv2  # type: ignore

                interp_map = {
                    "cv2_nearest": cv2.INTER_NEAREST,
                    "cv2_linear": cv2.INTER_LINEAR,
                    "cv2_area": cv2.INTER_AREA,
                    "cv2_cubic": cv2.INTER_CUBIC,
                    "cv2_lanczos4": cv2.INTER_LANCZOS4,
                }
                interp = interp_map.get(scene.algorithm, cv2.INTER_LINEAR)
                recon = cv2.resize(
                    image,
                    (recon_w, recon_h),
                    interpolation=interp,
                ).astype(np.float32)

            # Crop to (out_h,out_w) from recon using normalized (u,v)
            cx = int(round(kf.u * recon_w))
            cy = int(round(kf.v * recon_h))
            if kf.mode == "center":
                x0 = int(round(cx - out_w // 2))
                y0 = int(round(cy - out_h // 2))
            else:  # topleft
                x0 = int(round(cx))
                y0 = int(round(cy))

            frame = _crop_with_border(
                recon, x0, y0, out_w, out_h, scene.border_mode
            )
            frames.append(frame.astype(np.float32))
            frame_idx += 1

    return frames


def render_scene_to_pngs(
    image: np.ndarray[Any, Any],
    scene: Scene,
    params: SLPRParams,
    out_dir: Path,
    *,
    base_seed: Optional[int] = None,
    workers: Optional[int] = None,
) -> List[Path]:
    """Render frames and write them as PNGs (lossless) in out_dir.

    Returns list of file paths in frame order.
    """
    from .utils import save_image

    # Optionally render in parallel across frames using threads.
    # Build timeline first (same as render_scene_frames but without
    # reconstruct).
    h, w = image.shape[:2]
    out_h, out_w = scene.output_height, scene.output_width

    # Aspect ratio adjust like in render_scene_frames
    target_ar = out_w / float(max(1, out_h))
    input_ar = w / float(max(1, h))
    if abs(target_ar - input_ar) > 1e-6:
        if input_ar > target_ar:
            new_w = int(round(h * target_ar))
            x0c = max(0, (w - new_w) // 2)
            if image.ndim == 2:
                image = image[:, x0c:x0c + new_w]
            else:
                image = image[:, x0c:x0c + new_w, :]
            w = new_w
        else:
            new_h = int(round(w / target_ar))
            y0c = max(0, (h - new_h) // 2)
            if image.ndim == 2:
                image = image[y0c:y0c + new_h, :]
            else:
                image = image[y0c:y0c + new_h, :, :]
            h = new_h

    use_slpr = scene.algorithm == "slpr"
    sess = SLPRSession(image, scene.color_mode, params) if use_slpr else None
    fps = max(1, scene.fps)
    tasks: List[Tuple[int, Keyframe, int, int, int, int]] = []
    # (frame_idx, kf, recon_h, recon_w, x0, y0)
    frame_idx = 0
    first_phase = True
    for ph in scene.phases:
        nframes = max(1, int(round(ph.duration_sec * fps)))
        start_i = 0 if (first_phase or nframes == 1) else 1
        first_phase = False
        for i in range(start_i, nframes):
            t = 0.0 if nframes <= 1 else i / float(nframes - 1)
            kf = _interp_kf(ph.start, ph.end, t)
            recon_h = max(1, int(round(scene.output_height * kf.zoom)))
            recon_w = max(1, int(round(scene.output_width * kf.zoom)))
            cx = int(round(kf.u * recon_w))
            cy = int(round(kf.v * recon_h))
            if kf.mode == "center":
                x0 = int(round(cx - out_w // 2))
                y0 = int(round(cy - out_h // 2))
            else:
                x0 = int(round(cx))
                y0 = int(round(cy))
            tasks.append((frame_idx, kf, recon_h, recon_w, x0, y0))
            frame_idx += 1

    def _render_one(
        task: Tuple[int, Keyframe, int, int, int, int]
    ) -> Tuple[int, np.ndarray[Any, Any]]:
        idx, kf_loc, rh, rw, x0_loc, y0_loc = task
        if use_slpr:
            seed_i = None if base_seed is None else int(base_seed + idx)
            assert sess is not None
            recon = sess.reconstruct((rh, rw), seed=seed_i)
        else:
            import cv2  # type: ignore

            interp_map = {
                "cv2_nearest": cv2.INTER_NEAREST,
                "cv2_linear": cv2.INTER_LINEAR,
                "cv2_area": cv2.INTER_AREA,
                "cv2_cubic": cv2.INTER_CUBIC,
                "cv2_lanczos4": cv2.INTER_LANCZOS4,
            }
            interp = interp_map.get(scene.algorithm, cv2.INTER_LINEAR)
            recon = cv2.resize(
                image,
                (rw, rh),
                interpolation=interp,
            ).astype(np.float32)
        frame_arr = _crop_with_border(
            recon, x0_loc, y0_loc, out_w, out_h, scene.border_mode
        )
        return idx, frame_arr.astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    # Default workers to CPU cores
    if workers is None:
        workers = max(1, os.cpu_count() or 1)
    if workers <= 1:
        results = [_render_one(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_render_one, t) for t in tasks]
            results = [f.result() for f in futs]
    # Write in order
    results.sort(key=lambda x: x[0])
    for i, frame in results:
        p = out_dir / f"frame_{i:06d}.png"
        from .utils import save_image as _save_image
        _save_image(frame, p)
        paths.append(p)
    return paths


def _wrap_indices(n: int, idx: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if n <= 0:
        return np.zeros_like(idx)
    return np.mod(idx, n)


def _edge_indices(n: int, idx: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if n <= 0:
        return np.zeros_like(idx)
    return np.clip(idx, 0, n - 1)


def _reflect_indices(
    n: int, idx: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    if n <= 1:
        return np.zeros_like(idx)
    period = 2 * n - 2
    m = np.mod(idx, period)
    reflect = period - m
    out = np.where(m < n, m, reflect)
    return out


def _crop_with_border(
    img: np.ndarray[Any, Any],
    x0: int,
    y0: int,
    out_w: int,
    out_h: int,
    mode: Literal["black", "white", "mirror", "repeat", "edge"],
) -> np.ndarray[Any, Any]:
    h, w = img.shape[:2]
    if mode in ("black", "white"):
        fill = 0.0 if mode == "black" else 1.0
        if img.ndim == 2:
            out = np.full((out_h, out_w), fill, dtype=img.dtype)
        else:
            out = np.full((out_h, out_w, img.shape[2]), fill, dtype=img.dtype)
        # Intersection region to copy
        x1 = x0 + out_w
        y1 = y0 + out_h
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(w, x1)
        src_y1 = min(h, y1)
        dst_x0 = max(0, -x0)
        dst_y0 = max(0, -y0)
        if src_x1 > src_x0 and src_y1 > src_y0:
            if img.ndim == 2:
                out[
                    dst_y0:dst_y0 + (src_y1 - src_y0),
                    dst_x0:dst_x0 + (src_x1 - src_x0),
                ] = (img[src_y0:src_y1, src_x0:src_x1])
            else:
                out[
                    dst_y0:dst_y0 + (src_y1 - src_y0),
                    dst_x0:dst_x0 + (src_x1 - src_x0),
                    :,
                ] = (img[src_y0:src_y1, src_x0:src_x1, :])
        return out

    # Index-based modes: build index maps
    xs = np.arange(x0, x0 + out_w)
    ys = np.arange(y0, y0 + out_h)
    if mode == "repeat":
        xi = _wrap_indices(w, xs)
        yi = _wrap_indices(h, ys)
    elif mode == "edge":
        xi = _edge_indices(w, xs)
        yi = _edge_indices(h, ys)
    elif mode == "mirror":
        xi = _reflect_indices(w, xs)
        yi = _reflect_indices(h, ys)
    else:
        # Fallback to black
        return _crop_with_border(img, x0, y0, out_w, out_h, "black")

    if img.ndim == 2:
        return img[np.ix_(yi, xi)]
    return img[np.ix_(yi, xi, np.arange(img.shape[2]))]
