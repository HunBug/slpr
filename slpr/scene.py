from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import numpy as np
import yaml

from .api import SLPRParams, SLPRSession, ColorMode


@dataclass
class Source:
    name: str
    path: Path
    kind: Literal["image"] = "image"


@dataclass
class Keyframe:
    # Position in source image as normalized coords in [0,1]
    # mode "center": (u,v) is the crop center
    # mode "topleft": (u,v) is the crop's top-left
    mode: Literal["center", "topleft"] = "center"
    u: float = 0.5
    v: float = 0.5
    zoom: float = 1.0
    # Optional per-source weights mapping by source name
    weights: Optional[Dict[str, float]] = None


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
    sources: List[Source]
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
    blend_mode: Literal["weighted", "random"] = "weighted"
    # ROI optimization for large zooms: if zoom >= roi_min_zoom, crop source
    # region and process that only. pad factor applies to width/height.
    roi_min_zoom: float = 4.0
    roi_pad_scale: float = 2.0


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interp_kf(a: Keyframe, b: Keyframe, t: float) -> Keyframe:
    # Interpolate in the same mode; mode mismatch falls back to end's mode
    mode = b.mode if a.mode != b.mode else a.mode
    # Interpolate weights if present

    def _interp_weights(
        wa: Optional[Dict[str, float]],
        wb: Optional[Dict[str, float]],
        tloc: float,
    ) -> Optional[Dict[str, float]]:
        if wa is None and wb is None:
            return None
        wa = wa or {}
        wb = wb or {}
        keys = set(wa.keys()) | set(wb.keys())
        out: Dict[str, float] = {}
        for k in keys:
            va = float(wa.get(k, 0.0))
            vb = float(wb.get(k, 0.0))
            out[k] = _lerp(va, vb, tloc)
        return out

    return Keyframe(
        mode=mode,
        u=float(_lerp(a.u, b.u, t)),
        v=float(_lerp(a.v, b.v, t)),
        zoom=float(_lerp(a.zoom, b.zoom, t)),
        weights=_interp_weights(a.weights, b.weights, t),
    )


def load_scene(path: Path) -> Scene:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    fps = int(data.get("fps", 30))
    border_mode = data.get("border_mode", "black")
    algorithm = data.get("algorithm", "slpr")
    blend_mode = data.get("blend_mode", "weighted")
    roi_min_zoom = float(data.get("roi_min_zoom", 4.0))
    roi_pad_scale = float(data.get("roi_pad_scale", 2.0))
    # Inputs: support new schema `inputs: [{name, path}]` and legacy
    # `input_path`
    sources: List[Source] = []
    if "inputs" in data:
        for idx, src in enumerate(data["inputs"]):
            name = str(src.get("name") or f"src{idx}")
            sources.append(Source(name=name, path=Path(src["path"])))
    elif "input_path" in data:
        sources.append(
            Source(name="main", path=Path(data["input_path"]))
        )
    else:
        raise ValueError(
            "Scene YAML must include either 'inputs' or 'input_path'."
        )
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
                        weights=a.get("weights"),
                    ),
                    end=Keyframe(
                        mode=b.get("mode", "center"),
                        u=float(b.get("u", 0.5)),
                        v=float(b.get("v", 0.5)),
                        zoom=float(b.get("zoom", 1.0)),
                        weights=b.get("weights"),
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
                        weights=a.get("weights"),
                    ),
                    end=Keyframe(
                        mode=b.get("mode", "center"),
                        u=float(b.get("u", 0.5)),
                        v=float(b.get("v", 0.5)),
                        zoom=float(b.get("zoom", 1.0)),
                        weights=b.get("weights"),
                    ),
                    easing=ph.get("easing", "linear"),
                )
            )
    return Scene(
        sources=sources,
        color_mode=data.get("color_mode", "luma"),
        output_width=int(data["output_width"]),
        output_height=int(data["output_height"]),
        fps=fps,
        phases=phases,
        border_mode=border_mode,
        algorithm=algorithm,
        blend_mode=blend_mode,
        roi_min_zoom=roi_min_zoom,
        roi_pad_scale=roi_pad_scale,
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
    images: List[np.ndarray[Any, Any]],
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
    if len(images) == 0:
        return []
    # Prepare per-source images to match output aspect ratio
    prepped: List[np.ndarray[Any, Any]] = []
    for img in images:
        prepped.append(
            _match_output_aspect(
                img, scene.output_height, scene.output_width
            )
        )
    h, w = prepped[0].shape[:2]
    out_h, out_w = scene.output_height, scene.output_width

    # Aspect ratio already matched per source via _match_output_aspect

    use_slpr = scene.algorithm == "slpr"
    # One session per source
    sessions: Optional[List[SLPRSession]] = None
    if use_slpr:
        sessions = [
            SLPRSession(img, scene.color_mode, params) for img in prepped
        ]

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
            # Per-source reconstructions
            seed_i = None if base_seed is None else int(base_seed + frame_idx)
            frames_src: List[np.ndarray[Any, Any]] = []
            if use_slpr:
                assert sessions is not None
                for sess in sessions:
                    frames_src.append(
                        sess.reconstruct((recon_h, recon_w), seed=seed_i)
                    )
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
                for img in prepped:
                    frames_src.append(
                        cv2.resize(
                            img, (recon_w, recon_h), interpolation=interp
                        ).astype(np.float32)
                    )

            # Compute per-source weights
            wvec = _weights_for_sources(
                kf.weights, [s.name for s in scene.sources]
            )
            recon = _blend_frames(frames_src, wvec, scene.blend_mode, seed_i)

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
    images: List[np.ndarray[Any, Any]],
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
    from tqdm import tqdm

    # Optionally render in parallel across frames using threads.
    # Build timeline first.
    if len(images) == 0:
        return []
    # Prepare images per source to match output aspect ratio
    prepped: List[np.ndarray[Any, Any]] = [
        _match_output_aspect(img, scene.output_height, scene.output_width)
        for img in images
    ]
    h, w = prepped[0].shape[:2]
    out_h, out_w = scene.output_height, scene.output_width

    use_slpr = scene.algorithm == "slpr"
    base_sess_list: Optional[List[SLPRSession]] = None
    if use_slpr:
        base_sess_list = [
            SLPRSession(img, scene.color_mode, params) for img in prepped
        ]
    fps = max(1, scene.fps)
    tasks: List[Tuple[int, Keyframe, int, int, int, int, float]] = []
    # (frame_idx, kf, recon_h, recon_w, x0, y0, zoom)
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
            tasks.append((frame_idx, kf, recon_h, recon_w, x0, y0, kf.zoom))
            frame_idx += 1

    def _render_one(
        task: Tuple[int, Keyframe, int, int, int, int, float]
    ) -> Tuple[int, np.ndarray[Any, Any]]:
        idx, kf_loc, rh, rw, x0_loc, y0_loc, zoom_loc = task
        # ROI optimization
        use_roi = zoom_loc >= scene.roi_min_zoom
        frames_src: List[np.ndarray[Any, Any]] = []
        seed_i = None if base_seed is None else int(base_seed + idx)
        if use_roi:
            # Compute ROI per source
            rois: List[Tuple[int, int, int, int, np.ndarray[Any, Any]]] = []
            for img in prepped:
                rois.append(
                    _compute_roi_and_extract(
                        img,
                        (rw, rh),
                        (x0_loc, y0_loc),
                        (out_w, out_h),
                        scene.roi_pad_scale,
                    )
                )
            # Note: rrw/rrh/rx0/ry0 all computed from first ROI
            # (assume identical mapping across sources after AR crop)
            rrw, rrh, rx0, ry0, _ = rois[0]
            if use_slpr:
                for _, _, _, _, roi_img in rois:
                    sess_local = SLPRSession(
                        roi_img, scene.color_mode, params
                    )
                    frames_src.append(
                        sess_local.reconstruct((rrh, rrw), seed=seed_i)
                    )
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
                for _, _, _, _, roi_img in rois:
                    frames_src.append(
                        cv2.resize(
                            roi_img, (rrw, rrh), interpolation=interp
                        ).astype(np.float32)
                    )
        else:
            rrw, rrh, rx0, ry0 = rw, rh, x0_loc, y0_loc
            if use_slpr:
                assert base_sess_list is not None
                for sess in base_sess_list:
                    frames_src.append(
                        sess.reconstruct((rrh, rrw), seed=seed_i)
                    )
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
                for img in prepped:
                    frames_src.append(
                        cv2.resize(
                            img, (rrw, rrh), interpolation=interp
                        ).astype(np.float32)
                    )

        wvec = _weights_for_sources(
            kf_loc.weights, [s.name for s in scene.sources]
        )
        recon = _blend_frames(frames_src, wvec, scene.blend_mode, seed_i)
        frame_arr = _crop_with_border(
            recon, rx0, ry0, out_w, out_h, scene.border_mode
        )
        return idx, frame_arr.astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    # Default workers to CPU cores
    if workers is None:
        workers = max(1, os.cpu_count() or 1)
    total = len(tasks)
    if workers <= 1:
        for task in tqdm(tasks, desc="frames", total=total):
            i, frame = _render_one(task)
            p = out_dir / f"frame_{i:06d}.png"
            from .utils import save_image as _save_image
            _save_image(frame, p)
            paths.append(p)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_render_one, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=total, desc="frames"):
                i, frame = fut.result()
                p = out_dir / f"frame_{i:06d}.png"
                from .utils import save_image as _save_image
                _save_image(frame, p)
                paths.append(p)
        # Ensure paths ordered by index
        paths.sort()
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


def _compute_roi_and_extract(
    image: np.ndarray[Any, Any],
    recon_shape: Tuple[int, int],
    crop_xy: Tuple[int, int],
    out_wh: Tuple[int, int],
    pad_scale: float,
) -> Tuple[int, int, int, int, np.ndarray[Any, Any]]:
    """Compute a source ROI for a given recon crop and extract it.

    Returns (roi_recon_w, roi_recon_h, rx0, ry0, roi_img)
    where (rx0,ry0) is the crop's top-left in the roi's recon space.
    """
    h, w = image.shape[:2]
    rh, rw = recon_shape
    x0, y0 = crop_xy
    out_w, out_h = out_wh
    # Scale factors from source->recon
    sx = (rw - 1) / float(max(1, w - 1))
    sy = (rh - 1) / float(max(1, h - 1))
    # Map recon crop to source coords
    src_x0 = x0 / max(1e-6, sx)
    src_y0 = y0 / max(1e-6, sy)
    src_x1 = (x0 + out_w) / max(1e-6, sx)
    src_y1 = (y0 + out_h) / max(1e-6, sy)
    # Expand by pad_scale in width/height around center
    cx = 0.5 * (src_x0 + src_x1)
    cy = 0.5 * (src_y0 + src_y1)
    half_w = 0.5 * (src_x1 - src_x0) * pad_scale
    half_h = 0.5 * (src_y1 - src_y0) * pad_scale
    rx0s = int(np.floor(cx - half_w))
    ry0s = int(np.floor(cy - half_h))
    rx1s = int(np.ceil(cx + half_w))
    ry1s = int(np.ceil(cy + half_h))
    rx0s = max(0, rx0s)
    ry0s = max(0, ry0s)
    rx1s = min(w, rx1s)
    ry1s = min(h, ry1s)
    sw = max(1, rx1s - rx0s)
    sh = max(1, ry1s - ry0s)
    # Extract ROI
    if image.ndim == 2:
        roi = image[ry0s:ry1s, rx0s:rx1s]
    else:
        roi = image[ry0s:ry1s, rx0s:rx1s, :]
    # ROI recon size
    rrw = max(1, int(round(sw * sx)))
    rrh = max(1, int(round(sh * sy)))
    # Crop position in ROI recon space
    rx0 = int(round(x0 - rx0s * sx))
    ry0 = int(round(y0 - ry0s * sy))
    return rrw, rrh, rx0, ry0, roi


def _match_output_aspect(
    image: np.ndarray[Any, Any], out_h: int, out_w: int
) -> np.ndarray[Any, Any]:
    """Center-crop image to match desired output aspect ratio."""
    h, w = image.shape[:2]
    target_ar = out_w / float(max(1, out_h))
    input_ar = w / float(max(1, h))
    if abs(target_ar - input_ar) <= 1e-6:
        return image
    if input_ar > target_ar:
        # Crop width
        new_w = int(round(h * target_ar))
        x0c = max(0, (w - new_w) // 2)
        if image.ndim == 2:
            return image[:, x0c:x0c + new_w]
        return image[:, x0c:x0c + new_w, :]
    else:
        # Crop height
        new_h = int(round(w / target_ar))
        y0c = max(0, (h - new_h) // 2)
        if image.ndim == 2:
            return image[y0c:y0c + new_h, :]
        return image[y0c:y0c + new_h, :, :]


def _weights_for_sources(
    weights: Optional[Dict[str, float]], source_names: List[str]
) -> np.ndarray[Any, Any]:
    """Return a normalized weight vector aligned with source_names.

    If weights is None or sums to 0, default to [1,0,0,...] (favor first).
    """
    w = np.zeros((len(source_names),), dtype=np.float32)
    if weights:
        for i, name in enumerate(source_names):
            w[i] = float(weights.get(name, 0.0))
    s = float(w.sum())
    if s <= 0:
        if len(source_names) > 0:
            w[0] = 1.0
            s = 1.0
    return (w / s).astype(np.float32)


def _blend_frames(
    frames: List[np.ndarray[Any, Any]],
    weights: np.ndarray[Any, Any],
    mode: Literal["weighted", "random"],
    seed: Optional[int],
) -> np.ndarray[Any, Any]:
    if len(frames) == 1:
        return frames[0]
    # Ensure shapes match
    h, w = frames[0].shape[:2]
    for f in frames[1:]:
        assert f.shape[:2] == (h, w)
    if mode == "weighted":
        acc = np.zeros_like(frames[0], dtype=np.float32)
        for fi, fw in zip(frames, weights):
            if fw <= 0:
                continue
            acc = acc + (fi.astype(np.float32) * float(fw))
        return acc
    # random categorical selection per pixel
    rng = np.random.default_rng(None if seed is None else int(seed))
    # Stack frames for gather
    if frames[0].ndim == 2:
        stack = np.stack(frames, axis=0)  # (N,H,W)
        idx = rng.choice(len(frames), size=(h, w), p=weights.tolist())
        out = stack[idx, np.arange(h)[:, None], np.arange(w)[None, :]]
        return out.astype(np.float32)
    else:
        stack = np.stack(frames, axis=0)  # (N,H,W,C)
        idx = rng.choice(len(frames), size=(h, w), p=weights.tolist())
        out = stack[idx, np.arange(h)[:, None], np.arange(w)[None, :], :]
        return out.astype(np.float32)
