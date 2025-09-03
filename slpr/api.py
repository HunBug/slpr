from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Tuple

import numpy as np

from .pyramid import build_laplacian_pyramid
from .sampling import stochastic_reconstruct, stochastic_average
from .utils import rgb_to_ycbcr, ycbcr_to_rgb


ColorMode = Literal["gray", "rgb", "luma"]


@dataclass(frozen=True)
class SLPRParams:
    levels: int = 5
    patch_size: int = 3
    jitter: float = 1.0
    noise_strength: float = 0.1
    samples: int = 1  # 1 => single stochastic frame


class SLPRSession:
    """Stateful helper that reuses per-channel pyramids.

    Usage:
        sess = SLPRSession(image, color_mode, params)
        frame = sess.reconstruct(target_shape, seed)
    """

    def __init__(
        self,
        image: np.ndarray[Any, Any],
        color_mode: ColorMode,
        params: SLPRParams,
    ) -> None:
        self.color_mode = color_mode
        self.params = params
        self._init_channels(image)

    def _init_channels(self, image: np.ndarray[Any, Any]) -> None:
        if self.color_mode == "gray":
            if image.ndim != 2:
                raise ValueError("gray mode expects 2D array")
            self._channels = [image.astype(np.float32)]
            self._cb = None
            self._cr = None
        elif self.color_mode == "rgb":
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("rgb mode expects (H,W,3)")
            img = image.astype(np.float32)
            self._channels = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]
            self._cb = None
            self._cr = None
        elif self.color_mode == "luma":
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("luma mode expects (H,W,3) input")
            y, cb, cr = rgb_to_ycbcr(image.astype(np.float32))
            self._channels = [y]
            self._cb = cb
            self._cr = cr
        else:
            raise ValueError(f"Unsupported color mode: {self.color_mode}")

        # Build per-channel pyramids once
        self._pyramids: list[
            tuple[list[np.ndarray[Any, Any]], np.ndarray[Any, Any]]
        ] = []
        for ch in self._channels:
            laps, base, _ = build_laplacian_pyramid(ch, self.params.levels)
            self._pyramids.append((laps, base))

        # Original shape for convenience
        h = int(self._channels[0].shape[0])
        w = int(self._channels[0].shape[1])
        self.base_shape = (h, w)

    def reconstruct(
        self, target_shape: Tuple[int, int], seed: int | None = None
    ) -> np.ndarray[Any, Any]:
        """Reconstruct one frame at target_shape.

        Returns float32 array in [0,1], 2D for gray and 3D (H,W,3) otherwise.
        """
        ps = self.params
        if self.color_mode == "gray":
            laps, base = self._pyramids[0]
            if ps.samples == 1:
                y = stochastic_reconstruct(
                    laps,
                    base,
                    target_shape,
                    ps.patch_size,
                    ps.jitter,
                    ps.noise_strength,
                    seed=seed,
                )
            else:
                y = stochastic_average(
                    laps,
                    base,
                    target_shape,
                    ps.patch_size,
                    ps.jitter,
                    ps.noise_strength,
                    ps.samples,
                    seed=seed,
                )
            return y

        if self.color_mode == "rgb":
            outs = []
            for idx in range(3):
                laps, base = self._pyramids[idx]
                if ps.samples == 1:
                    ch = stochastic_reconstruct(
                        laps,
                        base,
                        target_shape,
                        ps.patch_size,
                        ps.jitter,
                        ps.noise_strength,
                        seed=seed,
                    )
                else:
                    ch = stochastic_average(
                        laps,
                        base,
                        target_shape,
                        ps.patch_size,
                        ps.jitter,
                        ps.noise_strength,
                        ps.samples,
                        seed=seed,
                    )
                outs.append(ch)
            return np.stack(outs, axis=2)

        # luma
        laps, base = self._pyramids[0]
        if ps.samples == 1:
            y = stochastic_reconstruct(
                laps,
                base,
                target_shape,
                ps.patch_size,
                ps.jitter,
                ps.noise_strength,
                seed=seed,
            )
        else:
            y = stochastic_average(
                laps,
                base,
                target_shape,
                ps.patch_size,
                ps.jitter,
                ps.noise_strength,
                ps.samples,
                seed=seed,
            )
        assert self._cb is not None and self._cr is not None
        th, tw = target_shape
        cb, cr = self._cb, self._cr
        if cb.shape != y.shape:
            try:
                import cv2  # type: ignore

                cb = cv2.resize(cb, (tw, th), interpolation=cv2.INTER_LINEAR)
                cr = cv2.resize(cr, (tw, th), interpolation=cv2.INTER_LINEAR)
            except Exception:
                from PIL import Image

                resample = 2  # bilinear
                cbi = Image.fromarray(
                    (np.clip(cb, 0.0, 1.0) * 255).astype(np.uint8)
                )
                cri = Image.fromarray(
                    (np.clip(cr, 0.0, 1.0) * 255).astype(np.uint8)
                )
                cb = (
                    np.asarray(cbi.resize((tw, th), resample))
                    .astype(np.float32)
                    / 255.0
                )
                cr = (
                    np.asarray(cri.resize((tw, th), resample))
                    .astype(np.float32)
                    / 255.0
                )
        return ycbcr_to_rgb(y, cb, cr)

    def reconstruct_with_params(
        self,
        target_shape: Tuple[int, int],
        params: SLPRParams,
        seed: int | None = None,
    ) -> np.ndarray[Any, Any]:
        """Reconstruct using provided params, reusing prebuilt pyramids.

        Note: The number of levels must match the levels used to build
        this session's pyramids; otherwise, create a new session.
        """
        if params.levels != self.params.levels:
            raise ValueError(
                "Params.levels mismatch; build a new SLPRSession "
                "for this levels"
            )
        # Temporarily use the provided params for reconstruction
        # Implementation mirrors reconstruct() but uses 'params' instead.
        ps = params
        if self.color_mode == "gray":
            laps, base = self._pyramids[0]
            if ps.samples == 1:
                y = stochastic_reconstruct(
                    laps,
                    base,
                    target_shape,
                    ps.patch_size,
                    ps.jitter,
                    ps.noise_strength,
                    seed=seed,
                )
            else:
                y = stochastic_average(
                    laps,
                    base,
                    target_shape,
                    ps.patch_size,
                    ps.jitter,
                    ps.noise_strength,
                    ps.samples,
                    seed=seed,
                )
            return y

        if self.color_mode == "rgb":
            outs = []
            for idx in range(3):
                laps, base = self._pyramids[idx]
                if ps.samples == 1:
                    ch = stochastic_reconstruct(
                        laps,
                        base,
                        target_shape,
                        ps.patch_size,
                        ps.jitter,
                        ps.noise_strength,
                        seed=seed,
                    )
                else:
                    ch = stochastic_average(
                        laps,
                        base,
                        target_shape,
                        ps.patch_size,
                        ps.jitter,
                        ps.noise_strength,
                        ps.samples,
                        seed=seed,
                    )
                outs.append(ch)
            return np.stack(outs, axis=2)

        # luma
        laps, base = self._pyramids[0]
        if ps.samples == 1:
            y = stochastic_reconstruct(
                laps,
                base,
                target_shape,
                ps.patch_size,
                ps.jitter,
                ps.noise_strength,
                seed=seed,
            )
        else:
            y = stochastic_average(
                laps,
                base,
                target_shape,
                ps.patch_size,
                ps.jitter,
                ps.noise_strength,
                ps.samples,
                seed=seed,
            )
        assert self._cb is not None and self._cr is not None
        th, tw = target_shape
        cb, cr = self._cb, self._cr
        if cb.shape != y.shape:
            try:
                import cv2  # type: ignore

                cb = cv2.resize(cb, (tw, th), interpolation=cv2.INTER_LINEAR)
                cr = cv2.resize(cr, (tw, th), interpolation=cv2.INTER_LINEAR)
            except Exception:
                from PIL import Image

                resample = 2  # bilinear
                cbi = Image.fromarray(
                    (np.clip(cb, 0.0, 1.0) * 255).astype(np.uint8)
                )
                cri = Image.fromarray(
                    (np.clip(cr, 0.0, 1.0) * 255).astype(np.uint8)
                )
                cb = (
                    np.asarray(cbi.resize((tw, th), resample))
                    .astype(np.float32)
                    / 255.0
                )
                cr = (
                    np.asarray(cri.resize((tw, th), resample))
                    .astype(np.float32)
                    / 255.0
                )
        return ycbcr_to_rgb(y, cb, cr)


def reconstruct_image(
    image: np.ndarray[Any, Any],
    color_mode: ColorMode,
    target_shape: Tuple[int, int],
    *,
    levels: int = 5,
    patch_size: int = 3,
    jitter: float = 1.0,
    noise_strength: float = 0.1,
    samples: int = 1,
    seed: int | None = None,
) -> np.ndarray[Any, Any]:
    """Stateless convenience API: reconstruct at target_shape.

    Builds pyramids internally (no reuse across calls). For many frames, prefer
    SLPRSession to reuse pyramids and gain performance.
    """
    params = SLPRParams(
        levels=levels,
        patch_size=patch_size,
        jitter=jitter,
        noise_strength=noise_strength,
        samples=samples,
    )
    sess = SLPRSession(image, color_mode, params)
    return sess.reconstruct(target_shape, seed=seed)
