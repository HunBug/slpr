from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import imageio.v2 as iio

from slpr.api import SLPRParams
from slpr.scene import load_scene, render_scene_to_pngs
from slpr.utils import load_image, now_stamp


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render a scene YAML to PNG frames and assemble GIF/MP4."
        )
    )
    p.add_argument(
        "--scene", required=True, type=Path, help="Path to scene YAML"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (defaults to outputs/<stamp>/<scene_stem>/)"
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Base seed (optional)"
    )
    # SLPR params
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--patch-size", type=int, default=3)
    p.add_argument("--jitter", type=float, default=1.0)
    p.add_argument("--noise-strength", type=float, default=0.1)
    p.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of stochastic samples to average per frame",
    )
    # Outputs
    p.add_argument("--no-gif", action="store_true", help="Skip GIF output")
    p.add_argument("--no-mp4", action="store_true", help="Skip MP4 output")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel threads for frame rendering (default: CPU cores)",
    )
    p.add_argument(
        "--algorithm",
        choices=[
            "slpr",
            "cv2_nearest",
            "cv2_linear",
            "cv2_area",
            "cv2_cubic",
            "cv2_lanczos4",
        ],
        help=(
            "Override scene algorithm: default is scene or 'slpr'. "
            "Use cv2_* to compare against OpenCV interpolation."
        ),
    )
    return p.parse_args()


def assemble_gif(frame_paths: list[Path], out_path: Path, fps: int) -> None:
    frames = [iio.imread(str(p)) for p in frame_paths]
    duration = 1.0 / max(1, fps)
    # Stubs in imageio can be strict; cast to relax typing for pyright.
    iio.mimsave(
        str(out_path), cast(list[Any], frames), duration=duration, loop=0
    )


def assemble_mp4(frame_paths: list[Path], out_path: Path, fps: int) -> None:
    # Use yuv420p for broad compatibility
    with iio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    ) as w:
        for p in frame_paths:
            w.append_data(iio.imread(str(p)))  # type: ignore[attr-defined]


def main() -> None:
    args = build_args()
    scene = load_scene(args.scene)

    # Load input image according to scene color_mode
    image = load_image(scene.input_path, mode=scene.color_mode)

    params = SLPRParams(
        levels=args.levels,
        patch_size=args.patch_size,
        jitter=args.jitter,
        noise_strength=args.noise_strength,
        samples=args.samples,
    )

    stamp = now_stamp()
    out_dir: Path = (
        args.out_dir
        if args.out_dir is not None
        else Path("outputs") / stamp / scene.input_path.stem / args.scene.stem
    )

    frames_dir = out_dir / "frames"
    # Allow overriding algorithm via CLI
    if args.algorithm:
        from dataclasses import replace

        scene = replace(scene, algorithm=args.algorithm)
    frames = render_scene_to_pngs(
        image,
        scene,
        params,
        frames_dir,
        base_seed=args.seed,
        workers=args.workers,
    )

    # Assemble outputs
    if not args.no_gif:
        gif_path = out_dir / "animation.gif"
        assemble_gif(frames, gif_path, scene.fps)
        print(f"GIF written: {gif_path}")
    if not args.no_mp4:
        mp4_path = out_dir / "animation.mp4"
        assemble_mp4(frames, mp4_path, scene.fps)
        print(f"MP4 written: {mp4_path}")

    print(f"Frames written: {len(frames)} -> {frames_dir}")


if __name__ == "__main__":
    main()
