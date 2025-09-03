# Project summary

Stochastic Laplacian Pyramid Renderer (SLPR) in Python 3.11. This document explains what’s implemented, how it’s structured, how to run it, and where we’re going next.

## What’s implemented
- Core algorithms
  - Gaussian/Laplacian pyramid construction and visualization.
  - Stochastic reconstruction and multi-sample averaging; vectorized fast path.
  - Color modes: gray, RGB (per-channel), luma (Y with original chroma).
- Batch PoC
  - Config-driven runs over multiple assets and algorithm configs via `main.py`.
  - Parallel execution with joblib; deterministic with fixed seeds.
- Scene + CLI
  - Scene YAML with normalized (u,v) coordinates and duration-based phases.
  - Border modes beyond bounds: black, white, edge, repeat, mirror.
  - Algorithm selector: SLPR vs OpenCV (nearest/linear/area/cubic/lanczos4).
  - Streaming frame writes with an overall progress bar; default workers = CPU cores.
  - ROI optimization for large zooms to process only the necessary region.
  - Multi-source inputs with blend modes: weighted and random categorical; per-keyframe weights interpolation.
  - SLPR parameter controls:
    - Scene-level defaults via `slpr_params` (or flat keys) for: levels, patch_size, jitter, noise_strength, samples
    - Per-keyframe `params` with linear interpolation across frames (numbers only)
    - Precedence: per-keyframe > CLI > scene-level defaults > library defaults
- Progress & telemetry
  - Global progress, per-stage timings (coarse/fine), environment snapshot, resolved config.
- Outputs & tests
  - Saves pyramid visuals and reconstructions; basic tests for color utils and progress.

## Repository structure (key files)
- `main.py` — Batch PoC orchestration for assets/configs.
- `slpr/api.py` — Session-based API for reconstruction (arrays only).
- `slpr/scene.py` — Scene parsing and frame rendering with ROI and border handling.
- `slpr/scripts/sr_scene_cli.py` — CLI to render frames and assemble GIF/MP4.
- `slpr/pyramid.py` — Laplacian/Gaussian pyramids.
- `slpr/sampling.py` — Stochastic reconstruction core.
- `slpr/analysis.py` — Metrics (MSE/PSNR; SSIM optional).
- `slpr/render.py` — Pyramid visualization helpers.
- `slpr/utils.py` — I/O, color conversions, JSON, seeding.
- `slpr/progress.py` — Shared counter + tqdm glue.
- `slpr/logging_utils.py` — Logging setup.
- `slpr/env_info.py` — Environment capture.
- `assets/generate_assets.py` — Synthetic assets.
- `tests/` — Unit tests.

## Data flows
Batch run (PoC):
1) Load image per `config.yaml`.
2) Build pyramids and save visuals.
3) Reconstruct 100% averaged output; compute metrics.
4) Reconstruct requested zooms; save outputs.
5) Persist timings and logs.

Scene render (animation):
1) Load scene YAML and input(s).
2) Build frame timeline from `fps` and `duration_sec` per phase.
3) For each frame: compute crop and zoom; choose algorithm; compute ROI if needed.
4) Render, apply border mode, write PNG; assemble GIF/MP4 optionally.

## Configuration
- Batch PoC: `config.yaml` with `global`, `assets`, `configs`. Resolved config is saved to `outputs/.../resolved_config.json`.
- Scene: scene YAML supports `fps`, `out_width/height`, `border_mode`, `algorithm`, phases with keyframes (u,v,zoom), ROI tuning, multi-source weights, scene-level `slpr_params`, and per-keyframe `params`.

## How to run
- Generate assets:
  - python -m assets.generate_assets
- Batch PoC:
  - python main.py --config config.yaml
  - Quick smoke: python main.py --config quick_rgb.yaml | quick_luma.yaml
- Scene CLI (frames → GIF/MP4):
  - python -m slpr.scripts.sr_scene_cli --scene scenes/example_scene.yaml
  - Options: --workers, --algorithm, --blend-mode, --seed, and algorithm parameters.

## Dependencies
- Runtime: numpy, opencv-python, Pillow, PyYAML, tqdm, joblib, imageio, imageio-ffmpeg, scikit-image (optional SSIM).
- Dev: pytest, types-Pillow, types-PyYAML. `pyrightconfig.json` included.

## Tests
- tests/test_color_utils.py — RGB↔Gray and YCbCr roundtrip sanity.
- tests/test_progress_counter.py — Shared counter behavior.
- tests/test_api.py — Production API shapes, determinism, and luma path.

## Performance notes
- Vectorized reconstruction reduces Python overhead; reuse pyramids per channel.
- Inter-frame parallelism with single-threaded math inside workers scales well.
- ROI optimization avoids full-frame work on extreme zooms and keeps memory stable.
- Rebuilding pyramids is only needed when `levels` changes; other per-frame params reuse cached sessions.

## Future work
- Multi-source blending: weighted crossfades and random categorical masks.
- Video inputs with timestamped seeking; per-source time keyframes.
- Optional diagnostics (variance maps, level contributions) and CI.
