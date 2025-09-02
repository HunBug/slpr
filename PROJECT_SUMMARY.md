# Project Summary

Stochastic Laplacian Pyramid Renderer (SLPR) PoC in Python 3.11. This document captures what’s implemented, how it’s structured, how to run it, and where to extend.

## What’s implemented
- Core algorithms
  - Gaussian/Laplacian pyramid construction and visualization.
  - Stochastic per-level reconstruction and multi-sample averaging.
  - Color modes: gray, RGB (per-channel), and luma (Y with original Cb/Cr via YCbCr).
- Batch runner
  - Config-driven runs over multiple assets and algorithm configs.
  - Parallel execution with joblib; deterministic seeds when provided.
- Progress & telemetry
  - Aggregated global progress bar across workers (pixel-based), throttled updates.
  - Per-stage timing with optional profiling and coarse/fine aggregation.
  - End-of-run aggregated timing summary (JSON + logs) to compare stages.
- Metrics & outputs
  - MSE/PSNR metrics (SSIM optional via scikit-image) @100% recon.
  - Saves pyramid visuals and reconstructions for requested zoom levels.
- Developer ergonomics
  - Structured logging, environment snapshot, resolved config capture.
  - Typed helpers and light tests (color utilities, progress counter).

## Repository structure (key files)
- `main.py` — Orchestrates runs, parallelism, progress aggregation, timings, and saving outputs.
- `slpr/pyramid.py` — Laplacian/gaussian pyramid build + visualization helpers.
- `slpr/sampling.py` — Stochastic reconstruction and averaging (OpenCV resize with Pillow fallback).
- `slpr/analysis.py` — MSE/PSNR (+ optional SSIM if scikit-image present).
- `slpr/render.py` — Rendering/saving of pyramid visualizations.
- `slpr/utils.py` — Image I/O, color conversions (RGB↔Gray, RGB↔YCbCr), paths, JSON, seeding.
- `slpr/progress.py` — Shared counter + background global tqdm bar.
- `slpr/logging_utils.py` — File/console logging setup.
- `slpr/env_info.py` — Captures Python/platform/env details to JSON.
- `assets/generate_assets.py` — Creates gray and colorful sample inputs.
- `tests/` — Basic sanity tests.
- `quick_rgb.yaml`, `quick_luma.yaml` — Small configs for smoke runs.

## Data flow (per asset/config)
1) Load image (mode per `global.color_mode`).
2) Build grayscale pyramid for visualization; save pyramid images.
3) Reconstruct 100% averaged output; compute metrics vs original.
4) For each `zoom_levels` entry: reconstruct one sample and an averaged image; save both.
5) Persist per-case timings; at the end aggregate and summarize timings across the run.

## Configuration
YAML top-level keys: `global`, `assets`, `configs`.
- global
  - `output_root`: directory for timestamped run outputs.
  - `num_cores`: integer or "auto" (None) for joblib.
  - `samples_per_render`: averaging samples per reconstruction.
  - `zoom_levels`: list of zoom percentages (ints).
  - `color_mode`: "gray" | "rgb" | "luma".
  - `seed`: optional int for determinism.
  - `log_level`: DEBUG|INFO|WARN|ERROR.
  - `profiling` (bool, default true): enable timing.
  - `profiling_granularity` ("coarse"|"fine", default "coarse"): group stage names.
  - `progress_update_pixels` (int, default 25000): batch size for progress increments.
- assets: list of `{ name, path }`.
- configs: list of algorithm configs `{ name, pyramid_levels, patch_size, jitter, noise_strength }`.

The fully parsed configuration is saved to `outputs/<stamp>/resolved_config.json`.

## Progress & timings
- Global bar shows aggregated processed pixels across all workers.
- Updates are throttled by `progress_update_pixels` to reduce synchronization overhead.
- Per-case timing saved as `timings.json`; aggregated timings saved as `run_timings.json` with a log summary sorted by total time per stage bucket (pyramid, stochastic, save, metrics when coarse).
- Disable profiling by setting `profiling: false` to minimize overhead.

## Outputs layout
`outputs/<stamp>/` contains:
- `run.log`, `environment.json`, `resolved_config.json`, `run_timings.json`.
- For each asset/config: pyramid images, reconstructions per zoom, metrics at 100%, and per-case `timings.json`.

## How to run
- Generate assets:
  - `python -m assets.generate_assets`
- Execute a run:
  - `python main.py --config config.yaml`
  - Quick smoke tests:
    - `python main.py --config quick_rgb.yaml`
    - `python main.py --config quick_luma.yaml`

## Dependencies
- Runtime: numpy, opencv-python, Pillow, PyYAML, tqdm, joblib, tqdm-joblib, scikit-image (optional for SSIM).
- Dev: pytest, types-Pillow, types-PyYAML; `pyrightconfig.json` relaxes missing-stub noise.

## Tests
- `tests/test_color_utils.py` — RGB↔Gray and YCbCr roundtrip sanity.
- `tests/test_progress_counter.py` — Shared counter increment behavior.

## Performance notes
- Overhead sources: synchronization for progress updates and fine-grained timing.
- Mitigations implemented: progress batching and coarse timing groups (default). Toggle off profiling for minimal overhead.

## Next steps (optional)
- Finer-grained per-zoom timing buckets if needed for deeper profiling.
- CI workflow for tests and (optional) lint/type checks.
- Additional metrics or visual diagnostics.
