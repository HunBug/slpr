# SLPR PoC

Stochastic Laplacian Pyramid Renderer in Python 3.11.

See PROJECT_SUMMARY.md for a concise overview of the implementation and how to run/extend it.

Quick start:

1. Create assets:
   - python -m assets.generate_assets
2. Run:
   - python main.py --config config.yaml
   - For quick smoke tests: python main.py --config quick_rgb.yaml or quick_luma.yaml

Outputs will appear under `outputs/YYYY-MM-DD_HHMMSS/asset/config/`.

Requirements: see requirements.txt
Dev typing/stubs: see requirements-dev.txt. A pyrightconfig.json is included to reduce type-noise and ignore missing stubs for joblib/tqdm-joblib.

Notes:
- Supports grayscale, RGB, and luma (Y with original Cb/Cr) modes via `global.color_mode`.
- Deterministic when `seed` is set.
- For SSIM, scikit-image must be installed (already in requirements.txt).

Environment & logging:
- The run root (e.g., `outputs/<stamp>/`) contains `environment.json` (Python, platform, numpy, basic env vars) and `resolved_config.json` (fully parsed config) plus a `run.log` with INFO-level logs.
- Adjust verbosity via `global.log_level` in `config.yaml` (DEBUG/INFO/WARN/ERROR).

Progress tracking:
- Outer case processing uses tqdm; inner per-pixel sampling shows level row progress for large outputs (> 256x256) to avoid noisy bars for small runs.

## Scene CLI (frames → GIF/MP4)

Render an animation defined by a scene YAML and assemble outputs:

- python -m slpr.scripts.sr_scene_cli --scene scenes/example_scene.yaml

Options:
- --out-dir /custom/output
- --seed 123
- --levels 5 --patch-size 3 --jitter 1.0 --noise-strength 0.1 --samples 1
- --workers N (defaults to CPU cores)
- --algorithm slpr|cv2_nearest|cv2_linear|cv2_area|cv2_cubic|cv2_lanczos4
- --no-gif / --no-mp4 to skip outputs

A minimal quick test is provided at `scenes/quick_test.yaml`.

Scene YAML extras:
- border_mode: black|white|mirror|repeat|edge (default: black)
- algorithm: slpr (default) or any cv2_* from the CLI choices above