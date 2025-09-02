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