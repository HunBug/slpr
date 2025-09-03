# SLPR – Implementation Plan & Roadmap

Project: Stochastic Laplacian Pyramid Renderer (SLPR)
Language: Python 3.11

This document captures where we are and the plan to evolve SLPR into a more capable, ergonomic tool. It focuses on concrete milestones with acceptance criteria so we can implement confidently and verify progress.

## 0) Current baseline (today)
- Batch PoC driven by YAML (`config.yaml`) and `main.py`.
- Core modules in `slpr/`: pyramids, stochastic reconstruction, rendering/analysis, progress & logging.
- Outputs saved to timestamped folders under `outputs/`.
- Assets generator for grids/lines/blobs under `assets/`.

Constraints and gaps:
- API is not yet a small, pure “production” API that returns arrays.
- No scene/animation system; no per-frame parallel rendering CLI.
- No border behavior for crops beyond image bounds.
- No algorithm selector to compare with OpenCV resamplers.
- No ROI optimization for extreme zooms.
- No multi-source blending or video input.

## 1) Milestones and acceptance criteria

M1: Production API (pure functions)
- Deliverables:
	- `slpr/api.py` exposing: build_channel_pyramids, reconstruct_from_pyramids, reconstruct_avg_from_pyramids, reconstruct_image.
	- Unit tests for shapes, determinism with fixed seeds, and luma path handling.
- Acceptance:
	- `pytest` green; functions documented via docstrings; no file I/O or logging.

M2: Scene system + CLI (animation)
- Deliverables:
	- `slpr/scene.py` for parsing a scene YAML and rendering frames.
	- `slpr/scripts/sr_scene_cli.py` to render frames to PNG and optionally assemble GIF/MP4.
	- Schema highlights:
		- Normalized (u,v) coordinates with aspect-ratio aware crops.
		- Phases defined by `duration_sec`; timeline built from `fps`.
		- `border_mode`: black|white|edge|repeat|mirror for out-of-bounds.
		- `algorithm`: slpr or cv2_{nearest,linear,area,cubic,lanczos4} to compare.
		- Defaults: workers = CPU cores; per-frame seed = base_seed + frame_index.
	- Streaming: write frames as they’re computed (low memory), global progress bar.
- Acceptance:
	- Renders example scene to frames under `outputs/.../frames/`.
	- Optional GIF/MP4 assembly via imageio/imageio-ffmpeg.
	- Deterministic with fixed seed; border modes visually correct for edge crops.

M3: ROI optimization for large zooms
- Deliverables:
	- Compute and process only the source region necessary for the target crop, with configurable padding (e.g., `roi_min_zoom`, `roi_pad_scale`).
- Acceptance:
	- Rendering of very large zooms remains stable in memory and faster than full-frame.
	- Visual equivalence to full-frame within crop tolerance.

M4: Multi-source blending (images)
- Deliverables:
	- Extend scene schema to support multiple inputs and per-phase weights.
	- Blend modes:
		- weighted: per-frame normalized weights for crossfades.
		- random: categorical mask per frame seeded by frame index for textural mixes.
- Acceptance:
	- Example crossfade scene that transitions between two images.
	- Deterministic masks when seed fixed.

Status: Implemented.
Notes: YAML parsing is isolated in `load_scene` and maps to a neutral Scene model (dataclasses). Renderers use the model + arrays; changing YAML structure is a small refactor in the loader only.

M5: Video inputs (optional follow-up)
- Deliverables:
	- Allow inputs to be videos with timestamped sampling; per-source keyframes include `time_sec`.
	- Frame seeking via imageio-ffmpeg.
- Acceptance:
	- Example scene combining stills and video with stable performance.

M6: Diagnostics and performance polish
- Deliverables:
	- Toggleable diagnostics (variance maps, level contributions) for a frame.
	- Fine/coarse timing summaries; environment capture for runs.
- Acceptance:
	- Diagnostics written only when enabled; overhead negligible when off.

## 2) Scene YAML (draft schema)
Top-level keys:
- `fps`: int
- `out_width`, `out_height`: ints
- `border_mode`: black|white|edge|repeat|mirror (default black)
- `algorithm`: slpr|cv2_nearest|cv2_linear|cv2_area|cv2_cubic|cv2_lanczos4 (default slpr)
- `inputs`: list of { name, path }
- `phases`: list of phases, each with:
	- `duration_sec`: float
	- `keyframes`: list of keyframes, each with:
		- `u`, `v`: floats in [0,1] (center position)
		- `zoom`: float (1.0 = native)
		- optional `weights`: { input_name: weight } for multi-source (normalized per frame)

Algorithm params (global defaults, CLI-overridable):
- `levels`, `patch_size`, `jitter`, `noise_strength`, `samples`, `seed`, `workers`.

Notes:
- Legacy convenience: allow a single `input` string; map to `inputs: [{name: main, path: ...}]`.
- Timeline interpolation: linear by default; smoothing (ease-in/out) may be added later.

## 3) Work plan (incremental PRs)
1. Implement M1 (API) with tests.
2. Implement M2 (scene + CLI) minimal path: single input, slpr algorithm, black border, streaming frames.
3. Add M2 extras: border modes, algorithm selector, default workers.
4. Implement M3 (ROI) and verify with a large-zoom example.
5. Implement M4 (multi-source) with a crossfade example.
6. Optional M5 video; then M6 diagnostics.

## 4) Verification
- Always run `pytest` after changes; keep tests small and deterministic.
- Provide one or two example scenes for manual smoke checks; commit small inputs.
- For performance claims (ROI), log timing deltas in `outputs/.../run.log`.

## 5) Dependencies
- Runtime: numpy, opencv-python, Pillow, PyYAML, tqdm, joblib, imageio, imageio-ffmpeg.
- Dev: pytest, typing stubs as needed; `pyrightconfig.json` configured.

## 6) Out-of-scope for now
- Learned models; neural SR; GPU acceleration.
- Fancy UIs—stick to CLI + YAML.

