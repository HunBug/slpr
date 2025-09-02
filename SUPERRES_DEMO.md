# Super-Resolution Demo: Production API and App Plan

This document defines a clean, production-style API for SLPR (returning arrays, no debug I/O) and a plan for a demo app that generates a zoom animation (frames in parallel) and assembles a GIF/MP4.

## Goals
- Provide a minimal, easy-to-call API that returns results only (no saving, no logs).
- Support gray/rgb/luma color modes with deterministic seeds.
- Efficient: reuse pyramids per channel, vectorized reconstruct, frame-parallel execution.
- Demo app: show original (dynamic noise), zoom to target factor while cropping to original size, hold at final zoom; build GIF/MP4.

## Production API (proposed)
Create `slpr/api.py` with the following functions. These are pure: they return NumPy arrays in [0,1], no disk writes, no logging.

- build_channel_pyramids(image_2d: np.ndarray, levels: int) -> tuple[list[np.ndarray], np.ndarray]
  - Inputs: 2D float32 array in [0,1].
  - Outputs: (laplacians, base_low_res).
  - Error: ValueError if ndim != 2.

- reconstruct_from_pyramids(
    laplacians: list[np.ndarray],
    base: np.ndarray,
    target_shape: tuple[int, int],
    patch_size: int,
    jitter: float,
    noise_strength: float,
    seed: int | None,
  ) -> np.ndarray
  - Returns a single stochastic reconstruction (no averaging).

- reconstruct_avg_from_pyramids(..., samples: int, seed: int | None) -> np.ndarray
  - Returns multi-sample average (deterministic if seed provided).

- reconstruct_image(
    image: np.ndarray,              # 2D gray or 3D (H,W,3) RGB in [0,1]
    color_mode: Literal["gray","rgb","luma"],
    levels: int,
    target_shape: tuple[int, int],
    patch_size: int,
    jitter: float,
    noise_strength: float,
    samples: int = 1,               # 1 => single stochastic frame (dynamic)
    seed: int | None = None,
  ) -> np.ndarray                    # shape: (H',W') or (H',W',3)
  - Internally: builds pyramids per channel once; calls (avg or single) reconstruct per channel; stacks.
  - Error: ValueError on unsupported mode or shape.

Notes
- Per-frame single-sample (samples=1) gives dynamic noise.
- For speed, callers can pre-build per-channel pyramids and reuse across frames.

## Demo App Plan (`scripts/sr_demo.py`)
A separate script that uses the production API to generate frames and assemble an animation.

### Behavior
- Timeline segments (all durations configurable):
  1) Hold original @1x for T_hold_start seconds; per-frame stochastic 100% recon (samples=1) for dynamic noise.
  2) Smooth zoom from 1x to Zx over T_zoom seconds; per-frame stochastic recon at each intermediate zoom.
  3) Hold final zoom Zx for T_hold_end seconds; per-frame stochastic recon for dynamic noise.
- Output: GIF or MP4 (user chooses). Frames cropped back to original size to keep a stable canvas.

### CLI (proposed)
- --input PATH: image file
- --output PATH: .gif or .mp4
- --zoom Z: target zoom factor (e.g., 8, 16)
- --fps N: frames per second (e.g., 30)
- --hold-start SEC: seconds at 1x (default 2.0)
- --zoom-sec SEC: seconds to reach final zoom (default 3.0)
- --hold-end SEC: seconds at final zoom (default 2.0)
- --color-mode gray|rgb|luma (default luma)
- --levels L: pyramid levels (default from config)
- --patch K: patch size (default 3)
- --jitter J: jitter stddev (default 1.0)
- --noise S: noise strength (default 0.1)
- --seed SEED: optional int (for determinism across runs)
- --cores N: frame-parallel workers (default: auto)

### Frame Generation Logic
- Load input as float32 [0,1]. Save (H,W) and RGB/luma conversion as needed.
- Precompute pyramids per channel once (reused across all frames). This avoids recomputing per frame.
- For each frame i at time t:
  - Compute zoom factor z(t):
    - Segment 1 (0..T_hold_start): z=1
    - Segment 2 (T_hold_start..T_hold_start+T_zoom): interpolate z from 1..Z (e.g., smoothstep)
    - Segment 3: z=Z
  - Compute target shape: (round(H*z), round(W*z))
  - Reconstruct single-sample frame (samples=1) at target shape using prebuilt pyramids.
  - If z>1, center-crop to (H,W) to keep original canvas size. If z<1 (not typical), center-pad or upscale first.
  - Convert back to RGB for luma mode using original Cb/Cr (constant chroma), then crop.
- Parallelization: schedule frames i=0..N-1 with joblib.Parallel(n_jobs=cores), using per-process env:
  - OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, OPENCV_NUM_THREADS=1
- Write frames as PNGs to a temp folder (e.g., outputs/<stamp>/frames/frame_000001.png). This keeps memory low and preserves order.

### Assembly
- GIF: use imageio (Pillow backend) to stack frame PNGs; set duration=1/fps per frame; optionally palette optimize.
- MP4: prefer imageio-ffmpeg (or system ffmpeg) for H.264 at given fps; stream frames in order.
- Clean up temp frames if desired (controlled by --keep-frames).

### Determinism
- With a fixed seed, per-frame RNG can be derived from i (e.g., base_seed + i) so animation is reproducible. If you want non-repro runs, omit seed.

### Pseudocode Sketch
- load image -> rgb or luma triplet
- per-channel build pyramids once
- build frame timeline (timestamps, zoom factors)
- Parallel map frames:
  - seed_i = None if seed is None else seed + i
  - img_i = reconstruct_from_pyramids(..., target_shape, seed=seed_i)
  - crop to (H,W), reassemble color
  - save frame_i.png
- assemble GIF or MP4

### Performance Notes
- Reusing pyramids per channel dramatically reduces compute when generating many frames.
- Single-sample (samples=1) keeps the noise dynamic and is faster than averaging.
- Using frame-level parallelism scales linearly with cores; keep math libs single-threaded per worker.

### Dependencies
- Runtime: existing requirements plus imageio and (optional) imageio-ffmpeg for MP4.
- If you prefer system ffmpeg, detect and call it; otherwise fall back to imageio’s writer.

### Testing & Acceptance
- Unit tests for API functions:
  - Gray/RGB/luma reconstruct returns expected shapes and ranges.
  - Deterministic when seed fixed.
- Smoke test for demo: generate a 2-second 1x hold at 10 fps and check frame count and output creation.

### Next Steps
- Implement `slpr/api.py` as above.
- Add `scripts/sr_demo.py` with CLI and parallel frame generation.
- Add imageio/imageio-ffmpeg to requirements.
- Provide a small README section showing how to run the demo.
