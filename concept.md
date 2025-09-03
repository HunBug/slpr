
# 🧠 Core concept: Stochastic Laplacian Pyramid

We decompose images with a Laplacian pyramid and reconstruct with randomized local sampling to produce plausible, grainy detail at arbitrary zooms.

1. Decompose into spatial frequency bands via a Laplacian pyramid
2. Reconstruct stochastically using local-neighborhood sampling with jitter
3. Add controlled randomness (dither/noise) per level
4. At high zoom, rely on lower-frequency structure and synthesize plausible detail
5. Results are:

   * **Dotty and grainy**
   * **Plausibly detailed** on zoom-in
   * **Resolution-agnostic** and expressive
   * Useful as **super-resolution / artistic zooms** of real images

---

## 🔁 Traditional vs. stochastic Laplacian

| Step           | Traditional Laplacian         | Stochastic Laplacian (Your Version)          |
| -------------- | ----------------------------- | -------------------------------------------- |
| Decomposition  | Fixed filters + downsampling  | Same                                         |
| Storage        | All detail layers (residuals) | Used as sampling guidance                    |
| Reconstruction | Deterministic blending        | Random sampling of each level’s detail       |
| Use case       | Compression / blending        | Zooming, texture synthesis, dithered art     |

---

## 🧱 Algorithm sketch

### ✅ 1. **Build the Laplacian Pyramid**

**Definition**:
Laplacian pyramid = Difference between consecutive Gaussian-blurred versions of the image.

#### Steps:

* Convert input image to float format (linearize if needed)
* Build Gaussian pyramid: repeatedly blur and downsample
* Compute Laplacian layers:

  $$
  L_i = G_i - \text{Upsample}(G_{i+1})
  $$
* Keep the lowest-resolution image (`G_N`) as the base layer

You now have:

* `L_0, L_1, ..., L_{N-1}`: **high-frequency detail**
* `G_N`: **low-frequency base**

---

### 🎨 2. Stochastic reconstruction

The core twist: don’t just blend the layers — **sample from them**.

For each pixel (x, y) at target resolution:

* Compute its corresponding location in each pyramid level
* For each level:

  * **Look up a small patch** (3x3, 5x5) around that location
  * **Sample randomly** from these pixels (possibly weighted by Gaussian)
  * Optionally jitter location slightly for grainy look
  * Accumulate sampled values from each level, with appropriate scaling

Results:

* Each reconstruction is **different** (grainy, dithered)
* Averaging multiple outputs converges toward original
* Zooming into the image still gives you **stochastic texture**, not blur

---

### 🔍 3. Zooming / super-resolution

At zoom level Z:

* You sample the pyramid **at higher spatial frequencies**
* Since the Laplacian pyramid already separates detail scales:

  * Lower levels guide the **structure**
  * Higher levels inject **texture + detail**
* If zoom is beyond input resolution:

  * Lower-level detail still gives plausible grain
  * You can add **procedural noise or extrapolated samples** to simulate high-frequency detail

Optional:

* Mix real Laplacian components with **synthesized high-frequency noise** (e.g. Perlin, Gaussian, texture patches)

---

## 🖼️ Visual properties

| Feature                    | Result                                                   |
| -------------------------- | -------------------------------------------------------- |
| Local sampling             | Image is structured, but has randomized dots             |
| Patch-based randomness     | Maintains texture, breaks symmetry                       |
| Per-pixel sampling jitter  | Adds natural film-like grain                             |
| Varying patch radius       | Controls fuzziness/sharpness                             |
| Averaging multiple samples | Approaches original image                                |
| Zooming                    | Smooth fade from low to high frequency; plausible detail |

---

## 🧪 Useful knobs

| Parameter                  | Effect                                              |
| -------------------------- | --------------------------------------------------- |
| Pyramid depth              | More depth = smoother base, longer detail chain     |
| Sampling window size       | Larger = blurrier, smaller = crisper                |
| Sampling noise level       | Controls randomness vs determinism                  |
| Per-level noise scaling    | Add more noise to lower levels for dreamier effects |
| Jitter offset range        | Strong jitter = bolder grain                        |
| Procedural noise injection | Good for hallucinating fine details at large zooms  |

---

## ✨ Enhancements

### ➕ Color preservation

* Do Laplacian on **luma** only
* Preserve **original chroma**
* Ensures stochastic sampling doesn't create hue shifts

### ➕ Adaptive pyramid

* Split more in high-detail areas (like adaptive quadtree)
* Use saliency to guide pyramid resolution

### ➕ Multimodal sampling

* Instead of 1 sample per patch, draw **from local histogram**
* Simulates plausible variation from texture-like areas

---

## 🧠 Why this works psychovisually

* Human vision tolerates high-frequency *noise* better than low-frequency *error*
* Grain/noise is often perceived as **texture**
* Local coherence (sampling near pixels) preserves structure
* Random detail fills in **subjective gaps**, useful for super-resolution
* Like **film grain** or **paint stippling**, this feels *organic*

---

## 💾 Storage model (optional)

If you want to store this representation for future use:

* Store the Laplacian pyramid as-is (lossy or lossless)
* Omit exact reconstruction formula
* Instead: store **sampling hints** (e.g. per-level noise maps, masks)

This becomes a **texturable, generative version** of the image.

---

## 📷 Use cases

## 🚀 Current features and next steps
- Implemented
  - Session-based pyramid reuse and vectorized reconstruction.
  - Color modes: gray, RGB, luma (Y with original chroma).
  - Scene/CLI to render animations with normalized coordinates and duration-based phases.
  - Border modes beyond bounds: black, white, edge, repeat, mirror.
  - Algorithm selector: SLPR vs OpenCV (nearest/linear/area/cubic/lanczos4).
  - ROI optimization for large zooms and streaming frame writes with progress.
- Next
  - Multi-source blending (weighted crossfade and random categorical masks).
  - Video inputs with timestamped seeking.
  - Optional diagnostics (variance maps, level contributions).


* Artistic zoomable renderers
* Procedural texture generation from photos
* Stochastic super-resolution
* Data augmentation (noisy variants from same source)
* Retro film-grain stylization
* Pointillism-inspired image renderers
* Experiments in perception (what “feels” like an image?)

