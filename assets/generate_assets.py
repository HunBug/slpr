from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def grid(w: int, h: int, step: int = 16, thickness: int = 1) -> Image.Image:
    img = np.ones((h, w), dtype=np.float32)
    for y in range(0, h, step):
        img[max(0, y - thickness // 2) : min(h, y + thickness // 2 + 1), :] = 0.0
    for x in range(0, w, step):
        img[:, max(0, x - thickness // 2) : min(w, x + thickness // 2 + 1)] = 0.0
    return Image.fromarray((img * 255).astype(np.uint8), mode="L")


def blobs(w: int, h: int, n: int = 20, r: int = 6, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n):
        cy = int(rng.integers(r, h - r))
        cx = int(rng.integers(r, w - r))
        rr = float(rng.integers(r // 2, r * 2))
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        img += np.exp(-d2 / (2.0 * (rr ** 2)))
    img = img / (img.max() + 1e-6)
    return Image.fromarray((img * 255).astype(np.uint8), mode="L")


def main() -> None:
    out = Path("assets")
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.png").parent.mkdir(parents=True, exist_ok=True)
    grid_img = grid(512, 512, step=24, thickness=2)
    blobs_img = blobs(512, 512, n=40, r=10, seed=42)
    grid_img.save(out / "grid.png")
    blobs_img.save(out / "blobs.png")

    # Colorful patterns
    def save_color(arr: np.ndarray, path: Path) -> None:
        img = Image.fromarray(
            (np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB"
        )
        img.save(path)

    # Horizontal/vertical/diagonal lines in color
    sz = 512
    img = Image.new("RGB", (sz, sz), (0, 0, 0))
    dr = ImageDraw.Draw(img)
    step = 32
    # horizontal red lines
    for y in range(0, sz, step):
        dr.line([(0, y), (sz, y)], fill=(255, 64, 64), width=3)
    # vertical green lines
    for x in range(0, sz, step):
        dr.line([(x, 0), (x, sz)], fill=(64, 255, 64), width=3)
    # diagonal blue lines
    for d in range(-sz, sz, step * 2):
        p0 = (max(0, -d), max(0, d))
        p1 = (min(sz, sz - d), min(sz, sz + d))
        dr.line([p0, p1], fill=(64, 64, 255), width=3)
    img.save(out / "color_lines.png")

    # Colored blobs
    rng = np.random.default_rng(0)
    color_blobs = np.zeros((sz, sz, 3), dtype=np.float32)
    for _ in range(40):
        cx, cy = int(rng.integers(0, sz)), int(rng.integers(0, sz))
        r = int(rng.integers(10, 60))
        color = rng.random(3)
        yy, xx = np.ogrid[:sz, :sz]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        for c in range(3):
            ch = color_blobs[:, :, c]
            ch[mask] = np.maximum(ch[mask], color[c])
            color_blobs[:, :, c] = ch
    save_color(color_blobs, out / "color_blobs.png")

    # Curves in color
    img2 = Image.new("RGB", (sz, sz), (0, 0, 0))
    dr2 = ImageDraw.Draw(img2)
    for i, col in enumerate([(255, 128, 0), (0, 255, 255), (255, 0, 255)]):
        pts = []
        for x in range(0, sz, 4):
            y = int(
                sz / 2
                + (sz / 4) * np.sin(2 * np.pi * (x / sz) * (i + 1) + i * 0.7)
            )
            pts.append((x, y))
        dr2.line(pts, fill=col, width=3)
    img2.save(out / "color_curves.png")


if __name__ == "__main__":
    main()
