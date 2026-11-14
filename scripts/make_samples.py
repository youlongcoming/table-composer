"""Generate synthetic dish photos so the pipeline can be tested offline.

These are NOT dataset images. They only prove the pipeline runs end to end
before you download Food-101 / FoodSeg103.

Each sample is a light plate on a darker table, which is what real food photos
look like and what the threshold segmenter needs.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "samples"

# darker, varied tabletops so a white plate separates cleanly
BACKDROPS = [
    (74, 58, 44),
    (58, 62, 66),
    (92, 74, 56),
    (66, 52, 46),
    (80, 86, 70),
]

FOODS = [
    [(196, 78, 48), (226, 140, 64), (120, 60, 40)],
    [(86, 130, 62), (150, 180, 96), (60, 92, 48)],
    [(212, 176, 108), (238, 210, 158), (168, 128, 72)],
    [(150, 60, 92), (200, 110, 140), (110, 42, 68)],
    [(70, 104, 148), (120, 156, 196), (48, 74, 108)],
]


def make_dish(idx: int, size: int, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size), BACKDROPS[idx % len(BACKDROPS)])
    d = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    r = size * 0.34

    # soft shadow under the plate, gives the segmenter a clean outer edge
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse(
        [cx - r * 1.02, cy - r * 0.96, cx + r * 1.02, cy + r * 1.10], fill=(0, 0, 0, 90)
    )
    img.paste(sh.filter(ImageFilter.GaussianBlur(size / 45)), (0, 0), sh)

    # plate
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 249, 246))
    d.ellipse([cx - r * 0.90, cy - r * 0.90, cx + r * 0.90, cy + r * 0.90], fill=(243, 242, 238))
    d.arc([cx - r, cy - r, cx + r, cy + r], 200, 340, fill=(214, 210, 202), width=max(2, size // 120))

    # food
    palette = FOODS[idx % len(FOODS)]
    n = rng.randint(9, 16)
    for i in range(n):
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0, r * 0.52)
        px = cx + math.cos(a) * rr
        py = cy + math.sin(a) * rr
        s = rng.uniform(r * 0.11, r * 0.24)
        d.ellipse([px - s, py - s, px + s, py + s], fill=palette[i % len(palette)])

    # sauce streak
    d.arc(
        [cx - r * 0.6, cy - r * 0.6, cx + r * 0.6, cy + r * 0.6],
        rng.randint(0, 180),
        rng.randint(200, 340),
        fill=palette[-1],
        width=max(1, size // 200),
    )

    return img.filter(ImageFilter.GaussianBlur(size / 900))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    sizes = [900, 820, 760, 700, 640]
    for i, s in enumerate(sizes):
        img = make_dish(i, s, rng)
        p = OUT / f"dish_{i + 1:02d}.png"
        img.save(p)
        print(f"wrote {p}  {s}x{s}")


if __name__ == "__main__":
    main()
