"""Generate a tiny synthetic segmentation dataset to smoke-test training.

Not a substitute for FoodSeg103 — it only proves the training loop, the
checkpoint format and the inference path work end to end without downloading
anything.

    python scripts/make_seg_dataset.py --n 120 --out data/synthetic_seg
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

BACKDROPS = [(74, 58, 44), (58, 62, 66), (92, 74, 56), (66, 52, 46), (80, 86, 70)]
FOODS = [
    [(196, 78, 48), (226, 140, 64)],
    [(86, 130, 62), (150, 180, 96)],
    [(212, 176, 108), (238, 210, 158)],
    [(150, 60, 92), (200, 110, 140)],
    [(70, 104, 148), (120, 156, 196)],
]


def make_one(idx: int, size: int, rng: random.Random):
    img = Image.new("RGB", (size, size), BACKDROPS[idx % len(BACKDROPS)])
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    md = ImageDraw.Draw(mask)

    cx = size / 2 + rng.uniform(-0.08, 0.08) * size
    cy = size / 2 + rng.uniform(-0.08, 0.08) * size
    r = size * rng.uniform(0.26, 0.36)

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(
        [cx - r * 1.02, cy - r * 0.96, cx + r * 1.02, cy + r * 1.10], fill=(0, 0, 0, 90)
    )
    img.paste(shadow.filter(ImageFilter.GaussianBlur(size / 45)), (0, 0), shadow)

    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 249, 246))
    md.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    palette = FOODS[idx % len(FOODS)]
    for _ in range(rng.randint(8, 15)):
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0, r * 0.5)
        px, py = cx + math.cos(a) * rr, cy + math.sin(a) * rr
        s = rng.uniform(r * 0.11, r * 0.24)
        d.ellipse([px - s, py - s, px + s, py + s], fill=palette[rng.randint(0, 1)])

    return img.filter(ImageFilter.GaussianBlur(size / 900)), mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", default="data/synthetic_seg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.out)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for i in range(args.n):
        img, mask = make_one(i, args.size, rng)
        img.save(root / "images" / f"img_{i:04d}.png")
        mask.save(root / "annotations" / f"ann_{i:04d}.png")
    print(f"wrote {args.n} pairs to {root}")


if __name__ == "__main__":
    main()
