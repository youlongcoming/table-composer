"""Backgrounds, contact shadows and compositing."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_background(w: int, h: int, kind: str = "wood", seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy = np.linspace(0.0, 1.0, h)[:, None, None]
    xx = np.linspace(0.0, 1.0, w)[None, :, None]

    if kind == "wood":
        base = np.array([0.40, 0.27, 0.18], dtype=np.float32)
        tone = 0.88 + 0.30 * yy
        grain = 0.016 * np.sin((yy * 14.0 + 0.25 * np.sin(xx * 3.0)) * 6.2832)
        streak = 0.010 * np.sin((xx * 5.0 + yy * 2.0) * 6.2832)
        noise = rng.normal(0.0, 0.010, (h, w, 1)).astype(np.float32)
        img = base.reshape(1, 1, 3) * tone + grain + streak + noise
    elif kind == "linen":
        base = np.array([0.78, 0.73, 0.66], dtype=np.float32)
        weave = 0.006 * np.sin(xx * 240.0) * np.sin(yy * 240.0)
        noise = rng.normal(0.0, 0.01, (h, w, 1)).astype(np.float32)
        img = base.reshape(1, 1, 3) * (0.95 + 0.10 * yy) + weave + noise
    elif kind == "dark":
        img = np.full((h, w, 3), 0.09, dtype=np.float32) + rng.normal(0.0, 0.008, (h, w, 1))
    elif kind == "solid":
        img = np.full((h, w, 3), 0.62, dtype=np.float32)
    else:
        raise ValueError(f"unknown background: {kind}")

    # vignette
    r = np.sqrt((xx - 0.5) ** 2 + ((yy - 0.5) * 0.72) ** 2)
    vig = 1.0 - 0.38 * np.clip(r * 1.5 - 0.30, 0.0, 1.0)
    img = np.clip(img * vig, 0.0, 1.0)
    return Image.fromarray((img * 255.0).astype(np.uint8), "RGB")


def _shadow(d: float, strength: float) -> Image.Image:
    """Contact shadow that reads from above: a slightly larger, blurred ellipse.

    A plate seen from above casts a thin halo around its rim, thicker at the
    bottom edge. Drawing it larger than the plate is what makes the cutout look
    like it is sitting on the table instead of pasted onto it.
    """
    span = 2.6 * d
    r = 0.55 * d
    layer = Image.new("L", (int(span), int(span)), 0)
    draw = ImageDraw.Draw(layer)
    c = span / 2.0
    draw.ellipse([c - r, c - r * 0.97, c + r, c + r * 1.03], fill=int(255 * max(0.0, min(1.0, strength))))
    layer = layer.filter(ImageFilter.GaussianBlur(max(1.0, d * 0.035)))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.putalpha(layer)
    return out


def render(
    background: Image.Image,
    placements,
    *,
    shadow: float = 0.45,
    shadow_offset: float = 0.035,
) -> Image.Image:
    canvas = background.convert("RGBA")
    for p in sorted(placements, key=lambda q: q.z):
        plate = p.image.resize(
            (max(1, int(round(p.size))), max(1, int(round(p.size)))),
            Image.Resampling.LANCZOS,
        )
        sh = _shadow(p.size, shadow)
        sh_x = int(round(p.cx - sh.width / 2))
        sh_y = int(round(p.cy + p.size * shadow_offset - sh.height / 2))
        canvas.alpha_composite(sh, (sh_x, sh_y))
        canvas.alpha_composite(
            plate, (int(round(p.cx - plate.width / 2)), int(round(p.cy - plate.height / 2)))
        )
    return canvas.convert("RGB")
