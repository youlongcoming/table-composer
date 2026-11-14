"""Turn a mask into a feathered, colour-matched RGBA cutout."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def build(
    img: Image.Image,
    mask: np.ndarray,
    *,
    feather: float = 2.0,
    color_match: float = 0.08,
    target_rgb: np.ndarray | None = None,
) -> tuple[Image.Image, dict]:
    """Return (RGBA cutout cropped to the mask bbox, metadata)."""
    img = img.convert("RGB")
    alpha = Image.fromarray((mask.astype(np.uint8)) * 255).filter(
        ImageFilter.GaussianBlur(feather)
    )
    a = np.asarray(alpha, dtype=np.float32)[..., None] / 255.0
    rgb = np.asarray(img, dtype=np.float32)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("empty mask")

    # Subtle colour transfer: nudge the plate toward the table's colour so the
    # cutout stops reading as "pasted from another photo".
    if color_match > 0 and target_rgb is not None:
        opaque = a[..., 0] > 0.6
        if opaque.any():
            mean = rgb[opaque].mean(axis=0)
            shift = (np.asarray(target_rgb, dtype=np.float32) - mean) * float(color_match)
            rgb = np.clip(rgb + shift[None, None, :], 0, 255)

    rgba = np.dstack([rgb, a * 255.0]).astype(np.uint8)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = rgba[y0:y1, x0:x1]

    meta = {
        "source_size": [int(img.width), int(img.height)],
        "bbox": [x0, y0, x1, y1],
        "diameter_px": int(max(x1 - x0, y1 - y0)),
        "mask_area": int(mask.sum()),
    }
    return Image.fromarray(crop, "RGBA"), meta
