"""Small image helpers shared by the pipeline."""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit_long_side(img: Image.Image, target: int = 1024) -> Image.Image:
    """Downscale so the long side is at most ``target``. Never upscales."""
    w, h = img.size
    m = max(w, h)
    if m <= target:
        return img
    s = target / m
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)


def mean_color(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(axis=0)
