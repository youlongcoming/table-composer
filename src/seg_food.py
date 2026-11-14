"""Segmentation inference: plate mask from a trained checkpoint.

The trained model predicts *food* (FoodSeg103 has no plate annotations), so the
plate is inferred from the food region: take its bounding ellipse and expand it
to the rim. That assumption holds for round plates, which is most of them.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

_CACHE: dict[str, object] = {}
DEFAULT_WEIGHTS = Path("weights/seg_foodseg.pt")


def _load(weights: str | os.PathLike, device: str = "cpu"):
    key = f"{weights}|{device}"
    if key in _CACHE:
        return _CACHE[key]

    import torch

    from .unet import UNet

    ckpt = torch.load(str(weights), map_location=device)
    base = int(ckpt.get("base", 32))
    size = int(ckpt.get("image_size", 384))
    net = UNet(base=base)
    net.load_state_dict(ckpt["model"])
    net.to(device).eval()
    _CACHE[key] = (net, size, device)
    return net, size, device


def food_probability(img: Image.Image, weights, device: str = "cpu") -> np.ndarray:
    """Full-resolution probability map of the food region."""
    import torch

    net, size, dev = _load(weights, device)
    rgb = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    x = torch.from_numpy(arr).permute(2, 0, 1)[None].float().to(dev)
    with torch.no_grad():
        logits = net(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return np.asarray(
        Image.fromarray((prob * 255).astype(np.uint8)).resize(img.size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def plate_from_food(mask: np.ndarray, expand: float = 1.55) -> np.ndarray:
    """Grow a food mask into a plate mask by fitting its bounding ellipse."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return mask
    h, w = mask.shape
    cy = (ys.min() + ys.max()) / 2.0
    cx = (xs.min() + xs.max()) / 2.0
    ry = max(1.0, (ys.max() - ys.min()) / 2.0) * expand
    rx = max(1.0, (xs.max() - xs.min()) / 2.0) * expand
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def segment(img: Image.Image, weights=DEFAULT_WEIGHTS, device: str = "cpu",
            threshold: float = 0.5, expand: float = 1.55,
            return_food: bool = False) -> np.ndarray:
    prob = food_probability(img, weights, device)
    food = prob > threshold
    # drop pepper noise without pulling in scipy
    soft = Image.fromarray((food.astype(np.uint8)) * 255).filter(ImageFilter.MedianFilter(5))
    food = np.asarray(soft) > 127
    if return_food:
        return food
    return plate_from_food(food, expand=expand)
