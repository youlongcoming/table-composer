"""Pluggable segmentation of a single dish photo.

Backends
--------
auto       pick rembg when installed, otherwise threshold
rembg      U2Net via onnxruntime; best quality on real photos, runs on CPU
threshold  border-colour distance + Otsu + largest connected component
center     centred ellipse; last-resort fallback, keeps the demo runnable

Every backend returns a boolean mask at the *original* image resolution.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _otsu(hist: np.ndarray) -> int:
    total = float(hist.sum())
    if total <= 0:
        return 127
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    thr = 127
    for t in range(256):
        w_b += float(hist[t])
        if w_b <= 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break
        sum_b += float(t) * float(hist[t])
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            thr = t
    return thr


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Flood-fill the biggest 4-connected blob. Masks are small (<=320px)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best_ys: list[int] = []
    best_xs: list[int] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            ys: list[int] = []
            xs: list[int] = []
            while q:
                cy, cx = q.popleft()
                ys.append(cy)
                xs.append(cx)
                if cy > 0 and mask[cy - 1, cx] and not seen[cy - 1, cx]:
                    seen[cy - 1, cx] = True
                    q.append((cy - 1, cx))
                if cy < h - 1 and mask[cy + 1, cx] and not seen[cy + 1, cx]:
                    seen[cy + 1, cx] = True
                    q.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not seen[cy, cx - 1]:
                    seen[cy, cx - 1] = True
                    q.append((cy, cx - 1))
                if cx < w - 1 and mask[cy, cx + 1] and not seen[cy, cx + 1]:
                    seen[cy, cx + 1] = True
                    q.append((cy, cx + 1))
            if len(ys) > len(best_ys):
                best_ys, best_xs = ys, xs
    out = np.zeros_like(mask)
    if best_ys:
        out[best_ys, best_xs] = True
    return out


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def _segment_threshold(img: Image.Image, work: int = 320) -> np.ndarray | None:
    small = img.copy()
    small.thumbnail((work, work), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    h, w = arr.shape[:2]
    b = max(2, min(6, min(h, w) // 20))
    border = np.concatenate(
        [
            arr[:b].reshape(-1, 3),
            arr[-b:].reshape(-1, 3),
            arr[:, :b].reshape(-1, 3),
            arr[:, -b:].reshape(-1, 3),
        ]
    )
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(arr - bg, axis=2)
    mx = float(diff.max())
    if mx < 6.0:
        return None
    d8 = (diff / mx * 255.0).astype(np.uint8)
    hist = np.histogram(d8, bins=256, range=(0, 256))[0]
    thr = max(_otsu(hist), 25)
    mask = d8 > thr
    if mask.mean() < 0.02 or mask.mean() > 0.95:
        return None
    mask = _largest_component(mask)
    out = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        img.size, Image.Resampling.BILINEAR
    )
    return np.asarray(out) > 127


def _segment_center(img: Image.Image) -> np.ndarray:
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = w * 0.42, h * 0.42
    cx, cy = w / 2.0, h / 2.0
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _segment_rembg(img: Image.Image) -> np.ndarray:
    from rembg import remove  # imported lazily

    out = remove(img.convert("RGB"))
    return np.asarray(out)[..., 3] > 127


# --------------------------------------------------------------------------- #
# public
# --------------------------------------------------------------------------- #
def segment(img: Image.Image, backend: str = "auto") -> tuple[np.ndarray, str]:
    img = img.convert("RGB")
    if backend == "auto":
        try:
            import rembg  # noqa: F401

            backend = "rembg"
        except Exception:
            backend = "threshold"

    if backend == "rembg":
        return _segment_rembg(img), "rembg"
    if backend == "threshold":
        mask = _segment_threshold(img)
        if mask is not None:
            return mask, "threshold"
        return _segment_center(img), "center(fallback)"
    if backend == "center":
        return _segment_center(img), "center"
    raise ValueError(f"unknown segment backend: {backend}")
