"""Rule-based table layout.

The whole point of the project lives here: given N cutouts, decide where each
plate sits, how big it is, and which ones overlap. No model involved.

A layout is not one fixed arrangement. ``plan`` draws several jittered
candidates and keeps the best-scoring one, which is what stops small inputs
from collapsing into a straight line.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Item:
    name: str
    aspect: float = 1.0  # width / height of the cutout bbox
    rank: int = 0


@dataclass
class Placement:
    index: int
    name: str
    cx: float
    cy: float
    size: float
    z: int
    meta: dict = field(default_factory=dict)


def _resolve_overlaps(pl: list[Placement], w: int, h: int, locked: set[int]) -> None:
    """Push circles apart until they no longer intersect, keeping them on canvas."""
    for _ in range(300):
        moved = False
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                a, b = pl[i], pl[j]
                dx = b.cx - a.cx
                dy = b.cy - a.cy
                d = math.hypot(dx, dy) or 1e-6
                # a little overlap is what makes it look like a table, not a shelf
                min_d = (a.size + b.size) * 0.5 * 0.88
                if d >= min_d:
                    continue
                ux, uy = dx / d, dy / d
                push = (min_d - d) * 0.5 + 0.5
                wa = 0.0 if a.index in locked else 1.0
                wb = 0.0 if b.index in locked else 1.0
                tot = wa + wb
                if tot == 0:
                    continue
                a.cx -= ux * push * (wa / tot) * 2
                a.cy -= uy * push * (wa / tot) * 2
                b.cx += ux * push * (wb / tot) * 2
                b.cy += uy * push * (wb / tot) * 2
                moved = True
        for p in pl:
            if p.index in locked:
                continue
            # keep the plate fully on canvas: a sliver cut off at the edge reads
            # as a mistake, not as a crop
            p.cx = min(max(p.cx, p.size * 0.5), w - p.size * 0.5)
            p.cy = min(max(p.cy, p.size * 0.5), h - p.size * 0.5)
        if not moved:
            break


def _candidate(
    items: list[Item],
    w: int,
    h: int,
    *,
    main_ratio: float,
    side_ratio: float,
    ring: float,
    headroom: float,
    jitter: float,
    angle0: float,
    rot: float,
    seed: int,
) -> list[Placement]:
    n = len(items)
    rng = random.Random(seed)
    cy_main = h * (1.0 - headroom) * 0.54 + h * headroom * 0.35

    pl: list[Placement] = [
        Placement(
            index=0,
            name=items[0].name,
            cx=w * 0.5,
            cy=cy_main,
            size=main_ratio * w,
            z=0,
        )
    ]
    if n == 1:
        return pl

    rx = ring * w
    ry = ring * w
    m = n - 1
    # Two sides at 180 degrees would draw a straight line through the main dish,
    # so cap the step: three plates form a triangle, not a column.
    if m == 1:
        base_angles = [math.radians(angle0 + 60 + rot)]
        step = 0.0
    else:
        step = min(2 * math.pi / m, math.radians(140))
        base_angles = [math.radians(angle0 + rot) + (k - 1) * step for k in range(1, n)]

    for k in range(1, n):
        theta = base_angles[k - 1] + math.radians(rng.uniform(-jitter, jitter))
        rr = 1.0 + rng.uniform(-0.08, 0.10)
        size = min(side_ratio, max(0.16, side_ratio - 0.018 * (k - 1)))
        size *= rng.uniform(0.94, 1.06) * w
        pl.append(
            Placement(
                index=k,
                name=items[k].name,
                cx=w * 0.5 + rx * rr * math.cos(theta),
                cy=cy_main + ry * rr * math.sin(theta),
                size=size,
                z=0,
            )
        )

    _resolve_overlaps(pl, w, h, locked={0})
    return pl


def _score(pl: list[Placement], w: int, h: int) -> float:
    """Higher is better. Balance, frame coverage, and not stacking plates."""
    if len(pl) == 1:
        return 0.0
    total = sum(p.size for p in pl) or 1.0
    cxm = sum(p.cx * p.size for p in pl) / total
    cym = sum(p.cy * p.size for p in pl) / total
    score = -1.0 * math.hypot(cxm - w / 2, cym - h / 2) / (w * 0.5)

    minx = min(p.cx - p.size / 2 for p in pl)
    maxx = max(p.cx + p.size / 2 for p in pl)
    miny = min(p.cy - p.size / 2 for p in pl)
    maxy = max(p.cy + p.size / 2 for p in pl)
    score += 1.5 * (maxx - minx) / w + 1.1 * (maxy - miny) / h
    # the visible composition should sit in the middle of the frame, whatever
    # the size weighting says
    score -= 2.4 * math.hypot((minx + maxx) / 2 - w / 2, (miny + maxy) / 2 - h / 2) / (0.5 * h)

    # 2D spread: a straight line of plates scores poorly here even though it may
    # be balanced and cover the full frame.
    pts = np.array([[p.cx, p.cy] for p in pl], dtype=np.float64)
    minor = math.sqrt(max(float(np.linalg.eigvalsh(np.cov(pts.T))[0]), 0.0))
    score += 3.0 * min(minor, 0.45 * w) / (0.45 * w)

    for i in range(len(pl)):
        for j in range(i + 1, len(pl)):
            a, b = pl[i], pl[j]
            d = math.hypot(a.cx - b.cx, a.cy - b.cy)
            min_d = (a.size + b.size) * 0.5
            if d < min_d * 0.78:
                score -= 1.6 * (min_d * 0.78 - d) / min_d

    for p in pl:
        if p.cx - p.size / 2 < -0.01 * w or p.cx + p.size / 2 > 1.01 * w:
            score -= 1.2
        if p.cy + p.size / 2 > h + 0.01 * h or p.cy - p.size / 2 < -0.01 * h:
            score -= 1.2
    return score


def plan(
    items: list[Item],
    w: int,
    h: int,
    *,
    main_ratio: float = 0.48,
    side_ratio: float = 0.27,
    ring: float = 0.42,
    headroom: float = 0.0,
    jitter: float = 14.0,
    angle0: float = -90.0,
    seed: int = 0,
    candidates: int = 16,
) -> list[Placement]:
    """Lay out ``items`` on a ``w x h`` canvas.

    The largest item becomes the main dish on the lower third; the rest go on an
    ellipse around it. ``headroom`` reserves empty space at the top for a person
    or for text.
    """
    if not items:
        return []

    best: list[Placement] | None = None
    best_score = -1e9
    n_cand = max(1, candidates)
    m = max(1, len(items) - 1)
    sweep = 360.0 / m
    for c in range(n_cand):
        pl = _candidate(
            items,
            w,
            h,
            main_ratio=main_ratio,
            side_ratio=side_ratio,
            ring=ring,
            headroom=headroom,
            jitter=jitter,
            angle0=angle0,
            rot=sweep * c / n_cand,
            seed=seed * 1000 + c,
        )
        s = _score(pl, w, h)
        if s > best_score:
            best, best_score = pl, s

    assert best is not None
    for z, p in enumerate(sorted(best, key=lambda q: q.cy)):
        p.z = z
    return best


def to_dict(pl: list[Placement], w: int, h: int) -> dict:
    return {
        "canvas": {"width": w, "height": h},
        "plates": [
            {
                "index": p.index,
                "source": p.name,
                "cx": round(p.cx, 1),
                "cy": round(p.cy, 1),
                "size": round(p.size, 1),
                "z": p.z,
                **p.meta,
            }
            for p in sorted(pl, key=lambda q: q.z)
        ],
    }
