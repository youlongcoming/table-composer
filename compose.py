#!/usr/bin/env python
"""Compose several single-dish photos into one table layout.

    python compose.py --input samples/*.png --out out/ --bg wood

Produces result.png and layout.json inside the output directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cutout import build  # noqa: E402
from src.layout import Item, plan, to_dict  # noqa: E402
from src.render import make_background, render  # noqa: E402
from src.segment import segment  # noqa: E402
from src.utils import fit_long_side, load_rgb, mean_color  # noqa: E402


def expand_inputs(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        files.extend(hits if hits else [pat])
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if not os.path.isfile(f):
            print(f"[skip] not a file: {f}", file=sys.stderr)
            continue
        key = os.path.abspath(f)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compose dish photos into one table shot.")
    ap.add_argument("--input", nargs="+", required=True, help="image paths or globs")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--bg", default="wood", choices=["wood", "linen", "dark", "solid"])
    ap.add_argument("--segment", default="auto", choices=["auto", "rembg", "threshold", "center"])
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=1500)
    ap.add_argument("--main-ratio", type=float, default=0.45, help="main dish diameter / canvas width")
    ap.add_argument("--ring", type=float, default=0.42, help="orbit radius / canvas width")
    ap.add_argument("--headroom", type=float, default=0.0, help="empty band at the top (0-0.3)")
    ap.add_argument("--angle0", type=float, default=-90.0, help="first side dish angle, degrees")
    ap.add_argument("--feather", type=float, default=2.0, help="alpha feather in source pixels")
    ap.add_argument("--color-match", type=float, default=0.05, help="0 disables colour transfer")
    ap.add_argument("--shadow", type=float, default=0.45, help="contact shadow opacity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="also write a bounding-circle overlay")
    args = ap.parse_args()

    paths = expand_inputs(args.input)
    if not paths:
        print("no input images", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    background = make_background(args.width, args.height, args.bg, seed=args.seed)
    target = mean_color(background)

    items: list[Item] = []
    cutouts: list[Image.Image] = []
    report: list[dict] = []

    for path in paths:
        try:
            img = fit_long_side(load_rgb(path), 1024)
            mask, backend = segment(img, args.segment)
            plate, meta = build(
                img,
                mask,
                feather=args.feather,
                color_match=args.color_match,
                target_rgb=target,
            )
        except Exception as exc:  # keep the rest of the batch alive
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue
        items.append(Item(name=os.path.basename(path), aspect=plate.width / plate.height))
        cutouts.append(plate)
        report.append({"source": path, "segment": backend, **meta})
        print(f"[ok]   {os.path.basename(path):<28} segment={backend:<18} d={meta['diameter_px']}")

    if not items:
        print("nothing segmented", file=sys.stderr)
        return 1

    # main dish = biggest cutout in source pixels
    order = sorted(range(len(items)), key=lambda i: report[i]["diameter_px"], reverse=True)
    items = [items[i] for i in order]
    cutouts = [cutouts[i] for i in order]
    report = [report[i] for i in order]

    placements = plan(
        items,
        args.width,
        args.height,
        main_ratio=args.main_ratio,
        ring=args.ring,
        headroom=args.headroom,
        angle0=args.angle0,
        seed=args.seed,
    )
    for p in placements:
        p.image = cutouts[p.index]

    result = render(background, placements, shadow=args.shadow)
    result_path = out_dir / "result.png"
    result.save(result_path)

    data = to_dict(placements, args.width, args.height)
    data["background"] = args.bg
    data["sources"] = report
    (out_dir / "layout.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

    if args.debug:
        dbg = result.convert("RGBA")
        from PIL import ImageDraw

        draw = ImageDraw.Draw(dbg)
        for p in placements:
            r = p.size / 2
            draw.ellipse([p.cx - r, p.cy - r, p.cx + r, p.cy + r], outline=(255, 40, 40, 255), width=3)
            draw.text((p.cx - r + 6, p.cy - r + 6), f"#{p.index} d={int(p.size)}", fill=(255, 255, 0, 255))
        dbg.convert("RGB").save(out_dir / "debug.png")

    print(f"\nwrote {result_path}  ({len(placements)} plates)")
    print(f"wrote {out_dir / 'layout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
