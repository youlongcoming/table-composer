"""Print what a FoodSeg103 checkout actually looks like, before training.

    python scripts/inspect_foodseg.py --data data/FoodSeg103

Answers three questions the training script cannot guess: are images and masks
paired, what dtype/range do the masks use, and how many classes are present.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/FoodSeg103")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.data)
    print(f"root: {root.resolve()}")
    if not root.exists():
        print("路径不存在。先按 README 的说明下载 FoodSeg103。")
        return 2

    for sub in ("images", "annotations"):
        d = root / sub
        n = len([p for p in d.iterdir() if p.suffix.lower() in EXT]) if d.is_dir() else 0
        print(f"  {sub:12} {'missing' if not d.is_dir() else f'{n} files'}")

    for f in ("train.txt", "test.txt", "category_id.txt"):
        p = root / f
        print(f"  {f:12} {'missing' if not p.exists() else f'{p.stat().st_size} bytes'}")

    imgs = {p.stem for p in (root / "images").iterdir() if p.suffix.lower() in EXT} if (root / "images").is_dir() else set()
    anns = {p.stem for p in (root / "annotations").iterdir() if p.suffix.lower() in EXT} if (root / "annotations").is_dir() else set()
    print(f"\n唯一 images {len(imgs)}, annotations {len(anns)}, 同名配对 {len(imgs & anns)}")

    if not anns:
        return 1

    sample = sorted(anns)[: args.samples]
    for stem in sample:
        p = root / "annotations" / f"{stem}.png"
        if not p.exists():
            hits = list((root / "annotations").glob(f"{stem}.*"))
            if not hits:
                continue
            p = hits[0]
        m = Image.open(p)
        arr = np.asarray(m)
        uniq = np.unique(arr)
        print(f"\n{stem}: mode={m.mode} shape={arr.shape} dtype={arr.dtype}")
        print(f"  values: min={arr.min()} max={arr.max()} unique={len(uniq)} 前十个={uniq[:10].tolist()}")
        print(f"  前景占比: {(arr > 0).mean():.3f}")

    # how often does a mask have actual annotations?
    empty = 0
    checked = 0
    for p in list((root / "annotations").iterdir())[:200]:
        if p.suffix.lower() not in EXT:
            continue
        a = np.asarray(Image.open(p).convert("L"))
        checked += 1
        if not (a > 0).any():
            empty += 1
    if checked:
        print(f"\n抽查 {checked} 张，全空标注 {empty} 张")

    cats = root / "category_id.txt"
    if cats.exists():
        names = [ln.strip() for ln in cats.read_text("utf-8", errors="ignore").splitlines() if ln.strip()]
        print(f"\ncategory_id.txt 共 {len(names)} 行，示例: {names[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
