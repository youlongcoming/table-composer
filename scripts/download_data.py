"""Fetch the demo datasets into data/.

Both datasets are research-only. This script does not redistribute them; it
downloads from the official source and unpacks locally.

    python scripts/download_data.py --dataset food101 --dest data/
    python scripts/download_data.py --dataset foodseg103
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

FOOD101_URL = "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/food-101.tar.gz"
FOODSEG103_PAGE = "https://xkyong.github.io/FoodSeg103/"
FOODSEG103_REPO = "https://github.com/LARC-CMU-SMU/FoodSeg103"


def _progress(done: int, total: int) -> None:
    if total <= 0:
        return
    pct = done * 100.0 / total
    bar = "#" * int(pct // 2)
    sys.stdout.write(f"\r  {pct:5.1f}% [{bar:<50}] {done/1e6:6.1f} MB")
    sys.stdout.flush()


def download_food101(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tgz = dest / "food-101.tar.gz"
    if not tgz.exists():
        print(f"downloading {FOOD101_URL}")
        urllib.request.urlretrieve(FOOD101_URL, tgz, _progress)  # noqa: S310
        print()
    else:
        print(f"archive exists: {tgz}")

    out = dest / "food-101"
    if out.exists():
        print(f"already extracted: {out}")
        return
    print(f"extracting to {dest}")
    with tarfile.open(tgz) as tf:
        tf.extractall(dest)  # noqa: S202
    print(f"done: {out}")


def download_foodseg103() -> None:
    print("FoodSeg103 需要手动下载（Google Drive，脚本无法直连）。\n")
    print(f"  1. 打开 {FOODSEG103_PAGE}")
    print(f"  2. 按页面说明从 {FOODSEG103_REPO} 下载 images 和 annotations")
    print("  3. 解压到 data/FoodSeg103/ 下，保持 images/ 和 annotations/ 两个目录\n")
    print("只需要分割标注时，其实可以先跳过它：rembg 零样本就能跑。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["food101", "foodseg103"])
    ap.add_argument("--dest", default="data")
    args = ap.parse_args()

    dest = Path(args.dest)
    if args.dataset == "food101":
        download_food101(dest)
    else:
        download_foodseg103()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
