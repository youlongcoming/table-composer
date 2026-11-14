"""Train the binary food segmenter on FoodSeg103.

    python -m src.train_seg --data data/FoodSeg103 --epochs 20 --device auto

FoodSeg103 annotates ingredients, not plates. The target here is therefore
"anything that is food" (mask > 0), and the plate is reconstructed at inference
time by ``src.seg_food.plate_from_food``.

Expected layout (that of the official release):

    data/FoodSeg103/
        images/       img_0001.jpg ...
        annotations/  ann_0001.png ...
        train.txt     (optional) one image stem per line
        test.txt      (optional)
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.unet import UNet  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def pair_files(root: Path, split_file: Path | None = None) -> list[tuple[Path, Path]]:
    img_dir = root / "images"
    ann_dir = root / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise SystemExit(f"{root} 下找不到 images/ 和 annotations/")

    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    anns = {p.stem: p for p in ann_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    if not imgs or not anns:
        raise SystemExit("images/ 或 annotations/ 是空的")

    # match by stem, then by replacing the img/ann prefix, then by digits
    ann_by_digits = {"".join(c for c in s if c.isdigit()): p for s, p in anns.items()}
    pairs: list[tuple[Path, Path]] = []
    for stem, ip in sorted(imgs.items()):
        a = anns.get(stem) or anns.get(stem.replace("img", "ann"))
        if a is None:
            a = ann_by_digits.get("".join(c for c in stem if c.isdigit()))
        if a is not None:
            pairs.append((ip, a))

    if split_file and split_file.exists():
        wanted = {ln.strip() for ln in split_file.read_text("utf-8").splitlines() if ln.strip()}
        picked = [(i, a) for i, a in pairs if i.stem in wanted]
        if picked:
            pairs = picked
    return pairs


class SegDataset(Dataset):
    def __init__(self, pairs, size: int = 384, train: bool = True):
        self.pairs = pairs
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        ip, ap = self.pairs[i]
        img = Image.open(ip).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        msk = Image.open(ap)
        if msk.mode not in ("L", "P", "I;16"):
            msk = msk.convert("L")
        msk = msk.resize((self.size, self.size), Image.Resampling.NEAREST)
        mask = (np.asarray(msk) > 0).astype(np.float32)

        x = np.asarray(img, dtype=np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                x = x[:, ::-1]
                mask = mask[:, ::-1]
            if random.random() < 0.5:
                x = np.clip(x * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05), 0, 1)

        x = (x - x.mean()) / (x.std() + 1e-6)
        xt = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float()
        mt = torch.from_numpy(np.ascontiguousarray(mask))[None].float()
        return xt, mt


# --------------------------------------------------------------------------- #
# loss / metrics
# --------------------------------------------------------------------------- #
def dice_loss(logits, target, eps: float = 1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def iou_score(logits, target, eps: float = 1e-6):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * target).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean()


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/FoodSeg103")
    ap.add_argument("--out", default="weights/seg_foodseg.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--base", type=int, default=32, help="U-Net width")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="only use N images (debug)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.data)
    pairs = pair_files(root, root / "train.txt")
    if not pairs:
        raise SystemExit("没有配对成功的图像/标注，先跑 scripts/inspect_foodseg.py 看看格式")
    if args.limit:
        pairs = pairs[: args.limit]

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"device={device}  train={len(train_pairs)}  val={len(val_pairs)}")

    dl = dict(num_workers=args.workers, pin_memory=(device == "cuda"))
    train_loader = DataLoader(SegDataset(train_pairs, args.image_size, True),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **dl)
    val_loader = DataLoader(SegDataset(val_pairs, args.image_size, False),
                            batch_size=max(1, args.batch_size // 2), shuffle=False, **dl)

    net = UNet(base=args.base).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_epoch = 0
    best = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("val_iou", -1.0))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        net.train()
        t0 = time.time()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = net(x)
                loss = 0.5 * F.binary_cross_entropy_with_logits(logits, y) + 0.5 * dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.detach())
        sched.step()

        net.eval()
        ious = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = net(x)
                ious.append(float(iou_score(logits.float(), y)))
        val_iou = float(np.mean(ious)) if ious else 0.0
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss={run_loss / max(1, len(train_loader)):.4f}  "
              f"val_iou={val_iou:.4f}  {time.time() - t0:.1f}s")

        state = {
            "model": net.state_dict(),
            "base": args.base,
            "image_size": args.image_size,
            "epoch": epoch,
            "val_iou": val_iou,
            "target": "food-binary",
        }
        if val_iou > best:
            best = val_iou
            torch.save(state, out_path)
            print(f"  saved {out_path}  (best val_iou={best:.4f})")

    torch.save(state, out_path.with_name(out_path.stem + "_last.pt"))
    print(f"done. best val_iou={best:.4f}  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
