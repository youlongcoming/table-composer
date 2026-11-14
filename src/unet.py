"""A compact U-Net for binary plate/food segmentation.

Self-contained: only depends on torch, no torchvision weights needed, so the
checkpoint is small and inference stays cheap.

    net = UNet(base=32)          # ~7.8M params
    mask = net(x)                # logits, (N, 1, H, W)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        b = base
        self.d1 = DoubleConv(in_ch, b)
        self.d2 = DoubleConv(b, b * 2)
        self.d3 = DoubleConv(b * 2, b * 4)
        self.d4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)
        self.u4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.c4 = DoubleConv(b * 16, b * 8)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.c3 = DoubleConv(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.c2 = DoubleConv(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.c1 = DoubleConv(b * 2, b)
        self.head = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        e1 = self.d1(x)
        e2 = self.d2(self.pool(e1))
        e3 = self.d3(self.pool(e2))
        e4 = self.d4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        x = self.c4(torch.cat([self.u4(b), e4], dim=1))
        x = self.c3(torch.cat([self.u3(x), e3], dim=1))
        x = self.c2(torch.cat([self.u2(x), e2], dim=1))
        x = self.c1(torch.cat([self.u1(x), e1], dim=1))
        return self.head(x)
