"""Lightweight PyTorch U-Net for 4-class Mars terrain semantic segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import (
    AI4MARS_CLASS_NAMES,
    AI4MARS_CLASS_ORDER,
    AI4MARS_NUM_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class DoubleConv(nn.Module):
    """Two 3x3 Conv2d + BatchNorm2d + ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """Encoder downsampling block with MaxPool2d followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Decoder upsampling block with bilinear upsampling, skip concatenation, and DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        merged = torch.cat([skip, x_up], dim=1)
        return self.conv(merged)


class LightweightUNet(nn.Module):
    """Compact 4-level U-Net predicting [soil, bedrock, sand, big_rock] logits.

    Input shape:  (batch, 3, H, W)
    Output shape: (batch, num_classes=4, H, W) raw unnormalized logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = AI4MARS_NUM_CLASSES,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2; got {num_classes}.")
        if base_channels < 4:
            raise ValueError(f"base_channels must be >= 4; got {base_channels}.")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)

        c1 = self.base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8

        self.inc = DoubleConv(self.in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)

        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        bottleneck = self.down3(s3)

        d3 = self.up3(bottleneck, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        return self.head(d1)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return total number of (trainable) parameters in a PyTorch module."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    path: str | Path,
    model: LightweightUNet,
    epoch: int,
    best_val_miou: float,
    image_size: tuple[int, int] = (256, 256),
    history: dict[str, Any] | None = None,
) -> Path:
    """Save a LightweightUNet checkpoint with architecture and class metadata."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_config": {
            "in_channels": model.in_channels,
            "num_classes": model.num_classes,
            "base_channels": model.base_channels,
        },
        "class_order": list(AI4MARS_CLASS_ORDER),
        "class_names": dict(AI4MARS_CLASS_NAMES),
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "image_size": list(image_size),
        "epoch": int(epoch),
        "best_val_miou": float(best_val_miou),
        "history": history or {},
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[LightweightUNet, dict[str, Any]]:
    """Load a LightweightUNet checkpoint onto `device` and set eval mode."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: '{ckpt_path}'.")

    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    payload = torch.load(ckpt_path, map_location=target_device, weights_only=False)
    cfg = payload.get("model_config", {})
    model = LightweightUNet(
        in_channels=int(cfg.get("in_channels", 3)),
        num_classes=int(cfg.get("num_classes", AI4MARS_NUM_CLASSES)),
        base_channels=int(cfg.get("base_channels", 16)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(target_device)
    model.eval()
    return model, payload
