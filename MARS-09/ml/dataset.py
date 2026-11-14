"""AI4MARS RGB mask decoding and synchronized PyTorch Dataset implementation.

Mandatory rules enforced here:
- Segmentation masks are NEVER converted to grayscale.
- Exact RGB triples are decoded:
  (0,0,0)->0 (soil), (1,1,1)->1 (bedrock), (2,2,2)->2 (sand),
  (3,3,3)->3 (big_rock), (255,255,255)->255 (unlabelled).
- Unexpected RGB colors are counted/reported and mapped to ignore index 255.
- Masks are always resized using nearest-neighbor interpolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ml.config import (
    AI4MARS_IGNORE_INDEX,
    AI4MARS_RGB_TO_ID,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def decode_ai4mars_rgb_mask(
    mask_rgb: np.ndarray | Image.Image,
) -> tuple[np.ndarray, dict[tuple[int, int, int], int]]:
    """Decode an RGB AI4MARS label mask into integer class IDs without grayscale conversion.

    Args:
        mask_rgb: 3-channel RGB array of shape (H, W, 3) or RGB PIL Image.

    Returns:
        Tuple of:
        - id_mask: 2D uint8 array of shape (H, W) with values in {0, 1, 2, 3, 255}.
        - unexpected_colors: Dict mapping unexpected (R, G, B) triples to pixel counts.

    Raises:
        ValueError: If input is not a 3-channel RGB image/array (grayscale is rejected).
    """
    if isinstance(mask_rgb, Image.Image):
        if mask_rgb.mode != "RGB":
            raise ValueError(
                f"AI4MARS mask must be RGB mode, got '{mask_rgb.mode}'. "
                "Do not convert segmentation masks to grayscale."
            )
        arr = np.asarray(mask_rgb, dtype=np.uint8)
    else:
        arr = np.asarray(mask_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"AI4MARS mask must have shape (H, W, 3) in RGB format; got shape {arr.shape}. "
                "Grayscale masks are not permitted."
            )
        arr = arr.astype(np.uint8, copy=False)

    h, w, _ = arr.shape
    id_mask = np.full((h, w), AI4MARS_IGNORE_INDEX, dtype=np.uint8)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    known_mask = np.zeros((h, w), dtype=bool)
    for (kr, kg, kb), class_id in AI4MARS_RGB_TO_ID.items():
        match = (r == kr) & (g == kg) & (b == kb)
        id_mask[match] = np.uint8(class_id)
        known_mask |= match

    unexpected_colors: dict[tuple[int, int, int], int] = {}
    if not np.all(known_mask):
        unexpected_pixels = arr[~known_mask]
        unique_rgb, counts = np.unique(unexpected_pixels, axis=0, return_counts=True)
        for rgb_row, cnt in zip(unique_rgb, counts):
            key = (int(rgb_row[0]), int(rgb_row[1]), int(rgb_row[2]))
            unexpected_colors[key] = int(cnt)

    return id_mask, unexpected_colors


def normalize_rgb_image(
    image_rgb: np.ndarray,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Convert an (H, W, 3) uint8/float RGB array into a normalized (3, H, W) float32 tensor."""
    arr = np.asarray(image_rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB array with shape (H, W, 3); got {arr.shape}.")
    if arr.max() > 1.0 or arr.dtype == np.uint8:
        arr = arr / 255.0

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (arr - mean_arr) / std_arr
    chw = np.transpose(normalized, (2, 0, 1)).copy()
    return torch.from_numpy(chw)


class AI4MARSSegmentationDataset(Dataset):
    """PyTorch Dataset for AI4MARS paired RGB images and RGB-encoded segmentation masks."""

    def __init__(
        self,
        samples: Sequence[dict[str, str]],
        image_size: tuple[int, int] = (256, 256),
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize dataset from list of {'image_path': ..., 'mask_path': ...} dicts."""
        self.samples = list(samples)
        if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError(f"image_size must be positive (height, width); got {image_size}.")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.samples[index]
        img_path = Path(entry["image_path"])
        mask_path = Path(entry["mask_path"])

        with Image.open(img_path) as img_raw:
            image_pil = img_raw.convert("RGB")
        with Image.open(mask_path) as mask_raw:
            if mask_raw.mode != "RGB":
                raise ValueError(
                    f"Mask '{mask_path}' has mode '{mask_raw.mode}', expected 'RGB'. "
                    "Do not store or convert AI4MARS masks as grayscale."
                )
            mask_pil = mask_raw.copy()

        target_h, target_w = self.image_size
        # PIL resize expects (width, height)
        image_resized = image_pil.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
        # Mandatory nearest-neighbor interpolation for segmentation masks
        mask_resized = mask_pil.resize((target_w, target_h), resample=Image.Resampling.NEAREST)

        img_np = np.asarray(image_resized, dtype=np.uint8)
        mask_rgb_np = np.asarray(mask_resized, dtype=np.uint8)
        mask_ids, _ = decode_ai4mars_rgb_mask(mask_rgb_np)

        if self.augment:
            # Synchronized horizontal flip
            if self.rng.random() < 0.5:
                img_np = np.ascontiguousarray(img_np[:, ::-1, :])
                mask_ids = np.ascontiguousarray(mask_ids[:, ::-1])

            # Mild brightness/contrast jitter applied ONLY to RGB image (never mask)
            if self.rng.random() < 0.5:
                alpha = float(self.rng.uniform(0.90, 1.10))
                beta = float(self.rng.uniform(-12.0, 12.0))
                img_np = np.clip(img_np.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)

        image_tensor = normalize_rgb_image(img_np)
        mask_tensor = torch.from_numpy(mask_ids.astype(np.int64, copy=True))
        return image_tensor, mask_tensor
