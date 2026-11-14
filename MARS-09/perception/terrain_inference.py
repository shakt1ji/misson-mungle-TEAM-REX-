"""Member 1-compatible terrain semantic segmentation inference API (Member 2).

Supports two strictly separated operating modes:
- Mode A (Image-level inference): `predict_terrain(image, checkpoint_path=...)`
  Preserves the input rover image's native (H, W) resolution and returns
  channel-last `(H, W, 4)` probabilities + `(H, W)` `np.int32` class IDs.
- Mode B (DEM-grid inference): `predict_terrain_on_dem_grid(image, dem_shape, is_orthorectified_to_dem=...)`
  Strictly validates that the image is explicitly flagged as orthorectified/registered
  to the DEM grid AND that `(H, W) == dem_shape`. Never silently resizes an
  unregistered camera image onto a DEM grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ml.config import (
    AI4MARS_CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_palette_lut,
)
from ml.dataset import normalize_rgb_image
from ml.model import LightweightUNet, load_checkpoint


class SpatialAlignmentError(ValueError):
    """Raised when an image is not spatially aligned/registered with a target DEM grid."""


def _coerce_rgb_numpy(image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Validate and convert input image path, PIL Image, or RGB array to (H, W, 3) uint8."""
    if isinstance(image, (str, Path)):
        img_path = Path(image)
        if not img_path.is_file():
            raise FileNotFoundError(f"Input image not found: '{img_path}'.")
        with Image.open(img_path) as pil_img:
            arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image array with shape (H, W, 3); got shape {arr.shape}."
            )
        if arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError(f"Input image must have positive height and width; got {arr.shape}.")
        if np.issubdtype(arr.dtype, np.floating):
            if not np.all(np.isfinite(arr)):
                raise ValueError("Input floating-point image contains NaN or Inf values.")
            scale = 255.0 if float(np.max(arr)) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise TypeError(
            f"Unsupported image type '{type(image).__name__}'. "
            "Expected str, Path, PIL.Image.Image, or (H, W, 3) numpy.ndarray."
        )

    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"Invalid RGB image shape: {arr.shape}.")
    return arr


class TerrainInferenceEngine:
    """Reusable inference engine that keeps a trained LightweightUNet loaded in memory."""

    def __init__(
        self,
        model: LightweightUNet,
        device: torch.device,
        image_size: tuple[int, int] = (256, 256),
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self._palette_lut = build_palette_lut()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> TerrainInferenceEngine:
        """Load a TerrainInferenceEngine from a saved checkpoint file."""
        ckpt = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
        model, payload = load_checkpoint(ckpt, device=device)
        norm = payload.get("normalization", {})
        mean = tuple(norm.get("mean", IMAGENET_MEAN))
        std = tuple(norm.get("std", IMAGENET_STD))
        img_sz = tuple(payload.get("image_size", (256, 256)))
        target_device = next(model.parameters()).device
        return cls(
            model=model,
            device=target_device,
            image_size=(int(img_sz[0]), int(img_sz[1])),
            mean=mean,  # type: ignore[arg-type]
            std=std,    # type: ignore[arg-type]
        )

    @torch.inference_mode()
    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray,
    ) -> dict[str, Any]:
        """Run Mode A image-level terrain segmentation preserving native (H, W) resolution."""
        rgb = _coerce_rgb_numpy(image)
        orig_h, orig_w, _ = rgb.shape

        model_h, model_w = self.image_size
        if (orig_h, orig_w) != (model_h, model_w):
            pil_resized = Image.fromarray(rgb, mode="RGB").resize(
                (model_w, model_h), resample=Image.Resampling.BILINEAR
            )
            infer_rgb = np.asarray(pil_resized, dtype=np.uint8)
        else:
            infer_rgb = rgb

        x = normalize_rgb_image(infer_rgb, mean=self.mean, std=self.std).unsqueeze(0).to(self.device)
        logits = self.model(x)  # (1, 4, model_h, model_w)

        if logits.shape[2:] != (orig_h, orig_w):
            logits = F.interpolate(
                logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )

        probs_chw = torch.softmax(logits, dim=1).squeeze(0)  # (4, H, W)
        # Convert channel-first PyTorch tensor -> channel-last (H, W, 4) NumPy array
        probs_hwc = probs_chw.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

        # Guarantee strict finite normalization for Member 1 compatibility
        probs_hwc = np.clip(probs_hwc, 0.0, 1.0)
        row_sums = np.sum(probs_hwc, axis=2, keepdims=True)
        probs_hwc = np.where(row_sums > 0.0, probs_hwc / np.maximum(row_sums, 1e-8), 0.25).astype(
            np.float32
        )

        terrain_class_ids = np.argmax(probs_hwc, axis=2).astype(np.int32)
        confidence = np.max(probs_hwc, axis=2).astype(np.float32)
        uncertainty_score = np.clip(1.0 - confidence, 0.0, 1.0).astype(np.float32)
        color_mask = self._palette_lut[terrain_class_ids]

        return {
            "terrain_class_ids": terrain_class_ids,
            "class_probabilities": probs_hwc,
            "confidence": confidence,
            "uncertainty_score": uncertainty_score,
            "color_mask": color_mask,
            "class_names": dict(AI4MARS_CLASS_NAMES),
        }


_ENGINE_CACHE: dict[tuple[str, str], TerrainInferenceEngine] = {}


def clear_inference_cache() -> None:
    """Clear cached checkpoint model instances from memory."""
    _ENGINE_CACHE.clear()


def predict_terrain(
    image: str | Path | Image.Image | np.ndarray,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    engine: TerrainInferenceEngine | None = None,
) -> dict[str, Any]:
    """Predict pixel-wise Mars terrain classes and probabilities at original image resolution.

    Args:
        image: Image file path, PIL Image, or RGB NumPy array of shape (H, W, 3).
        checkpoint_path: Path to trained LightweightUNet checkpoint (.pt).
            Defaults to `models/ai4mars_unet_best.pt`.
        device: Optional torch device string ('cpu' or 'cuda').
        engine: Optional preloaded `TerrainInferenceEngine` to avoid reloading weights.

    Returns:
        Dictionary with exact Member 1-compatible fields:
        - terrain_class_ids: (H, W) np.int32 in {0, 1, 2, 3}
        - class_probabilities: (H, W, 4) np.float32 in [soil, bedrock, sand, big_rock] order
        - confidence: (H, W) np.float32 max class probability
        - uncertainty_score: (H, W) np.float32 (1 - confidence)
        - color_mask: (H, W, 3) np.uint8 RGB visualization mask
        - class_names: dict {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
    """
    if engine is not None:
        return engine.predict(image)

    ckpt = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
    resolved_ckpt = str(ckpt.resolve())
    target_device = (
        str(torch.device(device))
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache_key = (resolved_ckpt, target_device)

    cached_engine = _ENGINE_CACHE.get(cache_key)
    if cached_engine is None:
        cached_engine = TerrainInferenceEngine.from_checkpoint(
            checkpoint_path=ckpt, device=target_device
        )
        _ENGINE_CACHE[cache_key] = cached_engine

    return cached_engine.predict(image)


def predict_terrain_on_dem_grid(
    image: str | Path | Image.Image | np.ndarray,
    dem_shape: tuple[int, int],
    is_orthorectified_to_dem: bool = False,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    engine: TerrainInferenceEngine | None = None,
) -> dict[str, Any]:
    """Mode B: Run terrain inference ONLY when image is verified aligned to the DEM grid.

    Never resizes an unregistered rover camera image to match a DEM grid.

    Raises:
        SpatialAlignmentError: If `is_orthorectified_to_dem` is False or if
            image `(H, W)` does not exactly match `dem_shape` `(rows, cols)`.
    """
    if not is_orthorectified_to_dem:
        raise SpatialAlignmentError(
            "Direct ML-to-DEM fusion is unavailable: input image is not marked as "
            "orthorectified/registered to the DEM grid (is_orthorectified_to_dem=False). "
            "Use Mode A (`predict_terrain`) for image-level evaluation instead of "
            "resizing an unregistered rover camera image onto a DEM."
        )

    if len(dem_shape) != 2 or dem_shape[0] < 1 or dem_shape[1] < 1:
        raise ValueError(f"dem_shape must be (rows, cols) with positive dims; got {dem_shape}.")

    rgb = _coerce_rgb_numpy(image)
    img_hw = (rgb.shape[0], rgb.shape[1])
    expected_hw = (int(dem_shape[0]), int(dem_shape[1]))

    if img_hw != expected_hw:
        raise SpatialAlignmentError(
            f"Orthorectified image spatial shape {img_hw} does not match DEM grid shape "
            f"{expected_hw}. Refusing to silently resize or warp image."
        )

    return predict_terrain(
        rgb,
        checkpoint_path=checkpoint_path,
        device=device,
        engine=engine,
    )
