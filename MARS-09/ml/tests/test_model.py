"""Tests for LightweightUNet, checkpoints, perception API, and Member 1 integration."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.config import (
    AI4MARS_CLASS_NAMES,
    AI4MARS_NUM_CLASSES,
    create_ai4mars_terrain_class_config,
)
from ml.model import LightweightUNet, count_parameters, load_checkpoint, save_checkpoint
from perception.terrain_inference import (
    SpatialAlignmentError,
    TerrainInferenceEngine,
    clear_inference_cache,
    predict_terrain,
    predict_terrain_on_dem_grid,
)
from terrain.terrain_cost import compute_terrain_cost_map


def test_unet_output_shape_and_parameter_count() -> None:
    model = LightweightUNet(in_channels=3, num_classes=AI4MARS_NUM_CLASSES, base_channels=8)
    assert count_parameters(model) > 0

    x = torch.randn(2, 3, 64, 80)
    logits = model(x)
    assert logits.shape == (2, AI4MARS_NUM_CLASSES, 64, 80)


def test_checkpoint_roundtrip_and_predict_terrain_schema(tmp_path: Path) -> None:
    clear_inference_cache()
    model = LightweightUNet(in_channels=3, num_classes=AI4MARS_NUM_CLASSES, base_channels=8)
    ckpt_path = tmp_path / "test_unet.pt"
    save_checkpoint(ckpt_path, model=model, epoch=1, best_val_miou=0.5, image_size=(64, 64))

    loaded_model, payload = load_checkpoint(ckpt_path, device="cpu")
    assert payload["epoch"] == 1
    assert loaded_model.num_classes == 4

    # Non-square native rover image resolution (e.g., 50x70) must be preserved
    rgb = np.random.default_rng(7).integers(0, 256, size=(50, 70, 3), dtype=np.uint8)
    result = predict_terrain(rgb, checkpoint_path=ckpt_path, device="cpu")

    assert set(result.keys()) == {
        "terrain_class_ids",
        "class_probabilities",
        "confidence",
        "uncertainty_score",
        "color_mask",
        "class_names",
    }

    ids = result["terrain_class_ids"]
    probs = result["class_probabilities"]
    conf = result["confidence"]
    unc = result["uncertainty_score"]
    color = result["color_mask"]

    assert ids.shape == (50, 70)
    assert ids.dtype == np.int32
    assert set(np.unique(ids).tolist()).issubset({0, 1, 2, 3})

    assert probs.shape == (50, 70, 4)
    assert np.issubdtype(probs.dtype, np.floating)
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(np.sum(probs, axis=2), 1.0, atol=1e-5)

    assert conf.shape == (50, 70)
    assert unc.shape == (50, 70)
    assert np.allclose(unc, 1.0 - conf, atol=1e-5)
    assert color.shape == (50, 70, 3)
    assert color.dtype == np.uint8
    assert result["class_names"] == AI4MARS_CLASS_NAMES


def test_dem_grid_spatial_alignment_validation_and_member1_cost_integration(tmp_path: Path) -> None:
    model = LightweightUNet(in_channels=3, num_classes=AI4MARS_NUM_CLASSES, base_channels=8)
    ckpt_path = tmp_path / "test_unet.pt"
    save_checkpoint(ckpt_path, model=model, epoch=1, best_val_miou=0.5, image_size=(32, 32))
    engine = TerrainInferenceEngine.from_checkpoint(ckpt_path, device="cpu")

    rgb_unaligned = np.zeros((48, 64, 3), dtype=np.uint8)

    # Mode B MUST reject unregistered rover images
    with pytest.raises(SpatialAlignmentError, match="orthorectified"):
        predict_terrain_on_dem_grid(
            rgb_unaligned,
            dem_shape=(20, 20),
            is_orthorectified_to_dem=False,
            engine=engine,
        )

    # Mode B MUST reject mismatched shape even if marked orthorectified
    with pytest.raises(SpatialAlignmentError, match="does not match DEM grid shape"):
        predict_terrain_on_dem_grid(
            rgb_unaligned,
            dem_shape=(20, 20),
            is_orthorectified_to_dem=True,
            engine=engine,
        )

    # Mode B succeeds when orthorectified AND shape == dem_shape, and feeds Member 1
    dem_shape = (24, 30)
    ortho_rgb = np.full((24, 30, 3), 100, dtype=np.uint8)
    pred = predict_terrain_on_dem_grid(
        ortho_rgb,
        dem_shape=dem_shape,
        is_orthorectified_to_dem=True,
        engine=engine,
    )

    slope_deg = np.full(dem_shape, 5.0, dtype=np.float64)
    valid_mask = np.ones(dem_shape, dtype=bool)
    class_cfg = create_ai4mars_terrain_class_config()

    cost_map = compute_terrain_cost_map(
        slope_deg=slope_deg,
        valid_mask=valid_mask,
        resolution_m=(1.0, 1.0),
        terrain_class_ids=pred["terrain_class_ids"],
        class_probabilities=pred["class_probabilities"],
        class_config=class_cfg,
    )

    assert cost_map.total_cost.shape == dem_shape
    assert np.all(np.isfinite(cost_map.total_cost))
    assert np.allclose(cost_map.uncertainty_risk, pred["uncertainty_score"], atol=1e-5)
