"""Tests for perception visualization helpers."""

from pathlib import Path

import numpy as np

from ml.config import AI4MARS_COLOR_PALETTE, AI4MARS_IGNORE_INDEX
from perception.visualization import (
    colorize_terrain_mask,
    compute_class_distribution,
    render_prediction_comparison,
)


def test_colorize_terrain_mask_and_ignore_override() -> None:
    ids = np.array([[0, 1], [2, 3]], dtype=np.int32)
    ignore_mask = np.array([[False, False], [False, True]], dtype=bool)
    rgb = colorize_terrain_mask(ids, ignore_mask=ignore_mask)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == AI4MARS_COLOR_PALETTE[0]
    assert tuple(rgb[0, 1]) == AI4MARS_COLOR_PALETTE[1]
    assert tuple(rgb[1, 0]) == AI4MARS_COLOR_PALETTE[2]
    assert tuple(rgb[1, 1]) == AI4MARS_COLOR_PALETTE[AI4MARS_IGNORE_INDEX]


def test_compute_class_distribution_excludes_ignored() -> None:
    ids = np.array(
        [
            [0, 0, 1],
            [2, 3, AI4MARS_IGNORE_INDEX],
        ],
        dtype=np.int32,
    )
    dist = compute_class_distribution(ids)

    assert dist["soil"]["pixel_count"] == 2
    assert dist["soil"]["percentage"] == 40.0
    assert dist["bedrock"]["pixel_count"] == 1
    assert dist["sand"]["pixel_count"] == 1
    assert dist["big_rock"]["pixel_count"] == 1


def test_render_prediction_comparison_saves_png(tmp_path: Path) -> None:
    img = np.zeros((16, 20, 3), dtype=np.uint8)
    pred = np.zeros((16, 20), dtype=np.int32)
    gt = np.full((16, 20), AI4MARS_IGNORE_INDEX, dtype=np.int32)
    gt[:8, :] = 1
    conf = np.full((16, 20), 0.85, dtype=np.float32)
    unc = 1.0 - conf

    out_png = tmp_path / "compare.png"
    render_prediction_comparison(
        image_rgb=img,
        terrain_class_ids=pred,
        confidence=conf,
        uncertainty_score=unc,
        ground_truth_ids=gt,
        save_path=out_png,
        show=False,
    )
    assert out_png.is_file() and out_png.stat().st_size > 0
