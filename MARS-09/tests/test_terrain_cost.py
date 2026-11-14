"""Tests for terrain.traversability and terrain.terrain_cost."""

import numpy as np
import pytest

from terrain.terrain_cost import CostWeights, compute_terrain_cost_map
from terrain.traversability import (
    TerrainClassConfig,
    TraversabilityConfig,
    compute_prediction_uncertainty,
)


def test_high_slope_increases_risk_and_blocks_above_threshold() -> None:
    slope_deg = np.array(
        [
            [0.0, 10.0, 20.0],
            [29.0, 35.0, 50.0],
        ],
        dtype=np.float64,
    )
    valid_mask = np.ones_like(slope_deg, dtype=bool)
    trav_cfg = TraversabilityConfig(max_safe_slope_deg=30.0)

    cost_map = compute_terrain_cost_map(
        slope_deg=slope_deg,
        valid_mask=valid_mask,
        traversability_config=trav_cfg,
        obstacle_buffer_cells=0,
    )

    assert cost_map.slope_risk[0, 0] < cost_map.slope_risk[0, 1] < cost_map.slope_risk[0, 2] < cost_map.slope_risk[1, 0]
    assert cost_map.traversable_mask[1, 0] is np.bool_(True)
    assert cost_map.blocked_mask[1, 1] is np.bool_(True)
    assert cost_map.blocked_mask[1, 2] is np.bool_(True)
    assert np.isinf(cost_map.total_cost[1, 1])


def test_uncertain_ml_predictions_increase_cost_without_blocking() -> None:
    slope_deg = np.zeros((2, 2), dtype=np.float64)
    valid_mask = np.ones((2, 2), dtype=bool)
    # Cell (0,0) high confidence; cell (0,1) uniform uncertainty across 3 classes
    probs = np.array(
        [
            [[0.95, 0.03, 0.02], [0.34, 0.33, 0.33]],
            [[0.90, 0.05, 0.05], [0.50, 0.30, 0.20]],
        ],
        dtype=np.float64,
    )

    cost_map = compute_terrain_cost_map(
        slope_deg=slope_deg,
        valid_mask=valid_mask,
        class_probabilities=probs,
        weights=CostWeights(wd=1.0, ws=0.0, wt=0.0, wo=0.0, wu=2.0),
        obstacle_buffer_cells=0,
    )

    # Uncertain cell remains traversable, not automatically blocked
    assert cost_map.traversable_mask[0, 1] is np.bool_(True)
    assert cost_map.blocked_mask[0, 1] is np.bool_(False)
    # Higher uncertainty yields strictly higher uncertainty_risk and total_cost
    assert cost_map.uncertainty_risk[0, 1] > cost_map.uncertainty_risk[0, 0]
    assert cost_map.total_cost[0, 1] > cost_map.total_cost[0, 0]


def test_missing_values_never_become_free_traversable_cells() -> None:
    slope_deg = np.array([[0.0, np.nan], [5.0, 5.0]], dtype=np.float64)
    valid_mask = np.array([[True, False], [True, True]], dtype=bool)

    cost_map = compute_terrain_cost_map(slope_deg=slope_deg, valid_mask=valid_mask)

    assert cost_map.valid_mask[0, 1] is np.bool_(False)
    assert cost_map.traversable_mask[0, 1] is np.bool_(False)
    assert cost_map.blocked_mask[0, 1] is np.bool_(True)
    assert np.isinf(cost_map.total_cost[0, 1])


def test_configurable_terrain_class_penalties_and_impassable_classes() -> None:
    slope_deg = np.zeros((2, 3), dtype=np.float64)
    valid_mask = np.ones((2, 3), dtype=bool)
    class_ids = np.array([[0, 1, 2], [0, 0, 99]], dtype=np.int32)

    class_cfg = TerrainClassConfig(
        class_names={0: "bedrock", 1: "loose_regolith", 2: "cliff"},
        class_penalties={0: 0.0, 1: 2.5, 2: 10.0},
        impassable_class_ids={2},
        default_unknown_class_penalty=1.0,
    )

    cost_map = compute_terrain_cost_map(
        slope_deg=slope_deg,
        valid_mask=valid_mask,
        terrain_class_ids=class_ids,
        class_config=class_cfg,
        weights=CostWeights(wd=1.0, ws=0.0, wt=1.0, wo=0.0, wu=0.0),
        obstacle_buffer_cells=0,
    )

    assert cost_map.terrain_risk[0, 0] == pytest.approx(0.0)
    assert cost_map.terrain_risk[0, 1] == pytest.approx(2.5)
    assert cost_map.terrain_risk[1, 2] == pytest.approx(1.0)  # unknown class fallback
    assert cost_map.blocked_mask[0, 2] is np.bool_(True)
    assert np.isinf(cost_map.total_cost[0, 2])


def test_entropy_uncertainty_validation() -> None:
    probs = np.array([[[1.0, 0.0], [0.5, 0.5]]], dtype=np.float64)
    unc = compute_prediction_uncertainty(probs, method="entropy")
    assert unc[0, 0] == pytest.approx(0.0)
    assert unc[0, 1] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="3D"):
        compute_prediction_uncertainty(np.ones((2, 2)))
