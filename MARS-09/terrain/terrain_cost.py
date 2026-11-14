"""Weighted terrain mobility cost model for autonomous rover navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from terrain.traversability import (
    TerrainClassConfig,
    TraversabilityConfig,
    evaluate_traversability,
)


@dataclass(frozen=True)
class CostWeights:
    """Nonnegative weights for the MARS-09 terrain cost model."""

    wd: float = 1.0
    ws: float = 1.0
    wt: float = 1.0
    wo: float = 5.0
    wu: float = 1.5

    def __post_init__(self) -> None:
        for name, val in (
            ("wd", self.wd),
            ("ws", self.ws),
            ("wt", self.wt),
            ("wo", self.wo),
            ("wu", self.wu),
        ):
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(f"Weight '{name}' must be finite and nonnegative; got {val}.")
        if self.wd <= 0.0:
            raise ValueError("Weight 'wd' must be strictly positive (> 0.0).")


@dataclass(frozen=True)
class TerrainCostMap:
    """Composite terrain cost grid and individual diagnostic risk layers."""

    total_cost: np.ndarray
    base_distance_cost: np.ndarray
    slope_risk: np.ndarray
    terrain_risk: np.ndarray
    obstacle_risk: np.ndarray
    uncertainty_risk: np.ndarray
    blocked_mask: np.ndarray
    traversable_mask: np.ndarray
    valid_mask: np.ndarray
    weights: CostWeights
    resolution_m: tuple[float, float]


def _compute_terrain_class_risk(
    shape: tuple[int, int],
    terrain_class_ids: np.ndarray | None,
    class_config: TerrainClassConfig,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Map 2D integer terrain class IDs to nonnegative risk penalties."""
    risk = np.zeros(shape, dtype=np.float64)
    if terrain_class_ids is None:
        return np.where(valid_mask, risk, np.inf)

    cls_arr = np.asarray(terrain_class_ids)
    risk.fill(float(class_config.default_unknown_class_penalty))

    for class_id, penalty in class_config.class_penalties.items():
        risk[cls_arr == int(class_id)] = float(penalty)

    for impassable_id in class_config.impassable_class_ids:
        risk[cls_arr == int(impassable_id)] = np.inf

    return np.where(valid_mask, risk, np.inf)


def _compute_obstacle_proximity_risk(
    blocked_mask: np.ndarray,
    obstacle_buffer_cells: int,
) -> np.ndarray:
    """Compute nonnegative obstacle and proximity clearance risk."""
    if obstacle_buffer_cells < 0:
        raise ValueError("obstacle_buffer_cells must be >= 0.")

    risk = np.zeros(blocked_mask.shape, dtype=np.float64)
    if not np.any(blocked_mask):
        return risk

    if obstacle_buffer_cells > 0:
        dist_to_blocked = distance_transform_edt(~blocked_mask)
        within_buffer = (~blocked_mask) & (dist_to_blocked <= float(obstacle_buffer_cells))
        risk[within_buffer] = 1.0 / (dist_to_blocked[within_buffer] + 1.0)

    risk[blocked_mask] = np.inf
    return risk


def compute_terrain_cost_map(
    slope_deg: np.ndarray,
    valid_mask: np.ndarray,
    resolution_m: tuple[float, float] = (1.0, 1.0),
    terrain_class_ids: np.ndarray | None = None,
    class_probabilities: Any | None = None,
    obstacle_mask: np.ndarray | None = None,
    class_config: TerrainClassConfig | None = None,
    traversability_config: TraversabilityConfig | None = None,
    weights: CostWeights | None = None,
    obstacle_buffer_cells: int = 1,
) -> TerrainCostMap:
    """Compute the composite weighted terrain cost map and individual risk layers."""
    if not isinstance(resolution_m, (tuple, list)) or len(resolution_m) != 2:
        raise ValueError("resolution_m must be a 2-tuple of (row_spacing_m, col_spacing_m).")
    dy_m, dx_m = float(resolution_m[0]), float(resolution_m[1])
    if not (np.isfinite(dy_m) and np.isfinite(dx_m)) or dy_m <= 0.0 or dx_m <= 0.0:
        raise ValueError(f"resolution_m must be positive and finite; got {resolution_m}.")

    cls_cfg = class_config or TerrainClassConfig()
    trav_cfg = traversability_config or TraversabilityConfig()
    w = weights or CostWeights()

    trav = evaluate_traversability(
        slope_deg=slope_deg,
        valid_mask=valid_mask,
        terrain_class_ids=terrain_class_ids,
        class_probabilities=class_probabilities,
        obstacle_mask=obstacle_mask,
        class_config=cls_cfg,
        traversability_config=trav_cfg,
    )

    slope_arr = np.asarray(slope_deg, dtype=np.float64)
    shape = slope_arr.shape

    base_distance_cost = np.where(trav.traversable_mask, 1.0, np.inf)

    slope_risk = np.full(shape, np.inf, dtype=np.float64)
    normalized_slope = np.zeros(shape, dtype=np.float64)
    normalized_slope[trav.valid_mask] = slope_arr[trav.valid_mask] / trav_cfg.max_safe_slope_deg
    slope_risk[trav.traversable_mask] = np.square(normalized_slope[trav.traversable_mask])

    terrain_risk = _compute_terrain_class_risk(
        shape=shape,
        terrain_class_ids=terrain_class_ids,
        class_config=cls_cfg,
        valid_mask=trav.valid_mask,
    )
    terrain_risk = np.where(trav.traversable_mask, terrain_risk, np.inf)

    obstacle_risk = _compute_obstacle_proximity_risk(
        blocked_mask=trav.blocked_mask,
        obstacle_buffer_cells=obstacle_buffer_cells,
    )

    uncertainty_risk = np.where(trav.traversable_mask, trav.uncertainty_score, np.inf)

    total_cost = np.full(shape, np.inf, dtype=np.float64)
    traversable = trav.traversable_mask
    total_cost[traversable] = (
        w.wd * base_distance_cost[traversable]
        + w.ws * slope_risk[traversable]
        + w.wt * terrain_risk[traversable]
        + w.wo * obstacle_risk[traversable]
        + w.wu * uncertainty_risk[traversable]
    )

    return TerrainCostMap(
        total_cost=total_cost,
        base_distance_cost=base_distance_cost,
        slope_risk=slope_risk,
        terrain_risk=terrain_risk,
        obstacle_risk=obstacle_risk,
        uncertainty_risk=uncertainty_risk,
        blocked_mask=trav.blocked_mask,
        traversable_mask=trav.traversable_mask,
        valid_mask=trav.valid_mask,
        weights=w,
        resolution_m=(dy_m, dx_m),
    )
