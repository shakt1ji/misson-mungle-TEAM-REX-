"""Terrain analysis helper service for the MARS-09 dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np

from simulation.simulation_controller import RoverSimulationController
from simulation.simulation_state import _compute_uncertainty_grid


CLASS_LABELS: dict[int, str] = {
    0: "Soil",
    1: "Bedrock",
    2: "Sand",
    3: "Big Rock",
    255: "Unknown",
}


def summarize_terrain(controller: RoverSimulationController) -> dict[str, Any]:
    """Compute elevation, slope, cost, uncertainty, and terrain class distribution stats."""
    state = controller.state
    elev = state.dem.elevation
    slope = state.slope_deg
    valid = state.dem.valid_mask
    blocked = state.cost_map.blocked_mask
    finite_cost = state.cost_map.total_cost[np.isfinite(state.cost_map.total_cost)]
    uncertainty = _compute_uncertainty_grid(state.class_probabilities)

    total_cells = int(elev.size)
    counts: dict[str, int] = {}
    for cid, label in CLASS_LABELS.items():
        counts[label] = int(np.sum(state.terrain_class_ids == cid))

    return {
        "grid_shape": elev.shape,
        "resolution_m": state.dem.resolution_m,
        "elevation_min_m": float(np.nanmin(elev[valid])) if np.any(valid) else 0.0,
        "elevation_max_m": float(np.nanmax(elev[valid])) if np.any(valid) else 0.0,
        "slope_mean_deg": float(np.nanmean(slope[valid])) if np.any(valid) else 0.0,
        "slope_max_deg": float(np.nanmax(slope[valid])) if np.any(valid) else 0.0,
        "blocked_cells": int(np.sum(blocked)),
        "blocked_pct": round(100.0 * float(np.sum(blocked)) / max(total_cells, 1), 1),
        "mean_cost": float(np.mean(finite_cost)) if finite_cost.size else float("inf"),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "class_counts": counts,
    }
