"""Simulation state models, mission status enum, and JSON serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from navigation.astar import NavigationResult
from simulation.dynamic_obstacles import DynamicRockObstacle
from simulation.physics import RoverPhysicsTelemetry, TERRAIN_PHYSICS_MODEL
from terrain.dem_loader import DEMData
from terrain.terrain_cost import TerrainCostMap


class MissionStatus(str, Enum):
    """Exact mission lifecycle states required for the MARS-09 interactive simulator."""

    IDLE = "IDLE"
    MOVING = "MOVING"
    OBSTACLE_DETECTED = "OBSTACLE_DETECTED"
    REPLANNING = "REPLANNING"
    ROUTE_UPDATED = "ROUTE_UPDATED"
    GOAL_REACHED = "GOAL_REACHED"
    NO_SAFE_PATH = "NO_SAFE_PATH"


@dataclass(frozen=True)
class ReplanEvent:
    """Timestamped log entry for obstacle detection and route recalculation events."""

    event_id: int
    timestamp_sec: float
    trigger: str
    rover_position: tuple[int, int]
    old_remaining_steps: int
    new_steps: int
    new_cost: float
    nodes_expanded: int
    success: bool
    message: str


@dataclass
class RoverSimulationState:
    """Mutable full state of the interactive Mars rover simulation."""

    scenario_name: str
    dem: DEMData
    slope_deg: np.ndarray
    terrain_class_ids: np.ndarray
    class_probabilities: np.ndarray
    static_blocked_mask: np.ndarray
    cost_map: TerrainCostMap

    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    rover_cell: tuple[int, int]
    rover_row_f: float
    rover_col_f: float
    heading_rad: float

    mission_status: MissionStatus = MissionStatus.IDLE
    is_running: bool = False
    speed_multiplier: float = 1.0
    auto_replan: bool = True
    show_cost_overlay: bool = False
    show_uncertainty_overlay: bool = False
    show_previous_route: bool = True

    sim_time_sec: float = 0.0
    distance_travelled_m: float = 0.0
    replan_count: int = 0
    alert_message: str = "Ready — press Start Mission or drag obstacles to test dynamic avoidance."
    alert_seq: int = 0

    original_route: list[tuple[int, int]] = field(default_factory=list)
    previous_route: list[tuple[int, int]] = field(default_factory=list)
    active_route: list[tuple[int, int]] = field(default_factory=list)
    expanded_route: list[tuple[int, int]] = field(default_factory=list)
    route_step_index: int = 0
    segment_progress: float = 0.0

    travelled_cells: list[tuple[int, int]] = field(default_factory=list)
    travelled_points: list[tuple[float, float]] = field(default_factory=list)

    last_plan_result: NavigationResult | None = None
    physics: RoverPhysicsTelemetry | None = None
    replan_events: list[ReplanEvent] = field(default_factory=list)


def _compute_uncertainty_grid(class_probabilities: np.ndarray) -> np.ndarray:
    """Compute normalized Shannon entropy [0, 1] across the 4 AI4MARS terrain classes."""
    probs = np.clip(np.asarray(class_probabilities, dtype=np.float64), 1e-9, 1.0)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    entropy = -np.sum(probs * np.log(probs), axis=-1)
    max_entropy = float(np.log(probs.shape[-1]))
    return np.clip(entropy / max(max_entropy, 1e-9), 0.0, 1.0)


def serialize_simulation_state(
    state: RoverSimulationState,
    obstacles: list[DynamicRockObstacle],
    include_grids: bool = True,
) -> dict[str, Any]:
    """Serialize state into a clean JSON-compatible dictionary for web/Streamlit UIs."""
    rows, cols = state.dem.elevation.shape
    cr, cc = state.rover_cell
    class_id = int(state.terrain_class_ids[cr, cc])
    soil_model = TERRAIN_PHYSICS_MODEL.get(class_id, TERRAIN_PHYSICS_MODEL[255])

    total_cost_clean = np.where(
        np.isfinite(state.cost_map.total_cost),
        np.round(state.cost_map.total_cost, 3),
        -1.0,
    )

    payload: dict[str, Any] = {
        "scenario_name": state.scenario_name,
        "grid_shape": [int(rows), int(cols)],
        "resolution_m": [float(state.dem.resolution_m[0]), float(state.dem.resolution_m[1])],
        "start_cell": [int(state.start_cell[0]), int(state.start_cell[1])],
        "goal_cell": [int(state.goal_cell[0]), int(state.goal_cell[1])],
        "rover": {
            "row": int(state.rover_cell[0]),
            "col": int(state.rover_cell[1]),
            "row_f": round(float(state.rover_row_f), 4),
            "col_f": round(float(state.rover_col_f), 4),
            "heading_rad": round(float(state.heading_rad), 4),
            "elevation_m": round(float(state.dem.elevation[cr, cc]), 3),
            "terrain_class": soil_model.name,
            "terrain_class_id": class_id,
        },
        "mission_status": state.mission_status.value,
        "is_running": bool(state.is_running),
        "speed_multiplier": float(state.speed_multiplier),
        "auto_replan": bool(state.auto_replan),
        "show_cost_overlay": bool(state.show_cost_overlay),
        "show_uncertainty_overlay": bool(state.show_uncertainty_overlay),
        "show_previous_route": bool(state.show_previous_route),
        "sim_time_sec": round(float(state.sim_time_sec), 2),
        "distance_travelled_m": round(float(state.distance_travelled_m), 2),
        "replan_count": int(state.replan_count),
        "alert_message": state.alert_message,
        "alert_seq": int(state.alert_seq),
        "obstacles": [
            {
                "obstacle_id": obs.obstacle_id,
                "row": int(obs.row),
                "col": int(obs.col),
                "radius_cells": int(obs.radius_cells),
                "height_m": round(float(obs.height_m), 2),
                "label": obs.label,
                "is_draggable": bool(obs.is_draggable),
                "footprint": [[int(rr), int(cc2)] for rr, cc2 in obs.footprint_cells(rows, cols)],
            }
            for obs in obstacles
        ],
        "routes": {
            "original_route": [[int(r), int(c)] for r, c in state.original_route],
            "previous_route": [[int(r), int(c)] for r, c in state.previous_route],
            "active_route": [[int(r), int(c)] for r, c in state.active_route],
            "expanded_route": [[int(r), int(c)] for r, c in state.expanded_route],
            "travelled_cells": [[int(r), int(c)] for r, c in state.travelled_cells],
            "travelled_points": [
                [round(float(r), 3), round(float(c), 3)] for r, c in state.travelled_points
            ],
            "remaining_steps": max(0, len(state.expanded_route) - 1 - state.route_step_index)
            if state.expanded_route
            else 0,
        },
        "planner_metrics": {
            "success": bool(state.last_plan_result.success) if state.last_plan_result else False,
            "total_cost": round(float(state.last_plan_result.total_cost), 3)
            if state.last_plan_result and np.isfinite(state.last_plan_result.total_cost)
            else None,
            "path_distance_m": round(float(state.last_plan_result.path_distance_m), 2)
            if state.last_plan_result
            else 0.0,
            "nodes_expanded": int(state.last_plan_result.nodes_expanded)
            if state.last_plan_result
            else 0,
            "termination_reason": state.last_plan_result.termination_reason
            if state.last_plan_result
            else "none",
        },
        "physics": {
            "grade_slope_deg": state.physics.grade_slope_deg if state.physics else 0.0,
            "pitch_deg": state.physics.pitch_deg if state.physics else 0.0,
            "roll_deg": state.physics.roll_deg if state.physics else 0.0,
            "slip_ratio": state.physics.slip_ratio if state.physics else 0.0,
            "sinkage_cm": state.physics.sinkage_cm if state.physics else 0.0,
            "rolling_resistance_n": state.physics.rolling_resistance_n if state.physics else 0.0,
            "grade_resistance_n": state.physics.grade_resistance_n if state.physics else 0.0,
            "tractive_force_n": state.physics.tractive_force_n if state.physics else 0.0,
            "effective_speed_mps": state.physics.effective_speed_mps if state.physics else 0.0,
            "motor_power_w": state.physics.motor_power_w if state.physics else 35.0,
            "energy_wh": state.physics.energy_wh if state.physics else 0.0,
            "wheel_spin_rad": state.physics.wheel_spin_rad if state.physics else 0.0,
            "tip_over_margin_deg": state.physics.tip_over_margin_deg if state.physics else 28.0,
        },
        "replan_events": [
            {
                "event_id": ev.event_id,
                "timestamp_sec": round(ev.timestamp_sec, 2),
                "trigger": ev.trigger,
                "rover_position": [int(ev.rover_position[0]), int(ev.rover_position[1])],
                "old_remaining_steps": int(ev.old_remaining_steps),
                "new_steps": int(ev.new_steps),
                "new_cost": round(float(ev.new_cost), 3) if np.isfinite(ev.new_cost) else None,
                "nodes_expanded": int(ev.nodes_expanded),
                "success": bool(ev.success),
                "message": ev.message,
            }
            for ev in state.replan_events
        ],
    }

    if include_grids:
        uncertainty = _compute_uncertainty_grid(state.class_probabilities)
        payload["grids"] = {
            "elevation": np.round(state.dem.elevation, 3).tolist(),
            "slope_deg": np.round(state.slope_deg, 2).tolist(),
            "terrain_class_ids": state.terrain_class_ids.astype(int).tolist(),
            "blocked_mask": state.cost_map.blocked_mask.astype(int).tolist(),
            "total_cost": total_cost_clean.tolist(),
            "uncertainty": np.round(uncertainty, 3).tolist(),
        }

    return payload
