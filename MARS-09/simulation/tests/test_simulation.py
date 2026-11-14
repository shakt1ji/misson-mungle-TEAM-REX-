"""Comprehensive Pytest suite for dynamic obstacle avoidance, UARA* replanning, and Martian physics."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.dynamic_obstacles import (
    InvalidObstaclePlacementError,
    canvas_to_grid,
)
from simulation.physics import step_rover_physics
from simulation.simulation_controller import RoverSimulationController
from simulation.simulation_state import MissionStatus


def _first_future_route_cell(ctrl: RoverSimulationController) -> tuple[int, int]:
    """Return an interior cell along the remaining expanded route strictly ahead of the rover."""
    expanded = ctrl.state.expanded_route
    idx = ctrl.state.route_step_index
    remaining = expanded[idx + 2 : -2]
    assert remaining, "Expected at least 3 future steps along route for test."
    return remaining[len(remaining) // 2]


def test_adding_obstacle_updates_cost_and_mask() -> None:
    ctrl = RoverSimulationController(scenario_name="normal_navigation")
    target = (15, 5)
    assert not ctrl.state.cost_map.blocked_mask[target]

    obs = ctrl.add_obstacle(row=target[0], col=target[1], radius_cells=0, label="New Basalt")
    assert ctrl.state.cost_map.blocked_mask[target]
    assert np.isinf(ctrl.state.cost_map.total_cost[target])
    assert any(o.obstacle_id == obs.obstacle_id for o in ctrl.obstacle_manager.obstacles)


def test_moving_obstacle_frees_old_and_blocks_new() -> None:
    ctrl = RoverSimulationController(scenario_name="dynamic_obstacle")
    first = ctrl.obstacle_manager.obstacles[0]
    old_pos = (first.row, first.col)
    new_pos = (10, 10)

    ctrl.move_obstacle(first.obstacle_id, new_row=new_pos[0], new_col=new_pos[1])
    moved = ctrl.obstacle_manager.get(first.obstacle_id)
    assert moved is not None
    assert (moved.row, moved.col) == new_pos
    assert not ctrl.state.cost_map.blocked_mask[old_pos]
    assert ctrl.state.cost_map.blocked_mask[new_pos]


def test_removing_obstacle_unblocks_cells() -> None:
    ctrl = RoverSimulationController(scenario_name="normal_navigation")
    obs = ctrl.add_obstacle(row=14, col=6, radius_cells=1)
    assert ctrl.state.cost_map.blocked_mask[14, 6]

    ctrl.remove_obstacle(obs.obstacle_id)
    assert not ctrl.state.cost_map.blocked_mask[14, 6]
    assert ctrl.obstacle_manager.get(obs.obstacle_id) is None


def test_obstacle_on_active_route_triggers_detection_and_automatic_replan() -> None:
    ctrl = RoverSimulationController(scenario_name="dynamic_obstacle")
    ctrl.start()
    for _ in range(8):
        ctrl.step(0.5)

    assert ctrl.state.mission_status == MissionStatus.MOVING
    block_target = _first_future_route_cell(ctrl)
    draggable = ctrl.obstacle_manager.obstacles[0]

    # Drag large boulder onto the active route ahead of the moving rover
    ctrl.move_obstacle(draggable.obstacle_id, new_row=block_target[0], new_col=block_target[1])

    assert ctrl.state.replan_count >= 1
    assert ctrl.state.mission_status == MissionStatus.ROUTE_UPDATED
    assert "Obstacle detected" in ctrl.state.alert_message
    assert len(ctrl.state.replan_events) >= 1
    assert block_target not in ctrl.state.expanded_route


def test_auto_replan_disabled_halts_in_obstacle_detected_state() -> None:
    ctrl = RoverSimulationController(scenario_name="normal_navigation")
    ctrl.set_controls(auto_replan=False)
    ctrl.start()
    ctrl.step(0.5)

    block_target = _first_future_route_cell(ctrl)
    ctrl.add_obstacle(row=block_target[0], col=block_target[1], radius_cells=0)

    assert ctrl.state.mission_status == MissionStatus.OBSTACLE_DETECTED
    assert not ctrl.state.is_running
    assert "Obstacle detected" in ctrl.state.alert_message


def test_replanning_starts_from_current_rover_position_and_preserves_travelled_path() -> None:
    ctrl = RoverSimulationController(scenario_name="dynamic_obstacle")
    ctrl.start()
    for _ in range(10):
        ctrl.step(0.5)

    pos_before = ctrl.state.rover_cell
    assert pos_before != ctrl.state.start_cell
    travelled_snapshot = list(ctrl.state.travelled_cells)
    assert len(travelled_snapshot) >= 2

    block_target = _first_future_route_cell(ctrl)
    ctrl.add_obstacle(row=block_target[0], col=block_target[1], radius_cells=1)

    assert ctrl.state.rover_cell == pos_before
    assert ctrl.state.travelled_cells == travelled_snapshot
    assert ctrl.state.active_route[0] == pos_before
    assert ctrl.state.expanded_route[0] == pos_before
    assert ctrl.state.expanded_route[-1] == ctrl.state.goal_cell


def test_no_safe_path_when_all_routes_blocked_and_recovery_when_removed() -> None:
    ctrl = RoverSimulationController(scenario_name="no_safe_path")
    assert ctrl.state.mission_status == MissionStatus.NO_SAFE_PATH
    assert ctrl.state.active_route == []
    assert not ctrl.state.is_running

    # Remove middle barrier rocks to open a safe corridor
    mid_rocks = [
        obs for obs in ctrl.obstacle_manager.obstacles if 12 <= obs.row <= 18
    ]
    for obs in mid_rocks:
        ctrl.remove_obstacle(obs.obstacle_id)

    assert ctrl.state.mission_status == MissionStatus.ROUTE_UPDATED
    assert len(ctrl.state.active_route) >= 2
    assert ctrl.state.expanded_route[0] == ctrl.state.rover_cell
    assert ctrl.state.expanded_route[-1] == ctrl.state.goal_cell


def test_terrain_modification_updates_costs_and_physics_slip() -> None:
    ctrl = RoverSimulationController(scenario_name="normal_navigation")
    cell = (ctrl.state.rover_cell[0], ctrl.state.rover_cell[1])

    ctrl.paint_terrain([cell], "bedrock")
    bedrock_telemetry = step_rover_physics(
        elevation=ctrl.state.dem.elevation,
        slope_deg=ctrl.state.slope_deg,
        valid_mask=ctrl.state.dem.valid_mask,
        terrain_class_ids=ctrl.state.terrain_class_ids,
        resolution_m=ctrl.state.dem.resolution_m,
        row_f=float(cell[0]),
        col_f=float(cell[1]),
        heading_rad=0.0,
        speed_multiplier=1.0,
        dt_sec=0.25,
        is_moving=True,
        prev_energy_wh=0.0,
        prev_wheel_spin_rad=0.0,
    )

    ctrl.paint_terrain([cell], "sand")
    sand_telemetry = step_rover_physics(
        elevation=ctrl.state.dem.elevation,
        slope_deg=ctrl.state.slope_deg,
        valid_mask=ctrl.state.dem.valid_mask,
        terrain_class_ids=ctrl.state.terrain_class_ids,
        resolution_m=ctrl.state.dem.resolution_m,
        row_f=float(cell[0]),
        col_f=float(cell[1]),
        heading_rad=0.0,
        speed_multiplier=1.0,
        dt_sec=0.25,
        is_moving=True,
        prev_energy_wh=0.0,
        prev_wheel_spin_rad=0.0,
    )

    assert sand_telemetry.slip_ratio > bedrock_telemetry.slip_ratio
    assert sand_telemetry.sinkage_cm > bedrock_telemetry.sinkage_cm
    assert sand_telemetry.rolling_resistance_n > bedrock_telemetry.rolling_resistance_n


def test_rover_reaches_goal_safely_after_dynamic_replanning() -> None:
    ctrl = RoverSimulationController(scenario_name="dynamic_obstacle")
    ctrl.set_controls(speed_multiplier=4.0)
    ctrl.start()

    for _ in range(12):
        ctrl.step(0.5)

    block_target = _first_future_route_cell(ctrl)
    draggable = ctrl.obstacle_manager.obstacles[0]
    ctrl.move_obstacle(draggable.obstacle_id, new_row=block_target[0], new_col=block_target[1])
    assert ctrl.state.replan_count >= 1

    for _ in range(300):
        ctrl.step(0.5)
        if ctrl.state.mission_status == MissionStatus.GOAL_REACHED:
            break

    assert ctrl.state.mission_status == MissionStatus.GOAL_REACHED
    assert ctrl.state.rover_cell == ctrl.state.goal_cell
    assert ctrl.state.distance_travelled_m > 0.0
    assert ctrl.state.physics is not None and ctrl.state.physics.energy_wh > 0.0
    # Verify rover never stepped on any blocked cell
    for cell in ctrl.state.travelled_cells:
        assert not ctrl.state.cost_map.blocked_mask[cell]


def test_canvas_to_grid_coordinate_conversion() -> None:
    assert canvas_to_grid(0, 0, 600, 600, (30, 30)) == (0, 0)
    assert canvas_to_grid(599, 599, 600, 600, (30, 30)) == (29, 29)
    assert canvas_to_grid(310, 150, 600, 600, (30, 30)) == (7, 15)
    assert canvas_to_grid(750, -10, 600, 600, (30, 30), clamp=True) == (0, 29)

    with pytest.raises(InvalidObstaclePlacementError):
        canvas_to_grid(-5, 10, 600, 600, (30, 30), clamp=False)


def test_invalid_obstacle_placement_rejected() -> None:
    ctrl = RoverSimulationController(scenario_name="normal_navigation")
    rover_pos = ctrl.state.rover_cell
    goal_pos = ctrl.state.goal_cell

    with pytest.raises(InvalidObstaclePlacementError):
        ctrl.add_obstacle(row=rover_pos[0], col=rover_pos[1])

    with pytest.raises(InvalidObstaclePlacementError):
        ctrl.add_obstacle(row=goal_pos[0], col=goal_pos[1])

    with pytest.raises(InvalidObstaclePlacementError):
        ctrl.add_obstacle(row=-1, col=5)

    with pytest.raises(InvalidObstaclePlacementError):
        ctrl.add_obstacle(row=99, col=5)
