"""Tests for navigation.rover_simulator."""

import numpy as np

from navigation.astar import plan_path_uara_star
from navigation.path_smoothing import smooth_path_line_of_sight
from navigation.rover_simulator import RoverSimulator, SimulationStatus
from terrain.terrain_cost import CostWeights, compute_terrain_cost_map


def test_rover_reaches_destination_along_planned_path() -> None:
    slope = np.zeros((6, 6), dtype=np.float64)
    valid = np.ones((6, 6), dtype=bool)
    obstacles = np.zeros((6, 6), dtype=bool)
    obstacles[1:4, 3] = True

    cost_map = compute_terrain_cost_map(
        slope_deg=slope,
        valid_mask=valid,
        obstacle_mask=obstacles,
        weights=CostWeights(wd=1.0, ws=1.0, wt=1.0, wo=2.0, wu=1.0),
    )

    nav = plan_path_uara_star(cost_map, start=(2, 0), goal=(2, 5))
    assert nav.success is True

    smoothed = smooth_path_line_of_sight(nav.path, blocked_mask=cost_map.blocked_mask)
    rover = RoverSimulator(start_position=(2, 0))
    sim = rover.follow_path(smoothed, blocked_mask=cost_map.blocked_mask)

    assert sim.goal_reached is True
    assert sim.replan_required is False
    assert sim.final_status == SimulationStatus.GOAL_REACHED
    assert sim.final_position == (2, 5)
    assert sim.visited_cells[0] == (2, 0)
    assert sim.visited_cells[-1] == (2, 5)
    assert all(not cost_map.blocked_mask[r, c] for r, c in sim.visited_cells)


def test_rover_stops_safely_on_dynamic_obstacle_and_signals_replan() -> None:
    blocked = np.zeros((5, 5), dtype=bool)
    path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    rover = RoverSimulator(start_position=(0, 0))
    sim = rover.follow_path(
        path,
        blocked_mask=blocked,
        dynamic_obstacles_by_step={2: [(0, 2)]},
    )

    assert sim.goal_reached is False
    assert sim.replan_required is True
    assert sim.final_status == SimulationStatus.BLOCKED_REPLAN_REQUIRED
    # Rover must stop at (0, 1) before entering newly blocked (0, 2)
    assert sim.final_position == (0, 1)
    assert (0, 2) not in sim.visited_cells
    assert sim.events[-1].blocked_cell == (0, 2)


def test_rover_replans_and_resumes_to_goal() -> None:
    from navigation.astar import UARAStarPlanner

    slope = np.zeros((5, 5), dtype=np.float64)
    valid = np.ones((5, 5), dtype=bool)
    cost_map = compute_terrain_cost_map(slope_deg=slope, valid_mask=valid, obstacle_buffer_cells=0)

    planner = UARAStarPlanner()
    initial_nav = planner.plan(cost_map, start=(2, 0), goal=(2, 4))
    rover = RoverSimulator(start_position=(2, 0))

    halted = rover.follow_path(
        initial_nav.path,
        blocked_mask=cost_map.blocked_mask,
        dynamic_obstacles_by_step={2: [(2, 2)]},
    )
    assert halted.replan_required is True
    assert halted.final_position == (2, 1)

    updated_map, replanned = planner.replan_with_updates(
        cost_map=cost_map,
        current_position=rover.position,
        goal=(2, 4),
        newly_blocked_cells=[halted.events[-1].blocked_cell],
    )
    assert replanned.success is True

    resumed = rover.follow_path(replanned.path, blocked_mask=updated_map.blocked_mask)
    assert resumed.goal_reached is True
    assert resumed.final_position == (2, 4)
    assert (2, 2) not in resumed.visited_cells


def test_rover_rejects_out_of_bounds_start_position() -> None:
    blocked = np.zeros((4, 4), dtype=bool)
    rover = RoverSimulator(start_position=(-1, 0))
    sim = rover.follow_path([(-1, 0), (0, 0)], blocked_mask=blocked)

    assert sim.goal_reached is False
    assert sim.final_status == SimulationStatus.FAILED_INVALID_PATH

