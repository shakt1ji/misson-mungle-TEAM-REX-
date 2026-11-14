"""Tests for UARA* navigation, heuristics, and line-of-sight path smoothing."""

import numpy as np
import pytest

from navigation.astar import UARAConfig, UARAStarPlanner, plan_path_uara_star
from navigation.path_smoothing import has_line_of_sight, smooth_path_line_of_sight
from terrain.terrain_cost import CostWeights, compute_terrain_cost_map
from terrain.traversability import TerrainClassConfig


def _flat_cost_map(
    rows: int,
    cols: int,
    obstacle_mask: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    class_cfg: TerrainClassConfig | None = None,
    weights: CostWeights | None = None,
    obstacle_buffer_cells: int = 0,
):
    slope = np.zeros((rows, cols), dtype=np.float64)
    valid = np.ones((rows, cols), dtype=bool)
    return compute_terrain_cost_map(
        slope_deg=slope,
        valid_mask=valid,
        resolution_m=(1.0, 1.0),
        terrain_class_ids=class_ids,
        class_probabilities=probs,
        obstacle_mask=obstacle_mask,
        class_config=class_cfg,
        weights=weights or CostWeights(wd=1.0, ws=1.0, wt=1.0, wo=2.0, wu=3.0),
        obstacle_buffer_cells=obstacle_buffer_cells,
    )


def test_obstacle_avoidance() -> None:
    obstacles = np.zeros((7, 7), dtype=bool)
    obstacles[1:6, 3] = True  # Vertical wall leaving top/bottom gaps
    cost_map = _flat_cost_map(7, 7, obstacle_mask=obstacles)

    result = plan_path_uara_star(cost_map, start=(3, 1), goal=(3, 5))

    assert result.success is True
    assert result.path[0] == (3, 1)
    assert result.path[-1] == (3, 5)
    for r, c in result.path:
        assert not obstacles[r, c]


def test_high_risk_uncertainty_route_avoidance() -> None:
    # Direct center corridor has high ML uncertainty; detour via row 0 or row 6 is low-risk
    rows, cols = 7, 7
    probs = np.zeros((rows, cols, 2), dtype=np.float64)
    probs[..., 0] = 0.98
    probs[..., 1] = 0.02

    # Inject high uncertainty band across rows 2..4, cols 2..4
    probs[2:5, 2:5, 0] = 0.50
    probs[2:5, 2:5, 1] = 0.50

    cost_map = _flat_cost_map(
        rows,
        cols,
        probs=probs,
        weights=CostWeights(wd=1.0, ws=0.0, wt=0.0, wo=0.0, wu=10.0),
    )

    result = plan_path_uara_star(
        cost_map,
        start=(3, 0),
        goal=(3, 6),
        config=UARAConfig(connectivity=8, safety_weight=1.0, uncertainty_weight=1.0),
    )

    assert result.success is True
    high_unc_cells = {(r, c) for r in range(2, 5) for c in range(2, 5)}
    assert all(cell not in high_unc_cells for cell in result.path)


def test_diagonal_corner_cutting_prevented() -> None:
    # Diagonal barrier from (0,1) and (1,0) blocks moving directly (0,0) <-> (1,1)
    obstacles = np.zeros((3, 3), dtype=bool)
    obstacles[0, 1] = True
    obstacles[1, 0] = True
    cost_map = _flat_cost_map(3, 3, obstacle_mask=obstacles)

    res_safe = plan_path_uara_star(
        cost_map,
        start=(0, 0),
        goal=(1, 1),
        config=UARAConfig(connectivity=8, prevent_corner_cutting=True),
    )
    assert res_safe.success is False
    assert res_safe.termination_reason == "unreachable"


def test_unreachable_goal() -> None:
    obstacles = np.zeros((5, 5), dtype=bool)
    obstacles[:, 2] = True  # Complete barrier splitting map
    cost_map = _flat_cost_map(5, 5, obstacle_mask=obstacles)

    result = plan_path_uara_star(cost_map, start=(2, 0), goal=(2, 4))

    assert result.success is False
    assert result.path == []
    assert np.isinf(result.total_cost)
    assert result.termination_reason == "unreachable"


def test_start_equals_goal() -> None:
    cost_map = _flat_cost_map(4, 4)
    result = plan_path_uara_star(cost_map, start=(1, 2), goal=(1, 2))

    assert result.success is True
    assert result.path == [(1, 2)]
    assert result.total_cost == 0.0
    assert result.path_length_steps == 0
    assert result.termination_reason == "start_equals_goal"


def test_path_smoothing_preserves_safety_and_endpoints() -> None:
    obstacles = np.zeros((7, 7), dtype=bool)
    obstacles[2:5, 3] = True
    cost_map = _flat_cost_map(7, 7, obstacle_mask=obstacles)

    nav = plan_path_uara_star(cost_map, start=(3, 1), goal=(3, 5))
    assert nav.success is True

    smoothed = smooth_path_line_of_sight(
        nav.path,
        blocked_mask=cost_map.blocked_mask,
        prevent_corner_cutting=True,
    )

    assert smoothed[0] == nav.path[0]
    assert smoothed[-1] == nav.path[-1]
    assert len(smoothed) <= len(nav.path)

    # Every smoothed segment must have verified obstacle-free line-of-sight
    for a, b in zip(smoothed[:-1], smoothed[1:]):
        assert has_line_of_sight(a, b, cost_map.blocked_mask, prevent_corner_cutting=True)
    # Must NOT collapse to direct [start, goal] because the wall blocks line-of-sight
    assert len(smoothed) >= 3


def test_line_of_sight_rejects_corner_cut_and_endpoint_only_shortcut() -> None:
    blocked = np.zeros((4, 4), dtype=bool)
    blocked[1, 2] = True
    blocked[2, 1] = True

    # Endpoints (0, 0) and (3, 3) are both free, but middle is blocked across diagonal
    assert has_line_of_sight((0, 0), (3, 3), blocked, prevent_corner_cutting=True) is False
    # Diagonal corner cut between (1, 1) and (2, 2) must also be rejected
    assert has_line_of_sight((1, 1), (2, 2), blocked, prevent_corner_cutting=True) is False


def test_replan_interface_with_newly_blocked_cell() -> None:
    cost_map = _flat_cost_map(6, 6)
    planner = UARAStarPlanner()
    initial = planner.plan(cost_map, start=(2, 0), goal=(2, 5))
    assert initial.success is True

    updated_map, replanned = planner.replan_with_updates(
        cost_map=cost_map,
        current_position=(2, 1),
        goal=(2, 5),
        newly_blocked_cells=[(2, 2), (2, 3)],
    )
    assert replanned.success is True
    assert updated_map.blocked_mask[2, 2] is np.bool_(True)
    assert (2, 2) not in replanned.path
    assert (2, 3) not in replanned.path


def test_invalid_and_blocked_start_goal_edge_cases() -> None:
    obstacles = np.zeros((4, 4), dtype=bool)
    obstacles[0, 0] = True
    obstacles[3, 3] = True
    cost_map = _flat_cost_map(4, 4, obstacle_mask=obstacles)

    assert plan_path_uara_star(cost_map, (-1, 0), (2, 2)).termination_reason == "invalid_start"
    assert plan_path_uara_star(cost_map, (1, 1), (4, 0)).termination_reason == "invalid_goal"
    assert plan_path_uara_star(cost_map, (0, 0), (2, 2)).termination_reason == "start_blocked"
    assert plan_path_uara_star(cost_map, (1, 1), (3, 3)).termination_reason == "goal_blocked"


def test_four_connected_and_anisotropic_metric_distance() -> None:
    slope = np.zeros((5, 5), dtype=np.float64)
    valid = np.ones((5, 5), dtype=bool)
    cost_map = compute_terrain_cost_map(
        slope_deg=slope,
        valid_mask=valid,
        resolution_m=(2.0, 5.0),
        weights=CostWeights(wd=1.0, ws=0.0, wt=0.0, wo=0.0, wu=0.0),
        obstacle_buffer_cells=0,
    )

    res_4 = plan_path_uara_star(
        cost_map,
        start=(0, 0),
        goal=(2, 3),
        config=UARAConfig(connectivity=4, use_metric_distance=True),
    )
    assert res_4.success is True
    assert res_4.path_distance_m == pytest.approx(2 * 2.0 + 3 * 5.0)
    assert res_4.total_cost == pytest.approx(2 * 2.0 + 3 * 5.0)
    assert res_4.is_optimal_guaranteed is True


def test_smoothing_respects_cost_grid_ceiling() -> None:
    # Center cell (2, 2) is traversable but very high risk; original path avoids it
    probs = np.zeros((5, 5, 2), dtype=np.float64)
    probs[..., 0] = 0.98
    probs[..., 1] = 0.02
    probs[1:4, 1:4, 0] = 0.50
    probs[1:4, 1:4, 1] = 0.50

    cost_map = _flat_cost_map(
        5,
        5,
        probs=probs,
        weights=CostWeights(wd=1.0, ws=0.0, wt=0.0, wo=0.0, wu=10.0),
    )
    nav = plan_path_uara_star(cost_map, start=(2, 0), goal=(2, 4))
    assert nav.success is True

    smoothed = smooth_path_line_of_sight(
        nav.path,
        blocked_mask=cost_map.blocked_mask,
        cost_grid=cost_map.total_cost,
    )
    # Must NOT shortcut straight across high-risk center (2, 2)
    assert len(smoothed) >= 3
    assert has_line_of_sight(
        smoothed[0],
        smoothed[-1],
        blocked_mask=cost_map.blocked_mask,
        cost_grid=cost_map.total_cost,
        max_cell_cost=max(float(cost_map.total_cost[r, c]) for r, c in nav.path),
    ) is False


def test_large_grid_200x200_navigation_performance() -> None:
    import time

    rows, cols = 200, 200
    obstacles = np.zeros((rows, cols), dtype=bool)
    obstacles[20:180, 100] = True  # Long obstacle wall requiring detour
    cost_map = _flat_cost_map(rows, cols, obstacle_mask=obstacles)

    t0 = time.perf_counter()
    res = plan_path_uara_star(cost_map, start=(100, 10), goal=(100, 190))
    elapsed = time.perf_counter() - t0

    assert res.success is True
    assert res.path[0] == (100, 10)
    assert res.path[-1] == (100, 190)
    assert all(not obstacles[r, c] for r, c in res.path)
    assert elapsed < 2.0

