"""UARA*: Uncertainty-Aware Risk-Adaptive A* navigation planner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import heapq
import math
from typing import Iterable

import numpy as np

from navigation.heuristics import (
    manhattan_distance,
    metric_distance_lower_bound,
    octile_distance,
)
from terrain.terrain_cost import TerrainCostMap


@dataclass(frozen=True)
class UARAConfig:
    """Configuration parameters for Uncertainty-Aware Risk-Adaptive A* (UARA*)."""

    connectivity: int = 8
    safety_weight: float = 1.0
    uncertainty_weight: float = 1.0
    heuristic_weight: float = 1.0
    use_metric_distance: bool = False
    prevent_corner_cutting: bool = True

    def __post_init__(self) -> None:
        if self.connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8; got {self.connectivity}.")
        if self.safety_weight < 0.0 or not np.isfinite(self.safety_weight):
            raise ValueError("safety_weight must be finite and >= 0.0.")
        if self.uncertainty_weight < 0.0 or not np.isfinite(self.uncertainty_weight):
            raise ValueError("uncertainty_weight must be finite and >= 0.0.")
        if self.heuristic_weight < 1.0 or not np.isfinite(self.heuristic_weight):
            raise ValueError("heuristic_weight must be finite and >= 1.0.")


@dataclass(frozen=True)
class NavigationResult:
    """Result and search telemetry returned by UARA* navigation."""

    success: bool
    path: list[tuple[int, int]]
    total_cost: float
    path_length_steps: int
    path_distance_m: float
    nodes_expanded: int
    nodes_generated: int
    termination_reason: str
    is_optimal_guaranteed: bool


_CARDINAL_MOVES: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

_DIAGONAL_MOVES: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _in_bounds(cell: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = cell
    return 0 <= r < rows and 0 <= c < cols


def _compute_heuristic(
    node: tuple[int, int],
    goal: tuple[int, int],
    cost_map: TerrainCostMap,
    config: UARAConfig,
) -> float:
    """Compute base admissible heuristic scaled by wd (before heuristic_weight)."""
    wd = cost_map.weights.wd
    if config.use_metric_distance:
        base_dist = metric_distance_lower_bound(
            node, goal, cost_map.resolution_m, connectivity=config.connectivity
        )
    elif config.connectivity == 4:
        base_dist = manhattan_distance(node, goal)
    else:
        base_dist = octile_distance(node, goal)
    return wd * base_dist


def _compute_path_distance_m(
    path: list[tuple[int, int]],
    resolution_m: tuple[float, float],
) -> float:
    """Compute total physical Euclidean path length in meters along grid steps."""
    if len(path) < 2:
        return 0.0
    dy_m, dx_m = resolution_m
    total_m = 0.0
    for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
        total_m += math.hypot((r2 - r1) * dy_m, (c2 - c1) * dx_m)
    return total_m


def plan_path_uara_star(
    cost_map: TerrainCostMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    config: UARAConfig | None = None,
) -> NavigationResult:
    """Plan a path from `start` to `goal` using Uncertainty-Aware Risk-Adaptive A*."""
    cfg = config or UARAConfig()
    rows, cols = cost_map.total_cost.shape
    blocked = cost_map.blocked_mask

    start_cell = (int(start[0]), int(start[1]))
    goal_cell = (int(goal[0]), int(goal[1]))

    if not _in_bounds(start_cell, rows, cols):
        return NavigationResult(
            success=False,
            path=[],
            total_cost=float("inf"),
            path_length_steps=0,
            path_distance_m=0.0,
            nodes_expanded=0,
            nodes_generated=0,
            termination_reason="invalid_start",
            is_optimal_guaranteed=False,
        )

    if not _in_bounds(goal_cell, rows, cols):
        return NavigationResult(
            success=False,
            path=[],
            total_cost=float("inf"),
            path_length_steps=0,
            path_distance_m=0.0,
            nodes_expanded=0,
            nodes_generated=0,
            termination_reason="invalid_goal",
            is_optimal_guaranteed=False,
        )

    if blocked[start_cell]:
        return NavigationResult(
            success=False,
            path=[],
            total_cost=float("inf"),
            path_length_steps=0,
            path_distance_m=0.0,
            nodes_expanded=0,
            nodes_generated=0,
            termination_reason="start_blocked",
            is_optimal_guaranteed=False,
        )

    if blocked[goal_cell]:
        return NavigationResult(
            success=False,
            path=[],
            total_cost=float("inf"),
            path_length_steps=0,
            path_distance_m=0.0,
            nodes_expanded=0,
            nodes_generated=0,
            termination_reason="goal_blocked",
            is_optimal_guaranteed=False,
        )

    if start_cell == goal_cell:
        return NavigationResult(
            success=True,
            path=[start_cell],
            total_cost=0.0,
            path_length_steps=0,
            path_distance_m=0.0,
            nodes_expanded=0,
            nodes_generated=1,
            termination_reason="start_equals_goal",
            is_optimal_guaranteed=True,
        )

    dy_m, dx_m = cost_map.resolution_m
    diag_m = math.hypot(dy_m, dx_m)
    sqrt2 = math.sqrt(2.0)

    moves: list[tuple[int, int, float, bool]] = []
    for dr, dc in _CARDINAL_MOVES:
        if cfg.use_metric_distance:
            step_len = dy_m if dr != 0 else dx_m
        else:
            step_len = 1.0
        moves.append((dr, dc, step_len, False))

    if cfg.connectivity == 8:
        for dr, dc in _DIAGONAL_MOVES:
            step_len = diag_m if cfg.use_metric_distance else sqrt2
            moves.append((dr, dc, step_len, True))

    w = cost_map.weights
    cell_risk_grid = cfg.safety_weight * (
        w.ws * cost_map.slope_risk
        + w.wt * cost_map.terrain_risk
        + w.wo * cost_map.obstacle_risk
        + cfg.uncertainty_weight * w.wu * cost_map.uncertainty_risk
    )

    traversable_cells = ~blocked
    all_risks_nonnegative = bool(
        np.all(cell_risk_grid[traversable_cells] >= 0.0)
    ) if np.any(traversable_cells) else True

    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    g_score[start_cell] = 0.0
    parent_r = np.full((rows, cols), -1, dtype=np.int32)
    parent_c = np.full((rows, cols), -1, dtype=np.int32)
    closed = np.zeros((rows, cols), dtype=bool)

    h_start = _compute_heuristic(start_cell, goal_cell, cost_map, cfg)
    open_heap: list[tuple[float, float, int, int]] = [
        (cfg.heuristic_weight * h_start, 0.0, start_cell[0], start_cell[1])
    ]

    nodes_expanded = 0
    nodes_generated = 1
    is_optimal = (
        cfg.heuristic_weight == 1.0
        and w.wd > 0.0
        and all_risks_nonnegative
    )

    goal_r, goal_c = goal_cell

    while open_heap:
        _, neg_g, r, c = heapq.heappop(open_heap)
        current_g = -neg_g

        if closed[r, c]:
            continue
        if current_g > g_score[r, c]:
            continue

        if r == goal_r and c == goal_c:
            path: list[tuple[int, int]] = [(r, c)]
            cr, cc = r, c
            while not (cr == start_cell[0] and cc == start_cell[1]):
                pr, pc = int(parent_r[cr, cc]), int(parent_c[cr, cc])
                if pr < 0 or pc < 0:
                    break
                path.append((pr, pc))
                cr, cc = pr, pc
            path.reverse()

            return NavigationResult(
                success=True,
                path=path,
                total_cost=float(current_g),
                path_length_steps=len(path) - 1,
                path_distance_m=_compute_path_distance_m(path, cost_map.resolution_m),
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                termination_reason="goal_reached",
                is_optimal_guaranteed=is_optimal,
            )

        closed[r, c] = True
        nodes_expanded += 1

        for dr, dc, step_len, is_diag in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if blocked[nr, nc] or closed[nr, nc]:
                continue
            if is_diag and cfg.prevent_corner_cutting:
                if blocked[r + dr, c] or blocked[r, c + dc]:
                    continue

            neighbor_risk = float(cell_risk_grid[nr, nc])
            if not np.isfinite(neighbor_risk) or neighbor_risk < 0.0:
                continue

            step_cost = step_len * (w.wd + neighbor_risk)
            tentative_g = current_g + step_cost

            if tentative_g < g_score[nr, nc]:
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                g_score[nr, nc] = tentative_g
                h_val = _compute_heuristic((nr, nc), goal_cell, cost_map, cfg)
                f_val = tentative_g + cfg.heuristic_weight * h_val
                heapq.heappush(open_heap, (f_val, -tentative_g, nr, nc))
                nodes_generated += 1

    return NavigationResult(
        success=False,
        path=[],
        total_cost=float("inf"),
        path_length_steps=0,
        path_distance_m=0.0,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        termination_reason="unreachable",
        is_optimal_guaranteed=False,
    )


class PlannerInterface(ABC):
    """Abstract planner contract designed for future incremental replanners (e.g., D* Lite)."""

    @abstractmethod
    def plan(
        self,
        cost_map: TerrainCostMap,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> NavigationResult:
        """Compute an initial path from start to goal."""

    @abstractmethod
    def replan_with_updates(
        self,
        cost_map: TerrainCostMap,
        current_position: tuple[int, int],
        goal: tuple[int, int],
        newly_blocked_cells: Iterable[tuple[int, int]],
    ) -> tuple[TerrainCostMap, NavigationResult]:
        """Update map state with newly discovered blocked cells and replan."""


class UARAStarPlanner(PlannerInterface):
    """Stateful-friendly wrapper around UARA* implementing PlannerInterface."""

    def __init__(self, config: UARAConfig | None = None) -> None:
        self.config = config or UARAConfig()

    def plan(
        self,
        cost_map: TerrainCostMap,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> NavigationResult:
        return plan_path_uara_star(cost_map, start, goal, config=self.config)

    def replan_with_updates(
        self,
        cost_map: TerrainCostMap,
        current_position: tuple[int, int],
        goal: tuple[int, int],
        newly_blocked_cells: Iterable[tuple[int, int]],
    ) -> tuple[TerrainCostMap, NavigationResult]:
        """Return an updated TerrainCostMap copy and a replanned NavigationResult."""
        rows, cols = cost_map.total_cost.shape
        updated_blocked = cost_map.blocked_mask.copy()
        updated_total = cost_map.total_cost.copy()
        updated_obs_risk = cost_map.obstacle_risk.copy()

        for cell in newly_blocked_cells:
            r, c = int(cell[0]), int(cell[1])
            if 0 <= r < rows and 0 <= c < cols:
                updated_blocked[r, c] = True
                updated_total[r, c] = np.inf
                updated_obs_risk[r, c] = np.inf

        updated_traversable = (~updated_blocked) & cost_map.valid_mask
        updated_map = replace(
            cost_map,
            total_cost=updated_total,
            obstacle_risk=updated_obs_risk,
            blocked_mask=updated_blocked,
            traversable_mask=updated_traversable,
        )
        result = self.plan(updated_map, current_position, goal)
        return updated_map, result
