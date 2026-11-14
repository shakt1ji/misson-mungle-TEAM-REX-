"""Admissible distance heuristics for 4-connected and 8-connected grid navigation."""

from __future__ import annotations

import math


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Compute L1 (4-connected grid step) distance between two (row, col) cells."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def euclidean_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Compute L2 Euclidean grid-cell distance between two (row, col) cells."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def octile_distance(
    a: tuple[int, int],
    b: tuple[int, int],
    cardinal_cost: float = 1.0,
    diagonal_cost: float = math.sqrt(2.0),
) -> float:
    """Compute octile distance for 8-connected grid movement."""
    if cardinal_cost < 0.0 or diagonal_cost < 0.0:
        raise ValueError("Step costs must be nonnegative.")
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return cardinal_cost * (dr + dc) + (diagonal_cost - 2.0 * cardinal_cost) * min(dr, dc)


def metric_distance_lower_bound(
    a: tuple[int, int],
    b: tuple[int, int],
    resolution_m: tuple[float, float],
    connectivity: int = 8,
) -> float:
    """Compute an admissible physical-meter lower bound for anisotropic grids."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8; got {connectivity}.")

    dy_m = float(resolution_m[0])
    dx_m = float(resolution_m[1])
    if dy_m <= 0.0 or dx_m <= 0.0:
        raise ValueError(f"resolution_m values must be positive; got {resolution_m}.")

    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])

    if connectivity == 4:
        return dy_m * dr + dx_m * dc

    diag_m = math.hypot(dy_m, dx_m)
    return dy_m * dr + dx_m * dc + (diag_m - dy_m - dx_m) * min(dr, dc)
