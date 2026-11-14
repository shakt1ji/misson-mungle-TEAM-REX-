"""Safety-preserving path smoothing via supercover line-of-sight checks."""

from __future__ import annotations

import numpy as np


def grid_line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Return all grid cells intersected by the segment from `a` to `b` (supercover)."""
    r0, c0 = int(a[0]), int(a[1])
    r1, c1 = int(b[0]), int(b[1])

    cells: list[tuple[int, int]] = [(r0, c0)]
    if (r0, c0) == (r1, c1):
        return cells

    y = r0
    x = c0
    dy = r1 - r0
    dx = c1 - c0

    ystep = 1 if dy > 0 else -1
    xstep = 1 if dx > 0 else -1
    dy = abs(dy)
    dx = abs(dx)
    ddy = 2 * dy
    ddx = 2 * dx

    if ddx >= ddy:
        errorprev = dx
        error = dx
        for _ in range(dx):
            x += xstep
            error += ddy
            if error > ddx:
                y += ystep
                error -= ddx
                if error + errorprev < ddx:
                    cells.append((y - ystep, x))
                elif error + errorprev > ddx:
                    cells.append((y, x - xstep))
                else:
                    cells.append((y - ystep, x))
                    cells.append((y, x - xstep))
            cells.append((y, x))
            errorprev = error
    else:
        errorprev = dy
        error = dy
        for _ in range(dy):
            y += ystep
            error += ddx
            if error > ddy:
                x += xstep
                error -= ddy
                if error + errorprev < ddy:
                    cells.append((y, x - xstep))
                elif error + errorprev > ddy:
                    cells.append((y - ystep, x))
                else:
                    cells.append((y, x - xstep))
                    cells.append((y - ystep, x))
            cells.append((y, x))
            errorprev = error

    deduped: list[tuple[int, int]] = []
    for cell in cells:
        if not deduped or deduped[-1] != cell:
            deduped.append(cell)
    return deduped


def bresenham_centerline(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Return single-pixel-wide Bresenham cells from `a` to `b` for discrete stepping."""
    r0, c0 = int(a[0]), int(a[1])
    r1, c1 = int(b[0]), int(b[1])

    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def has_line_of_sight(
    a: tuple[int, int],
    b: tuple[int, int],
    blocked_mask: np.ndarray,
    prevent_corner_cutting: bool = True,
    cost_grid: np.ndarray | None = None,
    max_cell_cost: float | None = None,
) -> bool:
    """Check whether a collision-free line of sight exists between `a` and `b`."""
    mask = np.asarray(blocked_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"blocked_mask must be 2D; got shape {mask.shape}.")
    rows, cols = mask.shape

    if cost_grid is not None:
        costs = np.asarray(cost_grid, dtype=np.float64)
        if costs.shape != mask.shape:
            raise ValueError(
                f"cost_grid shape {costs.shape} must match blocked_mask shape {mask.shape}."
            )
    else:
        costs = None

    cells_to_check = (
        grid_line_cells(a, b) if prevent_corner_cutting else bresenham_centerline(a, b)
    )

    for r, c in cells_to_check:
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        if mask[r, c]:
            return False
        if costs is not None and max_cell_cost is not None:
            cell_cost = float(costs[r, c])
            if not np.isfinite(cell_cost) or cell_cost > max_cell_cost:
                return False

    return True


def smooth_path_line_of_sight(
    path: list[tuple[int, int]],
    blocked_mask: np.ndarray,
    prevent_corner_cutting: bool = True,
    cost_grid: np.ndarray | None = None,
    max_cell_cost: float | None = None,
) -> list[tuple[int, int]]:
    """Smooth a grid path by removing redundant waypoints with safe line-of-sight checks."""
    if not path:
        return []
    if len(path) <= 2:
        return [tuple(map(int, p)) for p in path]

    mask = np.asarray(blocked_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"blocked_mask must be 2D; got shape {mask.shape}.")

    effective_max_cost = max_cell_cost
    if cost_grid is not None and effective_max_cost is None:
        costs = np.asarray(cost_grid, dtype=np.float64)
        path_costs = [float(costs[int(r), int(c)]) for r, c in path]
        if all(np.isfinite(c) for c in path_costs):
            effective_max_cost = max(path_costs) + 1e-9

    normalized_path = [(int(p[0]), int(p[1])) for p in path]
    smoothed: list[tuple[int, int]] = [normalized_path[0]]

    current_idx = 0
    n = len(normalized_path)

    while current_idx < n - 1:
        next_idx = current_idx + 1
        for candidate_idx in range(n - 1, current_idx, -1):
            if has_line_of_sight(
                normalized_path[current_idx],
                normalized_path[candidate_idx],
                blocked_mask=mask,
                prevent_corner_cutting=prevent_corner_cutting,
                cost_grid=cost_grid,
                max_cell_cost=effective_max_cost,
            ):
                next_idx = candidate_idx
                break

        smoothed.append(normalized_path[next_idx])
        current_idx = next_idx

    return smoothed
