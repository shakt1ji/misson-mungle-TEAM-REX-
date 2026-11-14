"""Dynamic obstacle management, coordinate conversion, and placement validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np


class InvalidObstaclePlacementError(ValueError):
    """Raised when an obstacle placement is out of bounds or collides with rover/goal."""


@dataclass(frozen=True)
class DynamicRockObstacle:
    """Represents a physical rock/boulder obstacle on the Martian terrain grid."""

    obstacle_id: str
    row: int
    col: int
    radius_cells: int = 0
    height_m: float = 1.15
    label: str = "Boulder"
    is_draggable: bool = True

    def footprint_cells(self, rows: int, cols: int) -> list[tuple[int, int]]:
        """Return all in-bounds grid cells occupied by this rock obstacle."""
        if self.radius_cells <= 0:
            if 0 <= self.row < rows and 0 <= self.col < cols:
                return [(self.row, self.col)]
            return []

        occupied: list[tuple[int, int]] = []
        r_min = max(0, self.row - self.radius_cells)
        r_max = min(rows - 1, self.row + self.radius_cells)
        c_min = max(0, self.col - self.radius_cells)
        c_max = min(cols - 1, self.col + self.radius_cells)
        threshold = float(self.radius_cells) + 0.15
        for rr in range(r_min, r_max + 1):
            for cc in range(c_min, c_max + 1):
                if math.hypot(rr - self.row, cc - self.col) <= threshold:
                    occupied.append((rr, cc))
        return occupied


def canvas_to_grid(
    canvas_x: float,
    canvas_y: float,
    canvas_width: float,
    canvas_height: float,
    grid_shape: tuple[int, int],
    clamp: bool = False,
) -> tuple[int, int]:
    """Convert pixel canvas coordinates (x, y) into integer grid (row, col) coordinates."""
    if canvas_width <= 0.0 or canvas_height <= 0.0:
        raise ValueError("Canvas width and height must be positive.")
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"grid_shape must have positive dimensions; got {grid_shape}.")

    col_f = (float(canvas_x) / float(canvas_width)) * cols
    row_f = (float(canvas_y) / float(canvas_height)) * rows
    col = int(math.floor(col_f))
    row = int(math.floor(row_f))

    if clamp:
        row = max(0, min(rows - 1, row))
        col = max(0, min(cols - 1, col))
    elif not (0 <= row < rows and 0 <= col < cols):
        raise InvalidObstaclePlacementError(
            f"Canvas coordinate ({canvas_x}, {canvas_y}) maps to out-of-bounds cell "
            f"({row}, {col}) for grid {grid_shape}."
        )
    return (row, col)


class DynamicObstacleManager:
    """Manages interactive rock obstacles and rasterizes them to a 2D occupancy mask."""

    def __init__(self, obstacles: Iterable[DynamicRockObstacle] | None = None) -> None:
        self._obstacles: dict[str, DynamicRockObstacle] = {}
        self._counter = 0
        if obstacles is not None:
            for obs in obstacles:
                self._obstacles[obs.obstacle_id] = obs
                self._counter += 1

    @property
    def obstacles(self) -> list[DynamicRockObstacle]:
        """Return deterministic list of all current rock obstacles."""
        return list(self._obstacles.values())

    def get(self, obstacle_id: str) -> DynamicRockObstacle | None:
        return self._obstacles.get(obstacle_id)

    def _validate_cell(
        self,
        row: int,
        col: int,
        radius_cells: int,
        grid_shape: tuple[int, int],
        rover_cell: tuple[int, int],
        goal_cell: tuple[int, int],
        start_cell: tuple[int, int] | None = None,
    ) -> list[tuple[int, int]]:
        rows, cols = int(grid_shape[0]), int(grid_shape[1])
        if not (0 <= row < rows and 0 <= col < cols):
            raise InvalidObstaclePlacementError(
                f"Obstacle center ({row}, {col}) is out of bounds for grid {(rows, cols)}."
            )
        if radius_cells < 0:
            raise InvalidObstaclePlacementError("radius_cells must be >= 0.")

        probe = DynamicRockObstacle(
            obstacle_id="__probe__",
            row=row,
            col=col,
            radius_cells=radius_cells,
        )
        cells = probe.footprint_cells(rows, cols)
        rover_pos = (int(rover_cell[0]), int(rover_cell[1]))
        goal_pos = (int(goal_cell[0]), int(goal_cell[1]))

        if rover_pos in cells:
            raise InvalidObstaclePlacementError(
                f"Cannot place obstacle at ({row}, {col}): overlaps current rover position {rover_pos}."
            )
        if goal_pos in cells:
            raise InvalidObstaclePlacementError(
                f"Cannot place obstacle at ({row}, {col}): overlaps destination {goal_pos}."
            )
        if start_cell is not None and (int(start_cell[0]), int(start_cell[1])) in cells:
            # Allow placing on start cell ONLY if rover already moved away from start
            if rover_pos == (int(start_cell[0]), int(start_cell[1])):
                raise InvalidObstaclePlacementError(
                    f"Cannot place obstacle at ({row}, {col}): overlaps rover start position {start_cell}."
                )
        return cells

    def add_obstacle(
        self,
        row: int,
        col: int,
        grid_shape: tuple[int, int],
        rover_cell: tuple[int, int],
        goal_cell: tuple[int, int],
        start_cell: tuple[int, int] | None = None,
        radius_cells: int = 0,
        height_m: float = 1.15,
        label: str = "Boulder",
        obstacle_id: str | None = None,
        is_draggable: bool = True,
    ) -> DynamicRockObstacle:
        """Validate and add a new rock obstacle."""
        r_i, c_i, rad_i = int(row), int(col), int(radius_cells)
        self._validate_cell(
            r_i, c_i, rad_i, grid_shape, rover_cell, goal_cell, start_cell=start_cell
        )
        if obstacle_id is None:
            self._counter += 1
            obs_id = f"rock_{self._counter:03d}"
            while obs_id in self._obstacles:
                self._counter += 1
                obs_id = f"rock_{self._counter:03d}"
        else:
            obs_id = str(obstacle_id)

        obs = DynamicRockObstacle(
            obstacle_id=obs_id,
            row=r_i,
            col=c_i,
            radius_cells=rad_i,
            height_m=float(height_m),
            label=str(label),
            is_draggable=bool(is_draggable),
        )
        self._obstacles[obs_id] = obs
        return obs

    def move_obstacle(
        self,
        obstacle_id: str,
        new_row: int,
        new_col: int,
        grid_shape: tuple[int, int],
        rover_cell: tuple[int, int],
        goal_cell: tuple[int, int],
        start_cell: tuple[int, int] | None = None,
    ) -> tuple[DynamicRockObstacle, tuple[int, int]]:
        """Move an existing obstacle to (new_row, new_col), returning (updated, previous_pos)."""
        if obstacle_id not in self._obstacles:
            raise KeyError(f"Unknown obstacle_id '{obstacle_id}'.")
        existing = self._obstacles[obstacle_id]
        nr, nc = int(new_row), int(new_col)
        self._validate_cell(
            nr,
            nc,
            existing.radius_cells,
            grid_shape,
            rover_cell,
            goal_cell,
            start_cell=start_cell,
        )
        prev = (existing.row, existing.col)
        updated = replace(existing, row=nr, col=nc)
        self._obstacles[obstacle_id] = updated
        return updated, prev

    def remove_obstacle(self, obstacle_id: str) -> DynamicRockObstacle:
        """Remove an existing obstacle by ID."""
        if obstacle_id not in self._obstacles:
            raise KeyError(f"Unknown obstacle_id '{obstacle_id}'.")
        return self._obstacles.pop(obstacle_id)

    def find_obstacle_at(
        self,
        row: int,
        col: int,
        grid_shape: tuple[int, int],
    ) -> DynamicRockObstacle | None:
        """Return the obstacle occupying (row, col), prioritizing exact center matches."""
        rows, cols = int(grid_shape[0]), int(grid_shape[1])
        target = (int(row), int(col))
        for obs in self._obstacles.values():
            if (obs.row, obs.col) == target:
                return obs
        for obs in self._obstacles.values():
            if target in obs.footprint_cells(rows, cols):
                return obs
        return None

    def rasterize_mask(self, grid_shape: tuple[int, int]) -> np.ndarray:
        """Rasterize all dynamic rock obstacles into a 2D boolean occupancy mask."""
        rows, cols = int(grid_shape[0]), int(grid_shape[1])
        mask = np.zeros((rows, cols), dtype=bool)
        for obs in self._obstacles.values():
            for rr, cc in obs.footprint_cells(rows, cols):
                mask[rr, cc] = True
        return mask

    def clear(self) -> None:
        self._obstacles.clear()
