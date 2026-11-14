"""Discrete rover path execution simulator with dynamic obstacle stop/replan events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from navigation.path_smoothing import bresenham_centerline


class SimulationStatus(str, Enum):
    """Lifecycle states for simulated rover navigation."""

    IDLE = "IDLE"
    MOVING = "MOVING"
    GOAL_REACHED = "GOAL_REACHED"
    BLOCKED_REPLAN_REQUIRED = "BLOCKED_REPLAN_REQUIRED"
    FAILED_INVALID_PATH = "FAILED_INVALID_PATH"


@dataclass(frozen=True)
class SimulationStepEvent:
    """Telemetry event emitted at each simulated movement attempt."""

    step_index: int
    position: tuple[int, int]
    status: SimulationStatus
    blocked_cell: tuple[int, int] | None
    message: str


@dataclass(frozen=True)
class SimulationResult:
    """Summary of a simulated rover traversal along a planned path."""

    final_status: SimulationStatus
    final_position: tuple[int, int]
    goal_position: tuple[int, int]
    visited_cells: list[tuple[int, int]]
    events: list[SimulationStepEvent]
    goal_reached: bool
    replan_required: bool
    remaining_path: list[tuple[int, int]]


def _expand_waypoints_to_steps(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Expand sparse smoothed waypoints into single-step grid cells."""
    if not path:
        return []
    expanded: list[tuple[int, int]] = [(int(path[0][0]), int(path[0][1]))]
    for target in path[1:]:
        curr = expanded[-1]
        nxt = (int(target[0]), int(target[1]))
        if curr == nxt:
            continue
        if max(abs(nxt[0] - curr[0]), abs(nxt[1] - curr[1])) <= 1:
            expanded.append(nxt)
        else:
            segment = bresenham_centerline(curr, nxt)
            expanded.extend(segment[1:])
    return expanded


class RoverSimulator:
    """Simulates discrete rover movement along a planned grid or waypoint path."""

    def __init__(self, start_position: tuple[int, int]) -> None:
        self.current_row: int = int(start_position[0])
        self.current_col: int = int(start_position[1])
        self.visited_cells: list[tuple[int, int]] = [(self.current_row, self.current_col)]
        self.status: SimulationStatus = SimulationStatus.IDLE

    @property
    def position(self) -> tuple[int, int]:
        """Return current rover grid coordinate as (row, column)."""
        return (self.current_row, self.current_col)

    def step_to(
        self,
        next_cell: tuple[int, int],
        blocked_mask: np.ndarray,
        step_index: int = 0,
        prevent_corner_cutting: bool = True,
    ) -> SimulationStepEvent:
        """Attempt a single step from the rover's current cell to `next_cell`."""
        mask = np.asarray(blocked_mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"blocked_mask must be 2D; got shape {mask.shape}.")
        rows, cols = mask.shape

        nr, nc = int(next_cell[0]), int(next_cell[1])
        if not (0 <= nr < rows and 0 <= nc < cols):
            self.status = SimulationStatus.FAILED_INVALID_PATH
            return SimulationStepEvent(
                step_index=step_index,
                position=self.position,
                status=self.status,
                blocked_cell=(nr, nc),
                message=f"Target cell {(nr, nc)} is out of bounds for grid {(rows, cols)}.",
            )

        if mask[nr, nc]:
            self.status = SimulationStatus.BLOCKED_REPLAN_REQUIRED
            return SimulationStepEvent(
                step_index=step_index,
                position=self.position,
                status=self.status,
                blocked_cell=(nr, nc),
                message=f"Blocked cell encountered at {(nr, nc)}; rover halted for replanning.",
            )

        dr = nr - self.current_row
        dc = nc - self.current_col
        if prevent_corner_cutting and abs(dr) == 1 and abs(dc) == 1:
            if mask[self.current_row + dr, self.current_col] or mask[self.current_row, self.current_col + dc]:
                blocked_corner = (
                    (self.current_row + dr, self.current_col)
                    if mask[self.current_row + dr, self.current_col]
                    else (self.current_row, self.current_col + dc)
                )
                self.status = SimulationStatus.BLOCKED_REPLAN_REQUIRED
                return SimulationStepEvent(
                    step_index=step_index,
                    position=self.position,
                    status=self.status,
                    blocked_cell=blocked_corner,
                    message=(
                        f"Diagonal move from {self.position} to {(nr, nc)} blocked by "
                        f"corner obstacle at {blocked_corner}; rover halted for replanning."
                    ),
                )

        self.current_row = nr
        self.current_col = nc
        self.visited_cells.append(self.position)
        self.status = SimulationStatus.MOVING

        return SimulationStepEvent(
            step_index=step_index,
            position=self.position,
            status=self.status,
            blocked_cell=None,
            message=f"Moved to {self.position}.",
        )

    def follow_path(
        self,
        path: list[tuple[int, int]],
        blocked_mask: np.ndarray,
        dynamic_obstacles_by_step: dict[int, list[tuple[int, int]]] | None = None,
        prevent_corner_cutting: bool = True,
    ) -> SimulationResult:
        """Simulate following `path` step-by-step until goal or safe obstacle halt."""
        mask = np.asarray(blocked_mask, dtype=bool).copy()
        if mask.ndim != 2:
            raise ValueError(f"blocked_mask must be 2D; got shape {mask.shape}.")

        if not path:
            self.status = SimulationStatus.FAILED_INVALID_PATH
            return SimulationResult(
                final_status=self.status,
                final_position=self.position,
                goal_position=self.position,
                visited_cells=list(self.visited_cells),
                events=[
                    SimulationStepEvent(
                        step_index=0,
                        position=self.position,
                        status=self.status,
                        blocked_cell=None,
                        message="Cannot simulate empty path.",
                    )
                ],
                goal_reached=False,
                replan_required=False,
                remaining_path=[],
            )

        steps = _expand_waypoints_to_steps(path)
        goal = steps[-1]
        events: list[SimulationStepEvent] = []

        if steps[0] != self.position:
            self.status = SimulationStatus.FAILED_INVALID_PATH
            return SimulationResult(
                final_status=self.status,
                final_position=self.position,
                goal_position=goal,
                visited_cells=list(self.visited_cells),
                events=[
                    SimulationStepEvent(
                        step_index=0,
                        position=self.position,
                        status=self.status,
                        blocked_cell=None,
                        message=f"Path start {steps[0]} does not match rover position {self.position}.",
                    )
                ],
                goal_reached=False,
                replan_required=False,
                remaining_path=steps,
            )

        rows, cols = mask.shape
        if not (0 <= self.current_row < rows and 0 <= self.current_col < cols):
            self.status = SimulationStatus.FAILED_INVALID_PATH
            return SimulationResult(
                final_status=self.status,
                final_position=self.position,
                goal_position=goal,
                visited_cells=list(self.visited_cells),
                events=[
                    SimulationStepEvent(
                        step_index=0,
                        position=self.position,
                        status=self.status,
                        blocked_cell=self.position,
                        message=f"Rover position {self.position} is out of bounds for grid {(rows, cols)}.",
                    )
                ],
                goal_reached=False,
                replan_required=False,
                remaining_path=steps,
            )

        if mask[self.position]:
            self.status = SimulationStatus.BLOCKED_REPLAN_REQUIRED
            return SimulationResult(
                final_status=self.status,
                final_position=self.position,
                goal_position=goal,
                visited_cells=list(self.visited_cells),
                events=[
                    SimulationStepEvent(
                        step_index=0,
                        position=self.position,
                        status=self.status,
                        blocked_cell=self.position,
                        message=f"Rover start cell {self.position} is blocked.",
                    )
                ],
                goal_reached=False,
                replan_required=True,
                remaining_path=steps,
            )

        if len(steps) == 1:
            self.status = SimulationStatus.GOAL_REACHED
            events.append(
                SimulationStepEvent(
                    step_index=0,
                    position=self.position,
                    status=self.status,
                    blocked_cell=None,
                    message=f"Already at goal {goal}.",
                )
            )
            return SimulationResult(
                final_status=self.status,
                final_position=self.position,
                goal_position=goal,
                visited_cells=list(self.visited_cells),
                events=events,
                goal_reached=True,
                replan_required=False,
                remaining_path=[],
            )

        rows, cols = mask.shape
        for step_idx, next_cell in enumerate(steps[1:], start=1):
            if dynamic_obstacles_by_step and step_idx in dynamic_obstacles_by_step:
                for obs_r, obs_c in dynamic_obstacles_by_step[step_idx]:
                    if 0 <= obs_r < rows and 0 <= obs_c < cols:
                        mask[obs_r, obs_c] = True

            event = self.step_to(
                next_cell,
                blocked_mask=mask,
                step_index=step_idx,
                prevent_corner_cutting=prevent_corner_cutting,
            )
            events.append(event)

            if event.status == SimulationStatus.BLOCKED_REPLAN_REQUIRED:
                return SimulationResult(
                    final_status=self.status,
                    final_position=self.position,
                    goal_position=goal,
                    visited_cells=list(self.visited_cells),
                    events=events,
                    goal_reached=False,
                    replan_required=True,
                    remaining_path=[self.position, *steps[step_idx:]],
                )

            if event.status == SimulationStatus.FAILED_INVALID_PATH:
                return SimulationResult(
                    final_status=self.status,
                    final_position=self.position,
                    goal_position=goal,
                    visited_cells=list(self.visited_cells),
                    events=events,
                    goal_reached=False,
                    replan_required=False,
                    remaining_path=steps[step_idx:],
                )

        self.status = SimulationStatus.GOAL_REACHED
        events.append(
            SimulationStepEvent(
                step_index=len(steps) - 1,
                position=self.position,
                status=self.status,
                blocked_cell=None,
                message=f"Goal {goal} reached safely.",
            )
        )
        return SimulationResult(
            final_status=self.status,
            final_position=self.position,
            goal_position=goal,
            visited_cells=list(self.visited_cells),
            events=events,
            goal_reached=True,
            replan_required=False,
            remaining_path=[],
        )
