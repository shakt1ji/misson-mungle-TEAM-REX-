"""Interactive Mars rover simulation controller with live UARA* replanning and physics."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from navigation.astar import NavigationResult, UARAConfig, plan_path_uara_star
from navigation.path_smoothing import smooth_path_line_of_sight
from navigation.rover_simulator import (
    RoverSimulator,
    SimulationStatus,
    _expand_waypoints_to_steps,
)
from simulation.dynamic_obstacles import (
    DynamicObstacleManager,
    DynamicRockObstacle,
    InvalidObstaclePlacementError,
)
from simulation.environment_editor import (
    EnvironmentBuildResult,
    SCENARIO_DESCRIPTIONS,
    apply_terrain_edit,
    build_scenario_environment,
    recompute_cost_map,
)
from simulation.physics import RoverPhysicsParams, step_rover_physics
from simulation.simulation_state import (
    MissionStatus,
    ReplanEvent,
    RoverSimulationState,
    serialize_simulation_state,
)
from terrain.terrain_cost import CostWeights, TerrainCostMap
from terrain.traversability import TerrainClassConfig, TraversabilityConfig


class RoverSimulationController:
    """Coordinates terrain cost maps, Member 1 UARA* planner, RoverSimulator, and physics."""

    def __init__(
        self,
        scenario_name: str = "dynamic_obstacle",
        rows: int = 30,
        cols: int = 30,
        resolution_m: tuple[float, float] = (2.0, 2.0),
        uara_config: UARAConfig | None = None,
        physics_params: RoverPhysicsParams | None = None,
    ) -> None:
        self._rows = int(rows)
        self._cols = int(cols)
        self._resolution_m = (float(resolution_m[0]), float(resolution_m[1]))
        self._uara_config = uara_config or UARAConfig(
            connectivity=8,
            safety_weight=1.0,
            uncertainty_weight=1.0,
            heuristic_weight=1.0,
            prevent_corner_cutting=True,
        )
        self._physics_params = physics_params or RoverPhysicsParams()

        self._class_config: TerrainClassConfig
        self._traversability_config: TraversabilityConfig
        self._weights: CostWeights
        self._obstacle_manager = DynamicObstacleManager()
        self._rover_sim: RoverSimulator
        self._state: RoverSimulationState
        self._event_counter = 0

        self.load_scenario(scenario_name)

    @property
    def state(self) -> RoverSimulationState:
        return self._state

    @property
    def obstacle_manager(self) -> DynamicObstacleManager:
        return self._obstacle_manager

    def to_payload(self, include_grids: bool = True) -> dict[str, Any]:
        return serialize_simulation_state(
            self._state,
            self._obstacle_manager.obstacles,
            include_grids=include_grids,
        )

    def load_scenario(self, scenario_name: str) -> RoverSimulationState:
        """Load one of the 5 demonstration scenarios and plan the initial UARA* route."""
        env: EnvironmentBuildResult = build_scenario_environment(
            scenario_name=scenario_name,
            rows=self._rows,
            cols=self._cols,
            resolution_m=self._resolution_m,
        )
        self._class_config = env.class_config
        self._traversability_config = env.traversability_config
        self._weights = env.weights
        self._obstacle_manager = DynamicObstacleManager(env.obstacles)
        self._event_counter = 0

        dyn_mask = self._obstacle_manager.rasterize_mask(env.dem.elevation.shape)
        slope_deg, cost_map = recompute_cost_map(
            dem=env.dem,
            terrain_class_ids=env.terrain_class_ids,
            class_probabilities=env.class_probabilities,
            dynamic_obstacle_mask=dyn_mask,
            static_blocked_mask=env.static_blocked_mask,
            class_config=self._class_config,
            traversability_config=self._traversability_config,
            weights=self._weights,
            obstacle_buffer_cells=1,
        )

        self._rover_sim = RoverSimulator(env.start)
        self._state = RoverSimulationState(
            scenario_name=scenario_name,
            dem=env.dem,
            slope_deg=slope_deg,
            terrain_class_ids=env.terrain_class_ids,
            class_probabilities=env.class_probabilities,
            static_blocked_mask=env.static_blocked_mask,
            cost_map=cost_map,
            start_cell=env.start,
            goal_cell=env.goal,
            rover_cell=env.start,
            rover_row_f=float(env.start[0]),
            rover_col_f=float(env.start[1]),
            heading_rad=self._initial_heading(env.start, env.goal),
            mission_status=MissionStatus.IDLE,
            is_running=False,
            travelled_cells=[env.start],
            travelled_points=[(float(env.start[0]), float(env.start[1]))],
        )

        self._plan_initial_route(record_original=True)
        self._update_physics(dt_sec=0.0, is_moving=False)
        return self._state

    def reset(self) -> RoverSimulationState:
        """Reset current scenario back to its initial state."""
        speed = self._state.speed_multiplier
        auto_replan = self._state.auto_replan
        show_cost = self._state.show_cost_overlay
        show_unc = self._state.show_uncertainty_overlay
        show_prev = self._state.show_previous_route
        scenario = self._state.scenario_name

        self.load_scenario(scenario)
        self._state.speed_multiplier = speed
        self._state.auto_replan = auto_replan
        self._state.show_cost_overlay = show_cost
        self._state.show_uncertainty_overlay = show_unc
        self._state.show_previous_route = show_prev
        return self._state

    def start(self) -> RoverSimulationState:
        """Start or resume autonomous rover motion."""
        if self._state.mission_status == MissionStatus.GOAL_REACHED:
            return self._state
        if self._state.mission_status == MissionStatus.NO_SAFE_PATH:
            self._state.is_running = False
            self._state.alert_message = "No safe path available — remove obstacle or edit terrain."
            return self._state
        if self._state.mission_status == MissionStatus.OBSTACLE_DETECTED:
            # Attempt replan before resuming
            self.replan_from_current_position(trigger="manual_resume")
            if self._state.mission_status == MissionStatus.NO_SAFE_PATH:
                return self._state

        if self._state.rover_cell == self._state.goal_cell:
            self._state.mission_status = MissionStatus.GOAL_REACHED
            self._state.is_running = False
            return self._state

        if not self._state.expanded_route:
            self.replan_from_current_position(trigger="start_replan")
            if self._state.mission_status == MissionStatus.NO_SAFE_PATH:
                return self._state

        self._state.is_running = True
        self._state.mission_status = MissionStatus.MOVING
        self._state.alert_message = "Rover traversing autonomous UARA* route."
        return self._state

    def pause(self) -> RoverSimulationState:
        """Pause rover motion while preserving exact continuous state and route."""
        self._state.is_running = False
        if self._state.mission_status in (MissionStatus.MOVING, MissionStatus.ROUTE_UPDATED):
            self._state.mission_status = MissionStatus.IDLE
            self._state.alert_message = "Simulation paused — modify obstacles or press Resume."
        self._update_physics(dt_sec=0.0, is_moving=False)
        return self._state

    def resume(self) -> RoverSimulationState:
        return self.start()

    def set_controls(
        self,
        speed_multiplier: float | None = None,
        auto_replan: bool | None = None,
        show_cost_overlay: bool | None = None,
        show_uncertainty_overlay: bool | None = None,
        show_previous_route: bool | None = None,
    ) -> RoverSimulationState:
        """Update interactive simulation control flags."""
        if speed_multiplier is not None:
            val = float(speed_multiplier)
            if val <= 0.0:
                raise ValueError("speed_multiplier must be positive.")
            self._state.speed_multiplier = float(np.clip(val, 0.25, 8.0))
        if auto_replan is not None:
            was_off = not self._state.auto_replan
            self._state.auto_replan = bool(auto_replan)
            if was_off and self._state.auto_replan and self._state.mission_status == MissionStatus.OBSTACLE_DETECTED:
                self.replan_from_current_position(trigger="auto_replan_enabled")
        if show_cost_overlay is not None:
            self._state.show_cost_overlay = bool(show_cost_overlay)
        if show_uncertainty_overlay is not None:
            self._state.show_uncertainty_overlay = bool(show_uncertainty_overlay)
        if show_previous_route is not None:
            self._state.show_previous_route = bool(show_previous_route)
        return self._state

    def add_obstacle(
        self,
        row: int,
        col: int,
        radius_cells: int = 0,
        height_m: float = 1.25,
        label: str = "Boulder",
        obstacle_id: str | None = None,
    ) -> DynamicRockObstacle:
        """Add a rock obstacle and trigger live collision check + UARA* replanning."""
        obs = self._obstacle_manager.add_obstacle(
            row=row,
            col=col,
            grid_shape=self._state.dem.elevation.shape,
            rover_cell=self._state.rover_cell,
            goal_cell=self._state.goal_cell,
            start_cell=self._state.start_cell,
            radius_cells=radius_cells,
            height_m=height_m,
            label=label,
            obstacle_id=obstacle_id,
        )
        self._on_environment_mutated(trigger=f"obstacle_added:{obs.obstacle_id}")
        return obs

    def move_obstacle(
        self,
        obstacle_id: str,
        new_row: int,
        new_col: int,
    ) -> DynamicRockObstacle:
        """Move an existing rock obstacle (mouse drag-and-drop) and trigger replanning."""
        obs, prev = self._obstacle_manager.move_obstacle(
            obstacle_id=obstacle_id,
            new_row=new_row,
            new_col=new_col,
            grid_shape=self._state.dem.elevation.shape,
            rover_cell=self._state.rover_cell,
            goal_cell=self._state.goal_cell,
            start_cell=self._state.start_cell,
        )
        if (obs.row, obs.col) != prev:
            self._on_environment_mutated(trigger=f"obstacle_moved:{obs.obstacle_id}")
        return obs

    def remove_obstacle(self, obstacle_id: str) -> DynamicRockObstacle:
        """Remove an existing rock obstacle and recover path if previously blocked."""
        removed = self._obstacle_manager.remove_obstacle(obstacle_id)
        self._on_environment_mutated(trigger=f"obstacle_removed:{removed.obstacle_id}")
        return removed

    def remove_obstacle_at(self, row: int, col: int) -> DynamicRockObstacle | None:
        """Remove the rock obstacle at (row, col), if one exists."""
        obs = self._obstacle_manager.find_obstacle_at(row, col, self._state.dem.elevation.shape)
        if obs is None:
            return None
        return self.remove_obstacle(obs.obstacle_id)

    def paint_terrain(
        self,
        cells: Iterable[tuple[int, int]],
        terrain_type: str,
    ) -> int:
        """Edit terrain class or blocked status at specified cells and update cost/route."""
        cell_list = [(int(r), int(c)) for r, c in cells]
        if not cell_list:
            return 0
        if terrain_type.strip().lower() == "blocked_region":
            if self._state.rover_cell in cell_list:
                raise InvalidObstaclePlacementError(
                    f"Cannot block current rover position {self._state.rover_cell}."
                )
            if self._state.goal_cell in cell_list:
                raise InvalidObstaclePlacementError(
                    f"Cannot block destination {self._state.goal_cell}."
                )
        updated = apply_terrain_edit(
            terrain_class_ids=self._state.terrain_class_ids,
            class_probabilities=self._state.class_probabilities,
            static_blocked_mask=self._state.static_blocked_mask,
            cells=cell_list,
            terrain_type=terrain_type,
        )
        self._on_environment_mutated(trigger=f"terrain_edit:{terrain_type}")
        return updated

    def set_start_goal(
        self,
        start_cell: tuple[int, int] | None = None,
        goal_cell: tuple[int, int] | None = None,
    ) -> RoverSimulationState:
        """Update rover start or destination position and recompute UARA* route."""
        rows, cols = self._state.dem.elevation.shape
        new_start = (
            (int(start_cell[0]), int(start_cell[1]))
            if start_cell is not None
            else self._state.start_cell
        )
        new_goal = (
            (int(goal_cell[0]), int(goal_cell[1]))
            if goal_cell is not None
            else self._state.goal_cell
        )

        for label, cell in (("start", new_start), ("goal", new_goal)):
            if not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
                raise InvalidObstaclePlacementError(
                    f"New {label} cell {cell} is out of bounds for {(rows, cols)}."
                )
            if self._state.cost_map.blocked_mask[cell]:
                raise InvalidObstaclePlacementError(
                    f"Cannot place {label} on blocked cell {cell}."
                )

        if start_cell is not None:
            self._state.start_cell = new_start
            self._state.rover_cell = new_start
            self._state.rover_row_f = float(new_start[0])
            self._state.rover_col_f = float(new_start[1])
            self._state.travelled_cells = [new_start]
            self._state.travelled_points = [(float(new_start[0]), float(new_start[1]))]
            self._state.distance_travelled_m = 0.0
            self._state.is_running = False
            self._state.mission_status = MissionStatus.IDLE
            self._rover_sim = RoverSimulator(new_start)

        if goal_cell is not None:
            self._state.goal_cell = new_goal
            if self._state.mission_status == MissionStatus.GOAL_REACHED and self._state.rover_cell != new_goal:
                self._state.mission_status = MissionStatus.IDLE

        self._state.heading_rad = self._initial_heading(self._state.rover_cell, self._state.goal_cell)
        self._plan_initial_route(record_original=(start_cell is not None))
        self._update_physics(dt_sec=0.0, is_moving=False)
        return self._state

    def step(self, dt_sec: float = 0.25) -> RoverSimulationState:
        """Advance continuous Martian physics and rover traversal by `dt_sec`."""
        dt = max(0.0, float(dt_sec))
        if not self._state.is_running:
            self._update_physics(dt_sec=0.0, is_moving=False)
            return self._state

        if self._state.mission_status == MissionStatus.GOAL_REACHED:
            self._state.is_running = False
            self._update_physics(dt_sec=0.0, is_moving=False)
            return self._state

        if self._state.mission_status in (MissionStatus.NO_SAFE_PATH, MissionStatus.OBSTACLE_DETECTED):
            self._state.is_running = False
            self._update_physics(dt_sec=0.0, is_moving=False)
            return self._state

        if self._state.mission_status == MissionStatus.ROUTE_UPDATED:
            self._state.mission_status = MissionStatus.MOVING

        # Safety check before moving: never enter a blocked cell
        if self._is_remaining_route_blocked():
            self._handle_route_blocked(trigger="lookahead_obstacle_check")
            if self._state.mission_status in (
                MissionStatus.NO_SAFE_PATH,
                MissionStatus.OBSTACLE_DETECTED,
            ):
                self._update_physics(dt_sec=0.0, is_moving=False)
                return self._state

        expanded = self._state.expanded_route
        if not expanded or self._state.route_step_index >= len(expanded) - 1:
            if self._state.rover_cell == self._state.goal_cell:
                self._finish_goal()
            else:
                self.replan_from_current_position(trigger="empty_route_guard")
            return self._state

        self._state.sim_time_sec += dt
        telemetry = self._update_physics(dt_sec=dt, is_moving=True)
        travel_budget_m = max(0.05, telemetry.effective_speed_mps * dt)

        dy_m, dx_m = self._state.dem.resolution_m
        while travel_budget_m > 1e-6 and self._state.route_step_index < len(expanded) - 1:
            curr_cell = expanded[self._state.route_step_index]
            next_cell = expanded[self._state.route_step_index + 1]

            if self._step_pair_blocked(curr_cell, next_cell):
                self._handle_route_blocked(trigger="runtime_collision_guard")
                break

            seg_len_m = math.hypot(
                (next_cell[0] - curr_cell[0]) * dy_m,
                (next_cell[1] - curr_cell[1]) * dx_m,
            )
            seg_len_m = max(seg_len_m, 1e-6)
            self._state.heading_rad = math.atan2(
                next_cell[0] - curr_cell[0],
                next_cell[1] - curr_cell[1],
            )

            remaining_in_seg_m = (1.0 - self._state.segment_progress) * seg_len_m
            if travel_budget_m >= remaining_in_seg_m - 1e-9:
                # Use Member 1 RoverSimulator to verify and record integer cell transition
                if next_cell != self._rover_sim.position:
                    sim_ev = self._rover_sim.step_to(
                        next_cell,
                        blocked_mask=self._state.cost_map.blocked_mask,
                        step_index=self._state.route_step_index + 1,
                        prevent_corner_cutting=self._uara_config.prevent_corner_cutting,
                    )
                    if sim_ev.status != SimulationStatus.MOVING:
                        self._handle_route_blocked(trigger="rover_simulator_halt")
                        break

                travel_budget_m -= remaining_in_seg_m
                self._state.distance_travelled_m += remaining_in_seg_m
                self._state.route_step_index += 1
                self._state.segment_progress = 0.0
                self._state.rover_cell = next_cell
                self._state.rover_row_f = float(next_cell[0])
                self._state.rover_col_f = float(next_cell[1])

                if not self._state.travelled_cells or self._state.travelled_cells[-1] != next_cell:
                    self._state.travelled_cells.append(next_cell)
                self._append_travelled_point(self._state.rover_row_f, self._state.rover_col_f)
                self._sync_active_route_tail()

                if next_cell == self._state.goal_cell:
                    self._finish_goal()
                    break
            else:
                frac_gain = travel_budget_m / seg_len_m
                self._state.segment_progress = min(1.0, self._state.segment_progress + frac_gain)
                self._state.distance_travelled_m += travel_budget_m
                travel_budget_m = 0.0
                p = self._state.segment_progress
                self._state.rover_row_f = curr_cell[0] + (next_cell[0] - curr_cell[0]) * p
                self._state.rover_col_f = curr_cell[1] + (next_cell[1] - curr_cell[1]) * p
                self._append_travelled_point(self._state.rover_row_f, self._state.rover_col_f)

        self._update_physics(dt_sec=0.0, is_moving=self._state.is_running)
        return self._state

    def replan_from_current_position(self, trigger: str = "manual_replan") -> NavigationResult:
        """Recompute safe UARA* path strictly from the rover's current position to goal."""
        old_remaining = (
            max(0, len(self._state.expanded_route) - 1 - self._state.route_step_index)
            if self._state.expanded_route
            else 0
        )
        if self._state.active_route:
            self._state.previous_route = list(self._state.active_route)

        self._state.mission_status = MissionStatus.REPLANNING
        result = plan_path_uara_star(
            cost_map=self._state.cost_map,
            start=self._state.rover_cell,
            goal=self._state.goal_cell,
            config=self._uara_config,
        )
        self._state.last_plan_result = result
        self._event_counter += 1

        if result.success:
            smoothed = smooth_path_line_of_sight(
                result.path,
                blocked_mask=self._state.cost_map.blocked_mask,
                prevent_corner_cutting=self._uara_config.prevent_corner_cutting,
                cost_grid=self._state.cost_map.total_cost,
            )
            expanded = _expand_waypoints_to_steps(smoothed)
            self._state.active_route = smoothed
            self._state.expanded_route = expanded
            self._state.route_step_index = 0
            self._state.segment_progress = 0.0
            self._state.rover_row_f = float(self._state.rover_cell[0])
            self._state.rover_col_f = float(self._state.rover_cell[1])
            if len(expanded) >= 2:
                self._state.heading_rad = math.atan2(
                    expanded[1][0] - expanded[0][0],
                    expanded[1][1] - expanded[0][1],
                )
            self._state.replan_count += 1
            self._state.mission_status = (
                MissionStatus.GOAL_REACHED
                if self._state.rover_cell == self._state.goal_cell
                else MissionStatus.ROUTE_UPDATED
            )
            self._state.alert_seq += 1
            self._state.alert_message = "Obstacle detected — recalculating route"
            self._state.replan_events.insert(
                0,
                ReplanEvent(
                    event_id=self._event_counter,
                    timestamp_sec=self._state.sim_time_sec,
                    trigger=trigger,
                    rover_position=self._state.rover_cell,
                    old_remaining_steps=old_remaining,
                    new_steps=max(0, len(expanded) - 1),
                    new_cost=result.total_cost,
                    nodes_expanded=result.nodes_expanded,
                    success=True,
                    message=(
                        f"Replanned from {self._state.rover_cell} -> {len(expanded) - 1} steps "
                        f"(cost {result.total_cost:.2f})"
                    ),
                ),
            )
        else:
            self._state.active_route = []
            self._state.expanded_route = []
            self._state.route_step_index = 0
            self._state.segment_progress = 0.0
            self._state.is_running = False
            self._state.mission_status = MissionStatus.NO_SAFE_PATH
            self._state.alert_seq += 1
            self._state.alert_message = "No safe path available — remove an obstacle to resume."
            self._state.replan_events.insert(
                0,
                ReplanEvent(
                    event_id=self._event_counter,
                    timestamp_sec=self._state.sim_time_sec,
                    trigger=trigger,
                    rover_position=self._state.rover_cell,
                    old_remaining_steps=old_remaining,
                    new_steps=0,
                    new_cost=float("inf"),
                    nodes_expanded=result.nodes_expanded,
                    success=False,
                    message=f"No safe path from {self._state.rover_cell} to {self._state.goal_cell} ({result.termination_reason}).",
                ),
            )
        # Cap event history cleanly
        self._state.replan_events = self._state.replan_events[:30]
        return result

    def _plan_initial_route(self, record_original: bool = True) -> None:
        result = plan_path_uara_star(
            cost_map=self._state.cost_map,
            start=self._state.rover_cell,
            goal=self._state.goal_cell,
            config=self._uara_config,
        )
        self._state.last_plan_result = result
        if result.success:
            smoothed = smooth_path_line_of_sight(
                result.path,
                blocked_mask=self._state.cost_map.blocked_mask,
                prevent_corner_cutting=self._uara_config.prevent_corner_cutting,
                cost_grid=self._state.cost_map.total_cost,
            )
            expanded = _expand_waypoints_to_steps(smoothed)
            if record_original:
                self._state.original_route = list(smoothed)
            self._state.active_route = smoothed
            self._state.expanded_route = expanded
            self._state.route_step_index = 0
            self._state.segment_progress = 0.0
            if len(expanded) >= 2:
                self._state.heading_rad = math.atan2(
                    expanded[1][0] - expanded[0][0],
                    expanded[1][1] - expanded[0][1],
                )
            desc = SCENARIO_DESCRIPTIONS.get(self._state.scenario_name, self._state.scenario_name)
            self._state.alert_message = f"Route ready ({len(expanded) - 1} steps). {desc}"
        else:
            if record_original:
                self._state.original_route = []
            self._state.active_route = []
            self._state.expanded_route = []
            self._state.mission_status = MissionStatus.NO_SAFE_PATH
            self._state.is_running = False
            self._state.alert_seq += 1
            self._state.alert_message = "No safe path available — remove an obstacle to unlock route."

    def _on_environment_mutated(self, trigger: str) -> None:
        """Recompute DEM slope/cost map and trigger replanning if path is impacted or blocked."""
        dyn_mask = self._obstacle_manager.rasterize_mask(self._state.dem.elevation.shape)
        slope_deg, cost_map = recompute_cost_map(
            dem=self._state.dem,
            terrain_class_ids=self._state.terrain_class_ids,
            class_probabilities=self._state.class_probabilities,
            dynamic_obstacle_mask=dyn_mask,
            static_blocked_mask=self._state.static_blocked_mask,
            class_config=self._class_config,
            traversability_config=self._traversability_config,
            weights=self._weights,
            obstacle_buffer_cells=1,
        )
        self._state.slope_deg = slope_deg
        self._state.cost_map = cost_map

        if self._state.mission_status == MissionStatus.GOAL_REACHED:
            self._update_physics(dt_sec=0.0, is_moving=False)
            return

        # Recover automatically from NO_SAFE_PATH when an obstacle is removed or terrain cleared
        if self._state.mission_status == MissionStatus.NO_SAFE_PATH:
            self.replan_from_current_position(trigger=f"recovery:{trigger}")
            if self._state.mission_status == MissionStatus.ROUTE_UPDATED:
                self._state.alert_message = "Path cleared — safe UARA* route restored."
            self._update_physics(dt_sec=0.0, is_moving=False)
            return

        if self._is_remaining_route_blocked():
            self._handle_route_blocked(trigger=trigger)
        elif trigger.startswith("terrain_edit:") and self._state.auto_replan:
            # High-cost sand or uncertainty paint should let UARA* optimize around cost spikes
            self.replan_from_current_position(trigger=trigger)

        self._update_physics(dt_sec=0.0, is_moving=self._state.is_running)

    def _step_pair_blocked(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        blocked = self._state.cost_map.blocked_mask
        if blocked[b]:
            return True
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if self._uara_config.prevent_corner_cutting and abs(dr) == 1 and abs(dc) == 1:
            if blocked[a[0] + dr, a[1]] or blocked[a[0], a[1] + dc]:
                return True
        return False

    def _is_remaining_route_blocked(self) -> bool:
        expanded = self._state.expanded_route
        if not expanded:
            return self._state.rover_cell != self._state.goal_cell
        blocked = self._state.cost_map.blocked_mask
        idx = max(0, self._state.route_step_index)
        for i in range(idx, len(expanded)):
            if blocked[expanded[i]]:
                return True
            if i + 1 < len(expanded) and self._step_pair_blocked(expanded[i], expanded[i + 1]):
                return True
        return False

    def _handle_route_blocked(self, trigger: str) -> None:
        """Snap rover safely to current cell, emit OBSTACLE_DETECTED, and replan if enabled."""
        self._state.segment_progress = 0.0
        self._state.rover_row_f = float(self._state.rover_cell[0])
        self._state.rover_col_f = float(self._state.rover_cell[1])
        self._state.mission_status = MissionStatus.OBSTACLE_DETECTED
        self._state.alert_seq += 1
        self._state.alert_message = "Obstacle detected — recalculating route"

        if self._state.auto_replan:
            self.replan_from_current_position(trigger=trigger)
        else:
            self._state.is_running = False

    def _finish_goal(self) -> None:
        self._state.is_running = False
        self._state.mission_status = MissionStatus.GOAL_REACHED
        self._state.active_route = [self._state.goal_cell]
        self._state.alert_seq += 1
        self._state.alert_message = f"Destination {self._state.goal_cell} reached safely."
        self._update_physics(dt_sec=0.0, is_moving=False)

    def _sync_active_route_tail(self) -> None:
        rem = self._state.expanded_route[self._state.route_step_index :]
        if rem:
            self._state.active_route = smooth_path_line_of_sight(
                rem,
                blocked_mask=self._state.cost_map.blocked_mask,
                prevent_corner_cutting=self._uara_config.prevent_corner_cutting,
            )

    def _append_travelled_point(self, row_f: float, col_f: float) -> None:
        pts = self._state.travelled_points
        if not pts or math.hypot(row_f - pts[-1][0], col_f - pts[-1][1]) >= 0.12:
            pts.append((float(row_f), float(col_f)))

    def _update_physics(self, dt_sec: float, is_moving: bool) -> Any:
        prev_energy = self._state.physics.energy_wh if self._state.physics else 0.0
        prev_spin = self._state.physics.wheel_spin_rad if self._state.physics else 0.0
        telemetry = step_rover_physics(
            elevation=self._state.dem.elevation,
            slope_deg=self._state.slope_deg,
            valid_mask=self._state.dem.valid_mask,
            terrain_class_ids=self._state.terrain_class_ids,
            resolution_m=self._state.dem.resolution_m,
            row_f=self._state.rover_row_f,
            col_f=self._state.rover_col_f,
            heading_rad=self._state.heading_rad,
            speed_multiplier=self._state.speed_multiplier,
            dt_sec=dt_sec,
            is_moving=is_moving,
            prev_energy_wh=prev_energy,
            prev_wheel_spin_rad=prev_spin,
            params=self._physics_params,
        )
        self._state.physics = telemetry
        return telemetry

    @staticmethod
    def _initial_heading(start: tuple[int, int], goal: tuple[int, int]) -> float:
        return math.atan2(goal[0] - start[0], goal[1] - start[1])
