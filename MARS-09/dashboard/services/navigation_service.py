"""Navigation & interactive simulation session service for the MARS-09 dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st

from simulation.dynamic_obstacles import InvalidObstaclePlacementError
from simulation.simulation_controller import RoverSimulationController


CONTROLLER_KEY = "mars09_sim_controller"
LAST_CANVAS_EVENT_KEY = "mars09_last_canvas_event_id"
ERROR_KEY = "mars09_last_error"


def get_controller() -> RoverSimulationController:
    """Retrieve or initialize the persistent RoverSimulationController in session state."""
    if CONTROLLER_KEY not in st.session_state:
        st.session_state[CONTROLLER_KEY] = RoverSimulationController(
            scenario_name="dynamic_obstacle",
            rows=30,
            cols=30,
            resolution_m=(2.0, 2.0),
        )
    if LAST_CANVAS_EVENT_KEY not in st.session_state:
        st.session_state[LAST_CANVAS_EVENT_KEY] = ""
    if ERROR_KEY not in st.session_state:
        st.session_state[ERROR_KEY] = ""
    return st.session_state[CONTROLLER_KEY]


def clear_error() -> None:
    st.session_state[ERROR_KEY] = ""


def pop_error() -> str:
    msg = str(st.session_state.get(ERROR_KEY, ""))
    st.session_state[ERROR_KEY] = ""
    return msg


def handle_canvas_action(
    controller: RoverSimulationController,
    event: dict[str, Any] | None,
) -> bool:
    """Process deduplicated bidirectional canvas events (drag rock, add/remove, paint, start/goal)."""
    if not event or not isinstance(event, dict):
        return False
    event_id = str(event.get("event_id", ""))
    if not event_id or event_id == st.session_state.get(LAST_CANVAS_EVENT_KEY):
        return False

    st.session_state[LAST_CANVAS_EVENT_KEY] = event_id
    action = str(event.get("action", ""))

    try:
        if action == "move_obstacle":
            controller.move_obstacle(
                obstacle_id=str(event["obstacle_id"]),
                new_row=int(event["row"]),
                new_col=int(event["col"]),
            )
        elif action == "add_obstacle":
            controller.add_obstacle(
                row=int(event["row"]),
                col=int(event["col"]),
                radius_cells=int(event.get("radius_cells", 0)),
                height_m=float(event.get("height_m", 1.25)),
                label=str(event.get("label", "Interactive Boulder")),
            )
        elif action == "remove_obstacle":
            obs_id = event.get("obstacle_id")
            if obs_id:
                controller.remove_obstacle(str(obs_id))
            else:
                controller.remove_obstacle_at(int(event["row"]), int(event["col"]))
        elif action == "paint_terrain":
            cells = [(int(c[0]), int(c[1])) for c in (event.get("cells") or [])]
            if not cells and "row" in event and "col" in event:
                cells = [(int(event["row"]), int(event["col"]))]
            controller.paint_terrain(cells=cells, terrain_type=str(event["terrain_type"]))
        elif action == "set_start":
            controller.set_start_goal(start_cell=(int(event["row"]), int(event["col"])))
        elif action == "set_goal":
            controller.set_start_goal(goal_cell=(int(event["row"]), int(event["col"])))
        st.session_state[ERROR_KEY] = ""
        return True
    except (InvalidObstaclePlacementError, KeyError, ValueError) as exc:
        st.session_state[ERROR_KEY] = str(exc)
        return False
