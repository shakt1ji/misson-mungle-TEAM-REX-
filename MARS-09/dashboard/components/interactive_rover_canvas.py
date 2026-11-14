"""Streamlit wrapper for the interactive 2D drag-and-drop Mars rover canvas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

from dashboard.services.navigation_service import handle_canvas_action
from simulation.simulation_controller import RoverSimulationController


_FRONTEND_DIR = Path(__file__).resolve().parent / "rover_canvas_frontend"
_rover_canvas_component = components.declare_component(
    "mars09_interactive_canvas",
    path=str(_FRONTEND_DIR),
)


def render_interactive_rover_canvas(
    controller: RoverSimulationController,
    interaction_mode: str,
    terrain_brush: str,
    rock_radius_cells: int,
) -> bool:
    """Render the interactive canvas and dispatch any drag/click action into `controller`."""
    payload = controller.to_payload(include_grids=True)
    event: dict[str, Any] | None = _rover_canvas_component(
        sim_state=payload,
        interaction_mode=interaction_mode,
        terrain_brush=terrain_brush,
        rock_radius_cells=int(rock_radius_cells),
        key="mars09_canvas_widget",
        default=None,
    )
    return handle_canvas_action(controller, event)
