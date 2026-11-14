"""Interactive 2D Mars Rover Simulation with Real-Time Dynamic Obstacle Avoidance."""

from __future__ import annotations

import time

import streamlit as st

from dashboard.components.interactive_rover_canvas import render_interactive_rover_canvas
from dashboard.components.mission_monitor import render_mission_monitor
from dashboard.components.simulation_controls import render_simulation_controls
from dashboard.services.navigation_service import get_controller, pop_error


def render_interactive_simulator_page() -> None:
    st.title("MARS-09 Interactive Mars Rover Simulator")
    st.caption(
        "Drag rocks directly on the Martian surface, add/remove obstacles, or paint terrain "
        "while the rover is moving to trigger live UARA* obstacle detection and route recalculation."
    )

    controller = get_controller()
    mode, terrain_brush, rock_radius = render_simulation_controls(controller)

    err = pop_error()
    if err:
        st.toast(err, icon="⚠️")

    col_map, col_mon = st.columns([1.15, 1.0], gap="large")
    with col_map:
        mutated = render_interactive_rover_canvas(
            controller=controller,
            interaction_mode=mode,
            terrain_brush=terrain_brush,
            rock_radius_cells=rock_radius,
        )
        if mutated:
            st.rerun()
    with col_mon:
        render_mission_monitor(controller)

    if controller.state.is_running:
        time.sleep(0.18)
        controller.step(0.25)
        st.rerun()


if __name__ == "__main__":
    render_interactive_simulator_page()
