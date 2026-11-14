"""Streamlit control bar, scenario selector, and interactive editor tools."""

from __future__ import annotations

import streamlit as st

from simulation.environment_editor import SCENARIO_DESCRIPTIONS
from simulation.simulation_controller import RoverSimulationController
from simulation.simulation_state import MissionStatus


INTERACTION_MODES: dict[str, str] = {
    "drag": "Drag / Move Rock",
    "add": "Add Rock Obstacle",
    "remove": "Remove Rock Obstacle",
    "paint": "Paint Terrain Brush",
    "start": "Move Rover Start",
    "goal": "Move Destination",
}

TERRAIN_BRUSH_OPTIONS: list[str] = [
    "soil",
    "bedrock",
    "sand",
    "big_rock",
    "unknown",
    "blocked_region",
]


def render_simulation_controls(
    controller: RoverSimulationController,
) -> tuple[str, str, int]:
    """Render transport buttons, scenario dropdown, speed control, and editor tools."""
    state = controller.state

    c1, c2, c3, c4, c5 = st.columns([1.2, 1.0, 1.0, 1.0, 1.8])
    with c1:
        if not state.is_running:
            label = "Resume" if state.sim_time_sec > 0 and state.mission_status != MissionStatus.GOAL_REACHED else "Start Mission"
            if st.button(label, type="primary", use_container_width=True):
                controller.start()
                st.rerun()
        else:
            if st.button("Pause", use_container_width=True):
                controller.pause()
                st.rerun()
    with c2:
        if st.button("Step +0.5s", use_container_width=True, disabled=state.is_running):
            was_running = state.is_running
            controller._state.is_running = True
            controller.step(0.5)
            if not was_running and state.mission_status == MissionStatus.MOVING:
                controller.pause()
            st.rerun()
    with c3:
        if st.button("Replan Now", use_container_width=True):
            controller.replan_from_current_position(trigger="manual_button")
            st.rerun()
    with c4:
        if st.button("Reset", use_container_width=True):
            controller.reset()
            st.rerun()
    with c5:
        speed_options = [0.5, 1.0, 2.0, 4.0]
        cur_speed = state.speed_multiplier if state.speed_multiplier in speed_options else 1.0
        selected_speed = st.select_slider(
            "Rover Speed",
            options=speed_options,
            value=cur_speed,
            format_func=lambda v: f"{v:.1f}x",
        )
        if abs(selected_speed - state.speed_multiplier) > 1e-6:
            controller.set_controls(speed_multiplier=selected_speed)

    with st.sidebar:
        st.subheader("Mission Scenario")
        scenario_keys = list(SCENARIO_DESCRIPTIONS.keys())
        cur_idx = scenario_keys.index(state.scenario_name) if state.scenario_name in scenario_keys else 0
        chosen_scenario = st.selectbox(
            "Select Scenario",
            options=scenario_keys,
            index=cur_idx,
            format_func=lambda k: k.replace("_", " ").title(),
        )
        if chosen_scenario != state.scenario_name:
            controller.load_scenario(chosen_scenario)
            st.rerun()
        st.caption(SCENARIO_DESCRIPTIONS.get(chosen_scenario, ""))

        st.divider()
        st.subheader("Interactive Canvas Tool")
        mode = st.radio(
            "Canvas Mode",
            options=list(INTERACTION_MODES.keys()),
            format_func=lambda k: INTERACTION_MODES[k],
        )

        rock_radius = st.radio(
            "New Rock Size",
            options=[0, 1],
            format_func=lambda r: "Single Cell (1x1)" if r == 0 else "Large Boulder (3x3)",
            horizontal=True,
            disabled=(mode != "add"),
        )
        terrain_brush = st.selectbox(
            "Terrain Brush Type",
            options=TERRAIN_BRUSH_OPTIONS,
            disabled=(mode != "paint"),
        )

        st.divider()
        st.subheader("Planner & Layer Overlays")
        auto_replan = st.checkbox("Auto-Replan on Obstacles", value=state.auto_replan)
        show_cost = st.checkbox("Terrain Cost Heatmap", value=state.show_cost_overlay)
        show_unc = st.checkbox("AI4MARS Uncertainty Overlay", value=state.show_uncertainty_overlay)
        show_prev = st.checkbox("Show Previous Route", value=state.show_previous_route)
        if (
            auto_replan != state.auto_replan
            or show_cost != state.show_cost_overlay
            or show_unc != state.show_uncertainty_overlay
            or show_prev != state.show_previous_route
        ):
            controller.set_controls(
                auto_replan=auto_replan,
                show_cost_overlay=show_cost,
                show_uncertainty_overlay=show_unc,
                show_previous_route=show_prev,
            )
            st.rerun()

    return mode, terrain_brush, int(rock_radius)
