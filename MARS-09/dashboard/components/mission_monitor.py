"""Mission status banner, physics telemetry readouts, and replanning event log."""

from __future__ import annotations

import streamlit as st

from simulation.simulation_controller import RoverSimulationController
from simulation.simulation_state import MissionStatus


def render_mission_monitor(controller: RoverSimulationController) -> None:
    """Render live mission status, Martian terramechanics physics, and replan log."""
    payload = controller.to_payload(include_grids=False)
    status = MissionStatus(payload["mission_status"])
    alert_msg = payload["alert_message"]
    rover = payload["rover"]
    physics = payload["physics"]
    routes = payload["routes"]
    planner = payload["planner_metrics"]

    if status == MissionStatus.NO_SAFE_PATH:
        st.error(f"**{status.value}** — {alert_msg}")
    elif status in (MissionStatus.OBSTACLE_DETECTED, MissionStatus.REPLANNING, MissionStatus.ROUTE_UPDATED):
        st.warning(f"**{status.value}** — {alert_msg}")
    elif status == MissionStatus.GOAL_REACHED:
        st.success(f"**{status.value}** — {alert_msg}")
    else:
        st.info(f"**{status.value}** — {alert_msg}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rover Cell", f"({rover['row']}, {rover['col']})", rover["terrain_class"].upper())
    m2.metric("Elevation", f"{rover['elevation_m']:.1f} m", f"Pitch {physics['pitch_deg']:+.1f}°")
    m3.metric("Distance", f"{payload['distance_travelled_m']:.1f} m", f"Remaining {routes['remaining_steps']} steps")
    m4.metric("Replans", str(payload["replan_count"]), f"Cost {planner['total_cost'] if planner['total_cost'] is not None else 'INF'}")

    st.markdown("#### Martian Terramechanics & Physics (`g = 3.721 m/s²`)")
    p1, p2, p3 = st.columns(3)
    p1.metric("Ground Slope", f"{physics['grade_slope_deg']:.1f}°", f"Roll {physics['roll_deg']:+.1f}°")
    p2.metric("Wheel Slip", f"{physics['slip_ratio'] * 100.0:.1f}%", f"Sinkage {physics['sinkage_cm']:.1f} cm")
    p3.metric("Speed / Power", f"{physics['effective_speed_mps']:.2f} m/s", f"{physics['motor_power_w']:.0f} W ({physics['energy_wh']:.2f} Wh)")

    st.markdown("#### Replanning Event Log")
    events = payload["replan_events"]
    if not events:
        st.caption("No dynamic replanning events yet. Drag a rock onto the active route to test live UARA* avoidance.")
    else:
        for ev in events[:8]:
            st.text(
                f"t={ev['timestamp_sec']:5.1f}s | {ev['trigger']} | {ev['message']}"
            )
