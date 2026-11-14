"""Route Planning & Terrain Intelligence Analysis page."""

from __future__ import annotations

import streamlit as st

from dashboard.services.navigation_service import get_controller
from dashboard.services.terrain_service import summarize_terrain
from dashboard.visualizations.route_map import (
    build_elevation_3d_figure,
    build_tactical_route_figure,
)


def render_route_planning_page() -> None:
    st.title("MARS-09 Terrain Intelligence & UARA* Route Analysis")
    controller = get_controller()
    stats = summarize_terrain(controller)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid Size", f"{stats['grid_shape'][0]} x {stats['grid_shape'][1]}")
    c2.metric("Elevation Range", f"{stats['elevation_min_m']:.1f}–{stats['elevation_max_m']:.1f} m")
    c3.metric("Mean / Max Slope", f"{stats['slope_mean_deg']:.1f}° / {stats['slope_max_deg']:.1f}°")
    c4.metric("Blocked Cells", f"{stats['blocked_cells']} ({stats['blocked_pct']}%)")

    layer = st.radio(
        "Inspection Layer",
        options=["total_cost", "elevation", "slope", "uncertainty"],
        format_func=lambda k: k.replace("_", " ").title(),
        horizontal=True,
    )

    col_left, col_right = st.columns([1.15, 1.0])
    with col_left:
        st.plotly_chart(build_tactical_route_figure(controller, layer=layer), use_container_width=True)
    with col_right:
        st.plotly_chart(build_elevation_3d_figure(controller), use_container_width=True)

    st.subheader("AI4MARS Terrain Class Breakdown")
    st.bar_chart(stats["class_counts"])


if __name__ == "__main__":
    render_route_planning_page()
