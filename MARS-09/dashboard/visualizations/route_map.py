"""Plotly terrain, cost, 3D elevation, and route visualization builders."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from simulation.simulation_controller import RoverSimulationController
from simulation.simulation_state import _compute_uncertainty_grid


def build_tactical_route_figure(
    controller: RoverSimulationController,
    layer: str = "total_cost",
) -> go.Figure:
    """Build an interactive Plotly tactical map with terrain layers, obstacles, and routes."""
    state = controller.state
    rows, cols = state.dem.elevation.shape

    if layer == "elevation":
        z_data = state.dem.elevation
        colorscale = "YlOrBr"
        colorbar_title = "Elevation (m)"
    elif layer == "slope":
        z_data = state.slope_deg
        colorscale = "Inferno"
        colorbar_title = "Slope (deg)"
    elif layer == "uncertainty":
        z_data = _compute_uncertainty_grid(state.class_probabilities)
        colorscale = "Magma"
        colorbar_title = "Uncertainty"
    else:
        finite = state.cost_map.total_cost
        cap = float(np.nanpercentile(finite[np.isfinite(finite)], 95)) * 1.35 if np.any(np.isfinite(finite)) else 10.0
        z_data = np.where(np.isfinite(finite), finite, cap)
        colorscale = "Cividis"
        colorbar_title = "Cost"

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z_data,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title),
            hovertemplate="Row %{y}, Col %{x}<br>Value: %{z:.2f}<extra></extra>",
        )
    )

    if state.show_previous_route and len(state.previous_route) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[c for _, c in state.previous_route],
                y=[r for r, _ in state.previous_route],
                mode="lines",
                name="Previous Route",
                line=dict(color="#94a3b8", width=2, dash="dash"),
            )
        )

    if len(state.travelled_cells) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[c for _, c in state.travelled_cells],
                y=[r for r, _ in state.travelled_cells],
                mode="lines",
                name="Travelled Path",
                line=dict(color="#10b981", width=3),
            )
        )

    if len(state.active_route) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[c for _, c in state.active_route],
                y=[r for r, _ in state.active_route],
                mode="lines+markers",
                name="Active UARA* Route",
                line=dict(color="#00e5ff", width=3.5),
                marker=dict(size=5, color="#3b82f6"),
            )
        )

    obstacles = controller.obstacle_manager.obstacles
    if obstacles:
        fig.add_trace(
            go.Scatter(
                x=[obs.col for obs in obstacles],
                y=[obs.row for obs in obstacles],
                mode="markers+text",
                name="Rock Obstacles",
                text=[obs.obstacle_id for obs in obstacles],
                textposition="top center",
                marker=dict(size=14, color="#ef4444", symbol="hexagon", line=dict(width=1.5, color="#ffffff")),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[state.start_cell[1]],
            y=[state.start_cell[0]],
            mode="markers",
            name="Start",
            marker=dict(size=12, color="#22c55e", symbol="circle", line=dict(width=2, color="#0b1120")),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[state.goal_cell[1]],
            y=[state.goal_cell[0]],
            mode="markers",
            name="Destination",
            marker=dict(size=14, color="#f59e0b", symbol="star", line=dict(width=1.5, color="#0b1120")),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[state.rover_col_f],
            y=[state.rover_row_f],
            mode="markers",
            name="MARS-09 Rover",
            marker=dict(size=15, color="#38bdf8", symbol="diamond", line=dict(width=2, color="#ffffff")),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=540,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis=dict(title="Column (X)", range=[-0.5, cols - 0.5], constrain="domain"),
        yaxis=dict(title="Row (Y)", range=[rows - 0.5, -0.5], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_elevation_3d_figure(controller: RoverSimulationController) -> go.Figure:
    """Build a 3D Martian DEM surface with draped rover trajectory."""
    state = controller.state
    elev = state.dem.elevation
    rows, cols = elev.shape
    dy_m, dx_m = state.dem.resolution_m

    x_m = np.arange(cols) * dx_m
    y_m = np.arange(rows) * dy_m

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_m,
                y=y_m,
                z=elev,
                colorscale="YlOrBr",
                opacity=0.92,
                showscale=True,
                colorbar=dict(title="Elev (m)"),
            )
        ]
    )

    if len(state.active_route) >= 2:
        rx = [c * dx_m for _, c in state.active_route]
        ry = [r * dy_m for r, _ in state.active_route]
        rz = [float(elev[r, c]) + 0.45 for r, c in state.active_route]
        fig.add_trace(
            go.Scatter3d(
                x=rx,
                y=ry,
                z=rz,
                mode="lines",
                name="UARA* Route",
                line=dict(color="#00e5ff", width=6),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=10, r=10, t=25, b=10),
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="South (m)",
            zaxis_title="Elevation (m)",
            aspectmode="data",
        ),
    )
    return fig
