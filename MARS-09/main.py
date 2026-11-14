"""MARS-09 Member 1 minimal end-to-end integration entry point.

Runs the terrain intelligence and UARA* navigation pipeline on either:
1. A real projected GeoTIFF DEM (--dem path/to/dem.tif), or
2. A clearly labeled synthetic demo fixture (default when --dem is omitted).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from navigation.astar import UARAConfig, plan_path_uara_star
from navigation.path_smoothing import smooth_path_line_of_sight
from navigation.rover_simulator import RoverSimulator
from terrain.dem_loader import create_demo_dem_fixture, load_dem_geotiff
from terrain.slope import compute_slope_degrees
from terrain.terrain_cost import CostWeights, compute_terrain_cost_map
from terrain.traversability import TerrainClassConfig, TraversabilityConfig


def _create_demo_ml_overlay(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, TerrainClassConfig]:
    """Create deterministic synthetic class IDs and probabilities for the demo."""
    rows, cols = shape
    class_ids = np.zeros((rows, cols), dtype=np.int32)
    probs = np.zeros((rows, cols, 3), dtype=np.float64)

    # High-confidence bedrock baseline (class 0)
    probs[..., 0] = 0.90
    probs[..., 1] = 0.07
    probs[..., 2] = 0.03

    # Synthetic loose-sand / high-uncertainty corridor (class 1) near lower middle
    r0, r1 = int(rows * 0.55), int(rows * 0.78)
    c0, c1 = int(cols * 0.25), int(cols * 0.65)
    class_ids[r0:r1, c0:c1] = 1
    probs[r0:r1, c0:c1, 0] = 0.34
    probs[r0:r1, c0:c1, 1] = 0.35
    probs[r0:r1, c0:c1, 2] = 0.31

    class_cfg = TerrainClassConfig(
        class_names={0: "bedrock", 1: "loose_sand", 2: "hazard_outcrop"},
        class_penalties={0: 0.1, 1: 1.5, 2: 4.0},
        impassable_class_ids={2},
    )
    return class_ids, probs, class_cfg


def visualize_mission(
    elevation: np.ndarray,
    slope_deg: np.ndarray,
    total_cost: np.ndarray,
    blocked_mask: np.ndarray,
    raw_path: list[tuple[int, int]],
    smoothed_path: list[tuple[int, int]],
    visited_cells: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Render a 3-panel diagnostic visualization of elevation, slope, and navigation."""
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    im0 = axes[0].imshow(elevation, cmap="terrain")
    axes[0].set_title("DEM Elevation (m)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(slope_deg, cmap="inferno")
    axes[1].set_title("Terrain Slope (deg)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    finite_cost = np.where(np.isfinite(total_cost), total_cost, np.nan)
    im2 = axes[2].imshow(finite_cost, cmap="viridis")
    axes[2].imshow(np.where(blocked_mask, 1.0, np.nan), cmap="Reds", alpha=0.65, vmin=0, vmax=1)
    axes[2].set_title("Terrain Cost & UARA* Route")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    if raw_path:
        rr, rc = zip(*raw_path)
        axes[2].plot(rc, rr, color="cyan", linewidth=1.8, linestyle="--", label="UARA* Path")
    if smoothed_path:
        sr, sc = zip(*smoothed_path)
        axes[2].plot(sc, sr, color="white", linewidth=2.0, label="Smoothed Waypoints")
    if visited_cells:
        vr, vc = zip(*visited_cells)
        axes[2].scatter(vc, vr, color="lime", s=10, alpha=0.8, label="Rover Visited")

    axes[2].scatter([start[1]], [start[0]], color="dodgerblue", s=70, marker="o", label="Start")
    axes[2].scatter([goal[1]], [goal[0]], color="gold", s=90, marker="*", label="Goal")
    axes[2].legend(loc="lower right", fontsize=8)

    for ax in axes:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def run_demo(
    dem_path: Path | None = None,
    save_plot: Path | None = None,
    show_plot: bool = True,
) -> dict[str, Any]:
    """Execute the end-to-end MARS-09 terrain and navigation demo."""
    if dem_path is not None:
        dem = load_dem_geotiff(dem_path)
        print(f"Loaded GeoTIFF DEM: {dem.source_name} | shape={dem.elevation.shape} | res={dem.resolution_m} m")
    else:
        dem = create_demo_dem_fixture(rows=40, cols=40, resolution_m=(2.0, 2.0))
        print(f"Loaded SYNTHETIC demo DEM fixture | shape={dem.elevation.shape} | res={dem.resolution_m} m")

    slope_deg = compute_slope_degrees(
        elevation=dem.elevation,
        resolution_m=dem.resolution_m,
        valid_mask=dem.valid_mask,
    )

    class_ids, probs, class_cfg = _create_demo_ml_overlay(dem.elevation.shape)

    cost_map = compute_terrain_cost_map(
        slope_deg=slope_deg,
        valid_mask=dem.valid_mask,
        resolution_m=dem.resolution_m,
        terrain_class_ids=class_ids,
        class_probabilities=probs,
        class_config=class_cfg,
        traversability_config=TraversabilityConfig(max_safe_slope_deg=25.0),
        weights=CostWeights(wd=1.0, ws=1.2, wt=1.0, wo=4.0, wu=2.0),
        obstacle_buffer_cells=1,
    )

    rows, cols = dem.elevation.shape
    start = (rows - 4, 3)
    goal = (4, cols - 5)

    nav_result = plan_path_uara_star(
        cost_map=cost_map,
        start=start,
        goal=goal,
        config=UARAConfig(connectivity=8, safety_weight=1.0, uncertainty_weight=1.0),
    )

    if not nav_result.success:
        print(f"Navigation failed: {nav_result.termination_reason}")
        return {"navigation": nav_result, "simulation": None}

    smoothed_path = smooth_path_line_of_sight(
        nav_result.path,
        blocked_mask=cost_map.blocked_mask,
        prevent_corner_cutting=True,
        cost_grid=cost_map.total_cost,
    )

    simulator = RoverSimulator(start_position=start)
    sim_result = simulator.follow_path(smoothed_path, blocked_mask=cost_map.blocked_mask)

    print(
        f"UARA* Result: steps={nav_result.path_length_steps}, "
        f"distance={nav_result.path_distance_m:.2f} m, "
        f"total_cost={nav_result.total_cost:.2f}, "
        f"expanded={nav_result.nodes_expanded}, "
        f"smoothed_waypoints={len(smoothed_path)}"
    )
    print(
        f"Simulator: status={sim_result.final_status.value}, "
        f"final_pos={sim_result.final_position}, "
        f"visited={len(sim_result.visited_cells)} cells"
    )

    if show_plot or save_plot is not None:
        visualize_mission(
            elevation=dem.elevation,
            slope_deg=slope_deg,
            total_cost=cost_map.total_cost,
            blocked_mask=cost_map.blocked_mask,
            raw_path=nav_result.path,
            smoothed_path=smoothed_path,
            visited_cells=sim_result.visited_cells,
            start=start,
            goal=goal,
            save_path=save_plot,
            show=show_plot,
        )

    return {"navigation": nav_result, "simulation": sim_result}


def main() -> None:
    parser = argparse.ArgumentParser(description="MARS-09 Terrain & UARA* Navigation Demo")
    parser.add_argument("--dem", type=Path, default=None, help="Optional path to projected GeoTIFF DEM")
    parser.add_argument("--save-plot", type=Path, default=None, help="Optional path to save output PNG plot")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive matplotlib window")
    args = parser.parse_args()

    run_demo(dem_path=args.dem, save_plot=args.save_plot, show_plot=not args.no_show)


if __name__ == "__main__":
    main()
