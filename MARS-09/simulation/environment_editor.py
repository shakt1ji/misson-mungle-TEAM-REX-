"""Environment editor and demonstration scenario builder using Member 1 & Member 2 models."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Iterable

import numpy as np

from simulation.dynamic_obstacles import DynamicRockObstacle
from terrain.dem_loader import DEMData, create_synthetic_dem
from terrain.slope import compute_slope_degrees
from terrain.terrain_cost import CostWeights, TerrainCostMap, compute_terrain_cost_map
from terrain.traversability import TerrainClassConfig, TraversabilityConfig


def _load_ai4mars_config_fn():
    try:
        from ml.config import create_ai4mars_terrain_class_config
        return create_ai4mars_terrain_class_config
    except ImportError:
        config_path = Path(__file__).resolve().parents[1] / "ml" / "config.py"
        if config_path.exists():
            spec = importlib.util.spec_from_file_location("mars09_ml_config_standalone", config_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.create_ai4mars_terrain_class_config
        return lambda: TerrainClassConfig(
            class_names={0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"},
            class_penalties={0: 0.4, 1: 0.2, 2: 1.5, 3: 3.0},
            impassable_class_ids=set(),
            default_unknown_class_penalty=1.0,
        )


create_ai4mars_terrain_class_config = _load_ai4mars_config_fn()


TERRAIN_TYPE_TO_CLASS_ID: dict[str, int] = {
    "soil": 0,
    "bedrock": 1,
    "sand": 2,
    "big_rock": 3,
    "unknown": 255,
    "blocked_region": 3,
}

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "normal_navigation": "Normal Navigation — Open Martian basin with gentle slopes and scattered off-path rocks.",
    "dynamic_obstacle": "Dynamic Obstacle — Large draggable boulder parked beside the active route; drag onto path to trigger live UARA* replanning.",
    "multiple_obstacles": "Multiple Obstacles — Dense boulder field requiring UARA* detour planning around multiple rock clusters.",
    "dangerous_terrain": "Dangerous Terrain — High-slip loose sand and steep crater flank; UARA* selects safer bedrock corridor.",
    "no_safe_path": "No Safe Path — Complete canyon barrier blocks all routes (NO_SAFE_PATH); remove a barrier rock to resume.",
}


@dataclass
class EnvironmentBuildResult:
    """Initial terrain, DEM, probabilities, obstacles, and endpoints for a scenario."""

    scenario_name: str
    dem: DEMData
    slope_deg: np.ndarray
    terrain_class_ids: np.ndarray
    class_probabilities: np.ndarray
    static_blocked_mask: np.ndarray
    obstacles: list[DynamicRockObstacle]
    start: tuple[int, int]
    goal: tuple[int, int]
    class_config: TerrainClassConfig
    traversability_config: TraversabilityConfig
    weights: CostWeights


def _make_class_probabilities(terrain_class_ids: np.ndarray, sand_uncertain: bool = True) -> np.ndarray:
    """Generate consistent (rows, cols, 4) [soil, bedrock, sand, big_rock] probability tensor."""
    rows, cols = terrain_class_ids.shape
    probs = np.full((rows, cols, 4), 0.04, dtype=np.float32)

    for cid in (0, 1, 2, 3):
        mask = terrain_class_ids == cid
        if np.any(mask):
            probs[mask] = 0.04
            probs[mask, cid] = 0.88

    if sand_uncertain:
        sand_mask = terrain_class_ids == 2
        if np.any(sand_mask):
            # Sand transitions carry higher semantic uncertainty
            probs[sand_mask, 0] = 0.24
            probs[sand_mask, 1] = 0.08
            probs[sand_mask, 2] = 0.62
            probs[sand_mask, 3] = 0.06

    unk_mask = terrain_class_ids == 255
    if np.any(unk_mask):
        probs[unk_mask] = 0.25

    return probs


def recompute_cost_map(
    dem: DEMData,
    terrain_class_ids: np.ndarray,
    class_probabilities: np.ndarray,
    dynamic_obstacle_mask: np.ndarray,
    static_blocked_mask: np.ndarray,
    class_config: TerrainClassConfig,
    traversability_config: TraversabilityConfig,
    weights: CostWeights,
    obstacle_buffer_cells: int = 1,
) -> tuple[np.ndarray, TerrainCostMap]:
    """Recompute DEM slope and Member 1 TerrainCostMap from current environment state."""
    slope_deg = compute_slope_degrees(
        elevation=dem.elevation,
        resolution_m=dem.resolution_m,
        valid_mask=dem.valid_mask,
    )
    combined_obstacles = np.asarray(dynamic_obstacle_mask, dtype=bool) | np.asarray(
        static_blocked_mask, dtype=bool
    )
    cost_map = compute_terrain_cost_map(
        slope_deg=slope_deg,
        valid_mask=dem.valid_mask,
        resolution_m=dem.resolution_m,
        terrain_class_ids=terrain_class_ids,
        class_probabilities=class_probabilities,
        obstacle_mask=combined_obstacles,
        class_config=class_config,
        traversability_config=traversability_config,
        weights=weights,
        obstacle_buffer_cells=obstacle_buffer_cells,
    )
    return slope_deg, cost_map


def apply_terrain_edit(
    terrain_class_ids: np.ndarray,
    class_probabilities: np.ndarray,
    static_blocked_mask: np.ndarray,
    cells: Iterable[tuple[int, int]],
    terrain_type: str,
) -> int:
    """Modify terrain type across specified grid cells; returns number of updated cells."""
    key = terrain_type.strip().lower()
    if key not in TERRAIN_TYPE_TO_CLASS_ID:
        raise ValueError(
            f"Unsupported terrain_type '{terrain_type}'. "
            f"Expected one of {list(TERRAIN_TYPE_TO_CLASS_ID.keys())}."
        )

    rows, cols = terrain_class_ids.shape
    class_id = TERRAIN_TYPE_TO_CLASS_ID[key]
    updated = 0

    for cell in cells:
        r, c = int(cell[0]), int(cell[1])
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"Terrain edit cell {(r, c)} is out of bounds for {(rows, cols)}.")

        terrain_class_ids[r, c] = class_id
        if key == "blocked_region":
            static_blocked_mask[r, c] = True
            class_probabilities[r, c] = [0.02, 0.03, 0.05, 0.90]
        elif key == "unknown":
            static_blocked_mask[r, c] = False
            class_probabilities[r, c] = [0.25, 0.25, 0.25, 0.25]
        elif key == "sand":
            static_blocked_mask[r, c] = False
            class_probabilities[r, c] = [0.24, 0.08, 0.62, 0.06]
        else:
            static_blocked_mask[r, c] = False
            probs_row = np.array([0.04, 0.04, 0.04, 0.04], dtype=np.float32)
            probs_row[class_id] = 0.88
            class_probabilities[r, c] = probs_row
        updated += 1

    return updated


def build_scenario_environment(
    scenario_name: str = "dynamic_obstacle",
    rows: int = 30,
    cols: int = 30,
    resolution_m: tuple[float, float] = (2.0, 2.0),
) -> EnvironmentBuildResult:
    """Build a realistic Martian DEM + terrain + obstacle configuration for a named scenario."""
    if scenario_name not in SCENARIO_DESCRIPTIONS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. Available: {list(SCENARIO_DESCRIPTIONS.keys())}"
        )

    r_idx, c_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    dy_m, dx_m = float(resolution_m[0]), float(resolution_m[1])
    y_m = r_idx * dy_m
    x_m = c_idx * dx_m

    # Realistic Martian regolith topography: rolling crater rims, ridge, and gentle ripples
    h_m = rows * dy_m
    w_m = cols * dx_m
    ridge = 5.5 * np.exp(-(((y_m - 0.34 * h_m) ** 2) / (2.0 * (6.5 ** 2)) + ((x_m - 0.56 * w_m) ** 2) / (2.0 * (7.0 ** 2))))
    crater = -3.8 * np.exp(-(((y_m - 0.72 * h_m) ** 2) + ((x_m - 0.30 * w_m) ** 2)) / (2.0 * (6.0 ** 2)))
    ripples = 0.45 * np.sin(x_m * 0.24) * np.cos(y_m * 0.19) + 0.25 * np.sin((x_m + y_m) * 0.35)
    regional_tilt = 0.04 * x_m - 0.03 * y_m

    elevation = 100.0 + regional_tilt + ridge + crater + ripples

    # Base Martian terrain classes: soil (0) baseline with bedrock (1) outcrops & sand (2) drifts
    terrain_class_ids = np.zeros((rows, cols), dtype=np.int32)
    terrain_class_ids[(ridge > 2.2) | ((r_idx < 7) & (c_idx > cols // 2))] = 1
    terrain_class_ids[(crater < -1.5) | ((r_idx > rows - 8) & (c_idx < cols // 2))] = 2

    static_blocked = np.zeros((rows, cols), dtype=bool)
    obstacles: list[DynamicRockObstacle] = []

    start = (rows - 4, 3)
    goal = (4, cols - 4)

    if scenario_name == "normal_navigation":
        obstacles = [
            DynamicRockObstacle("rock_001", rows // 2 - 5, cols // 3, radius_cells=0, height_m=0.95, label="Basalt Rock A"),
            DynamicRockObstacle("rock_002", rows // 2 + 5, (2 * cols) // 3, radius_cells=0, height_m=1.05, label="Basalt Rock B"),
            DynamicRockObstacle("rock_003", 7, 8, radius_cells=0, height_m=0.85, label="Outcrop C"),
        ]
    elif scenario_name == "dynamic_obstacle":
        # Park a large draggable boulder near the middle diagonal ready to be dragged onto the path
        mid_r = rows // 2
        mid_c = cols // 2
        obstacles = [
            DynamicRockObstacle("rock_001", max(2, mid_r - 4), mid_c + 2, radius_cells=1, height_m=1.65, label="Draggable Boulder"),
            DynamicRockObstacle("rock_002", mid_r + 5, mid_c - 4, radius_cells=0, height_m=1.05, label="Side Rock"),
            DynamicRockObstacle("rock_003", 6, cols // 3, radius_cells=0, height_m=0.95, label="Ridge Rock"),
        ]
    elif scenario_name == "multiple_obstacles":
        mid_r = rows // 2
        mid_c = cols // 2
        obstacles = [
            DynamicRockObstacle("rock_001", mid_r, mid_c, radius_cells=1, height_m=1.55, label="Central Boulder"),
            DynamicRockObstacle("rock_002", mid_r - 3, mid_c - 2, radius_cells=1, height_m=1.40, label="North-West Cluster"),
            DynamicRockObstacle("rock_003", mid_r + 3, mid_c + 2, radius_cells=1, height_m=1.45, label="South-East Cluster"),
            DynamicRockObstacle("rock_004", mid_r - 6, mid_c + 4, radius_cells=0, height_m=1.10, label="Flank Rock 1"),
            DynamicRockObstacle("rock_005", mid_r + 6, mid_c - 4, radius_cells=0, height_m=1.10, label="Flank Rock 2"),
        ]
    elif scenario_name == "dangerous_terrain":
        # Steep crater rim & wide loose sand trap directly across the center corridor
        r0, r1 = rows // 3, (2 * rows) // 3
        c0, c1 = cols // 4, (3 * cols) // 4
        terrain_class_ids[r0:r1, c0:c1] = 2  # Loose high-slip sand
        # Add steep elevation slope spike in the upper-middle of the sand trap
        steep_y = (r0 + r1) * 0.5 * dy_m
        steep_x = (c0 + c1) * 0.5 * dx_m
        elevation += 9.5 * np.exp(-(((y_m - steep_y) ** 2) + ((x_m - steep_x) ** 2)) / (2.0 * (4.2 ** 2)))
        obstacles = [
            DynamicRockObstacle("rock_001", r0 - 1, c0 + 2, radius_cells=0, height_m=1.10, label="Rim Boulder"),
        ]
    elif scenario_name == "no_safe_path":
        # Complete rock barrier dividing the map horizontally/diagonally around mid-column
        mid_c = cols // 2
        obs_idx = 1
        for rr in range(0, rows, 2):
            obstacles.append(
                DynamicRockObstacle(
                    f"rock_{obs_idx:03d}",
                    rr,
                    mid_c,
                    radius_cells=1,
                    height_m=1.50,
                    label=f"Barrier Rock {obs_idx}",
                )
            )
            obs_idx += 1

    # Ensure start & goal cells are always clean bedrock/soil and never covered by obstacles
    terrain_class_ids[start] = 1
    terrain_class_ids[goal] = 1
    filtered_obstacles: list[DynamicRockObstacle] = []
    for obs in obstacles:
        fp = obs.footprint_cells(rows, cols)
        if start not in fp and goal not in fp:
            filtered_obstacles.append(obs)

    dem = create_synthetic_dem(
        elevation=elevation,
        resolution_m=(dy_m, dx_m),
        source_name=f"mars09_{scenario_name}",
    )
    slope_deg = compute_slope_degrees(
        elevation=dem.elevation,
        resolution_m=dem.resolution_m,
        valid_mask=dem.valid_mask,
    )
    probs = _make_class_probabilities(terrain_class_ids)
    class_cfg = create_ai4mars_terrain_class_config()
    trav_cfg = TraversabilityConfig(max_safe_slope_deg=28.0, min_confidence_threshold=0.0)
    weights = CostWeights(wd=1.0, ws=1.2, wt=1.0, wo=4.0, wu=1.8)

    return EnvironmentBuildResult(
        scenario_name=scenario_name,
        dem=dem,
        slope_deg=slope_deg,
        terrain_class_ids=terrain_class_ids,
        class_probabilities=probs,
        static_blocked_mask=static_blocked,
        obstacles=filtered_obstacles,
        start=start,
        goal=goal,
        class_config=class_cfg,
        traversability_config=trav_cfg,
        weights=weights,
    )
