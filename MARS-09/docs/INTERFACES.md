# MARS-09 Interface Contract (Member 1)

This document defines the shared data conventions, coordinate systems, data structures, and public function signatures owned by **Member 1** (`terrain/`, `navigation/`).

Member 2 (`ml/`, `perception/`) and Member 3 (`dashboard/`, `science/`) must treat these signatures and array conventions as stable integration contracts. Do not change public function signatures without documenting the reason and updating this contract.

---

## 1. Global Spatial & Array Conventions

| Concept | Convention |
|---|---|
| Grid Coordinates | `(row, column)` as `tuple[int, int]`, 0-indexed from top-left |
| Array Indexing | `array[row, column]` with shape `(rows, cols)` |
| Elevation Units | Meters (`float64`), `np.nan` for invalid/NoData cells |
| Pixel Spacing | `resolution_m = (row_spacing_m, col_spacing_m)` in meters (`> 0`) |
| Slope | Degrees (`float64`), in `[0.0, 90.0)`, `np.nan` where invalid |
| Terrain Cost | Nonnegative `float64` (`np.inf` for blocked/impassable cells) |
| Blocked Cells | Explicit 2D Boolean mask `blocked_mask[row, col]` (`True` = blocked) |
| Traversable Cells | Explicit 2D Boolean mask `traversable_mask[row, col]` (`True` = traversable) |
| Invalid DEM Cells | Explicit 2D Boolean mask `valid_mask[row, col]` (`True` = valid data, `False` = NoData/NaN) |
| Terrain Class IDs | 2D integer array `terrain_class_ids` with shape `(rows, cols)` |
| ML Class Probabilities | 3D float array `class_probabilities` with shape `(rows, cols, num_classes)` |

### Important Rules
1. **No Geographic-as-Meters Assumption**: `dem_loader.py` rejects geographic CRS (degrees) and missing CRS unless explicitly overridden in synthetic test utilities.
2. **Missing Data Never Free**: Any cell where `valid_mask` is `False` is excluded from traversability (`traversable_mask[r, c] == False`, `blocked_mask[r, c] == True`, `total_cost[r, c] == np.inf`).
3. **Configurable Semantics**: `terrain_class_ids` mapping (`TerrainClassConfig`) must be supplied or configured explicitly; no hard-coded assumption about an unverified dataset class ID encoding.

---

## 2. `terrain/` Public Interfaces

### `terrain/dem_loader.py`

#### Exceptions
- `DEMLoadError(Exception)`
- `UnsupportedCRSUnitError(DEMLoadError)`
- `InvalidDEMError(DEMLoadError)`

#### Dataclass `DEMData`
- `elevation: np.ndarray` (`(rows, cols)`, `float64`, meters, `np.nan` where invalid)
- `valid_mask: np.ndarray` (`(rows, cols)`, `bool`)
- `resolution_m: tuple[float, float]` (`(row_spacing_m, col_spacing_m)`)
- `crs: str | None`
- `transform: object | None`
- `nodata_value: float | None`
- `source_name: str`
- `is_synthetic: bool`

#### Functions
- `validate_crs_and_units(crs: object, elevation_unit: str = "meter") -> tuple[float, str]`
- `load_dem_geotiff(file_path: str | Path, band: int = 1, elevation_unit: str = "meter", allow_missing_crs: bool = False) -> DEMData`
- `create_synthetic_dem(elevation: np.ndarray, resolution_m: tuple[float, float] = (1.0, 1.0), nodata_value: float | None = None, source_name: str = "synthetic_dem") -> DEMData`
- `create_demo_dem_fixture(rows: int = 40, cols: int = 40, resolution_m: tuple[float, float] = (2.0, 2.0)) -> DEMData`

---

### `terrain/slope.py`

#### Functions
- `compute_slope_degrees(elevation: np.ndarray, resolution_m: tuple[float, float] = (1.0, 1.0), valid_mask: np.ndarray | None = None) -> np.ndarray`

---

### `terrain/traversability.py`

#### Dataclasses
- `TerrainClassConfig(class_names: dict[int, str], class_penalties: dict[int, float], impassable_class_ids: set[int], default_unknown_class_penalty: float = 1.0)`
- `TraversabilityConfig(max_safe_slope_deg: float = 30.0, min_confidence_threshold: float = 0.0, uncertainty_method: str = "one_minus_max", treat_invalid_dem_as_blocked: bool = True)`
- `TraversabilityResult(obstacle_mask: np.ndarray, traversable_mask: np.ndarray, blocked_mask: np.ndarray, valid_mask: np.ndarray, uncertainty_score: np.ndarray)`

#### Functions
- `compute_prediction_uncertainty(class_probabilities: Any, method: str = "one_minus_max") -> np.ndarray`
- `evaluate_traversability(slope_deg: np.ndarray, valid_mask: np.ndarray, terrain_class_ids: np.ndarray | None = None, class_probabilities: Any | None = None, obstacle_mask: np.ndarray | None = None, class_config: TerrainClassConfig | None = None, traversability_config: TraversabilityConfig | None = None) -> TraversabilityResult`

---

### `terrain/terrain_cost.py`

#### Dataclasses
- `CostWeights(wd: float = 1.0, ws: float = 1.0, wt: float = 1.0, wo: float = 5.0, wu: float = 1.5)`
- `TerrainCostMap`:
  - `total_cost: np.ndarray`
  - `base_distance_cost: np.ndarray`
  - `slope_risk: np.ndarray`
  - `terrain_risk: np.ndarray`
  - `obstacle_risk: np.ndarray`
  - `uncertainty_risk: np.ndarray`
  - `blocked_mask: np.ndarray`
  - `traversable_mask: np.ndarray`
  - `valid_mask: np.ndarray`
  - `weights: CostWeights`
  - `resolution_m: tuple[float, float]`

#### Functions
- `compute_terrain_cost_map(slope_deg: np.ndarray, valid_mask: np.ndarray, resolution_m: tuple[float, float] = (1.0, 1.0), terrain_class_ids: np.ndarray | None = None, class_probabilities: Any | None = None, obstacle_mask: np.ndarray | None = None, class_config: TerrainClassConfig | None = None, traversability_config: TraversabilityConfig | None = None, weights: CostWeights | None = None, obstacle_buffer_cells: int = 1) -> TerrainCostMap`

---

## 3. `navigation/` Public Interfaces

### `navigation/heuristics.py`
- `manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> float`
- `euclidean_distance(a: tuple[int, int], b: tuple[int, int]) -> float`
- `octile_distance(a: tuple[int, int], b: tuple[int, int], cardinal_cost: float = 1.0, diagonal_cost: float = math.sqrt(2.0)) -> float`
- `metric_distance_lower_bound(a: tuple[int, int], b: tuple[int, int], resolution_m: tuple[float, float], connectivity: int = 8) -> float`

### `navigation/astar.py`
- `UARAConfig(connectivity: int = 8, safety_weight: float = 1.0, uncertainty_weight: float = 1.0, heuristic_weight: float = 1.0, use_metric_distance: bool = False, prevent_corner_cutting: bool = True)`
- `NavigationResult(success: bool, path: list[tuple[int, int]], total_cost: float, path_length_steps: int, path_distance_m: float, nodes_expanded: int, nodes_generated: int, termination_reason: str, is_optimal_guaranteed: bool)`
- `plan_path_uara_star(cost_map: TerrainCostMap, start: tuple[int, int], goal: tuple[int, int], config: UARAConfig | None = None) -> NavigationResult`
- `PlannerInterface` (ABC) & `UARAStarPlanner(PlannerInterface)`
  - `plan(cost_map: TerrainCostMap, start: tuple[int, int], goal: tuple[int, int]) -> NavigationResult`
  - `replan_with_updates(cost_map: TerrainCostMap, current_position: tuple[int, int], goal: tuple[int, int], newly_blocked_cells: Iterable[tuple[int, int]]) -> tuple[TerrainCostMap, NavigationResult]`

### `navigation/path_smoothing.py`
- `grid_line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]`
- `bresenham_centerline(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]`
- `has_line_of_sight(a: tuple[int, int], b: tuple[int, int], blocked_mask: np.ndarray, prevent_corner_cutting: bool = True, cost_grid: np.ndarray | None = None, max_cell_cost: float | None = None) -> bool`
- `smooth_path_line_of_sight(path: list[tuple[int, int]], blocked_mask: np.ndarray, prevent_corner_cutting: bool = True, cost_grid: np.ndarray | None = None, max_cell_cost: float | None = None) -> list[tuple[int, int]]`

### `navigation/rover_simulator.py`
- `SimulationStatus` (Enum): `IDLE`, `MOVING`, `GOAL_REACHED`, `BLOCKED_REPLAN_REQUIRED`, `FAILED_INVALID_PATH`
- `SimulationStepEvent(step_index: int, position: tuple[int, int], status: SimulationStatus, blocked_cell: tuple[int, int] | None, message: str)`
- `SimulationResult(final_status: SimulationStatus, final_position: tuple[int, int], goal_position: tuple[int, int], visited_cells: list[tuple[int, int]], events: list[SimulationStepEvent], goal_reached: bool, replan_required: bool, remaining_path: list[tuple[int, int]])`
- `RoverSimulator(start_position: tuple[int, int])`
  - `step_to(next_cell: tuple[int, int], blocked_mask: np.ndarray, step_index: int = 0, prevent_corner_cutting: bool = True) -> SimulationStepEvent`
  - `follow_path(path: list[tuple[int, int]], blocked_mask: np.ndarray, dynamic_obstacles_by_step: dict[int, list[tuple[int, int]]] | None = None, prevent_corner_cutting: bool = True) -> SimulationResult`
