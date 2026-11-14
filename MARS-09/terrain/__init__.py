"""MARS-09 Terrain Intelligence package (Member 1)."""

from terrain.dem_loader import (
    DEMData,
    DEMLoadError,
    InvalidDEMError,
    UnsupportedCRSUnitError,
    create_demo_dem_fixture,
    create_synthetic_dem,
    load_dem_geotiff,
    validate_crs_and_units,
)
from terrain.slope import compute_slope_degrees
from terrain.terrain_cost import (
    CostWeights,
    TerrainCostMap,
    compute_terrain_cost_map,
)
from terrain.traversability import (
    TerrainClassConfig,
    TraversabilityConfig,
    TraversabilityResult,
    compute_prediction_uncertainty,
    evaluate_traversability,
)

__all__ = [
    "CostWeights",
    "DEMData",
    "DEMLoadError",
    "InvalidDEMError",
    "TerrainClassConfig",
    "TerrainCostMap",
    "TraversabilityConfig",
    "TraversabilityResult",
    "UnsupportedCRSUnitError",
    "compute_prediction_uncertainty",
    "compute_slope_degrees",
    "compute_terrain_cost_map",
    "create_demo_dem_fixture",
    "create_synthetic_dem",
    "evaluate_traversability",
    "load_dem_geotiff",
    "validate_crs_and_units",
]
