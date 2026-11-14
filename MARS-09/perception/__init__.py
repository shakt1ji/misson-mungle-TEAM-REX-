"""MARS-09 Perception package (Member 2)."""

from perception.terrain_inference import (
    SpatialAlignmentError,
    TerrainInferenceEngine,
    clear_inference_cache,
    predict_terrain,
    predict_terrain_on_dem_grid,
)
from perception.visualization import (
    colorize_terrain_mask,
    compute_class_distribution,
    render_confidence_heatmap,
    render_prediction_comparison,
    render_uncertainty_heatmap,
)

__all__ = [
    "SpatialAlignmentError",
    "TerrainInferenceEngine",
    "clear_inference_cache",
    "colorize_terrain_mask",
    "compute_class_distribution",
    "predict_terrain",
    "predict_terrain_on_dem_grid",
    "render_confidence_heatmap",
    "render_prediction_comparison",
    "render_uncertainty_heatmap",
]
