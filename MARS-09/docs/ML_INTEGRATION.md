# MARS-09 ML & Perception Integration Guide (Member 2 -> Member 1 & Member 3)

This document defines how Member 2's semantic segmentation subsystem (`ml/`, `perception/`) integrates with Member 1's terrain and navigation pipeline (`terrain/`, `navigation/`, `docs/INTERFACES.md`) and Member 3's visualization/science modules (`dashboard/`, `science/`).

---

## 1. Terrain Class IDs & Channel Ordering

All outputs follow the verified **AI4MARS** class schema defined in `ml/config.py`:

| Raw Mask RGB | Class Name | `terrain_class_ids` Value | `class_probabilities` Channel Index |
|---|---|---:|---:|
| `(0, 0, 0)` | `soil` | `0` | `0` |
| `(1, 1, 1)` | `bedrock` | `1` | `1` |
| `(2, 2, 2)` | `sand` | `2` | `2` |
| `(3, 3, 3)` | `big_rock` | `3` | `3` |
| `(255, 255, 255)` | `unlabelled` (`ignore_index=255`) | *(excluded from training loss & eval; model outputs `0..3`)* | — |

- Channel order is fixed as `("soil", "bedrock", "sand", "big_rock")`.
- Segmentation masks are always decoded from exact RGB triples and resized strictly with nearest-neighbor interpolation (`Image.Resampling.NEAREST`); masks are never converted to grayscale.

---

## 2. Output Schema, Shapes & Data Types

Both `predict_terrain(...)` and `predict_terrain_on_dem_grid(...)` return a `dict[str, Any]` matching `docs/INTERFACES.md`:

| Key | Shape | Dtype | Value Semantics |
|---|---|---|---|
| `terrain_class_ids` | `(H, W)` | `np.int32` | Per-pixel `argmax` class ID in `{0, 1, 2, 3}` |
| `class_probabilities` | `(H, W, 4)` | `np.float32` | Channel-last softmax probabilities `[soil, bedrock, sand, big_rock]`, finite and normalized to sum to `~1.0` along `axis=2` |
| `confidence` | `(H, W)` | `np.float32` | `max(class_probabilities, axis=2)` in `[0.0, 1.0]` |
| `uncertainty_score` | `(H, W)` | `np.float32` | `1.0 - confidence` in `[0.0, 1.0]` (matches Member 1's `one_minus_max` default) |
| `color_mask` | `(H, W, 3)` | `np.uint8` | RGB visualization mask for UI/plots |
| `class_names` | `dict[int, str]` | — | `{0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}` |

---

## 3. Loading the Model & Running Inference

### Option A: Cached Functional API

```python
from perception.terrain_inference import predict_terrain

# Accepts file path (str | Path), PIL.Image.Image, or (H, W, 3) uint8/float RGB ndarray
pred = predict_terrain("data/raw/ai4mars/test_min3/images/test_min3_00000.png")

terrain_class_ids = pred["terrain_class_ids"]      # (H, W) np.int32
class_probabilities = pred["class_probabilities"]  # (H, W, 4) np.float32
```

### Option B: Explicit Reusable Engine (Recommended for Batch / Dashboard Use)

```python
from perception.terrain_inference import TerrainInferenceEngine, predict_terrain

engine = TerrainInferenceEngine.from_checkpoint("models/ai4mars_unet_best.pt", device="cpu")
pred = predict_terrain(image_rgb, engine=engine)
```

- Internally, input images are resized with bilinear interpolation to the checkpoint's training resolution `(256, 256)` for network inference, and output logits are bilinearly upsampled back to the original input `(H, W)` before `softmax`.

---

## 4. Two Strictly Separated Operating Modes (Spatial Alignment)

Rover Mastcam/Navcam imagery in AI4MARS is perspective camera imagery, **not** an overhead orthorectified map aligned to an orbital HiRISE/CTX DEM grid. To prevent invalid spatial overlays:

1. **Mode A — Image-Level Inference (`predict_terrain`)**:
   - Preserves the native `(H, W)` resolution of the rover camera image.
   - Use for rover-view perception, uncertainty maps, class distributions, and visual overlays (`perception/visualization.py`).
2. **Mode B — DEM-Grid Inference (`predict_terrain_on_dem_grid`)**:
   - Required before fusing `terrain_class_ids` / `class_probabilities` cell-by-cell with Member 1's DEM `slope_deg` and `valid_mask`.
   - Raises `SpatialAlignmentError` unless **both** `is_orthorectified_to_dem=True` **and** `image.shape[:2] == dem_shape`.
   - Never silently resizes or warps an unregistered camera image to match a DEM grid.

---

## 5. Configuring `TerrainClassConfig` & Calling `compute_terrain_cost_map()`

Use `create_ai4mars_terrain_class_config()` from `ml/config.py` to build Member 1's `TerrainClassConfig` explicitly:

```python
from ml.config import create_ai4mars_terrain_class_config

class_cfg = create_ai4mars_terrain_class_config()
# Default penalties (demonstration heuristics only):
# {0: 0.4 (soil), 1: 0.2 (bedrock), 2: 1.5 (sand), 3: 3.0 (big_rock)}
# Default impassable_class_ids = set() (no class auto-blocks unless configured)
```

### End-to-End Integration Example (Mode B Orthorectified Image + Member 1 `terrain`)

```python
import numpy as np

from ml.config import create_ai4mars_terrain_class_config
from perception.terrain_inference import TerrainInferenceEngine, predict_terrain_on_dem_grid
from terrain.dem_loader import create_demo_dem_fixture
from terrain.slope import compute_slope_degrees
from terrain.terrain_cost import CostWeights, compute_terrain_cost_map
from terrain.traversability import TraversabilityConfig

# 1. Load DEM and compute slope via Member 1
dem = create_demo_dem_fixture(rows=40, cols=40, resolution_m=(2.0, 2.0))
slope_deg = compute_slope_degrees(
    elevation=dem.elevation,
    resolution_m=dem.resolution_m,
    valid_mask=dem.valid_mask,
)

# 2. Obtain an image that is ALREADY orthorectified & registered to dem.elevation.shape
orthorectified_rgb = np.full((40, 40, 3), 128, dtype=np.uint8)

engine = TerrainInferenceEngine.from_checkpoint("models/ai4mars_unet_best.pt", device="cpu")
ml_pred = predict_terrain_on_dem_grid(
    orthorectified_rgb,
    dem_shape=dem.elevation.shape,
    is_orthorectified_to_dem=True,
    engine=engine,
)

# 3. Configure Member 1 TerrainClassConfig and compute composite terrain cost map
class_cfg = create_ai4mars_terrain_class_config(
    # Optional mission policy override, e.g. treat big_rock as impassable:
    # impassable_class_ids={3},
)

cost_map = compute_terrain_cost_map(
    slope_deg=slope_deg,
    valid_mask=dem.valid_mask,
    resolution_m=dem.resolution_m,
    terrain_class_ids=ml_pred["terrain_class_ids"],
    class_probabilities=ml_pred["class_probabilities"],
    class_config=class_cfg,
    traversability_config=TraversabilityConfig(max_safe_slope_deg=25.0),
    weights=CostWeights(wd=1.0, ws=1.2, wt=1.0, wo=4.0, wu=2.0),
    obstacle_buffer_cells=1,
)
```

---

## 6. Member 3 Visualization Helpers

`perception/visualization.py` exposes reusable helpers for dashboard panels:
- `colorize_terrain_mask(terrain_class_ids, ignore_mask=None) -> np.ndarray`
- `compute_class_distribution(terrain_class_ids, class_names=None, ignore_index=255) -> dict`
- `render_confidence_heatmap(confidence, save_path=None, show=False)`
- `render_uncertainty_heatmap(uncertainty_score, save_path=None, show=False)`
- `render_prediction_comparison(image_rgb, terrain_class_ids, confidence, uncertainty_score, ground_truth_ids=None, save_path=None, show=False)`

---

## 7. Known Limitations

1. **Perspective vs. Orbital Geometry**: AI4MARS images are surface-level Curiosity rover images; without camera-to-DEM orthorectification/projection, pixel coordinates do not correspond to DEM `(row, col)` meters.
2. **No Physical Geometry or Composition Inference**: The U-Net predicts only the 4 visual AI4MARS surface classes (`soil`, `bedrock`, `sand`, `big_rock`). It does **not** predict terrain elevation, slope, bearing strength, mineral composition, water, or ice.
3. **Heuristic Traversability Penalties**: `DEMO_AI4MARS_CLASS_PENALTIES` are demonstration heuristics for cost-map integration, not experimentally calibrated rover slip or clearance limits.
4. **Ignore Label Handling**: Ground-truth pixels labelled `255` (sky, rover hardware, distant horizon, or unlabelled regions) are masked during training/evaluation, whereas at inference time the model assigns one of `{0, 1, 2, 3}` to every pixel; downstream consumers should inspect `uncertainty_score` or rover/range masks when available.
