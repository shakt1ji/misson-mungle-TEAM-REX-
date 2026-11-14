"""Configuration constants and Member 1 TerrainClassConfig bridge for AI4MARS."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from terrain.traversability import TerrainClassConfig

# Repository paths (resolved relative to MARS-09 root)
MARS09_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = MARS09_ROOT / "data" / "raw" / "ai4mars"
DATA_PROCESSED_DIR = MARS09_ROOT / "data" / "processed"
MODELS_DIR = MARS09_ROOT / "models"
OUTPUTS_DIR = MARS09_ROOT / "outputs"
DEFAULT_CHECKPOINT_PATH = MODELS_DIR / "ai4mars_unet_best.pt"

# Hugging Face dataset identifier
AI4MARS_DATASET_ID = "hassanjbara/AI4MARS"

# Exact AI4MARS RGB -> integer class ID encoding
AI4MARS_IGNORE_INDEX = 255
AI4MARS_NUM_CLASSES = 4

AI4MARS_CLASS_ORDER: tuple[str, ...] = ("soil", "bedrock", "sand", "big_rock")

AI4MARS_CLASS_NAMES: dict[int, str] = {
    0: "soil",
    1: "bedrock",
    2: "sand",
    3: "big_rock",
}

AI4MARS_RGB_TO_ID: dict[tuple[int, int, int], int] = {
    (0, 0, 0): 0,
    (1, 1, 1): 1,
    (2, 2, 2): 2,
    (3, 3, 3): 3,
    (255, 255, 255): AI4MARS_IGNORE_INDEX,
}

# Distinct RGB visualization colors (uint8) for human-readable masks
AI4MARS_COLOR_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (166, 124, 82),    # soil -> warm brown
    1: (110, 118, 129),   # bedrock -> slate gray
    2: (235, 200, 120),   # sand -> pale ochre
    3: (196, 69, 54),     # big_rock -> rust red
    255: (30, 30, 30),    # unlabelled -> dark charcoal
}

# ImageNet normalization statistics used by LightweightUNet preprocessing
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Demonstration heuristic terrain penalties (NOT experimentally established rover limits)
DEMO_AI4MARS_CLASS_PENALTIES: dict[int, float] = {
    0: 0.4,  # soil (demonstration heuristic)
    1: 0.2,  # bedrock (demonstration heuristic)
    2: 1.5,  # sand (demonstration heuristic)
    3: 3.0,  # big_rock (demonstration heuristic)
}


def create_ai4mars_terrain_class_config(
    class_penalties: dict[int, float] | None = None,
    impassable_class_ids: set[int] | None = None,
    default_unknown_class_penalty: float = 1.0,
) -> TerrainClassConfig:
    """Create Member 1's TerrainClassConfig with the AI4MARS 4-class mapping.

    Note:
        Default `DEMO_AI4MARS_CLASS_PENALTIES` are demonstration heuristics for
        hackathon integration, not experimentally calibrated Mars rover limits.
        `impassable_class_ids` defaults to an empty set so semantic classes do not
        automatically block cells without an explicit operator policy.
    """
    penalties = dict(DEMO_AI4MARS_CLASS_PENALTIES) if class_penalties is None else dict(class_penalties)
    impassable = set() if impassable_class_ids is None else set(impassable_class_ids)
    return TerrainClassConfig(
        class_names=dict(AI4MARS_CLASS_NAMES),
        class_penalties=penalties,
        impassable_class_ids=impassable,
        default_unknown_class_penalty=float(default_unknown_class_penalty),
    )


def build_palette_lut() -> np.ndarray:
    """Return a (256, 3) uint8 lookup table mapping class ID -> RGB display color."""
    lut = np.full((256, 3), 30, dtype=np.uint8)
    for class_id, rgb in AI4MARS_COLOR_PALETTE.items():
        lut[int(class_id)] = np.asarray(rgb, dtype=np.uint8)
    return lut
