"""Terrain traversability analysis, obstacle masking, and ML uncertainty quantification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TerrainClassConfig:
    """Configurable mapping from integer terrain class IDs to names, penalties, and hazards."""

    class_names: dict[int, str] = field(default_factory=dict)
    class_penalties: dict[int, float] = field(default_factory=dict)
    impassable_class_ids: set[int] = field(default_factory=set)
    default_unknown_class_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.default_unknown_class_penalty < 0.0:
            raise ValueError("default_unknown_class_penalty must be nonnegative.")
        for class_id, penalty in self.class_penalties.items():
            if penalty < 0.0 or not np.isfinite(penalty):
                raise ValueError(
                    f"Penalty for class ID {class_id} must be finite and nonnegative; got {penalty}."
                )


@dataclass(frozen=True)
class TraversabilityConfig:
    """Configurable safety thresholds for terrain traversability evaluation."""

    max_safe_slope_deg: float = 30.0
    min_confidence_threshold: float = 0.0
    uncertainty_method: str = "one_minus_max"
    treat_invalid_dem_as_blocked: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_safe_slope_deg) or self.max_safe_slope_deg <= 0.0:
            raise ValueError("max_safe_slope_deg must be positive and finite.")
        if not (0.0 <= self.min_confidence_threshold <= 1.0):
            raise ValueError("min_confidence_threshold must be in [0.0, 1.0].")
        if self.uncertainty_method not in {"one_minus_max", "entropy"}:
            raise ValueError(
                "uncertainty_method must be 'one_minus_max' or 'entropy'."
            )


@dataclass(frozen=True)
class TraversabilityResult:
    """Outputs of terrain traversability and uncertainty evaluation."""

    obstacle_mask: np.ndarray
    traversable_mask: np.ndarray
    blocked_mask: np.ndarray
    valid_mask: np.ndarray
    uncertainty_score: np.ndarray


def compute_prediction_uncertainty(
    class_probabilities: Any,
    method: str = "one_minus_max",
) -> np.ndarray:
    """Compute per-cell prediction uncertainty in [0.0, 1.0] from ML class probabilities."""
    if method not in {"one_minus_max", "entropy"}:
        raise ValueError(f"Unsupported uncertainty method '{method}'.")

    if hasattr(class_probabilities, "detach") and hasattr(class_probabilities, "cpu"):
        probs = np.asarray(class_probabilities.detach().cpu().numpy(), dtype=np.float64)
    else:
        probs = np.asarray(class_probabilities, dtype=np.float64)

    if probs.ndim != 3 or probs.shape[2] < 1:
        raise ValueError(
            f"class_probabilities must be 3D with shape (rows, cols, num_classes >= 1); "
            f"got shape {probs.shape}."
        )

    valid_probs = np.all(np.isfinite(probs), axis=2) & np.all(probs >= 0.0, axis=2)
    row_sums = np.sum(np.where(valid_probs[..., None], probs, 0.0), axis=2)
    valid_probs &= row_sums > 0.0

    uncertainty = np.ones((probs.shape[0], probs.shape[1]), dtype=np.float64)

    if not np.any(valid_probs):
        return uncertainty

    normalized = np.zeros_like(probs, dtype=np.float64)
    normalized[valid_probs] = probs[valid_probs] / row_sums[valid_probs, None]

    if method == "one_minus_max":
        max_p = np.max(normalized[valid_probs], axis=1)
        uncertainty[valid_probs] = np.clip(1.0 - max_p, 0.0, 1.0)
    else:
        num_classes = probs.shape[2]
        if num_classes == 1:
            uncertainty[valid_probs] = 0.0
        else:
            p_valid = normalized[valid_probs]
            safe_p = np.where(p_valid > 0.0, p_valid, 1.0)
            entropy = -np.sum(np.where(p_valid > 0.0, p_valid * np.log(safe_p), 0.0), axis=1)
            uncertainty[valid_probs] = np.clip(entropy / np.log(float(num_classes)), 0.0, 1.0)

    return uncertainty


def evaluate_traversability(
    slope_deg: np.ndarray,
    valid_mask: np.ndarray,
    terrain_class_ids: np.ndarray | None = None,
    class_probabilities: Any | None = None,
    obstacle_mask: np.ndarray | None = None,
    class_config: TerrainClassConfig | None = None,
    traversability_config: TraversabilityConfig | None = None,
) -> TraversabilityResult:
    """Evaluate per-cell traversability, obstacles, and prediction uncertainty."""
    slope_arr = np.asarray(slope_deg, dtype=np.float64)
    valid_arr = np.asarray(valid_mask, dtype=bool)

    if slope_arr.ndim != 2:
        raise ValueError(f"slope_deg must be 2D; got shape {slope_arr.shape}.")
    if valid_arr.shape != slope_arr.shape:
        raise ValueError(
            f"valid_mask shape {valid_arr.shape} must match slope_deg shape {slope_arr.shape}."
        )

    cfg = traversability_config or TraversabilityConfig()
    cls_cfg = class_config or TerrainClassConfig()

    effective_valid = valid_arr & np.isfinite(slope_arr) & (slope_arr >= 0.0)

    combined_obstacle_mask = np.zeros(slope_arr.shape, dtype=bool)
    if obstacle_mask is not None:
        obs_arr = np.asarray(obstacle_mask, dtype=bool)
        if obs_arr.shape != slope_arr.shape:
            raise ValueError(
                f"obstacle_mask shape {obs_arr.shape} must match slope_deg shape {slope_arr.shape}."
            )
        combined_obstacle_mask |= obs_arr

    if terrain_class_ids is not None:
        cls_arr = np.asarray(terrain_class_ids)
        if cls_arr.shape != slope_arr.shape:
            raise ValueError(
                f"terrain_class_ids shape {cls_arr.shape} must match slope_deg shape {slope_arr.shape}."
            )
        if not np.issubdtype(cls_arr.dtype, np.integer):
            raise ValueError(
                f"terrain_class_ids must have an integer dtype; got {cls_arr.dtype}."
            )
        if cls_cfg.impassable_class_ids:
            impassable_mask = np.isin(cls_arr, list(cls_cfg.impassable_class_ids))
            combined_obstacle_mask |= impassable_mask

    if class_probabilities is not None:
        uncertainty = compute_prediction_uncertainty(
            class_probabilities, method=cfg.uncertainty_method
        )
        if uncertainty.shape != slope_arr.shape:
            raise ValueError(
                f"class_probabilities spatial shape {uncertainty.shape} must match "
                f"slope_deg shape {slope_arr.shape}."
            )
        uncertainty = np.where(effective_valid, uncertainty, 1.0)
    else:
        uncertainty = np.where(effective_valid, 0.0, 1.0)

    slope_blocked = np.where(effective_valid, slope_arr > cfg.max_safe_slope_deg, True)

    confidence_blocked = np.zeros(slope_arr.shape, dtype=bool)
    if cfg.min_confidence_threshold > 0.0 and class_probabilities is not None:
        max_conf = 1.0 - compute_prediction_uncertainty(
            class_probabilities, method="one_minus_max"
        )
        confidence_blocked = max_conf < cfg.min_confidence_threshold

    invalid_blocked = (~effective_valid) if cfg.treat_invalid_dem_as_blocked else np.zeros_like(effective_valid)

    blocked_mask = (
        combined_obstacle_mask
        | slope_blocked
        | confidence_blocked
        | invalid_blocked
    )
    traversable_mask = (~blocked_mask) & effective_valid

    return TraversabilityResult(
        obstacle_mask=combined_obstacle_mask,
        traversable_mask=traversable_mask,
        blocked_mask=blocked_mask,
        valid_mask=effective_valid,
        uncertainty_score=uncertainty,
    )
