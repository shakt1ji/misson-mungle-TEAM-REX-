"""Reusable terrain visualization helpers for Member 3 dashboard integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ml.config import AI4MARS_CLASS_NAMES, AI4MARS_IGNORE_INDEX, build_palette_lut


def colorize_terrain_mask(
    terrain_class_ids: np.ndarray,
    ignore_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert 2D integer terrain class IDs into a (H, W, 3) uint8 RGB mask."""
    ids = np.asarray(terrain_class_ids)
    if ids.ndim != 2:
        raise ValueError(f"terrain_class_ids must be 2D; got shape {ids.shape}.")

    lut = build_palette_lut()
    clipped = np.clip(ids, 0, 255).astype(np.uint8)
    rgb = lut[clipped]

    if ignore_mask is not None:
        ign = np.asarray(ignore_mask, dtype=bool)
        if ign.shape != ids.shape:
            raise ValueError(f"ignore_mask shape {ign.shape} must match {ids.shape}.")
        rgb[ign] = lut[AI4MARS_IGNORE_INDEX]

    return rgb


def compute_class_distribution(
    terrain_class_ids: np.ndarray,
    class_names: dict[int, str] | None = None,
    ignore_index: int = AI4MARS_IGNORE_INDEX,
) -> dict[str, dict[str, float | int]]:
    """Summarize pixel counts and percentages per terrain class (excluding ignore_index)."""
    ids = np.asarray(terrain_class_ids)
    if ids.ndim != 2:
        raise ValueError(f"terrain_class_ids must be 2D; got shape {ids.shape}.")

    names = class_names or AI4MARS_CLASS_NAMES
    valid = ids != int(ignore_index)
    total_valid = int(np.sum(valid))

    summary: dict[str, dict[str, float | int]] = {}
    for class_id, label in names.items():
        count = int(np.sum((ids == int(class_id)) & valid))
        pct = (100.0 * count / total_valid) if total_valid > 0 else 0.0
        summary[label] = {
            "class_id": int(class_id),
            "pixel_count": count,
            "percentage": round(pct, 3),
        }
    return summary


def render_confidence_heatmap(
    confidence: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "ML Prediction Confidence",
    show: bool = False,
) -> Any:
    """Render a [0, 1] maximum class confidence heatmap."""
    if not show:
        matplotlib.use("Agg", force=False)
    conf = np.asarray(confidence, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    im = ax.imshow(conf, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Max Class Probability")

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def render_uncertainty_heatmap(
    uncertainty_score: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "ML Prediction Uncertainty (1 - max p)",
    show: bool = False,
) -> Any:
    """Render a [0, 1] prediction uncertainty heatmap."""
    if not show:
        matplotlib.use("Agg", force=False)
    unc = np.asarray(uncertainty_score, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    im = ax.imshow(unc, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Uncertainty Score")

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def render_prediction_comparison(
    image_rgb: np.ndarray,
    terrain_class_ids: np.ndarray,
    confidence: np.ndarray,
    uncertainty_score: np.ndarray,
    ground_truth_ids: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Any:
    """Render side-by-side comparison: original, optional ground truth, prediction, confidence, uncertainty."""
    if not show:
        matplotlib.use("Agg", force=False)

    pred_rgb = colorize_terrain_mask(terrain_class_ids)
    panels: list[tuple[str, np.ndarray, dict[str, Any]]] = [
        ("Rover RGB Image", np.asarray(image_rgb, dtype=np.uint8), {}),
    ]
    if ground_truth_ids is not None:
        gt_arr = np.asarray(ground_truth_ids)
        gt_rgb = colorize_terrain_mask(gt_arr, ignore_mask=(gt_arr == AI4MARS_IGNORE_INDEX))
        panels.append(("Ground Truth (255=ignore)", gt_rgb, {}))

    panels.extend(
        [
            ("Predicted Terrain Mask", pred_rgb, {}),
            ("Confidence (max p)", np.asarray(confidence, dtype=np.float32), {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0}),
            ("Uncertainty (1 - max p)", np.asarray(uncertainty_score, dtype=np.float32), {"cmap": "magma", "vmin": 0.0, "vmax": 1.0}),
        ]
    )

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), constrained_layout=True)
    for ax, (title, img, kwargs) in zip(axes, panels):
        im = ax.imshow(img, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if kwargs:
            fig.colorbar(im, ax=ax, shrink=0.75)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig
