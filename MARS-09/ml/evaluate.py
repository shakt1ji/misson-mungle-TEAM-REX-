"""Evaluation on the untouched AI4MARS test_min3 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ml.config import (
    AI4MARS_CLASS_NAMES,
    AI4MARS_IGNORE_INDEX,
    AI4MARS_NUM_CLASSES,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DEFAULT_CHECKPOINT_PATH,
    OUTPUTS_DIR,
)
from ml.dataset import decode_ai4mars_rgb_mask
from ml.prepare_dataset import _collect_split_pairs
from perception.terrain_inference import TerrainInferenceEngine
from perception.visualization import render_prediction_comparison


def compute_detailed_metrics(conf: np.ndarray) -> dict[str, Any]:
    """Compute confusion matrix, pixel accuracy, per-class P/R/F1/IoU, and mIoU."""
    tp = np.diag(conf).astype(np.float64)
    fp = np.sum(conf, axis=0).astype(np.float64) - tp
    fn = np.sum(conf, axis=1).astype(np.float64) - tp

    total = float(np.sum(conf))
    pixel_accuracy = float(np.sum(tp) / total) if total > 0 else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    valid_ious: list[float] = []

    for cid in range(AI4MARS_NUM_CLASSES):
        name = AI4MARS_CLASS_NAMES[cid]
        precision = float(tp[cid] / (tp[cid] + fp[cid])) if (tp[cid] + fp[cid]) > 0 else 0.0
        recall = float(tp[cid] / (tp[cid] + fn[cid])) if (tp[cid] + fn[cid]) > 0 else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        union = tp[cid] + fp[cid] + fn[cid]
        iou = float(tp[cid] / union) if union > 0 else 0.0
        support = int(np.sum(conf[cid, :]))
        if union > 0:
            valid_ious.append(iou)

        per_class[name] = {
            "class_id": cid,
            "support_pixels": support,
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            "f1": round(f1, 5),
            "iou": round(iou, 5),
        }

    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0

    return {
        "total_labelled_pixels": int(total),
        "pixel_accuracy": round(pixel_accuracy, 5),
        "mean_iou": round(mean_iou, 5),
        "per_class": per_class,
        "confusion_matrix": conf.tolist(),
    }


def evaluate_on_test_min3(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    splits_path: Path = DATA_PROCESSED_DIR / "ai4mars_splits.json",
    raw_test_dir: Path = DATA_RAW_DIR / "test_min3",
    num_visual_samples: int = 4,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate trained checkpoint on untouched `test_min3` split at native resolution."""
    if splits_path.is_file():
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        test_samples = splits.get("test_min3", [])
    else:
        test_samples = _collect_split_pairs(raw_test_dir)

    if max_eval_samples is not None:
        test_samples = test_samples[:max_eval_samples]

    if not test_samples:
        raise RuntimeError("No test_min3 samples found for evaluation.")

    engine = TerrainInferenceEngine.from_checkpoint(checkpoint_path)
    conf = np.zeros((AI4MARS_NUM_CLASSES, AI4MARS_NUM_CLASSES), dtype=np.int64)
    ignored_pixels = 0

    plots_dir = OUTPUTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: list[str] = []

    for idx, entry in enumerate(test_samples):
        with Image.open(entry["image_path"]) as img_raw:
            img_rgb = np.asarray(img_raw.convert("RGB"), dtype=np.uint8)
        with Image.open(entry["mask_path"]) as mask_raw:
            gt_ids, _ = decode_ai4mars_rgb_mask(mask_raw)

        pred = engine.predict(img_rgb)
        pred_ids = pred["terrain_class_ids"]

        valid = (gt_ids != AI4MARS_IGNORE_INDEX) & (gt_ids >= 0) & (gt_ids < AI4MARS_NUM_CLASSES)
        ignored_pixels += int(np.sum(~valid))

        if np.any(valid):
            t = gt_ids[valid].astype(np.int64)
            p = pred_ids[valid].astype(np.int64)
            encoded = AI4MARS_NUM_CLASSES * t + p
            conf += np.bincount(encoded, minlength=AI4MARS_NUM_CLASSES ** 2).reshape(
                AI4MARS_NUM_CLASSES, AI4MARS_NUM_CLASSES
            )

        if idx < num_visual_samples:
            plot_path = plots_dir / f"test_min3_eval_{idx:02d}_{entry.get('sample_id', idx)}.png"
            render_prediction_comparison(
                image_rgb=img_rgb,
                terrain_class_ids=pred_ids,
                confidence=pred["confidence"],
                uncertainty_score=pred["uncertainty_score"],
                ground_truth_ids=gt_ids,
                save_path=plot_path,
                show=False,
            )
            saved_plots.append(str(plot_path))

    metrics = compute_detailed_metrics(conf)
    metrics["test_split"] = "test_min3"
    metrics["evaluated_samples"] = len(test_samples)
    metrics["ignored_pixels"] = ignored_pixels
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["saved_plots"] = saved_plots

    report_path = OUTPUTS_DIR / "reports" / "evaluation_test_min3.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LightweightUNet on untouched AI4MARS test_min3")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--splits", type=Path, default=DATA_PROCESSED_DIR / "ai4mars_splits.json")
    parser.add_argument("--num-visuals", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    metrics = evaluate_on_test_min3(
        checkpoint_path=args.checkpoint,
        splits_path=args.splits,
        num_visual_samples=args.num_visuals,
        max_eval_samples=args.max_samples,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
