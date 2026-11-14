"""Prepare deterministic train/validation splits and measure class balance on AI4MARS.

Important:
- `train` is split into `train` and `val` deterministically.
- `test_min3` is kept completely untouched as the final evaluation split.
- Class weights are computed ONLY from the training split.
"""

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
)
from ml.dataset import decode_ai4mars_rgb_mask


def _collect_split_pairs(split_dir: Path) -> list[dict[str, str]]:
    """Collect sorted, paired image/mask paths inside a split directory."""
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        return []

    pairs: list[dict[str, str]] = []
    for img_path in sorted(images_dir.glob("*.png")):
        mask_path = masks_dir / img_path.name
        if mask_path.is_file():
            pairs.append(
                {
                    "sample_id": img_path.stem,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                }
            )
    return pairs


def measure_class_balance(
    samples: list[dict[str, str]],
) -> dict[str, Any]:
    """Measure per-class labelled pixel counts and compute bounded ENet class weights."""
    class_counts = np.zeros(AI4MARS_NUM_CLASSES, dtype=np.int64)
    ignore_count = 0
    unexpected_total = 0

    for entry in samples:
        with Image.open(entry["mask_path"]) as mask_raw:
            id_mask, unexpected = decode_ai4mars_rgb_mask(mask_raw)
        if unexpected:
            unexpected_total += sum(unexpected.values())

        ignore_count += int(np.sum(id_mask == AI4MARS_IGNORE_INDEX))
        for cid in range(AI4MARS_NUM_CLASSES):
            class_counts[cid] += int(np.sum(id_mask == cid))

    total_labelled = int(np.sum(class_counts))
    if total_labelled == 0:
        raise RuntimeError("No labelled pixels (classes 0..3) found in training split.")

    frequencies = class_counts.astype(np.float64) / float(total_labelled)
    # Bounded inverse-log weighting: w_c = 1 / ln(1.02 + p_c)
    raw_weights = 1.0 / np.log(1.02 + np.maximum(frequencies, 1e-6))
    normalized_weights = raw_weights / np.mean(raw_weights)

    return {
        "total_labelled_pixels": total_labelled,
        "ignored_pixels": ignore_count,
        "unexpected_color_pixels": unexpected_total,
        "class_counts": {AI4MARS_CLASS_NAMES[i]: int(class_counts[i]) for i in range(AI4MARS_NUM_CLASSES)},
        "class_frequencies": {AI4MARS_CLASS_NAMES[i]: round(float(frequencies[i]), 6) for i in range(AI4MARS_NUM_CLASSES)},
        "recommended_class_weights": [round(float(w), 6) for w in normalized_weights],
    }


def prepare_splits(
    raw_dir: Path = DATA_RAW_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Create deterministic train/val split from `train` and preserve `test_min3` separately."""
    if not (0.0 < val_fraction < 0.5):
        raise ValueError(f"val_fraction must be in (0.0, 0.5); got {val_fraction}.")

    all_train = _collect_split_pairs(raw_dir / "train")
    test_min3 = _collect_split_pairs(raw_dir / "test_min3")

    if len(all_train) < 2:
        raise RuntimeError(
            f"Need at least 2 paired samples in '{raw_dir / 'train'}' to create train/val splits; "
            f"found {len(all_train)}."
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_train))
    rng.shuffle(indices)

    val_count = max(1, int(round(len(all_train) * val_fraction)))
    val_indices = set(indices[:val_count].tolist())

    train_samples = [all_train[i] for i in range(len(all_train)) if i not in val_indices]
    val_samples = [all_train[i] for i in range(len(all_train)) if i in val_indices]

    balance_stats = measure_class_balance(train_samples)

    payload = {
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test_min3": len(test_min3),
        },
        "train_class_balance": balance_stats,
        "train": train_samples,
        "val": val_samples,
        "test_min3": test_min3,
    }

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_file = processed_dir / "ai4mars_splits.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic AI4MARS train/val splits")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = prepare_splits(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print("Prepared AI4MARS splits:", payload["counts"])
    print("Train class balance:", json.dumps(payload["train_class_balance"], indent=2))


if __name__ == "__main__":
    main()
