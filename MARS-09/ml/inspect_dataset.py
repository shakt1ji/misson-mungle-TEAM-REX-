"""Dataset validation, RGB label verification, class statistics, and visual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ml.config import (
    AI4MARS_CLASS_NAMES,
    AI4MARS_IGNORE_INDEX,
    AI4MARS_NUM_CLASSES,
    DATA_RAW_DIR,
    OUTPUTS_DIR,
)
from ml.dataset import decode_ai4mars_rgb_mask
from perception.visualization import colorize_terrain_mask


def inspect_split(
    split_dir: Path,
    split_name: str,
    plots_dir: Path | None = None,
    num_visual_samples: int = 4,
) -> dict[str, Any]:
    """Validate image-mask pairs, dimensions, RGB encodings, and class frequencies."""
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"

    if not images_dir.is_dir() or not masks_dir.is_dir():
        return {
            "split": split_name,
            "paired_samples": 0,
            "error": f"Missing images/ or masks/ directory in '{split_dir}'.",
        }

    img_files = sorted(images_dir.glob("*.png"))
    mask_map = {p.name: p for p in masks_dir.glob("*.png")}

    paired: list[tuple[Path, Path]] = []
    missing_masks: list[str] = []
    for img_p in img_files:
        if img_p.name in mask_map:
            paired.append((img_p, mask_map[img_p.name]))
        else:
            missing_masks.append(img_p.name)

    image_dims: dict[str, int] = {}
    mask_dims: dict[str, int] = {}
    corrupt_samples: list[str] = []
    dimension_mismatches: list[str] = []
    unexpected_rgb_colors: dict[str, int] = {}

    class_counts = np.zeros(AI4MARS_NUM_CLASSES, dtype=np.int64)
    ignored_pixels = 0
    total_pixels = 0

    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
    saved_visuals: list[str] = []

    for idx, (img_p, mask_p) in enumerate(paired):
        try:
            with Image.open(img_p) as img_raw:
                img_rgb = np.asarray(img_raw.convert("RGB"), dtype=np.uint8)
            with Image.open(mask_p) as mask_raw:
                if mask_raw.mode != "RGB":
                    raise ValueError(f"Mask mode is '{mask_raw.mode}', expected 'RGB'.")
                id_mask, unexpected = decode_ai4mars_rgb_mask(mask_raw)
        except Exception as exc:
            corrupt_samples.append(f"{img_p.name}: {exc}")
            continue

        ih, iw = img_rgb.shape[:2]
        mh, mw = id_mask.shape[:2]
        image_dims[f"{ih}x{iw}"] = image_dims.get(f"{ih}x{iw}", 0) + 1
        mask_dims[f"{mh}x{mw}"] = mask_dims.get(f"{mh}x{mw}", 0) + 1

        if (ih, iw) != (mh, mw):
            dimension_mismatches.append(f"{img_p.name}: image={(ih, iw)} mask={(mh, mw)}")
            continue

        for rgb_key, cnt in unexpected.items():
            key_str = str(rgb_key)
            unexpected_rgb_colors[key_str] = unexpected_rgb_colors.get(key_str, 0) + cnt

        total_pixels += ih * iw
        ignored_pixels += int(np.sum(id_mask == AI4MARS_IGNORE_INDEX))
        for cid in range(AI4MARS_NUM_CLASSES):
            class_counts[cid] += int(np.sum(id_mask == cid))

        if plots_dir is not None and len(saved_visuals) < num_visual_samples:
            matplotlib.use("Agg", force=False)
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
            axes[0].imshow(img_rgb)
            axes[0].set_title(f"{split_name}: {img_p.stem}")
            axes[1].imshow(
                colorize_terrain_mask(id_mask, ignore_mask=(id_mask == AI4MARS_IGNORE_INDEX))
            )
            axes[1].set_title("Decoded RGB Ground-Truth Mask")
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            out_plot = plots_dir / f"inspect_{split_name}_{len(saved_visuals):02d}_{img_p.stem}.png"
            fig.savefig(out_plot, dpi=140)
            plt.close(fig)
            saved_visuals.append(str(out_plot))

    labelled_pixels = int(np.sum(class_counts))
    class_freq_labelled = {
        AI4MARS_CLASS_NAMES[i]: round(float(class_counts[i] / labelled_pixels), 6)
        if labelled_pixels > 0
        else 0.0
        for i in range(AI4MARS_NUM_CLASSES)
    }

    return {
        "split": split_name,
        "paired_samples": len(paired),
        "missing_mask_files": len(missing_masks),
        "corrupt_samples": corrupt_samples,
        "dimension_mismatches": dimension_mismatches,
        "image_dimensions": image_dims,
        "mask_dimensions": mask_dims,
        "total_pixels": total_pixels,
        "labelled_pixels": labelled_pixels,
        "ignored_pixels": ignored_pixels,
        "ignored_fraction": round(float(ignored_pixels / total_pixels), 6) if total_pixels > 0 else 0.0,
        "class_pixel_counts": {AI4MARS_CLASS_NAMES[i]: int(class_counts[i]) for i in range(AI4MARS_NUM_CLASSES)},
        "class_frequencies_among_labelled": class_freq_labelled,
        "unexpected_rgb_colors": unexpected_rgb_colors,
        "sample_visuals": saved_visuals,
    }


def inspect_ai4mars_dataset(
    raw_dir: Path = DATA_RAW_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
    num_visual_samples: int = 4,
) -> dict[str, Any]:
    """Inspect both `train` and `test_min3` downloaded AI4MARS splits and save report."""
    plots_dir = outputs_dir / "plots"
    reports_dir = outputs_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_report = inspect_split(
        raw_dir / "train",
        split_name="train",
        plots_dir=plots_dir,
        num_visual_samples=num_visual_samples,
    )
    test_report = inspect_split(
        raw_dir / "test_min3",
        split_name="test_min3",
        plots_dir=plots_dir,
        num_visual_samples=num_visual_samples,
    )

    report = {
        "raw_dir": str(raw_dir),
        "train": train_report,
        "test_min3": test_report,
    }
    report_path = reports_dir / "dataset_inspection.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate downloaded AI4MARS dataset")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--num-visuals", type=int, default=4)
    args = parser.parse_args()

    report = inspect_ai4mars_dataset(
        raw_dir=args.raw_dir,
        outputs_dir=args.outputs_dir,
        num_visual_samples=args.num_visuals,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
