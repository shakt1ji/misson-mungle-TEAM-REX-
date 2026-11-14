"""Training pipeline for LightweightUNet on AI4MARS semantic segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.config import (
    AI4MARS_CLASS_NAMES,
    AI4MARS_IGNORE_INDEX,
    AI4MARS_NUM_CLASSES,
    DATA_PROCESSED_DIR,
    DEFAULT_CHECKPOINT_PATH,
    OUTPUTS_DIR,
)
from ml.dataset import AI4MARSSegmentationDataset
from ml.model import LightweightUNet, count_parameters, save_checkpoint
from ml.prepare_dataset import prepare_splits


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = AI4MARS_NUM_CLASSES,
    ignore_index: int = AI4MARS_IGNORE_INDEX,
) -> np.ndarray:
    """Accumulate (num_classes, num_classes) confusion matrix ignoring unlabelled pixels."""
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    t = targets[valid].astype(np.int64)
    p = preds[valid].astype(np.int64)
    encoded = num_classes * t + p
    counts = np.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def _metrics_from_confusion(conf: np.ndarray) -> dict[str, Any]:
    """Compute pixel accuracy, per-class IoU, and mean IoU from confusion matrix."""
    tp = np.diag(conf).astype(np.float64)
    fp = np.sum(conf, axis=0).astype(np.float64) - tp
    fn = np.sum(conf, axis=1).astype(np.float64) - tp
    denom = tp + fp + fn

    per_class_iou: dict[str, float] = {}
    valid_ious: list[float] = []
    for cid in range(conf.shape[0]):
        if denom[cid] > 0:
            iou = float(tp[cid] / denom[cid])
            valid_ious.append(iou)
        else:
            iou = 0.0
        per_class_iou[AI4MARS_CLASS_NAMES[cid]] = iou

    total_pixels = float(np.sum(conf))
    pixel_acc = float(np.sum(tp) / total_pixels) if total_pixels > 0 else 0.0
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return {
        "pixel_accuracy": pixel_acc,
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou,
    }


def train_model(
    splits_path: Path = DATA_PROCESSED_DIR / "ai4mars_splits.json",
    checkpoint_out: Path = DEFAULT_CHECKPOINT_PATH,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1e-3,
    image_size: tuple[int, int] = (256, 256),
    base_channels: int = 16,
    seed: int = 42,
    use_class_weights: bool = True,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Train LightweightUNet on AI4MARS train/val splits and save best checkpoint."""
    set_reproducible_seed(seed)

    if not splits_path.is_file():
        splits = prepare_splits(processed_dir=splits_path.parent, seed=seed)
    else:
        splits = json.loads(splits_path.read_text(encoding="utf-8"))

    train_list = splits["train"][:max_train_samples] if max_train_samples else splits["train"]
    val_list = splits["val"][:max_val_samples] if max_val_samples else splits["val"]

    if not train_list or not val_list:
        raise RuntimeError("Train and validation sample lists must both be non-empty.")

    train_ds = AI4MARSSegmentationDataset(train_list, image_size=image_size, augment=True, seed=seed)
    val_ds = AI4MARSSegmentationDataset(val_list, image_size=image_size, augment=False, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightUNet(in_channels=3, num_classes=AI4MARS_NUM_CLASSES, base_channels=base_channels).to(device)

    weight_tensor: torch.Tensor | None = None
    if use_class_weights and "train_class_balance" in splits:
        w_list = splits["train_class_balance"].get("recommended_class_weights")
        if w_list and len(w_list) == AI4MARS_NUM_CLASSES:
            weight_tensor = torch.tensor(w_list, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=AI4MARS_IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    param_count = count_parameters(model)
    print(
        f"Training LightweightUNet ({param_count:,} params) on {device} | "
        f"train={len(train_ds)}, val={len(val_ds)}, image_size={image_size}, epochs={epochs}"
    )

    history: dict[str, list[Any]] = {
        "train_loss": [],
        "val_loss": [],
        "val_pixel_accuracy": [],
        "val_mean_iou": [],
    }
    best_val_miou = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += float(loss.item())
            train_batches += 1

        avg_train_loss = running_train_loss / max(train_batches, 1)

        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        conf = np.zeros((AI4MARS_NUM_CLASSES, AI4MARS_NUM_CLASSES), dtype=np.int64)

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                running_val_loss += float(loss.item())
                val_batches += 1

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                conf += _compute_confusion_matrix(preds, masks.cpu().numpy())

        avg_val_loss = running_val_loss / max(val_batches, 1)
        val_metrics = _metrics_from_confusion(conf)

        history["train_loss"].append(round(avg_train_loss, 5))
        history["val_loss"].append(round(avg_val_loss, 5))
        history["val_pixel_accuracy"].append(round(val_metrics["pixel_accuracy"], 5))
        history["val_mean_iou"].append(round(val_metrics["mean_iou"], 5))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_metrics['pixel_accuracy']:.4f} | val_mIoU={val_metrics['mean_iou']:.4f}"
        )

        if val_metrics["mean_iou"] > best_val_miou:
            best_val_miou = val_metrics["mean_iou"]
            best_epoch = epoch
            save_checkpoint(
                path=checkpoint_out,
                model=model,
                epoch=epoch,
                best_val_miou=best_val_miou,
                image_size=image_size,
                history=history,
            )

    summary = {
        "device": str(device),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "image_size": list(image_size),
        "base_channels": int(base_channels),
        "parameter_count": int(param_count),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "use_class_weights": bool(use_class_weights),
        "best_epoch": int(best_epoch),
        "best_val_miou": round(float(best_val_miou), 5),
        "checkpoint_path": str(checkpoint_out),
        "history": history,
    }

    report_path = OUTPUTS_DIR / "reports" / "training_history.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightweightUNet on AI4MARS")
    parser.add_argument("--splits", type=Path, default=DATA_PROCESSED_DIR / "ai4mars_splits.json")
    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    train_model(
        splits_path=args.splits,
        checkpoint_out=args.checkpoint_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=(args.image_size[0], args.image_size[1]),
        base_channels=args.base_channels,
        seed=args.seed,
        use_class_weights=not args.no_class_weights,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()
