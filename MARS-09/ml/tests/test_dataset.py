"""Tests for AI4MARS dataset pairing, nearest-neighbor mask resizing, and tensor shapes."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ml.config import AI4MARS_IGNORE_INDEX
from ml.dataset import AI4MARSSegmentationDataset
from ml.prepare_dataset import _collect_split_pairs


def test_dataset_pairing_and_nearest_neighbor_mask_resize(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    # Paired sample 0001
    img_arr = np.full((64, 80, 3), 120, dtype=np.uint8)
    mask_arr = np.full((64, 80, 3), 255, dtype=np.uint8)
    mask_arr[:32, :40] = (0, 0, 0)
    mask_arr[:32, 40:] = (3, 3, 3)
    mask_arr[32:, :40] = (1, 1, 1)
    mask_arr[32:, 40:] = (2, 2, 2)

    Image.fromarray(img_arr, mode="RGB").save(img_dir / "sample_0001.png")
    Image.fromarray(mask_arr, mode="RGB").save(mask_dir / "sample_0001.png")

    # Unpaired image should be ignored by _collect_split_pairs
    Image.fromarray(img_arr, mode="RGB").save(img_dir / "unpaired_0002.png")

    pairs = _collect_split_pairs(split_dir)
    assert len(pairs) == 1
    assert pairs[0]["sample_id"] == "sample_0001"

    ds = AI4MARSSegmentationDataset(pairs, image_size=(32, 40), augment=False)
    assert len(ds) == 1

    image_tensor, mask_tensor = ds[0]
    assert image_tensor.shape == (3, 32, 40)
    assert image_tensor.dtype == torch.float32
    assert mask_tensor.shape == (32, 40)
    assert mask_tensor.dtype == torch.int64

    # Nearest-neighbor resizing must preserve exact label IDs (no interpolated 1.5/128 values)
    unique_ids = set(torch.unique(mask_tensor).tolist())
    assert unique_ids.issubset({0, 1, 2, 3, AI4MARS_IGNORE_INDEX})
    assert {0, 1, 2, 3}.issubset(unique_ids)
