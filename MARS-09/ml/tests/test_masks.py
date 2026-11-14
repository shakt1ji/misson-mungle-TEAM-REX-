"""Tests for AI4MARS RGB segmentation mask decoding and error handling."""

import numpy as np
from PIL import Image
import pytest

from ml.config import AI4MARS_IGNORE_INDEX
from ml.dataset import decode_ai4mars_rgb_mask


def test_decode_exact_ai4mars_rgb_classes_and_unlabelled() -> None:
    mask_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    mask_rgb[0, 0] = (0, 0, 0)       # soil -> 0
    mask_rgb[0, 1] = (1, 1, 1)       # bedrock -> 1
    mask_rgb[0, 2] = (2, 2, 2)       # sand -> 2
    mask_rgb[1, 0] = (3, 3, 3)       # big_rock -> 3
    mask_rgb[1, 1] = (255, 255, 255) # unlabelled -> 255
    mask_rgb[1, 2] = (0, 0, 0)       # soil -> 0

    decoded, unexpected = decode_ai4mars_rgb_mask(mask_rgb)

    assert decoded.dtype == np.uint8
    assert decoded.shape == (2, 3)
    assert unexpected == {}
    assert decoded.tolist() == [
        [0, 1, 2],
        [3, AI4MARS_IGNORE_INDEX, 0],
    ]


def test_unexpected_rgb_colors_are_reported_and_ignored() -> None:
    mask_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    mask_rgb[0, 0] = (1, 1, 1)
    mask_rgb[0, 1] = (128, 0, 128)  # unexpected color
    mask_rgb[1, 0] = (128, 0, 128)  # unexpected color
    mask_rgb[1, 1] = (2, 2, 2)

    decoded, unexpected = decode_ai4mars_rgb_mask(mask_rgb)

    assert decoded[0, 0] == 1
    assert decoded[1, 1] == 2
    assert decoded[0, 1] == AI4MARS_IGNORE_INDEX
    assert decoded[1, 0] == AI4MARS_IGNORE_INDEX
    assert unexpected == {(128, 0, 128): 2}


def test_grayscale_mask_rejected() -> None:
    gray_arr = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="Grayscale"):
        decode_ai4mars_rgb_mask(gray_arr)

    gray_pil = Image.fromarray(gray_arr, mode="L")
    with pytest.raises(ValueError, match="RGB mode"):
        decode_ai4mars_rgb_mask(gray_pil)
