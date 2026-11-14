"""Memory-safe streaming downloader for hassanjbara/AI4MARS (`train` subset and `test_min3` split).

Streams only the required Parquet shards from Hugging Face Hub without downloading
the entire repository archive, reads only `image`, `label_mask`, and `has_labels`
in small Arrow batches (`batch_size=8`), filters to labelled samples, and saves
paired RGB images and RGB-encoded label masks under `data/raw/ai4mars/`.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
from pathlib import Path
import tempfile
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ml.config import AI4MARS_DATASET_ID, DATA_RAW_DIR


def _save_pil_rgb(pil_obj: Image.Image, out_path: Path) -> str:
    """Save PIL image as lossless RGB PNG and return its original PIL mode."""
    orig_mode = pil_obj.mode
    if orig_mode == "RGB":
        pil_obj.save(out_path, format="PNG", compress_level=1)
    elif orig_mode == "L":
        # Expand single-channel AI4MARS ID mask (0,1,2,3,255) to exact (v,v,v) RGB triple
        rgb_img = pil_obj.convert("RGB")
        try:
            rgb_img.save(out_path, format="PNG", compress_level=1)
        finally:
            rgb_img.close()
    else:
        rgb_img = pil_obj.convert("RGB")
        try:
            rgb_img.save(out_path, format="PNG", compress_level=1)
        finally:
            rgb_img.close()
    return orig_mode


def _decode_hf_image_cell(cell: Any) -> Image.Image:
    """Decode a Hugging Face Parquet image struct {'bytes': ..., 'path': ...} or PIL Image."""
    if isinstance(cell, Image.Image):
        return cell
    if isinstance(cell, dict):
        raw_bytes = cell.get("bytes")
        if raw_bytes is not None:
            return Image.open(io.BytesIO(raw_bytes))
        raw_path = cell.get("path")
        if raw_path:
            return Image.open(raw_path)
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell))
    raise TypeError(f"Unsupported HF image cell type: {type(cell).__name__}")


def _list_split_parquet_files(split_name: str) -> list[str]:
    """List sorted Parquet shard filenames for a given AI4MARS split on Hugging Face."""
    api = HfApi()
    all_files = api.list_repo_files(AI4MARS_DATASET_ID, repo_type="dataset")
    prefix = f"data/{split_name}-"
    shards = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".parquet"))
    if not shards:
        raise RuntimeError(
            f"No Parquet shards found for split '{split_name}' in '{AI4MARS_DATASET_ID}'."
        )
    return shards


def _download_split(
    split_name: str,
    out_split_dir: Path,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Stream Parquet shards one at a time and save paired labelled samples for a split."""
    images_dir = out_split_dir / "images"
    masks_dir = out_split_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    shards = _list_split_parquet_files(split_name)

    seen_rows = 0
    saved_count = 0
    skipped_unlabelled = 0
    skipped_corrupt = 0
    shards_read = 0
    source_image_modes: dict[str, int] = {}
    source_mask_modes: dict[str, int] = {}

    pbar = tqdm(desc=f"Streaming AI4MARS [{split_name}]", unit="img")
    try:
        for shard_file in shards:
            if max_samples is not None and saved_count >= max_samples:
                break

            shards_read += 1
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_parquet = hf_hub_download(
                    repo_id=AI4MARS_DATASET_ID,
                    filename=shard_file,
                    repo_type="dataset",
                    local_dir=tmp_dir,
                )
                pf = pq.ParquetFile(local_parquet)
                for batch in pf.iter_batches(
                    batch_size=8,
                    columns=["image", "label_mask", "has_labels"],
                ):
                    cols = batch.to_pydict()
                    for hl, img_cell, mask_cell in zip(
                        cols["has_labels"], cols["image"], cols["label_mask"]
                    ):
                        seen_rows += 1
                        if not bool(hl) or img_cell is None or mask_cell is None:
                            skipped_unlabelled += 1
                            continue

                        sample_id = f"{split_name}_{saved_count:05d}"
                        img_path = images_dir / f"{sample_id}.png"
                        mask_path = masks_dir / f"{sample_id}.png"

                        img_obj: Image.Image | None = None
                        mask_obj: Image.Image | None = None
                        try:
                            img_obj = _decode_hf_image_cell(img_cell)
                            mask_obj = _decode_hf_image_cell(mask_cell)

                            if img_obj.size != mask_obj.size:
                                raise ValueError(
                                    f"Image size {img_obj.size} does not match mask size {mask_obj.size}."
                                )

                            img_mode = _save_pil_rgb(img_obj, img_path)
                            mask_mode = _save_pil_rgb(mask_obj, mask_path)
                            source_image_modes[img_mode] = source_image_modes.get(img_mode, 0) + 1
                            source_mask_modes[mask_mode] = source_mask_modes.get(mask_mode, 0) + 1
                            saved_count += 1
                            pbar.update(1)
                        except Exception:
                            skipped_corrupt += 1
                            if img_path.exists():
                                img_path.unlink(missing_ok=True)
                            if mask_path.exists():
                                mask_path.unlink(missing_ok=True)
                            continue
                        finally:
                            if img_obj is not None:
                                img_obj.close()
                            if mask_obj is not None:
                                mask_obj.close()

                        if max_samples is not None and saved_count >= max_samples:
                            break

                    del cols, batch
                    pa.default_memory_pool().release_unused()
                    gc.collect()

                    if max_samples is not None and saved_count >= max_samples:
                        break
                del pf
                pa.default_memory_pool().release_unused()
                gc.collect()
    finally:
        pbar.close()
        pa.default_memory_pool().release_unused()
        gc.collect()

    return {
        "split": split_name,
        "shards_read": shards_read,
        "total_shards_available": len(shards),
        "seen_rows": seen_rows,
        "saved_labelled_pairs": saved_count,
        "skipped_unlabelled": skipped_unlabelled,
        "skipped_corrupt": skipped_corrupt,
        "source_image_modes": source_image_modes,
        "source_mask_modes": source_mask_modes,
    }


def download_ai4mars_subsets(
    output_dir: Path = DATA_RAW_DIR,
    max_train_samples: int = 1500,
    max_test_samples: int | None = None,
) -> dict[str, Any]:
    """Download labelled `train` subset and `test_min3` split via shard-bounded streaming."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_stats = _download_split(
        split_name="train",
        out_split_dir=output_dir / "train",
        max_samples=max_train_samples,
    )
    test_stats = _download_split(
        split_name="test_min3",
        out_split_dir=output_dir / "test_min3",
        max_samples=max_test_samples,
    )

    manifest = {
        "dataset": AI4MARS_DATASET_ID,
        "train": train_stats,
        "test_min3": test_stats,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download hassanjbara/AI4MARS subsets via streaming")
    parser.add_argument("--output-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--max-train-samples", type=int, default=1500)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    manifest = download_ai4mars_subsets(
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
