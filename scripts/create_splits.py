#!/usr/bin/env python3
"""Create a versioned train/val/test split file (NFR-015).

Stratifies splits by borehole so no borehole appears in more than one split.

Usage:
    python scripts/create_splits.py \
        --image-dir data/raw \
        --output data/annotations/splits/split_v1.yaml \
        --train 0.70 --val 0.15 --test 0.15 \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main() -> None:
    """Create and write a split YAML file."""
    parser = argparse.ArgumentParser(description="Create dataset split YAML")
    parser.add_argument("--image-dir", required=True, help="Root directory of raw images")
    parser.add_argument("--output", required=True, help="Output YAML file path")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError("Train + val + test ratios must sum to 1.0")

    image_dir = Path(args.image_dir)
    all_images = sorted(
        p for p in image_dir.rglob("*") if p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    if not all_images:
        logging.error("No images found in %s", image_dir)
        return

    # Group images by borehole (assumed: .../borehole_id/image.jpg)
    by_borehole: dict[str, list[Path]] = defaultdict(list)
    for img in all_images:
        borehole = img.parent.name
        by_borehole[borehole].append(img)

    boreholes = sorted(by_borehole.keys())
    random.seed(args.seed)
    random.shuffle(boreholes)

    n = len(boreholes)
    n_train = max(1, round(n * args.train))
    n_val = max(1, round(n * args.val))

    train_bh = boreholes[:n_train]
    val_bh = boreholes[n_train : n_train + n_val]
    test_bh = boreholes[n_train + n_val :]

    def _collect_paths(bhs: list[str]) -> list[str]:
        paths = []
        for bh in bhs:
            for img in by_borehole[bh]:
                paths.append(str(img.relative_to(image_dir.parent)))
        return sorted(paths)

    split_data = {
        "version": "1.0",
        "seed": args.seed,
        "split_date": datetime.now(timezone.utc).date().isoformat(),
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "splits": {
            "train": _collect_paths(train_bh),
            "val": _collect_paths(val_bh),
            "test": _collect_paths(test_bh),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(split_data, default_flow_style=False, sort_keys=False))

    logging.info(
        "Split written to %s: train=%d val=%d test=%d images",
        output_path,
        len(split_data["splits"]["train"]),
        len(split_data["splits"]["val"]),
        len(split_data["splits"]["test"]),
    )


if __name__ == "__main__":
    main()
