#!/usr/bin/env python3
# =============================================================================
# Galaxy Classifier — CHANCES Project
# Copyright (c) 2025 CHANCES Collaboration
# License: MIT (see LICENSE file)
# =============================================================================
"""
inference_custom.py — Standalone inference script for unlabeled images.

Use this script when you have a folder of new images and a trained checkpoint
but do **not** need to run the full pipeline via main.py.

Usage
-----
    python inference_custom.py \\
        --image_folder /path/to/new_images \\
        --checkpoint   /path/to/checkpoint.ckpt \\
        --output       predictions.csv \\
        --class_names  galaxies,nothing,spurious,offset,stars \\
        --img_size     256

See --help for all options.
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer import GalaxyClassifierTrainer  # noqa: E402
from src.data_preparation import build_inference_manifest  # noqa: E402


def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Galaxy Classifier — standalone inference on new images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--image_folder", required=True,
                   help="Folder containing images to classify.")
    p.add_argument("--checkpoint",   required=True,
                   help="Path to the trained model checkpoint (.ckpt).")

    # Optional
    p.add_argument("--output",       default="predictions.csv",
                   help="Output CSV filename.")
    p.add_argument("--class_names",  default=None,
                   help="Comma-separated class names, e.g. 'galaxies,stars'.")
    p.add_argument("--experiment_dir", default="./inference_temp/",
                   help="Temporary directory for the trainer.")
    p.add_argument("--img_size",     type=int,  default=256,
                   help="Image size used during training (pixels, square).")
    p.add_argument("--greyscale",    action="store_true",
                   help="Use greyscale images.")
    p.add_argument("--device",
                   choices=["auto", "cpu", "gpu"], default="auto",
                   help="Inference device.")
    p.add_argument("--recursive",    action="store_true",
                   help="Scan sub-directories recursively.")
    p.add_argument("--batch_size",   type=int, default=32,
                   help="Inference batch size.")

    return p.parse_args()


def resolve_num_classes(checkpoint_path: str, fallback: int | None = None) -> int:
    """
    Try to infer num_classes from the checkpoint's state dict.

    Falls back to *fallback* if inference fails, or prompts the user
    interactively if *fallback* is also None.
    """
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("state_dict", {})
        for key, tensor in state.items():
            if "classifier" in key and "weight" in key:
                return int(tensor.shape[0])
    except Exception as exc:
        logger.debug("Could not read num_classes from checkpoint: %s", exc)

    if fallback is not None:
        return fallback

    logger.warning("Could not determine num_classes from checkpoint.")
    try:
        return int(input("Enter the number of classes: "))
    except (ValueError, EOFError):
        logger.error(
            "Specify --class_names or provide a checkpoint with class info."
        )
        sys.exit(1)


def main():
    args = parse_args()

    # 1. Build manifest
    logger.info("Scanning folder: %s", args.image_folder)
    try:
        catalog_df = build_inference_manifest(args.image_folder, recursive=args.recursive)
    except Exception as exc:
        logger.error("Could not build manifest: %s", exc)
        sys.exit(1)

    # 2. Resolve class names and count
    if args.class_names:
        class_names = [n.strip() for n in args.class_names.split(",")]
        num_classes  = len(class_names)
    else:
        num_classes  = resolve_num_classes(args.checkpoint)
        class_names  = [f"class_{i}" for i in range(num_classes)]

    logger.info("Classes (%d): %s", num_classes, class_names)

    # 3. Run inference
    trainer = GalaxyClassifierTrainer(
        experiment_dir=args.experiment_dir,
        num_classes=num_classes,
        img_size=args.img_size,
        greyscale=args.greyscale,
    )

    try:
        trainer.run_inference(
            test_catalog=catalog_df,
            checkpoint_path=args.checkpoint,
            output_name=args.output,
            class_names=class_names,
            include_ground_truth=False,
        )
        logger.info("Done. Results saved to: %s", args.output)
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
