#!/usr/bin/env python3
# =============================================================================
# Galaxy Classifier — CHANCES Project
# Copyright (c) 2025 CHANCES Collaboration
# License: MIT (see LICENSE file)
# =============================================================================
"""
main.py — Entry point for the Galaxy Classifier pipeline.

Usage examples
--------------
Full pipeline (dataset → train → inference → evaluate):
    python main.py --config config.ini --step all

Training only (resumable):
    python main.py --config config.ini --step train --resume

Standard inference on the held-out test set:
    python main.py --config config.ini --step inference

Custom inference on a new image folder (no labels required):
    python main.py --config config.ini --step predict \\
        --image_folder /path/to/new_images \\
        --checkpoint   /path/to/checkpoint.ckpt \\
        --output       my_predictions.csv

Evaluation only (loads predictions from disk):
    python main.py --config config.ini --step evaluate
"""

import logging
import os
import sys
from datetime import datetime

import pandas as pd

from config.params import load_config_from_ini
from src.data_preparation import (
    build_dataset_manifest,
    build_inference_manifest,
    prepare_train_test_split,
    save_datasets,
)
from src.evaluator import ModelEvaluator
from src.trainer import GalaxyClassifierTrainer


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """Configure file + console logging and return the root logger."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"main_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_dataset(config: dict):
    """Step 1 — Scan image directories and create train/test CSVs."""
    logger.info("=" * 60)
    logger.info("STEP 1: Dataset creation and split")
    logger.info("=" * 60)

    paths = config["paths"]
    train = config["training"]

    dataset_df = build_dataset_manifest(paths.source_paths)
    if dataset_df.empty:
        logger.error("Dataset is empty. Check the paths in config.ini → [PATHS].")
        sys.exit(1)

    logger.info("Class distribution:")
    for lbl, cnt in dataset_df["label"].value_counts().sort_index().items():
        name = paths.class_names[lbl] if lbl < len(paths.class_names) else f"class_{lbl}"
        logger.info("  %-12s : %d images", name, cnt)

    train_df, test_df = prepare_train_test_split(
        dataset_df,
        test_size=train.test_size,
        random_state=train.random_state,
    )
    save_datasets(dataset_df, train_df, test_df, train.experiment_dir)
    logger.info("Datasets saved to: %s", train.experiment_dir)

    return dataset_df, train_df, test_df


def step_train(config: dict, resume: bool = False):
    """Step 2 — Fine-tune Zoobot on the training split."""
    logger.info("=" * 60)
    logger.info("STEP 2: Model training")
    logger.info("=" * 60)

    train = config["training"]
    train_path = os.path.join(train.experiment_dir, "train_split.csv")

    if not os.path.exists(train_path):
        logger.error(
            "Training CSV not found: %s  — run --step dataset first.", train_path
        )
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    logger.info("Training set loaded: %d images.", len(train_df))

    trainer = GalaxyClassifierTrainer(
        experiment_dir=train.experiment_dir,
        num_classes=train.num_classes,
        img_size=train.img_size,
        greyscale=train.greyscale,
    )

    try:
        trainer.run_training(
            train_catalog=train_df,
            epochs=train.epochs,
            batch_size=train.batch_size,
            accelerator=train.accelerator,
            patience=train.patience,
            devices=train.devices,
        )
        logger.info("Training complete.")
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)


def step_inference(config: dict) -> pd.DataFrame:
    """Step 3 — Run inference on the held-out test split."""
    logger.info("=" * 60)
    logger.info("STEP 3: Inference on test set")
    logger.info("=" * 60)

    train = config["training"]
    infer = config["inference"]
    paths = config["paths"]

    test_path = os.path.join(train.experiment_dir, "test_split.csv")
    if not os.path.exists(test_path):
        logger.error(
            "Test CSV not found: %s  — run --step dataset first.", test_path
        )
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    logger.info("Test set loaded: %d images.", len(test_df))

    trainer = GalaxyClassifierTrainer(
        experiment_dir=train.experiment_dir,
        num_classes=train.num_classes,
        img_size=train.img_size,
        greyscale=train.greyscale,
    )

    try:
        preds = trainer.run_inference(
            test_catalog=test_df,
            checkpoint_path=infer.checkpoint_path,
            output_name=infer.output_name,
            class_names=paths.class_names,
            include_ground_truth=True,
        )
        logger.info(
            "Predictions saved to: %s",
            os.path.join(train.experiment_dir, infer.output_name),
        )
        return preds
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        sys.exit(1)


def step_predict(config: dict, args) -> pd.DataFrame:
    """Step 4 (custom) — Inference on a new, unlabeled image folder."""
    logger.info("=" * 60)
    logger.info("STEP: Custom prediction on new images")
    logger.info("=" * 60)

    train = config["training"]
    paths = config["paths"]

    # Build manifest
    try:
        catalog_df = build_inference_manifest(
            args.image_folder,
            recursive=getattr(args, "recursive", False),
        )
    except Exception as exc:
        logger.error("Failed to build inference manifest: %s", exc)
        sys.exit(1)

    # Resolve class names (CLI > config)
    if getattr(args, "class_names", None):
        class_names = [n.strip() for n in args.class_names.split(",")]
    else:
        class_names = paths.class_names

    output_name = getattr(args, "output", None) or "custom_predictions.csv"

    trainer = GalaxyClassifierTrainer(
        experiment_dir=train.experiment_dir,
        num_classes=len(class_names),
        img_size=train.img_size,
        greyscale=train.greyscale,
    )

    try:
        preds = trainer.run_inference(
            test_catalog=catalog_df,
            checkpoint_path=args.checkpoint,
            output_name=output_name,
            class_names=class_names,
            include_ground_truth=False,
        )
        logger.info("Custom predictions saved to: %s", output_name)
        return preds
    except Exception as exc:
        logger.error("Custom inference failed: %s", exc)
        sys.exit(1)


def step_evaluate(config: dict, predictions_df: pd.DataFrame | None = None):
    """Step 5 — Compute metrics and generate plots."""
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluation")
    logger.info("=" * 60)

    train = config["training"]
    paths = config["paths"]
    infer = config["inference"]

    # Load predictions from disk if not passed in memory
    if predictions_df is None:
        pred_path = os.path.join(train.experiment_dir, infer.output_name)
        if not os.path.exists(pred_path):
            logger.error("Predictions file not found: %s", pred_path)
            sys.exit(1)
        predictions_df = pd.read_csv(pred_path)
        logger.info("Predictions loaded from: %s", pred_path)

    # Merge ground truth if missing
    if "label" not in predictions_df.columns:
        logger.warning("'label' column missing — merging from test split.")
        test_path = os.path.join(train.experiment_dir, "test_split.csv")
        if not os.path.exists(test_path):
            logger.error("Cannot evaluate without ground truth and test split.")
            sys.exit(1)
        test_df = pd.read_csv(test_path)
        predictions_df = pd.merge(
            predictions_df, test_df[["id_str", "label"]], on="id_str", how="left"
        )
        n_null = predictions_df["label"].isna().sum()
        if n_null:
            logger.warning("%d rows have no ground truth and will be dropped.", n_null)
            predictions_df = predictions_df.dropna(subset=["label"]).copy()
        predictions_df["label"] = predictions_df["label"].astype(int)

    evaluator = ModelEvaluator(class_names=paths.class_names)

    try:
        y_true, y_pred, metrics = evaluator.compute_metrics(predictions_df)
    except Exception as exc:
        logger.error("Metrics computation failed: %s", exc)
        logger.info("Available columns: %s", list(predictions_df.columns))
        sys.exit(1)

    evaluator.plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(train.experiment_dir, "confusion_matrix.png"),
    )
    evaluator.plot_class_distribution(
        predictions_df,
        save_path=os.path.join(train.experiment_dir, "class_distribution.png"),
    )
    evaluator.save_metrics_report(
        metrics,
        save_path=os.path.join(train.experiment_dir, "metrics_report.json"),
    )
    logger.info("Evaluation complete. Accuracy: %.4f", metrics["accuracy"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Galaxy Classifier — CHANCES Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", default="config.ini",
        help="Path to configuration INI file (default: config.ini)",
    )
    parser.add_argument(
        "--step",
        choices=["all", "dataset", "train", "inference", "evaluate", "predict"],
        default="all",
        help="Pipeline step to execute.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint.",
    )

    # ---- predict-only arguments -------------------------------------------
    predict = parser.add_argument_group("predict step arguments")
    predict.add_argument("--image_folder", default=None,
                         help="Folder of images to classify (predict step).")
    predict.add_argument("--checkpoint",   default=None,
                         help="Path to model checkpoint (.ckpt).")
    predict.add_argument("--output",       default=None,
                         help="Output CSV filename.")
    predict.add_argument("--class_names",  default=None,
                         help="Comma-separated class names override.")
    predict.add_argument("--recursive",    action="store_true",
                         help="Scan sub-directories recursively.")
    predict.add_argument("--batch_size",   type=int, default=None,
                         help="Inference batch size.")
    predict.add_argument("--device",
                         choices=["auto", "cpu", "gpu"], default=None,
                         help="Inference device.")

    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config_from_ini(args.config)

    if args.step in ("all", "dataset"):
        step_dataset(config)

    if args.step in ("all", "train"):
        step_train(config, resume=args.resume)

    if args.step in ("all", "inference"):
        preds = step_inference(config)

    if args.step == "predict":
        if not args.image_folder:
            logger.error("--image_folder is required for the 'predict' step.")
            sys.exit(1)
        if not args.checkpoint:
            logger.error("--checkpoint is required for the 'predict' step.")
            sys.exit(1)
        step_predict(config, args)

    if args.step in ("all", "evaluate"):
        if args.step == "all":
            train = config["training"]
            infer = config["inference"]
            pred_path = os.path.join(train.experiment_dir, infer.output_name)
            preds_df  = pd.read_csv(pred_path) if os.path.exists(pred_path) else None
            if preds_df is None:
                logger.error(
                    "Predictions not found at %s. "
                    "Run --step all from the beginning.", pred_path
                )
                sys.exit(1)
            step_evaluate(config, preds_df)
        else:
            step_evaluate(config)

    logger.info("=" * 60)
    logger.info("Pipeline finished successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
