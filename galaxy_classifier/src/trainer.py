# =============================================================================
# Galaxy Classifier — CHANCES Project
# Copyright (c) 2025 CHANCES Collaboration
# License: MIT (see LICENSE file)
# =============================================================================
"""
src/trainer.py — Zoobot fine-tuning and inference wrapper.

Wraps the Zoobot PyTorch API to provide a simple interface for:
  - Fine-tuning a pre-trained Zoobot encoder on custom classes (head-only).
  - Running batch inference over image catalogs.

Dependencies (must be installed separately):
    pip install zoobot[pytorch] galaxy-datasets
"""

import os
import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class GalaxyClassifierTrainer:
    """
    Fine-tune and run inference with a Zoobot-based galaxy classifier.

    The trainer wraps ``FinetuneableZoobotClassifier`` from the Zoobot library
    and provides a high-level API that integrates with the rest of this pipeline.

    Args:
        experiment_dir: Directory where checkpoints and result CSVs are saved.
        num_classes:    Number of output classes.
        img_size:       Square image resolution fed to the model (pixels).
        greyscale:      Whether to treat images as single-channel greyscale.

    Example:
        >>> trainer = GalaxyClassifierTrainer(
        ...     experiment_dir="./results/",
        ...     num_classes=5,
        ...     img_size=256,
        ... )
        >>> trainer.run_training(train_df, epochs=30)
        >>> preds = trainer.run_inference(test_df)
    """

    #: Pre-trained encoder used as backbone (hosted on HuggingFace Hub).
    ZOOBOT_ENCODER = "hf_hub:mwalmsley/zoobot-encoder-convnext_tiny"

    def __init__(
        self,
        experiment_dir: str,
        num_classes: int = 5,
        img_size: int = 256,
        greyscale: bool = False,
    ) -> None:
        self.save_dir = experiment_dir
        self.img_size = img_size
        self.num_classes = num_classes
        self.greyscale = greyscale

        os.makedirs(experiment_dir, exist_ok=True)
        self._import_zoobot()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _import_zoobot(self) -> None:
        """
        Lazily import Zoobot modules so the rest of the codebase does not
        break when Zoobot is not installed (e.g. during testing or linting).

        Raises:
            ImportError: If zoobot or galaxy_datasets are not installed.
        """
        try:
            from zoobot.pytorch.training import finetune
            from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
            from galaxy_datasets.transforms import (
                default_view_config,
                get_galaxy_transform,
            )
            from zoobot.pytorch.predictions import predict_on_catalog

            self.finetune            = finetune
            self.CatalogDataModule   = CatalogDataModule
            self.default_view_config = default_view_config
            self.get_galaxy_transform = get_galaxy_transform
            self.predict_on_catalog  = predict_on_catalog

        except ImportError as exc:
            logger.error(
                "Zoobot is not installed. "
                "Run: pip install 'zoobot[pytorch]' galaxy-datasets"
            )
            raise exc

    def _build_transform(self):
        """Return the standard galaxy image transform for the current config."""
        cfg = self.default_view_config()
        cfg.output_size = self.img_size
        cfg.greyscale   = self.greyscale
        return self.get_galaxy_transform(cfg)

    def _resolve_checkpoint(self, checkpoint_path: Optional[str]) -> str:
        """
        Resolve a checkpoint path: if *None*, pick the latest ``.ckpt`` file
        inside ``<experiment_dir>/checkpoints/``.

        Raises:
            FileNotFoundError: If no checkpoint can be found.
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            return checkpoint_path

        ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            candidates = sorted(
                f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")
            )
            if candidates:
                resolved = os.path.join(ckpt_dir, candidates[-1])
                logger.info("Auto-selected checkpoint: %s", resolved)
                return resolved

        raise FileNotFoundError(
            f"No checkpoint found at '{checkpoint_path}' "
            f"or inside '{os.path.join(self.save_dir, 'checkpoints')}'."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_training(
        self,
        train_catalog: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 32,
        accelerator: str = "auto",
        patience: int = 10,
        devices: str = "auto",
    ):
        """
        Fine-tune the Zoobot encoder head on *train_catalog*.

        Only the classification head is trained (``training_mode='head_only'``),
        keeping the pre-trained encoder frozen. This is typically sufficient
        and much faster than full fine-tuning.

        Args:
            train_catalog: DataFrame with at minimum ``file_loc`` and ``label``
                columns, as produced by :func:`~src.data_preparation.build_dataset_manifest`.
            epochs:        Maximum training epochs (early stopping may stop sooner).
            batch_size:    Images per gradient-update step.
            accelerator:   PyTorch Lightning accelerator — ``'auto'``, ``'cpu'``, ``'gpu'``.
            patience:      Early-stopping patience (epochs without val improvement).
            devices:       Number of accelerator devices or ``'auto'``.

        Returns:
            Trained ``FinetuneableZoobotClassifier`` model.
        """
        logger.info(
            "Starting training — %d images, %d classes, up to %d epochs.",
            len(train_catalog), self.num_classes, epochs,
        )

        transform = self._build_transform()

        dm = self.CatalogDataModule(
            catalog=train_catalog,
            label_cols=["label"],
            batch_size=batch_size,
            train_transform=transform,
            test_transform=transform,
        )

        model = self.finetune.FinetuneableZoobotClassifier(
            name=self.ZOOBOT_ENCODER,
            num_classes=self.num_classes,
            training_mode="head_only",
            label_col="label",
            greyscale=self.greyscale,
        )

        trainer = self.finetune.get_trainer(
            self.save_dir,
            accelerator=accelerator,
            max_epochs=epochs,
            patience=patience,
            devices=devices,
        )

        trainer.fit(model, dm)
        logger.info("Training complete. Checkpoints saved to: %s/checkpoints/", self.save_dir)
        return model

    def run_inference(
        self,
        test_catalog: pd.DataFrame,
        checkpoint_path: Optional[str] = None,
        output_name: str = "predictions.csv",
        class_names: Optional[List[str]] = None,
        include_ground_truth: bool = True,
    ) -> pd.DataFrame:
        """
        Run batch inference and return a DataFrame of class probabilities.

        Args:
            test_catalog:         DataFrame with at minimum ``id_str`` and
                ``file_loc`` columns.  If *include_ground_truth* is ``True``
                and a ``label`` column is present, it will be merged into the
                output for downstream evaluation.
            checkpoint_path:      Path to a ``.ckpt`` file. If ``None``, the
                latest checkpoint inside ``<experiment_dir>/checkpoints/`` is
                used automatically.
            output_name:          Filename for the CSV saved inside
                ``<experiment_dir>``.
            class_names:          If provided (length == num_classes), these
                become the probability column names. Otherwise columns are
                named ``class_0``, ``class_1``, …
            include_ground_truth: Merge the ``label`` column from *test_catalog*
                into the output when available.

        Returns:
            DataFrame with ``id_str``, ``file_loc``, optionally ``label``,
            and one probability column per class.
        """
        ckpt = self._resolve_checkpoint(checkpoint_path)
        logger.info("Loading checkpoint: %s", ckpt)

        model = self.finetune.FinetuneableZoobotClassifier.load_from_checkpoint(ckpt)

        # Column names for class probabilities
        if class_names and len(class_names) == self.num_classes:
            prob_cols = class_names
        else:
            prob_cols = [f"class_{i}" for i in range(self.num_classes)]
            if class_names:
                logger.warning(
                    "len(class_names)=%d != num_classes=%d; "
                    "falling back to generic column names.",
                    len(class_names), self.num_classes,
                )

        transform = self._build_transform()

        logger.info("Running inference on %d images …", len(test_catalog))
        raw_preds = self.predict_on_catalog.predict(
            test_catalog,
            model,
            inference_transform=transform,
            label_cols=prob_cols,
            save_loc=None,
            datamodule_kwargs={"batch_size": 32},
            trainer_kwargs={"accelerator": "auto", "devices": "auto"},
        )

        # Build list of metadata columns to merge
        merge_cols = ["id_str", "file_loc"]
        if include_ground_truth and "label" in test_catalog.columns:
            merge_cols.append("label")
            logger.info("Ground truth ('label') included in output.")

        missing = [c for c in merge_cols if c not in test_catalog.columns and c != "label"]
        if missing:
            raise KeyError(
                f"test_catalog is missing required columns: {missing}"
            )

        results = pd.merge(raw_preds, test_catalog[merge_cols], on="id_str")

        save_path = os.path.join(self.save_dir, output_name)
        results.to_csv(save_path, index=False)
        logger.info("Predictions saved to: %s", save_path)

        return results
