# =============================================================================
# Galaxy Classifier — CHANCES Project
# Copyright (c) 2025 CHANCES Collaboration
# License: MIT (see LICENSE file)
# =============================================================================
"""
src/evaluator.py — Metrics computation and visualisation for multi-class classifiers.

Provides :class:`ModelEvaluator`, which computes accuracy, F1-score, per-class
metrics, confusion matrices and class-distribution plots from a predictions
DataFrame produced by :meth:`~src.trainer.GalaxyClassifierTrainer.run_inference`.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate a multi-class image classifier from a predictions DataFrame.

    Args:
        class_names: Ordered list of class names matching label indices 0, 1, …

    Example:
        >>> evaluator = ModelEvaluator(["galaxies", "nothing", "spurious", "offset", "stars"])
        >>> y_true, y_pred, metrics = evaluator.compute_metrics(predictions_df)
        >>> evaluator.plot_confusion_matrix(y_true, y_pred)
    """

    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        df: pd.DataFrame,
        prob_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Compute classification metrics from a predictions DataFrame.

        The function expects:

        * A ``label`` column with integer ground-truth labels (0, 1, …).
        * One probability column per class.  Column detection order:

          1. Columns named after *class_names* (e.g. ``galaxies``, ``stars``).
          2. Columns named ``class_0``, ``class_1``, …
          3. First *N* numeric columns (fallback with a warning).

        Args:
            df:           Predictions DataFrame (output of
                :meth:`~src.trainer.GalaxyClassifierTrainer.run_inference`).
            prob_columns: Override automatic column detection.

        Returns:
            ``(y_true, y_pred, metrics_dict)`` where *metrics_dict* contains
            accuracy, macro/weighted F1, recall, and per-class breakdowns.

        Raises:
            ValueError: If the ``label`` column is absent.
            KeyError:   If probability columns cannot be found.
        """
        if "label" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'label' column with ground-truth labels. "
                "Ensure include_ground_truth=True was set in run_inference(), or merge "
                "manually with the test split before calling compute_metrics()."
            )

        prob_columns = self._resolve_prob_columns(df, prob_columns)

        y_true  = df["label"].values
        y_probs = df[prob_columns].values

        if y_probs.shape[1] != len(self.class_names):
            logger.warning(
                "Probability columns (%d) ≠ class count (%d). "
                "Truncating to first %d columns.",
                y_probs.shape[1], len(self.class_names), len(self.class_names),
            )
            y_probs = y_probs[:, : len(self.class_names)]

        y_pred = np.argmax(y_probs, axis=1)

        acc = accuracy_score(y_true, y_pred)

        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )

        metrics: Dict = {
            "accuracy":            acc,
            "f1_macro":            f1_macro,
            "f1_weighted":         f1_w,
            "recall_macro":        rec_macro,
            "recall_weighted":     rec_w,
            "precision_per_class": prec_pc.tolist(),
            "recall_per_class":    rec_pc.tolist(),
            "f1_per_class":        f1_pc.tolist(),
            "support_per_class":   sup_pc.tolist(),
        }

        self._log_report(y_true, y_pred, acc, f1_macro, f1_w)
        return y_true, y_pred, metrics

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "Blues",
        save_path: str = "confusion_matrix.png",
    ) -> None:
        """
        Plot and save a confusion matrix heatmap.

        Args:
            y_true:     Ground-truth labels.
            y_pred:     Predicted labels.
            normalize:  If ``True``, normalise rows to show per-class recall.
            figsize:    Matplotlib figure size.
            cmap:       Seaborn colormap name.
            save_path:  Output file path (PNG).
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm    = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            title = "Normalised Confusion Matrix (Recall)"
            fmt   = ".2f"
        else:
            title = "Confusion Matrix (Counts)"
            fmt   = "d"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
            square=True,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to: %s", save_path)
        plt.show()
        plt.close(fig)

    def plot_class_distribution(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        figsize: Tuple[int, int] = (10, 5),
        save_path: str = "class_distribution.png",
    ) -> None:
        """
        Plot and save a bar chart of class frequencies.

        Args:
            df:        DataFrame containing *label_col*.
            label_col: Name of the integer label column.
            figsize:   Matplotlib figure size.
            save_path: Output file path (PNG).
        """
        if label_col not in df.columns:
            logger.error("Column '%s' not found in DataFrame.", label_col)
            return

        counts      = df[label_col].value_counts().sort_index()
        tick_labels = [
            self.class_names[i] for i in counts.index if i < len(self.class_names)
        ]

        fig, ax = plt.subplots(figsize=figsize)
        palette = sns.color_palette("viridis", len(counts))
        bars    = ax.bar(tick_labels, counts.values, color=palette)

        for bar, n in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(n),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_title("Class Distribution in Dataset", fontsize=14, pad=20)
        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Number of Images", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Class distribution plot saved to: %s", save_path)
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_metrics_report(
        self,
        metrics: Dict,
        save_path: str = "metrics_report.json",
    ) -> None:
        """Serialise the metrics dictionary to a JSON file."""
        with open(save_path, "w") as fh:
            json.dump(metrics, fh, indent=4)
        logger.info("Metrics report saved to: %s", save_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_prob_columns(
        self,
        df: pd.DataFrame,
        prob_columns: Optional[List[str]],
    ) -> List[str]:
        """Return the list of probability column names, auto-detecting if needed."""
        if prob_columns is not None:
            return prob_columns

        # Option 1 — columns named after class_names
        if all(c in df.columns for c in self.class_names):
            logger.info("Using class-name columns: %s", self.class_names)
            return list(self.class_names)

        # Option 2 — generic class_0 … class_N columns
        generic = [f"class_{i}" for i in range(len(self.class_names))]
        missing = [c for c in generic if c not in df.columns]
        if not missing:
            logger.info("Using generic columns: %s", generic)
            return generic

        # Option 3 — first N numeric columns (last resort)
        numeric = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ("label", "id")
        ]
        if len(numeric) >= len(self.class_names):
            cols = numeric[: len(self.class_names)]
            logger.warning(
                "Could not find expected probability columns; "
                "falling back to first %d numeric columns: %s",
                len(cols), cols,
            )
            return cols

        raise KeyError(
            f"Cannot find probability columns. "
            f"Looked for: {self.class_names} and {generic}. "
            f"Available columns: {list(df.columns)}"
        )

    def _log_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        acc: float,
        f1_macro: float,
        f1_weighted: float,
    ) -> None:
        """Log a human-readable evaluation summary."""
        sep = "=" * 60
        logger.info(sep)
        logger.info("EVALUATION REPORT")
        logger.info(sep)
        logger.info("Total samples : %d", len(y_true))
        logger.info("Accuracy      : %.4f", acc)
        logger.info("F1 (macro)    : %.4f", f1_macro)
        logger.info("F1 (weighted) : %.4f", f1_weighted)
        logger.info("-" * 60)
        logger.info("Per-class metrics:")
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, digits=4
        )
        for line in report.split("\n"):
            logger.info(line)
