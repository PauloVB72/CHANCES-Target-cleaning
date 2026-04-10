# =============================================================================
# Galaxy Classifier — CHANCES Project
# Copyright (c) 2025 CHANCES Collaboration
# License: MIT (see LICENSE file)
# =============================================================================
"""
src/data_preparation.py — Dataset scanning, splitting and manifest utilities.

Provides functions to:
  - Build image manifests from labeled folder structures.
  - Build unlabeled manifests for custom inference.
  - Perform stratified train/test splits.
  - Save and load dataset CSVs.
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid image extensions recognised by the pipeline
# ---------------------------------------------------------------------------
VALID_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------

def build_dataset_manifest(path_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Scan labeled directories and create a training manifest.

    The function assigns an integer label (0, 1, 2, …) to each class in the
    order the keys appear in *path_mapping*, and adds one-hot columns
    ``class_0``, ``class_1``, … for compatibility with Zoobot's DataModule.

    Args:
        path_mapping: ``{class_name: folder_path}`` mapping. Order matters —
            the first key receives label 0, the second label 1, etc.

    Returns:
        DataFrame with columns: ``id_str``, ``file_loc``, ``filename``,
        ``label``, ``class_0`` … ``class_N``.  Returns an empty DataFrame
        (with no rows) if no valid images were found.

    Example:
        >>> manifest = build_dataset_manifest({
        ...     "galaxies": "/data/galaxies",
        ...     "stars":    "/data/stars",
        ... })
    """
    records = []

    for label_idx, (class_name, target_dir) in enumerate(path_mapping.items()):
        if not os.path.exists(target_dir):
            logger.warning(
                "Directory not found: %s  (skipping class '%s')",
                target_dir, class_name,
            )
            continue

        logger.info("Scanning class '%s' → %s", class_name, target_dir)

        for fname in os.listdir(target_dir):
            if fname.lower().endswith(VALID_EXTENSIONS):
                records.append({
                    "id_str":   os.path.splitext(fname)[0],
                    "file_loc": os.path.join(target_dir, fname),
                    "filename": fname,
                    "label":    label_idx,
                })

    if not records:
        logger.error("No images found. Check the paths in config.ini → [PATHS].")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # One-hot columns expected by some Zoobot utilities
    n_classes = len(path_mapping)
    for i in range(n_classes):
        df[f"class_{i}"] = (df["label"] == i).astype(int)

    logger.info(
        "Manifest created: %d images across %d classes.", len(df), n_classes
    )
    return df


def build_inference_manifest(
    image_folder: str,
    recursive: bool = False,
) -> pd.DataFrame:
    """
    Create an unlabeled manifest for inference over a folder of new images.

    Unlike :func:`build_dataset_manifest`, this function does **not** assign
    labels — useful when ground truth is unavailable.

    Args:
        image_folder: Path to the folder containing images.
        recursive:    If ``True``, walk all sub-directories.

    Returns:
        DataFrame with columns: ``id_str``, ``file_loc``, ``filename``.

    Raises:
        FileNotFoundError: If *image_folder* does not exist.
        ValueError:        If no valid images are found.
    """
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Folder not found: {image_folder}")

    records = []

    if recursive:
        for root, _, files in os.walk(image_folder):
            for fname in files:
                if fname.lower().endswith(VALID_EXTENSIONS):
                    records.append({
                        "id_str":   os.path.splitext(fname)[0],
                        "file_loc": os.path.join(root, fname),
                        "filename": fname,
                    })
    else:
        for fname in os.listdir(image_folder):
            if fname.lower().endswith(VALID_EXTENSIONS):
                records.append({
                    "id_str":   os.path.splitext(fname)[0],
                    "file_loc": os.path.join(image_folder, fname),
                    "filename": fname,
                })

    if not records:
        raise ValueError(f"No valid images found in: {image_folder}")

    df = pd.DataFrame(records)
    logger.info("Inference manifest: %d images found.", len(df))
    return df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def prepare_train_test_split(
    dataset_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified train/test split.

    Stratification ensures each class is proportionally represented in both
    subsets, which is especially important for imbalanced datasets.

    Args:
        dataset_df:   Full dataset DataFrame (must contain a ``label`` column).
        test_size:    Fraction of data reserved for testing (default 0.20).
        random_state: Random seed for reproducibility.

    Returns:
        ``(train_df, test_df)`` tuple.
    """
    if "label" not in dataset_df.columns:
        raise ValueError("dataset_df must contain a 'label' column.")

    train_df, test_df = train_test_split(
        dataset_df,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset_df["label"],
    )

    logger.info(
        "Split — train: %d images (%.0f%%)  |  test: %d images (%.0f%%)",
        len(train_df), 100 * (1 - test_size),
        len(test_df),  100 * test_size,
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_datasets(
    dataset_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "./",
) -> None:
    """
    Save the full dataset and its splits as CSV files.

    Files written:
      - ``complete_dataset.csv``
      - ``train_split.csv``
      - ``test_split.csv``

    Args:
        dataset_df: Complete dataset.
        train_df:   Training subset.
        test_df:    Test subset.
        output_dir: Directory where CSVs will be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_df.to_csv(os.path.join(output_dir, "complete_dataset.csv"), index=False)
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"),        index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"),          index=False)

    logger.info("Datasets saved to: %s", output_dir)
