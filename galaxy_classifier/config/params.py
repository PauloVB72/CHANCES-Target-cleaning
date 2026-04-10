# =============================================================================
# Galaxy Classifier — CHANCES Project
# Copyright (c) 2025 CHANCES Collaboration
# License: MIT (see LICENSE file)
# =============================================================================
"""
config/params.py — Configuration loader from INI file.

Parses config.ini and exposes typed dataclasses for each section.
"""

import configparser
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Dataclasses — one per INI section
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    """[PATHS] + [CLASSES] sections."""
    source_paths: Dict[str, str]          # {class_name: folder_path}
    class_names: List[str]                # ordered class name list


@dataclass
class TrainingConfig:
    """[TRAINING] section."""
    experiment_dir: str
    num_classes: int
    img_size: int
    greyscale: bool
    epochs: int
    batch_size: int
    accelerator: str
    patience: int
    devices: str
    test_size: float
    random_state: int


@dataclass
class InferenceConfig:
    """[INFERENCE] section."""
    checkpoint_path: Optional[str]
    output_name: str


@dataclass
class CustomInferenceConfig:
    """[CUSTOM_INFERENCE] section."""
    output_dir: str
    output_prefix: str
    recursive: bool
    batch_size: int
    device: str


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config_from_ini(ini_path: str) -> dict:
    """
    Parses a config.ini file and returns a dict of typed config objects.

    Args:
        ini_path: Path to the .ini configuration file.

    Returns:
        dict with keys: 'paths', 'training', 'inference', 'custom_inference'.

    Raises:
        FileNotFoundError: If the INI file does not exist.
        ValueError: If required keys are missing or values are malformed.
    """
    if not os.path.exists(ini_path):
        raise FileNotFoundError(f"Config file not found: {ini_path}")

    cfg = configparser.ConfigParser()
    cfg.read(ini_path)

    # -----------------------------------------------------------------------
    # [CLASSES] — parse class names
    # -----------------------------------------------------------------------
    raw_names = cfg.get("CLASSES", "names", fallback="")
    class_names = [n.strip() for n in raw_names.split(",") if n.strip()]
    if not class_names:
        raise ValueError("[CLASSES] names is empty or missing in config.ini")

    # -----------------------------------------------------------------------
    # [PATHS] — one entry per class, same keys as class_names
    # -----------------------------------------------------------------------
    source_paths: Dict[str, str] = {}
    for name in class_names:
        key = name.lower()
        if key not in cfg["PATHS"]:
            raise ValueError(
                f"[PATHS] is missing a key for class '{name}'. "
                f"Add: {key} = /your/path/to/{name}"
            )
        source_paths[name] = cfg.get("PATHS", key).strip()

    paths_config = PathsConfig(
        source_paths=source_paths,
        class_names=class_names,
    )

    # -----------------------------------------------------------------------
    # [TRAINING]
    # -----------------------------------------------------------------------
    t = cfg["TRAINING"]
    training_config = TrainingConfig(
        experiment_dir=t.get("experiment_dir", "./experiment_results/").strip(),
        num_classes=t.getint("num_classes", len(class_names)),
        img_size=t.getint("img_size", 256),
        greyscale=t.getboolean("greyscale", False),
        epochs=t.getint("epochs", 30),
        batch_size=t.getint("batch_size", 32),
        accelerator=t.get("accelerator", "auto").strip(),
        patience=t.getint("patience", 10),
        devices=t.get("devices", "auto").strip(),
        test_size=t.getfloat("test_size", 0.2),
        random_state=t.getint("random_state", 42),
    )

    # -----------------------------------------------------------------------
    # [INFERENCE]
    # -----------------------------------------------------------------------
    i = cfg["INFERENCE"]
    ckpt = i.get("checkpoint_path", "").strip()
    inference_config = InferenceConfig(
        checkpoint_path=ckpt if ckpt else None,
        output_name=i.get("output_name", "predictions.csv").strip(),
    )

    # -----------------------------------------------------------------------
    # [CUSTOM_INFERENCE]
    # -----------------------------------------------------------------------
    ci = cfg["CUSTOM_INFERENCE"] if "CUSTOM_INFERENCE" in cfg else {}
    custom_config = CustomInferenceConfig(
        output_dir=ci.get("output_dir", "").strip(),
        output_prefix=ci.get("output_prefix", "custom_predictions").strip(),
        recursive=cfg.getboolean("CUSTOM_INFERENCE", "recursive", fallback=False),
        batch_size=int(ci.get("batch_size", 32)),
        device=ci.get("device", "auto").strip(),
    )

    return {
        "paths": paths_config,
        "training": training_config,
        "inference": inference_config,
        "custom_inference": custom_config,
    }
