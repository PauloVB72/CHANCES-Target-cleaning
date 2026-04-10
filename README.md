# CHANCES-Target-cleaning
Repository to make prediction on differents catalogues using Zoobot, is a easy way to use the tool.

# 🌌 Galaxy Classifier

**A fine-tuning pipeline for multi-class galaxy image classification, built on [Zoobot](https://github.com/mwalmsley/zoobot) and PyTorch Lightning.**

> Developed as part of the **CHANCES** collaboration.

---

## Overview

Galaxy Classifier lets you train a compact ConvNeXt-Tiny model — pre-trained on millions of Galaxy Zoo images — to distinguish between your own galaxy morphology classes with only a few hundred labelled examples. The pipeline covers everything from raw image folders to a fully evaluated model:

```
images/ → manifest → train/test split → fine-tuning → inference → metrics + plots
```

You can change the categories:

| Label | Class      | Description                              |
|-------|------------|------------------------------------------|
| 0     | `name`     | description of the category              |
| .     | .,,,,,,,,  | .....................                    |


Any number of classes can be configured via `config.ini`.

---

## Project Structure

```
galaxy_classifier/
├── main.py                  # Pipeline entry point
├── inference_custom.py      # Standalone inference on new images
├── config.ini               # ← edit this before running
├── requirements.txt
├── LICENSE
│
├── config/
│   ├── __init__.py
│   └── params.py            # INI → typed dataclasses
│
└── src/
    ├── __init__.py
    ├── data_preparation.py  # Manifest creation & train/test split
    ├── trainer.py           # Zoobot fine-tuning & inference
    └── evaluator.py         # Metrics, confusion matrix, plots
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CHANCES-survey/galaxy-classifier.git
cd galaxy-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install PyTorch (with CUDA if available)

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and install
the wheel matching your CUDA version, for example:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `zoobot` and `galaxy-datasets` will download pre-trained weights
> (~200 MB) from HuggingFace Hub on first use. An internet connection is
> required for the first run.

---

## Configuration

Copy `config.ini` and fill in your paths:

```ini
[PATHS]
galaxies = /data/images/galaxies
nothing  = /data/images/nothing
spurious = /data/images/spurious
offset   = /data/images/offset
stars    = /data/images/stars

[CLASSES]
names = galaxies, nothing, spurious, offset, stars

[TRAINING]
experiment_dir = ./experiment_results/
num_classes    = 5
img_size       = 256
epochs         = 30
batch_size     = 32
accelerator    = auto     # auto | cpu | gpu
patience       = 10
test_size      = 0.2
random_state   = 42
```

> **Tip:** The order of keys in `[PATHS]` must match the order of names in
> `[CLASSES]`. The first key gets label 0, the second gets label 1, etc.

---

## Usage

### Full pipeline

```bash
python main.py --config config.ini --step all
```

### Individual steps

```bash
# 1. Scan directories and create train/test CSVs
python main.py --step dataset

# 2. Fine-tune the model
python main.py --step train

# 3. Inference on the held-out test set
python main.py --step inference

# 4. Compute metrics and generate plots
python main.py --step evaluate
```

### Custom inference on new images (no labels)

```bash
# Via main.py
python main.py --step predict \
    --image_folder /path/to/new_images \
    --checkpoint   experiment_results/checkpoints/last.ckpt \
    --output       new_predictions.csv

# Via the standalone script
python inference_custom.py \
    --image_folder /path/to/new_images \
    --checkpoint   experiment_results/checkpoints/last.ckpt \
    --class_names  galaxies,nothing,spurious,offset,stars \
    --output       new_predictions.csv
```

---

## Output Files

After running the full pipeline you will find inside `experiment_results/`:

| File                    | Description                                   |
|-------------------------|-----------------------------------------------|
| `complete_dataset.csv`  | Full image manifest with labels               |
| `train_split.csv`       | Training subset                               |
| `test_split.csv`        | Test subset                                   |
| `checkpoints/`          | PyTorch Lightning checkpoints                 |
| `predictions.csv`       | Model outputs on the test set                 |
| `metrics_report.json`   | Accuracy, F1-macro, F1-weighted, per-class    |
| `confusion_matrix.png`  | Normalised confusion matrix heatmap           |
| `class_distribution.png`| Bar chart of class frequencies                |
| `logs/`                 | Timestamped run logs                          |

---

## Hardware Requirements

| Setup          | Minimum              | Recommended           |
|----------------|----------------------|-----------------------|
| RAM            | 8 GB                 | 16 GB                 |
| GPU VRAM       | 4 GB (batch 16)      | 8 GB+ (batch 32–64)   |
| Storage        | 2 GB (model + data)  | 10 GB+                |

CPU training is supported but slow (~10× slower than GPU for 256 px images).

---

## Citation

If you use this software in your research, please cite:

```bibtex
@article{walmsley2023zoobot,
  title   = {Zoobot: Adaptable Deep Learning Models for Galaxy Morphology},
  author  = {Walmsley, Mike and others},
  journal = {Journal of Open Source Software},
  volume  = {8},
  number  = {85},
  pages   = {5312},
  year    = {2023},
  doi     = {10.21105/joss.05312}
}
```

---

## License

This project is released under the [MIT License](LICENSE).  
Copyright © 2025 CHANCES Collaboration.

---

## Acknowledgements

Built on top of [Zoobot](https://github.com/mwalmsley/zoobot) by Mike Walmsley et al.
and the [galaxy-datasets](https://github.com/mwalmsley/galaxy-datasets) package.







--------------------------------------------------------------------------------------------------







# 🌌 Galaxy Classifier

**A fine-tuning pipeline for multi-class galaxy image classification, built on [Zoobot](https://github.com/mwalmsley/zoobot) and PyTorch Lightning.**

> Developed as part of the **CHANCES** (Cluster Hosting Active Nuclei and Cluster Environment Survey) collaboration.

---

## Overview

Galaxy Classifier lets you train a compact ConvNeXt-Tiny model — pre-trained on millions of Galaxy Zoo images — to distinguish between your own galaxy morphology classes with only a few hundred labelled examples. The pipeline covers everything from raw image folders to a fully evaluated model:

```
images/ → manifest → train/test split → fine-tuning → inference → metrics + plots
```

Out of the box it targets five classes:

| Label | Class      | Description                              |
|-------|------------|------------------------------------------|
| 0     | `galaxies` | Clean galaxy detections                  |
| 1     | `nothing`  | Empty / blank cutouts                    |
| 2     | `spurious` | Artefacts, diffraction spikes, etc.      |
| 3     | `offset`   | Off-centre or blended sources            |
| 4     | `stars`    | Stellar sources mis-classified as galaxy |

Any number of classes can be configured via `config.ini`.

---

## Project Structure

```
galaxy_classifier/
├── main.py                  # Pipeline entry point
├── inference_custom.py      # Standalone inference on new images
├── config.ini               # ← edit this before running
├── requirements.txt
├── LICENSE
│
├── config/
│   ├── __init__.py
│   └── params.py            # INI → typed dataclasses
│
└── src/
    ├── __init__.py
    ├── data_preparation.py  # Manifest creation & train/test split
    ├── trainer.py           # Zoobot fine-tuning & inference
    └── evaluator.py         # Metrics, confusion matrix, plots
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CHANCES-survey/galaxy-classifier.git
cd galaxy-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install PyTorch (with CUDA if available)

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and install
the wheel matching your CUDA version, for example:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `zoobot` and `galaxy-datasets` will download pre-trained weights
> (~200 MB) from HuggingFace Hub on first use. An internet connection is
> required for the first run.

---

## Configuration

Copy `config.ini` and fill in your paths:

```ini
[PATHS]
galaxies = /data/images/galaxies
nothing  = /data/images/nothing
spurious = /data/images/spurious
offset   = /data/images/offset
stars    = /data/images/stars

[CLASSES]
names = galaxies, nothing, spurious, offset, stars

[TRAINING]
experiment_dir = ./experiment_results/
num_classes    = 5
img_size       = 256
epochs         = 30
batch_size     = 32
accelerator    = auto     # auto | cpu | gpu
patience       = 10
test_size      = 0.2
random_state   = 42
```

> **Tip:** The order of keys in `[PATHS]` must match the order of names in
> `[CLASSES]`. The first key gets label 0, the second gets label 1, etc.

---

## Usage

### Full pipeline

```bash
python main.py --config config.ini --step all
```

### Individual steps

```bash
# 1. Scan directories and create train/test CSVs
python main.py --step dataset

# 2. Fine-tune the model
python main.py --step train

# 3. Inference on the held-out test set
python main.py --step inference

# 4. Compute metrics and generate plots
python main.py --step evaluate
```

### Custom inference on new images (no labels)

```bash
# Via main.py
python main.py --step predict \
    --image_folder /path/to/new_images \
    --checkpoint   experiment_results/checkpoints/last.ckpt \
    --output       new_predictions.csv

# Via the standalone script
python inference_custom.py \
    --image_folder /path/to/new_images \
    --checkpoint   experiment_results/checkpoints/last.ckpt \
    --class_names  galaxies,nothing,spurious,offset,stars \
    --output       new_predictions.csv
```

---

## Output Files

After running the full pipeline you will find inside `experiment_results/`:

| File                    | Description                                   |
|-------------------------|-----------------------------------------------|
| `complete_dataset.csv`  | Full image manifest with labels               |
| `train_split.csv`       | Training subset                               |
| `test_split.csv`        | Test subset                                   |
| `checkpoints/`          | PyTorch Lightning checkpoints                 |
| `predictions.csv`       | Model outputs on the test set                 |
| `metrics_report.json`   | Accuracy, F1-macro, F1-weighted, per-class    |
| `confusion_matrix.png`  | Normalised confusion matrix heatmap           |
| `class_distribution.png`| Bar chart of class frequencies                |
| `logs/`                 | Timestamped run logs                          |

---

## Hardware Requirements

| Setup          | Minimum              | Recommended           |
|----------------|----------------------|-----------------------|
| RAM            | 8 GB                 | 16 GB                 |
| GPU VRAM       | 4 GB (batch 16)      | 8 GB+ (batch 32–64)   |
| Storage        | 2 GB (model + data)  | 10 GB+                |

CPU training is supported but slow (~10× slower than GPU for 256 px images).

---

## Citation

If you use this software in your research, please cite:

```bibtex
@article{walmsley2023zoobot,
  title   = {Zoobot: Adaptable Deep Learning Models for Galaxy Morphology},
  author  = {Walmsley, Mike and others},
  journal = {Journal of Open Source Software},
  volume  = {8},
  number  = {85},
  pages   = {5312},
  year    = {2023},
  doi     = {10.21105/joss.05312}
}
```

---

## License

This project is released under the [MIT License](LICENSE).  
Copyright © 2025 CHANCES Collaboration.

---

## Acknowledgements

Built on top of [Zoobot](https://github.com/mwalmsley/zoobot) by Mike Walmsley et al.
and the [galaxy-datasets](https://github.com/mwalmsley/galaxy-datasets) package.

---

## 🔭 Galaxy Viewer (Standalone Exploration Tool)

`Galaxy_Viewer_3_0.py` is an **interactive Streamlit dashboard** for exploring
classifier predictions. It is **not** part of the training pipeline — run it
separately at any time to inspect results visually.

```bash
pip install streamlit plotly pillow scipy
streamlit run Galaxy_Viewer_3_0.py
```

### Tabs

| Tab | What it does |
|-----|-------------|
| 🖼️ **Image Viewer** | Browse predictions filtered by class and threshold. Each card shows the galaxy ID badge, probability bar chart, and Add / Remove / Center mark buttons. |
| 🌠 **Group Analysis** | Friends-of-friends clustering. Includes an **interactive RA/Dec scatter plot** — hover over any point to see a floating panel with the galaxy thumbnail and full probability breakdown. Click-select updates the member table and thumbnails below. |
| 📋 **Filter & Match** | Build multi-class AND/OR filters, combine with the marked table from the viewer, and export the final catalogue as CSV. |

### What's new in v3.0

- **Galaxy ID badges** shown prominently on every card and thumbnail.
- **Hover-image preview** in the Individual Explorer scatter plot: move your
  mouse over any source and a floating panel appears with the galaxy image,
  class probabilities, and source ID — no click required.
- Unified **space observatory** dark theme (deep navy + cyan + gold palette,
  DM Mono + Syne typography).
- Dark-themed probability bar charts and histograms throughout.
- Improved image cards with colour-coded mark status.

### Requirements (viewer only)

```
streamlit>=1.33
plotly>=5.18
Pillow>=10.0
scipy>=1.11
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
```
