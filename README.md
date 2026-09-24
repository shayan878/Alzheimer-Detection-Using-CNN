# AD vs. MCI PyTorch Dataset Loader

A PyTorch `Dataset` implementation designed for loading and processing medical image data for Alzheimer's Disease (AD) and Mild Cognitive Impairment (MCI) classification tasks.

## Overview

This repository provides an optimized dataset class to handle binary medical image classification pipelines. It addresses common memory bottlenecks during training by utilizing lazy loading and applying data augmentations on the fly rather than caching transformed images in system memory.

## Key Features

- Lazy Image Loading: Reads image files from disk during batch execution rather than preloading the entire dataset into RAM, preventing memory crashes in constrained environments.
- Dynamic Data Augmentation: Applies random transformations (rotation, horizontal/vertical flips, and color jitter) dynamically per epoch to improve model generalization and mitigate overfitting.
- Native DataLoader Integration: Integrates seamlessly with `torch.utils.data.DataLoader` for batching, shuffling, and multi-worker fetching.

## Project structure

```text
Alzheimer-Detection-Using-CNN/
├── run.py
├── main.py
├── requirements.txt
├── README.md
└── src/
    ├── config.py
    ├── data/
    │   ├── dataset.py
    │   └── loaders.py
    ├── models/
    │   └── cnn_models.py
    ├── training/
    │   ├── engine.py
    │   └── metrics.py
    ├── evaluation/
    │   └── metrics.py
    ├── visualization/
    │   └── plots.py
    └── utils/
        └── seed.py
```

## Dataset Structure

The dataset implementation expects files to be organized into class-specific subdirectories under a single root directory:

```text
dataset_root/
├── AD/
│   ├── sample_01.jpg
│   ├── sample_02.jpg
│   └── ...
└── MCI/
    ├── sample_01.jpg
    ├── sample_02.jpg
    └── ...
```

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- Pillow

Dependencies can be installed via pip:

```bash
pip install torch torchvision pillow
```
or
```bash
pip install -r requirements.txt
```

## Usage

Import `CustomDataset` into your main training script:

```python
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset

# Define path to dataset
DATA_DIR = "./dataset_root"

# Instantiate train and evaluation datasets
train_dataset = CustomDataset(data_dir=DATA_DIR, is_train=True)
val_dataset = CustomDataset(data_dir=DATA_DIR, is_train=False)

# Wrap with DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Training loop
for images, labels in train_loader:
    # Model forward pass and training logic
    pass
```

## Transformation Pipeline

- Validation / Testing: Resizes images to 64x64 pixels and converts them to standard PyTorch tensors.
- Training: Applies dynamic random rotation (up to 15 degrees), horizontal flip, vertical flip, color jitter (brightness, contrast, saturation, hue), resizes to 64x64 pixels, and converts to tensors.

## Run

Update `DATA_DIR` in `main.py`, then run:

```bash
python main.py
```

## License

This project is licensed under the MIT License.
