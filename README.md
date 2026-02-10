# PyTorch Flower Classifier (CLI)

This repo contains my submission for the Udacity AIPND project **Create Your Own Image Classifier**.

It includes:
- **Part 1:** Jupyter notebook exploration/training (`Image Classifier Project.ipynb`)
- **Part 2:** Command-line training + prediction apps (`train.py`, `predict.py`)

## Repository contents

- `train.py` — train a model and save a checkpoint
- `predict.py` — load a checkpoint and predict the top-K classes for an image
- `model_utils.py` — model building + checkpoint save/load helpers
- `utils.py` — device helpers, transforms, datasets/dataloaders, image preprocessing
- `scripts_make_splits.py` — helper script to create `train/valid/test` splits (Oxford 102)
- `cat_to_name.json` — mapping from class index (1–102) to flower name
- `Image Classifier Project.ipynb` — project notebook (**required** for Part 1)
- `Image Classifier Project.html` — exported notebook (optional)

**Note:** Dataset files and model checkpoints are **not committed** (see `.gitignore`).

---

## Setup

### Option A: Conda (recommended)

    conda create -n aipnd python=3.11 -y
    conda activate aipnd
    pip install torch torchvision pillow matplotlib numpy scipy

### Option B: venv

    python -m venv .venv
    source .venv/bin/activate
    pip install torch torchvision pillow matplotlib numpy scipy

---

## Data

This project uses the **Oxford 102 Flowers** dataset.

The training script expects the dataset in this structure:

    assets/flower_data/
      train/1 ... train/102
      valid/1 ... valid/102
      test/1  ... test/102

If you already have the raw Oxford files under `assets/oxford102/`, you can generate the split folders using:

    python scripts_make_splits.py

---

## Train a model (Part 2)

### VGG16 (example)

    python train.py assets/flower_data \
      --gpu \
      --arch vgg16 \
      --hidden_1 512 \
      --hidden_2 256 \
      --drop_p 0.5 \
      --learning_rate 0.001 \
      --epochs 5 \
      --print_every 50 \
      --num_workers 4 \
      --pin_memory \
      --save_dir checkpoints/vgg16.pth

### ResNet18 (example)

    python train.py assets/flower_data \
      --gpu \
      --arch resnet18 \
      --hidden_1 512 \
      --hidden_2 256 \
      --drop_p 0.5 \
      --learning_rate 0.001 \
      --epochs 5 \
      --print_every 50 \
      --num_workers 4 \
      --pin_memory \
      --save_dir checkpoints/resnet18.pth

---

## Predict (Part 2)

### Predict top-5 classes (with flower names)

    python predict.py path/to/image.jpg checkpoints/vgg16.pth \
      --top_k 5 \
      --category_names cat_to_name.json \
      --gpu

### Predict top-5 classes (class indices only)

    python predict.py path/to/image.jpg checkpoints/resnet18.pth \
      --top_k 5 \
      --gpu

Output includes probabilities and (if `--category_names` is used) human-readable flower names.

---

## Quick sanity checks

1) Train for 1 epoch (fast check)

    python train.py assets/flower_data --arch vgg16 --epochs 1

2) Predict using a saved checkpoint

    python predict.py path/to/image.jpg checkpoints/vgg16.pth --top_k 5 --category_names cat_to_name.json

---

## Submission notes

- `Image Classifier Project.ipynb` is included for **Part 1** (required).
- Dataset folders under `assets/` and model checkpoints (`*.pth`) are ignored and not pushed.
- The training script supports two architectures (`vgg16`, `resnet18`) to satisfy the Part 2 rubric.