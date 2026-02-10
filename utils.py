import json
from pathlib import Path
from PIL import Image

import torch
from torchvision import datasets, transforms


def get_device(use_gpu: bool):
    """
    Use CUDA if available (when requested), else CPU.
    """
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_data_transforms():
    """
    Standard transforms for the Oxford 102 Flowers dataset (ImageNet normalization).
    """
    return {
        "train": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }


def load_datasets(data_dir: str, data_transforms: dict):
    """
    Load train/valid/test datasets from a directory structured for ImageFolder:
      data_dir/train/<class>/*.jpg
      data_dir/valid/<class>/*.jpg
      data_dir/test/<class>/*.jpg
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    test_dir  = data_dir / "test"

    image_datasets = {
        "train": datasets.ImageFolder(str(train_dir), transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(str(valid_dir), transform=data_transforms["valid"]),
        "test":  datasets.ImageFolder(str(test_dir),  transform=data_transforms["test"]),
    }
    return image_datasets


def get_dataloaders(
    image_datasets: dict,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
) -> dict:
    """
    Build PyTorch DataLoaders.
    - pin_memory: recommended when training on GPU
    - persistent_workers: avoids worker re-spawn each epoch (only valid if num_workers > 0)
    """
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"],
            shuffle=True,
            **common,
        ),
        "valid": torch.utils.data.DataLoader(
            image_datasets["valid"],
            shuffle=False,
            **common,
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"],
            shuffle=False,
            **common,
        ),
    }


def process_image(image_path: str, add_batch_dim: bool = True):
    """
    Convert a PIL image into a normalized tensor suitable for model input.
    If add_batch_dim=True, returns shape [1, 3, 224, 224] (ready for inference).
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    if add_batch_dim:
        image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def load_category_names(json_path: str):
    """
    Load a JSON mapping of class labels to human-readable names (e.g., cat_to_name.json).
    """
    with open(json_path, "r") as f:
        return json.load(f)