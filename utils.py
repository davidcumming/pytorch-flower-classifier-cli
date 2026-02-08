# utils.py
import json
from PIL import Image
import torch
from torchvision import datasets, transforms


def get_device(use_gpu: bool) -> torch.device:
    """
    Matches the notebook idea: use CUDA if available (when requested), else CPU.
    """
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_data_transforms():

    data_transforms = {
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
    return data_transforms


def load_datasets(data_dir: str, data_transforms: dict):
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"
    test_dir  = f"{data_dir}/test"

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
        "test":  datasets.ImageFolder(test_dir,  transform=data_transforms["test"]),
    }
    return image_datasets


def get_dataloaders(image_datasets: dict, batch_size: int = 64, num_workers: int = 0):
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"],
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers
            ),
        "valid": torch.utils.data.DataLoader(
            image_datasets["valid"],
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers
            ),
        "test":  torch.utils.data.DataLoader(
            image_datasets["test"],
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers),
    }
    return dataloaders


def process_image(image_path: str) -> torch.Tensor:

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    return image_tensor


def load_category_names(json_path: str):

    with open(json_path, "r") as f:
        return json.load(f)