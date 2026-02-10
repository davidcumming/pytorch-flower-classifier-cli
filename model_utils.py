from pathlib import Path
import torch
from torch import nn, optim
from torchvision.models import vgg16, VGG16_Weights, resnet18, ResNet18_Weights

def build_model(arch: str, num_classes: int, hidden_1: int=512, hidden_2: int=256, drop_p: float=0.5):
    """
    Build a pretrained model backbone + new classifier head for the requested architecture.
    Supported: vgg16, resnet18.
    """
    if arch == "vgg16":
        return build_model_vgg16(num_classes=num_classes, hidden_1=hidden_1, hidden_2=hidden_2, drop_p=drop_p)
    elif arch == "resnet18":
        return build_model_resnet18(num_classes=num_classes, hidden_1=hidden_1, hidden_2=hidden_2, drop_p=drop_p)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

def build_model_vgg16(num_classes: int, hidden_1: int = 512, hidden_2: int = 256, drop_p: float = 0.5):
    """
    Load pretrained VGG16, freeze feature parameters, and replace the classifier
    with a new feedforward head for num_classes outputs.
    """
    model = vgg16(weights=VGG16_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.classifier[0].in_features

    classifier = nn.Sequential(
        nn.Linear(n_inputs, hidden_1),
        nn.ReLU(),
        nn.Dropout(drop_p),
        nn.Linear(hidden_1, hidden_2),
        nn.ReLU(),
        nn.Dropout(drop_p),
        nn.Linear(hidden_2, num_classes),
    )

    model.classifier = classifier
    return model

def build_model_resnet18(num_classes: int, hidden_1: int = 512, hidden_2: int = 256, drop_p: float = 0.5):
    """
    Load pretrained ResNet18, freeze backbone parameters, and replace model.fc
    with a new classifier head for num_classes outputs.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features

    classifier = nn.Sequential(
        nn.Linear(n_inputs, hidden_1),
        nn.ReLU(),
        nn.Dropout(drop_p),
        nn.Linear(hidden_1, hidden_2),
        nn.ReLU(),
        nn.Dropout(drop_p),
        nn.Linear(hidden_2, num_classes),
    )

    model.fc = classifier
    return model

def save_checkpoint(
    filepath: str,
    model,
    optimizer,
    arch: str,
    hidden_1: int,
    hidden_2: int,
    drop_p: float,
    lr: float,
    epochs: int,
    num_classes: int,
):
    """
    Save a training checkpoint containing model weights, class_to_idx, and
    hyperparameters needed to rebuild the model. Creates the parent directory
    for filepath if it does not exist.
    """
    checkpoint = {
        "arch": arch,
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx,
        "hidden_1": hidden_1,
        "hidden_2": hidden_2,
        "drop_p": drop_p,
        "lr": lr,
        "epochs": epochs,
        "num_classes": num_classes,
        "optimizer_state": optimizer.state_dict(),
    }
    # Ensure the parent directory exists
    parent = Path(filepath).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, device: torch.device):
    """
    Load a saved checkpoint, rebuild the correct model architecture, restore weights,
    and restore optimizer state. Returns (model, optimizer).
    """
    checkpoint = torch.load(filepath, map_location=device)

    arch = checkpoint["arch"]

    model = build_model(
        arch=arch,
        num_classes=checkpoint["num_classes"],
        hidden_1=checkpoint["hidden_1"],
        hidden_2=checkpoint["hidden_2"],
        drop_p=checkpoint["drop_p"],
    )

    model.class_to_idx = checkpoint["class_to_idx"]

    model = model.to(device)
    model.load_state_dict(checkpoint["state_dict"])

    if arch == "vgg16":
        params = model.classifier.parameters()
    elif arch == "resnet18":
        params = model.fc.parameters()
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    optimizer = optim.Adam(params, lr=checkpoint["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    return model, optimizer