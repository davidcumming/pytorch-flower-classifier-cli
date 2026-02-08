import torch
from torch import nn, optim
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights

def build_model_vgg16(num_classes: int, hidden_1: int = 512, hidden_2: int = 256, drop_p: float = 0.5):
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
    Save everything needed to rebuild model later (and optionally keep training).
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
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, device: torch.device):
    """
    Loads checkpoint and rebuilds model + optimizer.
    Returns (model, optimizer).
    """
    checkpoint = torch.load(filepath, map_location=device)

    if checkpoint["arch"] != "vgg16":
        raise ValueError(
            f"Only vgg16 is supported here. Found arch={checkpoint['arch']}."
        )

    model = build_model_vgg16(
        num_classes=checkpoint["num_classes"],
        hidden_1=checkpoint["hidden_1"],
        hidden_2=checkpoint["hidden_2"],
        drop_p=checkpoint["drop_p"],
    )

    model.class_to_idx = checkpoint["class_to_idx"]

    model = model.to(device)
    model.load_state_dict(checkpoint["state_dict"])

    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    return model, optimizer