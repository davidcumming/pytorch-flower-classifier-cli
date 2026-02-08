import argparse

import torch
import torch.nn.functional as F

from utils import get_device, process_image, load_category_names
from model_utils import load_checkpoint


def predict(image_path: str, model, device: torch.device, topk: int = 5):

    model = model.to(device)
    model.eval()

    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)                 # logits
        probabilities = F.softmax(outputs, dim=1)           # probabilities
        top_probabilities, top_indices = probabilities.topk(topk, dim=1)

    top_probabilities = top_probabilities[0].cpu().tolist()
    top_indices = top_indices[0].cpu().tolist()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_indices]

    return top_probabilities, top_classes


def parse_args():
    parser = argparse.ArgumentParser(description = "Predict flower class from an image using a trained checkpoint.")
    parser.add_argument("image_path", type = str, help = "Path to image file.")
    parser.add_argument("checkpoint", type= str, help = "Path to checkpoint (.pth).")

    parser.add_argument("--top_k", type = int, default=5, help = "Return top K most likely classes.")
    parser.add_argument("--category_names", type = str, default = None, help = "Path to JSON mapping category to name.")
    parser.add_argument("--gpu", action = "store_true", help = "Use GPU (CUDA) if available.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = get_device(args.gpu)

    model, _ = load_checkpoint(args.checkpoint, device=device)

    probabilities, classes = predict(args.image_path, model, device=device, topk=args.top_k)

    if args.category_names:
        cat_to_name = load_category_names(args.category_names)
        names = [cat_to_name.get(c, c) for c in classes]
        print("Top predictions:")
        for p, name, cls in zip(probabilities, names, classes):
            print(f"{name} (class {cls}) : {p:.4f}")
    else:
        print("Probabilities:", probabilities)
        print("Classes:", classes)