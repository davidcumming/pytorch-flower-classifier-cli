import argparse
import torch
from torch import nn, optim
from utils import get_device, get_data_transforms, load_datasets, get_dataloaders
from model_utils import build_model_vgg16, save_checkpoint


def train_one_model(args):
    device = get_device(args.gpu)

    data_transforms = get_data_transforms()
    image_datasets = load_datasets(args.data_dir, data_transforms)
    dataloaders = get_dataloaders(image_datasets, batch_size=args.batch_size, num_workers = 0)

    num_classes = len(image_datasets["train"].classes)

    if args.arch != "vgg16":
        raise ValueError("This version supports only --arch vgg16 to match your notebook.")

    model = build_model_vgg16(
        num_classes = num_classes,
        hidden_1 = args.hidden_1,
        hidden_2 = args.hidden_2,
        drop_p = args.drop_p,
    )

    model.class_to_idx = image_datasets["train"].class_to_idx
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    steps = 0
    running_loss = 0.0

    for epoch in range(args.epochs):
        model.train()

        for inputs, labels in dataloaders["train"]:
            steps += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % args.print_every == 0:
                model.eval()

                valid_total_loss = 0.0
                valid_correct = 0
                valid_total = 0

                with torch.no_grad():
                    for valid_inputs, valid_labels in dataloaders["valid"]:
                        valid_inputs = valid_inputs.to(device)
                        valid_labels = valid_labels.to(device)

                        valid_outputs = model(valid_inputs)
                        valid_loss = criterion(valid_outputs, valid_labels)

                        valid_total_loss += valid_loss.item() * valid_labels.size(0)
                        valid_total += valid_labels.size(0)

                        _, preds = torch.max(valid_outputs, dim = 1)
                        valid_correct += (preds == valid_labels).sum().item()

                train_loss_avg = running_loss / args.print_every
                valid_loss_avg = valid_total_loss / valid_total
                valid_accuracy = valid_correct / valid_total

                print(f"Epoch {epoch+1}/{args.epochs}.. ")
                print(f"Training loss: {train_loss_avg:.3f}.. ")
                print(f"Validation loss: {valid_loss_avg:.3f}.. ")
                print(f"Validation accuracy: {valid_accuracy:.3f}")

                running_loss = 0.0
                model.train()

    print("Training complete.")

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for t_inputs, t_labels in dataloaders["test"]:
            t_inputs = t_inputs.to(device)
            t_labels = t_labels.to(device)

            t_outputs = model(t_inputs)
            _, t_preds = torch.max(t_outputs, 1)

            test_correct += (t_preds == t_labels).sum().item()
            test_total += t_labels.size(0)

    test_acc = test_correct / test_total
    print(f"Test accuracy: {test_acc:.3f}")

    checkpoint_path = args.save_dir
    # If user passes a directory, save checkpoint.pth in it; if they pass a filename, use it.
    if checkpoint_path.endswith("/"):
        checkpoint_path = checkpoint_path + "checkpoint.pth"
    elif checkpoint_path.lower().endswith(".pth") is False:
        checkpoint_path = checkpoint_path + "/checkpoint.pth"

    save_checkpoint(
        filepath = checkpoint_path,
        model = model,
        optimizer = optimizer,
        arch = args.arch,
        hidden_1 = args.hidden_1,
        hidden_2 = args.hidden_2,
        drop_p = args.drop_p,
        lr = args.learning_rate,
        epochs = args.epochs,
        num_classes = num_classes,
    )

    print(f"Checkpoint saved to: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image classifier and save a checkpoint.")
    parser.add_argument("data_dir", type=str, help="Path to dataset directory (with train/valid/test).")

    parser.add_argument("--save_dir", type=str, default="./", help="Directory or filepath to save checkpoint.")
    parser.add_argument("--arch", type=str, default="vgg16", help='Model architecture (only "vgg16" supported here).')

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_1", type=int, default=512)
    parser.add_argument("--hidden_2", type=int, default=256)
    parser.add_argument("--drop_p", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=50)

    parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of DataLoader worker processes')
    parser.add_argument('--pin_memory', action='store_true',
                    help='Pin CPU memory (recommended when using GPU)')

    parser.add_argument("--gpu", action="store_true", help="Use GPU (CUDA) if available.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_one_model(args)