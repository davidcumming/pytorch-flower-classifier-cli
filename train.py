import argparse
import time
import torch
from torch import nn, optim

from utils import get_device, get_data_transforms, load_datasets, get_dataloaders
from model_utils import build_model, save_checkpoint


def _fmt_time(seconds: float):
    """
    Format a duration in seconds into a human-readable string (e.g. 1m 12.3s).
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h:
        return f"{h}h {m}m {s:.1f}s"
    if m:
        return f"{m}m {s:.1f}s"
    return f"{s:.1f}s"


def train_one_model(args):
    """
    Train a model on the given dataset directory, evaluate on the test set,
    and save a checkpoint using the chosen architecture and hyperparameters.
    """
    device = get_device(args.gpu)

    data_transforms = get_data_transforms()
    image_datasets = load_datasets(args.data_dir, data_transforms)

    dataloaders = get_dataloaders(
        image_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    num_classes = len(image_datasets["train"].classes)

    model = build_model(
        arch = args.arch,
        num_classes = num_classes,
        hidden_1 = args.hidden_1,
        hidden_2 = args.hidden_2,
        drop_p = args.drop_p,
    )

    model.class_to_idx = image_datasets["train"].class_to_idx
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if args.arch == "vgg16":
        params = model.classifier.parameters()
    elif args.arch == "resnet18":
        params = model.fc.parameters()
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    optimizer = optim.Adam(params, lr=args.learning_rate)

    # Nice header
    print("\n" + "=" * 72)
    print("Training configuration")
    print("-" * 72)
    print(f"Device        : {device}")
    print(f"Arch          : {args.arch}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs}")
    print(f"LR            : {args.learning_rate}")
    print(f"Hidden units  : {args.hidden_1} -> {args.hidden_2}")
    print(f"Dropout       : {args.drop_p}")
    print(f"Print every   : {args.print_every} steps")
    print(f"num_workers   : {args.num_workers}")
    print(f"pin_memory    : {args.pin_memory}")
    print("=" * 72 + "\n")

    steps = 0
    running_loss = 0.0

    epoch_times = []
    t0_total = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        t0_epoch = time.perf_counter()

        for inputs, labels in dataloaders["train"]:
            steps += 1

            # non_blocking=True helps when pin_memory=True and using CUDA
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

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
                        valid_inputs = valid_inputs.to(device, non_blocking=True)
                        valid_labels = valid_labels.to(device, non_blocking=True)

                        valid_outputs = model(valid_inputs)
                        valid_loss = criterion(valid_outputs, valid_labels)

                        valid_total_loss += valid_loss.item() * valid_labels.size(0)
                        valid_total += valid_labels.size(0)

                        _, preds = torch.max(valid_outputs, dim=1)
                        valid_correct += (preds == valid_labels).sum().item()

                train_loss_avg = running_loss / args.print_every
                valid_loss_avg = valid_total_loss / valid_total
                valid_accuracy = valid_correct / valid_total

                # One clean block per report
                print(f"[Epoch {epoch+1:>2}/{args.epochs}] "
                      f"step {steps:<6} "
                      f"train_loss {train_loss_avg:>6.3f} | "
                      f"val_loss {valid_loss_avg:>6.3f} | "
                      f"val_acc {valid_accuracy:>6.3f}")

                running_loss = 0.0
                model.train()

        # End-of-epoch timing (sync GPU so time includes actual compute)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_sec = time.perf_counter() - t0_epoch
        epoch_times.append(epoch_sec)
        print(f"Epoch {epoch+1} finished in {_fmt_time(epoch_sec)}\n")

    # Total timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_sec = time.perf_counter() - t0_total
    print("Training complete.")
    print("\n" + "-" * 72)
    print("Timing summary")
    print("-" * 72)
    if epoch_times:
        avg_epoch = sum(epoch_times) / len(epoch_times)
        print("Epoch times   :", ", ".join(_fmt_time(t) for t in epoch_times))
        print("Avg / epoch   :", _fmt_time(avg_epoch))
    print("Total time    :", _fmt_time(total_sec))
    print("-" * 72 + "\n")

    # Test
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for t_inputs, t_labels in dataloaders["test"]:
            t_inputs = t_inputs.to(device, non_blocking=True)
            t_labels = t_labels.to(device, non_blocking=True)

            t_outputs = model(t_inputs)
            _, t_preds = torch.max(t_outputs, 1)

            test_correct += (t_preds == t_labels).sum().item()
            test_total += t_labels.size(0)

    test_acc = test_correct / test_total
    print(f"Test accuracy: {test_acc:.3f}")

    # Checkpoint path logic
    checkpoint_path = args.save_dir
    if checkpoint_path.endswith("/"):
        checkpoint_path = checkpoint_path + "checkpoint.pth"
    elif checkpoint_path.lower().endswith(".pth") is False:
        checkpoint_path = checkpoint_path + "/checkpoint.pth"

    save_checkpoint(
        filepath=checkpoint_path,
        model=model,
        optimizer=optimizer,
        arch=args.arch,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        drop_p=args.drop_p,
        lr=args.learning_rate,
        epochs=args.epochs,
        num_classes=num_classes,
    )

    print(f"Checkpoint saved to: {checkpoint_path}")


def parse_args():
    """
    Parse command-line arguments for configuring training.
    """
    parser = argparse.ArgumentParser(description="Train an image classifier and save a checkpoint.")
    parser.add_argument("data_dir", type=str, help="Path to dataset directory (with train/valid/test).")

    parser.add_argument("--save_dir", type=str, default="./", help="Directory or filepath to save checkpoint.")
    parser.add_argument("--arch", type=str, default="vgg16", help='Model architecture: "vgg16" or "resnet18".')

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_1", type=int, default=512)
    parser.add_argument("--hidden_2", type=int, default=256)
    parser.add_argument("--drop_p", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=50)

    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader worker processes (try 0, 2, 4, 8)")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Pin CPU memory (recommended when using GPU)")

    parser.add_argument("--gpu", action="store_true", help="Use GPU (CUDA) if available.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_one_model(args)