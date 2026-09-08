#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from tqdm.auto import tqdm

from unified_transport.data import FilteredVisionDataset, image_transform
from unified_transport.reference_classifier import (
    CIFAR10ReferenceClassifier,
    ImageNetResNet18Classifier,
    MNISTReferenceClassifier,
    save_reference_classifier,
)
from unified_transport.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an independent MNIST/CIFAR-10 evaluation classifier."
    )
    parser.add_argument("--mode", choices=("mnist", "cifar10", "tinyimagenet", "stl10"), default="mnist")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument(
        "--architecture", choices=("auto", "small_resnet", "resnet18"), default="auto"
    )
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device):
    correct = 0
    total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(1) == labels).sum())
        total += labels.numel()
    return loss_sum / total, correct / total


def plot_history(history, path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["test_loss"], marker="o", label="test")
    axes[0].set(xlabel="epoch", ylabel="cross entropy", title="Classifier loss")
    axes[0].legend()
    axes[1].plot(epochs, history["test_accuracy"], marker="o")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Test accuracy", ylim=(0.0, 1.0))
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    classes = [int(item.strip()) for item in args.classes.split(",") if item.strip()]
    if len(classes) != 10 or classes != list(range(10)):
        raise ValueError("the reference classifier currently expects local labels 0..9")
    image_size = args.image_size or {
        "mnist": 28, "cifar10": 32, "tinyimagenet": 64, "stl10": 96
    }[args.mode]
    if args.mode == "mnist":
        train_transform = image_transform(28, train=False, channels=1)
        test_transform = train_transform
        dataset_class = MNIST
        model = MNISTReferenceClassifier().to(device)
    elif args.mode in {"tinyimagenet", "stl10"}:
        train_set = FilteredVisionDataset(
            args.mode, args.data_root, True, image_size, classes, download=args.download,
            random_crop=True,
        )
        test_set = FilteredVisionDataset(
            args.mode, args.data_root, False, image_size, classes, download=args.download,
            random_crop=False,
        )
        architecture = "resnet18" if args.architecture == "auto" and args.mode == "stl10" else args.architecture
        if architecture == "auto":
            architecture = "small_resnet"
        model = (
            ImageNetResNet18Classifier(pretrained=True)
            if architecture == "resnet18"
            else CIFAR10ReferenceClassifier()
        ).to(device)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
            pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )
        test_transform = image_transform(32, train=False, channels=3)
        dataset_class = CIFAR10
        model = CIFAR10ReferenceClassifier().to(device)
    if args.mode not in {"tinyimagenet", "stl10"}:
        train_set = dataset_class(
            args.data_root, train=True, transform=train_transform, download=args.download
        )
        test_set = dataset_class(
            args.data_root, train=False, transform=test_transform, download=args.download
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
    )
    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}
    best_accuracy = -1.0
    weights_path = output / f"{args.mode}-classifier-best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        total = 0
        progress = tqdm(train_loader, desc=f"classifier epoch {epoch}/{args.epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * labels.numel()
            total += labels.numel()
            progress.set_postfix(loss=f"{loss_sum / total:.4f}")
        test_loss, test_accuracy = evaluate(model.eval(), test_loader, device)
        history["train_loss"].append(loss_sum / total)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        print(json.dumps({"epoch": epoch, "test_loss": test_loss, "test_accuracy": test_accuracy}))
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_reference_classifier(
                str(weights_path), model, epoch, test_accuracy, history, args.mode
            )
        scheduler.step()

    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    plot_history(history, output / "training-curves.png")
    print(json.dumps({"weights": str(weights_path.resolve()), "best_accuracy": best_accuracy}))


if __name__ == "__main__":
    main()
