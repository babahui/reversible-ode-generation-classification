#!/usr/bin/env python3
import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from unified_transport.checkpoint import load_for_inference
from unified_transport.data import FilteredVisionDataset
from unified_transport.interpolant import TransportSampler
from unified_transport.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Bayes classification accuracy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--cycle-batches", type=int, default=10, help="0 disables image-noise-image MSE"
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, latent, checkpoint = load_for_inference(args.checkpoint, device)
    config = checkpoint["config"]
    if latent is None:
        raise ValueError(
            "classification requires an MNIST/CIFAR image-noise checkpoint"
        )
    class_values = config.get("classes", list(range(config["num_classes"])))
    dataset = FilteredVisionDataset(
        mode=config["mode"],
        root=args.data_root,
        train=False,
        image_size=config["image_size"],
        classes=class_values,
        download=args.download,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    sampler = TransportSampler(model)
    correct = 0
    total = 0
    negative_log_posterior = 0.0
    cycle_squared_error = 0.0
    cycle_elements = 0
    for batch_index, (images, labels) in enumerate(tqdm(loader)):
        if args.max_batches and batch_index >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        encoded = sampler.integrate(
            images, source=0, target=1, steps=args.steps, method=args.method
        )
        log_posterior = latent.log_posterior(encoded).log_softmax(dim=1)
        predictions = log_posterior.argmax(dim=1)
        correct += int((predictions == labels).sum())
        negative_log_posterior -= float(
            log_posterior.gather(1, labels[:, None]).sum()
        )
        total += labels.numel()
        if args.cycle_batches and batch_index < args.cycle_batches:
            reconstructed = sampler.integrate(
                encoded, source=1, target=0, steps=args.steps, method=args.method
            )
            cycle_squared_error += float((reconstructed - images).square().sum())
            cycle_elements += images.numel()

    result = {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "posterior_nll": negative_log_posterior / total,
        "cycle_mse": cycle_squared_error / cycle_elements if cycle_elements else None,
        "class_values": class_values,
        "strict_markov_inverse": not config["use_history"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
