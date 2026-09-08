#!/usr/bin/env python3
"""Evaluate image->Gaussian classification for a generation-only flow."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from unified_transport.checkpoint import load_for_inference
from unified_transport.data import FilteredVisionDataset
from unified_transport.generation_only import GenerationOnlySampler
from unified_transport.reference_classifier import load_reference_classifier
from unified_transport.utils import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classifier", default="")
    parser.add_argument("--steps", nargs="+", type=int, default=[15, 30, 60, 120])
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-test-images", type=int, default=1000)
    parser.add_argument("--generated-per-class", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def confusion_image(matrix, destination, title):
    normalized = matrix.float() / matrix.sum(1, keepdim=True).clamp_min(1)
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(normalized.numpy(), cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(matrix.shape[0]), range(matrix.shape[0]))
    axis.set_yticks(range(matrix.shape[0]), range(matrix.shape[0]))
    axis.set(xlabel="predicted Gaussian component", ylabel="true image class", title=title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{normalized[row, column]:.2f}",
                      ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)


@torch.inference_mode()
def evaluate_step(args, model, latent, classifier, loader, device, steps):
    sampler = GenerationOnlySampler(model)
    classes = list(range(latent.num_classes))
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.long)
    correct = total = 0
    nll_sum = confidence_sum = cycle_sum = 0.0
    for images, labels in loader:
        if total >= args.max_test_images:
            break
        take = min(images.shape[0], args.max_test_images - total)
        images = images[:take].to(device, non_blocking=True)
        labels = labels[:take].to(device, non_blocking=True)
        encoded = sampler.integrate_interval(images, 1.0, 0.0, steps, args.method)
        log_posterior = latent.log_posterior(encoded).log_softmax(1)
        predictions = log_posterior.argmax(1)
        correct += int((predictions == labels).sum())
        total += take
        bins = torch.bincount(
            labels.cpu() * len(classes) + predictions.cpu(),
            minlength=len(classes) ** 2,
        )
        confusion += bins.reshape(len(classes), len(classes))
        nll_sum -= float(log_posterior.gather(1, labels[:, None]).sum())
        confidence_sum += float(log_posterior.exp().max(1).values.sum())
        reconstructed = sampler.integrate_interval(
            encoded, 0.0, 1.0, steps, args.method
        )
        cycle_sum += float((reconstructed - images).square().sum())
    set_seed(args.seed)
    labels = torch.arange(latent.num_classes, device=device).repeat_interleave(
        args.generated_per_class
    )
    initial = latent.sample(labels)
    generated_raw = sampler.integrate_interval(
        initial, 0.0, 1.0, steps, args.method
    )
    generated = generated_raw.clamp(-1, 1)
    regenerated = sampler.integrate_interval(
        generated_raw, 1.0, 0.0, steps, args.method
    )
    generated_posterior = latent.log_posterior(regenerated).log_softmax(1)
    generated_reverse_predictions = generated_posterior.argmax(1)
    result = {
        "steps": steps,
        "method": args.method,
        "real_reverse_accuracy": correct / max(total, 1),
        "real_reverse_posterior_nll": nll_sum / max(total, 1),
        "real_reverse_posterior_confidence": confidence_sum / max(total, 1),
        "real_cycle_mse": cycle_sum / max(total * initial[0].numel(), 1),
        "real_test_images": total,
        "generated_roundtrip_gaussian_accuracy": float(
            (generated_reverse_predictions == labels).float().mean()
        ),
        "generated_roundtrip_mse": float((regenerated - initial).square().mean()),
        "generated_external_accuracy": None,
        "confusion_matrix": confusion.tolist(),
    }
    if classifier is not None:
        result["generated_external_accuracy"] = float(
            (classifier(generated).argmax(1) == labels).float().mean()
        )
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, latent, checkpoint = load_for_inference(args.checkpoint, device)
    config = checkpoint["config"]
    if config.get("task") != "generation_only" or latent is None:
        raise ValueError("checkpoint must be a generation-only image/Gaussian model")
    classifier = None
    if args.classifier:
        classifier, _ = load_reference_classifier(args.classifier, device)
    dataset = FilteredVisionDataset(
        config["mode"], args.data_root, False, config["image_size"], config["classes"]
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    results = []
    for steps in args.steps:
        result = evaluate_step(args, model, latent, classifier, loader, device, steps)
        results.append(result)
        confusion_image(
            torch.tensor(result["confusion_matrix"]),
            output / f"confusion-{steps:03d}.png",
            f"Gaussian-only reverse classification ({steps} {args.method} steps)",
        )
        print(json.dumps(result))
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mode": config["mode"],
        "task": config["task"],
        "classes": config["classes"],
        "results": results,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
