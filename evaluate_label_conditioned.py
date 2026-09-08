#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from evaluate_checkpoints import collect_real_features, covariance, feature_frechet
from unified_transport.checkpoint import load_for_inference
from unified_transport.data import FilteredVisionDataset
from unified_transport.label_conditioned import LabelConditionalSampler
from unified_transport.reference_classifier import load_reference_classifier
from unified_transport.utils import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate label-conditioned standard-noise flow.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--generated-per-class", type=int, default=256)
    parser.add_argument("--visuals-per-class", type=int, default=10)
    parser.add_argument("--real-images", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_one(args, checkpoint_path, classifier, real_features, device):
    model, _, checkpoint = load_for_inference(checkpoint_path, device)
    config = checkpoint["config"]
    if config.get("task") != "label_conditioned_generation":
        raise ValueError("checkpoint is not label-conditioned generation")
    sampler = LabelConditionalSampler(model)
    classes = config["classes"]
    image_shape = (
        int(config["image_channels"]), int(config["image_size"]), int(config["image_size"])
    )
    set_seed(args.seed)
    generated_parts = []
    targets_parts = []
    class_metrics = {}
    for label in classes:
        labels = torch.full(
            (args.generated_per_class,), label, dtype=torch.long, device=device
        )
        generated = sampler.integrate(
            torch.randn(args.generated_per_class, *image_shape, device=device),
            labels, args.steps, args.method,
        ).clamp(-1, 1)
        logits = classifier(generated)
        probabilities = logits.softmax(1)
        predictions = logits.argmax(1)
        target = torch.full_like(predictions, label)
        generated_features = classifier(generated, return_features=True).cpu()
        real = real_features[label]
        generated_trace = float(torch.trace(covariance(generated_features)))
        real_trace = float(torch.trace(covariance(real)))
        class_metrics[str(label)] = {
            "accuracy": float((predictions == target).float().mean()),
            "target_confidence": float(probabilities[:, label].mean()),
            "feature_frechet": feature_frechet(real, generated_features),
            "feature_diversity_ratio": generated_trace / max(real_trace, 1e-12),
        }
        generated_parts.append(generated.cpu())
        targets_parts.append(target.cpu())
    generated = torch.cat(generated_parts)
    targets = torch.cat(targets_parts).to(device)
    logits = classifier(generated.to(device))
    result = {
        "step": int(checkpoint["step"]),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "generated_accuracy": float((logits.argmax(1) == targets).float().mean()),
        "generated_target_confidence": float(logits.softmax(1).gather(1, targets[:, None]).mean()),
        "mean_class_feature_frechet": sum(item["feature_frechet"] for item in class_metrics.values()) / len(class_metrics),
        "mean_feature_diversity_ratio": sum(item["feature_diversity_ratio"] for item in class_metrics.values()) / len(class_metrics),
        "classes": class_metrics,
        "ode_steps": args.steps,
        "generated_per_class": args.generated_per_class,
    }
    directory = Path(args.output) / f"step-{checkpoint['step']:07d}"
    directory.mkdir(parents=True, exist_ok=True)
    indices = []
    for class_index in range(len(classes)):
        offset = class_index * args.generated_per_class
        indices.extend(range(offset, offset + args.visuals_per_class))
    save_image(generated[indices].add(1).div(2), directory / "generated-grid.png", nrow=args.visuals_per_class)
    set_seed(args.seed)
    path_labels = torch.arange(len(classes), device=device)
    paths = sampler.integrate(
        torch.randn(len(classes), *image_shape, device=device),
        path_labels, args.steps, args.method, return_path=True,
    ).cpu()
    path_indices = torch.linspace(0, args.steps, 8).round().long().unique()
    save_image(paths[:, path_indices].reshape(-1, *image_shape).clamp(-1, 1).add(1).div(2), directory / "ode-path-grid.png", nrow=len(path_indices))
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plot_results(results, output):
    results = sorted(results, key=lambda item: item["step"])
    steps = [item["step"] for item in results]
    fields = [
        ("generated_accuracy", "Generated target accuracy"),
        ("generated_target_confidence", "Target confidence"),
        ("mean_class_feature_frechet", "Feature Frechet (lower is better)"),
        ("mean_feature_diversity_ratio", "Feature diversity ratio"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (field, title) in zip(axes.flat, fields):
        axis.plot(steps, [item[field] for item in results], marker="o")
        axis.set(xlabel="training step", title=title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "metric-curves.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.6), squeeze=False)
    for axis, result in zip(axes[0], results):
        image = np.asarray(Image.open(output / f"step-{result['step']:07d}" / "generated-grid.png"))
        axis.imshow(image)
        axis.set_title(f"Step {result['step']:,}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Label-conditioned standard Gaussian flow")
    figure.tight_layout()
    figure.savefig(output / "checkpoint-generation-comparison.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    classifier, classifier_checkpoint = load_reference_classifier(args.classifier, device)
    first = torch.load(args.checkpoints[0], map_location="cpu")
    config = first["config"]
    dataset = FilteredVisionDataset(config["mode"], args.data_root, False, config["image_size"], config["classes"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    real_features = collect_real_features(classifier, loader, config["classes"], device, args.real_images)
    results = []
    for path in sorted(args.checkpoints, key=lambda item: int(torch.load(item, map_location="cpu")["step"])):
        result = evaluate_one(args, path, classifier, real_features, device)
        results.append(result)
        print(json.dumps(result))
    (output / "metrics.json").write_text(json.dumps({"reference_classifier": {"path": str(Path(args.classifier).resolve()), "full_test_accuracy": classifier_checkpoint.get("accuracy")}, "results": results}, indent=2) + "\n")
    plot_results(results, output)


if __name__ == "__main__":
    main()
