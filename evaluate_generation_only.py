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
from unified_transport.generation_only import GenerationOnlySampler
from unified_transport.reference_classifier import load_reference_classifier
from unified_transport.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generation-only Gaussian-to-image flow checkpoints."
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--real-images", type=int, default=10000)
    parser.add_argument("--generated-per-class", type=int, default=256)
    parser.add_argument("--visuals-per-class", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_one(args, checkpoint_path, classifier, real_features, device):
    model, latent, checkpoint = load_for_inference(checkpoint_path, device)
    config = checkpoint["config"]
    if config.get("task") != "generation_only":
        raise ValueError("checkpoint is not generation-only")
    sampler = GenerationOnlySampler(model)
    class_values = config["classes"]
    set_seed(args.seed)
    generated_parts = []
    generated_targets = []
    class_metrics = {}
    for local_index, original_label in enumerate(class_values):
        labels = torch.full(
            (args.generated_per_class,), local_index, device=device, dtype=torch.long
        )
        generated = sampler.integrate(
            latent.sample(labels), args.steps, args.method
        ).clamp(-1, 1)
        logits = classifier(generated)
        probabilities = logits.softmax(1)
        predictions = logits.argmax(1)
        targets = torch.full_like(predictions, original_label)
        generated_features = classifier(generated, return_features=True).cpu()
        real_class_features = real_features[original_label]
        generated_trace = float(torch.trace(covariance(generated_features)))
        real_trace = float(torch.trace(covariance(real_class_features)))
        class_metrics[str(original_label)] = {
            "accuracy": float((predictions == targets).float().mean()),
            "target_confidence": float(
                probabilities[:, original_label].mean()
            ),
            "feature_frechet": feature_frechet(
                real_class_features, generated_features
            ),
            "feature_diversity_ratio": generated_trace / max(real_trace, 1e-12),
        }
        generated_parts.append(generated.cpu())
        generated_targets.append(targets.cpu())

    generated = torch.cat(generated_parts)
    targets = torch.cat(generated_targets).to(device)
    logits = classifier(generated.to(device))
    result = {
        "step": int(checkpoint["step"]),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "generated_accuracy": float((logits.argmax(1) == targets).float().mean()),
        "generated_target_confidence": float(
            logits.softmax(1).gather(1, targets[:, None]).mean()
        ),
        "mean_class_feature_frechet": sum(
            item["feature_frechet"] for item in class_metrics.values()
        ) / len(class_metrics),
        "mean_feature_diversity_ratio": sum(
            item["feature_diversity_ratio"] for item in class_metrics.values()
        ) / len(class_metrics),
        "classes": class_metrics,
        "ode_steps": args.steps,
        "generated_per_class": args.generated_per_class,
    }
    # Different runs can share a training step (for example, two 75K
    # checkpoints), so the visual output directory must be checkpoint-specific.
    checkpoint_path_obj = Path(checkpoint_path)
    checkpoint_slug = f"{checkpoint_path_obj.parent.name}-{checkpoint_path_obj.stem}"
    destination = (
        Path(args.output)
        / f"step-{checkpoint['step']:07d}-{checkpoint_slug}"
    )
    result["visual_directory"] = destination.name
    destination.mkdir(parents=True, exist_ok=True)
    indices = []
    for class_index in range(len(class_values)):
        offset = class_index * args.generated_per_class
        indices.extend(range(offset, offset + args.visuals_per_class))
    save_image(
        generated[indices].add(1).div(2),
        destination / "generated-grid.png",
        nrow=args.visuals_per_class,
    )
    set_seed(args.seed)
    path_labels = torch.arange(len(class_values), device=device)
    paths = sampler.integrate(
        latent.sample(path_labels), args.steps, args.method, return_path=True
    ).cpu()
    path_indices = torch.linspace(0, args.steps, 8).round().long().unique()
    save_image(
        paths[:, path_indices].reshape(-1, *paths.shape[2:])
        .clamp(-1, 1).add(1).div(2),
        destination / "ode-path-grid.png",
        nrow=len(path_indices),
    )
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plot_results(results, output: Path) -> None:
    results = sorted(results, key=lambda item: item["step"])
    steps = [item["step"] for item in results]
    fields = [
        ("generated_accuracy", "Generated target accuracy"),
        ("generated_target_confidence", "Target confidence"),
        ("mean_class_feature_frechet", "Feature Frechet (lower is better)"),
        ("mean_feature_diversity_ratio", "Diversity ratio"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (field, title) in zip(axes.flat, fields):
        axis.plot(steps, [item[field] for item in results], marker="o")
        axis.set(xlabel="training step", title=title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "metric-curves.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(results), figsize=(3.2 * len(results), 3.6), squeeze=False
    )
    for axis, result in zip(axes[0], results):
        image = np.asarray(Image.open(
            output / result["visual_directory"] / "generated-grid.png"
        ))
        axis.imshow(image)
        checkpoint_name = Path(result["checkpoint"]).parent.name
        axis.set_title(f"{checkpoint_name}\nStep {result['step']:,}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Generation-only flow: fixed Gaussian samples")
    figure.tight_layout()
    figure.savefig(output / "checkpoint-generation-comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    classifier, classifier_checkpoint = load_reference_classifier(
        args.classifier, device
    )
    first = torch.load(args.checkpoints[0], map_location="cpu")
    config = first["config"]
    dataset = FilteredVisionDataset(
        config["mode"], args.data_root, False, config["image_size"], config["classes"]
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    real_features = collect_real_features(
        classifier, loader, config["classes"], device, args.real_images
    )
    results = []
    for path in sorted(
        args.checkpoints,
        key=lambda item: int(torch.load(item, map_location="cpu")["step"]),
    ):
        result = evaluate_one(
            args, path, classifier, real_features, device
        )
        results.append(result)
        print(json.dumps(result))
    payload = {
        "reference_classifier": {
            "path": str(Path(args.classifier).resolve()),
            "full_test_accuracy": classifier_checkpoint.get("accuracy"),
        },
        "results": results,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_results(results, output)


if __name__ == "__main__":
    main()
