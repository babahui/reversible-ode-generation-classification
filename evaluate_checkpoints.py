#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from unified_transport.checkpoint import load_for_inference
from unified_transport.data import FilteredVisionDataset
from unified_transport.interpolant import TransportSampler
from unified_transport.reference_classifier import load_reference_classifier
from unified_transport.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate transport checkpoints over training.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-test-images", type=int, default=1024)
    parser.add_argument("--cycle-images", type=int, default=128)
    parser.add_argument("--generated-per-class", type=int, default=256)
    parser.add_argument("--visuals-per-class", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features.double() - features.double().mean(0, keepdim=True)
    return centered.T @ centered / max(features.shape[0] - 1, 1)


def feature_frechet(real: torch.Tensor, generated: torch.Tensor) -> float:
    real = real.double()
    generated = generated.double()
    mean_difference = (real.mean(0) - generated.mean(0)).square().sum()
    covariance_real = covariance(real)
    covariance_generated = covariance(generated)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_real)
    square_root = (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T
    middle = square_root @ covariance_generated @ square_root
    cross_trace = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum()
    distance = mean_difference + torch.trace(covariance_real + covariance_generated) - 2 * cross_trace
    return float(distance.clamp_min(0))


@torch.inference_mode()
def collect_real_features(classifier, loader, class_values, device, maximum):
    features = {label: [] for label in class_values}
    seen = 0
    for images, local_labels in loader:
        if seen >= maximum:
            break
        take = min(images.shape[0], maximum - seen)
        images = images[:take].to(device, non_blocking=True)
        local_labels = local_labels[:take]
        batch_features = classifier(images, return_features=True).cpu()
        for local_index, original_label in enumerate(class_values):
            mask = local_labels == local_index
            if mask.any():
                features[original_label].append(batch_features[mask])
        seen += take
    return {label: torch.cat(parts) for label, parts in features.items()}


@torch.inference_mode()
def evaluate_one(args, checkpoint_path, classifier, real_features, loader, device):
    model, latent, checkpoint = load_for_inference(checkpoint_path, device)
    config = checkpoint["config"]
    if config["mode"] not in {"mnist", "cifar10", "tinyimagenet", "stl10"} or latent is None:
        raise ValueError("checkpoint evaluation requires an image-noise model")
    if config.get("use_history", False):
        raise ValueError("strict bidirectional evaluation requires a checkpoint trained without history")
    class_values = config["classes"]
    sampler = TransportSampler(model)
    total = internal_correct = external_correct = 0
    posterior_nll = cycle_error = 0.0
    cycle_elements = 0
    inverse_confusion = torch.zeros(
        len(class_values), len(class_values), dtype=torch.long
    )
    reconstruction_pair = None

    for images, local_labels in tqdm(loader, desc=f"inverse step {checkpoint['step']}", leave=False):
        if total >= args.max_test_images:
            break
        take = min(images.shape[0], args.max_test_images - total)
        images = images[:take].to(device, non_blocking=True)
        local_labels = local_labels[:take].to(device, non_blocking=True)
        encoded = sampler.integrate(images, 0, 1, args.steps, args.method)
        log_posterior = latent.log_posterior(encoded).log_softmax(1)
        inverse_predictions = log_posterior.argmax(1)
        internal_correct += int((inverse_predictions == local_labels).sum())
        bins = torch.bincount(
            (local_labels * len(class_values) + inverse_predictions).cpu(),
            minlength=len(class_values) ** 2,
        )
        inverse_confusion += bins.reshape(len(class_values), len(class_values))
        posterior_nll -= float(log_posterior.gather(1, local_labels[:, None]).sum())
        original_labels = torch.tensor(class_values, device=device)[local_labels]
        external_correct += int((classifier(images).argmax(1) == original_labels).sum())

        remaining_cycle = max(args.cycle_images - cycle_elements // images[0].numel(), 0)
        if remaining_cycle:
            cycle_take = min(take, remaining_cycle)
            reconstructed = sampler.integrate(
                encoded[:cycle_take], 1, 0, args.steps, args.method
            )
            cycle_error += float((reconstructed - images[:cycle_take]).square().sum())
            cycle_elements += reconstructed.numel()
            if reconstruction_pair is None:
                reconstruction_pair = (images[:16].cpu(), reconstructed[:16].cpu())
        total += take

    set_seed(args.seed)
    generated_parts = []
    generated_labels = []
    class_metrics = {}
    for local_index, original_label in enumerate(class_values):
        labels = torch.full(
            (args.generated_per_class,), local_index, dtype=torch.long, device=device
        )
        initial = latent.sample(labels)
        generated = sampler.integrate(initial, 1, 0, args.steps, args.method).clamp(-1, 1)
        logits = classifier(generated)
        probabilities = logits.softmax(1)
        predictions = logits.argmax(1)
        target = torch.full_like(predictions, original_label)
        generated_features = classifier(generated, return_features=True).cpu()
        generated_parts.append(generated.cpu())
        generated_labels.append(target.cpu())
        real_class_features = real_features[original_label]
        generated_covariance_trace = float(torch.trace(covariance(generated_features)))
        real_covariance_trace = float(torch.trace(covariance(real_class_features)))
        class_metrics[str(original_label)] = {
            "accuracy": float((predictions == target).float().mean()),
            "target_confidence": float(probabilities[:, original_label].mean()),
            "feature_frechet": feature_frechet(real_class_features, generated_features),
            "feature_diversity_ratio": generated_covariance_trace / max(real_covariance_trace, 1e-12),
        }

    generated = torch.cat(generated_parts)
    generated_labels_tensor = torch.cat(generated_labels).to(device)
    generated_logits = classifier(generated.to(device))
    result = {
        "step": int(checkpoint["step"]),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "inverse_accuracy": internal_correct / total,
        "inverse_posterior_nll": posterior_nll / total,
        "inverse_per_class_accuracy": {
            str(original_label): float(
                inverse_confusion[local_index, local_index]
                / inverse_confusion[local_index].sum().clamp_min(1)
            )
            for local_index, original_label in enumerate(class_values)
        },
        "inverse_confusion_matrix": inverse_confusion.tolist(),
        "reference_test_accuracy": external_correct / total,
        "cycle_mse": cycle_error / cycle_elements,
        "generated_accuracy": float((generated_logits.argmax(1) == generated_labels_tensor).float().mean()),
        "generated_target_confidence": float(
            generated_logits.softmax(1).gather(1, generated_labels_tensor[:, None]).mean()
        ),
        "mean_class_feature_frechet": sum(item["feature_frechet"] for item in class_metrics.values()) / len(class_metrics),
        "classes": class_metrics,
        "ode_steps": args.steps,
        "generated_per_class": args.generated_per_class,
        "test_images": total,
    }

    step_directory = Path(args.output) / f"step-{checkpoint['step']:07d}"
    step_directory.mkdir(parents=True, exist_ok=True)
    visual_indices = []
    offset = 0
    for _ in class_values:
        visual_indices.extend(range(offset, offset + args.visuals_per_class))
        offset += args.generated_per_class
    save_image(
        generated[visual_indices].add(1).div(2),
        step_directory / "generated-grid.png",
        nrow=args.visuals_per_class,
    )
    if reconstruction_pair is not None:
        originals, reconstructions = reconstruction_pair
        save_image(
            torch.cat((originals, reconstructions)).clamp(-1, 1).add(1).div(2),
            step_directory / "reconstruction-grid.png",
            nrow=originals.shape[0],
        )

    set_seed(args.seed)
    path_labels = torch.arange(len(class_values), device=device)
    path_initial = latent.sample(path_labels)
    paths = sampler.integrate(
        path_initial, 1, 0, args.steps, args.method, return_path=True
    ).cpu()
    path_indices = torch.linspace(0, args.steps, 8).round().long().unique()
    selected_paths = paths[:, path_indices].reshape(-1, *paths.shape[2:])
    save_image(
        selected_paths.clamp(-1, 1).add(1).div(2),
        step_directory / "ode-path-grid.png",
        nrow=len(path_indices),
    )
    (step_directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plot_metrics(results: List[Dict], destination: Path) -> None:
    results = sorted(results, key=lambda item: item["step"])
    steps = [item["step"] for item in results]
    specifications: List[Tuple[str, str]] = [
        ("inverse_accuracy", "Inverse classification accuracy"),
        ("generated_accuracy", "Generated target accuracy"),
        ("generated_target_confidence", "Generated target confidence"),
        ("mean_class_feature_frechet", "Classifier-feature Frechet (lower is better)"),
        ("cycle_mse", "Cycle MSE (lower is better)"),
        ("inverse_posterior_nll", "Inverse posterior NLL"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, (key, title) in zip(axes.flat, specifications):
        axis.plot(steps, [item[key] for item in results], marker="o")
        axis.set(xlabel="training step", title=title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    classifier, classifier_checkpoint = load_reference_classifier(args.classifier, device)
    first_checkpoint = torch.load(args.checkpoints[0], map_location="cpu")
    first_config = first_checkpoint["config"]
    classifier_mode = classifier_checkpoint.get("mode", "mnist")
    if classifier_mode != first_config["mode"]:
        raise ValueError(
            f"reference classifier mode {classifier_mode!r} does not match checkpoint mode {first_config['mode']!r}"
        )
    class_values = first_config["classes"]
    dataset = FilteredVisionDataset(
        first_config["mode"],
        args.data_root,
        False,
        first_config["image_size"],
        class_values,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    real_features = collect_real_features(
        classifier, loader, class_values, device, args.max_test_images
    )
    metrics_path = Path(args.output) / "metrics.json"
    existing_results = []
    if metrics_path.exists():
        existing_results = json.loads(metrics_path.read_text()).get("results", [])
    new_results = []
    for checkpoint_path in sorted(
        args.checkpoints, key=lambda path: int(torch.load(path, map_location="cpu")["step"])
    ):
        result = evaluate_one(
            args, checkpoint_path, classifier, real_features, loader, device
        )
        new_results.append(result)
        print(json.dumps(result))
    results_by_step = {
        int(item["step"]): item for item in existing_results + new_results
    }
    results = [results_by_step[step] for step in sorted(results_by_step)]
    summary = {
        "reference_classifier": {
            "path": str(Path(args.classifier).resolve()),
            "full_test_accuracy": classifier_checkpoint.get("accuracy"),
            "epoch": classifier_checkpoint.get("epoch"),
        },
        "results": sorted(results, key=lambda item: item["step"]),
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_metrics(results, Path(args.output) / "metric-curves.png")


if __name__ == "__main__":
    main()
