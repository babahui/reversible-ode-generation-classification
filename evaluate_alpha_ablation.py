#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from unified_transport.checkpoint import load_for_inference
from unified_transport.data import FilteredVisionDataset
from unified_transport.interpolant import TransportSampler
from unified_transport.reference_classifier import load_reference_classifier
from unified_transport.utils import resolve_device, set_seed


class AlphaOverrideModel(nn.Module):
    def __init__(self, model: nn.Module, fixed_midpoint: bool = False):
        super().__init__()
        self.model = model
        self.num_marginals = model.num_marginals
        self.fixed_midpoint = fixed_midpoint

    def forward(self, x, alpha, previous=None, recent_velocity=None):
        if self.fixed_midpoint:
            alpha = torch.full_like(alpha, 1.0 / self.num_marginals)
        return self.model(x, alpha, previous, recent_velocity)


class SignedTransportSampler(TransportSampler):
    def __init__(self, model: nn.Module, velocity_sign: float = 1.0):
        super().__init__(model)
        self.velocity_sign = velocity_sign

    def _velocity(self, *args, **kwargs):
        return self.velocity_sign * super()._velocity(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate alpha/path ablations on MNIST.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--no-alpha-checkpoint", default="")
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-images", type=int, default=2048)
    parser.add_argument("--generated-per-class", type=int, default=256)
    parser.add_argument("--visuals-per-class", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_variant(
    name, model, latent, sampler, classifier, loader, class_values, args, device
):
    total = correct = 0
    nll = 0.0
    for images, labels in loader:
        if total >= args.test_images:
            break
        take = min(images.shape[0], args.test_images - total)
        images = images[:take].to(device, non_blocking=True)
        labels = labels[:take].to(device, non_blocking=True)
        encoded = sampler.integrate(images, 0, 1, args.steps, "heun")
        log_posterior = latent.log_posterior(encoded).log_softmax(1)
        correct += int((log_posterior.argmax(1) == labels).sum())
        nll -= float(log_posterior.gather(1, labels[:, None]).sum())
        total += take

    set_seed(args.seed)
    generated_parts = []
    generated_targets = []
    class_accuracy = {}
    for local_label, original_label in enumerate(class_values):
        labels = torch.full(
            (args.generated_per_class,), local_label, dtype=torch.long, device=device
        )
        initial = latent.sample(labels)
        generated = sampler.integrate(initial, 1, 0, args.steps, "heun").clamp(-1, 1)
        target = torch.full(
            (args.generated_per_class,), original_label, dtype=torch.long, device=device
        )
        predictions = classifier(generated).argmax(1)
        class_accuracy[str(original_label)] = float(
            (predictions == target).float().mean()
        )
        generated_parts.append(generated.cpu())
        generated_targets.append(target.cpu())
    generated = torch.cat(generated_parts)
    targets = torch.cat(generated_targets).to(device)
    logits = classifier(generated.to(device))
    probabilities = logits.softmax(1)
    result = {
        "variant": name,
        "inverse_accuracy": correct / total,
        "inverse_posterior_nll": nll / total,
        "generated_accuracy": float((logits.argmax(1) == targets).float().mean()),
        "generated_target_confidence": float(
            probabilities.gather(1, targets[:, None]).mean()
        ),
        "per_label_generated_accuracy": class_accuracy,
        "test_images": total,
        "generated_per_class": args.generated_per_class,
        "ode_steps": args.steps,
    }
    visual_indices = []
    offset = 0
    for _ in class_values:
        visual_indices.extend(range(offset, offset + args.visuals_per_class))
        offset += args.generated_per_class
    destination = Path(args.output) / name
    destination.mkdir(parents=True, exist_ok=True)
    save_image(
        generated[visual_indices].add(1).div(2),
        destination / "generated-grid.png",
        nrow=args.visuals_per_class,
    )
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plot_results(results, output: Path) -> None:
    names = [item["variant"] for item in results]
    inverse = [item["inverse_accuracy"] for item in results]
    generated = [item["generated_accuracy"] for item in results]
    figure, axis = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(names))
    width = 0.36
    axis.bar(positions - width / 2, inverse, width, label="image -> noise classification")
    axis.bar(positions + width / 2, generated, width, label="noise -> image generation")
    axis.set_xticks(positions, names, rotation=15, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("accuracy")
    axis.set_title("Alpha/path ablations")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "alpha-ablation-accuracy.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    classifier, classifier_checkpoint = load_reference_classifier(args.classifier, device)
    if classifier_checkpoint.get("mode", "mnist") != "mnist":
        raise ValueError("alpha ablation currently expects the MNIST reference classifier")
    model, latent, checkpoint = load_for_inference(args.checkpoint, device)
    config = checkpoint["config"]
    if config["mode"] != "mnist" or config.get("use_history", False):
        raise ValueError("alpha ablation expects a Markov MNIST checkpoint")
    class_values = config["classes"]
    dataset = FilteredVisionDataset(
        "mnist", args.data_root, False, config["image_size"], class_values
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    variants = [
        (
            "correct_alpha",
            model,
            latent,
            SignedTransportSampler(AlphaOverrideModel(model), 1.0),
        ),
        (
            "fixed_midpoint_alpha",
            model,
            latent,
            SignedTransportSampler(AlphaOverrideModel(model, fixed_midpoint=True), 1.0),
        ),
        (
            "wrong_alpha_dot_sign",
            model,
            latent,
            SignedTransportSampler(AlphaOverrideModel(model), -1.0),
        ),
    ]
    if args.no_alpha_checkpoint:
        no_alpha_model, no_alpha_latent, no_alpha_checkpoint = load_for_inference(
            args.no_alpha_checkpoint, device
        )
        if no_alpha_checkpoint["config"].get("condition_on_alpha", True):
            raise ValueError("--no-alpha-checkpoint was trained with alpha conditioning")
        variants.append(
            (
                "trained_without_alpha_condition",
                no_alpha_model,
                no_alpha_latent,
                SignedTransportSampler(no_alpha_model, 1.0),
            )
        )
    results = [
        evaluate_variant(
            name,
            variant_model,
            variant_latent,
            sampler,
            classifier,
            loader,
            class_values,
            args,
            device,
        )
        for name, variant_model, variant_latent, sampler in variants
    ]
    (output / "metrics.json").write_text(json.dumps({"results": results}, indent=2) + "\n")
    plot_results(results, output)
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
