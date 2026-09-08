#!/usr/bin/env python3
"""Record posterior margins along the complete image -> Gaussian ODE path.

This is a diagnostic only: it does not change the model or the usual
endpoint classification metric.  For each real test image we integrate from
physical time t=1 to t=0 and evaluate the fixed Gaussian posterior at every
saved state.  The resulting arrays make it possible to distinguish endpoint
errors from errors accumulated in the middle of the path.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_transport.checkpoint import load_for_inference
from unified_transport.data import FilteredVisionDataset
from unified_transport.generation_only import GenerationOnlySampler
from unified_transport.utils import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[15, 30, 60, 120])
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-test-images", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def posterior_margin(latent, states, labels):
    """Return true-class margin, posterior confidence, and prediction.

    ``states`` has shape [batch, time, channels, height, width].
    Log-softmax does not change a logit difference, but using it makes the
    returned values directly interpretable as log posterior probabilities.
    """
    batch, time = states.shape[:2]
    flat = states.reshape(batch * time, *states.shape[2:])
    logits = latent.log_posterior(flat).log_softmax(1).reshape(batch, time, -1)
    labels_view = labels[:, None]
    true_logp = logits.gather(2, labels_view.expand(-1, time)[:, :, None]).squeeze(2)
    other = logits.clone()
    other.scatter_(2, labels_view.expand(-1, time)[:, :, None], float("-inf"))
    margin = true_logp - other.max(2).values
    confidence, prediction = logits.exp().max(2)
    return margin, confidence, prediction


def summarize_path(margins, confidences, predictions, labels, steps, method):
    """Build JSON-serializable aggregate statistics for one solver setting."""
    count, path_len = margins.shape
    correct = predictions.eq(labels[:, None])
    reverse_progress = np.linspace(0.0, 1.0, path_len)
    physical_time = 1.0 - reverse_progress
    first_error = []
    first_change = []
    first_correct = []
    last_error = []
    stable_correct = []
    for row in range(count):
        error_indices = torch.nonzero(~correct[row], as_tuple=False).flatten()
        correct_indices = torch.nonzero(correct[row], as_tuple=False).flatten()
        change_indices = torch.nonzero(
            predictions[row, 1:] != predictions[row, :-1], as_tuple=False
        ).flatten()
        first_error.append(int(error_indices[0]) if error_indices.numel() else None)
        first_change.append(int(change_indices[0] + 1) if change_indices.numel() else None)
        first_correct.append(int(correct_indices[0]) if correct_indices.numel() else None)
        last_error.append(int(error_indices[-1]) if error_indices.numel() else None)
        # The first index after which the trajectory remains correctly
        # assigned all the way to the Gaussian endpoint.
        stable_index = None
        for index in range(path_len):
            if bool(correct[row, index:].all()):
                stable_index = index
                break
        stable_correct.append(stable_index)

    def segment(start, end):
        lo = int(round(start * (path_len - 1)))
        hi = int(round(end * (path_len - 1))) + 1
        values = margins[:, lo:hi]
        flags = correct[:, lo:hi]
        return {
            "reverse_progress": [start, end],
            "physical_time": [1.0 - start, 1.0 - end],
            "mean_margin": float(values.mean()),
            "median_margin": float(values.median()),
            "p10_margin": float(values.quantile(0.10)),
            "accuracy": float(flags.float().mean()),
            "negative_margin_rate": float((values < 0).float().mean()),
        }

    margin_np = margins.cpu().numpy()
    confidence_np = confidences.cpu().numpy()
    accuracy_np = correct.float().mean(0).cpu().tolist()
    flip_np = torch.cat(
        (torch.zeros((count, 1), dtype=torch.bool), predictions[:, 1:] != predictions[:, :-1]),
        dim=1,
    ).float().mean(0).cpu().tolist()
    first_error_valid = [index for index in first_error if index is not None]
    first_change_valid = [index for index in first_change if index is not None]
    first_correct_valid = [index for index in first_correct if index is not None]
    last_error_valid = [index for index in last_error if index is not None]
    stable_correct_valid = [index for index in stable_correct if index is not None]
    result = {
        "steps": steps,
        "method": method,
        "images": count,
        "reverse_progress": reverse_progress.tolist(),
        "physical_time": physical_time.tolist(),
        "mean_margin": margin_np.mean(0).tolist(),
        "median_margin": np.median(margin_np, 0).tolist(),
        "p10_margin": np.quantile(margin_np, 0.10, axis=0).tolist(),
        "p90_margin": np.quantile(margin_np, 0.90, axis=0).tolist(),
        "mean_confidence": confidence_np.mean(0).tolist(),
        "accuracy": accuracy_np,
        "negative_margin_rate": (margin_np < 0).mean(0).tolist(),
        "prediction_flip_rate": flip_np,
        "final_reverse_accuracy": float(accuracy_np[-1]),
        "initial_image_endpoint_accuracy": float(accuracy_np[0]),
        "final_noise_endpoint_mean_margin": float(margin_np[:, -1].mean()),
        "first_error_index_mean": float(np.mean(first_error_valid)) if first_error_valid else None,
        "first_error_reverse_progress_mean": (
            float(np.mean(np.asarray(first_error_valid) / max(path_len - 1, 1)))
            if first_error_valid else None
        ),
        "never_wrong_fraction": float(sum(index is None for index in first_error) / count),
        "ever_correct_fraction": float(sum(index is not None for index in first_correct) / count),
        "first_correct_index_mean": (
            float(np.mean(first_correct_valid)) if first_correct_valid else None
        ),
        "first_correct_reverse_progress_mean": (
            float(np.mean(np.asarray(first_correct_valid) / max(path_len - 1, 1)))
            if first_correct_valid else None
        ),
        "last_error_index_mean": float(np.mean(last_error_valid)) if last_error_valid else None,
        "last_error_reverse_progress_mean": (
            float(np.mean(np.asarray(last_error_valid) / max(path_len - 1, 1)))
            if last_error_valid else None
        ),
        "stable_correct_fraction": float(len(stable_correct_valid) / count),
        "stable_correct_index_mean": (
            float(np.mean(stable_correct_valid)) if stable_correct_valid else None
        ),
        "stable_correct_reverse_progress_mean": (
            float(np.mean(np.asarray(stable_correct_valid) / max(path_len - 1, 1)))
            if stable_correct_valid else None
        ),
        "first_prediction_change_index_mean": (
            float(np.mean(first_change_valid)) if first_change_valid else None
        ),
        "segments": {
            "image_endpoint": segment(0.0, 0.2),
            "middle": segment(0.2, 0.8),
            "noise_endpoint": segment(0.8, 1.0),
        },
    }
    return result


def plot_result(result, destination):
    progress = np.asarray(result["reverse_progress"])
    time = np.asarray(result["physical_time"])
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    axes[0].fill_between(progress, result["p10_margin"], result["p90_margin"], alpha=0.20,
                         color="#2563eb", label="10-90% margin")
    axes[0].plot(progress, result["mean_margin"], color="#1d4ed8", linewidth=2,
                 label="mean margin")
    axes[0].plot(progress, result["median_margin"], color="#f97316", linewidth=1.7,
                 label="median margin")
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_ylabel("true log-posterior margin")
    axes[0].set_title(f"Reverse posterior margin ({result['method']}, {result['steps']} steps)")
    axes[0].legend(loc="best", frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot(progress, result["accuracy"], color="#059669", linewidth=2, label="path accuracy")
    axes[1].plot(progress, result["negative_margin_rate"], color="#dc2626", linewidth=1.8,
                 label="negative-margin rate")
    axes[1].plot(progress, result["prediction_flip_rate"], color="#7c3aed", linewidth=1.5,
                 label="prediction flip rate")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("reverse progress (image $t=1$ $\\rightarrow$ noise $t=0$)")
    axes[1].set_ylabel("rate")
    axes[1].legend(loc="best", frameon=False)
    axes[1].grid(alpha=0.2)
    top = axes[0].twiny()
    top.set_xlim(axes[0].get_xlim())
    ticks = np.linspace(0, 1, 6)
    top.set_xticks(ticks)
    top.set_xticklabels([f"{1.0 - value:.1f}" for value in ticks])
    top.set_xlabel("physical time t")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_comparison(results, destination):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for result in results:
        progress = result["reverse_progress"]
        label = f"{result['method']} {result['steps']}"
        axes[0].plot(progress, result["mean_margin"], linewidth=1.8, label=label)
        axes[1].plot(progress, result["accuracy"], linewidth=1.8, label=label)
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Mean posterior margin")
    axes[0].set_ylabel("true log-posterior margin")
    axes[1].set_title("Accuracy along reverse path")
    axes[1].set_ylabel("accuracy")
    for axis in axes:
        axis.set_xlabel("reverse progress: image $\\rightarrow$ noise")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


@torch.inference_mode()
def evaluate_step(args, model, latent, loader, device, steps):
    sampler = GenerationOnlySampler(model)
    margin_rows, confidence_rows, prediction_rows, label_rows = [], [], [], []
    seen = 0
    for images, labels in loader:
        if seen >= args.max_test_images:
            break
        take = min(images.shape[0], args.max_test_images - seen)
        images = images[:take].to(device, non_blocking=True)
        labels = labels[:take].to(device, non_blocking=True)
        path = sampler.integrate_interval(images, 1.0, 0.0, steps, args.method, return_path=True)
        margins, confidence, predictions = posterior_margin(latent, path, labels)
        margin_rows.append(margins.cpu())
        confidence_rows.append(confidence.cpu())
        prediction_rows.append(predictions.cpu())
        label_rows.append(labels.cpu())
        seen += take
    margins = torch.cat(margin_rows, dim=0)[:args.max_test_images]
    confidences = torch.cat(confidence_rows, dim=0)[:args.max_test_images]
    predictions = torch.cat(prediction_rows, dim=0)[:args.max_test_images]
    labels = torch.cat(label_rows, dim=0)[:args.max_test_images]
    return summarize_path(margins, confidences, predictions, labels, steps, args.method), labels, predictions, margins


def save_sample_csv(destination, labels, predictions, margins):
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "true_label", *[f"pred_{i}" for i in range(predictions.shape[1])],
                         *[f"margin_{i}" for i in range(margins.shape[1])]])
        for index in range(labels.shape[0]):
            writer.writerow([int(index), int(labels[index]), *predictions[index].tolist(),
                             *[f"{value:.7g}" for value in margins[index].tolist()]])


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
    dataset = FilteredVisionDataset(config["mode"], args.data_root, False,
                                    config["image_size"], config["classes"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    results = []
    for steps in args.steps:
        result, labels, predictions, margins = evaluate_step(args, model, latent, loader, device, steps)
        step_dir = output / f"{args.method}-{steps:03d}"
        step_dir.mkdir(exist_ok=True)
        (step_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        plot_result(result, step_dir / "posterior-margin.png")
        save_sample_csv(step_dir / "per-sample-path.csv", labels, predictions, margins)
        results.append(result)
        print(json.dumps({
            "steps": steps,
            "method": args.method,
            "images": result["images"],
            "final_accuracy": result["final_reverse_accuracy"],
            "mean_margin": result["mean_margin"],
            "segments": result["segments"],
        }))
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mode": config["mode"],
        "task": config["task"],
        "classes": config["classes"],
        "max_test_images": args.max_test_images,
        "results": results,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_comparison(results, output / "posterior-margin-comparison.png")


if __name__ == "__main__":
    main()
