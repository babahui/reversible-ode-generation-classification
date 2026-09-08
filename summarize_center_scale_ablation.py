#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two center-scale experiments.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--stronger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-name", default="center scale 4")
    parser.add_argument("--stronger-name", default="center scale 12")
    return parser.parse_args()


def load_results(path):
    payload = json.loads((Path(path) / "metrics.json").read_text())
    return {int(item["step"]): item for item in payload["results"]}


def mean_diversity(item):
    return float(np.mean([entry["feature_diversity_ratio"] for entry in item["classes"].values()]))


def plot_metrics(steps, baseline, stronger, names, output):
    definitions = [
        ("inverse_accuracy", "Image -> noise accuracy", 100.0),
        ("generated_accuracy", "Noise -> image accuracy", 100.0),
        ("mean_class_feature_frechet", "Feature Frechet (lower is better)", 1.0),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    positions = np.arange(len(steps))
    width = 0.36
    for axis, (key, title, scale) in zip(axes, definitions):
        axis.bar(
            positions - width / 2,
            [baseline[step][key] * scale for step in steps],
            width,
            label=names[0],
        )
        axis.bar(
            positions + width / 2,
            [stronger[step][key] * scale for step in steps],
            width,
            label=names[1],
        )
        axis.set_xticks(positions, [f"{step // 1000}k" for step in steps])
        axis.set(xlabel="training step", title=title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("percent")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output / "center-scale-metrics.png", dpi=180)
    plt.close(figure)


def plot_grids(step, baseline_path, stronger_path, names, output):
    paths = [
        Path(baseline_path) / f"step-{step:07d}" / "generated-grid.png",
        Path(stronger_path) / f"step-{step:07d}" / "generated-grid.png",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for axis, path, name in zip(axes, paths, names):
        axis.imshow(Image.open(path))
        axis.set_title(f"{name}, step {step:,}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / f"generated-comparison-{step}.png", dpi=200)
    plt.close(figure)


def plot_confusions(step, baseline, stronger, names, output):
    matrices = []
    for result in (baseline[step], stronger[step]):
        matrix = np.asarray(result["inverse_confusion_matrix"], dtype=float)
        matrices.append(matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1))
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for axis, matrix, name in zip(axes, matrices, names):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        axis.set_title(name)
        axis.set(xlabel="predicted class", ylabel="true class")
        axis.set_xticks(range(10), range(10))
        axis.set_yticks(range(10), range(10))
    figure.colorbar(image, ax=axes, label="row-normalized fraction", shrink=0.85)
    figure.suptitle(f"Image -> noise confusion at step {step:,}")
    figure.subplots_adjust(left=0.07, right=0.9, bottom=0.12, top=0.86, wspace=0.25)
    figure.savefig(output / f"inverse-confusion-{step}.png", dpi=180)
    plt.close(figure)


def write_report(steps, baseline, stronger, names, output):
    lines = [
        "# CIFAR-10 Center Scale Ablation",
        "",
        "Only the class-Gaussian center norm changes from 4 to 12. Sigma, data, seed, model, optimizer, training budget, EMA, alpha path, ODE solver, and frozen evaluator are unchanged.",
        "",
        "| Step | Scale 4 inverse | Scale 12 inverse | Scale 4 generated | Scale 12 generated | Scale 4 Frechet | Scale 12 Frechet |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in steps:
        left, right = baseline[step], stronger[step]
        lines.append(
            f"| {step:,} | {100*left['inverse_accuracy']:.2f}% | {100*right['inverse_accuracy']:.2f}% | "
            f"{100*left['generated_accuracy']:.2f}% | {100*right['generated_accuracy']:.2f}% | "
            f"{left['mean_class_feature_frechet']:.2f} | {right['mean_class_feature_frechet']:.2f} |"
        )
    final_step = steps[-1]
    left, right = baseline[final_step], stronger[final_step]
    lines.extend([
        "",
        "![Metric comparison](center-scale-metrics.png)",
        "",
        f"![Generated comparison](generated-comparison-{final_step}.png)",
        "",
        f"![Inverse confusion](inverse-confusion-{final_step}.png)",
        "",
        "## Final checkpoint diagnostics",
        "",
        f"At {final_step:,} steps, inverse accuracy improves by {100*(right['inverse_accuracy']-left['inverse_accuracy']):.2f} percentage points and generated-label accuracy improves by {100*(right['generated_accuracy']-left['generated_accuracy']):.2f} points. Feature Frechet falls by {left['mean_class_feature_frechet']-right['mean_class_feature_frechet']:.2f}.",
        "",
        f"Mean feature-diversity ratio changes from {mean_diversity(left):.3f} to {mean_diversity(right):.3f}. The stronger class signal therefore improves control and inverse partitioning, but the generated distribution is somewhat less diverse in classifier feature space.",
        "",
        "Posterior NLL is not comparable across center scales because the Gaussian-distance logits change scale while sigma remains fixed.",
        "",
    ])
    (output / "center-scale-ablation.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = load_results(args.baseline)
    stronger = load_results(args.stronger)
    steps = sorted(set(baseline) & set(stronger))
    if not steps:
        raise ValueError("the evaluations have no common checkpoint steps")
    names = (args.baseline_name, args.stronger_name)
    plot_metrics(steps, baseline, stronger, names, output)
    plot_grids(steps[-1], args.baseline, args.stronger, names, output)
    if "inverse_confusion_matrix" in baseline[steps[-1]] and "inverse_confusion_matrix" in stronger[steps[-1]]:
        plot_confusions(steps[-1], baseline, stronger, names, output)
    write_report(steps, baseline, stronger, names, output)


if __name__ == "__main__":
    main()
