#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build annotated comparison figures from checkpoint evaluations."
    )
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step-(\d+)", path.name)
    if match is None:
        raise ValueError(f"cannot parse checkpoint step from {path}")
    return int(match.group(1))


def load_results(evaluation: Path):
    payload = json.loads((evaluation / "metrics.json").read_text())
    generation = {
        int(item["step"]): item
        for item in payload["results"]
    }
    classification = {}
    for path in evaluation.glob("full-test-step-*.json"):
        classification[checkpoint_step(path)] = json.loads(path.read_text())
    for step, item in generation.items():
        if step not in classification and "inverse_accuracy" in item:
            total = int(item["test_images"])
            classification[step] = {
                "accuracy": item["inverse_accuracy"],
                "correct": round(item["inverse_accuracy"] * total),
                "total": total,
                "posterior_nll": item["inverse_posterior_nll"],
            }
    steps = sorted(set(generation) & set(classification))
    if not steps:
        raise ValueError("no steps have both generation and full-test results")
    return steps, generation, classification


def plot_generation_grids(evaluation: Path, steps, output: Path) -> None:
    figure, axes = plt.subplots(1, len(steps), figsize=(3.2 * len(steps), 3.8))
    axes = np.atleast_1d(axes)
    for axis, step in zip(axes, steps):
        grid_path = evaluation / f"step-{step:07d}" / "generated-grid.png"
        image = np.asarray(Image.open(grid_path))
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(image)
        axis.set_title(f"Step {step:,}")
        tile_stride = image.shape[0] / 10
        centers = (np.arange(10) + 0.5) * tile_stride
        axis.set_yticks(centers, [str(label) for label in range(10)])
        axis.set_xticks([])
        axis.set_ylabel("target label")
    figure.suptitle("Fixed-seed class-conditional generation across checkpoints")
    figure.tight_layout()
    figure.savefig(output / "generation-checkpoint-comparison.png", dpi=180)
    plt.close(figure)


def plot_generation_by_label(evaluation: Path, steps, output: Path) -> None:
    """Arrange fixed samples as target-label rows and checkpoint columns."""
    figure, axes = plt.subplots(10, len(steps), figsize=(3.2 * len(steps), 7.0))
    for column, step in enumerate(steps):
        grid_path = evaluation / f"step-{step:07d}" / "generated-grid.png"
        image = np.asarray(Image.open(grid_path))
        padding = 2
        # Evaluation grids contain 10 columns and 10 class rows.
        tile_size = (image.shape[1] - 11 * padding) // 10
        stride = tile_size + padding
        for label in range(10):
            axis = axes[label, column]
            top = padding + label * stride
            strip = image[top : top + tile_size, :]
            if strip.ndim == 2:
                axis.imshow(strip, cmap="gray", vmin=0, vmax=255, aspect="auto")
            else:
                axis.imshow(strip, aspect="auto")
            axis.set_xticks([])
            axis.set_yticks([])
            if label == 0:
                axis.set_title(f"Step {step:,}", fontsize=10)
            if column == 0:
                axis.set_ylabel(str(label), rotation=0, labelpad=10, va="center")
            for spine in axis.spines.values():
                spine.set_visible(False)
    figure.supylabel("target label")
    figure.suptitle("Same fixed latent samples across checkpoints", y=0.995)
    figure.subplots_adjust(left=0.055, right=0.995, top=0.94, bottom=0.02, wspace=0.04, hspace=0.12)
    figure.savefig(output / "generation-by-label-and-checkpoint.png", dpi=200)
    plt.close(figure)


def plot_accuracy_heatmap(steps, generation, output: Path) -> None:
    values = np.array(
        [
            [generation[step]["classes"][str(label)]["accuracy"] for label in range(10)]
            for step in steps
        ]
    )
    figure, axis = plt.subplots(figsize=(11, 3.8))
    image = axis.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(10), range(10))
    axis.set_yticks(range(len(steps)), [f"{step:,}" for step in steps])
    axis.set(xlabel="target label", ylabel="training step", title="Generated-label accuracy")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] < 0.55 else "black"
            axis.text(
                column,
                row,
                f"{100 * values[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    figure.colorbar(image, ax=axis, label="accuracy")
    figure.tight_layout()
    figure.savefig(output / "generation-label-accuracy-heatmap.png", dpi=180)
    plt.close(figure)


def plot_inverse_confusion(step, generation, output: Path) -> None:
    result = generation[step]
    if "inverse_confusion_matrix" not in result:
        return
    matrix = np.asarray(result["inverse_confusion_matrix"], dtype=float)
    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(10), range(10))
    axis.set_yticks(range(10), range(10))
    axis.set(
        xlabel="predicted class",
        ylabel="true class",
        title=f"Image -> noise confusion at step {step:,}",
    )
    figure.colorbar(image, ax=axis, label="row-normalized fraction")
    figure.tight_layout()
    figure.savefig(output / "inverse-confusion-latest.png", dpi=180)
    plt.close(figure)


def write_report(steps, generation, classification, output: Path) -> None:
    total = classification[steps[-1]].get("total", generation[steps[-1]]["test_images"])
    lines = [
        "# Checkpoint Comparison",
        "",
        f"Classification uses {total:,} test images and 30-step Heun integration. Generation uses 256 fixed-seed samples per label and a frozen reference classifier.",
        "",
        "| Step | Test accuracy | Correct | Posterior NLL | Generated accuracy | Feature Frechet |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in steps:
        inverse = classification[step]
        generated = generation[step]
        lines.append(
            f"| {step:,} | {100 * inverse['accuracy']:.2f}% | "
            f"{inverse['correct']}/{inverse.get('total', generated['test_images'])} | {inverse['posterior_nll']:.4f} | "
            f"{100 * generated['generated_accuracy']:.2f}% | "
            f"{generated['mean_class_feature_frechet']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Visual Comparison",
            "",
            "Rows are target labels and columns are checkpoints. Every checkpoint uses the same random seed and latent sample order.",
            "",
            "![Generation by label and checkpoint](generation-by-label-and-checkpoint.png)",
            "",
            "![Generated-label accuracy heatmap](generation-label-accuracy-heatmap.png)",
            "",
            "![Checkpoint generation grids](generation-checkpoint-comparison.png)",
            "",
            "![Latest inverse confusion](inverse-confusion-latest.png)",
            "",
            "## Per-label Generated Accuracy",
            "",
            "| Step | " + " | ".join(str(label) for label in range(10)) + " |",
            "| ---: | " + " | ".join("---:" for _ in range(10)) + " |",
        ]
    )
    for step in steps:
        values = [
            generation[step]["classes"][str(label)]["accuracy"] for label in range(10)
        ]
        lines.append(
            f"| {step:,} | " + " | ".join(f"{100 * value:.1f}%" for value in values) + " |"
        )
    lines.extend(
        [
            "",
        ]
    )
    (output / "checkpoint-comparison.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    evaluation = Path(args.evaluation).resolve()
    output = Path(args.output).resolve() if args.output else evaluation
    output.mkdir(parents=True, exist_ok=True)
    steps, generation, classification = load_results(evaluation)
    plot_generation_grids(evaluation, steps, output)
    plot_generation_by_label(evaluation, steps, output)
    plot_accuracy_heatmap(steps, generation, output)
    plot_inverse_confusion(steps[-1], generation, output)
    write_report(steps, generation, classification, output)


if __name__ == "__main__":
    main()
