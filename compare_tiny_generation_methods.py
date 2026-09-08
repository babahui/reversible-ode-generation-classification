#!/usr/bin/env python3
"""Create a reproducible Tiny-ImageNet three-method comparison report."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_result(path: Path):
    payload = json.loads((path / "metrics.json").read_text())
    return {int(item["step"]): item for item in payload.get("results", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified", required=True)
    parser.add_argument("--gaussian", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = {
        "Unified": Path(args.unified),
        "Gaussian noise-only": Path(args.gaussian),
        "Label-conditioned": Path(args.label),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {name: load_result(root) for name, root in roots.items()}
    steps = sorted(set.intersection(*(set(values) for values in metrics.values())))
    summary = {str(step): {name: values[step] for name, values in metrics.items()} for step in steps}
    (output / "three-method-metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    fields = ["generated_accuracy", "generated_target_confidence", "mean_class_feature_frechet"]
    figure, axes = plt.subplots(1, len(fields), figsize=(14, 4))
    for axis, field in zip(axes, fields):
        for name, values in metrics.items():
            axis.plot(steps, [values[step][field] for step in steps], marker="o", label=name)
        axis.set(xlabel="training step", title=field.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "three-method-metric-curves.png", dpi=180)
    plt.close(figure)

    # Rows are methods and columns are checkpoints; each cell is the actual
    # generated grid, so differences cannot be hidden by a shared plot scale.
    figure, axes = plt.subplots(len(roots), len(steps), figsize=(3.2 * len(steps), 3.0 * len(roots)))
    axes = np.asarray(axes).reshape(len(roots), len(steps))
    for row, (name, root) in enumerate(roots.items()):
        for column, step in enumerate(steps):
            image = np.asarray(Image.open(root / f"step-{step:07d}" / "generated-grid.png"))
            axes[row, column].imshow(image)
            axes[row, column].set_title(f"{name}\n{step:,}", fontsize=9)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle("Tiny-ImageNet-10: fixed-seed generated grids")
    figure.tight_layout()
    figure.savefig(output / "three-method-checkpoint-grids.png", dpi=180)
    plt.close(figure)

    final_step = steps[-1]
    final_images = {
        name: np.asarray(Image.open(root / f"step-{final_step:07d}" / "generated-grid.png")).astype(np.float32) / 255
        for name, root in roots.items()
    }
    pairwise = {}
    names = list(final_images)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            difference = np.abs(final_images[left] - final_images[right])
            pairwise[f"{left}_vs_{right}"] = {
                "mae": float(difference.mean()),
                "mse": float(np.square(difference).mean()),
                "exact_pixel_fraction": float(np.mean(final_images[left] == final_images[right])),
            }
    (output / "final-pairwise-pixel-differences.json").write_text(json.dumps(pairwise, indent=2) + "\n")

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, name in enumerate(names):
        axes[0, column].imshow(final_images[name])
        axes[0, column].set_title(name)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
    for axis, (left, right) in zip(axes[1], [(names[0], names[1]), (names[0], names[2]), (names[1], names[2])]):
        difference = np.abs(final_images[left] - final_images[right])
        axis.imshow(difference / max(float(difference.max()), 1e-8))
        axis.set_title(f"|{left} - {right}|")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Tiny-ImageNet-10 final checkpoint ({final_step:,}): pixel differences")
    figure.tight_layout()
    figure.savefig(output / "three-method-final-differences.png", dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
