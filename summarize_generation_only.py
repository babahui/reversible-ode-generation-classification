#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare unified and generation-only CIFAR experiments."
    )
    parser.add_argument("--generation-evaluation", required=True)
    parser.add_argument("--unified-evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generation_directory = Path(args.generation_evaluation)
    unified_directory = Path(args.unified_evaluation)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    generation = {
        item["step"]: item
        for item in json.loads(
            (generation_directory / "metrics.json").read_text()
        )["results"]
    }
    unified = {
        item["step"]: item
        for item in json.loads(
            (unified_directory / "metrics.json").read_text()
        )["results"]
    }
    steps = sorted(set(generation) & set(unified))
    specifications = [
        ("generated_accuracy", "Generated target accuracy", False),
        ("mean_class_feature_frechet", "Classifier-feature Frechet", True),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, (field, title, _) in zip(axes[:2], specifications):
        axis.plot(
            steps, [generation[step][field] for step in steps],
            marker="o", label="generation-only"
        )
        axis.plot(
            steps, [unified[step][field] for step in steps],
            marker="o", label="unified"
        )
        axis.set(xlabel="training step", title=title)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[2].plot(
        steps,
        [generation[step]["mean_feature_diversity_ratio"] for step in steps],
        marker="o",
        label="generation-only",
    )
    axes[2].plot(
        steps,
        [
            np.mean([
                value["feature_diversity_ratio"]
                for value in unified[step]["classes"].values()
            ])
            for step in steps
        ],
        marker="o",
        label="unified",
    )
    axes[2].set(xlabel="training step", title="Feature diversity ratio")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output / "unified-vs-generation-only-metrics.png", dpi=180)
    plt.close(figure)

    final_step = max(steps)
    paths = [
        unified_directory / f"step-{final_step:07d}" / "generated-grid.png",
        generation_directory / f"step-{final_step:07d}" / "generated-grid.png",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for axis, path, title in zip(
        axes, paths, ("Unified generation + classification", "Generation only")
    ):
        axis.imshow(Image.open(path))
        axis.set_title(f"{title}\nstep {final_step:,}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Same class-Gaussian construction and fixed latent seed")
    figure.tight_layout()
    figure.savefig(output / "unified-vs-generation-only-75k.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
