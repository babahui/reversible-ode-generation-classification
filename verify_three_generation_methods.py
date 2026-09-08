#!/usr/bin/env python3
"""Audit that the three generation evaluations are not reusing image tensors."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified", required=True)
    parser.add_argument("--gaussian", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "unified": Path(args.unified),
        "gaussian_noise_only": Path(args.gaussian),
        "label_conditioned": Path(args.label),
    }
    images = {
        name: np.asarray(Image.open(path)).astype(np.float32) / 255.0
        for name, path in paths.items()
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    names = list(images)
    verification = {"image_shape": list(next(iter(images.values())).shape), "pairwise": {}}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            difference = np.abs(images[left] - images[right])
            verification["pairwise"][f"{left}_vs_{right}"] = {
                "mae": float(difference.mean()),
                "mse": float(np.square(difference).mean()),
                "exact_pixel_fraction": float(np.mean(images[left] == images[right])),
            }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, name in enumerate(names):
        axes[0, column].imshow(images[name])
        axes[0, column].set_title(name.replace("_", " "))
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
    pairs = [(names[0], names[1]), (names[0], names[2]), (names[1], names[2])]
    for axis, (left, right) in zip(axes[1], pairs):
        difference = np.abs(images[left] - images[right])
        axis.imshow(difference / max(float(difference.max()), 1e-8))
        axis.set_title(f"|{left} - {right}|")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Three generation methods: same fixed-seed grid and pixel differences")
    figure.tight_layout()
    figure.savefig(output / "three-method-verification.png", dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
