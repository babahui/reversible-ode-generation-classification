#!/usr/bin/env python3
"""Download or verify the CIFAR-10 layout expected by this repository."""

import argparse
import json
from pathlib import Path

from torchvision.datasets import CIFAR10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate CIFAR-10 through torchvision."
    )
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[2] / "all_datasets"),
        help="Directory containing cifar-10-batches-py (default: ../all_datasets)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Require an existing valid dataset and never access the network.",
    )
    return parser.parse_args()


def load_split(root: Path, train: bool, download: bool) -> CIFAR10:
    split = "train" if train else "test"
    try:
        return CIFAR10(root=str(root), train=train, download=download)
    except RuntimeError as error:
        action = "verify" if not download else "download or verify"
        raise RuntimeError(
            f"Could not {action} the CIFAR-10 {split} split under {root}: {error}"
        ) from error


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    train = load_split(root, train=True, download=not args.verify_only)
    test = load_split(root, train=False, download=not args.verify_only)
    expected = {"train_images": 50000, "test_images": 10000, "classes": 10}
    actual = {
        "train_images": len(train),
        "test_images": len(test),
        "classes": len(train.classes),
    }
    if actual != expected:
        raise RuntimeError(f"Unexpected CIFAR-10 metadata: expected {expected}, got {actual}")
    payload = {
        "status": "ready",
        "data_root": str(root),
        "extracted_directory": str(root / CIFAR10.base_folder),
        **actual,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
