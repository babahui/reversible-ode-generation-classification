#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from unified_transport.checkpoint import load_for_inference
from unified_transport.label_conditioned import LabelConditionalSampler
from unified_transport.utils import resolve_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Sample label-conditioned CIFAR flow.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, _, checkpoint = load_for_inference(args.checkpoint, device)
    if checkpoint["config"].get("task") != "label_conditioned_generation":
        raise ValueError("checkpoint is not label-conditioned generation")
    config = checkpoint["config"]
    classes = config["classes"]
    labels = torch.arange(len(classes), device=device).repeat_interleave(args.per_class)
    generated = LabelConditionalSampler(model).integrate(
        torch.randn(
            labels.shape[0], config["image_channels"],
            config["image_size"], config["image_size"], device=device,
        ),
        labels, args.steps, args.method,
    ).clamp(-1, 1)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(generated.add(1).div(2), destination, nrow=args.per_class)


if __name__ == "__main__":
    main()
