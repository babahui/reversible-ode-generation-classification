#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from unified_transport.checkpoint import load_for_inference
from unified_transport.data import load_image
from unified_transport.interpolant import TransportSampler
from unified_transport.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or transport with one model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--seed", type=int, default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    class_selector = generate.add_mutually_exclusive_group(required=True)
    class_selector.add_argument(
        "--class-label", type=int, help="Original dataset label, for example MNIST digit 2"
    )
    class_selector.add_argument(
        "--class-id", type=int, help="Zero-based latent component index"
    )
    generate.add_argument("--num", type=int, default=16)
    generate.add_argument("--output", required=True)

    transport = subparsers.add_parser("transport")
    transport.add_argument("--input", required=True)
    transport.add_argument("--source", type=int, required=True)
    transport.add_argument("--target", type=int, required=True)
    transport.add_argument("--output", required=True)

    classify = subparsers.add_parser("classify")
    classify.add_argument("--input", nargs="+", required=True)
    classify.add_argument("--save-latent", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, latent, checkpoint = load_for_inference(args.checkpoint, device)
    config = checkpoint["config"]
    sampler = TransportSampler(model)

    if args.command == "generate":
        if latent is None:
            raise ValueError("generate requires an MNIST/CIFAR image-noise checkpoint")
        class_values = config.get("classes", list(range(latent.num_classes)))
        if args.class_label is not None:
            if args.class_label not in class_values:
                raise ValueError(f"class-label must be one of {class_values}")
            local_class = class_values.index(args.class_label)
        else:
            local_class = args.class_id
        if not 0 <= local_class < latent.num_classes:
            raise ValueError("class-id is out of range")
        labels = torch.full((args.num,), local_class, device=device, dtype=torch.long)
        initial = latent.sample(labels)
        result = sampler.integrate(
            initial, source=1, target=0, steps=args.steps, method=args.method
        )
    elif args.command == "transport":
        if latent is not None:
            raise ValueError("use classify/generate for an image-noise checkpoint")
        initial = load_image(
            args.input, config["image_size"], config["image_channels"], device
        )
        result = sampler.integrate(
            initial,
            source=args.source,
            target=args.target,
            steps=args.steps,
            method=args.method,
        )
    else:
        if latent is None:
            raise ValueError("classify requires an MNIST/CIFAR image-noise checkpoint")
        images = torch.cat(
            [
                load_image(path, config["image_size"], config["image_channels"], device)
                for path in args.input
            ],
            dim=0,
        )
        encoded = sampler.integrate(
            images, source=0, target=1, steps=args.steps, method=args.method
        )
        posterior = latent.log_posterior(encoded).softmax(dim=1)
        local_predictions = posterior.argmax(dim=1).tolist()
        class_values = config.get("classes", list(range(latent.num_classes)))
        records = []
        for path, local_prediction, probabilities in zip(
            args.input, local_predictions, posterior.tolist()
        ):
            records.append(
                {
                    "input": str(Path(path).resolve()),
                    "prediction": class_values[local_prediction],
                    "posterior": {
                        str(label): probability
                        for label, probability in zip(class_values, probabilities)
                    },
                }
            )
        if args.save_latent:
            destination = Path(args.save_latent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoded.cpu(), destination)
        print(json.dumps(records, indent=2))
        return

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(result.clamp(-1, 1).add(1).div(2), destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
