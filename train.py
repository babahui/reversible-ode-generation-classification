#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from unified_transport.checkpoint import (
    ExponentialMovingAverage,
    build_latent,
    build_model,
    save_checkpoint,
)
from unified_transport.data import (
    FilteredVisionDataset,
    FolderMarginalsDataset,
    VisionLatentDataset,
)
from unified_transport.interpolant import (
    MultimarginalInterpolant,
    differentiable_edge_integrate,
)
from unified_transport.utils import resolve_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multimarginal stochastic-interpolant model."
    )
    parser.add_argument(
        "--mode", choices=("mnist", "cifar10", "tinyimagenet", "stl10", "folders"), required=True
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--domains", default="", help="Comma-separated subfolders for --mode folders"
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--random-crop",
        action="store_true",
        help="Use CIFAR-style 32x32 random crop with four-pixel padding",
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument(
        "--image-channels", type=int, default=0, help="0 selects 1 for MNIST and 3 otherwise"
    )
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--classes",
        default="",
        help="Original MNIST/CIFAR labels, for example 1,2,3 (default: 0..num-classes-1)",
    )
    parser.add_argument("--center-scale", type=float, default=4.0)
    parser.add_argument("--latent-sigma", type=float, default=0.5)
    parser.add_argument(
        "--endpoint-class-weight",
        type=float,
        default=0.0,
        help="Weight for image-endpoint class-center projection alignment",
    )
    parser.add_argument(
        "--ode-class-weight",
        type=float,
        default=0.0,
        help="Weight for class-center alignment after a short differentiable image-to-noise solve",
    )
    parser.add_argument("--ode-class-steps", type=int, default=5)
    parser.add_argument("--ode-class-batch", type=int, default=64)
    parser.add_argument(
        "--data-noise",
        type=float,
        default=0.01,
        help="Uniform dequantization half-width on the image marginal",
    )
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", default="1,2,4")
    history = parser.add_mutually_exclusive_group()
    history.add_argument(
        "--history",
        dest="use_history",
        action="store_true",
        help="Experimental non-Markov path memory; disables strict ODE inversion",
    )
    history.add_argument(
        "--no-history", dest="use_history", action="store_false", help=argparse.SUPPRESS
    )
    parser.set_defaults(use_history=False)
    parser.add_argument(
        "--no-alpha-condition",
        dest="condition_on_alpha",
        action="store_false",
        help="Ablation: do not expose the simplex coordinate to the UNet",
    )
    parser.set_defaults(condition_on_alpha=True)
    parser.add_argument("--history-step", type=float, default=0.02)
    parser.add_argument("--history-dropout", type=float, default=0.1)
    parser.add_argument("--history-noise", type=float, default=0.01)
    parser.add_argument("--simplex-probability", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=300000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-warmup", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--initialize-from",
        default="",
        help="Initialize model and latent from a checkpoint EMA without restoring optimizer/step",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Dict[str, Any]:
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    classes = (
        [int(item.strip()) for item in args.classes.split(",") if item.strip()]
        if args.classes
        else list(range(args.num_classes))
    )
    if args.mode == "folders" and len(domains) < 2:
        raise ValueError("--mode folders requires at least two comma-separated --domains")
    image_channels = args.image_channels or (1 if args.mode == "mnist" else 3)
    if args.mode in {"cifar10", "tinyimagenet", "stl10"} and image_channels != 3:
        raise ValueError("CIFAR-10 requires --image-channels 3")
    if args.mode == "mnist" and image_channels != 1:
        raise ValueError("MNIST requires --image-channels 1")
    if args.random_crop and args.mode not in {"cifar10", "tinyimagenet", "stl10"}:
        raise ValueError("--random-crop is only supported for RGB vision datasets")
    if args.mode in {"mnist", "cifar10", "tinyimagenet", "stl10"} and len(set(classes)) != len(classes):
        raise ValueError("--classes must not contain duplicates")
    if args.endpoint_class_weight < 0 or args.ode_class_weight < 0:
        raise ValueError("class-alignment weights must be non-negative")
    if (args.endpoint_class_weight > 0 or args.ode_class_weight > 0) and args.mode not in {"mnist", "cifar10"}:
        raise ValueError("class-alignment losses require MNIST or CIFAR-10 mode")
    if args.ode_class_weight > 0 and args.use_history:
        raise ValueError("--ode-class-weight requires a Markov model without history")
    if args.ode_class_steps < 1 or args.ode_class_batch < 1:
        raise ValueError("ODE class steps and batch must be positive")
    if args.resume and args.initialize_from:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    return {
        "mode": args.mode,
        "data_root": str(Path(args.data_root).resolve()),
        "domains": domains,
        "image_size": args.image_size,
        "image_channels": image_channels,
        "num_marginals": 2 if args.mode in {"mnist", "cifar10", "tinyimagenet", "stl10"} else len(domains),
        "classes": classes if args.mode in {"mnist", "cifar10", "tinyimagenet", "stl10"} else [],
        "num_classes": len(classes),
        "center_scale": args.center_scale,
        "latent_sigma": args.latent_sigma,
        "endpoint_class_weight": args.endpoint_class_weight,
        "ode_class_weight": args.ode_class_weight,
        "ode_class_steps": args.ode_class_steps,
        "ode_class_batch": args.ode_class_batch,
        "data_noise": args.data_noise,
        "random_crop": args.random_crop,
        "base_channels": args.base_channels,
        "channel_mults": [int(item) for item in args.channel_mults.split(",")],
        "use_history": args.use_history,
        "condition_on_alpha": args.condition_on_alpha,
        "history_step": args.history_step,
        "history_dropout": args.history_dropout,
        "history_noise": args.history_noise,
        "simplex_probability": args.simplex_probability,
        "seed": args.seed,
        "ema_warmup": args.ema_warmup,
        "initialized_from": str(Path(args.initialize_from).resolve()) if args.initialize_from else "",
    }


def main() -> None:
    args = parse_args()
    config = make_config(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(str(output / "config.json"), config)

    latent = build_latent(config)
    if args.mode in {"mnist", "cifar10", "tinyimagenet", "stl10"}:
        images = FilteredVisionDataset(
            args.mode,
            args.data_root,
            train=True,
            image_size=args.image_size,
            classes=config["classes"],
            download=args.download,
            random_crop=args.random_crop,
        )
        empirical_priors = images.class_counts.float()
        empirical_priors /= empirical_priors.sum()
        latent.log_priors.copy_(empirical_priors.log())
        dataset = VisionLatentDataset(images, latent)
    else:
        dataset = FolderMarginalsDataset(
            args.data_root, config["domains"], args.image_size, train=True
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = build_model(config).to(device)
    objective = MultimarginalInterpolant(
        model,
        history_step=args.history_step,
        history_dropout=args.history_dropout,
        history_noise=args.history_noise,
        simplex_probability=args.simplex_probability,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    ema = ExponentialMovingAverage(
        model, decay=args.ema_decay, warmup_steps=args.ema_warmup
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_step = 0

    if args.initialize_from:
        initialization = torch.load(args.initialize_from, map_location=device)
        source_config = initialization["config"]
        compatibility_keys = (
            "mode", "image_size", "image_channels", "num_marginals", "classes",
            "num_classes", "center_scale", "latent_sigma", "base_channels",
            "channel_mults", "use_history", "condition_on_alpha", "seed",
        )
        mismatches = [
            key for key in compatibility_keys
            if source_config.get(key) != config.get(key)
        ]
        if mismatches:
            raise ValueError(
                "initialization checkpoint is incompatible for: " + ", ".join(mismatches)
            )
        initialization_state = initialization["ema"]["model"]
        model.load_state_dict(initialization_state)
        ema.model.load_state_dict(initialization_state)
        if latent is not None:
            latent.load_state_dict(initialization["latent"])

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        resume_config = dict(config)
        resume_config["initialized_from"] = checkpoint["config"].get(
            "initialized_from", ""
        )
        if checkpoint["config"] != resume_config:
            raise ValueError("resume checkpoint config does not match current configuration")
        config = checkpoint["config"]
        write_json(str(output / "config.json"), config)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if latent is not None:
            latent.load_state_dict(checkpoint["latent"])
        start_step = int(checkpoint["step"])

    iterator = iter(loader)
    log_path = output / "train.jsonl"
    progress = tqdm(range(start_step, args.train_steps), initial=start_step, total=args.train_steps)
    for step_index in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        marginals = batch[0] if args.mode in {"mnist", "cifar10", "tinyimagenet", "stl10"} else batch
        labels = (
            batch[1].to(device, non_blocking=True)
            if args.mode in {"mnist", "cifar10", "tinyimagenet", "stl10"}
            else None
        )
        marginals = marginals.to(device, non_blocking=True)
        if args.mode in {"mnist", "cifar10", "tinyimagenet", "stl10"} and args.data_noise > 0:
            image_noise = torch.empty_like(marginals[:, 0]).uniform_(
                -args.data_noise, args.data_noise
            )
            marginals[:, 0] = (marginals[:, 0] + image_noise).clamp(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss, metrics = objective(marginals)
            transport_loss = loss
            if args.endpoint_class_weight > 0:
                endpoint_alpha = torch.zeros(
                    marginals.shape[0], config["num_marginals"],
                    device=device, dtype=marginals.dtype,
                )
                endpoint_alpha[:, 0] = 1
                predicted_noise = model(marginals[:, 0], endpoint_alpha)[:, 1]
                endpoint_loss = latent.endpoint_projection_loss(predicted_noise, labels)
                loss = loss + args.endpoint_class_weight * endpoint_loss
                metrics["endpoint_projection_loss"] = float(endpoint_loss.detach())
            else:
                metrics["endpoint_projection_loss"] = 0.0
            if args.ode_class_weight > 0:
                ode_batch = min(args.ode_class_batch, marginals.shape[0])
                encoded_subset = differentiable_edge_integrate(
                    model,
                    marginals[:ode_batch, 0],
                    source=0,
                    target=1,
                    steps=args.ode_class_steps,
                )
                ode_class_loss = latent.endpoint_projection_loss(
                    encoded_subset, labels[:ode_batch]
                )
                loss = loss + args.ode_class_weight * ode_class_loss
                metrics["ode_class_loss"] = float(ode_class_loss.detach())
            else:
                metrics["ode_class_loss"] = 0.0
            metrics["transport_loss"] = float(transport_loss.detach())
            metrics["loss"] = float(loss.detach())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        completed_step = step_index + 1

        if completed_step % args.log_every == 0 or completed_step == 1:
            record = {
                "step": completed_step,
                "loss": metrics["loss"],
                "history_speed": metrics["history_speed"],
                "transport_loss": metrics["transport_loss"],
                "endpoint_projection_loss": metrics["endpoint_projection_loss"],
                "ode_class_loss": metrics["ode_class_loss"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            progress.set_postfix(loss=f"{metrics['loss']:.5f}")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

        if completed_step % args.save_every == 0:
            save_checkpoint(
                str(output / f"model-{completed_step}.pt"),
                config,
                model,
                ema,
                optimizer,
                completed_step,
                latent,
                scaler,
            )
            save_checkpoint(
                str(output / "model-latest.pt"),
                config,
                model,
                ema,
                optimizer,
                completed_step,
                latent,
                scaler,
            )

    save_checkpoint(
        str(output / "model-latest.pt"),
        config,
        model,
        ema,
        optimizer,
        args.train_steps,
        latent,
        scaler,
    )


if __name__ == "__main__":
    main()
