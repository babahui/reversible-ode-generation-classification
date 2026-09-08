#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from unified_transport.checkpoint import (
    ExponentialMovingAverage,
    build_latent,
    build_model,
    save_checkpoint,
)
from unified_transport.data import FilteredVisionDataset, VisionLatentDataset
from unified_transport.generation_only import ConditionalFlowMatching
from unified_transport.ot_pairing import pair_latents_by_class
from unified_transport.utils import resolve_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a generation-only conditional flow from Gaussian classes."
    )
    parser.add_argument("--mode", choices=("cifar10", "tinyimagenet", "stl10"), default="cifar10")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--center-scale", type=float, default=12.0)
    parser.add_argument(
        "--center-mode",
        choices=("random", "low_frequency", "coded_low_frequency"),
        default="random",
    )
    parser.add_argument("--latent-sigma", type=float, default=0.5)
    parser.add_argument(
        "--pairing", choices=("independent", "sinkhorn"), default="independent",
        help="How to couple same-class Gaussian samples and images within each batch.",
    )
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-iterations", type=int, default=50)
    parser.add_argument(
        "--path-diagnostics-every", type=int, default=1000,
        help="Log local velocity/cosine diagnostics every N steps; 0 disables them.",
    )
    parser.add_argument(
        "--center-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary endpoint center-projection loss; keeps labels out of the model input.",
    )
    parser.add_argument(
        "--ode-class-weight",
        type=float,
        default=0.0,
        help="Auxiliary differentiable image-to-noise center-projection loss.",
    )
    parser.add_argument("--ode-class-steps", type=int, default=5)
    parser.add_argument("--ode-class-batch", type=int, default=64)
    parser.add_argument(
        "--ode-class-loss", choices=("projection", "ce"), default="projection",
        help="Reverse endpoint objective: center-coordinate regression or Gaussian posterior CE.",
    )
    parser.add_argument("--ode-class-temperature", type=float, default=1.0)
    parser.add_argument(
        "--ode-class-every", type=int, default=1,
        help="Compute the differentiable reverse loss every N optimizer steps.",
    )
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", default="1,2,4")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=75000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-warmup", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--data-noise", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--initialize-from",
        default="",
        help="Load model/EMA/latent weights from a compatible checkpoint and start a new optimizer run.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-random-crop", action="store_true")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--direction-adapter",
        action="store_true",
        help="Condition the Markov flow on the previous physical velocity via a zero-initialized adapter.",
    )
    parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="When initializing from a checkpoint, update only direction-adapter parameters.",
    )
    parser.add_argument("--direction-dropout", type=float, default=0.10)
    parser.add_argument("--direction-noise", type=float, default=0.02)
    return parser.parse_args()


def make_config(args: argparse.Namespace):
    classes = [int(item.strip()) for item in args.classes.split(",")]
    if len(set(classes)) != len(classes) or not classes:
        raise ValueError("--classes must be non-empty and contain no duplicates")
    return {
        "task": "generation_only",
        "objective": "conditional_flow_matching",
        "mode": args.mode,
        "data_root": str(Path(args.data_root).resolve()),
        "image_size": args.image_size or {"cifar10": 32, "tinyimagenet": 64, "stl10": 96}[args.mode],
        "image_channels": 3,
        "num_marginals": 2,
        "classes": classes,
        "num_classes": len(classes),
        "center_scale": args.center_scale,
        "center_mode": args.center_mode,
        "latent_sigma": args.latent_sigma,
        "pairing": args.pairing,
        "ot_epsilon": args.ot_epsilon,
        "ot_iterations": args.ot_iterations,
        "path_diagnostics_every": args.path_diagnostics_every,
        "center_loss_weight": args.center_loss_weight,
        "ode_class_weight": args.ode_class_weight,
        "ode_class_steps": args.ode_class_steps,
        "ode_class_batch": args.ode_class_batch,
        "ode_class_loss": args.ode_class_loss,
        "ode_class_temperature": args.ode_class_temperature,
        "ode_class_every": args.ode_class_every,
        "base_channels": args.base_channels,
        "channel_mults": [int(item) for item in args.channel_mults.split(",")],
        "use_history": False,
        "direction_adapter": args.direction_adapter,
        "direction_dropout": args.direction_dropout if args.direction_adapter else 0.0,
        "direction_noise": args.direction_noise if args.direction_adapter else 0.0,
        "condition_on_alpha": True,
        "random_crop": not args.no_random_crop,
        "data_noise": args.data_noise,
        "seed": args.seed,
        "ema_warmup": args.ema_warmup,
    }


def main() -> None:
    args = parse_args()
    if args.ode_class_every < 1:
        raise ValueError("--ode-class-every must be positive")
    config = make_config(args)
    if args.adapter_only and not args.direction_adapter:
        raise ValueError("--adapter-only requires --direction-adapter")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(str(output / "config.json"), config)

    latent = build_latent(config)
    images = FilteredVisionDataset(
        args.mode,
        args.data_root,
        train=True,
        image_size=config["image_size"],
        classes=config["classes"],
        download=args.download,
        random_crop=config["random_crop"],
    )
    priors = images.class_counts.float()
    latent.log_priors.copy_((priors / priors.sum()).log())
    if args.pairing == "sinkhorn":
        # The OT sampler must see the whole class-wise batch.  Keep the
        # dataset CPU-only for worker safety, and move this separate latent
        # module to the training device after DataLoader construction.
        latent = latent.to(device)
        train_dataset = images
    else:
        train_dataset = VisionLatentDataset(images, latent)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(config).to(device)
    # Independent-pairing DataLoader workers need a CPU latent sampler, while
    # auxiliary reverse losses compare GPU tensors against the centers.
    objective_latent = latent
    if args.pairing != "sinkhorn" and (
        args.center_loss_weight > 0 or args.ode_class_weight > 0
    ):
        objective_latent = copy.deepcopy(latent).to(device)
    objective = ConditionalFlowMatching(
        model,
        latent=objective_latent,
        center_loss_weight=args.center_loss_weight,
        ode_class_weight=args.ode_class_weight,
        ode_class_steps=args.ode_class_steps,
        ode_class_batch=args.ode_class_batch,
        ode_class_loss=args.ode_class_loss,
        ode_class_temperature=args.ode_class_temperature,
        direction_dropout=args.direction_dropout if args.direction_adapter else 0.0,
        direction_noise=args.direction_noise if args.direction_adapter else 0.0,
    )
    if args.adapter_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("direction_adapter_net."))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("direction adapter has no trainable parameters")
    else:
        trainable = list(model.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    ema = ExponentialMovingAverage(
        model, decay=args.ema_decay, warmup_steps=args.ema_warmup
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_step = 0
    if args.initialize_from and args.resume:
        raise ValueError("use only one of --initialize-from and --resume")
    if args.initialize_from:
        initialization = torch.load(args.initialize_from, map_location=device)
        source_config = initialization["config"]
        # Objective weights and pairing may change during fine-tuning, but
        # tensor shapes, latent geometry, and model conditioning must match.
        compatibility_keys = (
            "task", "objective", "mode", "image_size", "image_channels",
            "num_marginals", "classes", "num_classes", "center_scale",
            "center_mode", "latent_sigma", "base_channels", "channel_mults",
            "use_history", "condition_on_alpha",
        )
        mismatches = [
            key for key in compatibility_keys
            if source_config.get(key) != config.get(key)
        ]
        if mismatches:
            raise ValueError(
                "initialization checkpoint is incompatible for: "
                + ", ".join(mismatches)
            )
        source_direction_adapter = bool(source_config.get("direction_adapter", False))
        if source_direction_adapter and not args.direction_adapter:
            raise ValueError("cannot initialize a Markov model from a direction-adapter checkpoint")
        model.load_state_dict(initialization["model"], strict=False)
        ema.model.load_state_dict(initialization["ema"]["model"], strict=False)
        latent.load_state_dict(initialization["latent"])
        if objective_latent is not latent:
            objective_latent.load_state_dict(initialization["latent"])
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        saved_config = dict(checkpoint["config"])
        # Checkpoints written before the endpoint-loss option do not contain
        # its zero-valued default; keep those runs resumable.
        saved_config.setdefault("center_loss_weight", 0.0)
        saved_config.setdefault("ode_class_weight", 0.0)
        saved_config.setdefault("ode_class_steps", 5)
        saved_config.setdefault("ode_class_batch", 64)
        saved_config.setdefault("ode_class_loss", "projection")
        saved_config.setdefault("ode_class_temperature", 1.0)
        saved_config.setdefault("ode_class_every", 1)
        saved_config.setdefault("pairing", "independent")
        saved_config.setdefault("ot_epsilon", 0.05)
        saved_config.setdefault("ot_iterations", 50)
        saved_config.setdefault("path_diagnostics_every", 1000)
        saved_config.setdefault("direction_adapter", False)
        saved_config.setdefault("direction_dropout", 0.0)
        saved_config.setdefault("direction_noise", 0.0)
        if saved_config != config:
            raise ValueError("resume checkpoint config does not match")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        latent.load_state_dict(checkpoint["latent"])
        if objective_latent is not latent:
            objective_latent.load_state_dict(checkpoint["latent"])
        start_step = int(checkpoint["step"])

    iterator = iter(loader)
    log_path = output / "train.jsonl"
    progress = tqdm(
        range(start_step, args.train_steps), initial=start_step, total=args.train_steps
    )
    for step_index in progress:
        completed_step = step_index + 1
        try:
            batch, labels_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch, labels_batch = next(iterator)
        labels_batch = labels_batch.to(device, non_blocking=True).long()
        if args.pairing == "sinkhorn":
            images_batch = batch.to(device, non_blocking=True)
            noise_batch, pairing_metrics = pair_latents_by_class(
                images_batch,
                labels_batch,
                latent,
                epsilon=args.ot_epsilon,
                iterations=args.ot_iterations,
            )
        else:
            marginals = batch.to(device, non_blocking=True)
            images_batch = marginals[:, 0]
            noise_batch = marginals[:, 1]
            pairing_metrics = {}
        if args.data_noise > 0:
            images_batch = (
                images_batch
                + torch.empty_like(images_batch).uniform_(
                    -args.data_noise, args.data_noise
                )
            ).clamp(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss, metrics = objective(
                images_batch,
                noise_batch,
                labels_batch,
                compute_ode_class=(
                    args.ode_class_weight > 0
                    and completed_step % args.ode_class_every == 0
                ),
                diagnostics=(
                    args.path_diagnostics_every > 0
                    and completed_step % args.path_diagnostics_every == 0
                ),
            )
            # The reverse solve is intentionally sparse when requested.
            if "ode_class_loss" not in metrics:
                metrics["ode_class_loss"] = 0.0
        metrics.update(pairing_metrics)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        if completed_step % args.log_every == 0 or completed_step == 1:
            record = {"step": completed_step, **metrics}
            progress.set_postfix(loss=f"{metrics['loss']:.5f}")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        if completed_step % args.save_every == 0:
            for name in (f"model-{completed_step}.pt", "model-latest.pt"):
                save_checkpoint(
                    str(output / name), config, model, ema, optimizer,
                    completed_step, latent, scaler
                )
    save_checkpoint(
        str(output / "model-latest.pt"), config, model, ema, optimizer,
        args.train_steps, latent, scaler
    )


if __name__ == "__main__":
    main()
