#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from unified_transport.checkpoint import ExponentialMovingAverage, build_model, save_checkpoint
from unified_transport.data import FilteredVisionDataset
from unified_transport.label_conditioned import LabelConditionalFlowMatching
from unified_transport.utils import resolve_device, set_seed, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train standard-noise label-conditioned CIFAR flow.")
    parser.add_argument("--mode", choices=("cifar10", "tinyimagenet", "stl10"), default="cifar10")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
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
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    classes = [int(item.strip()) for item in args.classes.split(",")]
    if classes != list(range(len(classes))):
        raise ValueError("this baseline expects contiguous local labels 0..num_classes-1")
    config = {
        "task": "label_conditioned_generation",
        "objective": "conditional_flow_matching",
        "mode": args.mode,
        "data_root": str(Path(args.data_root).resolve()),
        "image_size": args.image_size or {"cifar10": 32, "tinyimagenet": 64, "stl10": 96}[args.mode],
        "image_channels": 3,
        "num_marginals": 2,
        "classes": classes,
        "num_classes": len(classes),
        "class_conditioning": True,
        "base_channels": args.base_channels,
        "channel_mults": [int(item) for item in args.channel_mults.split(",")],
        "use_history": False,
        "condition_on_alpha": True,
        "random_crop": args.mode in {"cifar10", "stl10"},
        "data_noise": args.data_noise,
        "seed": args.seed,
        "ema_warmup": args.ema_warmup,
    }
    start_step = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        saved_config = resume_checkpoint.get("config", {})
        if saved_config.get("task") != "label_conditioned_generation":
            raise ValueError("resume checkpoint is not label-conditioned generation")
        config = saved_config
        start_step = int(resume_checkpoint.get("step", 0))
        if start_step >= args.train_steps:
            raise ValueError("resume checkpoint step must be less than train_steps")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(str(output / "config.json"), config)
    dataset = FilteredVisionDataset(
        args.mode, args.data_root, True, config["image_size"], classes,
        download=args.download, random_crop=config["random_crop"],
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(config).to(device)
    objective = LabelConditionalFlowMatching(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    ema = ExponentialMovingAverage(model, args.ema_decay, args.ema_warmup)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        ema.load_state_dict(resume_checkpoint["ema"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if resume_checkpoint.get("scaler") is not None:
            scaler.load_state_dict(resume_checkpoint["scaler"])
    iterator = iter(loader)
    log_path = output / "train.jsonl"
    progress = tqdm(range(start_step, args.train_steps), total=args.train_steps, initial=start_step)
    for step_index in progress:
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        noise = torch.randn_like(images)
        if args.data_noise > 0:
            images = (images + torch.empty_like(images).uniform_(-args.data_noise, args.data_noise)).clamp(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss, metrics = objective(images, labels, noise)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        step = step_index + 1
        if step % args.log_every == 0 or step == 1:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **metrics}) + "\n")
            progress.set_postfix(loss=f"{metrics['loss']:.5f}")
        if step % args.save_every == 0:
            for name in (f"model-{step}.pt", "model-latest.pt"):
                save_checkpoint(str(output / name), config, model, ema, optimizer, step, None, scaler)
    save_checkpoint(str(output / "model-latest.pt"), config, model, ema, optimizer, args.train_steps, None, scaler)


if __name__ == "__main__":
    main()
