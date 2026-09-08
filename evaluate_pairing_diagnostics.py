#!/usr/bin/env python3
"""Compare path ambiguity for independent and class-wise OT couplings."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from unified_transport.checkpoint import load_for_inference
from unified_transport.data import FilteredVisionDataset
from unified_transport.ot_pairing import pair_latents_by_class, path_diagnostics
from unified_transport.utils import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-iterations", type=int, default=50)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, latent, checkpoint = load_for_inference(args.checkpoint, device)
    config = checkpoint["config"]
    if config.get("task") != "generation_only" or latent is None:
        raise ValueError("checkpoint must be a generation-only image/Gaussian model")
    dataset = FilteredVisionDataset(
        config["mode"], args.data_root, True, config["image_size"], config["classes"],
        random_crop=config.get("random_crop", True),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    latent = latent.to(device)
    accum = {name: {} for name in ("independent", "sinkhorn")}
    pairing_costs = {"independent": [], "sinkhorn": []}
    iterator = iter(loader)
    for _ in range(args.batches):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        time = torch.rand(images.shape[0], device=device, dtype=images.dtype)
        for method in ("independent", "sinkhorn"):
            if method == "independent":
                noise = latent.sample(labels)
            else:
                noise, pairing = pair_latents_by_class(
                    images, labels, latent,
                    epsilon=args.ot_epsilon, iterations=args.ot_iterations,
                )
                pairing_costs["sinkhorn"].append(pairing["ot_pair_cost"])
            pairing_costs[method].append(float((images - noise).square().mean()))
            time_view = time[:, None, None, None]
            state = (1 - time_view) * noise + time_view * images
            alpha = torch.stack((time, 1 - time), dim=1)
            predicted = model(state, alpha)
            result = path_diagnostics(
                state, images - noise, predicted, time,
                bins=10, neighbors=args.neighbors,
            )["path_diagnostics"]
            for row in result:
                index = int(round(row["t_start"] * 10))
                bucket = accum[method].setdefault(index, {"count": 0})
                count = row["count"]
                old_count = bucket["count"]
                new_count = old_count + count
                for key in (
                    "velocity_direction_cosine",
                    "local_velocity_variance",
                    "near_neighbor_direction_conflict",
                ):
                    bucket[key] = (
                        bucket.get(key, 0.0) * old_count + row[key] * count
                    ) / new_count
                bucket["count"] = new_count
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "ot_epsilon": args.ot_epsilon,
        "ot_iterations": args.ot_iterations,
        "methods": {
            method: [
                {"t_start": index / 10, "t_end": (index + 1) / 10, **values}
                for index, values in sorted(rows.items())
            ]
            for method, rows in accum.items()
        },
        "mean_pair_costs": {
            method: sum(values) / max(len(values), 1)
            for method, values in pairing_costs.items()
        },
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    fields = [
        ("local_velocity_variance", "Local velocity variance"),
        ("velocity_direction_cosine", "Predicted/target direction cosine"),
        ("near_neighbor_direction_conflict", "Near-neighbor direction conflict"),
    ]
    colors = {"independent": "#b26a35", "sinkhorn": "#3c8c70"}
    for axis, (field, title) in zip(axes, fields):
        for method, color in colors.items():
            rows = payload["methods"][method]
            axis.plot(
                [row["t_start"] for row in rows],
                [row[field] for row in rows],
                marker="o", label=method, color=color,
            )
        axis.set_title(title)
        axis.set_xlabel("t bin start")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("value")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output / "metrics.png", dpi=180)
    plt.close(figure)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
