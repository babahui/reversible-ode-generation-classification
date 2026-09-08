#!/usr/bin/env python3
"""Evaluate unpaired generation quality with FID/KID/IS and SSIM diagnostics."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch_msssim import ssim
import torch_fidelity

from unified_transport.checkpoint import load_for_inference
from unified_transport.data import FilteredVisionDataset
from unified_transport.generation_only import GenerationOnlySampler
from unified_transport.interpolant import TransportSampler
from unified_transport.label_conditioned import LabelConditionalSampler
from unified_transport.utils import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-per-class", type=int, default=500)
    parser.add_argument("--real-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--ssim-per-class", type=int, default=32)
    parser.add_argument("--ssim-real-candidates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def save_tensor_images(images, directory: Path, offset: int):
    directory.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu().clamp(-1, 1).add(1).div(2)
    for index, image in enumerate(images):
        save_image(image, directory / f"{offset + index:07d}.png")


@torch.inference_mode()
def generate(checkpoint_path, per_class, batch_size, steps, method, seed, device, directory):
    model, latent, checkpoint = load_for_inference(checkpoint_path, device)
    config = checkpoint["config"]
    task = config.get("task", "unified")
    shape = (config["image_channels"], config["image_size"], config["image_size"])
    set_seed(seed)
    samples = []
    labels_out = []
    offset = 0
    for label in range(config["num_classes"]):
        remaining = per_class
        while remaining:
            count = min(batch_size, remaining)
            labels = torch.full((count,), label, dtype=torch.long, device=device)
            if task == "generation_only":
                initial = latent.sample(labels)
                images = GenerationOnlySampler(model).integrate(initial, steps, method)
            elif task == "label_conditioned_generation":
                initial = torch.randn(count, *shape, device=device)
                images = LabelConditionalSampler(model).integrate(
                    initial, labels, steps, method
                )
            else:
                initial = latent.sample(labels)
                images = TransportSampler(model).integrate(
                    initial, 1, 0, steps, method
                )
            images = images.clamp(-1, 1)
            save_tensor_images(images, directory, offset)
            samples.append(images.cpu())
            labels_out.append(labels.cpu())
            offset += count
            remaining -= count
    return torch.cat(samples), torch.cat(labels_out), checkpoint


def collect_real(dataset, maximum, num_classes, directory, workers):
    images_out, labels_out = [], []
    target_per_class = maximum // num_classes
    counts = torch.zeros(num_classes, dtype=torch.long)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=workers)
    for images, labels in loader:
        for image, label in zip(images, labels):
            label_index = int(label)
            if counts[label_index] >= target_per_class:
                continue
            images_out.append(image)
            labels_out.append(label)
            counts[label_index] += 1
        if bool((counts >= target_per_class).all()):
            break
    images = torch.stack(images_out)
    labels = torch.stack(labels_out)
    save_tensor_images(images, directory, 0)
    return images, labels


def paired_ssim(images, labels, num_classes, per_class):
    values = []
    for label in range(num_classes):
        subset = images[labels == label][: per_class * 2].float().add(1).div(2)
        count = subset.shape[0] // 2
        if count:
            values.append(ssim(
                subset[:count], subset[count:count * 2], data_range=1.0,
                size_average=False,
            ))
    return float(torch.cat(values).mean())


def nearest_train_ssim(generated, generated_labels, train_set, num_classes, probes, candidates):
    train_images = [[] for _ in range(num_classes)]
    for image, label in train_set:
        if len(train_images[label]) < candidates:
            train_images[label].append(image)
        if all(len(items) >= candidates for items in train_images):
            break
    maxima = []
    for label in range(num_classes):
        queries = generated[generated_labels == label][:probes].float().add(1).div(2)
        references = torch.stack(train_images[label]).float().add(1).div(2)
        for query in queries:
            scores = []
            for chunk in references.split(32):
                repeated = query.unsqueeze(0).expand(chunk.shape[0], -1, -1, -1)
                scores.append(ssim(repeated, chunk, data_range=1.0, size_average=False))
            maxima.append(torch.cat(scores).max())
    return float(torch.stack(maxima).mean())


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output = Path(args.output)
    generated_dir = output / "generated"
    real_dir = output / "real"
    output.mkdir(parents=True, exist_ok=True)
    for directory in (generated_dir, real_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for old_image in directory.glob("*.png"):
            old_image.unlink()
    generated, generated_labels, checkpoint = generate(
        args.checkpoint, args.generated_per_class, args.batch_size, args.steps,
        args.method, args.seed, device, generated_dir,
    )
    config = checkpoint["config"]
    test_set = FilteredVisionDataset(
        config["mode"], args.data_root, False, config["image_size"], config["classes"]
    )
    train_set = FilteredVisionDataset(
        config["mode"], args.data_root, True, config["image_size"], config["classes"]
    )
    requested_real = min(args.real_images, len(test_set))
    real, real_labels = collect_real(
        test_set, requested_real, config["num_classes"], real_dir, args.workers
    )
    kid_subset_size = min(1000, generated.shape[0], real.shape[0])
    fidelity = torch_fidelity.calculate_metrics(
        input1=str(generated_dir), input2=str(real_dir), cuda=device.type == "cuda",
        isc=True, fid=True, kid=True, kid_subset_size=kid_subset_size,
        kid_subsets=100, verbose=False,
    )
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": int(checkpoint["step"]),
        "mode": config["mode"],
        "task": config.get("task", "unified"),
        "generated_images": int(generated.shape[0]),
        "real_test_images": int(real.shape[0]),
        "fid_inception": float(fidelity["frechet_inception_distance"]),
        "kid_mean": float(fidelity["kernel_inception_distance_mean"]),
        "kid_std": float(fidelity["kernel_inception_distance_std"]),
        "inception_score_mean": float(fidelity["inception_score_mean"]),
        "inception_score_std": float(fidelity["inception_score_std"]),
        "generated_pair_ssim": paired_ssim(
            generated, generated_labels, config["num_classes"], args.ssim_per_class
        ),
        "real_pair_ssim": paired_ssim(
            real, real_labels, config["num_classes"], args.ssim_per_class
        ),
        "nearest_train_ssim": nearest_train_ssim(
            generated, generated_labels, train_set, config["num_classes"],
            args.ssim_per_class, args.ssim_real_candidates,
        ),
        "ssim_interpretation": (
            "Pair SSIM is a diversity diagnostic (lower means more varied). "
            "Nearest-train SSIM is a resemblance/memorization diagnostic, not an unpaired quality score."
        ),
    }
    (output / "quality-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
