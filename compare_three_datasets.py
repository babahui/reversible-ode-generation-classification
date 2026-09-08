#!/usr/bin/env python3
"""Aggregate three-dataset transport experiments into tables and figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parent


def _load(path):
    return json.loads(Path(path).read_text())


def _result_at(path, step=75000):
    data = _load(path)
    if "results" in data:
        rows = data["results"]
    else:
        rows = data
    if isinstance(rows, dict):
        rows = list(rows.values())
    exact = [row for row in rows if int(row.get("step", -1)) == step]
    if exact:
        return exact[-1]
    return max(rows, key=lambda row: int(row.get("step", -1)))


def _quality(path):
    if not Path(path).exists():
        return {}
    row = _load(path)
    return {
        "fid": row.get("fid_inception"),
        "kid": row.get("kid_mean"),
        "is": row.get("inception_score_mean"),
        "generated_pair_ssim": row.get("generated_pair_ssim"),
    }


def manifest():
    # CIFAR Unified uses the 75k base64 run with endpoint supervision; this is
    # the only CIFAR Unified checkpoint with the same 75k budget as the other
    # three methods.  Its status is called out in the generated README.
    return {
        "CIFAR-10": {
            "Unified": {
                "metric": ROOT / "runs/cifar10-performance-base64/evaluation-75k/metrics.json",
                "quality": ROOT / "runs/cifar10-quality/unified/quality-metrics.json",
                "image": ROOT / "runs/cifar10-performance-base64/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/cifar10-performance-base64/evaluation/metrics.json",
            },
            "Gaussian-only": {
                "metric": ROOT / "runs/cifar10-generation-only-base64/evaluation/metrics.json",
                "quality": ROOT / "runs/cifar10-quality/gaussian-only/quality-metrics.json",
                "image": ROOT / "runs/cifar10-generation-only-base64/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/cifar10-generation-only-base64/evaluation/metrics.json",
                "reverse": ROOT / "runs/cifar10-generation-only-base64/reverse-evaluation/metrics.json",
            },
            "Low-frequency Gaussian-only": {
                "metric": ROOT / "runs/cifar10-generation-only-lowfreq-base64/evaluation/metrics.json",
                "quality": ROOT / "runs/cifar10-quality/lowfreq/quality-metrics.json",
                "image": ROOT / "runs/cifar10-generation-only-lowfreq-base64/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/cifar10-generation-only-lowfreq-base64/evaluation/metrics.json",
                "reverse": ROOT / "runs/cifar10-generation-only-lowfreq-base64/reverse-evaluation/metrics.json",
            },
            "Label-conditioned": {
                "metric": ROOT / "runs/cifar10-label-conditioned-base64/evaluation/metrics.json",
                "quality": ROOT / "runs/cifar10-quality/label-conditioned/quality-metrics.json",
                "image": ROOT / "runs/cifar10-label-conditioned-base64/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/cifar10-label-conditioned-base64/evaluation/metrics.json",
            },
        },
        "Tiny-ImageNet-10": {
            "Unified": {
                "metric": ROOT / "runs/tiny-imagenet-10-unified/evaluation/metrics.json",
                "quality": ROOT / "runs/tiny-imagenet-10-quality/unified/quality-metrics.json",
                "image": ROOT / "runs/tiny-imagenet-10-unified/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/tiny-imagenet-10-unified/evaluation/metrics.json",
            },
            "Gaussian-only": {
                "metric": ROOT / "runs/tiny-imagenet-10-generation-only/evaluation/metrics.json",
                "quality": ROOT / "runs/tiny-imagenet-10-quality/gaussian-only/quality-metrics.json",
                "image": ROOT / "runs/tiny-imagenet-10-generation-only/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/tiny-imagenet-10-generation-only/evaluation/metrics.json",
                "reverse": ROOT / "runs/tiny-imagenet-10-generation-only/reverse-evaluation/metrics.json",
            },
            "Low-frequency Gaussian-only": {
                "metric": ROOT / "runs/tiny-imagenet-10-generation-only-lowfreq/evaluation/metrics.json",
                "quality": ROOT / "runs/tiny-imagenet-10-quality/lowfreq/quality-metrics.json",
                "image": ROOT / "runs/tiny-imagenet-10-generation-only-lowfreq/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/tiny-imagenet-10-generation-only-lowfreq/evaluation/metrics.json",
                "reverse": ROOT / "runs/tiny-imagenet-10-generation-only-lowfreq/reverse-evaluation/metrics.json",
            },
            "Label-conditioned": {
                "metric": ROOT / "runs/tiny-imagenet-10-label-conditioned/evaluation/metrics.json",
                "quality": ROOT / "runs/tiny-imagenet-10-quality/label-conditioned/quality-metrics.json",
                "image": ROOT / "runs/tiny-imagenet-10-label-conditioned/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/tiny-imagenet-10-label-conditioned/evaluation/metrics.json",
            },
        },
        "STL-10": {
            "Unified": {
                "metric": ROOT / "runs/stl10-10class-unified/evaluation/metrics.json",
                "quality": ROOT / "runs/stl10-quality/unified/quality-metrics.json",
                "image": ROOT / "runs/stl10-10class-unified/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/stl10-10class-unified/evaluation/metrics.json",
                "reverse": ROOT / "runs/stl10-10class-unified/reverse-evaluation/evaluate-30.json",
            },
            "Gaussian-only": {
                "metric": ROOT / "runs/stl10-10class-generation-only/evaluation/metrics.json",
                "quality": ROOT / "runs/stl10-quality/gaussian-only/quality-metrics.json",
                "image": ROOT / "runs/stl10-10class-generation-only/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/stl10-10class-generation-only/evaluation/metrics.json",
                "reverse": ROOT / "runs/stl10-10class-generation-only/reverse-evaluation/metrics.json",
            },
            "Low-frequency Gaussian-only": {
                "metric": ROOT / "runs/stl10-10class-generation-only-lowfreq/evaluation/metrics.json",
                "quality": ROOT / "runs/stl10-quality/lowfreq-75k/quality-metrics.json",
                "image": ROOT / "runs/stl10-10class-generation-only-lowfreq/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/stl10-10class-generation-only-lowfreq/evaluation/metrics.json",
                "reverse": ROOT / "runs/stl10-10class-generation-only-lowfreq/reverse-evaluation/metrics.json",
            },
            "Label-conditioned": {
                "metric": ROOT / "runs/stl10-10class-label-conditioned/evaluation/metrics.json",
                "quality": ROOT / "runs/stl10-quality/label-conditioned/quality-metrics.json",
                "image": ROOT / "runs/stl10-10class-label-conditioned/evaluation/step-0075000/generated-grid.png",
                "curve": ROOT / "runs/stl10-10class-label-conditioned/evaluation/metrics.json",
            },
        },
    }


def collect():
    rows = []
    for dataset, methods in manifest().items():
        for method, paths in methods.items():
            metric = paths["metric"]
            quality = paths["quality"]
            row = {
                "dataset": dataset,
                "method": method,
                "step": None,
                "generated_accuracy": None,
                "reverse_ode_accuracy": None,
                "fid": None,
                "kid": None,
                "is": None,
                "generated_pair_ssim": None,
                "metric_path": str(metric),
                "quality_path": str(quality),
                "image_path": str(paths["image"]),
                "reverse_path": str(paths["reverse"]) if paths.get("reverse") else None,
            }
            if metric.exists():
                try:
                    result = _result_at(metric)
                    row["step"] = int(result.get("step", 75000))
                    row["generated_accuracy"] = result.get("generated_accuracy")
                    row["reverse_ode_accuracy"] = result.get(
                        "inverse_accuracy", result.get("real_reverse_accuracy")
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            reverse_path = paths.get("reverse")
            if reverse_path is not None and reverse_path.exists():
                try:
                    reverse = _load(reverse_path)
                    # `evaluate_generation_only_reverse.py` writes a results
                    # list, while `evaluate.py` writes one result object.
                    reverse_rows = reverse.get("results") if isinstance(reverse, dict) else reverse
                    if reverse_rows is None:
                        reverse_rows = [reverse]
                    elif isinstance(reverse_rows, dict):
                        reverse_rows = [reverse_rows]
                    reverse_rows = [r for r in reverse_rows if r.get("method", "heun") == "heun"]
                    if reverse_rows:
                        row["reverse_ode_accuracy"] = max(
                            reverse_rows, key=lambda item: int(item.get("steps", 0))
                        ).get("real_reverse_accuracy", reverse_rows[-1].get("accuracy"))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            row.update(_quality(quality))
            rows.append(row)
    return rows


def _save_bars(rows, output):
    datasets = list(manifest())
    methods = list(manifest()[datasets[0]])
    colors = ["#35618f", "#b26a35", "#3c8c70", "#9b4f67"]
    figures = [
        ("generated_accuracy", "Generated class accuracy", True),
        ("reverse_ode_accuracy", "Reverse ODE accuracy", True),
        ("fid", "FID (lower is better)", False),
        ("kid", "KID (lower is better)", False),
        ("is", "Inception Score", True),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (field, title, higher) in zip(axes.flat, figures):
        x = np.arange(len(datasets))
        width = 0.18
        for index, method in enumerate(methods):
            values = []
            for dataset in datasets:
                value = next(r[field] for r in rows if r["dataset"] == dataset and r["method"] == method)
                values.append(np.nan if value is None else float(value) * (100 if field in {"generated_accuracy", "reverse_ode_accuracy"} else 1))
            axis.bar(x + (index - 1.5) * width, values, width, label=method, color=colors[index])
        axis.set_xticks(x, datasets, rotation=12)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if field in {"generated_accuracy", "reverse_ode_accuracy"}:
            axis.set_ylim(0, 100)
            axis.set_ylabel("percent")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output / "metric-comparison.png", dpi=190)
    plt.close(figure)


def _save_curves(output):
    datasets = list(manifest())
    methods = list(manifest()[datasets[0]])
    colors = ["#35618f", "#b26a35", "#3c8c70", "#9b4f67"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for axis, dataset in zip(axes, datasets):
        for method, color in zip(methods, colors):
            path = manifest()[dataset][method]["curve"]
            if not path.exists():
                continue
            data = _load(path)
            results = data.get("results", data)
            if isinstance(results, dict):
                results = list(results.values())
            results = [r for r in results if r.get("generated_accuracy") is not None]
            if not results:
                continue
            axis.plot([r["step"] for r in results], [100 * r["generated_accuracy"] for r in results], marker="o", label=method, color=color)
        axis.set_title(dataset)
        axis.set_xlabel("training steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("generated class accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output / "checkpoint-accuracy-curves.png", dpi=190)
    plt.close(figure)


def _save_visuals(output):
    datasets = list(manifest())
    methods = list(manifest()[datasets[0]])
    cell_w, cell_h = 320, 320
    label_h = 32
    canvas = Image.new("RGB", (len(datasets) * cell_w, len(methods) * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, dataset in enumerate(datasets):
        for row, method in enumerate(methods):
            path = manifest()[dataset][method]["image"]
            x, y = column * cell_w, row * (cell_h + label_h)
            if path.exists():
                image = Image.open(path).convert("RGB")
                image.thumbnail((cell_w, cell_h))
                image = ImageOps.pad(image, (cell_w, cell_h), color="white")
                canvas.paste(image, (x, y))
            else:
                draw.rectangle((x, y, x + cell_w, y + cell_h), outline="#cc3333", width=2)
                draw.text((x + 8, y + 8), "pending", fill="#cc3333")
            draw.text((x + 6, y + cell_h + 7), f"{dataset} | {method}", fill="black")
    canvas.save(output / "final-generation-grids.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/three-dataset-comparison")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = collect()
    (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _save_bars(rows, output)
    _save_curves(output)
    _save_visuals(output)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
