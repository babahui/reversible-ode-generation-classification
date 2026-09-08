#!/usr/bin/env python3
"""Aggregate the 75k A/C/D CIFAR-10 experiments into one report."""

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "phase-c-d-75k"
RUNS = {
    "Low-frequency baseline": ROOT / "runs/cifar10-generation-only-lowfreq-base64",
    "Strong OT": ROOT / "runs/cifar10-generation-only-lowfreq-ot-eps005-10k-v2",
    "Phase C coded": ROOT / "runs/cifar10-generation-only-coded-lowfreq-base64",
    "Phase D joint": ROOT / "runs/cifar10-generation-only-coded-lowfreq-joint-base64",
}


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def metrics_for(name: str, run: Path):
    if name == "Low-frequency baseline":
        generation = read_json(run / "evaluation/metrics.json")
        generated = next(item for item in generation["results"] if item["step"] == 75000)
        reverse = read_json(run / "reverse-evaluation/metrics.json")
        reverse_item = reverse["results"][-1]
        quality = read_json(ROOT / "runs/cifar10-quality/lowfreq/quality-metrics.json")
    else:
        generation = read_json(run / "evaluation-75k/metrics.json")
        generated = generation["results"][0]
        reverse = read_json(run / "reverse-evaluation-75k/metrics.json")
        reverse_item = next(item for item in reverse["results"] if item["steps"] == 30)
        quality = read_json(run / "quality-75k/quality-metrics.json")
    return {
        "method": name,
        "step": 75000,
        "generated_accuracy": generated["generated_accuracy"],
        "reverse_ode_accuracy": reverse_item["real_reverse_accuracy"],
        "fid": quality["fid_inception"],
        "kid": quality["kid_mean"],
        "is": quality["inception_score_mean"],
        "pair_ssim": quality["generated_pair_ssim"],
    }


def make_visual():
    images = []
    for name, run in RUNS.items():
        if name == "Low-frequency baseline":
            path = run / "evaluation/step-0075000/generated-grid.png"
        else:
            path = run / "evaluation-75k/step-0075000/generated-grid.png"
        if not path.exists():
            return
        images.append((name, Image.open(path).convert("RGB")))
    width, height = images[0][1].size
    canvas = Image.new("RGB", (width * 2, (height + 34) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(images):
        x = (index % 2) * width
        y = (index // 2) * (height + 34)
        draw.text((x + 8, y + 8), name, fill="black")
        canvas.paste(image, (x, y + 34))
    canvas.save(OUT / "generation-comparison-75k.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    missing = []
    for name, run in RUNS.items():
        try:
            results.append(metrics_for(name, run))
        except FileNotFoundError as error:
            missing.append(str(error))
    (OUT / "metrics.json").write_text(json.dumps({"results": results, "missing": missing}, indent=2) + "\n")
    lines = [
        "# Phase C/D CIFAR-10 75k comparison",
        "",
        "| Method | Generation accuracy | Reverse ODE accuracy | FID | KID | IS | Pair SSIM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['method']} | {item['generated_accuracy']:.2%} | "
            f"{item['reverse_ode_accuracy']:.2%} | {item['fid']:.2f} | "
            f"{item['kid']:.5f} | {item['is']:.3f} | {item['pair_ssim']:.5f} |"
        )
    lines.extend([
        "",
        "Lower is better for FID/KID; higher is better for generation accuracy, reverse ODE accuracy, and IS.",
        "The baseline row is read from the previously completed 75k low-frequency run.",
        "",
    ])
    if missing:
        lines.extend(["## Pending artifacts", "", *[f"- `{item}`" for item in missing], ""])
    else:
        lines.extend(["![75k generated grids](generation-comparison-75k.png)", ""])
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    if not missing:
        make_visual()
    print(json.dumps({"results": results, "missing": missing}))


if __name__ == "__main__":
    main()
