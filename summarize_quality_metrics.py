#!/usr/bin/env python3
"""Combine comparable generative-quality JSON files into a table and figure."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Generative quality comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.metrics) != len(args.names):
        raise ValueError("--metrics and --names must have the same length")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(Path(path).read_text()) for path in args.metrics]
    fields = [
        ("fid_inception", "FID", "lower"),
        ("kid_mean", "KID", "lower"),
        ("inception_score_mean", "Inception Score", "higher"),
        ("generated_pair_ssim", "Generated pair SSIM", "lower diversity"),
    ]
    figure, axes = plt.subplots(1, len(fields), figsize=(14, 3.5))
    for axis, (field, title, direction) in zip(axes, fields):
        values = [row[field] for row in rows]
        bars = axis.bar(args.names, values, color=("#3b82f6", "#f59e0b", "#10b981"))
        axis.set_title(f"{title}\n({direction})")
        axis.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4g}",
                ha="center", va="bottom", fontsize=8,
            )
    figure.suptitle(args.title)
    figure.tight_layout()
    figure.savefig(output / "quality-comparison.png", dpi=180)
    plt.close(figure)

    real_pair_ssim = rows[0]["real_pair_ssim"]
    lines = [
        f"# {args.title}", "",
        "FID/KID use the same generated count, real reference set, Inception extractor, and seed.",
        "SSIM is reported as a diversity diagnostic because generated samples are not paired with ground truth.",
        "",
        "| Method | FID | KID mean | IS | Generated pair SSIM | Nearest-train SSIM |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in zip(args.names, rows):
        lines.append(
            f"| {name} | {row['fid_inception']:.3f} | {row['kid_mean']:.5f} | "
            f"{row['inception_score_mean']:.3f} | {row['generated_pair_ssim']:.4f} | "
            f"{row['nearest_train_ssim']:.4f} |"
        )
    lines.extend([
        "", f"Real-test class-pair SSIM: `{real_pair_ssim:.4f}`.", "",
        "![Quality comparison](quality-comparison.png)", "",
    ])
    (output / "QUALITY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
