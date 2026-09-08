#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable MNIST transport training and evaluation in stages."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument(
        "--stage-steps",
        type=int,
        default=50000,
        help="Iterations between checkpoint/evaluation stages; about one hour on the measured RTX 4080 setup",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--channel-mults", default="1,2,4")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ema-warmup", type=int, default=1000)
    parser.add_argument("--ode-steps", type=int, default=30)
    parser.add_argument("--eval-test-images", type=int, default=2048)
    parser.add_argument("--generated-per-class", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    if not path.exists():
        return 0
    return int(torch.load(path, map_location="cpu")["step"])


def run(command, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def main() -> None:
    args = parse_args()
    if args.stage_steps <= 0 or args.total_steps <= 0:
        raise ValueError("stage-steps and total-steps must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "model-latest.pt"
    evaluation_output = output / "evaluation"
    script_directory = Path(__file__).resolve().parent
    configuration = vars(args).copy()
    configuration["output"] = str(output)
    (output / "periodic-experiment-config.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )
    current_step = checkpoint_step(latest)

    while current_step < args.total_steps:
        target_step = min(current_step + args.stage_steps, args.total_steps)
        train_command = [
            sys.executable,
            str(script_directory / "train.py"),
            "--mode", "mnist",
            "--data-root", str(Path(args.data_root).resolve()),
            "--classes", args.classes,
            "--image-size", "28",
            "--output", str(output),
            "--base-channels", str(args.base_channels),
            "--channel-mults", args.channel_mults,
            "--batch-size", str(args.batch_size),
            "--train-steps", str(target_step),
            "--learning-rate", str(args.learning_rate),
            "--ema-warmup", str(args.ema_warmup),
            "--save-every", str(args.stage_steps),
            "--log-every", "100",
            "--workers", str(args.workers),
            "--seed", str(args.seed),
            "--device", args.device,
        ]
        if current_step:
            train_command.extend(("--resume", str(latest)))
        run(train_command, output / "periodic-training.log")
        current_step = checkpoint_step(latest)
        if current_step != target_step:
            raise RuntimeError(
                f"training stopped at step {current_step}, expected {target_step}"
            )
        named_checkpoint = output / f"model-{current_step}.pt"
        if not named_checkpoint.exists():
            temporary = named_checkpoint.with_suffix(".pt.tmp")
            shutil.copy2(latest, temporary)
            temporary.replace(named_checkpoint)
        checkpoint_for_evaluation = named_checkpoint
        evaluation_command = [
            sys.executable,
            str(script_directory / "evaluate_checkpoints.py"),
            "--checkpoints", str(checkpoint_for_evaluation),
            "--classifier", str(Path(args.classifier).resolve()),
            "--data-root", str(Path(args.data_root).resolve()),
            "--output", str(evaluation_output),
            "--steps", str(args.ode_steps),
            "--batch-size", str(args.batch_size),
            "--max-test-images", str(args.eval_test_images),
            "--cycle-images", "128",
            "--generated-per-class", str(args.generated_per_class),
            "--visuals-per-class", "10",
            "--workers", str(args.workers),
            "--seed", "2026",
            "--device", args.device,
        ]
        run(evaluation_command, output / "periodic-evaluation.log")
        print(
            json.dumps(
                {
                    "completed_step": current_step,
                    "checkpoint": str(checkpoint_for_evaluation),
                    "metrics": str(evaluation_output / "metrics.json"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
