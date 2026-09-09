#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/../all_datasets}"
OUTPUT="${OUTPUT:-$ROOT/runs/cifar10-generation-only-lowfreq-ot-ce-5k}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT/model-latest.pt}"
TARGET_STEPS="${TARGET_STEPS:-1200000}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WORKERS="${WORKERS:-4}"
SAVE_EVERY="${SAVE_EVERY:-5000}"

[[ "$TARGET_STEPS" =~ ^[0-9]+$ ]] || { echo "TARGET_STEPS must be an integer" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "missing CHECKPOINT=$CHECKPOINT" >&2; exit 2; }
mkdir -p "$OUTPUT"
cd "$ROOT"
dataset_args=(--data-root "$DATA_ROOT")
if [[ "${CIFAR10_OFFLINE:-0}" == "1" ]]; then
  dataset_args+=(--verify-only)
fi
python scripts/prepare_cifar10.py "${dataset_args[@]}"

args=(
  --mode cifar10 --data-root "$DATA_ROOT" --output "$OUTPUT"
  --classes 0,1,2,3,4,5,6,7,8,9 --image-size 32
  --center-scale 12 --center-mode low_frequency --latent-sigma 0.5
  --pairing sinkhorn --ot-epsilon 0.005 --ot-iterations 10
  --path-diagnostics-every 1000 --center-loss-weight 0
  --ode-class-weight 0.001 --ode-class-steps 4 --ode-class-batch 32
  --ode-class-loss ce --ode-class-temperature 1.0 --ode-class-every 4
  --base-channels 64 --channel-mults 1,2,4 --batch-size "$BATCH_SIZE"
  --train-steps "$TARGET_STEPS" --learning-rate 2e-4 --ema-decay 0.9999
  --ema-warmup 1000 --grad-clip 1 --data-noise 0.01 --workers "$WORKERS"
  --log-every 100 --save-every "$SAVE_EVERY" --seed 0 --device "$DEVICE"
  --resume "$CHECKPOINT"
)
echo "Continuing from $CHECKPOINT to step $TARGET_STEPS"
PYTHONPATH="$ROOT" python train_generation_only.py "${args[@]}" 2>&1 | tee -a "$OUTPUT/continue-training.log"
