#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/../all_datasets}"
CHECKPOINT="${CHECKPOINT:-$ROOT/runs/cifar10-generation-only-lowfreq-ot-ce-5k/model-1000000.pt}"
CLASSIFIER="${CLASSIFIER:-$ROOT/runs/reference-cifar10-classifier/cifar10-classifier-best.pt}"
OUTPUT="${OUTPUT:-$ROOT/runs/cifar10-generation-only-lowfreq-ot-ce-1m-evaluation}"
DEVICE="${DEVICE:-auto}"
[[ -f "$CHECKPOINT" ]] || { echo "missing CHECKPOINT=$CHECKPOINT" >&2; exit 2; }
[[ -f "$CLASSIFIER" ]] || { echo "missing CLASSIFIER=$CLASSIFIER" >&2; exit 2; }
mkdir -p "$OUTPUT"
cd "$ROOT"

dataset_args=(--data-root "$DATA_ROOT")
if [[ "${CIFAR10_OFFLINE:-0}" == "1" ]]; then
  dataset_args+=(--verify-only)
fi
python scripts/prepare_cifar10.py "${dataset_args[@]}"

echo "[$(date -u +%FT%TZ)] generation evaluation"
PYTHONPATH="$ROOT" python evaluate_generation_only.py \
  --checkpoints "$CHECKPOINT" --classifier "$CLASSIFIER" --data-root "$DATA_ROOT" \
  --output "$OUTPUT/generation" --steps 30 --method heun --batch-size 256 \
  --real-images 10000 --generated-per-class 256 --visuals-per-class 10 \
  --workers 4 --seed 20260828 --device "$DEVICE" 2>&1 | tee "$OUTPUT/generation.log"

echo "[$(date -u +%FT%TZ)] quality evaluation"
PYTHONPATH="$ROOT" python evaluate_generative_quality.py \
  --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" --output "$OUTPUT/quality" \
  --generated-per-class 256 --real-images 5000 --batch-size 64 --steps 30 \
  --method heun --ssim-per-class 32 --ssim-real-candidates 200 --workers 4 \
  --seed 20260828 --device "$DEVICE" 2>&1 | tee "$OUTPUT/quality.log"

echo "[$(date -u +%FT%TZ)] reverse ODE classification"
PYTHONPATH="$ROOT" python evaluate_reverse_margin.py \
  --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" --output "$OUTPUT/reverse-margin" \
  --steps 30 --method heun --batch-size 128 --max-test-images 10000 \
  --workers 4 --seed 20260828 --device "$DEVICE" 2>&1 | tee "$OUTPUT/reverse-margin.log"

date -u > "$OUTPUT/complete.txt"
echo "evaluation complete: $OUTPUT"
