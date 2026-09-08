#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/optimized_unified_transport"
RUN="$ROOT/runs/cifar10-generation-only-lowfreq-ot-ce-5k"
OUT="$ROOT/runs/cifar10-generation-only-lowfreq-ot-ce-500k-comparison"
CKPT="$RUN/model-500000.pt"

while [ ! -f "$CKPT" ]; do
  sleep 60
done

mkdir -p "$OUT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python evaluate_generation_only.py \
  --checkpoints "$CKPT" \
  --classifier runs/reference-cifar10-classifier/cifar10-classifier-best.pt \
  --data-root /root/autodl-tmp/all_datasets \
  --output "$OUT/generation" \
  --steps 30 --method heun --batch-size 256 --real-images 10000 \
  --generated-per-class 256 --visuals-per-class 10 --workers 4 \
  --seed 20260828 --device cuda > "$OUT/generation.log" 2>&1

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python evaluate_generative_quality.py \
  --checkpoint "$CKPT" \
  --data-root /root/autodl-tmp/all_datasets \
  --output "$OUT/quality" \
  --generated-per-class 256 --real-images 5000 --batch-size 64 \
  --steps 30 --method heun --ssim-per-class 32 --ssim-real-candidates 200 \
  --workers 4 --seed 20260828 --device cuda > "$OUT/quality.log" 2>&1

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python evaluate_reverse_margin.py \
  --checkpoint "$CKPT" \
  --data-root /root/autodl-tmp/all_datasets \
  --output "$OUT/reverse-margin" \
  --steps 30 --method heun --batch-size 128 --max-test-images 10000 \
  --workers 4 --seed 20260828 --device cuda > "$OUT/reverse-margin.log" 2>&1

date -u > "$OUT/complete.txt"
