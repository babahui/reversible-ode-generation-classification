#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/optimized_unified_transport"
RUN="$ROOT/runs/cifar10-generation-only-lowfreq-ot-ce-5k"
EVAL="$ROOT/runs/cifar10-generation-only-lowfreq-ot-ce-500k-comparison"

while [ ! -f "$RUN/model-500000.pt" ]; do
  sleep 60
done

while [ ! -f "$EVAL/complete.txt" ]; do
  sleep 60
done

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train_generation_only.py \
  --mode cifar10 --data-root /root/autodl-tmp/all_datasets \
  --output "$RUN" --classes 0,1,2,3,4,5,6,7,8,9 \
  --center-scale 12 --center-mode low_frequency --latent-sigma 0.5 \
  --pairing sinkhorn --ot-epsilon 0.005 --ot-iterations 10 \
  --path-diagnostics-every 1000 --center-loss-weight 0 \
  --ode-class-weight 0.001 --ode-class-steps 4 --ode-class-batch 32 \
  --ode-class-loss ce --ode-class-temperature 1.0 --ode-class-every 4 \
  --base-channels 64 --channel-mults 1,2,4 --batch-size 256 \
  --train-steps 1000000 --learning-rate 2e-4 --ema-decay 0.9999 \
  --ema-warmup 1000 --grad-clip 1 --data-noise 0.01 --workers 4 \
  --log-every 100 --save-every 5000 --seed 0 --device cuda \
  --resume "$RUN/model-latest.pt"
