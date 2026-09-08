#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
DATA_ROOT="../all_datasets"
CLASSIFIER="runs/reference-cifar10-classifier/cifar10-classifier-best.pt"
COMMON=(
  --mode cifar10 --data-root "$DATA_ROOT" --image-size 32
  --classes 0,1,2,3,4,5,6,7,8,9 --center-scale 12 --latent-sigma 0.5
  --base-channels 64 --channel-mults 1,2,4 --batch-size 256
  --train-steps 75000 --save-every 5000 --log-every 100 --workers 4
  --path-diagnostics-every 1000 --device cuda:0
)

mark() {
  local run="$1"
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" \
    | tee -a "$run/STATUS-live.log"
}

evaluate_run() {
  local run="$1"
  local checkpoint="$run/model-75000.pt"
  mark "$run" "evaluation started"
  python evaluate_generation_only.py \
    --checkpoints "$checkpoint" --classifier "$CLASSIFIER" \
    --data-root "$DATA_ROOT" --output "$run/evaluation-75k" \
    --steps 30 --method heun --batch-size 256 --real-images 5000 \
    --generated-per-class 256 --visuals-per-class 10 --workers 4 \
    --seed 20260828
  python evaluate_generation_only_reverse.py \
    --checkpoint "$checkpoint" --data-root "$DATA_ROOT" \
    --output "$run/reverse-evaluation-75k" --classifier "$CLASSIFIER" \
    --steps 15 30 --method heun --batch-size 256 --max-test-images 1000 \
    --generated-per-class 64 --workers 4 --seed 20260828
  python evaluate_generative_quality.py \
    --checkpoint "$checkpoint" --data-root "$DATA_ROOT" \
    --output "$run/quality-75k" --generated-per-class 500 \
    --real-images 5000 --batch-size 64 --steps 30 --method heun \
    --ssim-per-class 64 --ssim-real-candidates 256 --workers 4 \
    --seed 20260828
  mark "$run" "evaluation finished"
}

OT_RUN="runs/cifar10-generation-only-lowfreq-ot-eps005-10k-v2"
if [[ ! -f "$OT_RUN/model-75000.pt" ]]; then
  mark "$OT_RUN" "resuming strong OT from 10k to 75k"
  python train_generation_only.py \
    "${COMMON[@]}" --output "$OT_RUN" --center-mode low_frequency \
    --pairing sinkhorn --ot-epsilon 0.005 --ot-iterations 50 \
    --resume "$OT_RUN/model-10000.pt"
  evaluate_run "$OT_RUN"
fi

C_RUN="runs/cifar10-generation-only-coded-lowfreq-base64"
if [[ ! -f "$C_RUN/model-75000.pt" ]]; then
  mkdir -p "$C_RUN"
  mark "$C_RUN" "starting phase C coded low-frequency Gaussian-only"
  C_RESUME=()
  if [[ -f "$C_RUN/model-latest.pt" ]]; then
    C_RESUME=(--resume "$C_RUN/model-latest.pt")
  fi
  python train_generation_only.py \
    "${COMMON[@]}" --output "$C_RUN" --center-mode coded_low_frequency \
    --pairing independent "${C_RESUME[@]}"
  evaluate_run "$C_RUN"
fi

D_RUN="runs/cifar10-generation-only-coded-lowfreq-joint-base64"
if [[ ! -f "$D_RUN/model-75000.pt" ]]; then
  mkdir -p "$D_RUN"
  mark "$D_RUN" "starting phase D joint flow plus reverse ODE objective"
  D_RESUME=()
  if [[ -f "$D_RUN/model-latest.pt" ]]; then
    D_RESUME=(--resume "$D_RUN/model-latest.pt")
  fi
  python train_generation_only.py \
    "${COMMON[@]}" --output "$D_RUN" --center-mode coded_low_frequency \
    --pairing independent --center-loss-weight 0.001 \
    --ode-class-weight 0.001 --ode-class-steps 5 --ode-class-batch 64 \
    "${D_RESUME[@]}"
  evaluate_run "$D_RUN"
fi

python summarize_phase_c_d.py

printf '[%s] all phase C/D experiments complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
