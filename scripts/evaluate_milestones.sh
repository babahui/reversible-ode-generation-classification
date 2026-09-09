#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/../all_datasets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/runs/cifar10-generation-only-lowfreq-ot-ce-10m}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$ROOT/weights/model-1000000.pt}"
CLASSIFIER="${CLASSIFIER:-$ROOT/weights/cifar10-classifier-best.pt}"
OUTPUT="${OUTPUT:-$CHECKPOINT_DIR/evaluation}"
START_STEPS="${START_STEPS:-1000000}"
END_STEPS="${END_STEPS:-10000000}"
STAGE_STEPS="${STAGE_STEPS:-1000000}"
DEVICE="${DEVICE:-auto}"

for value in "$START_STEPS" "$END_STEPS" "$STAGE_STEPS"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "step values must be integers" >&2; exit 2; }
done
[[ -f "$CLASSIFIER" ]] || { echo "missing CLASSIFIER=$CLASSIFIER" >&2; exit 2; }
mkdir -p "$OUTPUT"

step="$START_STEPS"
while (( step <= END_STEPS )); do
  if (( step == START_STEPS )); then
    checkpoint="$INITIAL_CHECKPOINT"
  else
    checkpoint="$CHECKPOINT_DIR/step-${step}/model-latest.pt"
  fi
  [[ -f "$checkpoint" ]] || { echo "missing checkpoint for ${step}: $checkpoint" >&2; exit 2; }
  echo "[$(date -u +%FT%TZ)] evaluating step $step"
  CHECKPOINT="$checkpoint" CLASSIFIER="$CLASSIFIER" DATA_ROOT="$DATA_ROOT" \
    OUTPUT="$OUTPUT/step-${step}" DEVICE="$DEVICE" \
    "$ROOT/scripts/evaluate_1m.sh"
  step=$((step + STAGE_STEPS))
done

echo "Finished milestone evaluation. Results are under $OUTPUT/step-*"
