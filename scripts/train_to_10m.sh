#!/usr/bin/env bash
set -Eeuo pipefail

# Continue one million steps at a time so every milestone has a resumable
# checkpoint and a clearly named log entry.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/../all_datasets}"
OUTPUT="${OUTPUT:-$ROOT/runs/cifar10-generation-only-lowfreq-ot-ce-10m}"
CHECKPOINT="${CHECKPOINT:-$ROOT/weights/model-1000000.pt}"
START_STEPS="${START_STEPS:-1000000}"
END_STEPS="${END_STEPS:-10000000}"
STAGE_STEPS="${STAGE_STEPS:-1000000}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WORKERS="${WORKERS:-4}"
SAVE_EVERY="${SAVE_EVERY:-5000}"

for value in "$START_STEPS" "$END_STEPS" "$STAGE_STEPS"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "step values must be integers" >&2; exit 2; }
done
(( END_STEPS > START_STEPS )) || { echo "END_STEPS must be greater than START_STEPS" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "missing CHECKPOINT=$CHECKPOINT" >&2; exit 2; }

mkdir -p "$OUTPUT"
step="$START_STEPS"
checkpoint="$CHECKPOINT"
while (( step < END_STEPS )); do
  next=$((step + STAGE_STEPS))
  (( next > END_STEPS )) && next="$END_STEPS"
  stage_output="$OUTPUT/step-${next}"
  echo "[$(date -u +%FT%TZ)] continuing $step -> $next"
  DATA_ROOT="$DATA_ROOT" OUTPUT="$stage_output" CHECKPOINT="$checkpoint" \
    TARGET_STEPS="$next" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" \
    WORKERS="$WORKERS" SAVE_EVERY="$SAVE_EVERY" \
    "$ROOT/scripts/continue_from_1m.sh"
  checkpoint="$stage_output/model-latest.pt"
  step="$next"
done

echo "Finished at step $END_STEPS. Checkpoints and logs are under $OUTPUT/step-*"
