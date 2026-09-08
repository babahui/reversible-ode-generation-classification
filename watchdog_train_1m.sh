#!/usr/bin/env bash
set -u

ROOT="/root/autodl-tmp/optimized_unified_transport"
RUN="$ROOT/runs/cifar10-generation-only-lowfreq-ot-ce-5k"
LOG="$RUN/watchdog-1m.log"

latest_checkpoint() {
  find "$RUN" -maxdepth 1 -type f -name 'model-[0-9]*.pt' -printf '%f\n' \
    | sort -V | tail -1
}

prune_checkpoints() {
  python - "$RUN" <<'PY'
from pathlib import Path
import sys

run = Path(sys.argv[1])
milestones = {5000, 50000, 75000, 100000, 150000, 200000,
              225000, 300000, 400000, 450000, 500000, 1000000}
checkpoints = []
for path in run.glob("model-[0-9]*.pt"):
    step = int(path.stem.split("-")[1])
    checkpoints.append((step, path))
latest_steps = {step for step, _ in sorted(checkpoints)[-2:]}
for step, path in checkpoints:
    if step in milestones or step % 25000 == 0 or step in latest_steps:
        continue
    path.unlink()
PY
}

while [ ! -f "$RUN/model-1000000.pt" ]; do
  prune_checkpoints
  if pgrep -f '[p]ython train_generation_only.py.*cifar10-generation-only-lowfreq-ot-ce-5k' >/dev/null; then
    sleep 60
    continue
  fi

  checkpoint_name="$(latest_checkpoint)"
  if [ -z "$checkpoint_name" ]; then
    printf '%s no checkpoint found\n' "$(date -u --iso-8601=seconds)" >> "$LOG"
    sleep 60
    continue
  fi

  printf '%s restarting from %s\n' "$(date -u --iso-8601=seconds)" "$checkpoint_name" >> "$LOG"
  cd "$ROOT" || exit 1
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train_generation_only.py \
    --mode cifar10 --data-root /root/autodl-tmp/all_datasets \
    --output runs/cifar10-generation-only-lowfreq-ot-ce-5k \
    --classes 0,1,2,3,4,5,6,7,8,9 \
    --center-scale 12 --center-mode low_frequency --latent-sigma 0.5 \
    --pairing sinkhorn --ot-epsilon 0.005 --ot-iterations 10 \
    --path-diagnostics-every 1000 --center-loss-weight 0 \
    --ode-class-weight 0.001 --ode-class-steps 4 --ode-class-batch 32 \
    --ode-class-loss ce --ode-class-temperature 1.0 --ode-class-every 4 \
    --base-channels 64 --channel-mults 1,2,4 --batch-size 256 \
    --train-steps 1000000 --learning-rate 2e-4 --ema-decay 0.9999 \
    --ema-warmup 1000 --grad-clip 1 --data-noise 0.01 --workers 4 \
    --log-every 100 --save-every 5000 --seed 0 --device cuda \
    --resume "$RUN/$checkpoint_name" >> "$LOG" 2>&1

  status=$?
  printf '%s training command exited with status %s\n' \
    "$(date -u --iso-8601=seconds)" "$status" >> "$LOG"
  sleep 30
done

printf '%s reached model-1000000.pt\n' "$(date -u --iso-8601=seconds)" >> "$LOG"
