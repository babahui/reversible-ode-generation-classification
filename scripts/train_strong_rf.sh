#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
kind=${KIND:-gaussian}
output=${OUTPUT:-runs/strong-rf/$kind}
flags=(--kind "$kind" --output "$output" --data-root "${DATA_ROOT:-./data}"
  --steps "${STEPS:-500000}" --seed 0 --batch-size 128 --lr 0.0002
  --warmup 5000 --ema-decay 0.9999 --amp none --center-scale 12
  --save-every 25000 --snapshot-every 25000 --preview-every 25000
  --keep-previous-checkpoint)
case "$kind" in
  label) flags+=(--sigma 1) ;;
  gaussian|gaussian_ot) flags+=(--sigma 0.5) ;;
  gaussian_ce|gaussian_ot_ce)
    flags+=(--sigma 0.5 --ce-logits center --ce-weight 0.001
      --ce-every 4 --ce-steps 4 --ce-batch 16) ;;
  *) printf 'Unsupported study kind: %s\n' "$kind" >&2; exit 2 ;;
esac
if [[ ${RESUME:-0} == 1 ]]; then
  [[ -f "$output/training-latest.pt" ]] || { printf 'Missing full resume checkpoint\n' >&2; exit 2; }
  flags+=(--resume)
fi
exec "${PYTHON:-python}" -u -m strong_rf.train "${flags[@]}"
