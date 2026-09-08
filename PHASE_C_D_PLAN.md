# Phase C/D 75k experiments

These experiments are designed to separate Gaussian class encoding from the
joint generation/classification objective. All runs use CIFAR-10, 32x32,
`center_scale=12`, `sigma=0.5`, the same 8.28M-parameter generation-only U-Net,
batch size 256, EMA, and 75k optimizer steps.

## Phase A: strong OT control

`low_frequency` centers with class-wise Sinkhorn pairing (`epsilon=0.005`, 50
iterations). The run resumes from the already completed 10k checkpoint so its
total training step is exactly 75k. This is the direct test of whether the
improved path coupling beats the 75k low-frequency baseline (75.16% generation
accuracy, 75.30% reverse ODE accuracy).

## Phase C: redundant class encoding

`coded_low_frequency` centers use all sixteen 4x4 low-frequency DCT modes and
the first ten rows of a 16-dimensional Hadamard code. Each class is therefore
encoded over many low-frequency modes instead of a single mode. The codewords
are orthogonal after normalization, but have redundant support that should be
more robust to image-space averaging and downsampling. Labels are still absent
from the U-Net input.

## Phase D: joint objective

Phase C is augmented with two supervised terms while retaining the same flow
matching loss:

```text
L = L_flow
  + 0.001 * L_endpoint_center
  + 0.001 * L_reverse_ode_center
```

`L_endpoint_center` uses the current velocity to estimate the starting Gaussian
endpoint. `L_reverse_ode_center` differentiably integrates 5 Euler steps from
the image toward noise and applies the Gaussian-center projection loss to the
encoded endpoint. This is a joint objective on one velocity field, not a second
classifier head and not an explicit label condition at sampling time.

## Running and outputs

```bash
setsid sh -c 'cd /root/autodl-tmp/optimized_unified_transport && \
  exec ./run_phase_c_d_75k.sh' </dev/null \
  > runs/phase-c-d-75k-console.log 2>&1 &
```

The script runs the strong OT control, Phase C, and Phase D sequentially. Each
run saves 5k checkpoints, and after 75k it computes generation accuracy,
reverse ODE accuracy, FID, KID, IS, SSIM, and visual grids under its own output
directory. Live progress is recorded in `STATUS-live.log` for each run.
