# Release Verification

## Dual-Pad4 CE + JS Addition (2026-09-23)

Local execution used the actual 35,746,307-parameter U-Net, real CIFAR-100 data,
FM batch128, CE64 (32 original images, two views each), FP32 and Euler4.
GPU1 was shared briefly for validation; existing jobs were not stopped.
No 5M production training was started by the release operation.

Passed checks:

- Nine CPU unit tests: original prior/model/solver tests plus JS symmetry,
  temperature, both-view gradients, crop RNG isolation, original-image
  preservation, label-free differentiable reverse supervision, and total-step
  scheduling including the 1M fork and 5M target.
- Three report tests with explicitly synthetic fixtures check matched-step
  CSV/grid output, rejection of mismatched solvers and exclusion of partial
  evaluations. Fourteen existing RF tests also pass unchanged.
- Extracted the crop and JS functions from the actual CIFAR-10 dual-pad4 run.
  For fixed seeds, the 32x2 crops and the 100-class JS values/gradients match
  those functions exactly.
- CUDA pure-FM from-scratch updates 0-4, then a full-state fork into CE+JS.
- CUDA CE+JS updates 4-8, checkpoint/restart, then 8-12. CE, JS and diagnostic
  gradients are finite; the model has no learned classifier/head/label embedding.
- Independent continuous run from the same full step-4 source to step12,
  compared with the split/restarted run. CPU/CUDA/crop RNG states and Adam
  step counters are identical. Maximum absolute differences:
  model `4.5592e-7`, EMA `7.4506e-9`, Adam first moment `9.9174e-8`, second
  moment `4.5475e-11`. These pass `atol=2e-6, rtol=1e-5` for model/EMA.
  CUDA arithmetic is not promised to be bitwise deterministic across processes.
- Also ran CE+JS from scratch for four updates, with the same full batch sizes.
  Peak PyTorch allocated memory with gradient diagnostics: 15,132.7 MiB.
  This is per-process allocated memory, not total device memory or a throughput
  benchmark; the GPU was shared and the run was short.
- EMA checkpoint inference: Heun60 generation of all 100 classes (one image
  per class, ten grid pages) and Heun60 classification of eight real test images.
- Evaluation pipeline: 100 generated images and eight test images with Heun2,
  using the existing full CIFAR-100 reference feature cache. FID/IS computation,
  confusion matrix, class grids and protocol output completed with finite metrics.

The compressed four-step fork is a **software smoke test**, not the proposed
1M initialization or a classification/generation result. Its accuracy, FID and
grids are not included in the research performance table. Full 50K generation /
10K classification at trained 5M checkpoints remains to be run.

Local ignored artifacts: `runs/cifar100-dual-release-smoke/` and
`outputs/cifar100-dual-release-smoke/`. They are not uploaded.

## Original Pure-FM Release Checks

The release was checked on CPU using Python 3.8.10, PyTorch 2.0.0+cu118 and
torchvision 0.15.1+cu118. No running GPU training job was changed or interrupted.

- Five unit tests passed: codebook orthogonality and class mapping, Gaussian
  variance/posterior, forward/reverse solver direction and no-label contract,
  parameter count, quantization and study milestones.
- Seed-0 model initialization is tensor-identical to the historical isolated
  CIFAR-100 snapshot.
- The original 450K EMA checkpoint loaded strictly into the released model.
  CPU velocity outputs matched the original snapshot exactly on the same input.
- With the same model state, input, sampled noise and dropout RNG, FM loss and
  all parameter gradients matched the original implementation exactly.
- The independent classification CLI completed a two-image, Euler1 CPU smoke
  test using that checkpoint and CIFAR-100 test images. This is an execution
  check, **not a new accuracy measurement**.
- Training and evaluation CLI argument parsing passed.
- All eight archived full evaluation results were checked: each confusion
  matrix contains 10,000 images and its diagonal reproduces the reported
  accuracy; each generation set contains 500 images per class; all use Heun60.

Original 450K checkpoint SHA256:

```text
68d5b9e0c3d85efd8f4f604c7489264b4140769c6c6985576d6f5a77da59317a
```

The original training snapshot was under
`cifar100-scalar-fm-500k-20260917/code/strong_rf/`. Its vendored U-Net files
are byte-identical to `strong_rf/vendor_unet/` in this repository.
Results archive source: the completed `evaluations/gaussian-STEP-50k/metrics.json`
and corresponding `protocol.json`. Machine-specific paths were removed.

Not rerun for the original release: a fresh 500K GPU training, full 50K Inception
evaluation, or a CUDA resume equivalence experiment. The historical metrics
remain historical measurements; no new GPU performance claim is made.
