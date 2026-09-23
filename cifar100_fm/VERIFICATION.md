# Release Verification

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

Not rerun for this release: a fresh 500K GPU training, full 50K Inception
evaluation, or a CUDA resume equivalence experiment. The historical metrics
remain historical measurements; no new GPU performance claim is made.
