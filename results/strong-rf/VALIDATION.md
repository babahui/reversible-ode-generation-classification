# Publication Validation

Validated on 2026-09-15 in an isolated worktree based on remote `origin/main`.
The original experiment working tree and its uncommitted changes were not edited.

- All 26 repository unit tests passed using the recorded PyTorch 2.0.0+cu118
  environment. Tests include reverse gradients, no label leakage into the
  Gaussian reverse solve, OT permutation preservation, seed-matched trunks,
  checkpoint rotation, selective continuation and portable release commands.
- Actual GPU smoke training: `gaussian_ot_ce`, batch16, four updates,
  `--ce-logits center --ce-every 4 --ce-batch 2 --ce-steps 4`.
  The fourth update exercised differentiable reverse CE; flow loss, CE and
  gradients were finite. This is a smoke test, not a reported research run.
- Resumed that full optimizer/RNG checkpoint at step4 and completed step5.
- The independent sampling CLI loaded Gaussian 550K EMA and generated ten
  images with Heun60. Its reverse CLI classified sixteen real test images
  with Heun60, without the evaluation classifier. These tiny checks do not
  replace the archived full-test accuracy.
- Replayed the formal Gaussian550K first batch: 64 images, seed20260912,
  Heun60, FP32, no TF32. The checkpoint SHA256 matched the stored protocol;
  all output uint8 pixels exactly matched the archived batch (max difference0).
- SHA256 of all six core source files in `source-provenance.json` matched
  the original experiment code. Training, prior, pairing, solver and metric
  numerics were not changed for publication.
- Inspected the exported common400K generation comparison visually.

No new 50K-image metric evaluation or long training run was performed for the
publication. Tables are exports of completed experiments, not extrapolations.
