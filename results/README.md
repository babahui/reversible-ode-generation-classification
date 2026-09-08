# Reproduced 1M result

The numbers below are from the checkpoint at training step 1,000,000. The
checkpoint is intentionally excluded from Git because it is about 133 MB and
GitHub's normal file limit is 100 MB. Upload it to your own cloud drive and
place it at the path shown in `WEIGHTS.md` before running inference.

| metric | value | evaluation |
| --- | ---: | --- |
| conditional generation accuracy | 82.66% | 2,560 generated CIFAR-10 images, 256/class |
| reverse ODE classification accuracy | 90.23% | 10,000 test images, image -> Gaussian, Heun 30 steps |
| FID (Inception) | 53.25769 | 2,560 generated vs 5,000 real images |
| KID mean | 0.05192797 | 100 subsets |
| Inception Score | 6.56545 | 2,560 generated images |

The machine-readable source files are preserved under the local experiment
directory before publication:

```text
runs/cifar10-generation-only-lowfreq-ot-ce-1m-evaluation/
  generation/metrics.json
  quality/quality-metrics.json
  reverse-margin/metrics.json
```
