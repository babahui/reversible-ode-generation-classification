CIFAR-10 pure Gaussian FM 1M followed by 50K updates of Euler1 reverse CE and paired pad4 JS.

- EMA Euler1 test10K accuracy: **92.72%**; Euler4: 91.99%.
- Heun60 generation50K FID vs train50K: **4.8161**; generated-label accuracy: **90.438%**.
- Full Chinese training, resumption and inference instructions: [cifar10_ce1/README.md](https://github.com/babahui/reversible-ode-generation-classification/blob/experiment/cifar10-ce1/cifar10_ce1/README.md).
- `cifar10-ce1-1050000-ema.pt`: inference checkpoint.
- `cifar10-ce1-1050000-full.pt`: complete Adam/EMA/RNG state for continued CE1 training.
- `cifar10-fm-1000000-full.pt`: original pure-FM initialization for reproducing the experiment.
- `cifar10-reference-classifier.pt`: independent generated-label scorer, not used for model inference.
- SHA256 hashes are included in `SHA256SUMS` and the repository's `cifar10_ce1/weights.json`.

CE1 refers to one-step reverse classification; image generation uses Heun60. The +500K run is ongoing and is not represented by these +50K weights.
