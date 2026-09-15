# New backbone: TorchCFM U-Net, CIFAR-10

This is the `backbone/torchcfm-unet-cifar10` experiment branch. The legacy
`unified_transport/model.py` experiments remain available, but this study uses
the separate model constructed in [`common.py`](common.py).

[中文说明](USAGE_CN.md)

## Scope and backbone

- Vendored TorchCFM U-Net at commit `7c653857de4c25b979a740ef8b6a5d5ddb09b6c1`;
  see [source attribution and MIT license](vendor_unet/PROVENANCE.md).
- CIFAR-10, RGB 32x32; base channels 128, multipliers `[1,2,2,2]`, two residual
  blocks, attention at 16px and the bottleneck, dropout 0.1.
- Gaussian models: 35,746,307 parameters. Label model: 35,751,427 parameters.
- All five main methods start from scratch. Shared trunk initialization is
  seed-matched. Public pretrained I-CFM weights were used for a separate metric
  sanity check, **not** to initialize these comparisons.
- This is single-stage linear flow matching (rectified flow), not DDPM score
  training or a teacher-generated reflow experiment.
- No posterior head, classifier guidance, history adapter, or classifier output
  enters any Gaussian velocity field. The new state/history adapter proposal is
  a future experiment; no results are claimed for it on this backbone.

## Five implemented methods

| `--kind` | Initial noise | Velocity input | Coupling | Extra training loss |
|---|---|---|---|---|
| `label` | N(0,I) | state, time, label embedding | independent | none |
| `gaussian` | N(mu_y,0.25 I) | state, time | independent | none |
| `gaussian_ot` | same class Gaussian | state, time | class-wise Sinkhorn + permutation | none |
| `gaussian_ce` | same class Gaussian | state, time | independent | reverse endpoint CE |
| `gaussian_ot_ce` | same class Gaussian | state, time | class-wise Sinkhorn + permutation | reverse endpoint CE |

`label_ot` is also supported by the low-level trainer but is not one of the
five reported groups. Here "Gaussian-only" always means **low-frequency**
Gaussian, not the older random-center Gaussian experiment.

### Forward training and sampling

For an image-label pair `(x1,y)`, sample `x0` from the specified prior and
`t ~ Uniform(0,1)`. The input is `xt=(1-t)*x0+t*x1`; the target is `x1-x0`.
[`train.py`](train.py) minimizes mean squared error over batch, channel and pixels.
The new API is `model(t, xt, y=label_or_None)`, not the legacy alpha-vector API.
The output is a `[B,3,32,32]` velocity, not an image or class logits.

At generation time, integrate `dx/dt=v(t,x)` from 0 to 1. Gaussian generation
uses the requested label only to select an initial Gaussian; it is never passed
to the U-Net. Label-conditioned generation passes the label at every evaluation.
No clipping is applied inside the ODE; only the final display image is quantized.

### Gaussian centers and reverse classification

[`OrthogonalGaussianLatent`](../unified_transport/latent.py) constructs a
normalized separable DCT pattern for each class. Frequency pairs `(u,v)` are
enumerated in row-major order over `0..3`, excluding `(3,3)`; the first ten
are used. Pattern `i` is placed in channel `i % 3`, all other channels zero.
After unit L2 normalization, each center is multiplied by 12. Thus `mu_y` is
a full `[3,32,32]` tensor with norm 12, not a scalar pixel mean.
Noise is `mu_y + 0.5 * randn_like(mu_y)`, with covariance `0.25 I` and uniform
class priors. This is neither N(0,I) nor low-pass-filtered random noise.

For a real unaugmented image, integrate the **same field**, without a label,
from `t=1` to `t=0`, obtaining `z_hat`. Classify by the largest score
`-||z_hat-mu_k||^2/(2*sigma^2) + log(pi_k)`. `log_posterior()` returns these
unnormalized logits; apply softmax/log_softmax for normalized probabilities.
Equal norms and uniform priors make this argmax equivalent to center projection.
The Label model has no label-free reverse classification result: supplying its
ground-truth label to the reverse solve would leak the answer.

### OT implementation

[`pair_latents_by_class`](../unified_transport/ot_pairing.py) samples a batch of
class Gaussians, groups noise and images by class, computes mean squared pixel
distance, and runs log-domain Sinkhorn (epsilon 0.005, 10 iterations). SciPy
linear assignment maximizes mass in the resulting plan to obtain a one-to-one
permutation. Every sampled noise tensor is preserved, rather than averaged into
a barycenter. Singleton classes retain their original pairing. This is a
minibatch approximation, not global dataset OT or a guarantee of no crossing.

### Reverse CE implementation: important reproduction detail

The reported CE groups explicitly use **`--ce-logits center`**, not the trainer's
legacy default `posterior`. Let `c_k=mu_k/||mu_k||`. Every fourth update, 16 real
training images are integrated backwards with **4 differentiable Euler steps**.
The auxiliary loss is `0.001 * cross_entropy(z_hat.flatten(1) @ c.T, labels)`.
Dropout is disabled for this solve and its backward pass; gradients reach the
same U-Net. Flow matching still trains every update with normal training dropout.

For the current centers/sigma, Gaussian posterior logits equal 48 times these
projection logits, up to a class-independent offset. Their argmax agrees, but
their CE scale/temperature and optimization do not. Do not substitute one for
the other while claiming to reproduce the table. Evaluation uses Heun60, so
the train/eval integration discretizations also differ.

## Environment

Recorded environment: Python 3.8.10, CUDA 11.8, PyTorch 2.0.0+cu118,
torchvision 0.15.1+cu118. Install a compatible NVIDIA driver separately.

```bash
python3.8 -m venv .venv-rf
source .venv-rf/bin/activate
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r strong_rf/requirements.txt
python scripts/prepare_cifar10.py --data-root ./data
python -m unittest discover -s tests -p 'test_*rf*.py' -v
```

Run all commands from the repository root. This study requires CUDA for training
and full Inception evaluation. CPU supports unit tests and the small sampling
CLI, albeit slowly. Different hardware/software can change results; a common
seed is not a promise of bitwise identical training.

## Reproduce training (one process per GPU)

```bash
# Choose one of: label gaussian gaussian_ot gaussian_ce gaussian_ot_ce.
CUDA_VISIBLE_DEVICES=0 DATA_ROOT=./data KIND=gaussian_ot_ce \
  STEPS=500000 bash scripts/train_strong_rf.sh

# Continue the same run, keeping all optimizer/RNG/EMA state.
CUDA_VISIBLE_DEVICES=0 DATA_ROOT=./data KIND=gaussian_ot_ce \
  STEPS=1000000 RESUME=1 bash scripts/train_strong_rf.sh
```

The wrapper explicitly sets the reported CE options for both CE groups and
rejects a missing full checkpoint when resuming. Change `KIND` and use a distinct
`OUTPUT` for each method; never load a different kind into the same run.
The checked-in [configs](../configs/strong-rf/) archive the actual study settings.
The wrapper runs training and previews; full metrics are a separate command below.

Shared settings: batch 128; Adam (not AdamW), LR 2e-4, 5K linear warmup, EMA
0.9999, gradient norm clipping at 1, FP32, TF32 disabled, seed 0. Images are
normalized to [-1,1], horizontal flips only, no crop/padding. Parameters and
data permutations are matched, but OT/CE and dropout consume random numbers
differently, so not every later noise/time draw is identical across methods.

Every 25K the wrapper stores an EMA snapshot and a 30-step Heun preview. Full
`training-latest.pt` holds model, EMA, Adam and RNG states; `training-previous.pt`
is its prior recovery point. `model-STEP.pt` is for inference, not full resume.
`STEPS` is a cumulative optimizer-step target, not additional updates.

The original `run_five_gpu.py` and `run_study.py` are archived orchestration
code; their GPU mapping, data paths, quality gates and 20GiB disk guard reflect
the original server. Use the portable single-GPU commands here on other hosts.

## Inference without Inception or the evaluation classifier

```bash
python -m strong_rf.sample --checkpoint runs/strong-rf/gaussian/model-500000.pt \
  --output samples/gaussian-500k --steps 60
python -m strong_rf.sample --checkpoint runs/strong-rf/gaussian/model-500000.pt \
  --output samples/reverse-500k --reverse-data-root ./data --limit 10000 --steps 60
```

The first command generates ten images per class. The second reverses real
CIFAR-10 test images and writes accuracy, per-image predictions and a confusion
matrix; ground-truth labels are used only after integration to score predictions.
EMA is loaded in both cases. The sample CLI's generation seed layout is designed
for a compact grid, not a replacement for the batch-seeded formal metric command.

## Reproduce the reported evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m strong_rf.evaluate \
  --checkpoint runs/strong-rf/gaussian/model-500000.pt \
  --output runs/strong-rf/evaluations/gaussian-500000-50k \
  --data-root ./data --cache runs/strong-rf/reference-cache \
  --samples 50000 --batch-size 64 --method heun --steps 60 \
  --seed 20260912 --reverse-images 10000 \
  --classifier results/strong-rf/reference-classifier.pt
```

For `label`, change the checkpoint/output and set `--reverse-images 0`.
Use separate output directories for every checkpoint/protocol. Initial Inception
weights may download automatically. Evaluation caches original train features
and generated batches; allow approximately 0.75GiB per 50K generation evaluation,
plus shared reference features and weights. No training occurs in this command.

- Formal FID: torch-fidelity Inception-v3-compatible 2048 features, FP32,
  50K generated versus all **50K original training images**. IS uses the same
  generated images. Main-table FID is not clean-fid's default clean mode.
- Quantization: `(x*127.5+128).clamp(0,255).uint8`. No resizing or random crop.
- KID currently reported by this evaluator is **legacy 2560 generated vs 5000
  test reference images**, 100 subsets of up to 1000, seed 2020. It is explicitly
  labeled in exported data; it is **not KID50K**, despite the generation run size.
- Generated accuracy uses the included fixed, independently trained small
  CIFAR residual classifier (92.36% real-test accuracy, selected epoch 44).
  It is not WRN and does not guide sampling or reverse classification. Its
  imperfect accuracy makes generated-label accuracy a proxy, not ground truth.
- Reverse accuracy: all 10K unaugmented test images, Heun60 / NFE120, no labels
  to the velocity model. No test-time augmentation or external classifier.
- Checkpoint hashes and exact evaluation protocols accompany each result.
  Optional `--cross-check` adds legacy-TensorFlow clean-fid as a separate metric.

## Results, availability and limits

See [RESULTS.md](../results/strong-rf/RESULTS.md) for common-step tables, latest
completed metrics, grids, full CSV/JSON history, and checkpoint fingerprints.
The frozen export contains completed evaluations only; stale STATUS files do
not demonstrate a running job or completed 1M training.

The exact **11MB evaluation classifier** is included, with its hash. Large flow
EMA weights (~137MiB each), full optimizer checkpoints, datasets and generated
50K caches are **not uploaded in this branch**. Retrain using the commands above
or obtain the matching checkpoint from the experiment owner and verify its hash.
The result images/JSON allow inspection without the flow weights; exact published
sample regeneration still requires the matching weights. Only load trusted `.pt`
files: full resume checkpoints use Python pickle.

Interpretation limits: one training seed; different sigma for Label (1) and
Gaussian (0.5); multiple comparison choices; limited evaluator accuracy. The
new results also changed model size and evaluation protocol relative to the
old backbone, so do not attribute all FID improvement to architecture alone.
Neither 95% reverse accuracy nor SOTA equivalence has been demonstrated.
