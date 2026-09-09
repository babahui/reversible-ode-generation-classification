# Reversible ODE Generation and Classification

This repository contains the reproducible CIFAR-10 experiment for a single
reversible ODE that generates images from class-conditioned low-frequency
Gaussian noise and classifies images by integrating the same flow backwards.
Training uses class-wise Sinkhorn OT pairing, conditional flow matching, and a
small reverse-ODE Gaussian-posterior cross-entropy (CE) term.

## Terminal quick start

Run these commands from this directory. The scripts are resumable and write
logs continuously, so a terminal reconnect does not lose the training state.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-eval.txt  # needed for FID/KID/IS
python scripts/prepare_cifar10.py --data-root ../all_datasets
python -m unittest discover -s tests -v

# DATA_ROOT can point to an existing CIFAR-10 directory; --download is not
# used by the release command so an accidental network failure cannot corrupt a run.
DATA_ROOT=/path/to/all_datasets DEVICE=cuda ./scripts/train_1m.sh
CHECKPOINT=/path/to/model-1000000.pt \
CLASSIFIER=/path/to/cifar10-classifier-best.pt \
DATA_ROOT=/path/to/all_datasets DEVICE=cuda ./scripts/evaluate_1m.sh
```

The main result is documented in [results/README.md](results/README.md). The
checkpoint handoff and checksum procedure are in [WEIGHTS.md](WEIGHTS.md).
Dataset download, verification, directory layout, and offline setup are in
[DATASET.md](DATASET.md).
完整的中文环境、续训、推理和指标比较流程见 [USAGE_CN.md](USAGE_CN.md)。
Do not commit `runs/`, datasets, or `.pt` files; `.gitignore` is configured for
that purpose. The local 1M checkpoint is at
`runs/cifar10-generation-only-lowfreq-ot-ce-5k/model-1000000.pt`.

To prepare the local commit and, when a token is available, create and push the
GitHub repository automatically:

```bash
GITHUB_TOKEN=ghp_... ./scripts/publish_github.sh
```

The default target is `babahui/reversible-ode-generation-classification`; set
`GITHUB_REPO=owner/name` to override it. Without a token the script only makes
the local commit and prints the exact next action.

For future long Codex terminal jobs, use the retry wrapper. It persists JSONL
events, captures the session ID, and resumes the same session after a temporary
stream failure:

```bash
./scripts/codex_resilient.sh "your task"
# or: ./scripts/codex_resilient.sh /path/to/task.md
```

Set `CODEX_MAX_RETRIES` to change the default five retries. For a single failed
interactive session, the built-in direct recovery is `codex resume --last`.

## Why the terminal task used to disconnect

The original working directory contains roughly 59 GB of historical runs and
many checkpoints. A recursive Git or upload command over that directory can
produce a long silent operation and exceed the terminal transport window. Work
from this subdirectory, use the resumable scripts, and inspect logs under the
selected output directory. GitHub authentication is intentionally not stored
in this project; set up `git` credentials before pushing.

## Experiment configuration

The exact 1M command-line configuration is versioned in
`configs/cifar10_lowfreq_ot_ce_1m.json`. The implementation is in
`unified_transport/`; training entry points are `train_generation_only.py` and
`train_reference_classifier.py`; inference entry points are
`sample_generation_only.py`, `evaluate_generation_only.py`,
`evaluate_reverse_margin.py`, and `evaluate_generative_quality.py`.

---

这是从原实验中整理出的独立实现。它保留了线性 stochastic interpolant / rectified
flow 的训练方式，但修正了多域混训、路径条件、EMA、归一化、数据配对和分类潜空间。
原目录中的 notebook、checkpoint 和结果均未修改。

多边缘定义参考 Albergo et al., [Multimarginal generative modeling with stochastic
interpolants](https://arxiv.org/abs/2310.03695)。本实现增加的历史扩展状态是针对路径身份的
实验假设，不是该论文原始 Markov 场定理的一部分。

## 数学接口

给定 `K` 个边缘样本 `(x_0,...,x_{K-1})` 和单纯形坐标 `alpha`：

```text
x(alpha) = sum_k alpha_k x_k
g_k(alpha, x) = E[x_k | x(alpha)=x]
b(t, x) = sum_k alpha_dot_k(t) g_k(alpha(t), x)
```

网络一次输出全部 `g_k`。沿边 `i -> j` 时，速度就是 `g_j - g_i`，所以同一网络可做
任意边缘的正向、逆向和 all-to-all 传输。训练同时覆盖有向边，并以一定概率采样单纯形
内部。实现对应 `unified_transport/interpolant.py`。
`TransportSampler.integrate_path` 还接受任意分段线性单纯形路径，可比较直达、经过重心或经过
第三个边缘的路径；命令行的 `transport` 是直线边路径的便捷接口。

路径扩展状态为：

```text
h_t = (x_previous, (x_t - x_previous) / delta_t)
```

它只使用过去信息，训练和采样定义一致，内存与 ODE 步数无关。但统一生成/分类依赖一个
可逆 Markov ODE，所以默认不启用历史；`--history` 仅用于路径 correspondence 消融，启用后
不能再把反向过程严格解释为前向生成映射的逆。

## 统一生成与判别

MNIST/CIFAR 模式只训练一个双向网络：

```text
image distribution <-> mixture of class-conditional Gaussian latents
```

每类潜分布具有正交均值但保留高斯类内随机量。分类时将图像传到潜空间，并计算
`argmax_y log p(z|y) + log p(y)`；生成时从指定 `p(z|y)` 采样，再用同一模型沿反向边积分。
这避免了旧代码将整类压到一个固定原型、反向生成没有多样性的问题。
严格可行性、Bayes 后验中 Jacobian 抵消的推导，以及标准 DDPM 为什么不满足该设定，见
[THEORY.md](THEORY.md)。

## 训练

在本目录执行：

```bash
python train.py \
  --mode mnist \
  --data-root ../all_datasets \
  --classes 1,2,3 \
  --image-size 28 \
  --output runs/mnist-123 \
  --download
```

CIFAR-10 将 `--mode` 改为 `cifar10`、`--image-size` 改为 `32`。图像和噪声在线归一化/采样，
不会预先写出成倍数据。
EMA 在前 1000 步默认直接同步当前模型，之后使用 `0.9999` 衰减；可通过
`--ema-warmup` 和 `--ema-decay` 调整，避免早期采样长期停留在随机初始化。

十分类长期实验可按约一小时一个阶段自动训练、保存和评估：

```bash
python run_periodic_experiment.py \
  --data-root ../all_datasets \
  --classifier runs/reference-mnist-classifier/mnist-classifier-best.pt \
  --output runs/mnist-10class-periodic \
  --stage-steps 50000 \
  --total-steps 200000
```

该运行器可从 `model-latest.pt` 自动恢复。每阶段输出命名 checkpoint，并用固定随机种子、固定
Heun ODE 步数和冻结的外部 MNIST 分类器更新 `evaluation/metrics.json`、指标曲线、分类生成网格、
循环重建网格和 ODE 路径网格。在当前 RTX 4080 SUPER 实测约 16 iteration/s，50000 步接近
一小时；其他硬件应先根据吞吐量调整 `--stage-steps`。

STL-10 原生分辨率实验使用 `--mode stl10`、`--image-size 96`。在单张 RTX 4080 SUPER
上，当前 UNet、batch 32 的吞吐约 10.6 step/s，75k 步约 2 小时；三种方法合计约 6 小时。
类别 Gaussian 的 `center_scale=18` 按像素 RMS 对齐 64x64 实验中的 `center_scale=12`。
建议先训练独立的 ImageNet 预训练 ResNet-18 评估器：

```bash
python train_reference_classifier.py --mode stl10 \
  --data-root ../all_datasets --output runs/stl10-reference-classifier-resnet18 \
  --architecture resnet18 --epochs 30 --batch-size 128 --learning-rate 1e-4 \
  --image-size 96
```

三种 STL-10 模型的共同配置如下（仅 `train.py` 的 Unified 保留 image->noise 方向）：

```bash
python train.py --mode stl10 --data-root ../all_datasets \
  --output runs/stl10-10class-unified --image-size 96 --center-scale 18 \
  --classes 0,1,2,3,4,5,6,7,8,9 --batch-size 32 --train-steps 75000 \
  --random-crop --save-every 5000

python train_generation_only.py --mode stl10 --data-root ../all_datasets \
  --output runs/stl10-10class-generation-only --image-size 96 --center-scale 18 \
  --classes 0,1,2,3,4,5,6,7,8,9 --batch-size 32 --train-steps 75000 \
  --save-every 5000

python train_label_conditioned.py --mode stl10 --data-root ../all_datasets \
  --output runs/stl10-10class-label-conditioned --image-size 96 \
  --classes 0,1,2,3,4,5,6,7,8,9 --batch-size 32 --train-steps 75000 \
  --save-every 5000
```

为让 Gaussian-only 的类别信号经过 UNet 下采样仍可见，可使用低频正交 DCT 中心：

```bash
python train_generation_only.py --mode stl10 --data-root ../all_datasets \
  --output runs/stl10-10class-generation-only-lowfreq --image-size 96 \
  --classes 0,1,2,3,4,5,6,7,8,9 --center-scale 18 \
  --center-mode low_frequency --latent-sigma 0.5 --batch-size 32 \
  --train-steps 75000 --save-every 5000
```

该模型的反向分类评估为 `t=1 -> 0` 的同一向量场积分，再用 Gaussian posterior
判别端点分量：

```bash
python evaluate_generation_only_reverse.py \
  --checkpoint runs/stl10-10class-generation-only-lowfreq/model-75000.pt \
  --classifier runs/stl10-reference-classifier-resnet18/stl10-classifier-best.pt \
  --data-root ../all_datasets \
  --output runs/stl10-10class-generation-only-lowfreq/reverse-evaluation \
  --steps 15 30 --method heun --max-test-images 1000
```

该低频中心实验的完整生成、反向分类、FID/KID/IS 和可视化见
[runs/stl10-10class-generation-only-lowfreq/RESULTS.md](runs/stl10-10class-generation-only-lowfreq/RESULTS.md)。
训练时还可通过 `--center-loss-weight` 打开 endpoint center projection 辅助项；默认值为 0，
因此不改变原有 Gaussian-only 目标。

先训练冻结的 CIFAR-10 评估分类器，再训练传输模型：

```bash
python train_reference_classifier.py --mode cifar10 \
  --data-root ../all_datasets --epochs 50 \
  --output runs/reference-cifar10-classifier

python train.py --mode cifar10 --data-root ../all_datasets \
  --classes 0,1,2,3,4,5,6,7,8,9 --image-size 32 \
  --base-channels 32 --batch-size 256 --train-steps 100000 \
  --output runs/cifar10-10class-periodic
```

当前 CIFAR-10 结果与限制见
[runs/cifar10-10class-periodic/STATUS.md](runs/cifar10-10class-periodic/STATUS.md)。
将类别 Gaussian 中心范数从 4 提高到 12 的单变量实验，将 100k 的 inverse 分类从
14.49% 提高到 64.15%，具体控制变量、生成指标和多样性权衡见
[runs/cifar10-center12-ablation/STATUS.md](runs/cifar10-center12-ablation/STATUS.md)。
进一步使用 base channel 64、CIFAR random crop 和 `0.001` endpoint class-center projection
loss 后，75k 完整测试集 inverse 分类达到 92.97%，条件生成类别准确率为 68.09%。配置、
理论含义、生成质量权衡和最终权重见
[runs/cifar10-performance-base64/STATUS.md](runs/cifar10-performance-base64/STATUS.md)。
保持十个类别 Gaussian 不变、只训练 noise -> image 单向 flow 的控制实验在 75k 达到
72.89% 生成类别准确率和 169.97 classifier-feature Frechet；它与统一模型的严格对比及限制分析见
[runs/cifar10-generation-only-base64/STATUS.md](runs/cifar10-generation-only-base64/STATUS.md)。
进一步将类别改为显式 label condition、初始噪声改为纯 `N(0,I)` 的对照在 75k 达到
80.90% 生成类别准确率和 120.68 classifier-feature Frechet，说明当前高斯中心隐式类别信号
是重要实践瓶颈；详情见 [runs/cifar10-label-conditioned-base64/STATUS.md](runs/cifar10-label-conditioned-base64/STATUS.md)。

为检验 minibatch coupling 是否造成路径混叠，`train_generation_only.py` 支持按类别的
Sinkhorn pairing：

```bash
python train_generation_only.py --mode cifar10 --data-root ../all_datasets \
  --output runs/cifar10-generation-only-lowfreq-ot --image-size 32 \
  --classes 0,1,2,3,4,5,6,7,8,9 --center-scale 12 \
  --center-mode low_frequency --latent-sigma 0.5 --pairing sinkhorn \
  --ot-epsilon 0.05 --ot-iterations 50 --base-channels 64 \
  --channel-mults 1,2,4 --batch-size 256 --train-steps 75000 \
  --path-diagnostics-every 1000 --save-every 5000
```

训练日志中的 `path_diagnostics` 按时间段记录局部 velocity variance、预测/目标方向余弦和
近邻方向冲突率。对已有 checkpoint 进行 independent/OT 的配对诊断：

```bash
python evaluate_pairing_diagnostics.py --checkpoint runs/.../model-10000.pt \
  --data-root ../all_datasets --output runs/.../pairing-diagnostics-10000
```

CIFAR-10 的弱 OT 5k/10k pilot 结果见
[runs/cifar10-generation-only-lowfreq-ot-pilot/RESULTS.md](runs/cifar10-generation-only-lowfreq-ot-pilot/RESULTS.md)。
更强的 `epsilon=0.005` 配对结果见
[runs/cifar10-generation-only-lowfreq-ot-eps005-10k-v2/RESULTS.md](runs/cifar10-generation-only-lowfreq-ot-eps005-10k-v2/RESULTS.md)，
其中同时保存了 10k 生成网格、反向 ODE 分类、FID/KID/IS/SSIM 和逐时间段路径诊断。

阶段 C/D 的 75k 对照实验已由
[`run_phase_c_d_75k.sh`](run_phase_c_d_75k.sh) 排队执行，设计和损失定义见
[PHASE_C_D_PLAN.md](PHASE_C_D_PLAN.md)。阶段 C 使用冗余低频 Hadamard 类别编码；阶段 D
在同一个 generation-only 速度场上加入 endpoint 和可微 reverse-ODE 类别投影损失，仍不向
U-Net 输入 label。

多图像域训练要求 `DATA_ROOT/domain_name/...`：

```bash
python train.py \
  --mode folders \
  --data-root /path/to/data \
  --domains cat,dog,wild,flowers,celeba \
  --image-size 64 \
  --output runs/five-domains
```

folder 模式按批次从各域独立采样，不创建笛卡尔积。独立耦合只保证边缘分布传输；若要求
同一对象的语义对应，应让 Dataset 返回真实配对/多元组，或加入 OT/data-dependent
coupling。低训练损失本身不能证明语义 correspondence。

## 采样与评估

指定类别生成：

```bash
python sample.py \
  --checkpoint runs/mnist-123/model-latest.pt \
  --steps 50 \
  generate --class-label 2 --num 16 --output samples/digit-2.png
```

单张或多张图片执行 `image -> noise` 并输出 Bayes 后验：

```bash
python sample.py \
  --checkpoint runs/mnist-123/model-latest.pt \
  --steps 50 \
  classify --input digit-a.png digit-b.png --save-latent samples/encoded.pt
```

多域图像传输，域编号与 `--domains` 顺序一致：

```bash
python sample.py \
  --checkpoint runs/five-domains/model-latest.pt \
  transport --input cat.jpg --source 0 --target 1 --output dog.png
```

贝叶斯分类准确率：

```bash
python evaluate.py \
  --checkpoint runs/mnist-123/model-latest.pt \
  --data-root ../all_datasets
```

输出包含分类准确率、latent posterior NLL 和 `image -> noise -> image` cycle MSE。

无配对生成质量使用：

```bash
python evaluate_generative_quality.py \
  --checkpoint runs/stl10-10class-generation-only/model-75000.pt \
  --data-root ../all_datasets --output runs/stl10-quality/gaussian-only \
  --generated-per-class 500 --real-images 5000 --steps 30 --method heun
```

该脚本输出 FID、KID、Inception Score。由于没有真实配对图像，SSIM 不作为重建质量；脚本中的
`generated_pair_ssim`/`real_pair_ssim` 是类内成对 SSIM 的多样性诊断（越低通常表示更丰富），
`nearest_train_ssim` 仅用于近邻相似度/潜在记忆检查。用 `summarize_quality_metrics.py` 可将三组
JSON 合并为 `QUALITY.md` 和对比图。

MNIST 的 alpha/path 消融可复现为：

```bash
python train.py --mode mnist --data-root ../all_datasets \
  --classes 0,1,2,3,4,5,6,7,8,9 --no-alpha-condition \
  --base-channels 32 --batch-size 256 --train-steps 50000 \
  --output runs/mnist-no-alpha-ablation

python evaluate_alpha_ablation.py \
  --checkpoint runs/mnist-10class-periodic/model-100000.pt \
  --no-alpha-checkpoint runs/mnist-no-alpha-ablation/model-50000.pt \
  --classifier runs/reference-mnist-classifier/mnist-classifier-best.pt \
  --data-root ../all_datasets --output runs/mnist-alpha-ablation
```

结论、逐项定义和结果见
[runs/mnist-alpha-ablation/STATUS.md](runs/mnist-alpha-ablation/STATUS.md)。

## 三数据集四方法汇总

CIFAR-10 (32x32)、Tiny-ImageNet-10 (64x64) 和 STL-10 (96x96) 的 75k-step
对比已经统一汇总。方法包括 `Unified`、原始 `Gaussian-only`、低频正交中心的
`Gaussian-only` 和 `Label-conditioned`，同时报告生成类别准确率、反向 ODE 分类、
FID、KID、IS，并固定 30-step Heun 采样生成可视化网格。

结果、原始 JSON/CSV 和图表见
[runs/three-dataset-comparison/README.md](runs/three-dataset-comparison/README.md)：

- [指标对比图](runs/three-dataset-comparison/metric-comparison.png)
- [checkpoint 生成准确率曲线](runs/three-dataset-comparison/checkpoint-accuracy-curves.png)
- [相同 75k steps 的生成网格](runs/three-dataset-comparison/final-generation-grids.png)

重新汇总已有结果：

```bash
python compare_three_datasets.py --output runs/three-dataset-comparison
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 实验要求

至少报告以下消融：`history on/off`、边/域条件 on/off、Euler/Heun、ODE 步数、独立/语义或
OT coupling。多域传输报告每个有向边的 FID/KID 和域分类准确率；统一任务同时报告 CIFAR-10
分类准确率、每类生成 FID、类条件准确率和重建 cycle error。路径交叉应在 toy 数据上报告
交叉邻域内的方向余弦误差，而不是仅看训练 MSE。
