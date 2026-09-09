# 使用说明：从 1M 继续训练到 10M、推理与指标复现

本文档对应仓库中的 CIFAR-10 实验：低频 Gaussian 类条件噪声、类内 Sinkhorn OT 配对、
conditional flow matching，以及 reverse-ODE Gaussian posterior CE 辅助损失。

## 1. 代码在哪里

| 用途 | 文件 |
| --- | --- |
| 主训练入口 | `train_generation_only.py` |
| 低频 Gaussian 类中心和采样 | `unified_transport/latent.py` |
| Sinkhorn OT 类内配对 | `unified_transport/ot_pairing.py` |
| flow matching、CE 和 ODE sampler | `unified_transport/generation_only.py` |
| U-Net/向量场 | `unified_transport/model.py` |
| 生成采样 | `sample_generation_only.py` |
| 生成类别准确率 | `evaluate_generation_only.py` |
| 反向 ODE 分类 | `evaluate_reverse_margin.py` |
| FID/KID/IS/SSIM | `evaluate_generative_quality.py` |
| 1M 配置 | `configs/cifar10_lowfreq_ot_ce_1m.json` |
| 一键训练 | `scripts/train_1m.sh` |
| 从 1M 续训 | `scripts/continue_from_1m.sh` |
| 1M 分阶段训练到 10M | `scripts/train_to_10m.sh` |
| 一键评估 | `scripts/evaluate_1m.sh` |
| 1M 到 10M 里程碑评估 | `scripts/evaluate_milestones.sh` |

## 2. 环境设置

建议使用 Linux + NVIDIA GPU。CPU 可以运行单元测试和小规模 smoke test，但不适合完整
1M 训练。仓库提供 `environment.yml`；也可以用 Python venv：

```bash
git clone https://github.com/babahui/reversible-ode-generation-classification.git
cd reversible-ode-generation-classification

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-eval.txt  # FID/KID/IS/SSIM 评估需要
python -m unittest discover -s tests -v
```

如果使用 conda：

```bash
conda env create -f environment.yml
conda activate reversible-ode-gen-class
```

## 3. 准备 CIFAR-10

数据集不在 GitHub 中。自动下载并校验：

```bash
python scripts/prepare_cifar10.py --data-root /path/to/all_datasets
```

校验通过后应有 50,000 张训练图、10,000 张测试图和 10 个类别。服务器无网络时，把
`cifar-10-python.tar.gz` 或解压后的 `cifar-10-batches-py/` 放入数据根目录，再执行：

```bash
python scripts/prepare_cifar10.py \
  --data-root /path/to/all_datasets --verify-only
```

详细目录结构和下载地址见 [DATASET.md](DATASET.md)。

## 4. 下载并放置 1M 权重

GitHub 不存放 133 MB 的 checkpoint。将网盘中的主权重下载到任意位置，例如：

```text
weights/model-1000000.pt
```

当前 1M 权重下载地址：<https://pan.quark.cn/s/67b5254c9276>（夸克网盘文件名
`model-1000000.pt`）。下载后请按下方 SHA256 校验；不要把权重提交到 GitHub。

主权重的 SHA256 应为：

```text
fbd3a87ef31f3daf0aa862386a61d874129e18bf5bc79b82cf6a7d8b58ba682b
```

```bash
sha256sum weights/model-1000000.pt
```

生成类别准确率和 FID/KID/IS 还需要冻结的 reference classifier：

```text
weights/cifar10-classifier-best.pt
```

两个文件的来源和校验值见 [WEIGHTS.md](WEIGHTS.md)。

## 5. 单卡 batch size 与从头训练到 1M

当前实验在约 32 GB 显存的 RTX 4080 SUPER 上验证了 `batch size=256`，因此脚本默认值为
256。显存更小的卡请先用 128、64 或 32；显存更大的卡可以逐步试探。推荐先运行 2 步探测，
把仍能完成且没有 CUDA out-of-memory 的最大值设为 `BATCH_SIZE`：

```bash
for bs in 512 384 256 192 128 64 32; do
  echo "testing batch size $bs"
  BATCH_SIZE=$bs DATA_ROOT=/path/to/all_datasets OUTPUT="runs/batch-probe-$bs" \
    DEVICE=cuda TARGET_STEPS=2 CHECKPOINT=/path/to/model-1000000.pt \
    ./scripts/continue_from_1m.sh && break
done
```

探测时如果某个值 OOM，降低到下一个值；正式训练时固定该值，不要在不同里程碑之间改变。
`BATCH_SIZE`、`WORKERS`、`SAVE_EVERY` 均可通过环境变量覆盖。

```bash
DATA_ROOT=/path/to/all_datasets \
OUTPUT=runs/cifar10-generation-only-lowfreq-ot-ce-5k \
DEVICE=cuda BATCH_SIZE=256 \
./scripts/train_1m.sh
```

脚本会在线下载/校验数据，使用固定的低频 Gaussian + OT + CE 配置，并每 5,000 步保存：

```text
runs/cifar10-generation-only-lowfreq-ot-ce-5k/model-5000.pt
runs/cifar10-generation-only-lowfreq-ot-ce-5k/model-latest.pt
```

中断后重新执行同一命令会从 `model-latest.pt` 恢复。训练日志在该目录的 `train.log`。

## 6. 从 1M 继续训练

把 1M checkpoint 放在 `OUTPUT/model-latest.pt`，或显式指定路径。默认继续到 1.2M：

```bash
DATA_ROOT=/path/to/all_datasets \
OUTPUT=runs/cifar10-generation-only-lowfreq-ot-ce-5k \
CHECKPOINT=/path/to/model-1000000.pt \
TARGET_STEPS=1200000 \
DEVICE=cuda \
./scripts/continue_from_1m.sh
```

例如继续到 1.5M，只需改 `TARGET_STEPS=1500000`。每个目标步数应使用新的输出目录保存，
便于比较：

```bash
OUTPUT=runs/cifar10-generation-only-lowfreq-ot-ce-1500k \
CHECKPOINT=/path/to/model-1000000.pt TARGET_STEPS=1500000 \
DATA_ROOT=/path/to/all_datasets DEVICE=cuda ./scripts/continue_from_1m.sh
```

续训不会重置模型、EMA、latent 几何或 optimizer；`--resume` 会恢复 checkpoint 中的训练状态。

## 7. 持续训练到 10M，每 1M 保存一次

仓库提供阶段化运行器。它从 1M 开始依次训练到 2M、3M，直到 10M，每一阶段使用独立目录，
因此中断后可从最近的 `model-latest.pt` 继续：

```bash
DATA_ROOT=/path/to/all_datasets \
CHECKPOINT=/path/to/model-1000000.pt \
OUTPUT=runs/cifar10-generation-only-lowfreq-ot-ce-10m \
START_STEPS=1000000 END_STEPS=10000000 STAGE_STEPS=1000000 \
BATCH_SIZE=256 DEVICE=cuda \
./scripts/train_to_10m.sh
```

每个阶段的权重和日志位于 `OUTPUT/step-2000000/`、`OUTPUT/step-3000000/` 等目录。若任务中断，
可将最近阶段的 `model-latest.pt` 作为 `CHECKPOINT`，把 `START_STEPS` 改为该 checkpoint 的步数，
再运行同一脚本。训练脚本不会自动计算昂贵的 FID/KID/IS；每到一个整百万步，应等待该阶段完成，
然后用下一节的固定参数评估并保存 JSON。

## 8. 生成推理

生成 10 个类别、每类 16 张图片：

```bash
python sample_generation_only.py \
  --checkpoint weights/model-1000000.pt \
  --output samples/generated-1m.png \
  --per-class 16 --steps 30 --method heun --device cuda
```

输出网格按类别排列，像素已转换到 `[0, 1]` PNG。

## 9. 查看指标

### 8.1 生成类别准确率和重建

需要 reference classifier：

```bash
python evaluate_generation_only.py \
  --checkpoints weights/model-1000000.pt \
  --classifier weights/cifar10-classifier-best.pt \
  --data-root /path/to/all_datasets \
  --output evaluation/1m-generation \
  --steps 30 --method heun --batch-size 256 \
  --real-images 10000 --generated-per-class 256 \
  --visuals-per-class 10 --workers 4 --device cuda
```

主要结果在 `evaluation/1m-generation/metrics.json`，其中的 `generated_accuracy` 是条件
生成类别准确率。

### 8.2 反向 ODE 分类准确率

```bash
python evaluate_reverse_margin.py \
  --checkpoint weights/model-1000000.pt \
  --data-root /path/to/all_datasets \
  --output evaluation/1m-reverse \
  --steps 15 30 60 --method heun \
  --batch-size 128 --max-test-images 10000 --workers 4 --device cuda
```

查看 `evaluation/1m-reverse/metrics.json` 中每个步数的 `accuracy` 数组最后一个值；30 步
Heun 是当前结果的比较基准。还可以查看 `posterior-margin.png` 和每步路径统计。

### 8.3 FID、KID、IS 和 SSIM

```bash
python evaluate_generative_quality.py \
  --checkpoint weights/model-1000000.pt \
  --data-root /path/to/all_datasets \
  --output evaluation/1m-quality \
  --generated-per-class 256 --real-images 5000 \
  --batch-size 64 --steps 30 --method heun \
  --ssim-per-class 32 --ssim-real-candidates 200 \
  --workers 4 --device cuda
```

查看 `evaluation/1m-quality/quality-metrics.json`：

```text
fid_inception
kid_mean / kid_std
inception_score_mean / inception_score_std
generated_pair_ssim       # 多样性诊断，越低不代表质量越差
nearest_train_ssim        # 训练集相似度/潜在记忆诊断
```

也可以一次比较多个 checkpoint：

```bash
python evaluate_generation_only.py \
  --checkpoints weights/model-1000000.pt weights/model-1200000.pt \
  --classifier weights/cifar10-classifier-best.pt \
  --data-root /path/to/all_datasets --output evaluation/checkpoints \
  --steps 30 --method heun --device cuda
```

## 10. 每 1M 汇报和判断是否提升

训练完成后，可以自动对 `1M, 2M, ..., 10M` 每个 checkpoint 运行全部评估：

```bash
DATA_ROOT=/path/to/all_datasets \
CHECKPOINT_DIR=runs/cifar10-generation-only-lowfreq-ot-ce-10m \
INITIAL_CHECKPOINT=/path/to/model-1000000.pt \
CLASSIFIER=/path/to/cifar10-classifier-best.pt \
OUTPUT=runs/cifar10-generation-only-lowfreq-ot-ce-10m/evaluation \
DEVICE=cuda ./scripts/evaluate_milestones.sh
```

该命令会为每个 checkpoint 建立同名目录，例如：
`evaluation/step-2000000-generation`、`evaluation/step-2000000-reverse`、
`evaluation/step-2000000-quality`。每个里程碑都运行 9.1、9.2、9.3 三条命令，记录：

```text
step, generated_accuracy, reverse_ode_accuracy_30_heun,
fid_inception, kid_mean, kid_std, inception_score_mean, inception_score_std,
generated_pair_ssim, nearest_train_ssim
```

可将这些 JSON 汇总为 CSV 或表格后画出随训练步数变化的曲线。只有在数据集、样本数量、随机种子、
ODE 步数/solver、batch size 和 reference classifier 都一致时，才把指标变化解释为训练继续带来的
变化；FID/KID 越低通常越好，生成准确率、反向 ODE 准确率和 IS 通常越高越好，SSIM 主要用于监测
多样性和记忆风险。

对 1M、2M、...、10M 分别运行同样的评估参数，比较：

1. `generated_accuracy`：条件生成类别是否更准确。
2. 反向 ODE `accuracy[-1]`：图像到 Gaussian 的分类是否更准确。
3. `fid_inception`、`kid_mean`：分布质量是否改善，通常越低越好。
4. `inception_score_mean`：类别可分性和视觉质量的辅助指标，通常越高越好。
5. `generated_pair_ssim`、`nearest_train_ssim`：多样性与记忆风险，不能只看单一指标。

必须保持数据集、随机种子、生成数量、ODE 步数、solver 和 reference classifier 不变，
否则不同 checkpoint 的指标不能直接比较。当前 1M 基准记录在
`results/metrics_1m.json`：生成准确率 82.66%、反向 ODE 分类准确率 90.23%、FID 53.25769、
KID 0.05192797、IS 6.56545。
