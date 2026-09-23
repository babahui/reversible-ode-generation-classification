# CIFAR-100 DCT Gaussian Pure-FM Baseline

这是已运行的 CIFAR-100 实验的独立复现入口。使用一个 TorchCFM U-Net，正向 ODE
生成图片，反向 ODE 得到噪声，再用固定 Gaussian 中心分类。不输入预测/真实标签给
速度网络，不使用 posterior head、外部分类器引导、CE、OT、history 或预训练权重。

**原实验训练至 450K；完整评估到 400K。这里提供训练至 500K 的命令，不把它当作已完成结果。**

- [已验证结果、协议及生成图](results/README.md)
- [给另一个执行者的完整训练 prompt](TRAINING_PROMPT_ZH.md)
- [模型来源及 MIT 许可证](../strong_rf/vendor_unet/PROVENANCE.md)
- [发布验证记录与尚未重跑的项目](VERIFICATION.md)

## 1. 实验设置

| 项目 | 设置 |
|---|---|
| 数据 | CIFAR-100 fine labels 0-99；RGB 32x32；训练 50,000，测试 10,000 |
| 预处理 | 像素映射到 [-1,1]；训练仅随机水平翻转 p=0.5；无裁剪 |
| 网络 | TorchCFM U-Net，35,746,307 参数，base128，mult=[1,2,2,2] |
| 结构 | 每层 2 residual blocks；16px 与 bottleneck attention；dropout=0.1 |
| 时间输入 | 标量 t，经网络时间嵌入；不是 alpha=[t,1-t] |
| 初始分布 | z\|y ~ N(mu_y, 0.25 I)，100 个固定正交低频 DCT 中心，范数 12 |
| Pairing | 同类 noise 与真实图片独立配对，不做 OT |
| 优化器 | Adam，beta=(0.9,0.999)，eps=1e-8，无 weight decay |
| Batch / LR | 128；2e-4；前 5K 线性 warmup，之后固定 |
| 数值 | FP32，TF32 关闭，gradient norm clip=1 |
| EMA / seed | EMA=0.9999；初始化 seed=0，训练随机流 seed=12345 |
| 保存 | 每 5K 完整恢复状态；每 25K EMA 快照及 Heun30 预览 |
| 正式评估 | EMA、Heun60、batch64、seed=20260912；50K 生成、10K 测试分类 |

标签只用来选择初始 Gaussian，不送入 U-Net。训练每个 epoch 随机打乱，batch128
时每轮 390 个 batch，丢弃该轮末尾 80 张；不是永久丢弃固定 80 张。

## 2. 核心实现

### DCT Gaussian 中心

[`cifar100_prior.py`](cifar100_prior.py) 枚举 `(channel,u,v)`，channel=0..2、
u,v=0..5，按 `(u*u+v*v,u,v,channel)` 排序，取前 100 个模式，依次对应 fine label。
每个模式仅在一个 RGB 通道上放置二维 DCT pattern，其余通道为零。
将整个 `[3,32,32]` pattern 做 L2 归一化，再乘以 12，得到 `mu_y`。

```python
z = mu_y + 0.5 * torch.randn_like(mu_y)
```

低频的是**均值**，随机噪声仍是 3072 维白噪声；既不是 N(0,I)，也不是对随机噪声低通滤波。
类别先验均匀。100 类中心不是简单复制 CIFAR-10 的 10 类中心，不宜把两组实验解释成只改变类别数量。

### 训练

[`train.py`](train.py) 采样 `t ~ U(0,1)`，给定图片 x 和类别 y：

```python
z = prior.sample(y)
x_t = (1 - t) * z + t * x
prediction = model(t, x_t, y=None)  # [B,3,32,32] velocity
loss = ((prediction - (x - z)) ** 2).mean()
```

优化的是速度回归，而不是预测图片或类别 logits；本实验没有额外反向分类训练损失。

### 生成与分类

[`common.py`](common.py) 中同一速度场 `dx/dt=v_theta(t,x)`：

- 生成：指定 y，只据此采样 z，然后从 t=0 积分至 1，得到图片。
- 分类：真实图片作为 t=1 状态，从 1 积分至 0 得到 z_hat；全程 `y=None`。
- 分类分数为 `-||z_hat-mu_k||^2/(2*sigma^2)+log(pi_k)`，取 argmax。
  `log_posterior()` 返回省去公共项的 logits；softmax 后才是归一化概率。

无 history 的连续 ODE 在通常的解唯一性条件下定义可逆流；有限步数求解会有数值误差。
Gaussian 后验是模型后验，并不保证与真实数据后验相同，也不保证高测试准确率。

## 3. 安装与数据

原实验环境：Python 3.8.10、PyTorch 2.0.0+cu118、torchvision 0.15.1+cu118。
以下所有命令在**仓库根目录**执行；需要兼容 CUDA 11.8 的 NVIDIA 驱动。

```bash
python3.8 -m venv .venv-cifar100
source .venv-cifar100/bin/activate
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r cifar100_fm/requirements.txt
python -m cifar100_fm.prepare_data --data-root ./data
python -m unittest discover -s tests -p 'test_cifar100_fm.py' -v
```

下载使用 torchvision 官方 CIFAR-100 接口及其校验；如果已有数据，传入包含
`cifar-100-python/` 的父目录。训练不自动下载。首次 FID 评估会下载 torch-fidelity
Inception 特征提取器，它仅用于评分，不用于初始化训练模型或指导分类。

## 4. 从头训练、恢复与定期评估

**推荐：一张空闲 GPU 上串行运行，每 50K 完整评估，再继续训练。**
选用自己的 GPU 编号；勿占用其他任务的 GPU。

```bash
CUDA_VISIBLE_DEVICES=0 python -m cifar100_fm.run_study \
  --data-root ./data --output ./runs/cifar100-dct-fm \
  --steps 500000 --eval-every 50000
```

重新运行同一命令会从完整训练状态恢复、复用生成缓存、跳过已完成且匹配的评估。
`--steps` 是总步数，不是额外增加步数。评估与训练串行，无后台自动启动其他实验。
如需延长到 1M，用同一 output、同一 eval-every 改为 `--steps 1000000`。
不要删除阶段快照后期待队列自动重建已过去的阶段。

只训练、不跑完整指标：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cifar100_fm.train \
  --data-root ./data --output ./runs/cifar100-train-only \
  --steps 500000 --keep-previous-checkpoint
# 中断后执行相同命令并追加 --resume。
```

`training-latest.pt` 保存网络、EMA、Adam、随机数状态；`training-previous.pt` 是上一次
恢复点；`model-STEP.pt` 只含 EMA/config/step，**用于推理，不是完整断点续训状态**。
只加载自己信任的 checkpoint：这些文件用 PyTorch pickle 格式。
数据路径可以迁移，恢复时必须指向相同的官方数据；不要修改 batch、prior、LR 等再声称严格续训。

硬件时间需实测。历史纯训练约 0.225 秒/step，500K 纯更新约 31.3 小时，
还要加保存、预览、特征提取与 10 次完整评估；不能承诺不同显卡同样用时。
用前 1K-5K 日志测速估算余时，并单独测一次评估耗时。原配置在单张 32GiB GPU 上运行。
50K 样本图像、2048 维特征及 Inception logits 缓存每次评估约 0.8GB，加快照和恢复点，
建议预留至少 30GB；不要在未确认结果完整前删除恢复状态或缓存。

## 5. 单独推理和完整指标

```bash
CKPT=./runs/cifar100-dct-fm/training/model-400000.pt

# 100 类，每类 10 张，输出 10 页 class grid；不需要训练数据。
CUDA_VISIBLE_DEVICES=0 python -m cifar100_fm.infer sample \
  --checkpoint "$CKPT" --output ./outputs/cifar100-grid

# 完整测试集反向分类，保存逐样本预测与混淆矩阵。
CUDA_VISIBLE_DEVICES=0 python -m cifar100_fm.infer classify \
  --checkpoint "$CKPT" --data-root ./data --output ./outputs/cifar100-classification

# 正式复现表格：50K 生成与完整 10K 反向分类。
CUDA_VISIBLE_DEVICES=0 python -m cifar100_fm.evaluate \
  --checkpoint "$CKPT" --data-root ./data --cache ./runs/cifar100-reference-cache \
  --output ./outputs/cifar100-400k-full \
  --samples 50000 --reverse-images 10000 --batch-size 64 --method heun --steps 60
```

快速 smoke test 可设置 `--samples 100 --reverse-images 100 --steps 2`，使用**不同输出目录**；
这些值不能与正式表格比较。`infer` 支持 `--device cpu`，正式评估和训练需要 CUDA。
独立 grid 使用一次固定随机流，正式评估用 seed+batch_offset；两者都正确，但不应逐像素相等。

FID 主指标：50K 生成 vs 全部 50K 原始训练图，torch-fidelity Inception 2048 维特征。
另报 10K 测试参考 FID，不能拿两种参考 FID 混比。IS 使用 Inception logits。
两种指标都不是类别匹配率；本实验不报告 KID 或外部分类器生成准确率。
固定 batch64：采样随机种子按 batch offset 定义，改变 batch 会改变这组生成样本。
ODE 内不做裁剪；最终 uint8 转换严格为 `(x*127.5+128).clamp(0,255).uint8`。
评估输出包含 checkpoint SHA256、协议、指标、逐类结果、生成缓存和可视化。

## 6. 发布范围与验证

保留了原实验的 prior、模型构造、FM 更新和 ODE/指标公式；整理改动包括 portable paths、
去掉未启用的 CE/OT 训练分支、严格恢复检查、新增独立推理及阶段执行入口。
U-Net 复用仓库内相同字节的 vendored 文件，未改 backbone，不另行安装最新版 TorchCFM。
代码兼容原实验 `config + ema` 权重。归档结果是原实验实测，不是此次重新训练所得。
环境、硬件差异可能改变数值，固定 seed 不等于跨环境逐位一致。
仓库不包含 CIFAR-100 图片数据、训练权重或 `.npz` 缓存；因此全新下载需从头训练后才能推理。
