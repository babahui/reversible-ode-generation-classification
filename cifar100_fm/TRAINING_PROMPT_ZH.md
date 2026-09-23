# CIFAR-100 从头训练执行 Prompt

请复现这个仓库的 **CIFAR-100 DCT Gaussian Pure-FM Baseline**，不要运行仓库中其他
CIFAR-10、CE、OT、history 或 SymmFlow 实验。仓库和分支：

https://github.com/babahui/reversible-ode-generation-classification

`experiment/cifar100-dct-gaussian-fm`

首先阅读 `cifar100_fm/README.md`、`cifar100_fm/results/README.md` 与
`strong_rf/vendor_unet/PROVENANCE.md`。所有命令从仓库根目录执行。

## 目标与不可更改的设置

从随机初始化训练 CIFAR-100 100 类，目标 **500,000 次 optimizer updates**，不是额外
500K，也不是 500 epochs。不加载任何预训练生成/分类权重。单网络正向生成、反向分类，
速度网络始终只接收状态和标量时间，`y=None`。标签仅用于采样对应初始 Gaussian。

- 使用仓库固定 TorchCFM U-Net，base128、mult=[1,2,2,2]、2 residual blocks、
  16px/bottleneck attention、dropout0.1、35,746,307 参数。
- 数据 50K train/10K test，32x32；fine labels；[-1,1]；训练只有 p=0.5 水平翻转。
- 按已实现排序取 100 个 RGB DCT 模式，中心范数12，sigma0.5；不要换中心映射。
- 独立同类配对，直线 FM 速度 MSE；不增加 CE、OT、JS、history、posterior head。
- Batch128，Adam LR2e-4，warmup5K 后恒定，EMA0.9999，FP32，TF32关闭，clip1，seed0。
- 完整恢复状态每5K保存，EMA快照/Heun30预览每25K，完整评估每50K。
- 评估 EMA/Heun60/batch64/seed20260912；50K 生成，每类500张；全部10K测试反向分类。
- FID 用原始50K训练图为参考，torch-fidelity Inception2048；另报测试参考FID及IS。
  没有独立 CIFAR-100 评分分类器，不要编造生成类别准确率，不用 CIFAR-10 评分器代替。

## 执行步骤

1. 检查 GPU、磁盘、现有进程。只选择空闲 GPU，不能终止/抢占已有任务。至少预留30GB，
   若资源不足先报告。检查 Python/CUDA 兼容性，创建独立环境，不覆盖其他任务的环境。
2. 如尚未克隆，执行：

```bash
git clone --branch experiment/cifar100-dct-gaussian-fm \
  https://github.com/babahui/reversible-ode-generation-classification.git
cd reversible-ode-generation-classification
python3.8 -m venv .venv-cifar100
source .venv-cifar100/bin/activate
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r cifar100_fm/requirements.txt
python -m cifar100_fm.prepare_data --data-root ./data
python -m unittest discover -s tests -p 'test_cifar100_fm.py' -v
```

3. 确认新输出目录不存在旧权重。从头启动以下单卡串行任务；GPU编号按实际空闲卡修改。
   长任务用 tmux 或环境支持的作业调度器保持运行，持续记录标准输出。

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cifar100_fm.run_study \
  --data-root ./data --output ./runs/cifar100-dct-fm \
  --steps 500000 --eval-every 50000
```

4. 同一命令可以在中断后恢复。不得把 `model-STEP.pt` EMA-only 文件作为完整训练恢复状态；
   `training-latest.pt` 才包含模型、EMA、Adam及随机数状态。不要自动删除旧实验目录。
5. 每50K完成评估后汇报：总step、完整10K反向准确率、FID(train50K)、FID(test10K)、IS、
   样本数量/solver、生成图路径、权重路径、实测速率与预计剩余时间。未完成写“进行中”。
   记录实际命令、git commit、GPU、依赖版本和协议，保证可追溯。
6. 训练时间不要直接承诺。历史更新约0.225秒/step，500K纯更新约31小时，仅供参考。
   用本机前1K-5K测速和首次完整评估耗时估算训练+评估总时间，并持续修正。
7. 500K后停止并汇总，不自动增加损失、修改LR或无限续训。比较归档50K-400K结果时
   保持指标协议一致，不把先前400K结果当作本次结果。保存每阶段分类混淆矩阵与生成页图。

单独复查完整指标：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cifar100_fm.evaluate \
  --checkpoint ./runs/cifar100-dct-fm/training/model-500000.pt \
  --data-root ./data --cache ./runs/cifar100-dct-fm/reference-cache \
  --output ./outputs/cifar100-500k-full \
  --samples 50000 --reverse-images 10000 --batch-size 64 --method heun --steps 60
```

不要上传训练集、测试集、访问凭据、大型 checkpoint 或缓存到 Git。
Inception 预训练权重只允许用于度量，不得作为生成/分类网络的初始化或条件。
