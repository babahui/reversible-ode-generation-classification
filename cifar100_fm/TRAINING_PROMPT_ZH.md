# 两组 CIFAR-100 / 5000K 实验执行 Prompt

请按照以下要求运行两组 CIFAR-100 实验。先阅读仓库文档、检查资源，再开始执行。

仓库：https://github.com/babahui/reversible-ode-generation-classification

分支：`experiment/cifar100-dct-gaussian-fm`

必读：`cifar100_fm/README.md`、`cifar100_fm/DUAL_PAD4_JS.md`、
`cifar100_fm/VERIFICATION.md`、`cifar100_fm/results/README.md`。
不要运行仓库内无关的 CIFAR-10 / SymmFlow / OT / history 脚本。

## 1. 目标和计数

- A：DCT Gaussian + pure FM，目标总 **5,000,000 optimizer updates（5000K）**。
- B：相同 DCT Gaussian + FM + 双 pad4 reverse CE + JS，也到总 **5,000,000 updates**。
- 推荐严格沿用原 CIFAR-10 初始化方式：从随机初始化训练公共纯FM 1M，
  A继续纯FM至5M，B加载该1M的完整model/EMA/Adam/RNG，再训练4M至总5M。
- 公共1M也必须在本次CIFAR-100上从头训练，不使用外部预训练生成模型或分类器。
  B不是加载EMA重新初始化Adam，也不是额外训练5M到总6M。
- 如果用户明确要求“两个方法都从第1步就使用各自损失”，才使用下文的from-scratch B命令。
  将它标为另一个初始化协议，不要与1M分叉方案混报或同时启动第三组。

## 2. 固定设置

两组相同：CIFAR-100 fine labels，训练50K/测试10K，RGB32x32，[-1,1]，
FM仅水平翻转p=.5；固定100个DCT中心norm12、sigma.5；independent pairing。
网络为vendored TorchCFM U-Net base128，35,746,307参数，scalar-t，不输入类别。
FM batch128，Adam LR2e-4，初始5K warmup后恒定，EMA.9999，FP32，TF32关闭，clip1，seed0。

B额外设置，不能擅自改变：

- 每4次update取当前FM batch前32张，每张生成2个独立reflect-pad4裁剪视角，共CE64。
- 两个视角都裁剪回32x32，裁剪随机流seed20260918，单独保存/恢复，不修改FM图片。
- reverse Euler4可微积分，dropout关闭，`model(t,x,y=None)`，无真实/预测label条件。
- CE为固定Gaussian单位中心投影scores的交叉熵，不是posterior logits；权重.001，warmup2K。
- JS为两个视角softmax(scores/T)之间Jensen-Shannon散度，T=2，外乘T²，权重.001，warmup5K。
- 两个视角都反传梯度。CE/JS warmup按B训练开始之后的updates计算，不在续训时重置。
- 梯度诊断每500步；不加posterior head、OT、history或其它损失。

## 3. 环境、资源和数据

检查空闲GPU和磁盘，不停止已有任务，不覆盖其他实验输出。原配置单卡32GiB可运行；
两组保留全部50K评估缓存和100K快照建议至少150GB可用磁盘。
若资源不足先报告，不静默减小batch、不删除其他实验。脚本低于15GiB会退出保护。
创建独立环境。所有命令从仓库根目录运行：

```bash
git clone --branch experiment/cifar100-dct-gaussian-fm \
  https://github.com/babahui/reversible-ode-generation-classification.git
cd reversible-ode-generation-classification
python3.8 -m venv .venv-cifar100
source .venv-cifar100/bin/activate
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r cifar100_fm/requirements.txt
python -m cifar100_fm.prepare_data --data-root ./data
python -m unittest discover -s tests -p 'test_cifar100*.py' -v
```

如果已有仓库/环境/官方数据，先核对再复用，不要盲目覆盖。
首次FID下载的Inception权重仅作评估特征提取，不得用来初始化模型或引导生成/分类。

## 4. 运行顺序

先训练公共1M；GPU编号按实际空闲设备替换。长任务使用tmux或作业调度器，保存stdout日志。

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cifar100_fm.run_study \
  --kind gaussian --data-root ./data --output ./runs/cifar100-fm-5m \
  --steps 1000000 --eval-every 100000
```

确认 `./runs/cifar100-fm-5m/training/training-1000000.pt` 完整保存后，两张卡分别执行：

```bash
# A：同一纯FM训练继续，总目标5M。
CUDA_VISIBLE_DEVICES=0 python -u -m cifar100_fm.run_study \
  --kind gaussian --data-root ./data --output ./runs/cifar100-fm-5m \
  --steps 5000000 --eval-every 100000

# B：由相同1M完整状态分叉，引入CE+JS，总目标5M。
CUDA_VISIBLE_DEVICES=1 python -u -m cifar100_fm.run_study \
  --kind gaussian_dual_pad4_js --data-root ./data --output ./runs/cifar100-dual-pad4-js-5m \
  --finetune-from ./runs/cifar100-fm-5m/training/training-1000000.pt \
  --finetune-start 1000000 --steps 5000000 --eval-every 100000
```

两条命令在不同终端/作业中运行才会并行；只有一张空闲卡时串行执行。
不得误用 `model-1000000.pt`：它只有EMA，不能恢复Adam/RNG。
公共完整 `training-1000000.pt` 必须保留，不能被后续latest覆盖或删除。
中断后重新运行同一命令，脚本会恢复训练、复用缓存并核对已完成评估，不要新建重复实验。

仅当用户指定B从第1步使用CE+JS，改用下面命令替代上面的B，仍总5M：

```bash
CUDA_VISIBLE_DEVICES=1 python -u -m cifar100_fm.run_study \
  --kind gaussian_dual_pad4_js --data-root ./data \
  --output ./runs/cifar100-dual-pad4-js-scratch-5m --steps 5000000 --eval-every 100000
```

## 5. 保存、评估、汇报

- 每5K保存完整latest并保留previous；每100K保存EMA快照；每25K生成预览。
- 两组每100K正式评估。B另外在开始后+50K评估，分叉协议即总1050K；+100K即总1100K。
- 正式协议：EMA、Heun60、batch64、seed20260912，50K生成（每类500），完整10K干净测试集反向分类。
- FID主指标以原始50K训练图为参考，torch-fidelity Inception2048；同时保存test10K参考FID和IS。
- 无CIFAR-100独立评分分类器，不能编造生成类别准确率；不能把反向准确率或IS当成它。
- 每阶段汇报总step、分叉后step、反向准确率、FID、IS、权重/可视化路径、训练速率和ETA。
  保存confusion matrix、逐类指标、checkpoint SHA256、实际命令、git commit和依赖版本。
- 对比相同总step，不把FM 400K和CE+JS 1100K直接归因于方法。还要报告不同方法的计算时间。

```bash
python -m cifar100_fm.compare --baseline ./runs/cifar100-fm-5m \
  --dual ./runs/cifar100-dual-pad4-js-5m --output ./outputs/cifar100-comparison
```

该工具只对齐两组都已完成且评估协议相同的step，输出CSV/Markdown与逐类并排网格。
没有共同已完成结果时，报告尚不能比较，不制造空白提升数字。

## 6. 时间与边界

历史FM约.225秒/step仅供参考，5M纯更新约13天，另加50次完整评估。CE+JS更慢。
请用本机前1K-5K稳定吞吐和首次正式评估耗时重新估算；共享GPU上的短smoke测试不宜作ETA依据。
两组达到总5M后停止并汇总，不自动增加训练目标或改损失。历史完整指标只到400K，
新CE+JS只做过执行验证，不得写成已取得提升。不得把访问凭据、数据集、权重或缓存提交到Git。
