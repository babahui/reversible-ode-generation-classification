# RAE · ImageNet-1k 条件 Flow Matching

`RAE` 分支把已验证的 Imagenette 条件生成流程扩展到 **完整 ImageNet-1k** 和原版尺寸 DDT：

**标准高斯噪声 + 类别标签 → DDT 速度场 → RAE latent → 冻结解码器 → 256×256图片。**

使用1000类条件、纯FM、10%条件丢弃、CFG采样。**本分支不使用正交类别中心、反向CE或JS。** 不需要下载官方大DDT权重；DDT从随机初始化训练，RAE使用配套的官方预训练权重。

## 配置

默认配置：[configs/rae/imagenet256_large.json](configs/rae/imagenet256_large.json)。

| 项目 | 默认值 |
| --- | --- |
| 数据 | ImageNet-1k，train 1,281,167张 / val 50,000张，1000类 |
| 图像 | RGB、短边缩放256、中心裁剪256；训练随机水平翻转 |
| 冻结RAE | DINOv3-L K7，多层聚合11/13/15/17/19/21/23；配套ImageNet ViT-XL decoder和统计 |
| latent | 1024×16×16，官方mean/std归一化 |
| DDT主干 | 28层、1440宽、20头 |
| DDT预测头 | 2层、2048宽、16头；无IG辅助头 |
| tokens | 256个空间token + 4个时间token + 8个类别token |
| 参数 | 总869,589,396；可训练869,589,140，约8.70亿 |
| 条件 | 1000个真实类别 + 独立null索引1000；10%条件丢弃 |
| 先验 | 所有类别共享N(0,I)，latent形状；不加正交中心 |
| FM | 直接速度预测，uniform t，t=0噪声、t=1数据 |
| 优化器 | AdamW，betas=(.9,.95)，weight decay=.01，clip norm1 |
| 学习率 | warmup10000步至2e-4；cosine至800000步时2e-5，随后保持 |
| batch | 全局128；每卡microbatch2；根据卡数自动计算梯度累积 |
| 精度/显存 | BF16前向，FP32训练状态；激活重计算；默认CPU EMA=.999 |
| 训练期限 | 默认持续训练；可通过`--max-steps`指定绝对停止步数 |
| 保存/评估 | 每1000步保存完整状态；每10000步保存EMA并评估 |
| 周期评估 | Euler100，CFG=1/2各5000张，FID/KID、ResNet50生成类别一致性 |

8.68亿是去掉类别输入的版本；加入1000类embedding后实际约8.70亿。这里沿用原版网络尺寸，FM/优化器/采样配方是本实验实现，**不等同于官方RAEv2的x-prediction、logit-normal、IG、GMuon配置**。

全量ImageNet不再默认预缓存latent：一张FP16 latent约0.5MiB，训练原图+翻转缓存约1.22TiB。因此默认每个microbatch在线调用冻结编码器；训练不加载解码器，不创建全量latent缓存。均匀有放回采样所有训练图像，翻转由可恢复的样本流决定。800000步、全局batch128约等于80次数据集大小的图片呈现，不是精确逐轮遍历。

## 安装

在仓库根目录执行，推荐独立Python3.11环境：

```bash
python3.11 -m venv .venv-rae
source .venv-rae/bin/activate
python -m pip install -r requirements-rae.txt
python -m rae_imagenet.setup_vendor
```

安装与驱动匹配的PyTorch CUDA构建。上游源码固定为：

- RAEv2：`8a0d238f8dc3b261aba98b217f6c79c0182e8e94`
- DINOv3：`94a96ac83c2446f15f9bdcfae23cad3c6a9d4988`

`setup_vendor`只获取源码，不下载ImageNet或模型权重。已有源码可通过`--rae-repo`和`--dino-repo`指定。第三方来源、许可和改动说明见[THIRD_PARTY.md](rae_imagenet/THIRD_PARTY.md)。

## 数据与权重

从已有合法获取的ImageNet副本准备ImageFolder目录：

```text
/path/to/imagenet/
  train/
    n01440764/*.JPEG
    ...（1000个WNID目录）
  val/
    n01440764/*.JPEG
    ...（相同1000个WNID目录）
```

验证集必须按官方标签分到WNID目录；不支持未分类的扁平`val/*.JPEG`。标签按WNID字典序生成，train/val映射必须一致。默认检查1000类和官方图片总数；`--allow-subset`只用于少量数据的管线检查，不作为全量结果。

从[RAEv2官方权重库](https://huggingface.co/nyu-visionx/RAEv2-models)获取配套文件，例如：

```bash
hf download nyu-visionx/RAEv2-models \
  --include 'encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth' \
            'stage1/imagenet/dinov3l-k7/decoder.pt' \
            'stage1/imagenet/dinov3l-k7/stats.pt' \
  --local-dir pretrained_models
mkdir -p weights
ln -s ../pretrained_models/encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth weights/
ln -s ../pretrained_models/stage1/imagenet/dinov3l-k7/decoder.pt weights/
ln -s ../pretrained_models/stage1/imagenet/dinov3l-k7/stats.pt weights/
```

最终`--weights`目录下应有以上三个文件。训练和重建预检校验固定SHA256，避免编码器、解码器或统计不匹配。评估的Inception和ResNet50首次使用时自动获取公开评估权重；离线时用`--inception-weights`和`--classifier-weights`指定本地文件，后者为ResNet50 IMAGENET1K_V2。

## 先检查

参数统计使用meta设备，不分配大模型GPU权重：

```bash
python -m rae_imagenet.preflight \
  --data-root /path/to/imagenet --weights weights

python -m rae_imagenet.reconstruct \
  --data-root /path/to/imagenet --weights weights \
  --output outputs/rae-reconstruction --count 16

python -m unittest discover -s tests/rae -v
```

`reconstruct`检查真实图像编码/解码，输出PSNR和原图/重建图。测试覆盖图像配对、类别映射、null类别、CFG、时间方向、采样batch不变性、DDP数据分片、断点数据位置、完整优化器/EMA/RNG恢复、激活重计算的前向/梯度以及大模型参数量。模型测试需要上游源码，可设置`RAEV2_REPO=/path/to/RAEv2`。

## 训练与续训

单卡：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 \
python -m rae_imagenet.train \
  --data-root /path/to/imagenet --weights weights \
  --output outputs/imagenet-rae-large
```

8卡DDP，**全局batch仍为128**，每卡microbatch2、累积8次：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 \
torchrun --standalone --nproc_per_node=8 -m rae_imagenet.train \
  --data-root /path/to/imagenet --weights weights \
  --output outputs/imagenet-rae-large-ddp
```

训练默认连续进行，每10000次参数更新暂停训练、在rank0完成评估，然后继续。评估期间其余rank等待，不同时执行训练前向。用`--max-steps 800000`可显式设置绝对训练目标；默认JSON为null，无自动终止点。

SIGTERM/SIGINT请求在参数更新边界保存。再次运行同一命令自动加载`checkpoints/latest.pt`；也可指定`--resume /path/to/latest.pt`。恢复模型、EMA、AdamW、每rank的CPU/CUDA/NumPy/Python RNG和全局数据位置。仅EMA快照不能续训。

每个参数更新的数据批次由步数确定，DataLoader预取不会导致续训跳过图片；生成噪声按样本编号固定，改变采样batch不会改变噪声。严格续训要求相同world size及模型、优化器和数据清单。允许调整microbatch/worker/输出频率等运行参数，但改变microbatch会改变随机数分组和浮点累积，不能宣称逐位重现。

- `status.json`：训练、评估或暂停状态。
- `train.jsonl`：loss、lr、速度、峰值显存。
- `dataset.json`：类映射及路径/标签清单摘要；不校验全部图像内容哈希。
- `checkpoints/latest.pt`：完整训练状态，原子替换。
- `checkpoints/ema-XXXXXXXXX.pt`：周期EMA快照。
- `evaluations/step-XXXXXXXXX/`：每轮指标、样图和评估协议。

## 独立生成与评估

```bash
python -m rae_imagenet.sample \
  --checkpoint outputs/imagenet-rae-large/checkpoints/ema-000010000.pt \
  --weights weights --labels 0,207,483,491,497,566,569,571,574,701 \
  --cfg 2 --steps 100 --output outputs/imagenet-samples

python -m rae_imagenet.evaluate \
  --checkpoint outputs/imagenet-rae-large/checkpoints/ema-000010000.pt \
  --data-root /path/to/imagenet --weights weights \
  --samples 50000 --batch-size 2 --output outputs/imagenet-eval50k
```

标签编号以checkpoint内的WNID顺序为准。5000张周期评估每类5张；50000张评估每类50张。真实参考使用完整ImageNet验证集50000张。FID/KID口径与此前Imagenette3925张参考不同，不能直接混比数值。

CFG采用`v_null + w*(v_label-v_null)`；w=1为普通类别条件速度，w=2额外执行空条件前向。Euler100的NFE分别为100/200。模型输出速度，时间0为标准高斯、1为图像latent。生成类别一致性为外部分类器评价，不是反向分类。

评估按CFG子任务保存结果，可在中断后继续；如果已有同一步结果，修改采样协议时应选择新的评估输出目录，避免混合指标。

## 资源与验证范围

约8.70亿参数的FP32模型+梯度+AdamW+GPU EMA静态状态约16.20GiB，不含激活、编码器、BF16权重缓存和工作区。默认CPU EMA和激活重计算降低GPU占用，但每步CPU EMA更新存在传输开销。24GB卡需要microbatch短测；可用`--microbatch 1`保持全局batch128，不保证本机已验证大模型GPU峰值。更大显存/多卡适合正式全量训练。

一个完整checkpoint约12.96GiB，原子保存需要额外临时空间；每个EMA快照约3.24GiB。至少为最新文件及其临时副本预留约27GiB，周期快照、数据集、权重和评估缓存另计。代码在写入前检查空间，不自动删除其他实验文件。

本次发布通过11项测试（含两进程CPU/Gloo续训及评估中断恢复），并完成大模型meta参数统计、配套RAE在2张真实图像上的CPU编码—解码。验证记录见[VALIDATION.json](rae_imagenet/VALIDATION.json)。**未执行完整ImageNet大模型GPU训练或多卡吞吐测试，不提供已训练的ImageNet大模型权重或FID声明。** 已有本地Imagenette实验不由此分支自动启动或停止。

旧CIFAR代码仍保留，原说明见[README_CIFAR.md](README_CIFAR.md)。
