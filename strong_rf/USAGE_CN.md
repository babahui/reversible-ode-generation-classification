# 更换 Backbone 的五组对照实验

本分支 `backbone/torchcfm-unet-cifar10` 使用约 3575 万参数的 TorchCFM U-Net。
它与仓库原先的 U-Net、旧的 1M 结果不是同一套实验。完整命令见
[英文复现指南](README.md)，真实指标及同一步数网格见
[结果报告](../results/strong-rf/RESULTS.md)。

## 方法区别

| 方法 | 初始分布 | U-Net 输入 | 额外训练处理 |
|---|---|---|---|
| Label-conditioned | 标准高斯 | 时间、状态、真实类别 | 无 |
| Low-frequency Gaussian-only | 类别高斯 | 时间、状态 | 无 |
| Gaussian + OT | 类别高斯 | 时间、状态 | 按类别 Sinkhorn 后一对一配对 |
| Gaussian + reverse CE | 类别高斯 | 时间、状态 | 可微反向端点分类损失 |
| Gaussian + OT + reverse CE | 类别高斯 | 时间、状态 | 上述两者 |

类别高斯为 `N(mu_y, 0.25 I)`，其中 `mu_y` 是 `[3,32,32]` 的低频 DCT
中心张量，L2 范数为 12；随机部分仍是全维独立高斯，标准差为 0.5。

训练取 `x_t=(1-t)*noise+t*image`，网络预测速度，目标为 `image-noise`。
生成从 `t=0` 积分至 `t=1`；Gaussian 各组只通过起始噪声选择类别，网络不输入 label。
分类从真实图片 `t=1` 反向积分至 `t=0`，按终点距离哪个 Gaussian 中心最近分类。

CE 正式实验必须显式设置 `--ce-logits center`：每 4 次更新，对 16 张训练图片
进行 4 步可微 Euler 反向积分，再将终点投影到单位中心方向做 CE，权重 0.001。
这与默认 posterior logits 的 argmax 一致，但 CE 温度不同，不能混用。
提供的训练脚本已设置正确选项。

## 运行入口

```bash
# 环境及数据准备见 README.md；以下从仓库根目录运行。
CUDA_VISIBLE_DEVICES=0 KIND=gaussian DATA_ROOT=./data STEPS=500000 \
  bash scripts/train_strong_rf.sh
CUDA_VISIBLE_DEVICES=1 KIND=gaussian_ot_ce DATA_ROOT=./data STEPS=500000 \
  bash scripts/train_strong_rf.sh
```

每个方法使用独立输出目录。续训添加 `RESUME=1`，`STEPS` 表示累计步数。
训练每 25K 保存 EMA 权重和预览；完整指标使用 README 的独立评估命令。
`strong_rf.sample` 可以直接生成网格，或只进行反向分类，不需要额外分类器。

## 结果边界

- 五组均没有 posterior head、没有 history adapter；刚讨论的 history 对照尚未实现。
- 外部小型 ResNet 只给生成图片评分，不参与训练速度场、采样或反向分类；其真实测试准确率为 92.36%。
- 正式评估为 EMA、Heun60、生成 50K 张、原始训练集 50K 作为 FID 参考；反向分类测试完整 10K 张。
- 历史 KID 实际是 2560 张生成图片对 5000 张测试参考，不是 KID50K，报告中单独标注。
- 一组训练 seed 尚不足以确定小幅提升；Label 与 Gaussian 的噪声方差不同，也不能只归因为条件方式。
- 包含精确的评分分类器权重；大型 flow 权重和优化器状态没有上传。可以从头训练，或向实验作者索取与报告 SHA256 匹配的权重。
- 没有声称已达到 95% 反向分类或 SOTA。旧状态文件也不能代表当前训练仍在运行。
