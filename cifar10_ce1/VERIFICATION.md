# CE1发布验证

2026-10-05，在原实验环境Python 3.8.10 / PyTorch 2.0.0+cu118 / torchvision 0.15.1+cu118验证。

- 发布入口`evaluate_reverse_test`重新评估+50K EMA的完整test10K：9272/10000，92.72%，与原实验一致。结果在`results/release-euler1-test10k.json`。
- 发布入口`infer sample`在CPU加载原EMA，完成10类各1张、Euler1短步数接口检查。此短步数样例不用于声称生成质量；正式生成使用Heun60。
- CPU复核前4张真实测试图：预测[3,8,8,0]与原CUDA预测一致，中心logits最大绝对误差2.77e-5。
- 两项行为测试通过：一步反向积分等于x-v(1,x)且梯度有限非零；使用小型速度网络和合成数据执行真实训练循环，迁移数据/源/输出路径后恢复，与连续训练的model、EMA、Adam及CPU/CUDA/裁剪RNG逐位相同。
- 四个发布权重均保留原文件字节，SHA256及大小已核对，见weights.json；完整CE1状态的Adam计数为1,050,000。
- U-Net、prior构造和训练计算来自已完成实验的隔离代码快照。新增包内导入、路径迁移和CLI；迁移源路径仍需通过原source SHA256检查。
- 所有Python文件通过编译检查，git diff --check通过。

复测命令：

```bash
python -m unittest discover -s tests -p test_cifar10_ce1_release.py -v
python -m cifar10_ce1.evaluate_reverse_test \
  --checkpoint ./weights/cifar10-ce1-1050000-ema.pt \
  --data-root ./data --output ./outputs/release-euler1-test10k --steps 1
```

未重新执行完整1M/50K训练或50K生成：这些历史结果来自原始实验，详见results。上述迁移测试使用小型网络验证状态恢复逻辑，不替代完整网络长训复现。当前500K续训继续使用原隔离快照，不受发布整理影响。
