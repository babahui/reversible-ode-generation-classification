# 原代码审计与设计结论

## 结论

原图像实验不是标准 DDPM 训练，而是线性 rectified flow：随机采样 `t`，用
`x_t=(1-t)x_src+t*x_tgt` 构造输入，回归常速度 `x_tgt-x_src`。这可以放在 stochastic
interpolant 框架下讨论，但不能直接用 DDPM 的前向加噪/逆向去噪解释 checkpoint。

“条件路径交叉导致方向丢失”只对样本 correspondence 有直接含义。MSE 学到的是给定当前
观测的条件平均速度，conditional paths 在同一时刻交叉时会发生平均；但精确的 Lipschitz
ODE 流本身在同一时刻不会相交，并且条件平均场仍可能正确传输边缘分布。因此必须分别评估：

1. 边缘分布是否正确；
2. 样本/语义对应是否保持；
3. 交叉附近的速度方向是否恢复。

最先应加入的是单纯形/源目标条件和合理 coupling。历史状态用于仍然不可辨识、但需要保留
样本路径身份的情况。它把模型扩展成非 Markov 动力系统，不应宣称仍是原论文完全相同的
`g_k(alpha,x)`。

## 已发现的实现问题

- `unconditional-image-translation-5-maginal-Chain-Transport.py` 预先生成各相邻域的完整
  Cartesian product。5 个各约 5000 样本的域会产生约一亿索引，既不必要也没有建立语义
  coupling。
- 多域训练调用无条件 `model(x_mix,t)`，没有源域、目标域或单纯形坐标。同一点同一时间来自
  不同边时，网络只能平均速度。
- `denoising_diffusion_pytorch_path_guide.py` 的兼容 `Unet.forward` 丢弃路径参数；其
  `GaussianDiffusion` 也只传 `x_self_cond`。另一份路径版明确省略了 diffusion/trainer，路径
  条件没有端到端进入训练和采样。
- `ode.path_guide_euler_sampler` 写入 `path[:, i] = z`，但 `z` 是固定输入而不是更新后的 `x`；
  当 `sample_N != train_steps` 时索引语义也不一致。
- toy 的“真实历史训练”用真值直线路径，推理用模型生成历史，存在 exposure bias；两份命名
  不同的 path-memory notebook 核心实现实际相同。方向条件 `c=dt*(z1-z0)` 还近似直接泄露
  监督目标，不能单独作为有效证据。
- `ModelWithEMA.step` 从不递增，所以 `step_ema()` 永远处于 warm-up reset 分支；保存的 step
  也持续为 0。
- 学习率调度器在 step 0 调用，且外层条件与 `StepLR(step_size=10000)` 重复计数，实际衰减
  周期不等于代码表面含义。
- 训练图像、原型与推理图像在 `[0,1]`、`[-1,1]` 和按样本 max 缩放之间不一致。部分采样
  notebook 加载的 prototype 文件名还带尾随空格或与训练文件规则不同。
- 固定类原型把类内连续分布压成点质量，不可由可逆 ODE 无损实现，也不能支持多样生成。
- `slerp` 使用整个 tensor 的单一范数/点积，未 clamp `acos` 输入且平行向量时除零；按通道
  循环也改变了预期的球面几何。
- `DistributionContrastiveLoss` 最终只比较每张图的均值，忽略标准差和空间结构，label 语义与
  注释中的多个版本相互矛盾，并在每次 forward 打印 tensor。
- checkpoint 约 1GB/个但模型、优化器和 EMA 重复保存；旧结果缺少完整 config、数据版本和
  指标。抽查 cat-to-dog 和 CIFAR 分类的最终 checkpoint，二者保存的 `step` 均为 0，与
  EMA 计数缺陷一致；日志中的低训练 MSE不足以比较传输质量。
- 已保存二维图中，conditional paths 在原点附近大量交叉，最终绿色样本塌缩在交叉附近，
  没有覆盖三个目标高斯。这说明该 toy 设置确实失败，但还需要 edge/history/coupling 消融
  才能把失败归因于哪一种不可辨识性。
- 已保存图像中，训练集 cat 输入到 dog/celeba 的输出较连贯，验证集则出现明显源纹理叠加、
  结构破碎。当前证据更像过拟合或 coupling 泛化失败，不能只解释成 ODE 数值误差。

## 新实现的取舍

- 用论文的 marginal fields `g_k` 统一所有边，不为每条边训练独立网络。
- 统一生成/分类默认使用无历史 Markov ODE，以保留可逆映射和 Bayes 解释；路径消融可用
  `previous + recent_velocity`，避免完整帧序列的 `O(T*C*H*W)` 条件内存。
- 默认 Heun 减少 Euler 离散误差，同时保留 Euler 做消融。
- CIFAR 采用正交高斯混合潜空间，使生成与贝叶斯分类共享一个连续、可逆的双向模型。
- folder 数据默认独立 coupling，仅把它定位为边缘分布基线；语义 correspondence 需要真实
  tuple 或 data-dependent/OT coupling。
