# 类别噪声分布统一生成与分类：可行性

## 1. 对标准 diffusion 不可行

标准 DDPM 的前向过程设计为

```text
q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1-alpha_bar_t) I)
```

当 `t -> T` 时，`alpha_bar_T -> 0`，所有类别都趋近同一个 `N(0,I)`。因此类别 `Y` 与终点
噪声 `X_T` 的互信息趋近 0；从最终噪声判断 MNIST 类别在设计上不可行。随机 SDE 的一次正向
轨迹也不是生成轨迹的逐样本逆映射。

所以本项目中的可行模型应称为 class-structured stochastic interpolant / probability-flow
ODE，而不是“保持标准终点先验不变的 DDPM”。

## 2. 对一个共享的可逆 ODE 可行

令类别结构化基础分布为

```text
p_Z(z) = sum_y pi_y p_y(z)
p_y(z) = N(mu_y, sigma^2 I)
```

其中 `mu_y` 是正交类中心。训练一个不输入标签的共享 ODE 映射 `F: z -> x`。生成类别 `y`
时，采样 `z ~ p_y` 并积分 `z -> x`。分类时，用同一 ODE 反向得到 `z=F^{-1}(x)`。

由变量替换公式：

```text
p(x | y) = p_y(F^{-1}(x)) * |det dF^{-1}(x)/dx|
```

同一个 `F` 用于全部类别，因此 Jacobian 项与类别无关，在 Bayes 后验中抵消：

```text
p(y | x)
  = pi_y p_y(z) / sum_j pi_j p_j(z),  z=F^{-1}(x)
```

这正是 `OrthogonalGaussianLatent.log_posterior` 的计算。分类不需要估计 ODE divergence，
也不需要第二个 classifier。训练会从筛选后的训练集统计 `pi_y`，并随 checkpoint 保存。

## 3. 训练目标

训练样本先按标签耦合：

```text
y ~ p_data(y)
x ~ p_data(x | y)
z ~ N(mu_y, sigma^2 I)
```

在线性插值上采样

```text
x_t = (1-t) x + t z
```

网络预测两个 conditional mean：

```text
g_image(t,x_t) = E[x | x_t]
g_noise(t,x_t) = E[z | x_t]
v(t,x_t) = g_noise - g_image
```

沿 `image -> noise` 积分用于分类；改变路径方向后速度符号自动反转，沿 `noise -> image`
积分用于生成。这是同一个网络、同一组参数和同一个概率流。

## 4. 成立条件与限制

1. 必须是一个共享、Markov 的确定性 ODE。启用 `--history` 后，动力学依赖额外历史状态，
   简单反向积分不再保证得到 `F^{-1}`，上面的 Jacobian 抵消论证也不再严格成立。因此统一
   生成分类默认关闭历史。
2. 训练 coupling 必须保持标签对应。共享 flow 一定能传输总混合分布，但有限模型和有限数据
   下不保证每个噪声分量都自动对应指定图像类；分类准确率和每类生成准确率必须实际验证。
3. 高斯分量有重叠，因此理论 Bayes error 不为零。减小 `sigma` 或增大中心间距可降低重叠，
   但过大的尺度差会增加 ODE 运输代价和数值难度。
   默认 `center_scale=4.0, sigma=0.5` 时任意两中心距离约为 `5.66`，约为一维判别标准差的
   8 倍，分量重叠已经很低，不需要使用会导致后验数值过度饱和的极端尺度。
4. MNIST/CIFAR 像素是离散数据。训练默认用 `--data-noise 0.01` 做轻微 dequantization，使连续
   密度解释合理。
5. conditional interpolation paths 可以交叉，回归场会取条件平均；但一个满足唯一性条件的
   Markov ODE 解在同一时刻不会真正交叉。首先应使用分离良好的噪声分量、标签保持 coupling、
   Heun 求解和 reflow/OT pairing。路径历史更适合作为“保持特定样本 correspondence”的消融，
   不是可逆分类主模型的默认解法。
6. `image -> noise -> image` 的 cycle MSE 是必要的数值诊断，但 cycle 小不代表生成分布正确；
   还需报告分类准确率、每类生成样本的外部分类准确率和 FID/KID。

## 5. 结论

如果“diffusion”严格指所有类别最终都变成同一个标准高斯的 DDPM，该方案不可行。如果允许
把终点改为类别结构化高斯混合，并使用共享 stochastic-interpolant probability-flow ODE，
则方案理论上成立：噪声分量控制生成类别，逆向潜变量通过 Bayes posterior 完成分类。
