# 双 pad4 + reverse CE + JS: CIFAR-100 迁移说明

## 与 CIFAR-10 的对齐

来源是 `cifar10-dual-pad4-ce-js-500k-20260921` 的隔离训练代码及
[实际配置归档](results/cifar10-dual-pad4-source-config.json)，不是凭名称重新设计增强。

| 设置 | 原 CIFAR-10 | 本次 CIFAR-100 |
|---|---|---|
| 网络 | scalar-t TorchCFM U-Net base128 | 同一backbone、参数量35,746,307 |
| 初始化 | 本数据集纯FM 1M完整model/EMA/Adam/RNG | 推荐同样从本数据集纯FM 1M完整状态分叉 |
| FM | independent coupling，batch128，水平翻转 | 相同 |
| Prior | 10类DCT Gaussian，norm12，sigma0.5 | 原CIFAR-100的100类DCT codebook，norm12，sigma0.5 |
| CE来源 | 当前FM batch前32张，各2视角 | 相同，不是64张原图各2视角 |
| 裁剪 | normalized image反射pad4 -> uniform 32x32 crop | 相同；两个视角都裁剪，先后独立抽偏移 |
| 独立增强随机流 | seed20260918，保存/恢复状态 | 相同，不改变FM的噪声/时间/翻转随机流 |
| CE频率与ODE | 每4次optimizer update，Euler4 | 相同 |
| 网络在CE ODE中 | eval/dropout关闭，仍可微，y=None | 相同 |
| CE scores | 对单位Gaussian中心的投影 | 同一定义，输出100类固定几何分数 |
| CE权重 | .001，开始后2K线性warmup | 相同，按分叉后step计算 |
| JS | .001，5K warmup；T=2，外乘T²；两侧有梯度 | 相同 |
| 优化器 | Adam LR2e-4不下降，EMA.9999，clip1，FP32 | 相同；分叉时保留Adam，不重启LR warmup |
| 梯度诊断 | 每500步 | 相同；不影响目标函数 |
| 正式评估 | EMA Heun60，50K生成，完整10K分类 | 相同数量/solver，但必须用CIFAR-100真实参考数据 |

**改变的因素**：数据集、类别数与对应100类先验；以及用户要求的5M目标和长实验存储频率。
这不构成“只改变类别数”的完全单因素数据集实验。原CIFAR-10的指标不可当作CIFAR-100结果。

## 训练如何实现

FM路径仍为 `x_t=(1-t)z+tx`，目标速度 `x-z`，使用128张水平翻转后的图片。
另外取其中32张，用独立随机流从每张图得到 `a,b`，按 `[a_1,...,a_32,b_1,...,b_32]` 拼接。
反射pad4不是补黑边；归一化前后反射操作等价。两个裁剪可以碰巧相同，不强制排斥相同偏移。

```python
z_a, z_b = reverse_Euler4(model, concatenate(a, b)).chunk(2)
c_k = mu_k / norm(mu_k)
s_a = flatten(z_a) @ c.T
s_b = flatten(z_b) @ c.T
q_a = softmax(s_a / 2)
q_b = softmax(s_b / 2)
```

令 `m=(q_a+q_b)/2`：

```text
CE = [CE(s_a,y) + CE(s_b,y)] / 2
JS = [KL(q_a || m) + KL(q_b || m)] / 2
L  = L_FM + lambda_CE * CE + lambda_JS * 2^2 * JS
```

两视角都保留梯度，未 detach，也没有新增可学习分类头。JS单独只要求两者一致，
一致但错误也能低JS，所以必须同时保留有监督CE，监控完整测试集准确率。
CE并未每步乘4补偿：每4步启用一次，保持原CIFAR-10的有效权重。

这里的center scores和推理Gaussian posterior logits的argmax相同，但温度不同。
norm12、sigma.5时，posterior logits在类别无关项外等于center scores的48倍。
**不能用 `prior.log_posterior()` 直接替换训练center logits，或省略JS外部T²。**

实现入口：[dual_view.py](dual_view.py)、[train.py](train.py)。恢复时同时恢复model、
EMA、Adam、CPU/CUDA/NumPy RNG及独立crop RNG；EMA-only文件被拒绝用于训练恢复。

## 推理没有新增条件

生成仍从请求类别的Gaussian采样，ODE中`y=None`，t=0到1。
分类仍输入原始未裁剪图片，t=1到0，最后固定Gaussian后验argmax。
正式主指标用Heun60；Euler4分类可另外检查训练/推理solver差异，但不能混入主表。
**没有测试时多视角投票、classifier guidance、posterior head或真实标签泄漏。**

## 公平比较和结果边界

推荐对齐总步数1.1M、1.2M、...、5M；B的额外+50K只作早期趋势检查，A没有同step时不强行配对。
相同总步数不代表相同计算量：B的反向ODE反传更昂贵，必须同时记录wall-clock和GPU型号。
关注反向准确率、FID和IS，不预设CIFAR-100能达到CIFAR-10的95%目标。
原FM 400K：反向47.22%、FID11.14、IS9.61，是历史参考，不是新5M的对照结果。
新CE+JS/新5M结果目前待训练；本地短时测试仅证明可执行，不证明性能收益。
