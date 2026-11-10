---
intention: 定义 qmodel 算子级 FLOPs 估算接口及统一统计口径, 为后续 MFU 汇总提供基础.
---

# 算子级 FLOPs 估算

## 如何估算一个现有算子

`qmodel.flops.estimate_flops(module, input_shape, gradient_request)` 只读取模块配置、参数的 `requires_grad` 和输入 shape. 不执行 tensor 运算, 不替换算子, 不安装 hook. 实现在 `qmodel/flops/__init__.py`.

```python
from torch import nn
from qmodel.flops import GradientRequest, estimate_flops

module = nn.GRU(4, 5)
request = GradientRequest(backward=True, input_requires_grad=True)
result = estimate_flops(module, (3, 2, 4), request)
# FlopsEstimate(1620, 1320, 1620), total_flops = 4560.
```

`GradientRequest` 显式声明 backward、输入梯度和 h0 梯度需求. `h0_requires_grad` 默认为 False, 仅 GRU 使用; 为 True 表示调用者提供可求导 h0. 参数是否训练直接读取各参数的 `requires_grad`. `.eval()` 和全局 grad mode 不改变估算请求. 禁用 backward 时, 两个 backward 分量均为零.

结果包含 `forward_flops`, `backward_input_flops`, `backward_weight_flops` 和只读 `total_flops`. Backward Input 包含内部 activation 梯度传播; Backward Weight 只包含矩阵权重梯度.

## 统计哪些计算

乘加计 2 FLOPs. 只计 dense 矩阵乘和卷积, 不扣除输入零值、padding 零值或默认零 h0. 不计 bias 加法及梯度归约、activation、门控逐元素运算、dropout、padding 搬运、optimizer 或 checkpoint 重算. 参数 dtype 不改变理论 FLOPs.

估算假定所有返回值收到完整上游梯度. GRU 包含 `output` 和所有层的 `h_n`; 不针对下游切片或零梯度裁剪. 这是统一的理论矩阵计算口径, 不是 kernel 实际执行指令数, 也不是完整逐元素 FLOPs.

## Linear 和 Conv1d 的公式

Linear 输入最后一维为 I, 前导维度乘积为 M, 输出维度为 O. Forward 为 `2MIO`. 输入需要梯度时 Backward Input 为同一数值, weight 需要梯度时 Backward Weight 为同一数值. 支持 `(I,)` 和任意合法前导维度.

Conv1d 的 batch 为 B, 输出长度为 Lout, 通道为 Cin/Cout, groups 为 G, kernel size 为 K. Forward 为 `2 B Lout Cout (Cin/G) K`. 两类 backward 分别按同一 dense 连接数量计数, 按梯度需求启用. padding 边缘仍计完整 kernel, 因此不是去除边缘零项后的精确算术最小值.

数字 padding P 对应 `Lout = floor((Lin + 2P - dilation*(K-1) - 1)/stride) + 1`. `valid` 对应 P=0, `same` 对应 Lout=Lin. 支持合法的 stride、dilation、groups、padding mode, 以及 `(C,L)` 和 `(B,C,L)` 输入.

## GRU 的公式与梯度路径

令 batch 为 B, 序列长度为 T, 方向数为 D, hidden size 为 H. 第 l 层输入宽度 Il 在首层为 input_size, 后续层为 DH. Forward 为 `Σl 6BTDH(Il+H)`.

每层、每方向、每时间步的三个 gate 合并计算:

| 矩阵运算 | FLOPs |
| --- | --- |
| input-hidden | 6B H Il |
| hidden-hidden | 6B H² |

相应输入梯度和权重梯度分别采用同一矩阵计算量. Backward Weight 按各层、各方向的 `weight_ih` 和 `weight_hh` 独立计数.

Backward Input 根据展开计算的依赖关系计算. 先沿时间和层记录 activation 是否可求导, 再从 output 和每层最终状态反向传播需求. 输入不需要梯度只省去对外输入梯度, 不省去训练参数所需的层间或时间步间传播. h0 不需要梯度时省去到 h0 的矩阵乘. bias 本身不计 FLOPs, 但可训练 bias 仍使其后续状态需要梯度传播.

依赖分析只操作 Python 布尔值, 时间和空间复杂度为 O(layers × directions × T), 与 B 和 H 无关. 支持多层、双向、batch_first、unbatched 输入和显式 h0. PackedSequence 不支持并明确报错.

## 注册与避免重复计数

全局 `registry` 预注册三个精确类型. `FlopsRegistry()` 可创建独立 registry, 用 `register(module_type, estimator)` 注册函数. 函数接收 module、shape tuple 和 GradientRequest, 返回 FlopsEstimate. 重复注册报错, 不隐式覆盖; 子类必须显式注册.

单次估算从不递归访问 children. GRU 整体包含其所有层和方向. Sequential 等容器不受支持, 不会产生隐式汇总. 后续模型级汇总必须按执行边界归属计算量, 本接口不承担遍历和去重.

测试位于 `tests/unittest/test_flops.py`. 除固定公式外, 独立展开 GRU 使用 autograd 实际访问的矩阵乘计数, 并与 PyTorch 的 output、h_n 和参数/输入梯度比较. 本阶段不实现模型级汇总、实际耗时或 MFU.
