---
intention: 说明 qmodel 的职责、安装条件和接入契约, 帮助研究项目复用训练与推理执行能力.
---

# qmodel

qmodel 是面向 PyTorch/CUDA 的训练与推理框架, 提供 batch 数据加载、异步 CPU/GPU 执行、DDP、AMP、CUDA Graph、checkpoint 和 profiling. 研究项目提供 Dataset、模型、loss 和实验配置.

## 1. 什么时候使用

需要复用 GPU 执行循环、分布式采样或训练恢复时使用 qmodel. 它不负责市场数据采集、股票池、feature/label 定义、normalization、实验调度或策略回测.

| 需求 | 接入方式 |
| --- | --- |
| 按框架约定训练、保存并评价 checkpoint | `Trainer`, `Evaluator`, `QConfig` |
| 项目已有训练编排或输出格式 | `AsyncTrainer` / `AsyncInference` 的子类 |
| 对照同步与异步执行 | `SyncTrainer` / `SyncInference` 的子类 |
| 仅复用数据加载或执行组件 | 显式导入 sampler、loader、scheduler 等 |

顶层入口见 [qmodel/__init__.py](qmodel/__init__.py). 底层执行类见 [eval_trainer.py](qmodel/core/eval_trainer.py), 需要通过子类实现 `_pre_step` 和 `_post_step` hooks, 不能作为完整训练脚本直接使用. `Sync` 表示同步执行, 不表示 CPU backend.

与其他公共包的分工: `qdata` 读取市场数据, `qdataset` 提供因子存储和窗口访问, 消费者将它们适配为 qmodel 的 Dataset. `qartifact` 处理研究产物发布与完整性, `qeval` 提供统计计算. qmodel 不要求安装这些包; 当前顶层 `Evaluator` 的 global IC 仍由 NumPy 计算.

## 2. 如何安装

从仓库根目录构建. 当前 C++ sampler 扩展使用 pybind11、C++11 和 OpenMP, 构建环境需要支持 `-fopenmp` 的编译器. wheel 与 Python ABI 和平台有关, 不能视为纯 Python 通用 wheel.

```bash
python -m pip install build
python -m build --wheel
```

消费者安装 `dist/` 中与目标环境匹配的 wheel. 开发环境可执行 `python -m pip install -e .`. 构建配置见 [pyproject.toml](pyproject.toml) 和 [setup.py](setup.py).

当前环境契约有两处需要由消费者显式处理:

- metadata 声明 Python >=3.9, 但源码使用 `typing.TypeAlias` 和 `X | Y` 等 Python 3.10 接口/语法, 使用 Python >=3.10.
- metadata 仅声明 Torch 和 NumPy. 顶层 import 还涉及 `pandas`, `polars`, `tqdm`, `tensorboard`, `line-profiler`, `typing-extensions`; Feather 输出需要 `pyarrow`. 使用 `TmpFeather` 时额外需要 `fasteners`. 这些依赖应纳入消费者环境, 仅安装 wheel 的声明依赖不代表运行环境完整.

训练和推理执行路径需要 CUDA. Torch/CUDA/NCCL 组合应以目标项目已验证的环境为准, metadata 的版本下限不等于所有组合都已验证. 使用前核对实际安装版本和 import 来源, 避免实验目录中的旧副本遮蔽安装包:

```bash
python -c 'import qmodel; from importlib.metadata import version; print(version("qmodel")); print(qmodel.__file__)'
```

## 3. 如何接入训练

最小接入需要三个项目对象:

1. Dataset factory: 接收 `(group, dtype)`, 返回实现 batch 读取的数据集.
2. Model class: 接收一个 `ModelConfig` 对象, 返回 `nn.Module`; `forward(data)` 返回与 target 兼容的预测 tensor.
3. 完整配置: 继承或填充 `QConfig`, 显式提供模型、数据、优化器、执行和输出设置.

模型工厂实际调用 `model_class(config.model)`, 见 [models/__init__.py](qmodel/models/__init__.py). `QConfig` 的许多字段为 `field(init=False)` 且没有默认值, 因此 `Trainer(QConfig())` 不是可运行示例. 配置定义见 [config.py](qmodel/config.py).

配置至少需要覆盖以下内容:

- 数据与模型: `dataset_class`, `model_class`, `model`, `seed`, `window_size`, `ret_col_name`.
- 优化与精度: `criterion`, `optimizer_class`, `learning_rate`, `train_dtype`, `eval_dtype`, `amp_dtype`, `use_amp`, `use_lr_sched`, `lr_scheduler`.
- 执行: `batch_size`, `num_workers`, `num_iters`, `train_use_cuda_graph`, `num_normal_steps`, `distributed`.
- 输出与恢复: `root_dir`, `tensorboard_dir`, `load_from_iter`, `save_every`, `log_every`, `mean_loss_length`.
- 评价与 profiling: `eval_during`, `eval_every`, `eval_during_num_iters`, 完整的 `evaluator` 和 `profiler` 配置.

已有完整项目配置 `conf` 后, 入口如下. 两种操作分别调用, 不是无需数据和配置即可独立运行的示例:

```python
from qmodel import Evaluator, Trainer

# Train with the project-owned configuration.
Trainer(conf).train()

# Evaluate existing checkpoints in a separate invocation.
Evaluator(conf, group="test").evaluate()
```

可参考 [test_distributed_runtime.py](tests/unittest/test_distributed_runtime.py) 中的 `RuntimeDataset`, `RuntimeModel` 和 `RuntimeConfig` 理解完整契约. 这些对象属于测试, 消费者应实现自己的适配. `tests/train_config_*.py` 和 `tests/CQ/` 是依赖特定数据环境的历史实验示例.

## 4. Dataset 如何返回 batch

协议见 [dataset.py](qmodel/data/dataset.py), 实际加载行为见 [dataloader.py](qmodel/data/dataloader.py).

`dataset_class(group, dtype)` 的训练 group 为 `"train"`, 顶层评价 group 为 `"test"` 或 `"predict"`. Dataset 的 `__len__` 表示样本数, `__getitems__(indices)` 必须按传入顺序返回已组 batch 的 `(data, target, other_meta)`, 而不是单样本列表. 默认 collate 原样返回这个对象, 不再次 stack.

- `data` 是模型输入 tensor, 第 0 维为 batch.
- `target` 是 loss 或评价所需 tensor, 顺序与输入一致.
- `other_meta` 携带样本身份, 供项目 hook 或顶层评价使用.

标准 loader 启用 `pin_memory=True`. 自定义 iterable 接入异步 runtime 时, 输入 data 必须是 pinned CPU tensor, 训练 target 也必须 pinned; runtime 随后执行 non-blocking H2D. 不应在 Dataset 中擅自改变 sampler 已确定的样本选择或顺序.

## 5. 如何启用 DDP 和 CUDA Graph

设置 `config.distributed.enabled=True`, 再通过 `torchrun` 启动项目训练入口. runtime 从 `RANK`, `WORLD_SIZE`, `LOCAL_RANK` 读取布局, 每个 rank 使用对应 CUDA device. 实现见 [distributed.py](qmodel/components/distributed.py).

`batch_size` 和 `evaluator.eval_batch_size` 都是 global batch size, 必须能被 world size 整除. 每个 rank 获取 global batch 的连续一段; 训练丢弃最后不足的 global batch, 评价保留尾 batch. 采样实现见 [sampler.py](qmodel/data/sampler.py). 顶层 Trainer 使用 DDP 同步梯度, rank 0 写 checkpoint 和训练日志.

顶层 `train_use_cuda_graph=True` 要求 `use_lr_sched="custom"` 且 `use_amp` 为 `"custom"` 或 `"none"`. Optimizer 和模型运算也必须支持 capture, 例如 Adam 需要相应的 `capturable` 设置. Graph 训练使用稳定的输入 shape 和控制流; `num_normal_steps` 控制 eager warmup, 当前异步 runtime 要求至少为 12. 关闭顶层 graph 开关后仍使用 CUDA 异步执行.

底层 runtime 没有与顶层相同的 graph 布尔开关, 直接接入时需检查 `num_normal_steps` 与本次运行迭代范围. 一个 runtime 实例只能执行一次 `run`, 重复训练或推理需新建实例.

GPU 正式训练前, 用代表性输入测量 batch 等待、H2D、计算和端到端吞吐, 确认数据准备能与 GPU 计算重叠. `profiler.profile_section` 使用 `"train"`, `"eval"` 或 `"none"`; profiling 实现在 [profiler.py](qmodel/components/profiler.py).

## 6. 如何恢复和评价 checkpoint

顶层 checkpoint 写入 `root_dir/ckpt/iter_<iteration>.pt`, 保存 model、optimizer、scheduler、scaler、iteration 和 config. 设置 `load_from_iter` 后, Trainer 从当前 `root_dir` 加载状态, 并按 `iteration * batch_size` 恢复 sampler 的 global sample 偏移. 实现见 [trainer.py](qmodel/core/trainer.py) 和 [checkpoint.py](qmodel/components/checkpoint.py).

恢复依赖一致的数据顺序、采样设置和模型配置. 当前 checkpoint 不保存 Python/NumPy/Torch RNG 全状态, 也不验证数据或代码 identity; 任意实验的 bit-exact 恢复需要消费者另行验证. 保存发生在对应 iteration 的训练 step 之前, 不自动保证退出时额外保存最终 checkpoint.

训练中评价会读取对应 iteration 的 checkpoint, 因此评价时刻必须已有该 checkpoint. 独立 `evaluate()` 遍历 `evaluator.eval_checkpoint_iter` 中大于 0 的值; `eval_all_num_iters=0` 表示读取完整评价 loader.

顶层 `Evaluator` 有明确的金融数据格式: prediction 和 target 通常为 `(B, 1)`, metadata 为 `(B, 3)` 的 `code`, `date`, `time`. 它重建全局行顺序, 生成 `StockCode` / `DateTime`, 计算 global Pearson IC, 并写入 `root_dir/eval/eval_<iteration>.feather`. 日期与时间编码必须符合 [util.py](qmodel/util.py) 的转换规则, 不能直接假定支持任意日期表示.

不同输出维度、metadata 或评价口径应使用底层 `AsyncInference` 配合项目自己的输出和统计逻辑. 顶层评价不会自动使用 `qeval` 的 finite mask、ties 或分组契约. 具体实现见 [evaluator.py](qmodel/core/evaluator.py).

## 7. 如何验证修改

安装开发依赖和上述运行时依赖后, 在仓库根目录运行:

```bash
python -m pytest -q tests/unittest
```

该目录包含 sampler、AMP、同步/异步执行、profiling 和分布式运行测试. 部分测试需要 CUDA 或多张 GPU, skip 不代表相应路径已验证. 测试会产生临时训练输出, 不应在正式 run 的目录运行. 修改 CUDA Graph、DDP 或恢复行为时, 还需在目标项目验证数值一致性、样本顺序、checkpoint 恢复和供数吞吐.

算子级 FLOPs 估算的接口和统计边界见 [FLOPs 文档](docs/flops.md).
