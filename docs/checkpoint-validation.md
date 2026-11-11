---
intention: 记录 qmodel Checkpoint 的恢复语义、验证证据和已知限制.
---

# Checkpoint 验证

## 1. 保存和恢复解决的问题

Trainer 在 optimizer 更新完成后保存, 避免后台 post-step 与下一次更新交错. Torch AMP 跳过更新时保留 signal, global_step 不增加; 实际消费的 batch 位置单独记录. scheduler 恢复时重新绑定 optimizer 的 param groups.

启动从同一实验目录自动选择最大 global_step 的已提交文件, 使用文件中的 config. 新格式保存各 rank 的 RNG、模型状态和 sampler 消费位置. 原子保存失败直接报错, signal 不清零. 旧格式可加载时明确提示非精确续训.

实现入口为 qmodel/core/trainer.py, 持久化实现为 qmodel/components/checkpoint.py.

## 2. 续训如何验证

环境为 Python 3.10、Torch 2.5.0a0+872d972e41.nv24.08、三张 RTX 3080. 环境中的旧 ONNX/protobuf 需要设置 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python.

测试 tests/unittest/test_checkpoint_resume.py 使用独立进程完成连续训练、step 14 保存退出、自动恢复至 step 30. 模型包含 Dropout, 配置包含 Adam 和 scheduler. 恢复进程故意传入不同 learning_rate 和 num_iters, 验证 Checkpoint config 优先.

| 路径 | 最终模型 | optimizer 最大绝对差 |
| --- | --- | --- |
| 单 GPU eager | 完全一致 | 0 |
| 单 GPU Torch AMP, 2 workers | 完全一致 | 0 |
| 单 GPU CUDA Graph/custom AMP, 2 workers | 完全一致 | 0 |
| 3 GPU DDP eager | 完全一致 | 2.33e-10 |
| 3 GPU DDP CUDA Graph/custom AMP | 完全一致 | 1.17e-10 |

所有路径的 scheduler、AMP、RNG、global_step 和采样位置完全一致. DDP optimizer 浮点状态使用 rtol=1e-6、atol=1e-9 验证; 其他状态不放宽精度. DDP Graph 恢复后使用真实更新重新 warmup reducer, 测试覆盖再次 capture.

另行验证 Torch AMP 溢出跳步、signal 成功清零、原子序列化失败保留旧文件、临时文件不参与恢复、损坏文件报错、旧格式状态加载和 sampler 跨 epoch 位置恢复. 原有回归中的 fresh-start fixture 已适配自动恢复行为, 分布式训练/评价及 profiling 回归通过.

## 3. 性能和限制

小模型 16-step smoke 测得 39.6 steps/s, 平均 batch 准备等待 0.965 ms, GPU 换批空档 0.973 ms, CPU 数据准备与 GPU 计算累计重叠 5.87 ms. 当时三张 GPU 已有高负载, 结果仅证明该工作负载仍存在准备/计算重叠, 不代表正式训练吞吐.

精确数据衔接要求 Dataset 对相同 indices 返回相同 batch, 以及相同代码、数据和 DDP 布局. 不恢复 worker 内随机变换或外部数据状态. DDP reduction 和 CUDA 非确定性运算可能引入浮点舍入差异; 不承诺任意模型 bit-exact. 旧格式缺少 RNG 状态, 只保证恢复可用状态, 不保证连续训练轨迹一致.

构建前清理旧 build 产物, 防止相同时间戳的过期源码进入 wheel. 发布目标为 CPython 3.10 Linux x86_64 wheel, 不是通用 Python wheel.


## 4. 0.2.21 暂停与项目验收

新增接口为 TrainResult, get_project_state/load_project_state, save_pause_checkpoint, restore_training_state 和 close_training_resources. 不新增独立工具或服务; 持久化复用 CheckpointSaver, 测量复用 AsyncProfilingHelper 和 MyProfiler.

暂停对照使用真实返回和进程退出, 不再使用 os._exit. 项目状态测试在子类初始化后创建独立 generator 和滚动窗口, 验证它们与内置 loss window 一并恢复. 完整格式标记升级为 2, 仍能读取旧格式.

各项目应固定模型、Dataset、batch、精度和 GPU 布局, 配对比较 0.2.20 连续训练、0.2.21 连续训练和 0.2.21 恢复后的稳态. 测量期间禁用 profiling 的人为延迟, 使用相同数据区间, 分别记录 batch 等待、GPU 空档、重叠和有效 steps/s. 用基线重复测量的波动范围判断可测退化, 样本量不足或外部竞争明显时标记未确定. 加载、DDP warmup 和 Graph recapture 开销单独报告.

相同负载的 MFU 相对变化使用有效吞吐比; 绝对 MFU 使用项目已有 FLOPs 和硬件峰值口径. qmodel 的框架测试不能代替真实项目 MFU 验收.

框架 smoke 使用同一无人工延迟的小模型, 与 0.2.20 交替运行 5 对 16-step 测量. baseline: 中位数 42.29 steps/s, 范围 33.66-44.10. updated: 中位数 43.75 steps/s, 范围 38.28-45.70. 此短测量包含启动开销, 且 GPU 持续存在外部竞争; 观察到 CPU/GPU 重叠, 但稳态 MFU 无退化的结论仍为未确定, 需由项目真实负载验收.

最终全量测试 134 项通过, 6 项为 torchrun 子进程入口而跳过; 另行追加的后台统计异常测试通过. wheel 安装后的多 worker/Torch AMP 暂停重启状态对照通过, 暂停返回时无存活 DataLoader 子进程, 训练子进程退出后其 PID 不再出现在 GPU 进程列表. wheel 中全部 Python 源码与工作区逐文件一致, twine metadata 检查通过.
