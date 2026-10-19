---
id: training-hang-communication
title: 训练卡住排查：通信等待型 GPU Hang
status: draft
visibility: public
publication_approval: approved
publication_scope: adapted_body_only
deployment_status: published
published_url: https://chrisfang1999.github.io/aiinfra-notes/training-hang-communication.html
sync_status: synced
updated: 2026-10-14
---

# 训练卡住排查：通信等待型 GPU Hang

训练不再推进，GPU 却持续显示 100% 利用率，并不矛盾：通信 kernel 可能仍在执行等待或轮询，而有效训练已经停滞。排查的重点是找到通信无法继续的原因，而不是把高利用率直接等同于正常计算。

[快速排查 SOP](训练报错排查SOP.md) · [检测规则与详细笔记](训练报错排查详细解释.md)

## 一、问题定义与典型现象

训练任务没有退出，但连续几分钟 step 不再更新，训练进度日志也停止输出，可视为疑似训练卡住并进入排查。这里采用现场经验，不设置严格的时间阈值；日志中明确标记的 Checkpoint 保存等阶段不在本文讨论范围内。

当前环境中的典型通信等待型 GPU Hang 表现为：多数 GPU 持续 **100% 利用率且功率较低**，部分 GPU 为 0%；少数 rank 的连续堆栈停留在不同执行路径。通信等待的 GPU 对应的 CPU 堆栈不一定显示在 allreduce 调用处。

该组合是调查入口，不是所有故障必须具备的特征。全部 GPU 为 0% 的阻塞归入其他类别；GPU 执行异常时，致因 rank 也可能显示高利用率。本文覆盖常见路径，不覆盖所有训练阻塞。

## 二、现象背后的机制

### 为什么等待也会显示 100%

nvidia-smi／NVML 的 GPU utilization 表示：过去采样周期中，有一个或多个 kernel 在 GPU 上执行的时间占比。它不直接表示计算单元的利用程度、有效计算量或训练进度。[NVIDIA 指标说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

```text
GPU 利用率 ≈ 窗口内至少有一个 kernel 执行的时间 ÷ 窗口总时间
```

监控每 30 秒读取一次，是采集频率，不等于该指标统计最近 30 秒。

集合通信要求对应通信组内的 rank 匹配参与；并非整个任务的所有 rank 必须先同时就绪，kernel 才能启动。部分 rank 的通信 kernel 可以已经执行，再等待对端的数据或进度。[NCCL 集合通信说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

NCCL 的部分 GPU 通信实现通过循环检查对端进度或缓冲区状态来等待。当条件持续不满足时，kernel 仍执行轮询指令，却无法推进通信，因而可能持续显示 100% 利用率。[NCCL 设备端实现](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h)

一个典型因果链是：

```text
某个 rank 长时间停在 CPU 路径
    → 未推进对应的集合通信
    → 其他 rank 的通信 kernel 等待数据或进度
    → kernel 持续执行等待，利用率可保持 100%
    → 通信无法完成，训练 step 停止推进
```

### GPU 利用率与 SM 活跃率

GPU 利用率高，只能说明持续有 kernel 执行。要了解这些活动覆盖了多少 SM，可以结合 SM 活跃率（SM Active／SM Activity）观察。

| 指标 | 统计口径 |
| --- | --- |
| GPU 利用率 | 窗口内有一个或多个 kernel 执行的时间占比 |
| SM 活跃率 | 每个 SM 至少有一个活跃 warp 的时间占比，再对所有 SM 求平均 |

![GPU 利用率与 SM 活跃率的统计原理对比](配图/GPU利用率与SM活跃率统计原理示意图.png)

图：简化为 4 个 SM，只有一个在整个窗口内持续活跃，GPU 利用率仍可为 100%，全卡平均 SM 活跃率为 25%。该示例解释统计口径，不代表 NCCL 固定使用一个 SM，也不意味着通信等待时 SM 活跃率一定低。

SM Active 与 SM Occupancy（驻留 warp 相对容量）不同；等待内存或通信进度的 warp 也可能计为活跃。因此，GPU 和 SM 两项指标都不能单独证明有效训练计算正在推进，排障时仍要结合 step、功率、日志与堆栈。[NVIDIA DCGM 指标定义](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)

### 为什么功率却较低

等待或轮询的活动强度可能低于正常训练中的密集计算。因此，kernel 持续执行与较低功率可以同时存在。准确的说法是“没有推进有效训练计算”，而不是“完全没有执行指令”。

低功率与持续 100% 的组合是当前环境的经验判据，不能仅凭两项指标确认 GPU 上只有通信 kernel，也不能直接证明最终根因。

![通信等待时 GPU 利用率持续为 100% 的原理示意图](配图/GPU通信等待利用率原理示意图.png)

图：某个 rank 未推进对应通信，其他 rank 的通信 kernel 持续等待／轮询，因此利用率可保持 100%，而训练 step 不推进。该图是机制示意，不代表所有 Hang 都符合这一状态组合。

检测配置示例每 30 秒采样，先逐卡判断连续 14 次命中，再统计卡住的 GPU 是否达到任务总数的 50% 及以上。机型功率阈值及完整规则统一维护在[详细笔记](训练报错排查详细解释.md)，不将工具判据当作机制定义。

## 三、排查流程与证据判断

| 步骤 | 检查动作 | 判断与去向 |
| --- | --- | --- |
| 确认现场 | 检查 step、训练日志及任务是否退出 | 确认疑似卡住，区分存活现场与退出前留存记录 |
| 查多机日志 | 找具体代码异常、I/O 异常，追查 timeout 之前的事件 | 定位原始异常；同时保留最终 watchdog 日志中的通信信息 |
| 查平台与硬件 | 对齐平台事件、OOM 证据、网卡与网络异常、存储流量 | 判断是否与异常节点和时间吻合，不凭单项曲线归因 |
| 查跨 rank 堆栈 | 以分组为主要入口，结合连续快照 | 找偏离多数执行路径且持续停留的 rank，同步查其日志 |
| 进入原因分支 | 关联通信调用、进程状态及相关节点证据 | 区分阻塞位置与其发生原因，按下节处理 |

没有明显日志或硬件异常时，也直接查看堆栈，不必满足自动检测阈值才开始分析。

### 堆栈分析：横向找偏离，纵向看停留

工具从训练启动起每分钟采集堆栈，发生训练卡住时输出最近 10 次快照。先看跨 rank 分组，再看同一 rank 是否在连续快照中持续停留于同一位置。无响应 rank 的历史记录代表最后可观测位置。

典型情况是多数 GPU 已处于通信等待，少数 rank 的连续 CPU 堆栈停在数据 wait 或 NumPy 计算。后者是优先调查的阻塞来源；为何数据没有返回、为何计算未完成，仍需代码报错和系统日志支持。正常职责差异也会产生不同堆栈，不仅凭“少数不同”认定故障。

**CPU 堆栈不等于 GPU 当前执行位置。** GPU 调用通常异步执行，CPU 提交通信后可能继续运行后续代码，或停在其他同步位置。因此，处于通信等待的 GPU 对应的 CPU 堆栈不一定显示在 `all_reduce`；也不能仅凭 CPU 已进入后续代码就认定它跳过通信。应结合连续堆栈、日志与通信信息交叉判断。[PyTorch CUDA 异步执行说明](https://docs.pytorch.org/docs/2.14/notes/cuda.html)

### watchdog 日志仍有定位价值

timeout 通常是异常传播后的表现，但其中的通信操作和数据量仍能提供线索。比较 watchdog timeout 日志中的通信信息，可以发现跨 rank 通信 size 不一致的线索。比较时需确保对应同一通信组及匹配的通信操作，不能只看最后一条错误，也不能直接忽略它。

[PyTorch 对 watchdog timeout 与跨 rank 诊断的解释](https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/)说明了超时表现与致因 rank 不一定一致。

## 四、常见原因与对应处理

堆栈定位后，按 GPU 状态选择调查重点。以下是实战中的诊断入口，指标组合本身不能直接确认根因。

### 多数 GPU 为 100%，少数为 0%

本组案例已确认高利用率 GPU 在通信等待，重点检查低利用率 GPU 对应的 rank。

| 少数 rank 的表现 | 下一步检查 | 处理与验证 |
| --- | --- | --- |
| 长时间停在 NumPy 矩阵乘法 | 检查矩阵大小、维度和耗时，用相同输入离线复现 | 根据复现结果调整 API 或计算实现，必要时替换 NumPy 路径；验证结果与耗时 |
| 停在 DataLoader 数据队列等待 | 查 worker、`__getitem__` 和 `collate_fn` 报错及异常处理，追查为何数据未返回 | 修复数据生产路径，验证异常样本也能正确处理并完成组 batch |
| 有索引越界、Tensor 维度不匹配等明确错误 | 沿异常位置检查输入、代码及异常后的执行行为 | 修正代码，覆盖原触发条件验证 |
| 已进入后续代码，其他 rank 等待通信 | 检查是否手动跳过通信，或条件分支使通信顺序不同；结合通信记录排除正常异步推进 | 修正跨 rank 通信参与逻辑，再执行完整 epoch 冒烟测试 |

**NumPy API 的范围。** 常见入口包括 `np.matmul`、`@` 和二维输入下的 `np.dot`。高维输入下 `matmul` 与 `dot` 的语义不同，单纯换 API 不保证解决问题，应通过相同输入复现后选择等价且合适的实现。[NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)、[NumPy dot](https://numpy.org/doc/2.0/reference/generated/numpy.dot.html)

**DataLoader 异常分支。** 实战中既有 worker／collate 显式报错，也有 `__getitem__` 捕获问题后没有正确构造替代训练数据的情况。除了查日志，还应检查异常分支返回值的结构、shape、dtype 是否符合后续处理要求。队列 wait 只说明主进程没有取得所需数据，不能单凭该位置区分 worker 报错、数据处理未完成或异常返回；错误替代数据也可能引发后续异常或跨 rank 执行分歧。[PyTorch DataLoader 说明](https://docs.pytorch.org/docs/2.14/data.html)

数据供给、CPU 和存储性能的进一步排查见[训练慢详细解释](../训练慢与变慢排查/训练慢与变慢排查详细解释.md)。

### 所有 GPU 都为 100%

本组通信等待案例缺少明显的低利用率 rank，应转向通信匹配、并行配置与链路检查。不能仅凭所有 GPU 为 100% 就认定都在等待同一次通信，也不能预期 CPU 堆栈一定完全相同。

| 诊断方向 | 检查重点 | 处理与验证 |
| --- | --- | --- |
| 通信不匹配 | 同一通信组内，操作类型、顺序、tensor shape／size、数据类型及调用次数是否对应；各 rank 可能等在不同一次 collective | 查造成参数不匹配、跳过或顺序错位的逻辑，修正后覆盖触发条件验证 |
| 框架或并行策略相关阻塞 | 在显存允许的条件下，改变并行配置作消融对照，如 ZeRO-3 → ZeRO-0 | 实战中切换后恢复运行，作为当时有效的规避方案；底层原因仍未查明 |
| IB 通信异常 | 实际使用的 IB 端口、链路状态、相关报错与故障窗口是否吻合 | 当前任务换节点运行，原节点用其他训练任务及 NCCL Tests 交叉检查 |

通信计数在对应通信组内比较；size／shape 的匹配要求取决于具体 API，不概括为所有通信操作都要求各 rank 输入形状相同。

**ZeRO 切换的证据边界。** 此项是实际采用的处理经验：消融切换后可运行，但尚未确认原因。ZeRO-0 关闭 ZeRO 分片优化，仍有分布式通信；切换会改变显存需求和通信模式。恢复运行不能直接证明 ZeRO-3 或框架本身存在缺陷，应记录为规避方案而非已完成根因修复。[DeepSpeed ZeRO 说明](https://deepspeed.readthedocs.io/en/stable/zero3.html)

### 连接报错不等于网络已经损坏

可能遇到的关键词包括：

```text
socketProgress: Connection closed by remote peer
Connection reset by peer
ncclRemoteError
```

具体报错文本随版本和执行路径变化。NCCL 对远端错误的定义同时包含网络异常和远端进程提前退出，因此需要向对端更早的报错追查。[NCCL 错误类型](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html)、[socket 实现](https://github.com/NVIDIA/nccl/blob/master/src/misc/socket.cc)、[连接重置日志示例](https://github.com/NVIDIA/nccl/issues/947)

### GPU 执行异常的教学示例

假设某个 rank 的自定义 CUDA kernel 因循环或同步逻辑错误无法结束，排在其后或依赖它完成的通信就不能推进；其他 rank 可能等待或超时。此时真正出问题的 GPU 也可能是高利用率，CPU 堆栈则可能只显示等待 GPU 的位置。

该示例说明执行依赖：计算 kernel 无法完成也会阻塞后续通信，不据此直接认定硬件损坏。

## 五、恢复验证与预防

### 代码问题

先从逻辑上确认通信 size、参与条件和调用顺序已经匹配，再跑一个完整 epoch 的冒烟测试。以完整 epoch 的测试结果验证修复效果。对于依赖特定数据或分支的问题，测试应覆盖原触发条件。

### 网络问题

采用两条并行路径：

- 当前训练任务更换节点重新运行，使用已有恢复能力继续训练，观察 step 与吞吐是否恢复。
- 原节点运行其他训练任务和 NCCL Tests，检查报错或性能异常是否复现。

换节点后恢复，只能提示问题与原节点或关联路径有关，不能单独证明网卡损坏。其他任务及 NCCL Tests 不复现，也不能完全排除间歇性故障。重启成功不等于已经证明网络闪断。

### 观测与预防

- 对原节点测试记录实际使用的端口，比较测试前后相关错误计数器增量，结合流量、吞吐与链路事件判断。
- 修改手动通信逻辑时检查跨 rank 参数、分支及顺序；保留代表性触发条件用于回归。
- 保留日志、连续堆栈及故障窗口指标，为下次间歇性故障提供证据。

网卡累计值、发送等待与错误事件的区别见[详细笔记](训练报错排查详细解释.md)。恢复训练后仍需验证进度、吞吐与结果正确性。
