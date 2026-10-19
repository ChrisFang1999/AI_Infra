---
id: training-slow-sop
title: 训练慢与变慢排查SOP
status: draft
visibility: public
publication_approval: approved
publication_scope: adapted_body_only
deployment_status: published
published_url: https://chrisfang1999.github.io/aiinfra-notes/training-slow-sop.html
sync_status: local_changes_pending
updated: 2026-10-15
---

# 训练慢与变慢排查 SOP

用于相同硬件下的明显效率异常：训练持续慢、运行中变慢或耗时波动。以稳定阶段的 step time、吞吐和跨 rank 差异为入口，逐步定位数据、CPU、GPU 或通信问题。

[详细解释](训练慢与变慢排查详细解释.md) · [返回知识库](../../../README.md)

## data_get_time 是什么

启用多进程预取时，DataLoader 会让后台 worker 提前读取和处理样本，再由 `collate_fn` 把多个样本整理成一批数据（batch）。准备好的 batch 会先放入队列，等待训练进程取用。**这个队列可以理解为“待使用 batch 的缓冲区”**：训练处理当前 batch 的同时，后台继续准备后面的数据。

```text
后台准备：读取样本 → 预处理 → collate 组成 batch
                                    ↓
                              待使用 batch 队列
                                    ↓
训练进程：取出下一批数据 → 执行训练
```

**data_get_time 就是训练进程“开始取下一批数据”到“拿到这批数据”的时间。**

- 下一批数据已经准备好：通常可以很快拿到，data_get_time 较短。
- 下一批数据尚未准备好：训练进程需要等待，data_get_time 会变长，应继续检查数据生产与传递链路。

它测的是取数环节，不是后台读取、预处理和 collate 的总耗时。长时间等数据可能延后后续 GPU 计算，但 GPU 是否实际空闲，还需结合时间线确认。

需要在训练代码中自行计时，每个 rank、每个 micro-batch 独立记录。若取数接口还包含数据搬运等操作，应注明范围；接入示例见[详细解释](训练慢与变慢排查详细解释.md)。

## 快速排查路径

**确认比较条件 → 检查跨 rank 取数耗时 → 对比 GPU 与主机指标 → Profile 区分长操作和计算间隙 → 沿证据定位 → 对照验证。**

| 步骤 | 检查动作 | 判断与下一步 |
| --- | --- | --- |
| 1. 确认异常与 baseline | 对照正常的 step time、吞吐、耗时均匀性及近期变更 | 排除启动 warmup 和已知特殊阶段；确认是否是工作量、执行配置或环境变化 |
| 2. 检查数据供给 | 每 rank 独立观察 data_get_time，并关联对应 micro-batch | 长等待进入数据链路；没有该指标时先按训练代码接入，或用阶段标记与 trace 辅助调查，不凭 GPU 利用率反推取数耗时 |
| 3. 指标初筛 | 对比 GPU 利用率、功率、显存，辅助看 SM 活跃率；结合 CPU 与存储 | 找局部节点差异；怀疑设备异常时查温度、频率和降频原因，必要时隔离检测 |
| 4. 查单次异常长操作 | 用 Profile 区分 CPU 操作、GPU kernel、memcpy 与通信事件，对照正常执行 | 调查操作次数、输入规模、精度、执行路径及设备状态；结合正常耗时和输入规模判断 |
| 5. 查通信与计算间隙 | 通信长时跨 rank 对比；无长操作时看各 stream 与 CPU／CUDA 调用 | 通信等待不直接等于网络慢；间隙按“未及时提交”或“已提交但依赖未满足”分流 |
| 6. 针对性定位 | 数据查 worker／collate／存储；CPU 查串行计算／竞争／同步；GPU 查长操作和执行状态 | 找最先延后的环节，并关联其他 rank 的等待，不把等待最长者自动认作根因 |
| 7. 验证 | 相同条件下，一次调整一个因素，比较稳定窗口吞吐与耗时分布 | 确认端到端改善和正确性，记录未解决问题；正式验收不依赖重型采集开启时的性能 |

## 数据等待分流

| 分布 | 优先方向 |
| --- | --- |
| 不固定 rank／step，随机且反复变慢 | 整体数据链路、共享存储或 CPU 资源 |
| 固定 rank／节点持续慢，其他正常 | 局部节点、worker 或分配到的数据路径 |
| 长期稳定，偶发孤立慢 step | 异常样本或特殊处理，同时排除瞬时资源波动 |

上述是调查方向，仍需日志、输入、运行状态或复现证据确认。

## 耗时口径

- **step_time**：每个 micro-batch 获取数据前开始，到该 micro-batch 的训练调用返回结束。执行参数更新的 micro-batch 包含更新调用；其他 micro-batch 不人为等待后续更新。使用墙钟计时，不额外逐步同步 GPU。
- 同时记录 micro-step、是否更新参数和 warmup／稳定阶段；不同类型的 micro-batch 分组比较。

GPU 异步执行，不能把 step_time 减去 data_get_time 直接当作 GPU 计算时间。取数长也不自动证明整个等待期间 GPU 都空闲；结合时间线确认影响。

## 排查结果记录

留下 baseline 与实际配置、异常时间及 rank／节点、关键指标和 trace 线索、候选原因、对照改动及验证结果。证据不足则明确缺口，继续采证或复现。

CPU 与 GPU 的进一步诊断、耗时采集和分布式初筛见[详细解释](训练慢与变慢排查详细解释.md)。
