---
id: torchft-research-record
visibility: private
status: draft
updated: 2026-10-19
---

# torchft 调研与验证记录

## 范围与授权

用户要求另起 torchft 专题，重点为稳定性监控、容错和重启。本次直接授权调研与写入知识库，不涉及安装依赖、集群故障注入或博客发布。

正文《torchft：稳定性监测、容错与重启》与 NVRx 同属“稳定性与排障 / 训练容错与恢复”。本记录、来源核对信息与验证计划不进入公开正文。未代写用户理解或推断已掌握。

## 版本口径

- 调研日期：2026-10-16。
- GitHub ls-remote 实际查询 main 提交：612a7b0ac49684ab71b61b6af6bc2905a7321f52。
- 已按该提交下载 10 个官方源文件到临时目录 /tmp/torchft调研来源，仅静态阅读，不安装或执行。
- 固定提交源码与在线滚动 main API 文档分开对待。官方 PyPI 查询显示发布包 0.2.0；不宣称与本次 main 提交一致，不把新 API 自动下放到发布包。
- GitHub REST API 查询受限后改用 git ls-remote 固定提交；个别源码读取遇 TLS 中断，重试取得。失败请求不记为已核验内容。

## 已核对的核心机制

| 模块 | 核对内容 |
| --- | --- |
| README / pyproject | DDP/HSDP、LocalSGD/DiLoCo 范围；依赖；Rust/maturin；main 与 release 区别 |
| manager.py | start_quorum、should_commit、恢复、参与规模、重试及日志；step 计数在 wrapper 底层 optimizer 更新之前推进 |
| optim.py | zero_grad 发起 quorum，step 检查 should_commit 后才调用底层优化器；closure 不支持 |
| ddp.py | dummy PG 与 comm_hook 路由，初始状态同步、高级配置限制；恢复时 bucket 一致性安排 |
| process_group.py | 可重配置与错误处理；NCCL abort 尝试及死锁限制，不宣称所有 CUDA 故障可原进程恢复 |
| data.py | 副本组分片、故障下丢失数据的边界、StatefulDataLoader 建议 |
| src/manager.rs | 后台心跳任务与训练进展分离；组内提交一致；供体和恢复安排 |
| src/lighthouse.rs | 心跳、quorum、min_replicas、join 等待、状态页及 quorum_id 变化条件 |
| train_ddp.py | 具体 Manager/DDP/Optimizer 接入；模型与优化器状态回调；数据本地状态及持久保存仍需应用实现 |

在线资料包括 Protocol、Assumptions、Manager、Optimizers、Process Groups、Checkpointing、Data 和 DDP；官方链接在正文保留。

## 关键取舍与不确定项

1. 不把副本组当成固定一台机器或一个 rank；正文用 DDP 与 HSDP 两种布局区分。
2. 不把后台心跳当“有效训练前进”；以 commit、实际更新、吞吐和持久恢复点构成分层监测建议。
3. min_replicas 是可配置下限，不能取代协议一致性前提；官方 assumptions 的多数 / 交叠叙述与当前 Lighthouse 动态存活集合检查不能概括成固定原始世界多数。正文不宣称剩余任意一个副本就可安全继续。
4. 同一副本组内 should_commit 一致，不同组可能不同；不写成全局原子提交事务，也不承诺回滚任意应用副作用。
5. 现有普通 DDP 完整代码不能直接更换 import 后当作 torchft 生产模板。AMP、梯度累计、no_sync、scheduler 和外部平台生命周期仍需测试。
6. 源码有结构化 logger，依赖出现 OpenTelemetry，但未审计完整导出配置；不宣称自动提供完整 Prometheus 指标或硬件监控。建议告警清单明确为自行接入设计。
7. torchft 不负责重新申请节点；外层 torchrun / 平台可否只重启失效副本组需要环境验证。与 NVRx 叠加也不能按默认配置自动安全协同。
8. 研究未覆盖全部算法、后端与兼容性；LocalSGD/DiLoCo 不作为主线，不评判两工具性能优劣。

## 后续验证矩阵

优先做无故障多副本组基线，再依次测单组退出、组内 rank 失效、后台心跳仍正常的训练停滞、落后组恢复、Lighthouse 重启、连续提交失败、全组故障和持久 checkpoint 恢复。每项观察：协议日志、实际参数与 optimizer 状态、数据进度、恢复首个有效 update、检测 / 重配置 / 传输开销。

如要扩展为实战教程，先确认真实副本布局、平台退出行为、NCCL/PyTorch 组合和数据语义要求。不在没有测试时给出固定超时或成功率承诺。

## 检查范围

仅静态阅读指定模块和官方文档，检查 Markdown 结构、元数据与本地链接；没有运行 torchft、Lighthouse、GPU/NCCL、故障恢复或 benchmark。正文的示例片段仅解释调用顺序。

## 固定源码清单

以下散列对应临时下载的原始文件，用于后续复核；不是完整代码审计结论。

| 文件 | SHA256 |
| --- | --- |
| README.md | 199b8ae8d379bcaa8421ec75a6695e6277639980c6b6eb7b659a3a210cee3d7c |
| pyproject.toml | 7fe49bc9b04f858f188e467feccb4fcb47d8c0664a7e0a1fa720674a639426b8 |
| src/lighthouse.rs | 8b9c137a0a6508959a63860a608515bb29fbddb233b661613f73f6317efa96b3 |
| src/manager.rs | e3eaa8e00f877ba893608be00b7e35e27c8b07f2265c9a11e29bb9a2f611b874 |
| torchft/data.py | f6d5f392d45a3c5aafb321a963e9203a5eb24dd196dbb870c9ee516801a37b6a |
| torchft/ddp.py | 012ee40a798c794facea963def416656cb83cdad6db9b306f98398388ccdd343 |
| torchft/manager.py | b1f30d5e48d64546eaa930cc0418674c1ca13332abfbec0922e94c55297463eb |
| torchft/optim.py | 724f0020797c90ac4aa3c98ad4dd6674832098a7f2b9cb694b8efa0c006bc088 |
| torchft/process_group.py | 41e55cbf0f650ced70ac6090f7d24dbba55d694f4fd2bd023f9cdaac0cf86d30 |
| train_ddp.py | 04ed42f5638f6113c8f76da65a491feafcc1c610d0116c2df6ec502efa9da095 |


## 2026-10-16：可读性与 K8s 接入要求

按用户要求优化现有正文，前置容错逻辑与 Lighthouse / replica group / quorum / ProcessGroup / commit 概念；保留原有监测、训练语义和版本边界。新增 Kubernetes 与 Python 接入职责，明确 Lighthouse 可独立启动、不强制依赖 K8s；Deployment + Service 为建议组合，单实例重建不代表协调状态高可用，Recreate 也不能代替旧实例隔离。

核对固定提交 README 独立启动命令及官方协议假设，补充 Kubernetes 官方 Deployment / Service / Disruptions 资料。新增两张 PNG 原理图与中文命名绘图脚本，说明图中的 Pod / 副本组映射只是示意。已查看图像，检查章节编号、Markdown 围栏、本地链接及配图引用；未执行 torchft 或 K8s 部署实验，未发布博客。


## 2026-10-19：参考 NVRx 教程重组

### 授权与结构

用户要求参考 NVRx 讨论及修改建议优化 torchft 文档。本次执行文档编辑，不涉及博客发布或跨聊天通知。保留正式文档文件名、ID、公开草稿与待发布状态，正文中的协作取舍留在本记录。

- 前置功能收益、Lighthouse / Manager / 副本组 / quorum / commit，保留两张原有原理图。
- 增加 Quick Start：Python 改造骨架、关键参数注释、独立 Lighthouse 与三个单卡副本组启动命令。单机三卡仅为教学接入布局；三台各八卡的 HSDP 组织仍用于概念解释，没有声称改 nproc_per_node 即可完成 HSDP。
- 容错章节融合提交控制、重配置、外部进程重建、确定性代码错误与有限重试。
- 单独解释同伴状态恢复、持久 checkpoint 与异步保存的区别，不能将 NVRx AsyncCheckpointIO 或 FT Sections 能力移植为 torchft 原生能力。
- 监测章节补充阶段计时、慢 rank 与归因所需的外部观测入口、采集开销和低频输出边界；没有承诺仅 report 时同步或内置慢节点报告。
- 用户已具备跨 rank 日志检索与重启分组，正文采用通用的现有平台集成建议，不建议重建日志系统。
- 超时、版本安装和故障注入验收移入附录。保留数据丢弃、有效 batch 变化、AMP / 累计兼容性、NCCL 恢复和 Lighthouse 单实例边界。

### 来源与验证

沿用 2026-10-16 固定提交 612a7b0ac49684ab71b61b6af6bc2905a7321f52；本次没有重新认定最新发布版本。静态核对临时保存的 train_ddp.py、Manager、DDP、OptimizerWrapper、DistributedSampler 和 README；在线重读官方 Optimizers 页面。固定提交 GitHub 页面本次 web 读取 cache miss，未据该失败请求宣称获取新证据。

新增 Python 骨架通过 AST 语法检查，各 Bash 代码块通过 bash -n；本地文档链接与两张配图均存在，小节编号和代码围栏完整。骨架有明确的应用实现占位，不作为可直接运行的生产脚本。未安装依赖，未执行 GPU、故障注入、性能或 K8s 测试。未发布博客。
