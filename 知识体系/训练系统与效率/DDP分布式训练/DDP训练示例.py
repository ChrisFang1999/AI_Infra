"""原生 DDP 教学示例：一进程一卡；CPU/Gloo 仅用于流程自检。"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class SyntheticDataset(Dataset):
    """按索引生成确定性样本，不依赖下载；不是存储性能测试。"""
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        features = torch.randn(32, generator=generator)
        label = int(features[:4].sum() > 0)
        return features, label


def collate_batch(samples):
    # 多进程加载时，此函数在接到该 batch 索引任务的 worker 中执行。
    features, labels = zip(*samples)
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def seed_worker(worker_id):
    # PyTorch 已设置 worker 的 torch seed；为 Python 随机增强初始化种子。
    random.seed(torch.initial_seed() % (2**32))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='CPU/Gloo 流程自检')
    parser.add_argument('--amp', choices=['off', 'fp16', 'bf16'], default='off')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--samples', type=int, default=1040)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', default='运行输出')
    args = parser.parse_args()
    if min(args.epochs, args.samples, args.batch_size, args.accumulation) < 1 or args.workers < 0:
        parser.error('轮数、样本数、batch 和累计次数必须为正，workers 不能为负')
    if args.cpu and args.amp != 'off':
        parser.error('CPU 自检仅支持 --amp off')
    return args


def main():
    args = parse_args()
    # torchrun 为每个训练进程注入身份信息；LOCAL_RANK 是本机编号，不能当全局 RANK。
    # 双机各 8 卡时，每台 LOCAL_RANK 都为 0～7，整个任务的 RANK 为 0～15。
    local_rank = int(os.environ['LOCAL_RANK'])
    if args.cpu:
        device, backend = torch.device('cpu'), 'gloo'
        torch.set_num_threads(1)
    else:
        # 为当前进程选择默认 CUDA 设备；此处编号相对于 CUDA_VISIBLE_DEVICES 可见列表。
        # 一个训练进程独占一张 GPU；这一步只绑定设备，还没有建立分布式进程组。
        torch.cuda.set_device(local_rank)
        device, backend = torch.device('cuda', local_rank), 'nccl'
        if args.amp == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('当前设备不支持示例所需的 BF16')

    # 创建默认进程组：约定哪些 rank 共同通信，以及使用哪个后端执行 collective。
    # 所有参与训练的 rank 都要调用；此调用不创建训练子进程，也不包装或更新模型。
    # init_method 未传时采用 env://，从环境读取 RANK、WORLD_SIZE、MASTER_ADDR/PORT。
    # rank / world_size 也可显式指定；torchrun 已提供环境信息，本例不重复手工填写。
    # device_id 是可选设备参数，在支持的版本中可促使 NCCL communicator 提前创建；
    # 本例为兼容基础路径不传入，初始化返回后仍会进行实际 CUDA Tensor 通信验证。
    dist.init_process_group(
        backend=backend,                 # GPU 使用 nccl；CPU 自检使用 gloo。
        timeout=timedelta(minutes=5),     # 进程组操作的超时配置，不是整个训练的运行时限。
    )
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        # 测试实际 Tensor collective，而不只检查进程组初始化是否返回。
        probe = torch.ones(1, device=device)
        # AllReduce：对默认组内所有 rank 的 Tensor 求和，并原地写回每个 rank 的 probe。
        # 各 rank 必须以匹配顺序参与，且本例 Tensor 的元素数量和类型一致。
        # 每人贡献 1，结果应为 world_size；不是把结果只收集到 rank 0。
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        if probe.item() != world_size:
            raise RuntimeError('AllReduce 验证结果不符合预期')
        if rank == 0:
            print(f'通信验证通过：backend={backend}, world_size={world_size}', flush=True)

        torch.manual_seed(42)
        dataset = SyntheticDataset(args.samples)
        # 丢弃不能均分到 rank 的尾部，避免补齐重复；与 DataLoader.drop_last 不同。
        # num_replicas / rank 未传时从默认进程组读取；Sampler 分配索引，不传输样本内容。
        # shuffle=True 打乱全局索引；seed 在各 rank 相同，再用 set_epoch 改变每轮排列。
        sampler = DistributedSampler(dataset, shuffle=True, seed=42, drop_last=True)
        loader_options = {}
        if args.workers:
            # prefetch_factor=2：每 worker 的预取任务数，本例自动组 batch 时任务单位为 batch。
            # persistent_workers：跨 epoch 保留 worker；spawn：独立启动解释器，需入口保护。
            loader_options.update(prefetch_factor=2, persistent_workers=True,
                                  multiprocessing_context='spawn')
        # batch_size 是每 rank 的本地 batch，不是全局 batch；num_workers 也是每 rank 数量。
        # sampler 控制索引顺序，不能再同时设置 shuffle=True。
        # collate_fn 把样本列表整理成 batch；worker_init_fn 在各 worker 初始化时执行。
        # generator 控制 worker 基础种子，与 DistributedSampler 的全局 shuffle seed 分开。
        # pin_memory 在 GPU 路径使用页锁定主机内存，为后续主机到设备传输提供条件。
        loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                            drop_last=False, num_workers=args.workers,
                            collate_fn=collate_batch, worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(2026 + rank),
                            pin_memory=not args.cpu, **loader_options)
        if len(loader) == 0:
            raise ValueError('样本不足以分配给全部 rank')

        network = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
        # DDP 包装本进程已经放到设备上的完整模型：默认初始化同步模型状态，
        # 并注册 autograd hook / reducer，在 backward 中按 bucket 归约梯度。
        # 它不切分输入、不创建 DataLoader，也不代替 optimizer 更新参数。
        # device_ids：本进程模型使用的 CUDA 设备列表；一进程一卡传 [local_rank]，CPU 传 None。
        # process_group：默认使用上面创建的默认组；传入子组时需对应调整数据与训练语义。
        # bucket_cap_mb：控制梯度 bucket 容量，影响通信调用数量及计算通信重叠。
        # find_unused_parameters：是否遍历计算图识别未参与 loss 的参数；本模型全部参与，
        #   保持默认 False，不能用此开关修复任意显式 collective 不匹配。
        # buffer 运行时同步由 broadcast_buffers（较新版本为 forward_sync_buffers）等控制；
        #   本例沿用默认值，不代表每步广播参数，也不等于自动使用 SyncBatchNorm。
        # 其余可选参数按安装的 PyTorch 版本核对，不应为了性能盲目改动。
        model = DDP(network, device_ids=None if args.cpu else [local_rank])
        # 每个 rank 各自维护 AdamW 参数和动量状态；相同梯度还需要相同状态与超参数，
        # 才能得到一致更新。model.parameters() 返回被包装模型的参数引用。
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # step_size=10：每调用 scheduler.step() 10 次，学习率乘 gamma=0.9。
        # StepLR 常按 epoch 调用，但本例明确按成功参数更新调用，所以这里单位是更新次数。
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        # FP16 才启用 loss scaling；off / BF16 时此 scaler 透传相应操作。
        # 放大 loss 缓解下溢，与梯度累计中的 loss 权重归一化是两回事。
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp == 'fp16')
        # 本例分类目标是本地 batch 的样本平均 loss，后续按实际样本数加权累计。
        criterion = nn.CrossEntropyLoss(reduction='mean')
        dtype = torch.float16 if args.amp == 'fp16' else torch.bfloat16

        # 使用公开 optimizer hook，区分 scaler.step 被调用与真正完成参数更新。
        update_state = {'count': 0}
        def record_optimizer_step(optimizer, hook_args, hook_kwargs):
            update_state['count'] += 1
        hook = optimizer.register_step_post_hook(record_optimizer_step)
        micro_step = 0
        for epoch in range(args.epochs):
            # 切换训练模式，影响 Dropout / BatchNorm 等；不会自行执行 forward 或更新参数。
            model.train()
            sampler.set_epoch(epoch)  # 必须在 iter(loader) 前调用。
            # 首次创建迭代器时启动 worker，并开始预取；构造 DataLoader 不等于已完成加载。
            iterator = iter(loader)
            for start in range(0, len(loader), args.accumulation):
                window_size = min(args.accumulation, len(loader) - start)
                # 清理上个窗口的梯度；set_to_none 避免逐项填零，不能在窗口每次反向前都清理。
                optimizer.zero_grad(set_to_none=True)
                # 先取得本窗口 batch，以便按实际样本数加权，包括最后一个小 batch。
                # 最多额外持有一个累计窗口的 CPU 数据；生产代码也可由批次计划预知权重。
                batches, get_times = [], []
                for _ in range(window_size):
                    begin = time.perf_counter()
                    batches.append(next(iterator))
                    get_times.append(time.perf_counter() - begin)
                local_count = sum(target.numel() for _, target in batches)
                statistics = torch.zeros(2, device=device, dtype=torch.float64)
                for offset, ((inputs, targets), get_time) in enumerate(zip(batches, get_times)):
                    micro_step += 1
                    count = targets.numel()
                    # non_blocking 请求异步搬运；是否与计算重叠仍取决于内存、stream 和依赖。
                    inputs = inputs.to(device, non_blocking=not args.cpu)
                    targets = targets.to(device, non_blocking=not args.cpu)
                    # 前几次只在本地累计；最后一次正常 backward 同步整个窗口的累计贡献。
                    # no_sync 只关闭 DDP 梯度同步，不承诺关闭模型内部所有其他 collective。
                    context = nullcontext() if offset == window_size - 1 else model.no_sync()
                    # no_sync 必须覆盖 forward 与 backward。
                    with context:
                        with torch.autocast(device_type=device.type, dtype=dtype,
                                            enabled=args.amp != 'off'):
                            # model(inputs) 经 DDP 包装入口执行本地 forward；无需先归约 loss。
                            loss = criterion(model(inputs), targets)
                            # 按窗口实际样本数归一化，兼容尾部小 batch；等大 batch 时等价于除 K。
                            weighted_loss = loss * (count / local_count)
                        # backward 计算并累加 .grad；最后一个 micro-batch 还触发梯度归约。
                        # 一个累计窗口内保持相同 scale，不能中途 scaler.update 或 unscale。
                        scaler.scale(weighted_loss).backward()
                    # detach 只提取日志数值，不把统计逻辑接入 autograd 计算图。
                    statistics[0] += loss.detach().double() * count
                    statistics[1] += count
                    # 只用于理解取数等待；不强制 CUDA 同步，也不称作 GPU 执行耗时。
                    print(f'rank={rank} epoch={epoch} micro_step={micro_step} '
                          f'data_get_time={get_time:.6f}s', flush=True)

                # 各 rank 样本数相同是本示例 sampler/loader 配置保证的前提。
                # 最后一次 backward 同步的是本窗口全部累计贡献。
                # 还原梯度数值尺度后再裁剪；每个优化器每个更新窗口只 unscale 一次。
                scaler.unscale_(optimizer)
                # max_norm=1.0 是本示例裁剪阈值；裁剪同步后的完整梯度，而非各 rank 局部梯度。
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                before = update_state['count']
                scaler.step(optimizer)  # 非有限梯度时可能不执行 optimizer.step。
                # 根据非有限值检查结果调整下一窗口的 scale；不执行参数更新。
                scaler.update()
                if update_state['count'] > before:
                    scheduler.step()  # 本例按成功参数更新次数调度，各 rank 均执行。
                # 清理上个窗口的梯度；set_to_none 避免逐项填零，不能在窗口每次反向前都清理。
                optimizer.zero_grad(set_to_none=True)

                # 全部 rank 参与；这是日志统计的额外通信，不是 DDP 梯度同步。
                # 默认 op=SUM，原地累加 [loss 总和, 样本数]，两者相除得到全局样本平均。
                # 不能把这行放进 if rank == 0，否则其他 rank 未参与会造成通信不匹配。
                dist.all_reduce(statistics)
                if rank == 0:
                    mean_loss = (statistics[0] / statistics[1]).item()
                    print(f'update={update_state["count"]} loss={mean_loss:.6f} '
                          f'lr={optimizer.param_groups[0]["lr"]:.6g}', flush=True)

            if rank == 0:
                output = Path(args.output)
                output.mkdir(parents=True, exist_ok=True)
                # 同步更新边界保存；不是精确中途恢复的数据游标/RNG 完整快照。
                # model.module 取得原始模型，避免将 DDP 包装器的 module. 前缀写入权重键。
                # state_dict 只含状态，不是序列化整份模型对象；恢复时仍要创建相同模型结构。
                torch.save({'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(), 'epoch': epoch,
                            'update_step': update_state['count'], 'args': vars(args)},
                           output / '训练检查点.pt')
            dist.barrier()  # 保存边界协调，不能用 barrier 修复训练中的通信不匹配。
        hook.remove()
    finally:
        # 释放当前进程持有的进程组通信资源；各 rank 都需清理，不是只让 rank 0 执行。
        dist.destroy_process_group()


if __name__ == '__main__':
    main()  # spawn 模式下必须有入口保护。
