"""DDP 初始化与工具函数。"""

import os
import torch
import torch.distributed as dist


def init_ddp():
    """从环境变量初始化 DDP（torchrun 启动时自动设置）。"""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """跨所有 rank 求均值，用于同步 loss/metric。"""
    if get_world_size() == 1:
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / get_world_size()


def wrap_ddp(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    """把模型包装为 DDP，同步 BN。"""
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


# 启动命令示例（写在注释里方便 agent 查阅）：
# torchrun --nproc_per_node=8 --nnodes=1 main.py [args]
