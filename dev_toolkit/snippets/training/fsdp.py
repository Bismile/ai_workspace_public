"""
FSDP（Fully Sharded Data Parallel）封装。
适合超大模型（>1B 参数），显存压力比 DDP 小很多。

启动命令：
    torchrun --nproc_per_node=8 train_fsdp.py

用法示例见文件末尾。
"""

import os
from functools import partial
from typing import Type

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp import StateDictType, FullStateDictConfig


# ---- 初始化 ----

def init_fsdp():
    """从环境变量初始化进程组（torchrun 自动设置）。"""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def cleanup():
    dist.destroy_process_group()


# ---- 混合精度配置 ----

def get_mixed_precision(dtype: str = "bf16") -> MixedPrecision:
    """
    dtype: "bf16" | "fp16" | "fp32"
    推荐 bf16（A100/H100），数值更稳定；V100 只支持 fp16。
    """
    t = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    return MixedPrecision(
        param_dtype=t,
        reduce_dtype=t,
        buffer_dtype=t,
    )


# ---- 包装模型 ----

def wrap_fsdp(
    model: nn.Module,
    transformer_layer_cls: Type[nn.Module] | None = None,
    min_params_to_wrap: int = 1e6,
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision_dtype: str = "bf16",
    cpu_offload: bool = False,
) -> FSDP:
    """
    Args:
        transformer_layer_cls: Transformer block 的类（如 GPTBlock、DiTBlock），
            优先按层分片。传 None 则按参数量自动分片。
        min_params_to_wrap: 自动分片的最小参数量阈值（仅 transformer_layer_cls=None 时生效）。
        sharding_strategy: "FULL_SHARD"（最省显存）| "SHARD_GRAD_OP"（更快）| "NO_SHARD"（等价DDP）
        cpu_offload: 把参数卸载到 CPU，极端省显存但很慢，一般不开。
    """
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }

    if transformer_layer_cls is not None:
        auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={transformer_layer_cls})
    else:
        auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=int(min_params_to_wrap))

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=get_mixed_precision(mixed_precision_dtype),
        sharding_strategy=strategy_map[sharding_strategy],
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,   # 兼容 torch.compile 和部分 optimizer
    )


# ---- Checkpoint（FSDP 保存需要特殊处理）----

def save_fsdp_checkpoint(model: FSDP, optimizer, save_path: str, rank: int):
    """
    在所有 rank 上调用，只有 rank 0 实际写文件。
    使用 FULL_STATE_DICT：把分片权重聚合到 rank 0 再保存，
    恢复时不需要相同的并行配置。
    """
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()

    if rank == 0:
        torch.save({"model": state, "optimizer": optimizer.state_dict()}, save_path)
        print(f"[FSDP] checkpoint saved → {save_path}")


def load_fsdp_checkpoint(model: FSDP, optimizer, load_path: str, device="cuda"):
    """加载 FULL_STATE_DICT 格式的 checkpoint。"""
    ckpt = torch.load(load_path, map_location=device)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])


# ---- 使用示例 ----
"""
if __name__ == "__main__":
    local_rank = init_fsdp()

    model = MyTransformer(...)
    model = wrap_fsdp(
        model,
        transformer_layer_cls=MyTransformerBlock,
        sharding_strategy="FULL_SHARD",
        mixed_precision_dtype="bf16",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step, batch in enumerate(loader):
        loss = model(batch)
        loss.backward()
        model.clip_grad_norm_(1.0)   # FSDP 提供的接口
        optimizer.step()
        optimizer.zero_grad()

        if step % 500 == 0:
            save_fsdp_checkpoint(model, optimizer, f"ckpt_step{step}.pt", rank=dist.get_rank())

    cleanup()
"""
