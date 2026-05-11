"""
DeepSpeed 封装。
适合超大模型训练，ZeRO-2/3 大幅减少显存，支持 CPU/NVMe offload。

依赖：pip install deepspeed

启动命令：
    deepspeed train_ds.py --deepspeed ds_config.json
    # 或者
    torchrun --nproc_per_node=8 train_ds.py   # 不用 ds_config 时

用法示例见文件末尾。
"""

import json
from pathlib import Path
from typing import Literal


# ---- 配置生成 ----

def make_ds_config(
    stage: Literal[1, 2, 3] = 2,
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16",
    offload_optimizer: bool = False,
    offload_param: bool = False,       # 仅 stage 3 有效
    grad_accum_steps: int = 1,
    train_batch_size: int | None = None,   # 设置后 DeepSpeed 自动算 micro batch
    train_micro_batch_per_gpu: int = 8,
    zero_allgather_bucket_size: int = 5e8,
    zero_reduce_bucket_size: int = 5e8,
) -> dict:
    """
    返回 DeepSpeed config dict，可直接传给 deepspeed.initialize()
    或写成 json 文件用命令行指定。

    ZeRO 阶段说明：
        stage=1  — 分片 optimizer state（最低要求，提升有限）
        stage=2  — 分片 optimizer state + gradient（推荐，大多数场景够用）
        stage=3  — 分片 optimizer + gradient + parameter（最省显存，但通信开销更大）
    """
    assert not (offload_param and stage < 3), "offload_param 仅 stage=3 支持"

    zero_config: dict = {"stage": stage}

    if stage >= 2:
        zero_config["allgather_partitions"] = True
        zero_config["allgather_bucket_size"] = int(zero_allgather_bucket_size)
        zero_config["reduce_scatter"] = True
        zero_config["reduce_bucket_size"] = int(zero_reduce_bucket_size)
        zero_config["overlap_comm"] = True
        zero_config["contiguous_gradients"] = True

    if offload_optimizer:
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    if stage == 3 and offload_param:
        zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}

    config = {
        "zero_optimization": zero_config,
        "gradient_accumulation_steps": grad_accum_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }

    if train_batch_size is not None:
        config["train_batch_size"] = train_batch_size
    config["train_micro_batch_size_per_gpu"] = train_micro_batch_per_gpu

    if dtype == "bf16":
        config["bf16"] = {"enabled": True}
    elif dtype == "fp16":
        config["fp16"] = {"enabled": True, "loss_scale": 0, "loss_scale_window": 1000,
                          "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1}

    return config


def save_ds_config(config: dict, path: str = "ds_config.json"):
    Path(path).write_text(json.dumps(config, indent=2))
    print(f"[DeepSpeed] config saved → {path}")


# ---- 初始化 ----

def init_deepspeed(model, optimizer, lr_scheduler, config: dict | str, args=None):
    """
    包装模型为 DeepSpeed engine。

    Args:
        config: dict 或 json 文件路径
        args:   命令行 args（含 --local_rank），可传 None

    Returns:
        engine, optimizer, _, lr_scheduler
        （engine 替代 model 使用，optimizer/lr_scheduler 由 DS 接管）
    """
    import deepspeed

    if isinstance(config, dict):
        ds_config = config
    else:
        with open(config) as f:
            ds_config = json.load(f)

    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
        args=args,
    )
    return engine, optimizer, lr_scheduler


# ---- Checkpoint ----

def save_ds_checkpoint(engine, save_dir: str, tag: str = "latest"):
    """engine.save_checkpoint 会自动保存 ZeRO 分片状态。"""
    engine.save_checkpoint(save_dir, tag=tag)
    print(f"[DeepSpeed] checkpoint saved → {save_dir}/{tag}")


def load_ds_checkpoint(engine, load_dir: str, tag: str = "latest"):
    _, client_state = engine.load_checkpoint(load_dir, tag=tag)
    return client_state   # 用户自定义状态（step、epoch 等）


# ---- 使用示例 ----
"""
import deepspeed

config = make_ds_config(stage=2, dtype="bf16", offload_optimizer=False)

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = build_scheduler(optimizer, warmup_steps=500, total_steps=10000)

engine, optimizer, scheduler = init_deepspeed(model, optimizer, scheduler, config)

for step, batch in enumerate(loader):
    loss = engine(batch)
    engine.backward(loss)       # 不用 loss.backward()
    engine.step()               # 不用 optimizer.step() / zero_grad()

    if step % 1000 == 0:
        save_ds_checkpoint(engine, "checkpoints/", tag=f"step{step}")
"""
