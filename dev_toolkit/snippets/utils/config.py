"""dataclass Config + argparse 自动绑定。

用法：
    @dataclass
    class Config(BaseConfig):
        lr: float = 1e-4
        num_epochs: int = 100

    cfg = Config.from_args()
"""

import argparse
from dataclasses import dataclass, fields, asdict
from pathlib import Path
import yaml


@dataclass
class BaseConfig:

    @classmethod
    def from_args(cls):
        """从命令行参数构造 Config，所有字段自动暴露为 --xxx 参数。"""
        parser = argparse.ArgumentParser()
        inst = cls()
        for f in fields(cls):
            val = getattr(inst, f.name)
            t = type(val) if val is not None else str
            parser.add_argument(f"--{f.name}", type=t, default=val,
                                help=f"default: {val}")
        args = parser.parse_args()
        return cls(**vars(args))

    @classmethod
    def from_yaml(cls, path: str):
        """从 yaml 文件加载，命令行参数可覆盖。"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, allow_unicode=True)

    def __repr__(self):
        lines = [f"{f.name}: {getattr(self, f.name)}" for f in fields(self)]
        return "\n".join(lines)
