from dataclasses import dataclass, field, asdict
from typing import Any, Optional




@dataclass
class BaseConfig:
    name: str = ''
    args: dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BaseConfig':
        return cls(**data)
    
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class DataLoaderConfig:
    dataset: Any
    shuffle: bool = False
    batch_size: Optional[int] = None # NOTE: DO NOT SET THIS OPTION MANUALLY IF YOU WANT TO DO WANDB SWEEPS, USE THE CONFIG ATTRIBUTES 'train_batch_size' AND 'eval_batch_size'.
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None