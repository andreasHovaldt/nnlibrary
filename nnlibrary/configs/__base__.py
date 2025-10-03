from dataclasses import dataclass, field, asdict
from typing import Any




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
    batch_size: int
    shuffle: bool = False
    num_workers: int | None = None
    pin_memory: bool | None = None