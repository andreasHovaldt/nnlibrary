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
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DataLoaderConfig':
        return cls(**data)
    
    def to_dict(self) -> dict:
        return asdict(self)