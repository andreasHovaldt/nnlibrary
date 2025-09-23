from dataclasses import dataclass, field
from typing import Any




@dataclass
class BaseConfig:
    name: str = ''
    args: dict = field(default_factory=dict)

@dataclass
class DataLoaderConfig:
    dataset: Any = None
    batch_size: int = 512
    shuffle: bool = True