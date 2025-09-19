import os
import torch

import numpy as np
import torch.nn as nn

from torch import Tensor
from pathlib import Path
from torch.utils.data import Dataset


class MultiTaskLoss(nn.Module):
    def __init__(self, reg_weight = 1.0, cls_weight = 1.0):
        super().__init__()
        self.regression_criterion = nn.SmoothL1Loss()
        self.classification_criterion = nn.CrossEntropyLoss()
        self.reg_weight = float(reg_weight)
        self.cls_weight = float(cls_weight)
    
    def forward(self, input: tuple[Tensor, Tensor], target: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        regression_loss = self.regression_criterion(input[0], target[0])
        classification_loss = self.classification_criterion(input[1], target[1])
        combined_weighted_loss = (regression_loss * self.reg_weight) + (classification_loss * self.cls_weight)
        
        return combined_weighted_loss, regression_loss, classification_loss




class MpcDataset(Dataset):
    def __init__(self, root_dir: os.PathLike, transform=None, target_transform=None):
        """
        Arguments:
            root_dir (PathLike): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.dataset_length = len(list(self.root_dir.iterdir()))
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # Load data
        data_dir = self.root_dir / f"{index:06d}"
        x = np.load(data_dir / "input.npy")
        y = np.load(data_dir / "output.npy")
        
        # Apply transforms
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
            
        # Convert to tensors and transfer to correct device
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        
        return x, y