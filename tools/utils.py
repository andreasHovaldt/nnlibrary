import os
import torch

import numpy as np
import torch.nn as nn

from torch import Tensor
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
    def __init__(self, root_dir: os.PathLike, transform=None, device='cpu'):
        """
        Arguments:
            root_dir (PathLike): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        
        data_dict: dict[str, np.ndarray] = np.load(self.root_dir, allow_pickle=True)
        ahu_states_measurements, ahu_actuators_measurements, ahu_actuator_commands, ahu_mode_commands, _ = data_dict.values()
        
        self.mpc_input = np.concatenate([ahu_states_measurements, ahu_actuators_measurements], axis=1).astype(np.float32)
        self.mpc_command_output = ahu_actuator_commands.astype(np.float32)
        self.mpc_mode_output = ahu_mode_commands.astype(np.float32)
        assert self.mpc_input.shape[0] == self.mpc_command_output.shape[0] == self.mpc_mode_output.shape[0] 
        
    def __len__(self):
        return self.mpc_input.shape[0]
    
    def __getitem__(self, index) -> Tensor:
        
        
        
        return torch.tensor([])