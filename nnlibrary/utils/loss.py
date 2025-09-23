import torch.nn as nn
from torch import Tensor


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
