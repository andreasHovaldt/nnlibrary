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


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self, 
        weight: Tensor | None = None, 
        size_average=None, 
        ignore_index: int = -100, 
        reduce=None, 
        reduction: str = "mean", 
        label_smoothing: float = 0
    ) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # If batch_y is class indices (shape: [batch_size])
        if target.dim() == 1:
            target = target.long()
        # If batch_y is one-hot (shape: [batch_size, num_classes])
        elif target.dim() == 2:
            target = target.argmax(dim=1).long()
        else:
            raise ValueError(f"Target tensor must be 1D (class indices) or 2D (one-hot), got {target.dim()}D")

        return super().forward(input, target)