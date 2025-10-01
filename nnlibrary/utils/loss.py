import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, Union, List



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
        # If batch_y is class indices (shape: [batch_size,])
        if target.dim() == 1:
            target = target.long()
        # If batch_y is one-hot (shape: [batch_size, num_classes])
        elif target.dim() == 2:
            target = target.argmax(dim=1).long()
        else:
            raise ValueError(f"Target tensor must be 1D (class indices) or 2D (one-hot), got {target.dim()}D")

        return super().forward(input, target)



class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    
    Shamelessly taken and modified from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/e1dd0e84e897d9883cf9fe3ccae52b30dd1d4d27/focal_loss.py

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Union[Tensor, List[float]]] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper. If gamma is 0.0, it is essentially the same as CrossEntropyLoss. 
                Defaults to 2.0
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        
        if isinstance(alpha, List):
            self.alpha = torch.as_tensor(alpha)
        else:
            self.alpha = alpha
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=self.alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        # Ensure targets are integer class indices for loss/indexing
        if y.dtype != torch.long:
            y = y.long()

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return x.new_zeros(())
        x = x[unignored_mask]

        # Ensure NLLLoss weights (alpha) are on the same device as inputs
        if getattr(self.nll_loss, 'weight', None) is not None and self.nll_loss.weight is not None:
            if self.nll_loss.weight.device != x.device:
                # Move the loss module (and its weight buffer) to the correct device
                self.nll_loss = self.nll_loss.to(x.device)

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(x.shape[0], device=x.device)
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp().clamp_min(1e-12)
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss: Tensor = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss