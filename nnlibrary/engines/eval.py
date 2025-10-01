import os
import functools

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import nnlibrary.utils.comm as comm



class EvaluatorBase:
    def __init__(self) -> None:
        self.dataloader: DataLoader
    
    def eval(self, model: nn.Module) -> dict:
        model.eval()
        self.before_evaluation()
        result = self.run_evaluation(model=model)
        self.after_evaluation()
        model.train()
        return result
        
    
    def before_evaluation(self):
        pass
    
    def run_evaluation(self, model: nn.Module) -> dict:
        raise NotImplementedError
    
    def after_evaluation(self):
        # Sync GPUs
        comm.synchronize()
        
    def _average_sample_accuracy(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        correct = (y_true == y_pred).sum().item()
        total = y_pred.size(dim=0) # first dim should be batch size
        return 100 * correct / total
    
    def _average_class_accuracy(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, torch.Tensor]:
        """Returns the average class accuracy and a tensor containing the individual class accuracies"""
        class_ids: torch.Tensor = y_true.unique(sorted=True)
        class_accuracies = []

        for class_id in class_ids:
            # Find all samples that truly belong to this class
            class_mask: torch.Tensor = (y_true == class_id)
            
            # Of those, how many did we predict correctly?
            correct: torch.types.Number = (y_pred[class_mask] == class_id).sum().item()
            total: torch.types.Number = class_mask.sum().item()
            
            class_acc: float = 100 * correct / total if total > 0 else 0
            class_accuracies.append(class_acc)
        
        return float(np.mean(class_accuracies)), torch.as_tensor(class_accuracies, dtype=torch.float32)
    
    def _confusion_matrix(self, y_true: torch.Tensor, y_pred: torch.Tensor, class_names: list[str]):
        
        # Build confusion matrix and per-class stats from raw predictions if present
        from sklearn.metrics import confusion_matrix
        
        num_classes = len(class_names)
        labels = list(range(num_classes))
        
        cm_norm = confusion_matrix(y_true.tolist(), y_pred.tolist(), labels=labels, normalize="true")

        # Create confusion matrix image
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create figure and draw heatmap on this axes
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix (Ground-truth-normalized)')
        fig.tight_layout()

        return fig, ax
    
    
    
    
class Evaluator(EvaluatorBase):
    def __init__(
        self, 
        dataloader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        amp_enable: bool,
        amp_dtype: torch.dtype,
        class_names: list[str],
        detailed: bool = False,
        
    ) -> None:
        super().__init__()
    
        self.dataloader = dataloader # self.trainer.build_dataloader(self.trainer.cfg.dataset.val)
        self.loss_fn = loss_fn
        self.device = device
        self.amp_enable = amp_enable
        self.amp_dtype = amp_dtype
        self.class_names = class_names
        self.detailed = detailed
        
    
    def run_evaluation(self, model: nn.Module):
        # Use AMP autocast for evaluation (same as training)
        autocast = functools.partial(torch.autocast, device_type=self.device)
        
        loss_eval_batch = 0.0
        eval_total = 0
        
        y_true: torch.Tensor = torch.as_tensor([])
        y_pred: torch.Tensor = torch.as_tensor([])
        
        with torch.no_grad():
            for batch_x, batch_y in self.dataloader:
                
                # TODO: Add better logging
                
                # Move to correct torch device
                if isinstance(batch_x, torch.Tensor): 
                    batch_x = batch_x.to(device=self.device, non_blocking=True)
                if isinstance(batch_y, torch.Tensor): 
                    batch_y = batch_y.to(device=self.device, non_blocking=True)
                
                # Forward pass with AMP
                with autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                    pred_y: torch.Tensor = model(batch_x)
                    pred_loss: torch.Tensor = self.loss_fn(pred_y, batch_y)
                
                
                # TODO: Accuracy might need to be more modular depending on the task
                
                # Calculate accuracy (supports index or one-hot targets)
                if isinstance(batch_y, torch.Tensor) and batch_y.dim() == 1:
                    target = batch_y.long()
                else:
                    target = batch_y.argmax(dim=-1).long()

                # Get argmax of last dim, usually the actual labels
                predicted = pred_y.argmax(dim=-1)
                
                # Add the batch size to the total validation samples
                eval_total += target.size(0)
                
                # Collect preds and labels for metric calc
                y_true = torch.cat((y_true, target.view(-1)))
                y_pred = torch.cat((y_pred, predicted.view(-1)))

                # Add the current batch loss
                loss_eval_batch += pred_loss.item()
                
        
        # Calculate metrics
        loss = loss_eval_batch / len(self.dataloader)
        avg_sample_accuracy = self._average_sample_accuracy(y_true, y_pred)
        avg_class_accuracy, class_accuracies = self._average_class_accuracy(y_true, y_pred)
        
        # TODO: Add this to logging
        # print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Create metric dict with results
        result: dict = dict(
            loss = loss,
            avg_sample_accuracy = avg_sample_accuracy,
            avg_class_accuracy = avg_class_accuracy,
        )
        
        # Add the individual class accuracies
        for cls, cls_acc in enumerate(class_accuracies):
            result[f"class_{self.class_names[cls]}_accuracy"] = cls_acc.item()
            
        
        # If detailed calculate the confusion matrix
        if (self.detailed and len(y_true) > 0):
            
            result.update({
                "confusion_matrix": self._confusion_matrix(y_true=y_true, y_pred=y_pred, class_names=self.class_names),
            })
        
        return result