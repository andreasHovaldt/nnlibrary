import functools

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
    
    
    
class Evaluator(EvaluatorBase):
    def __init__(
        self, 
        dataloader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        amp_enable: bool,
        amp_dtype: torch.dtype,
        detailed: bool = False,
        
    ) -> None:
        super().__init__()
    
        self.dataloader = dataloader # self.trainer.build_dataloader(self.trainer.cfg.dataset.val)
        self.loss_fn = loss_fn
        self.device = device
        self.amp_enable = amp_enable
        self.amp_dtype = amp_dtype
        self.detailed = detailed
        
    
    def run_evaluation(self, model: nn.Module):
        # Use AMP autocast for evaluation (same as training)
        autocast = functools.partial(torch.autocast, device_type=self.device)
        
        loss_eval_batch = 0.0
        eval_correct = 0
        eval_total = 0
        
        y_true: list[int] = []
        y_pred: list[int] = []
        
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
                
                # Count the amount of correct predictions
                eval_correct += (predicted == target).sum().item()
                
                # Detailed stats: collect true/pred labels for confusion matrix
                if self.detailed:
                    y_true.extend(target.view(-1).tolist())
                    y_pred.extend(predicted.view(-1).tolist())

                # Add the current batch loss
                loss_eval_batch += pred_loss.item()
        
        # Calculate metrics
        loss = loss_eval_batch / len(self.dataloader)
        accuracy = 100 * eval_correct / eval_total
        
        # TODO: Add this to logging
        # print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        result: dict = dict(
            loss = loss,
            accuracy = accuracy,
        )
        
        if self.detailed and len(y_true) > 0:
            # Defer heavy computations (confusion matrix, per-class stats) to the caller.
            result.update({
                "y_true": y_true,
                "y_pred": y_pred,
            })
        
        return result