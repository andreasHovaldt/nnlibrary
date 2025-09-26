import functools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import nnlibrary.utils.comm as comm



class ValidatorBase:
    def __init__(self) -> None:
        self.valloader: DataLoader
    
    def validate(self, model: nn.Module) -> dict:
        model.eval()
        self.before_validation()
        result = self.run_validation(model=model)
        self.after_validation()
        model.train()
        return result
        
    
    def before_validation(self):
        pass
    
    def run_validation(self, model: nn.Module) -> dict:
        raise NotImplementedError
    
    def after_validation(self):
        # Sync GPUs
        comm.synchronize()
    
    
    
class Validator(ValidatorBase):
    def __init__(
        self, 
        validation_loader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        amp_enable: bool,
        amp_dtype: torch.dtype,
        
    ) -> None:
        super().__init__()
    
        self.valloader = validation_loader # self.trainer.build_dataloader(self.trainer.cfg.dataset.val)
        self.loss_fn = loss_fn
        self.device = device
        self.amp_enable = amp_enable
        self.amp_dtype = amp_dtype
        
    
    def run_validation(self, model: nn.Module):
        # Use AMP autocast for validation (same as training)
        autocast = functools.partial(torch.autocast, device_type=self.device)
        
        loss_val_batch = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.valloader:
                
                # TODO: Add better logging
                
                # Move to correct torch device
                if isinstance(batch_x, torch.Tensor): 
                    batch_x = batch_x.to(device=self.device, non_blocking=True)
                if isinstance(batch_y, torch.Tensor): 
                    batch_y = batch_y.to(device=self.device, non_blocking=True)
                
                # Forward pass with AMP
                with autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                    pred_y: torch.Tensor = model(batch_x)
                    loss: torch.Tensor = self.loss_fn(pred_y, batch_y)
                
                
                # TODO: Accuracy might need to be more modular depending on the task
                
                # Calculate accuracy (supports index or one-hot targets)
                if isinstance(batch_y, torch.Tensor) and batch_y.dim() == 1:
                    target = batch_y.long()
                else:
                    target = batch_y.argmax(dim=-1).long()

                # Get argmax of last dim, usually the actual labels
                predicted = pred_y.argmax(dim=-1)
                
                # Add the batch size to the total validation samples
                val_total += target.size(0)
                
                # Count the amount of correct predictions
                val_correct += (predicted == target).sum().item()
                
                # Add the current batch loss
                loss_val_batch += loss.item()
        
        # Calculate metrics
        loss_val = loss_val_batch / len(self.valloader)
        val_accuracy = 100 * val_correct / val_total
        
        # TODO: Add this to logging
        # print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        return dict(
            loss_val = loss_val,
            val_accuracy = val_accuracy,
        )