import os
import functools

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import nnlibrary.utils.comm as comm
from nnlibrary.utils.operations import Standardize



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
    
    
    
    
class ClassificationEvaluator(EvaluatorBase):
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

        # Ensure model is on the correct device before evaluation
        if isinstance(model, nn.Module):
            try:
                param_device_type = next(model.parameters()).device.type
            except StopIteration:
                param_device_type = self.device  # model with no parameters
            if param_device_type != self.device:
                model.to(self.device)
        
        loss_eval_batch = 0.0

        # Accumulate predictions and targets (class indices) on CPU
        y_true_list: list[torch.Tensor] = []
        y_pred_list: list[torch.Tensor] = []

        with torch.inference_mode():
            for batch_x, batch_y in self.dataloader:
                # Move to device
                if isinstance(batch_x, torch.Tensor):
                    batch_x = batch_x.to(device=self.device, non_blocking=True)
                if isinstance(batch_y, torch.Tensor):
                    batch_y = batch_y.to(device=self.device, non_blocking=True)

                # Forward pass
                with autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                    logits: torch.Tensor = model(batch_x)
                    pred_loss: torch.Tensor = self.loss_fn(logits, batch_y)

                # Targets as class indices
                if isinstance(batch_y, torch.Tensor) and batch_y.dim() == 1:
                    target = batch_y.long()
                else:
                    target = batch_y.argmax(dim=-1).long()

                # Predicted class indices, argmax of last dim should usually represent the class pred
                predicted = logits.argmax(dim=-1)

                # Append flattened to CPU lists
                y_true_list.append(target.detach().to('cpu').view(-1))
                y_pred_list.append(predicted.detach().to('cpu').view(-1))

                # Add loss of current batch
                loss_eval_batch += float(pred_loss.item())

        # Concatenate once
        y_true = torch.cat(y_true_list, dim=0) if y_true_list else torch.empty(0, dtype=torch.long)
        y_pred = torch.cat(y_pred_list, dim=0) if y_pred_list else torch.empty(0, dtype=torch.long)

        # Metrics
        num_batches = max(len(self.dataloader), 1)
        loss = loss_eval_batch / num_batches
        avg_sample_accuracy = self._average_sample_accuracy(y_true, y_pred) if y_true.numel() > 0 else float('nan')
        avg_class_accuracy, class_accuracies = self._average_class_accuracy(y_true, y_pred) if y_true.numel() > 0 else (float('nan'), torch.tensor([]))

        result: dict = dict(
            loss=loss,
            avg_sample_accuracy=avg_sample_accuracy,
            avg_class_accuracy=avg_class_accuracy,
        )

        # Per-class accuracies
        if y_true.numel() > 0 and len(self.class_names) > 0:
            for cls, cls_acc in enumerate(class_accuracies):
                name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
                result[f"class_{name}_accuracy"] = float(cls_acc.item())

        # Detailed outputs: confusion matrix and sequences
        if self.detailed and y_true.numel() > 0:
            result.update({
                "confusion_matrix": self._confusion_matrix(y_true=y_true, y_pred=y_pred, class_names=self.class_names),
                "y_true_seq": y_true,
                "y_pred_seq": y_pred,
            })

        return result
    
    
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


from typing import Callable, Optional

class RegressionEvaluator(EvaluatorBase):
    def __init__(
        self, 
        dataloader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        amp_enable: bool,
        amp_dtype: torch.dtype,
        output_names: list[str] | None = None,  # Names for each output dimension
        detailed: bool = False,
        inverse_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        
    ) -> None:
        super().__init__()
    
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.amp_enable = amp_enable
        self.amp_dtype = amp_dtype
        self.output_names = output_names
        self.detailed = detailed
        self.inverse_transform = inverse_transform
        
    
    def run_evaluation(self, model: nn.Module):
        # Use AMP autocast for evaluation (same as training)
        autocast = functools.partial(torch.autocast, device_type=self.device)
        
        # Ensure model is on the correct device before evaluation
        if isinstance(model, nn.Module):
            try:
                param_device_type = next(model.parameters()).device.type
            except StopIteration:
                param_device_type = self.device  # model with no parameters
            if param_device_type != self.device:
                model.to(self.device)
        
        loss_eval_batch = 0.0
        
        # Initialize CPU-side containers for metrics
        y_true_list: list[torch.Tensor] = []
        y_pred_list: list[torch.Tensor] = []
        
        # inference_mode is faster than no_grad for eval-only paths
        with torch.inference_mode():
            for batch_x, batch_y in self.dataloader:
                
                # Move to correct torch device
                if isinstance(batch_x, torch.Tensor): 
                    batch_x = batch_x.to(device=self.device, non_blocking=True)
                if isinstance(batch_y, torch.Tensor): 
                    batch_y = batch_y.to(device=self.device, non_blocking=True)
                
                # Forward pass with AMP
                with autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                    pred_y: torch.Tensor = model(batch_x)
                    pred_loss: torch.Tensor = self.loss_fn(pred_y, batch_y)
                
                # Collect preds and labels for metric calculation
                y_true_list.append(batch_y.detach().to('cpu'))
                y_pred_list.append(pred_y.detach().to('cpu'))

                # Add the current batch loss
                loss_eval_batch += pred_loss.item()
                
        
        # Concatenate metrics once
        y_true = torch.cat(y_true_list, dim=0) if y_true_list else torch.empty(0, dtype=torch.float32)
        y_pred = torch.cat(y_pred_list, dim=0) if y_pred_list else torch.empty(0, dtype=torch.float32)
        
        # # Inverse transform if provided - Removed again due to it making the aggregated metrics unfairly distributed, again features with higher values will affect MAE and RMSE more than others
        # if self.inverse_transform is not None and y_true.numel() > 0:
        #     y_true = self.inverse_transform(y_true)
        #     y_pred = self.inverse_transform(y_pred)

        # Calculate metrics on original scale
        num_batches = max(len(self.dataloader), 1)
        loss = loss_eval_batch / num_batches
        mae = self._mean_absolute_error(y_true, y_pred)
        rmse = self._root_mean_squared_error(y_true, y_pred)
        r2 = self._r2_score(y_true, y_pred)
        
        # Per-output metrics
        per_output_mae = self._per_output_mae(y_true, y_pred)
        per_output_rmse = self._per_output_rmse(y_true, y_pred)
        
        # Create metric dict with results
        result: dict = dict(
            loss=loss,
            mae=mae,
            rmse=rmse,
            r2_score=r2,
        )
        
        # Add per-output metrics with names if available
        if self.output_names is not None and len(per_output_mae) == len(self.output_names):
            for i, name in enumerate(self.output_names):
                result[f"{name}_mae"] = per_output_mae[i].item()
                result[f"{name}_rmse"] = per_output_rmse[i].item()
        else:
            for i in range(len(per_output_mae)):
                result[f"output_{i}_mae"] = per_output_mae[i].item()
                result[f"output_{i}_rmse"] = per_output_rmse[i].item()
        
        # If detailed, return raw sequences and create plots
        if self.detailed and len(y_true) > 0:
            result.update({
                "y_true_seq": y_true,
                "y_pred_seq": y_pred,
                "prediction_plots": self._create_prediction_plots(y_true, y_pred, draw_quantiles = False, draw_standard_deviation = True),
            })
        
        return result

    def _mean_absolute_error(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Calculate MAE across all outputs"""
        return torch.mean(torch.abs(y_true - y_pred)).item()
    
    def _root_mean_squared_error(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Calculate RMSE across all outputs"""
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    
    def _r2_score(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Calculate R² score across all outputs"""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        
        ss_tot_val = float(ss_tot.item()) if isinstance(ss_tot, torch.Tensor) else float(ss_tot)
        if ss_tot_val == 0.0:
            return 0.0
        
        r2 = 1.0 - float(ss_res.item()) / ss_tot_val
        return r2
    
    def _per_output_mae(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate MAE for each output dimension separately"""
        if y_true.dim() == 1:
            return torch.tensor([torch.mean(torch.abs(y_true - y_pred)).item()])
        return torch.mean(torch.abs(y_true - y_pred), dim=0)
    
    def _per_output_rmse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate RMSE for each output dimension separately"""
        if y_true.dim() == 1:
            return torch.tensor([torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()])
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=0))
    
    def _create_prediction_plots(self, y_true: torch.Tensor, y_pred: torch.Tensor, draw_quantiles = False, draw_standard_deviation = False):
        """Create scatter plots comparing predictions vs ground truth"""
        import matplotlib.pyplot as plt
        
        # Handle both single and multi-output cases
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)
        
        # Inverse transform if provided
        if self.inverse_transform is not None and y_true.numel() > 0:
            y_true = self.inverse_transform(y_true)
            y_pred = self.inverse_transform(y_pred)
        
        num_outputs = y_true.shape[1]
        output_names = self.output_names if self.output_names else [f"Output {i}" for i in range(num_outputs)]
        
        # Create subplots for each output
        fig, axes = plt.subplots(1, num_outputs, figsize=(6 * num_outputs, 5))
        if num_outputs == 1:
            axes = [axes]
        
        for i, (ax_i, name) in enumerate(zip(axes, output_names)):
            y_true_i = y_true[:, i].numpy()
            y_pred_i = y_pred[:, i].numpy()
            
            # Scatter plot
            ax_i.scatter(y_true_i, y_pred_i, alpha=0.5, s=10)
            
            # Perfect prediction line
            min_val = min(y_true_i.min(), y_pred_i.min())
            max_val = max(y_true_i.max(), y_pred_i.max())
            ax_i.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

            # Symmetric error-quantile bands around y = x
            # Compute absolute residual quantiles so that:
            #  - 25% of points lie within ±q25
            #  - 50% of points lie within ±q50
            #  - 75% of points lie within ±q75
            if draw_quantiles:
                abs_err = np.abs(y_pred_i - y_true_i)
                q25, q50, q75 = np.quantile(abs_err, [0.25, 0.50, 0.75])

                # Helper to draw a band defined by offset q around y=x
                def draw_band(q: float, color: str, label: str | None = None, lw: float = 1.5):
                    # Upper: y = x + q, Lower: y = x - q
                    ax_i.plot([min_val, max_val], [min_val + q, max_val + q], color=color, linestyle='-', linewidth=lw, alpha=0.85, label=label)
                    ax_i.plot([min_val, max_val], [min_val - q, max_val - q], color=color, linestyle='-', linewidth=lw, alpha=0.85)

                # Draw from inner to outer so outer lines don't hide inner ones
                # Colors: inner=green, mid=orange, outer=purple (distinct from red perfect line)
                draw_band(q25, color='#27ae60', label='±q25 (25%)')
                draw_band(q50, color='#f39c12', label='±q50 (50%)')
                draw_band(q75, color='#8e44ad', label='±q75 (75%)')
            elif draw_standard_deviation:
                # Draw standard deviation bands based on residuals r = y_pred - y_true
                residuals = y_pred_i - y_true_i
                std = residuals.std()

                if std > 0:
                    def draw_sigma(k: float, color: str, label: str | None = None, lw: float = 1.5):
                        off = k * std
                        # Upper: y = x + k*std, Lower: y = x - k*std
                        ax_i.plot([min_val, max_val], [min_val + off, max_val + off], color=color, linestyle='-', linewidth=lw, alpha=0.85, label=label)
                        ax_i.plot([min_val, max_val], [min_val - off, max_val - off], color=color, linestyle='-', linewidth=lw, alpha=0.85)

                    # Use distinct colors; annotate approximate Gaussian coverage for intuition
                    draw_sigma(1.0, color='#3498db', label='±1σ (~68%)')
                    draw_sigma(2.0, color='#e67e22', label='±2σ (~95%)')
                    draw_sigma(3.0, color='#c0392b', label='±3σ (~99.7%)')
                else:
                    # If std is zero, all points are on the line; no bands to draw
                    pass
            
            ax_i.set_xlabel(f'True {name}')
            ax_i.set_ylabel(f'Predicted {name}')
            ax_i.set_title(f'{name}: Predictions vs Ground Truth')
            ax_i.legend()
            ax_i.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig, axes