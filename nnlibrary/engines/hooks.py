import os
import time
import shutil
import numpy as np
import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
import wandb
import torch
from nnlibrary.engines.eval import ClassificationEvaluator, RegressionEvaluator
import nnlibrary.utils.comm as comm
from nnlibrary.utils.operations import Standardize, MinMaxNormalize

if TYPE_CHECKING:
    from .train import TrainerBase, Trainer


class Hookbase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    
    NOTE: self.trainer cannot be accessed duing hook __init__
    
    To avoid circular reference, hooks and trainer cannot own each other.
    This normally does not matter, but will cause memory leak if the
    involved objects contain __del__:
    See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
    """
    
    # if TYPE_CHECKING:
    #     trainer: TrainerBase | Trainer  # A weak reference to the trainer object.
    trainer = None
    
    def before_train(self):
        pass
    
    def before_epoch(self):
        pass
    
    def before_step(self):
        pass
    
    def after_step(self):
        pass
    
    def after_epoch(self):
        pass
    
    def after_train(self):
        pass


class WandbHook(Hookbase):
    def __init__(self) -> None:
        self.current_iter = 0
    
    def before_train(self):
        if self.trainer and self.trainer.wandb_run:
            self.trainer.wandb_run.define_metric(name="params/*", step_metric="Iter")
            self.trainer.wandb_run.define_metric(name="train_batch/*", step_metric="Iter")
            self.trainer.wandb_run.define_metric(name="train/*", step_metric="Epoch")
            
            if self.trainer.cfg.validate_model:
                self.trainer.wandb_run.define_metric(name="validation/*", step_metric="Epoch")
            
        
    def before_step(self):
        self.current_iter += 1
    
    def after_step(self):
        if self.trainer and self.trainer.wandb_run:
            current_lr = self.trainer.info["current_lr"][0]
            loss_iter = self.trainer.info["loss_iter"]
            
            self.trainer.wandb_run.log(
                data={
                    "Iter": self.current_iter,
                    "params/lr": current_lr,
                    "train_batch/loss": loss_iter,
                }, 
                step=self.current_iter
            )
            
    def after_epoch(self):
        if self.trainer and self.trainer.wandb_run:
            epoch = self.trainer.info["epoch"]
            epoch_loss = self.trainer.info["loss_epoch_avg"]
            # print("after_epoch: ", epoch, epoch_loss)
            
            self.trainer.wandb_run.log(
                data={
                    "Epoch": epoch+1,
                    "train/loss": epoch_loss,
                },
                step=self.trainer.wandb_run.step,
            )
            
            # Log validation metric if available
            if "validation_result" in self.trainer.info.keys():
                val_result: dict = self.trainer.info["validation_result"]
                if isinstance(val_result, dict): # Check if dict so pylance doesn't cry
                    for key, value in val_result.items():
                        
                        # Specific logging cases
                        if key == "confusion_matrix":
                            fig, ax = value
                            self.trainer.wandb_run.log(
                                data={
                                    f"validation/{key}": wandb.Image(fig)
                                },
                                step=self.trainer.wandb_run.step,
                            )
                        
                        # Dump the remaining test metrics
                        else:
                            try:
                                self.trainer.wandb_run.log(
                                    data={
                                        f"validation/{key}": value,
                                    },
                                    step=self.trainer.wandb_run.step,
                                )
                            except:
                                print(f"WARN: wandb.log failed for key '{key}' with type '{type(value)}'")
                                
    
    def after_train(self):
        if self.trainer and self.trainer.wandb_run:
            
            # Logging test results
            
            if "test_result" in self.trainer.info.keys():
                test_result: dict = self.trainer.info["test_result"]
                if isinstance(test_result, dict):
                    
                    for key, value in test_result.items():
                        
                        # Specific logging cases
                        if key in ("confusion_matrix", "prediction_plots"):
                            fig, ax = value
                            self.trainer.wandb_run.log(
                                data={
                                    f"test/{key}": wandb.Image(fig)
                                }
                            )
                        
                        elif key in ('y_true', 'y_pred'):
                            continue
                        
                        # Dump the remaining test metrics
                        else:
                            try:
                                self.trainer.wandb_run.log(
                                    data={
                                        f"test/{key}": value,
                                    },
                                    step=self.trainer.wandb_run.step,
                                )
                            except:
                                print(f"WARN: wandb.log failed for key '{key}' with type '{type(value)}'")
                
    
    
class TensorBoardHook(Hookbase):
    def __init__(self) -> None:
        self.current_iter = 0
        
    def before_step(self):
        self.current_iter += 1
    
    def after_step(self):
        if self.trainer and self.trainer.tensorboard_writer:
            current_lr = self.trainer.info["current_lr"][0]
            loss_iter = self.trainer.info["loss_iter"]
            
            self.trainer.tensorboard_writer.add_scalar("progress/Iter", self.current_iter, self.current_iter)
            self.trainer.tensorboard_writer.add_scalar("params/lr", current_lr, self.current_iter)
            self.trainer.tensorboard_writer.add_scalar("train_batch/loss", loss_iter, self.current_iter)
            
    def after_epoch(self):
        if self.trainer and self.trainer.tensorboard_writer:
            epoch = self.trainer.info["epoch"]
            epoch_loss = self.trainer.info["loss_epoch_avg"]
            
            self.trainer.tensorboard_writer.add_scalar("progress/Epoch", epoch, self.current_iter)
            self.trainer.tensorboard_writer.add_scalar("train/loss", epoch_loss, epoch)
            
            # Log validation metric if available
            if "validation_result" in self.trainer.info.keys():
                val_result: dict = self.trainer.info["validation_result"]
                if isinstance(val_result, dict): # Check if dict so pylance doesn't cry
                    for key, value in val_result.items():
                        # TODO: Implement plot logging for tensorboard
                        if key == "confusion_matrix" or "prediction_plots":
                            continue
                        # Only log scalars to TensorBoard
                        try:
                            if isinstance(value, (int, float)):
                                self.trainer.tensorboard_writer.add_scalar(f"validation/{key}", value, epoch)
                            else: print(f"WARN: '{key}' is not an int or float and was not logged to tensorboard, type: {type(value)}")
                        except Exception as e:
                            print(f"WARN: Encountered an exception when logging to tensorboard: '{e}'")
        

class ValidationHook(Hookbase):
    def __init__(self) -> None:
        self.validator = None
            
    def after_epoch(self):
        
        if self.trainer and self.trainer.cfg.validate_model:
            
            if self.validator is None:
                validation_loader = self.trainer.build_dataloader(self.trainer.cfg.dataset.val)
                
                if self.trainer.cfg.task == "classification":
                    self.validator = ClassificationEvaluator(
                        dataloader=validation_loader,
                        loss_fn=self.trainer.loss_fn,
                        device=self.trainer.device,
                        amp_enable=self.trainer.amp_enable,
                        amp_dtype=self.trainer.amp_dtype,
                        class_names=self.trainer.cfg.dataset.info["class_names"],
                        detailed=self.trainer.cfg.validation_plot,
                    )
                elif self.trainer.cfg.task == "regression":
                    
                    # Provide inverse-transform for calculating on original value ranges if available
                    inv_transform = None
                    transform = getattr(self.trainer, 'target_transform', None)
                    if transform is not None and hasattr(transform, 'inverse_transform'):
                        inv_transform = transform.inverse_transform
                    
                    self.validator = RegressionEvaluator(
                        dataloader=validation_loader,
                        loss_fn=self.trainer.loss_fn,
                        device=self.trainer.device,
                        amp_enable=self.trainer.amp_enable,
                        amp_dtype=self.trainer.amp_dtype,
                        output_names=self.trainer.cfg.dataset.info["class_names"],
                        detailed=self.trainer.cfg.validation_plot,
                        inverse_transform=inv_transform,
                    )
                else: 
                    raise ValueError(f"Unknown task '{self.trainer.cfg.task}' for ValidationHook")
        
            # Time the validation run (wall clock), sync CUDA if applicable
            is_cuda = (self.trainer.device == 'cuda' and torch.cuda.is_available())
            
            timing: bool = self.trainer.cfg.timing
            if timing:
            
                if is_cuda:
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()
                
                result = self.validator.eval(model=self.trainer.model)
                
                if is_cuda:
                    torch.cuda.synchronize()
                
                val_wall_s = time.perf_counter() - _t0
            
            else:
                if is_cuda:
                    torch.cuda.synchronize()
                
                result = self.validator.eval(model=self.trainer.model)
                
                if is_cuda:
                    torch.cuda.synchronize()
                
            
            self.trainer.info["validation_result"] = result
            
            # Print and log validation timing
            if comm.is_main_process() and timing:
                # print(f"[Timing] Validation wall time: {val_wall_s:.2f}s")
                if self.trainer.wandb_run:
                    try:
                        self.trainer.wandb_run.log({
                            'timing/val_wall_s': val_wall_s,
                            'Epoch': self.trainer.info.get('epoch', 0) + 1,
                        }, step=getattr(self.trainer.wandb_run, 'step', None))
                    except Exception:
                        pass
                if self.trainer.tensorboard_writer:
                    try:
                        self.trainer.tensorboard_writer.add_scalar('timing/val_wall_s', val_wall_s, self.trainer.info.get('epoch', 0) + 1)
                    except Exception:
                        pass
        return None


class TestHook(Hookbase):
    def __init__(self, checkpoint_name: str = 'best') -> None:
        self.tester = None
        self.checkpoint_name = checkpoint_name
    
    def after_train(self):
        
        if self.trainer and self.trainer.cfg.test_model:
            
            if self.tester is None:
                test_loader = self.trainer.build_dataloader(self.trainer.cfg.dataset.test)
                
                if self.trainer.cfg.task == "classification":
                    self.tester = ClassificationEvaluator(
                        dataloader=test_loader,
                        loss_fn=self.trainer.loss_fn,
                        device=self.trainer.device,
                        amp_enable=self.trainer.amp_enable,
                        amp_dtype=self.trainer.amp_dtype,
                        class_names=self.trainer.cfg.dataset.info["class_names"],
                        detailed=True,
                    )
                elif self.trainer.cfg.task == "regression":
                    
                    # Provide inverse-transform for calculating on original value ranges if available
                    inv_transform = None
                    transform: Optional[Union[Standardize, MinMaxNormalize]] = getattr(self.trainer, 'target_transform', None)
                    if transform is not None and hasattr(transform, 'inverse_transform'):
                        inv_transform = transform.inverse_transform
                    
                    self.tester = RegressionEvaluator(
                        dataloader=test_loader,
                        loss_fn=self.trainer.loss_fn,
                        device=self.trainer.device,
                        amp_enable=self.trainer.amp_enable,
                        amp_dtype=self.trainer.amp_dtype,
                        output_names=self.trainer.cfg.dataset.info["class_names"],
                        detailed=True,
                        inverse_transform=inv_transform,
                    )
                else:
                    raise ValueError(f"Unknown task '{self.trainer.cfg.task}' for TestHook")

            # Load in the desired checkpointed model
            checkpoint_path = Path(self.trainer.save_path / "model" / f"model_{self.checkpoint_name}.pth")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            checkpoint_model: torch.nn.Module = self.trainer.build_model(self.trainer.cfg.model_config)
            checkpoint_model.load_state_dict(checkpoint["state_dict"])
            
            # Perform the test inference
            result = self.tester.eval(
                model=checkpoint_model
            )
            self.trainer.info["test_result"] = result
            
            print(f"Final test result:")
            for key, value in result.items():
                if key in ("confusion_matrix", "prediction_plots", "y_true", "y_pred"):
                    continue
                try:
                    print(f"   {key}: {float(value):.4f}")
                except Exception:
                    print(f"WARN: test result print skipped for key '{key}' (type {type(value)})")

            figure_dir = Path(self.trainer.save_path / "figures")
            figure_dir.mkdir(parents=True, exist_ok=True)

            # Save classification confusion matrix if present
            if "confusion_matrix" in result:
                try:
                    fig, ax = result["confusion_matrix"]
                    fig.savefig(figure_dir / "confusion_matrix_test_split.png")
                except Exception:
                    pass

            # Save regression prediction plots if present
            if "prediction_plots" in result:
                try:
                    fig, axes = result["prediction_plots"]
                    fig.savefig(figure_dir / "pred_vs_true_test.png")
                except Exception:
                    pass
                
        return None
  

class CheckpointerHook(Hookbase):
    def __init__(self) -> None:
        self.best_metric_value_high: float = 0.0
        self.best_metric_value_low: float = np.inf
        
    def after_epoch(self):
        if self.trainer is not None and comm.is_main_process():
            save_dir = Path(self.trainer.save_path / "model")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the last model
            torch.save(
                {
                    "epoch": self.trainer.info["epoch"] + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    
                },
                save_dir / "model_last.pth.tmp"
            )
            os.replace(save_dir / "model_last.pth.tmp", save_dir / "model_last.pth")
            
            
            # Save a copy as best if validation improved
            if self.trainer.cfg.validate_model:
                metric_val = None
                val_result = self.trainer.info.get("validation_result")
                
                if isinstance(val_result, dict):
                    metric_val = val_result.get(self.trainer.cfg.validation_metric_name)
                
                if self.trainer.cfg.validation_metric_higher_is_better:
                    # Copy the last model if it was the best
                    if metric_val is not None and metric_val > self.best_metric_value_high:
                        print(f"New best model! Previous best metric value: {self.best_metric_value_high:.4f}. New best metric value: {metric_val:.4f}")
                        shutil.copyfile(
                            save_dir / "model_last.pth",
                            save_dir / "model_best.pth",
                        )
                        self.best_metric_value_high = metric_val
                else:
                    if metric_val is not None and metric_val < self.best_metric_value_low:
                        print(f"New best model! Previous best metric value: {self.best_metric_value_low:.4f}. New best metric value: {metric_val:.4f}")
                        shutil.copyfile(
                            save_dir / "model_last.pth",
                            save_dir / "model_best.pth",
                        )
                        self.best_metric_value_low = metric_val
            

class TimingHook(Hookbase):
    """Lightweight timing hook that prints wall-clock times and (optionally) per-phase averages.

    - Measures total, per-epoch, and per-step wall times using perf_counter.
    - If the Trainer collects per-phase timings in trainer.info (timing_last_step, timing_epoch_avg),
      this hook will also print those averages each epoch.
    - Printing is limited to the main process.
    """

    def __init__(self, log_every: int = 50) -> None:
        """Create a TimingHook.

        Args:
            log_every: If > 0, prints a brief step timing every N steps.
        """
        self.log_every = int(log_every) if log_every is not None else 0
        if TYPE_CHECKING:
            from .train import TrainerBase as _TrainerBase
            self.trainer: Optional[_TrainerBase]
        self._t_train_start = None
        self._t_epoch_start = None
        self._t_step_start = None
        self._step_counter = 0
        self._epoch_step_time_ms_total = 0.0
        self._epoch_samples_total = 0

    def before_train(self):
        if self.trainer and self.trainer.cfg.timing:
            if comm.is_main_process():
                self._t_train_start = time.perf_counter()

    def before_epoch(self):
        if self.trainer and self.trainer.cfg.timing:
            if comm.is_main_process():
                # Ensure previous GPU work is accounted for before starting epoch timer
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                self._t_epoch_start = time.perf_counter()
                self._step_counter = 0
                self._epoch_step_time_ms_total = 0.0
                self._epoch_samples_total = 0

    def before_step(self):
        if self.trainer and self.trainer.cfg.timing:
            if comm.is_main_process():
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                self._t_step_start = time.perf_counter()
                self._step_counter += 1

    def after_step(self):
        if self.trainer and self.trainer.cfg.timing:
            if not comm.is_main_process():
                return
            if self._t_step_start is None:
                return
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            dt_ms = (time.perf_counter() - self._t_step_start) * 1000.0
            self._epoch_step_time_ms_total += dt_ms
            # Try to infer batch size to compute throughput
            trainer = getattr(self, 'trainer', None)
            if trainer is not None and hasattr(trainer, 'info'):
                iter_data = trainer.info.get('iter_data')  # type: ignore[attr-defined]
                if isinstance(iter_data, (tuple, list)) and len(iter_data) > 0:
                    x0 = iter_data[0]
                    if isinstance(x0, torch.Tensor) and x0.dim() > 0:
                        self._epoch_samples_total += int(x0.shape[0])
            
            # Log the step times
            if self.log_every and (self._step_counter % self.log_every == 0):

                # If a WandB run exists, log step wall time
                if trainer is not None and getattr(trainer, 'wandb_run', None):
                    try:
                        trainer.wandb_run.log(
                            data={
                                'timing/train_step_ms': dt_ms,
                            }, 
                            step=trainer.wandb_run.step
                        )
                    except Exception:
                        pass
                    
                # If TensorBoard is available, log step timing
                if trainer is not None and getattr(trainer, 'tensorboard_writer', None):
                    try:
                        trainer.tensorboard_writer.add_scalar('timing/train_step_ms', dt_ms, self._step_counter)
                    except Exception:
                        pass

    def after_epoch(self):
        if self.trainer and self.trainer.cfg.timing:
            if not comm.is_main_process():
                return
            if self._t_epoch_start is None:
                return
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            epoch_ms = (time.perf_counter() - self._t_epoch_start) * 1000.0
            trainer = getattr(self, 'trainer', None)
            epoch_num = 0
            if trainer is not None and hasattr(trainer, 'info'):
                epoch_num = int(trainer.info.get('epoch', 0))
            steps = max(self._step_counter, 1)
            avg_step_ms = self._epoch_step_time_ms_total / steps
            steps_per_s = 1000.0 / avg_step_ms if avg_step_ms > 0 else 0.0
            samples_per_s = (self._epoch_samples_total / (epoch_ms / 1000.0)) if epoch_ms > 0 else 0.0
            msg = f"[Timing] Epoch {epoch_num+1}/{self.trainer.num_epochs} wall={epoch_ms:.1f}ms | avg_step={avg_step_ms:.1f}ms ({steps_per_s:.1f} steps/s, {samples_per_s:.1f} samples/s)"
            
            # Estimate remaining time:
            if self._t_train_start:
                total_s = time.perf_counter() - self._t_train_start
                avg_epoch_s = total_s / (epoch_num + 1)
                est_remain_s = int((avg_epoch_s * (self.trainer.num_epochs - epoch_num - 1)))
                msg += f" | avg. epoch time: {avg_epoch_s:.2f}s/epoch | Est. remaining training time: {str(dt.timedelta(seconds=est_remain_s))}"
                
            
            print(msg)
            
            # Log epoch timing metrics to WandB/TensorBoard if available
            if trainer is not None and getattr(trainer, 'wandb_run', None):
                try:
                    trainer.wandb_run.log(
                        data={
                            'timing/train_epoch_wall_s': epoch_ms / 1000.0,
                            'timing/train_epoch_avg_step_ms': avg_step_ms,
                            'timing/train_epoch_steps_per_s': steps_per_s,
                            'timing/train_epoch_samples_per_s': samples_per_s,
                        }, 
                        step=trainer.wandb_run.step
                    )
                except Exception:
                    pass
            if trainer is not None and getattr(trainer, 'tensorboard_writer', None):
                try:
                    writer = trainer.tensorboard_writer
                    writer.add_scalar('timing/train_epoch_wall_s', epoch_ms / 1000.0, epoch_num + 1)
                    writer.add_scalar('timing/train_epoch_avg_step_ms', avg_step_ms, epoch_num + 1)
                    writer.add_scalar('timing/train_epoch_steps_per_s', steps_per_s, epoch_num + 1)
                    writer.add_scalar('timing/train_epoch_samples_per_s', samples_per_s, epoch_num + 1)
                except Exception:
                    pass

    def after_train(self):
        if self.trainer and self.trainer.cfg.timing:
            if not comm.is_main_process():
                return
            if self._t_train_start is None:
                return
            total_s = time.perf_counter() - self._t_train_start
            print(f"[Timing] Total training wall time: {total_s:.2f}s")


class SaveTrainingRun(Hookbase):
    def before_train(self):
        # Access trainer.model, trainer.optimizer, etc. through weak proxy
        if self.trainer:
            # Save the codebase, model state, etc.
            cfg = self.trainer.cfg
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"{cfg.model_config.name}_{timestamp}"
            #self.save_path = Path(cfg.save_path / run_name) this already exists
            # TODO: save shiez
            
        raise NotImplementedError