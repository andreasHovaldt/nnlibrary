import os
import shutil
import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import wandb
import torch
from nnlibrary.engines.eval import Evaluator
import nnlibrary.utils.comm as comm

if TYPE_CHECKING:
    from .train import TrainerBase, Trainer


class Hookbase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    
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
            # print("after_step: ", self.current_iter, current_lr, loss_iter)
            
            self.trainer.wandb_run.log(
                data={
                    "Iter": self.current_iter,
                    "params/lr": current_lr,
                    "train_batch/loss": loss_iter,
                },
                #step=self.trainer.wandb_run.step,
                step=self.current_iter,
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
                            self.trainer.wandb_run.log(
                                data={
                                    f"validation/{key}": value,
                                },
                                step=self.trainer.wandb_run.step,
                            )
    
    def after_train(self):
        if self.trainer and self.trainer.wandb_run:
            
            # Logging test results
            
            if "test_result" in self.trainer.info.keys():
                test_result: dict = self.trainer.info["test_result"]
                if isinstance(test_result, dict):
                    
                    for key, value in test_result.items():
                        
                        # Specific logging cases
                        if key == "confusion_matrix":
                            fig, ax = value
                            self.trainer.wandb_run.log(
                                data={
                                    f"test/{key}": wandb.Image(fig)
                                }
                            )
                        
                        # Dump the remaining test metrics
                        else:
                            self.trainer.wandb_run.log(
                                data={
                                    f"test/{key}": value,
                                },
                                step=self.trainer.wandb_run.step,
                            )
                
    
    
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
                        if key == "confusion_matrix": continue
                        self.trainer.tensorboard_writer.add_scalar(f"validation/{key}", value, epoch)
        

class ValidationHook(Hookbase):
    def __init__(self) -> None:
        self.validator = None
            
    
    def after_epoch(self):
        
        if self.trainer and self.trainer.cfg.validate_model:
            
            if self.validator is None:
                validation_loader = self.trainer.build_dataloader(self.trainer.cfg.dataset.val)
                self.validator = Evaluator(
                    dataloader=validation_loader,
                    loss_fn=self.trainer.loss_fn,
                    device=self.trainer.device,
                    amp_enable=self.trainer.amp_enable,
                    amp_dtype=self.trainer.amp_dtype,
                    class_names=self.trainer.cfg.dataset.info["class_names"],
                    detailed=self.trainer.cfg.validation_confusion_matrix,
                )
        
            result = self.validator.eval(
                model=self.trainer.model
            )
            self.trainer.info["validation_result"] = result
        return None


class TestHook(Hookbase):
    def __init__(self) -> None:
        self.tester = None
            
    
    def after_train(self):
        
        if self.trainer and self.trainer.cfg.test_model:
            
            if self.tester is None:
                test_loader = self.trainer.build_dataloader(self.trainer.cfg.dataset.test)
                self.tester = Evaluator(
                    dataloader=test_loader,
                    loss_fn=self.trainer.loss_fn,
                    device=self.trainer.device,
                    amp_enable=self.trainer.amp_enable,
                    amp_dtype=self.trainer.amp_dtype,
                    class_names=self.trainer.cfg.dataset.info["class_names"],
                    detailed=True,
                )
        
            result = self.tester.eval(
                model=self.trainer.model
            )
            self.trainer.info["test_result"] = result
            
            print(f"Final test result:")
            for key, value in result.items():
                if key == "confusion_matrix": continue
                print(f"   {key}: {value:.4f}")
            
            if "confusion_matrix" in result:
                import matplotlib.pyplot as plt
                from matplotlib.figure import Figure
                from matplotlib.axes import Axes
                
                fig, ax = result["confusion_matrix"]
                fig: Figure
                ax: Axes
                
                figure_dir = Path(self.trainer.save_path / "figures")
                figure_dir.mkdir(parents=True, exist_ok=True)
                
                fig.savefig(figure_dir / "confusion_matrix_test_split.png")
                
        return None
  

class CheckpointerHook(Hookbase):
    def __init__(self) -> None:
        self.best_metric_value: float = 0.0
        
        
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
                if metric_val is not None and metric_val > self.best_metric_value:
                    # Copy the last model if it was the best
                    shutil.copyfile(
                        save_dir / "model_last.pth",
                        save_dir / "model_best.pth",
                    )
                    self.best_metric_value = metric_val
            


class SaveTrainingRun(Hookbase):
    def before_train(self):
        # Access trainer.model, trainer.optimizer, etc. through weak proxy
        if self.trainer:
            # Save the codebase, model state, etc.
            cfg = self.trainer.cfg
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"{cfg.model_config.name}_{timestamp}"
            self.save_path = Path(cfg.save_path / run_name)
            # TODO: save shiez
            
        raise NotImplementedError