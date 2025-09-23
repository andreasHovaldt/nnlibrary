import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .train import TrainerBase


class Hookbase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    
    To avoid circular reference, hooks and trainer cannot own each other.
    This normally does not matter, but will cause memory leak if the
    involved objects contain __del__:
    See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
    """
    
    trainer = None  # A weak reference to the trainer object.
    
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
    def after_step(self):
        # Access trainer through weak proxy
        if self.trainer:
            loss = self.trainer.info["loss_iter"]
            # Log to wandb
        raise NotImplementedError
    
    def after_epoch(self):
        # Access epoch loss from trainer
        if self.trainer:
            epoch_loss = self.trainer.info["loss_epoch_avg"]
            # Log to wandb
        raise NotImplementedError


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


class TestHook(Hookbase):
    def before_train(self):
        print("Test hook - Before train")
    
    def before_epoch(self):
        print("Test hook - Before epoch")
    
    def after_epoch(self):
        print("Test hook - After epoch")
        
        if self.trainer:
            epoch_loss = self.trainer.info["loss_epoch_avg"]
            print(epoch_loss)
    
    def after_train(self):
        print("Test hook - After train")