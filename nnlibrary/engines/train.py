import tensorboardX
import wandb
import logging
import functools
import weakref

import datetime as dt

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import nnlibrary.models
import nnlibrary.datasets
import nnlibrary.utils.comm as comm

from .hooks import Hookbase

from nnlibrary.configs import BaseConfig, DataLoaderConfig


AMP_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class TrainerBase:
    def __init__(self) -> None:
        
        self.save_path: Path
        
        self.hooks: list[Hookbase]
        self.logger: logging.Logger
        self.epochs: int
        self.start_epoch: int
        self.max_epoch: int
        
        self.amp_enable: bool
        self.amp_dtype: torch.dtype
        
        self.model: nn.Module
        self.trainloader: DataLoader
        self.valloader: DataLoader
        
        self.optimizer: optim.Optimizer
        self.scheduler: optim.lr_scheduler.LRScheduler
        self.loss_fn: nn.Module
        
        self.tensorboard_writer: tensorboardX.SummaryWriter
        self.wandb_run: wandb.Run # Basically the writer used for writing logs to wandb
        
        # Dict used for passing around various runtime information
        self.info = dict()
        
        
    def train(self) -> None:
        
        self.before_train()
        
        for self.epoch in tqdm(range(self.start_epoch, self.max_epoch)):
            self.before_epoch()
            
            for self.info["epoch_iter"], self.info["iter_data"] in enumerate(self.trainloader):
                self.before_step()
                self.run_step()
                self.after_step()
            
            self.after_epoch()
        self.after_train()
        
    
    def before_train(self):
        for hook in self.hooks:
            hook.before_train()
    
    def before_epoch(self):
        for hook in self.hooks:
            hook.before_epoch()
    
    def before_step(self):
        for hook in self.hooks:
            hook.before_step()
    
    def run_step(self):
        raise NotImplementedError
    
    def after_step(self):
        for hook in self.hooks:
            hook.after_step()
    
    def after_epoch(self):
        for hook in self.hooks:
            hook.after_epoch()
    
    def after_train(self):
        
        # Sync GPU
        comm.synchronize()
        
        for hook in self.hooks:
            hook.after_train()
    
        # Shut off logging writers
        if comm.is_main_process():
            self.tensorboard_writer.close()
            self.wandb_run.finish()
    

import nnlibrary.configs.hvac_mode_classifier as cfg_example
class Trainer(TrainerBase):
    def __init__(self, cfg: cfg_example) -> None:
        super().__init__()
        
        self.cfg = cfg
                
        # Initialize hooks and pass self to them
        self.hooks: list[Hookbase] = self.register_hooks()
        
            
        self.logger: logging.Logger = logging.getLogger()
        self.epochs: int = cfg.epochs
        self.start_epoch: int = cfg.start_epoch
        self.max_epoch: int = cfg.max_epoch
        # self.max_iter = 0 # <- it seems this only relates to calculating an ETA, so maybe the number of epochs * len(dataloader)? (Total batches to process during training)
        
        self.device: str = cfg.device
        self.amp_enable: bool = cfg.amp_enable
        self.amp_dtype: torch.dtype = AMP_DTYPES[cfg.amp_dtype]
        
        self.model = self.build_model(cfg.model_config)
        self.trainloader = self.build_dataloader(cfg.dataset.train)
        self.valloader = self.build_dataloader(cfg.dataset.val)
        
        self.optimizer = self.build_optimizer(cfg.optimizer)
        self.scheduler = self.build_scheduler(cfg.scheduler)
        self.loss_fn = self.build_loss_fn(cfg.loss_fn)
        
        # self.tensorboard_writer = self.build_tensorboard_writer()
        # self.wandb_run = self.build_wandb_run()
        
        # Dict used for passing around various runtime information
        self.info = dict()
        

    def run_step(self):
        
        # Use Automatic Mixed Precision to speed up computations
        # https://docs.pytorch.org/docs/stable/amp
        autocast = functools.partial(torch.autocast, device_type=self.device)
        
        # Obtain iteration data
        X_batch, y_batch = self.info["iter_data"]
        
        # Move to correct torch device
        if isinstance(X_batch, torch.Tensor): X_batch = X_batch.to(device=self.device, non_blocking=True)
        if isinstance(y_batch, torch.Tensor): y_batch = y_batch.to(device=self.device, non_blocking=True)
            
        # Train model
        with autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss: torch.Tensor = self.loss_fn(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Log the train loss
            self.info["loss_iter"] = loss.item()
            self.info["loss_epoch_total"] += loss.item()
            
            
    def before_epoch(self):
        self.info["loss_epoch_total"] = 0
        return super().before_epoch()
    
    
    def after_epoch(self):
        self.info["loss_epoch_avg"] = self.info["loss_epoch_total"] / len(self.trainloader)
        return super().after_epoch()


    def build_model(self, model_config: BaseConfig) -> nn.Module:
        """Build model from configuration.
    
        Args:
            model_config (dict): Model configuration containing:
                name (str): Model class name from './nnlibrary/models'
                args (dict): Constructor arguments for the model
        
        Returns:
            nn.Module: Configured model instance
            
        Raises:
            ValueError: If model class doesn't exist
        """
        model_name = model_config.name
        model_args = model_config.args
        
        # Get the model class from the models module
        if hasattr(nnlibrary.models, model_name):
            model_class = getattr(nnlibrary.models, model_name)
            return model_class(**model_args)
        else:
            raise ValueError(f"Model '{model_name}' not found in nnlibrary.models")


    def build_dataset(self, dataset_config: BaseConfig) -> Dataset:
        """Build dataset from configuration.
    
        Args:
            dataset_config (dict): Dataset configuration containing:
                name (str): Dataset class name from './nnlibrary/datasets'
                args (dict): Constructor arguments for the dataset
        
        Returns:
            DataLoader: Configured dataloader instance
            
        Raises:
            ValueError: If dataset class doesn't exist
        """
        dataset_name = dataset_config.name
        dataset_args = dataset_config.args
        
        # Get the dataset class from the datasets module
        if hasattr(nnlibrary.datasets, dataset_name):
            dataset_class = getattr(nnlibrary.datasets, dataset_name)
            return dataset_class(**dataset_args)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found in nnlibrary.datasets")
    
    
    def build_dataloader(self, dataloader_config: DataLoaderConfig) -> DataLoader:
        dataset = self.build_dataset(dataset_config=dataloader_config.dataset)
        return DataLoader(dataset=dataset, batch_size=dataloader_config.batch_size, shuffle=dataloader_config.shuffle)
    
    
    def build_optimizer(self, optimizer_config: BaseConfig) -> optim.Optimizer:
        """Build optimizer from configuration.
    
        Args:
            optimizer_config (dict): Optimizer configuration containing:
                name (str): Optimizer class name from torch.optim
                args (dict): Constructor arguments for the optimizer
        
        Returns:
            Optimizer: Configured optimizer instance
            
        Raises:
            ValueError: If optimizer class doesn't exist
        """
        optimizer_name = optimizer_config.name
        optimizer_args = optimizer_config.args
        
        # Get the optimizer class from the optim module
        if hasattr(torch.optim, optimizer_name):
            optimizer_class = getattr(torch.optim, optimizer_name)
            return optimizer_class(params=self.model.parameters(), lr=self.cfg.lr, **optimizer_args)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim")
        
    
    def build_scheduler(self, scheduler: BaseConfig) -> optim.lr_scheduler.LRScheduler:
        """Build scheduler from configuration.
    
        Args:
            scheduler (dict): Scheduler configuration containing:
                name (str): Scheduler class name from torch.optim
                args (dict): Constructor arguments for the scheduler
        
        Returns:
            sycheduler: Configured scheduler instance
            
        Raises:
            ValueError: If scheduler class doesn't exist
        """
        scheduler_name = scheduler.name
        scheduler_args = scheduler.args
        
        # Get the scheduler class from the lr_scheduler module
        if hasattr(optim.lr_scheduler, scheduler_name):
            scheduler_class = getattr(optim.lr_scheduler, scheduler_name)
            return scheduler_class(optimizer=self.optimizer, **scheduler_args)
        else:
            raise ValueError(f"Scheduler '{scheduler_name}' not found in torch.optim.lr_scheduler")

    
    # TODO: Make compatible with custom loss functions
    #   - Could make it search for loss functions in utils/loss.py with fallback to torch.nn
    def build_loss_fn(self, loss_fn: BaseConfig) -> nn.Module:
        """Build loss function from configuration.
    
        Args:
            loss_fn (dict): Loss function configuration containing:
                name (str): Loss function class name from torch.optim
                args (dict): Constructor arguments for the Loss function
        
        Returns:
            sycheduler: Configured loss function instance
            
        Raises:
            ValueError: If loss function class doesn't exist
        """
        loss_fn_name = loss_fn.name
        loss_fn_args = loss_fn.args
        
        # Get the loss_fn class from the torch.nn module
        if hasattr(torch.nn, loss_fn_name):
            loss_fn_class = getattr(torch.nn, loss_fn_name)
            return loss_fn_class(**loss_fn_args)
        else:
            raise ValueError(f"Loss function '{loss_fn_name}' not found in torch.nn")
        

    def build_tensorboard_writer(self) -> tensorboardX.SummaryWriter:
        writer = tensorboardX.SummaryWriter(self.cfg.save_path)
        
        raise NotImplementedError
    
    def build_wandb_run(self) -> wandb.Run:
        raise NotImplementedError
    
    def register_hooks(self) -> list[Hookbase]:
        """
        Register hooks and assign weak proxy references to avoid circular references.
        
        To avoid circular reference, hooks and trainer cannot own each other.
        This normally does not matter, but will cause memory leak if the
        involved objects contain __del__:
        See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
        """
        import inspect
        
        hooks = []
        for hook_class_or_instance in self.cfg.hooks:
            if inspect.isclass(hook_class_or_instance):
                # It's a class, instantiate it
                hook_instance = hook_class_or_instance()
            else:
                # It's already an instance
                hook_instance = hook_class_or_instance
            
            assert isinstance(hook_instance, Hookbase), f"Hook {hook_instance} must inherit from Hookbase"
            
            # Assign weak proxy to avoid circular reference
            hook_instance.trainer = weakref.proxy(self)
            hooks.append(hook_instance)
        
        return hooks