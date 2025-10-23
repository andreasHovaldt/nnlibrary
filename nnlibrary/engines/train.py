import tensorboardX
import wandb
import logging
import functools
import weakref
import os
import random
import numpy as np

from pathlib import Path
from typing import Any, Union, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import nnlibrary.models
import nnlibrary.datasets
import nnlibrary.utils.loss
import nnlibrary.utils.schedulers
import nnlibrary.utils.comm as comm

from .hooks import Hookbase

from nnlibrary.configs import BaseConfig, DataLoaderConfig


AMP_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class TrainerBase:
    def __init__(self) -> None:
        
        self.cfg: Any
        self.save_path: Path
        
        self.hooks: list[Hookbase]
        self.logger: logging.Logger
        self.num_epochs: int
        
        self.amp_enable: bool
        self.amp_dtype: torch.dtype
        
        self.model: nn.Module
        self.trainloader: DataLoader
        
        self.optimizer: optim.Optimizer
        self.scheduler: optim.lr_scheduler.LRScheduler
        self.loss_fn: nn.Module
        
        # The writers are None if tensorboard or wandb is turned off, respectively
        self.tensorboard_writer: tensorboardX.SummaryWriter | None
        self.wandb_run: wandb.Run | None
        # Track ownership of the W&B run to avoid finishing a run we didn't create
        self._wandb_run_owned = False
        
        # Dict used for passing around various runtime information
        #  Mostly used by the hooks
        self.info = dict()
        
        
    def train(self) -> None:
        
        print("Started training...")
        self.before_train()
        
        for self.info["epoch"] in range(self.num_epochs):
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
            if self.tensorboard_writer:
                self.tensorboard_writer.flush()
                self.tensorboard_writer.close()
            # Only finish the W&B run if this trainer created/owns it
            if self.wandb_run and getattr(self, "_wandb_run_owned", False):
                self.wandb_run.finish()
    


class Trainer(TrainerBase):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.parse_dict_configs(cfg) # Individual checks are still there in the build methods, since this conversion was added afterwards and the check doesn't destroy anything
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Seed everything early for reproducibility if configured
        self._seed_everything(getattr(self.cfg, 'seed', None))

        self.save_path = Path().cwd().resolve() / self.cfg.save_path / self.cfg.dataset_name / self.cfg.model_config.name
            
        self.hooks: list[Hookbase] = self.register_hooks() # Initialize hooks and pass self to them
        self.num_epochs: int = self.cfg.num_epochs
        
        self.device: str = self.cfg.device
        self.amp_enable: bool = self.cfg.amp_enable
        self.amp_dtype: torch.dtype = AMP_DTYPES[self.cfg.amp_dtype]
        
        self.model = self.build_model(self.cfg.model_config)
        self.logger.debug("Successfully built model!")
        
        # Set the batch sizes based on the config
        #   This is done here instead of directly in the config
        #   to make it compatible with wandb sweeps, if you set 
        #   the batch_size parameters directly in the config file, 
        #   sweeps which change 'train_batch_size' or 
        #   'eval_batch_size' do not work as intended. 
        if self.cfg.dataset.train.batch_size is None:
            self.cfg.dataset.train.batch_size = self.cfg.train_batch_size
        if self.cfg.dataset.val.batch_size is None:
            self.cfg.dataset.val.batch_size = self.cfg.eval_batch_size
        if self.cfg.dataset.test.batch_size is None:
            self.cfg.dataset.test.batch_size = self.cfg.eval_batch_size
        
        self.trainloader = self.build_dataloader(self.cfg.dataset.train)
        self.logger.debug("Successfully built trainloader!")
        
        self.optimizer = self.build_optimizer(self.cfg.optimizer)
        self.logger.debug("Successfully built optimizer!")
        self.scheduler = self.build_scheduler(self.cfg.scheduler)
        self.logger.debug("Successfully built scheduler!")
        self.loss_fn = self.build_loss_fn(self.cfg.loss_fn)
        self.logger.debug("Successfully built loss function!")
        
        self.tensorboard_writer = self.build_tensorboard_writer()
        self.wandb_run = self.build_wandb_run()
        if self.wandb_run is not None: self.logger.info(f"WandB logging dir: {self.wandb_run.dir}")
        
        # Dict used for passing around various runtime information
        self.info = dict()
    
    @staticmethod
    def parse_dict_configs(cfg: Any):
        """Converts some of the dict configs to BaseConfig"""
        
        for attr_name in ['model_config', 'loss_fn', 'optimizer', 'scheduler']:
            current_attr = getattr(cfg, attr_name)
            if isinstance(current_attr, dict):
                # Convert the attribute from dict to BaseConfig
                setattr(cfg, attr_name, BaseConfig.from_dict(current_attr))

        # Dataset special case
        for split_name in ['train', 'val', 'test']:
            if hasattr(cfg.dataset, split_name):
                split_config = getattr(cfg.dataset, split_name)
                if hasattr(split_config, 'dataset') and isinstance(split_config.dataset, dict):
                    # Convret the dataset dict to BaseConfig 
                    setattr(split_config, 'dataset', BaseConfig.from_dict(split_config.dataset))
        return cfg

    def _seed_everything(self, seed: int | None) -> None:
        """Set seeds for Python, NumPy, and PyTorch. Enable deterministic behavior where possible.

        Args:
            seed: Integer seed or None to skip seeding.
        
        Note:
            Deterministic behavior guarantees bit-wise reproducibility ONLY when:
            - Using the same PyTorch, CUDA, and cuBLAS versions
            - Running on GPUs with identical architecture and SM count
            - Using the same random seed
            
            Results may differ across:
            - Different CUDA toolkit versions
            - Different GPU models (even same generation)
            - CPU vs GPU execution
            
            Performance impact: Deterministic mode may be 10-50% slower depending on operations.
        """
        if seed is None:
            self.logger.debug("No seed was provided. The following run is non deterministic!")
            return
        self.logger.info(f"A seed was provided ({seed}), setting random seeds and deterministic behaviour!")
        try:
            os.environ["PYTHONHASHSEED"] = str(seed)
        except Exception as e:
            self.logger.info(f"Failed setting PYTHONHASHSEED environment variable: {e}")
        try:
            random.seed(seed)
        except Exception as e:
            self.logger.info(f"Failed setting seed for native python random variable generator: {e} ")
        try:
            np.random.seed(seed)
        except Exception as e:
            self.logger.info(f"Failed settomg seed fpr numpy rng: {e}")
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception as e:
            self.logger.info(f"Failed setting torch random seed: {e}")
        # Deterministic settings (may reduce performance)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            self.logger.info(f"Failed setting cudnn backends to deterministic behavior: {e}")
        # Optional: enforce deterministic algorithms where supported
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        except Exception as e:
            # Not all environments or ops support full determinism
            self.logger.info(f"Faile setting torch to use deterministic algorithms: {e}")

    def run_step(self):
        
        # Use Automatic Mixed Precision to speed up computations
        # https://docs.pytorch.org/docs/stable/amp
        autocast = functools.partial(torch.autocast, device_type=self.device)
        
        # Obtain iteration data
        X_batch, y_batch = self.info["iter_data"]
        
        # Move to correct torch device
        if isinstance(X_batch, torch.Tensor): 
            X_batch = X_batch.to(device=self.device, non_blocking=True)
        if isinstance(y_batch, torch.Tensor): 
            y_batch = y_batch.to(device=self.device, non_blocking=True)
            
        # Train model
        with autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
            # Forward pass
            self.optimizer.zero_grad()
            y_pred: torch.Tensor = self.model(X_batch)
            loss: torch.Tensor = self.loss_fn(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Log the train loss
            self.info["loss_iter"] = loss.item()
            self.info["loss_epoch_total"] += loss.item()
            
    def before_epoch(self):
        # Make sure the model is in training mode at the start of every epoch
        # (important if any validation/eval toggled it previously)
        if hasattr(self, "model") and isinstance(self.model, nn.Module):
            self.model.train()
        self.info["loss_epoch_total"] = 0
        return super().before_epoch()
    
    def before_step(self):
        self.info["current_lr"] = self.scheduler.get_last_lr()
        return super().before_step()
    
    
    def after_epoch(self):
        self.info["loss_epoch_avg"] = self.info["loss_epoch_total"] / len(self.trainloader)
        return super().after_epoch()


    def build_model(self, model_config: Union[BaseConfig, Dict]) -> nn.Module:
        """Build model from configuration.
    
        Args:
            model_config (BaseConfig): Model configuration containing:
                name (str): Model class name from './nnlibrary/models'
                args (dict): Constructor arguments for the model
        
        Returns:
            nn.Module: Configured model instance
            
        Raises:
            ValueError: If model class doesn't exist
        """
        # Parse config
        if isinstance(model_config, BaseConfig):
            model_name = model_config.name
            model_args = model_config.args
        elif isinstance(model_config, dict):
            model_name = model_config["name"]
            model_args = model_config["args"]
        else:
            raise ValueError(f"Config type is not supported: {type(model_config)}")
        
        # Get the model class from the models module
        if hasattr(nnlibrary.models, model_name):
            model_class = getattr(nnlibrary.models, model_name)
            
            # Extract module name for logging
            module_path: str = model_class.__module__  # e.g., "nnlibrary.models.mlp"
            module_name = module_path.split('.')[-1]  # "mlp"
            
            # Store for logging purposes
            self.model_module = module_name
            
            return model_class(**model_args).to(device=self.device)
        else:
            raise ValueError(f"Model '{model_name}' not found in nnlibrary.models")


    def build_dataset(self, dataset_config: Union[BaseConfig, Dict], standardize_target = False, normalize_target = False) -> Dataset:
        """Build dataset from configuration.
    
        Args:
            dataset_config (BaseConfig): Dataset configuration containing:
                name (str): Dataset class name from './nnlibrary/datasets'
                args (dict): Constructor arguments for the dataset
        
        Returns:
            DataLoader: Configured dataloader instance
            
        Raises:
            ValueError: If dataset class doesn't exist
        """
        # Parse config
        if isinstance(dataset_config, BaseConfig):
            dataset_name = dataset_config.name
            dataset_args = dataset_config.args
        elif isinstance(dataset_config, dict):
            dataset_name = dataset_config["name"]
            dataset_args = dataset_config["args"]
        else:
            raise ValueError(f"Config type is not supported: {type(dataset_config)}")
        
        # Make sure both standardization and normalization isn't enabled
        assert not (standardize_target and normalize_target), "Cannot use both standardize_target and normalize_target at the same time"
        
        # Get the dataset class from the datasets module
        if hasattr(nnlibrary.datasets, dataset_name):
            dataset_class = getattr(nnlibrary.datasets, dataset_name)
            if (not standardize_target) and (not normalize_target):
                return dataset_class(**dataset_args)
            else:
                stats_dir = self.cfg.data_root / "stats"
                try:
                    import numpy as np
                    if standardize_target:
                        from nnlibrary.utils.operations import Standardize
                        self.target_transform = Standardize(
                            mean=np.load(stats_dir / "target_mean.npy").astype(float).tolist(),
                            std=np.load(stats_dir / "target_std.npy").astype(float).tolist(),
                        )
                        print("Using standardized targets!")
                    
                    elif normalize_target:
                        from nnlibrary.utils.operations import MinMaxNormalize
                        self.target_transform = MinMaxNormalize(
                            min_vals=np.load(stats_dir / "target_min.npy").astype(float).tolist(),
                            max_vals=np.load(stats_dir / "target_max.npy").astype(float).tolist(),
                        )
                        print("Using normalized targets!")
                    
                    return dataset_class(target_transform=self.target_transform, **dataset_args)
                
                except TypeError as e:
                    print(e)
                    print(f"WARN: {e}!")
                    if input("Try to continue without augmented targets? (y/n) ").lower() == 'y':
                        print("Continuing without augmented targets...")
                        return dataset_class(**dataset_args)
                    else: exit()
                
                except FileNotFoundError as e:
                    print(f"WARN: {e}!")
                    if input("Continue without augmented targets? (y/n) ").lower() == 'y':
                        print("Continuing without augmented targets..")
                        return dataset_class(**dataset_args)
                    else: exit()
                
                except Exception as e:
                    print(e)
                    exit()
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found in nnlibrary.datasets")
    
    
    def build_dataloader(self, dataloader_config: DataLoaderConfig) -> DataLoader: # TODO: Make it able to accept a dict as config input
        dataset: Dataset[Any] = self.build_dataset(
            dataset_config=dataloader_config.dataset, 
            standardize_target=self.cfg.dataset.info.get("standardize_target", False),
            normalize_target=self.cfg.dataset.info.get("normalize_target", False),
        )
        
        # Gracefully support optional perf params even if not present in config
        if dataloader_config.num_workers: 
            num_workers = dataloader_config.num_workers
        else: 
            num_workers = max(1, (os.cpu_count() or 2) // 2)
        
        if dataloader_config.pin_memory: 
            pin_memory = dataloader_config.pin_memory
        else: 
            pin_memory = (self.cfg.device == 'cuda')

        # Reproducible shuffling and per-worker seeding
        generator = None
        worker_init_fn = None
        
        seed = getattr(self.cfg, 'seed', None)
        if seed is not None:
            # DataLoader RNG for shuffling (only matters if shuffle=True)
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            
            # Seed Python and NumPy in each worker deterministically
            def _seed_worker(worker_id: int):
                worker_seed = (torch.initial_seed() + worker_id) % 2**32
                random.seed(worker_seed)
                np.random.seed(worker_seed)
            
            worker_init_fn = _seed_worker
        
        return DataLoader(
            dataset=dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=dataloader_config.shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
    
    
    def build_optimizer(self, optimizer_config: Union[BaseConfig, Dict]) -> optim.Optimizer:
        """Build optimizer from configuration.
    
        Args:
            optimizer_config (BaseConfig): Optimizer configuration containing:
                name (str): Optimizer class name from torch.optim
                args (dict): Constructor arguments for the optimizer
        
        Returns:
            Optimizer: Configured optimizer instance
            
        Raises:
            ValueError: If optimizer class doesn't exist
        """
        # Parse config
        if isinstance(optimizer_config, BaseConfig):
            optimizer_name = optimizer_config.name
            optimizer_args = optimizer_config.args
        elif isinstance(optimizer_config, dict):
            optimizer_name = optimizer_config["name"]
            optimizer_args = optimizer_config["args"]
        else:
            raise ValueError(f"Config type is not supported: {type(optimizer_config)}")
        
        # Get the optimizer class from the optim module
        if hasattr(torch.optim, optimizer_name):
            optimizer_class = getattr(torch.optim, optimizer_name)
            return optimizer_class(params=self.model.parameters(), lr=self.cfg.lr, **optimizer_args)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim")
        
    
    def build_scheduler(self, scheduler_config: Union[BaseConfig, Dict]) -> optim.lr_scheduler.LRScheduler:
        """Build scheduler from configuration.
    
        Args:
            scheduler (BaseConfig): Scheduler configuration containing:
                name (str): Scheduler class name from torch.optim
                args (dict): Constructor arguments for the scheduler
        
        Returns:
            scheduler: Configured scheduler instance
            
        Raises:
            ValueError: If scheduler class doesn't exist
        """ 
        # Parse config
        if isinstance(scheduler_config, BaseConfig):
            scheduler_name = scheduler_config.name
            scheduler_args = scheduler_config.args.copy() # Copy to avoid modifying original # TODO: wtf is this?
        elif isinstance(scheduler_config, dict):
            scheduler_name = scheduler_config["name"]
            scheduler_args = scheduler_config["args"].copy()
        else:
            raise ValueError(f"Config type is not supported: {type(scheduler_config)}")
        
        
        # Get the scheduler class from the nnlibrary.utils.schedulers module
        if hasattr(nnlibrary.utils.schedulers, scheduler_name):
            scheduler_class = getattr(nnlibrary.utils.schedulers, scheduler_name)
            # Pass weak proxy to trainer to avoid circular references (same as hooks)
            return scheduler_class(optimizer=self.optimizer, trainer=weakref.proxy(self), **scheduler_args)
        
        # Else get the scheduler class from the lr_scheduler module
        elif hasattr(optim.lr_scheduler, scheduler_name):
            scheduler_class = getattr(optim.lr_scheduler, scheduler_name)
            return scheduler_class(optimizer=self.optimizer, **scheduler_args)
        else:
            raise ValueError(f"Scheduler '{scheduler_name}' not found")

    
    def build_loss_fn(self, loss_fn_config: Union[BaseConfig, Dict]) -> nn.Module:
        """Build loss function from configuration.
    
        Args:
            loss_fn (BaseConfig): Loss function configuration containing:
                name (str): Loss function class name from nnlibrary.utils.loss or torch.nn
                args (dict): Constructor arguments for the loss function
        
        Returns:
            loss_fn: Configured loss function instance
            
        Raises:
            ValueError: If loss function class doesn't exist
        """
        # Parse config
        if isinstance(loss_fn_config, BaseConfig):
            loss_fn_name = loss_fn_config.name
            loss_fn_args = loss_fn_config.args
        elif isinstance(loss_fn_config, dict):
            loss_fn_name = loss_fn_config["name"]
            loss_fn_args = loss_fn_config["args"]
        else:
            raise ValueError(f"Config type is not supported: {type(loss_fn_config)}")
        
        # First try nnlibrary.utils.loss for custom loss functions
        if hasattr(nnlibrary.utils.loss, loss_fn_name):
            loss_fn_class = getattr(nnlibrary.utils.loss, loss_fn_name)
            return loss_fn_class(**loss_fn_args)
        
        # Then fall back to trying torch.nn
        elif hasattr(torch.nn, loss_fn_name):
            loss_fn_class = getattr(torch.nn, loss_fn_name)
            return loss_fn_class(**loss_fn_args)
        else:
            raise ValueError(f"Loss function '{loss_fn_name}' not found")
        

    def build_tensorboard_writer(self) -> tensorboardX.SummaryWriter | None:
        if self.cfg.enable_tensorboard and comm.is_main_process():
            writer = tensorboardX.SummaryWriter(str(self.save_path / "tensorboard"))
            self.logger.info(f"Tensorboard writer logging dir: {self.save_path / 'tensorboard'}")
            return writer
        else:
            return None
    
    def build_wandb_run(self) -> wandb.Run | None:
        if self.cfg.enable_wandb and comm.is_main_process():
            
            if self.cfg.dataset.info.get("standardize_target", False):
                target_transform = 'standardization'
            elif self.cfg.dataset.info.get("normalize_target", False):
                target_transform = 'normalization'
            else:
                target_transform = None

            # If an active W&B run already exists (e.g., created by a sweep script), reuse it
            if wandb.run is not None:
                self.logger.info("Reusing active Weights & Biases run (likely sweep-managed); will not re-initialize.")
                self._wandb_run_owned = False

                # Best-effort: update any missing config fields without overwriting sweep-controlled values
                try:
                    existing_cfg = wandb.run.config
                    supplemental_cfg = dict(
                        dataset=f"{self.cfg.dataset_name}/{self.cfg.data_root.name}",
                        target_transform=target_transform,
                        task=self.cfg.task,
                        architecture=self.model_module,
                        model_name=self.cfg.model_config.name,
                        epochs=self.cfg.num_epochs,
                        train_batch_size=self.cfg.train_batch_size,
                        lr=self.cfg.lr,
                        loss_fn=self.cfg.loss_fn.name,
                        optimizer=self.cfg.optimizer.name,
                        scheduler=self.cfg.scheduler.name,
                    )
                    # Only include keys not present in the existing config
                    missing = {k: v for k, v in supplemental_cfg.items() if k not in existing_cfg}
                    if missing:
                        existing_cfg.update(missing, allow_val_change=False)
                except Exception as e:
                    print(f'Tried to update run info but failed: {e}')
                    # Config update is best-effort; ignore if not supported
                    pass

                return wandb.run

            # Otherwise, initialize a fresh run and mark ownership
            run = wandb.init(
                entity=self.cfg.wandb_group_name,
                project=self.cfg.wandb_project_name,
                # name=f"{self.cfg.dataset_name}/{self.cfg.model_config.name}",
                tags=[f"{self.cfg.dataset_name}/{self.cfg.data_root.name}", self.model_module, self.cfg.model_config.name],
                group=self.cfg.dataset_name,
                # sync_tensorboard=True, # TODO: Look into this
                dir=self.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key) if self.cfg.wandb_key else None,
                config=dict(
                    dataset=f"{self.cfg.dataset_name}/{self.cfg.data_root.name}",  # self.cfg.dataset_name,
                    target_transform=target_transform,
                    task=self.cfg.task,
                    architecture=self.model_module,
                    model_name=self.cfg.model_config.name,
                    epochs=self.cfg.num_epochs,
                    train_batch_size=self.cfg.train_batch_size,
                    lr=self.cfg.lr,
                    loss_fn=self.cfg.loss_fn.name,  # TODO: Look into logging the args as well
                    optimizer=self.cfg.optimizer.name,
                    scheduler=self.cfg.scheduler.name,
                ),
            )
            self._wandb_run_owned = True
            return run
        else:
            return None
            
            
    
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
            self.logger.debug(f"Successfully registered hook: {hook_instance}")
        
        return hooks
    



""" 
Out of date - written keys might not be accurate and info contains more keys now

What self.info contans and when it is accessible for potential hooks:

# Before epoch
self.info["epoch"] == The current epoch number

# Before step
self.info["epoch_iter"] == Current iteration within the current epoch (batch number)
self.info["iter_data"] == Contains current batch data, X,y
self.info["current_lr"] == Contains current learning rate

# After step
self.info["loss_iter"] == Loss of current batch
self.info["loss_epoch_total"] == The summed loss for the current epoch, is only useful in after epoch

# After epoch
self.info["loss_epoch_avg"] == The average loss across all batches for the current epoch
self.info["validation_result"] == The result of the validation run

# After train
self.info["test_result"] == The result of the test run

"""