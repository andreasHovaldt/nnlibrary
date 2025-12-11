import os
import gc
import sys
import wandb
import importlib
import importlib.util
import pkgutil
import logging
import argparse
from functools import partial
from pathlib import Path
from typing import Optional
from os import PathLike

parser = argparse.ArgumentParser(description="Helper script used for running hyperparameter sweeps on a config file")
parser.add_argument("-n", "--config-name", type=str, required=True, help="Name of config, either as shorthand 'TCN-reg', as dotted module 'nnlibrary.configs.TCN-reg' or as file path '~/nnlibrary/configs/TCN-reg.py'.")
parser.add_argument("--logging", action="store_true", help="Whether to display logger prints.")
parser.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help="The logging level used if logging is enabled. Default: INFO")

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Dont know why but I could import the function from train.py >:(
def load_cfg(name: str):
    """Load a config module by short name, dotted path, or file path.

    Resolution order:
      1) nnlibrary.configs.<name>
      2) <name> as a dotted module path
      3) file path under nnlibrary/configs (relative or absolute)
    """
    # 1) Try nnlibrary.configs.<name>
    try:
        return importlib.import_module(f"nnlibrary.configs.{name}")
    except ModuleNotFoundError:
        pass
    # 2) Try dotted path directly
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pass
    # 3) Treat as file path relative to configs dir by default
    cfg_path = Path(name)
    if not cfg_path.is_absolute():
        cfg_path = project_root / "nnlibrary" / "configs" / cfg_path
    if cfg_path.suffix != ".py":
        cfg_path = cfg_path.with_suffix(".py")
    if cfg_path.exists():
        spec = importlib.util.spec_from_file_location(cfg_path.stem, cfg_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Enable relative imports by setting the package if the file is in nnlibrary/configs
            if "nnlibrary" in cfg_path.parts and "configs" in cfg_path.parts:
                 module.__package__ = "nnlibrary.configs"
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module
    # If all else fails, list available configs
    try:
        configs_pkg = importlib.import_module("nnlibrary.configs")
        available = [name for _, name, ispkg in pkgutil.iter_modules(configs_pkg.__path__) if not ispkg and not name.startswith("__")]
    except Exception:
        available = []
    raise SystemExit(f"Config '{name}' not found. Available: {', '.join(available) if available else 'no configs discovered'}")


# Set wandb environment variable
def set_wandb_env_var(api_file: Optional[PathLike]):
    wandb_key = None

    # If no key in config, try to read from .secrets/wandb file
    if api_file:
        secrets_file = Path(api_file).resolve()
    else:
        secrets_file = Path().cwd().resolve() / '.secrets' / 'wandb'
    if secrets_file.exists():
        try:
            wandb_key = secrets_file.read_text().strip()
        except Exception as e:
            print(f"Warning: Could not read WandB key from {secrets_file}: {e}")
    else:
        print(f"Could not find the wandb secrets file at: '{secrets_file}'")

    # Set the environment variable if we found a key
    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key
        print("WandB API key loaded successfully")
        return True
    else:
        print("No WandB API key found!")
        return False


def get_run_save_dir(root_dir: Path) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    run_number = len(list(root_dir.iterdir()))
    run_name = random_name_gen(prefix='sweep')
    return root_dir / f"{run_name}-{run_number}"


def sweeper_func(config_name: str, sweep_root: Path):
    cfg = load_cfg(config_name)

    save_dir = get_run_save_dir(root_dir=sweep_root)
    try:
        save_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        print(f"Run directory already exists! Terminating execution to ensure data is not overwritten...")
        exit()
    
    run = wandb.init(name=save_dir.name, dir=save_dir)
    
    # Apply the curent sweep run settings to the config
    for key, value in run.config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            print(f"Attr {key} could not be set, falling back to config default!")
    
    trainer = Trainer(cfg=cfg)
    trainer.train()
    
    # Sweep metrics
    if cfg.task.lower() == 'regression':
        sweep_metrics = ['loss','mae', 'rmse', 'r2_score'] # FIXME: Make this accessible when calling the script, or make it defineable in the config being swept
    elif cfg.task.lower() == 'classification':
        sweep_metrics = ['loss', 'avg_sample_accuracy', 'avg_class_accuracy']
    else: 
        raise ValueError("Config task must be either 'classification' or 'regression'!")
    
    sweep_log_dict = {metric_name: trainer.info["test_result"][metric_name] for metric_name in sweep_metrics}
    
    run.log(data=sweep_log_dict)
    run.finish()
    del trainer, cfg
    gc.collect()


if __name__ == "__main__":
    args = parser.parse_args()
    if not set_wandb_env_var(api_file=None): exit()
    
    from nnlibrary.engines import Trainer
    from nnlibrary.utils.misc import random_name_gen, REPO_ROOT
    from nnlibrary.configs.__base__ import BaseConfig
    
    if args.logging:
        logging.getLogger('matplotlib').setLevel(logging.INFO) # Supress matplib debug logging
        logging.getLogger('urllib3').setLevel(logging.INFO) # Supress urllib3 (connectionpool logs) debug logging
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s | %(filename)s:%(funcName)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            stream=sys.stdout
        )
    else: 
        logger = None 
    
    config_name = str(args.config_name)
    
    cfg = load_cfg(config_name)
    
    
    # Assurances before trying to run sweep
    if not hasattr(cfg, 'sweep_configuration'):
        raise AttributeError("The loaded config did not contain a 'sweep_configuration' attribute, please define one for performing a hyperparameter sweep!")
    assert cfg.enable_wandb is True, "Please enable WandB in the config, the hyperparameter sweep depends on WandB integration"
    
    if getattr(cfg, 'seed') is None:
        print("Seed has not been set, this is recommended for hyperparameter sweeps to remove randomness between the runs.")
        if input("Do you want to continue the sweep without a set seed? y/n: ").lower() not in ['y', 'yes']:
            print("Terminating sweep...")
            exit()
        print("Continuing sweep...")
            

    
    sweep_id = wandb.sweep(sweep=cfg.sweep_configuration, project=cfg.wandb_project_name + '-sweep')

    # Create sweep root save dir 
    if isinstance(cfg.model_config, dict):
        model_config = BaseConfig.from_dict(cfg.model_config)
    else:
        model_config = cfg.model_config
    sweep_save_root: Path = REPO_ROOT / cfg.save_path / cfg.dataset_name / model_config.name / 'sweeps' / cfg.sweep_configuration["name"]
    
    try:
        sweep_save_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if args.logging and (logger is not None): 
            logger.warning(f"Sweep exists at: '{sweep_save_root}', falling back to saving using sweep id as identifier!")
        sweep_save_root = sweep_save_root.parent / sweep_id
        sweep_save_root.mkdir(parents=True, exist_ok=False)
    finally:
        if args.logging and (logger is not None): 
            logger.info(f"Saving the sweep runs to: '{sweep_save_root}'")
    
    # Ensure W&B uses the sweep root for any agent/sweep metadata (defaults to ./wandb otherwise)
    # This affects files like ./wandb/sweep-<id>/config-<id>.yaml which would otherwise be created at CWD.
    os.environ['WANDB_DIR'] = str(sweep_save_root)

    # Now allow multiple run inits for this process and finalize W&B setup
    # https://docs.wandb.ai/guides/runs/multiple-runs-per-process/
    wandb.setup(wandb.Settings(reinit="create_new"))
    
    
    # Pass the function with arguments without executing it, via functools.partial
    wandb.agent(sweep_id, function=partial(sweeper_func, config_name, sweep_save_root))
    wandb.finish()