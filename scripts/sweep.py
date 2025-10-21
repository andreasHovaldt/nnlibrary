import os
import sys
import wandb
import importlib
import importlib.util
import pkgutil
from functools import partial
from pathlib import Path

# Set project root to repo root
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from nnlibrary.engines import Trainer

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
wandb_key = None

# If no key in config, try to read from .secrets/wandb file
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
else:
    print("No WandB API key found - WandB logging has been disabled")


# Ensure multiple runs inits are allowed for the current runtime
wandb.setup(wandb.Settings(reinit="create_new"))
# When a new wandb.init is run, a new run is created while still allowing to log data to previous runs
# It is important to handle the runs with the specific run object and not use the general wandb.log function
# On second thought, I dont actually know if it is needed, but it works with it so I wont change it :-)


def sweeper_func(cfg):
    if isinstance(cfg.model_config, dict):
        out_dir: Path = Path().cwd().resolve() / cfg.save_path / cfg.dataset_name / cfg.model_config['name']
    else:
        out_dir: Path = Path().cwd().resolve() / cfg.save_path / cfg.dataset_name / cfg.model_config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run = wandb.init(dir=out_dir)
    
    # Apply the curent sweep run settings to the config
    for key, value in run.config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            print(f"Attr {key} could not be set, falling back to config default!")
    
    trainer = Trainer(cfg=cfg)
    trainer.train()
    
    # Sweep metrics
    sweep_metrics = ['loss','mae', 'rmse', 'r2_score'] # FIXME: Make this accessible when calling the script, or make it defineable in the config being swept
    sweep_log_dict = {metric_name: trainer.info["test_result"][metric_name] for metric_name in sweep_metrics}
    
    run.log(data=sweep_log_dict)
    run.finish()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sweep.py <config_name>")
        sys.exit(1)
    config_name = str(sys.argv[1])
    
    cfg = load_cfg(config_name)
    
    # Assurances before trying to run sweep
    if not hasattr(cfg, 'sweep_configuration'):
        raise AttributeError("The loaded config did not contain a 'sweep_configuration' attribute, please define one for performing a hyperparameter sweep!")
    assert cfg.enable_wandb is True, "Please enable WandB in the config, the hyperparameter sweep depends on WandB integration"
    
    sweep_id = wandb.sweep(sweep=cfg.sweep_configuration, project=cfg.wandb_project_name)
    
    # Pass the function with arguments without executing it, via functools.partial
    wandb.agent(sweep_id, function=partial(sweeper_func, cfg), count=18) # FIXME: How to automate the 'count' variable?
    wandb.finish()