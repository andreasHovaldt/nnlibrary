import os
import sys
import wandb
import importlib
from functools import partial
from pathlib import Path
from .train import load_cfg

project_root = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from nnlibrary.engines import Trainer

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
    out_dir = Path().cwd().resolve() / cfg.save_path / cfg.dataset_name / cfg.model_config.name
    run = wandb.init(dir=out_dir)
    
    # Apply the curent sweep run settings to the config
    for key, value in run.config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            print(f"Attr {key} could not be set, falling back to config default!")
    
    trainer = Trainer(cfg=cfg)
    trainer.train()
    
    run.log(
        data={
            "test/loss": trainer.info["test_result"]["loss"]
        }
    )
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