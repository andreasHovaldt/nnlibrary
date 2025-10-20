import os
import sys
import wandb
import importlib
from pathlib import Path

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



def main():
    config_name = "TCN-reg"
    cfg = importlib.import_module(f"nnlibrary.configs.{config_name}")
    out_dir = (Path(__file__).parent / 'sweep').resolve()

    # Important: For sweeps, the agent provides project/name; ignore custom project warnings
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


# 2: Define the search space
sweep_configuration = {
    "name": "sweep-demo",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "test/loss"},
    "parameters": {
        "num_epochs": {"values": [5, 10]},
        "lr": {"values": [1e-2, 1e-3, 1e-4]},
        "train_batch_size": {"values": [256, 512, 1024]},
    },
}


if __name__ == "__main__":
    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
    wandb.agent(sweep_id, function=main, count=18)
    wandb.finish()