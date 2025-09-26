import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import nnlibrary.configs.hvac_mode_classifier as cfg
from nnlibrary.engines import Trainer

# Set environment variables
wandb_key = None

# First try to get from config
if hasattr(cfg, 'wandb_key') and cfg.wandb_key:
    wandb_key = cfg.wandb_key
    if wandb_key: print("The wandb key was read from the config file, it is recommended to put it in a secrets file under '.secrets/wandb' instead for safety!")

# If no key in config, try to read from .secrets/wandb file
if wandb_key is None:
    secrets_file = Path().cwd().resolve() / '.secrets' / 'wandb'
    if secrets_file.exists():
        try:
            wandb_key = secrets_file.read_text().strip()
        except Exception as e:
            print(f"Warning: Could not read WandB key from {secrets_file}: {e}")

# Set the environment variable if we found a key
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key
    print("WandB API key loaded successfully")
else:
    cfg.enable_wandb = False
    print("No WandB API key found - WandB logging has been disabled")

# TODO: This option could be useful with a passed param for how many gpus to use
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use specific GPU

trainer = Trainer(cfg=cfg)

trainer.train()