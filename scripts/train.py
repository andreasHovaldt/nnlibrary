if __name__ == "__main__":
    import sys
    import os
    import importlib
    import importlib.util
    import pkgutil
    import logging
    import argparse
    from types import ModuleType
    from pathlib import Path


    parser = argparse.ArgumentParser(description="Helper script used for training on the defined config file")
    parser.add_argument("-n", "--config-name", type=str, required=True, help="Name of config, either as shorthand 'TCN-reg', as dotted module 'nnlibrary.configs.TCN-reg' or as file path '~/nnlibrary/configs/TCN-reg.py'.")
    parser.add_argument("--logging", action="store_true", help="Whether to display logger prints.")
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help="The logging level used if logging is enabled. Default: INFO")
    args = parser.parse_args()
    
    if args.logging:    
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s | %(filename)s:%(funcName)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            stream=sys.stdout
        )
    
    config_name = str(args.config_name)

    # Add project root to Python path
    project_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(project_root))

    def load_cfg(name: str) -> ModuleType:
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

    cfg = load_cfg(config_name)

    from nnlibrary.engines import Trainer

    if cfg.enable_wandb:
        # Set wandb environment variable
        wandb_key = None

        # First try to get from config
        if hasattr(cfg, 'wandb_key') and getattr(cfg, 'wandb_key'):
            wandb_key = getattr(cfg, 'wandb_key')
            if wandb_key:
                print("The wandb key was read from the config file, it is recommended to put it in a secrets file under '.secrets/wandb' instead for safety!")

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
            if hasattr(cfg, 'enable_wandb'):
                setattr(cfg, 'enable_wandb', False)
            print("No WandB API key found - WandB logging has been disabled")

    # TODO: This option could be useful with a passed param for how many gpus to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use specific GPU

    trainer = Trainer(cfg=cfg)
    trainer.train()