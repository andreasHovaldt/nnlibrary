"""
Model Analyzer Script
=====================
Loads a model from a config file and displays a summary using torchinfo.

Usage:
    python dumpster/model_analyzer.py -n TCN-reg
    python dumpster/model_analyzer.py -n MLP-reg.py --depth 3
    python dumpster/model_analyzer.py -n transformer-cls --device cuda
"""

import sys
import torch
import pkgutil
import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Tuple, Optional
from torchinfo import ModelStatistics, summary

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))


# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(
    description="Analyze a model defined in a config file using torchinfo summary"
)
parser.add_argument(
    "-n", "--config-name",
    type=str,
    required=True,
    help="Name of config: shorthand 'TCN-reg', dotted 'nnlibrary.configs.TCN-reg', or path 'nnlibrary/configs/TCN-reg.py'"
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    choices=["cpu", "cuda"],
    help="Device to place the model on (default: cpu)"
)
parser.add_argument(
    "--depth",
    type=int,
    default=4,
    help="Depth of the model summary (default: 4)"
)
parser.add_argument(
    "--mode",
    type=str,
    default='eval',
    choices=['train','eval',],
    help="The mode of the model. Equal to either passing pytorch model as model.train() or model.eval(). (default: 'eval')"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="The batch size of the dummy input (default: 1)"
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Show additional model details"
)


# -----------------------------
# Config Loading
# -----------------------------
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
        available = [n for _, n, ispkg in pkgutil.iter_modules(configs_pkg.__path__) if not ispkg and not n.startswith("__")]
    except Exception:
        available = []
    raise SystemExit(f"Config '{name}' not found. Available: {', '.join(available) if available else 'no configs discovered'}")


# -----------------------------
# Helper Functions
# -----------------------------
def _extract_name_args(obj: Any) -> Tuple[str, dict]:
    """Extract model name and args from config object."""
    if hasattr(obj, 'name') and hasattr(obj, 'args'):
        return getattr(obj, 'name'), getattr(obj, 'args')
    if isinstance(obj, dict) and 'name' in obj and 'args' in obj:
        return obj['name'], obj['args']
    raise SystemExit(f"Unsupported config format: {type(obj)}. Expected attrs 'name'/'args' or a dict with those keys.")


def build_model(cfg: Any) -> torch.nn.Module:
    """Build a model from config."""
    import nnlibrary.models as models
    model_name, model_args = _extract_name_args(cfg.model_config)
    if hasattr(models, model_name):
        model_cls = getattr(models, model_name)
        return model_cls(**model_args)
    raise SystemExit(f"Model '{model_name}' not found in nnlibrary.models")


def infer_input_shape(cfg: Any, batch_size: int = 1) -> Tuple[int, ...]:
    """Infer the input shape from the config's model_config."""
    _, model_args = _extract_name_args(cfg.model_config)
    
    # Common patterns in model configs
    window_size = model_args.get('window_size', model_args.get('seq_len', 32))
    feature_dim = model_args.get('feature_dim', model_args.get('input_dim', model_args.get('d_input', 34)))
    
    # Return shape as (batch_size, sequence_length, features)
    return (batch_size, window_size, feature_dim)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config: {args.config_name}")
    cfg = load_cfg(args.config_name)
    
    # Build model
    model_name, model_args = _extract_name_args(cfg.model_config)
    print(f"Building model: {model_name}")
    model = build_model(cfg)
    model = model.to(args.device)
    model.eval()
    
    # Infer input shape
    input_shape = infer_input_shape(cfg, args.batch_size)
    print(f"Input shape: {input_shape}")
    
    # Print verbose model args if requested
    if args.verbose:
        print("\nModel Arguments:")
        for key, value in model_args.items():
            print(f"  {key}: {value}")
        print()
    
    print("\n" + "=" * 60)
    print("Model Eval Summary (torchinfo)")
    print("=" * 60)
    summary(
        model,
        input_size=input_shape,
        device=args.device,
        depth=args.depth,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        row_settings=["var_names"],
        mode=args.mode,
    )
    
    # Additional model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 60)
    print("Manual Parameter Summary")
    print("=" * 60)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print(f"Model size (MB):      {total_params * 4 / 1024 / 1024:.2f} (assuming float32)")
