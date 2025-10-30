#!/usr/bin/env python3
"""
Export a trained model checkpoint to ONNX.

Examples:
  # Export latest checkpoint for a run
  python scripts/export_onnx_model.py -n TCN-reg -r witty-salamander-4

  # Export with dynamic batch and sequence axes and a custom opset
  python scripts/export_onnx_model.py -n TCN-reg -r witty-salamander-4 \
      --dynamic-batch --dynamic-seq --opset 17

  # Force sequence length for dummy input if not in config
  python scripts/export_onnx_model.py -n TCN-reg -r witty-salamander-4 --seq-len 32
"""

from __future__ import annotations

import sys
import onnx
import torch
import pkgutil
import argparse
import importlib
import importlib.util
from torch.export import Dim

from pathlib import Path
from typing import Any, Tuple, Optional



PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------
# Config loading (mirrors eval_visualization)
# -----------------------------

def load_cfg(name: str):
    """Load a config module by short name, dotted path, or file path."""
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
    # 3) Treat as file path under nnlibrary/configs
    cfg_path = Path(name)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / "nnlibrary" / "configs" / cfg_path
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
        available = [n for _, n, ispkg in pkgutil.iter_modules(configs_pkg.__path__) if not ispkg and not n.startswith("__")]
    except Exception:
        available = []
    raise SystemExit(f"Config '{name}' not found. Available: {', '.join(available) if available else 'no configs discovered'}")


def _extract_name_args(obj: Any) -> Tuple[str, dict]:
    if hasattr(obj, 'name') and hasattr(obj, 'args'):
        return getattr(obj, 'name'), getattr(obj, 'args')
    if isinstance(obj, dict) and 'name' in obj and 'args' in obj:
        return obj['name'], obj['args']
    raise SystemExit(f"Unsupported config format: {type(obj)}. Expected attrs 'name'/'args' or a dict with those keys.")


def _get_model_name(cfg: Any) -> str:
    name, _ = _extract_name_args(cfg.model_config)
    return name


def _resolve_run_dirs(cfg: Any, run_name: str) -> Tuple[Path, Path, Path]:
    """Return (run_dir, model_dir, onnx_dir) consistent with Trainer.
    run_dir = <root>/exp/<dataset>/<model>/<run_name>
    """
    model_name = _get_model_name(cfg)
    run_dir = PROJECT_ROOT / cfg.save_path / cfg.dataset_name / model_name / run_name
    model_dir = run_dir / 'model'
    onnx_dir = run_dir / 'onnx'
    onnx_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, model_dir, onnx_dir


def _restore_checkpoint(model: torch.nn.Module, cfg: Any, device: str, run_name: str) -> Path:
    _, model_dir, _ = _resolve_run_dirs(cfg, run_name)
    best_path = model_dir / 'model_best.pth'
    last_path = model_dir / 'model_last.pth'
    ckpt_path = best_path if best_path.exists() else last_path
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint found at {best_path} or {last_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict)
    return ckpt_path


# -----------------------------
# Model building
# -----------------------------

def build_model(cfg: Any) -> torch.nn.Module:
    import nnlibrary.models as models
    model_name, model_args = _extract_name_args(cfg.model_config)
    if hasattr(models, model_name):
        model_cls = getattr(models, model_name)
        return model_cls(**model_args)
    raise SystemExit(f"Model '{model_name}' not found in nnlibrary.models")


# -----------------------------
# Dummy input inference
# -----------------------------

def infer_dummy_input(
    cfg: Any,
    model_name: str,
    model_args: dict,
    batch_size_override: Optional[int] = None,
    feature_dim_override: Optional[int] = None,
    seq_len_override: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """Infer a plausible dummy input tensor using config metadata when available.

    Priority:
      - feature_dim: cfg.dataset_metadata['dataset_info']['feature_dim'] or cfg.model_config.args['input_dim']
      - window/seq_len: cfg.dataset_metadata['temporal_settings']['window'] or cfg.model_config.args['sequence_length'|'window_size'|'seq_len'|'max_seq_length'] or --seq-len
      - batch size: cfg.eval_batch_size, else cfg.train_batch_size, else 1

    Returns (input_tensor, shape_info) where shape_info has: layout ('BTC' or 'BCT'), B, L, C.
    """
    # Defaults
    L: Optional[int] = None
    C: Optional[int] = None
    B: Optional[int] = None

    # Try to read dataset metadata embedded in config module
    metadata = getattr(cfg, 'dataset_metadata', None)
    if metadata is None:
        # Best-effort: try reading from data_root/stats/metadata.json if present
        try:
            from pathlib import Path as _P
            import json as _json
            data_root = getattr(cfg, 'data_root', None)
            if data_root is not None:
                meta_path = _P(data_root) / 'stats' / 'metadata.json'
                if meta_path.exists():
                    metadata = _json.loads(meta_path.read_text())
        except Exception:
            metadata = None

    # Feature dimension
    if feature_dim_override is not None:
        C = int(feature_dim_override)
    if metadata is not None:
        try:
            C = int(metadata['dataset_info']['feature_dim'])
        except Exception:
            C = None
    if C is None:
        for key in ('input_dim', 'feature_dim', 'in_channels'):
            if key in model_args and isinstance(model_args[key], int):
                C = int(model_args[key])
                break
    if C is None:
        C = 32

    # Sequence/window length
    if seq_len_override is not None:
        L = int(seq_len_override)
    if L is None and metadata is not None:
        try:
            L = int(metadata['temporal_settings']['window'])
        except Exception:
            L = None
    if L is None:
        for key in ('sequence_length', 'window_size', 'seq_len', 'max_seq_length'):
            if key in model_args and isinstance(model_args[key], int):
                L = int(model_args[key])
                break
    if L is None:
        L = 32

    # Batch size: override > eval > train > 1
    if batch_size_override is not None:
        B = int(batch_size_override)
    elif hasattr(cfg, 'eval_batch_size') and isinstance(getattr(cfg, 'eval_batch_size'), int):
        B = int(getattr(cfg, 'eval_batch_size'))
    elif hasattr(cfg, 'train_batch_size') and isinstance(getattr(cfg, 'train_batch_size'), int):
        B = int(getattr(cfg, 'train_batch_size'))
    else:
        B = 1

    # Determine layout
    # - Most models here use (B, L, C)
    # - TCNResidualBlock (standalone) expects (B, C, L)
    layout = 'BTC'
    if model_name == 'TCNResidualBlock':
        layout = 'BCT'

    shape_info = dict(layout=layout, B=B, L=L, C=C)

    if layout == 'BTC':
        x = torch.randn((B, L, C), dtype=torch.float32)
    else:
        x = torch.randn((B, C, L), dtype=torch.float32)

    return x, shape_info


def build_dynamic_signatures(
    shape_info: dict,
    input_name: str,
    dynamic_batch: bool,
    dynamic_seq: bool,
) -> dict:
    """Build dynamo dynamic_shapes."""

    # Dynamo dynamic_shapes: map input name to Dim symbols
    dyn_shapes: dict = {}
    if dynamic_batch or dynamic_seq:
        dims: dict[int, Dim] = {}
        if dynamic_batch:
            dims[0] = Dim('batch')
        if dynamic_seq:
            if shape_info['layout'] == 'BTC':
                dims[1] = Dim('seq_len')
            else:
                dims[2] = Dim('seq_len')
        dyn_shapes[input_name] = dims

    return dyn_shapes


# -----------------------------
# Export
# -----------------------------

def export_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    save_path: Path,
    dynamic_shapes: Optional[dict],
    input_name: str = 'input',
    output_name: str = 'output',
    train_mode: bool = False,
) -> None:
    # Set desired mode for export
    if train_mode:
        model.train()
    else:
        model.eval()
    # Dynamo exporter path on CPU; prefer dynamic_shapes over dynamic_axes
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (example_input,),
        f=str(save_path),
        input_names=[input_name],
        output_names=[output_name],
        dynamo=True,
        dynamic_shapes=dynamic_shapes if dynamic_shapes else None,
    )


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained model checkpoint to ONNX")
    parser.add_argument('-n', '--config-name', required=True, type=str,
                        help="Config name: short (e.g. 'TCN-reg'), dotted module, or path to a .py file")
    parser.add_argument('-r', '--run_name', required=True, type=str, help="Run directory name (e.g. 'witty-salamander-4', or the path from the model directory, sweeps/TCN-reg-sweep/neat-reindeer-1)")
    parser.add_argument('--filename', type=str, default=None, help="Output filename (defaults to <ModelName>.onnx)")
    
    parser.add_argument('--train-mode', action='store_true', help="Export the model in training mode (default: eval mode)")
    
    parser.add_argument('--batch-size', type=int, default=None, help="Override dummy input batch size (optional; default from config)")
    parser.add_argument('--feature-dim', type=int, default=None, help="Override dummy input feature dimension (optional; default from metadata/config)")
    parser.add_argument('--seq-len', type=int, default=None, help="Override dummy input sequence length (optional; default from metadata/config)")
    
    parser.add_argument('--dynamic-batch', action='store_true', help="Mark batch dimension as dynamic")
    parser.add_argument('--dynamic-seq', action='store_true', help="Mark sequence length as dynamic (if applicable)")
    
    args = parser.parse_args()

    cfg = load_cfg(args.config_name)

    # Build and restore model
    model = build_model(cfg)
    # Always export on CPU for portability
    model.to('cpu')
    ckpt_path = _restore_checkpoint(model, cfg, 'cpu', args.run_name)
    print(f"Loaded checkpoint: {ckpt_path}")

    # Prepare dummy input on CPU for export stability
    model_name, model_args = _extract_name_args(cfg.model_config)
    example_input, shape_info = infer_dummy_input(
        cfg=cfg,
        model_name=model_name,
        model_args=model_args,
        batch_size_override=args.batch_size,
        feature_dim_override=args.feature_dim,
        seq_len_override=args.seq_len,
    )
    example_input = example_input.to('cpu')
    model.to('cpu').eval()

    # Build dynamic axes
    input_name = 'input'
    output_name = 'output'
    dynamic_shapes = build_dynamic_signatures(
        shape_info, input_name, args.dynamic_batch, args.dynamic_seq
    )

    # Resolve save path near the run directory (keep artifacts together)
    _, _, onnx_dir = _resolve_run_dirs(cfg, args.run_name)
    fname = args.filename or f"{model.__class__.__name__}.onnx"
    save_path = onnx_dir / fname

    # Export
    print(f"Exporting ONNX to: {save_path}")
    export_onnx(model, example_input, save_path, dynamic_shapes, input_name, output_name, train_mode=args.train_mode)

    # Always validate the exported ONNX
    try:
        m = onnx.load(str(save_path))
        onnx.checker.check_model(m)
        print("ONNX model check: OK")
    except Exception as e:
        raise SystemExit(f"onnx.checker failed: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
