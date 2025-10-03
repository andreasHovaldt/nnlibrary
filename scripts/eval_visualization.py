#!/usr/bin/env python3
"""
Post-training evaluation visualization.

Usage:
    python scripts/eval_visualization.py <config_name> [--split test|val] [--interactive]

Examples:
    python scripts/eval_visualization.py hvac_mode_classifier
    python scripts/eval_visualization.py nnlibrary.configs.hvac_mode_classifier --split val
    python scripts/eval_visualization.py hvac_mode_classifier.py --interactive
"""
from __future__ import annotations

import sys
import os
import importlib
import importlib.util
import pkgutil
from types import ModuleType
from pathlib import Path
from typing import Any, cast

import torch
import matplotlib.pyplot as plt

# Make repo root importable
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


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
        available = [name for _, name, ispkg in pkgutil.iter_modules(configs_pkg.__path__) if not ispkg and not name.startswith("__")]
    except Exception:
        available = []
    raise SystemExit(f"Config '{name}' not found. Available: {', '.join(available) if available else 'no configs discovered'}")


def build_model(cfg: Any) -> torch.nn.Module:
    import nnlibrary.models as models
    model_name = cfg.model_config.name
    model_args = cfg.model_config.args
    if hasattr(models, model_name):
        model_cls = getattr(models, model_name)
        return model_cls(**model_args)
    raise SystemExit(f"Model '{model_name}' not found in nnlibrary.models")


def build_dataset(dl_cfg: Any):
    import nnlibrary.datasets as datasets
    ds_name = dl_cfg.dataset.name
    ds_args = dl_cfg.dataset.args
    if hasattr(datasets, ds_name):
        return getattr(datasets, ds_name)(**ds_args)
    raise SystemExit(f"Dataset '{ds_name}' not found in nnlibrary.datasets")


def build_dataloader(cfg: Any, dl_cfg: Any) -> torch.utils.data.DataLoader:
    from torch.utils.data import DataLoader
    dataset = build_dataset(dl_cfg)

    # Defaults similar to Trainer
    cpu_count = os.cpu_count() or 2
    default_workers = max(1, cpu_count // 2)

    num_workers = getattr(dl_cfg, 'num_workers', None)
    if num_workers is None:
        num_workers = default_workers
    pin_memory_default = (cfg.device == 'cuda')
    pin_memory = getattr(dl_cfg, 'pin_memory', pin_memory_default)
    persistent_workers = getattr(dl_cfg, 'persistent_workers', False)
    if not num_workers:
        persistent_workers = False
    prefetch_factor = getattr(dl_cfg, 'prefetch_factor', 2 if num_workers else None)

    kwargs = dict(
        dataset=dataset,
        batch_size=int(dl_cfg.batch_size),
        shuffle=bool(dl_cfg.shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
    )
    if prefetch_factor is not None and num_workers > 0:
        kwargs['prefetch_factor'] = int(prefetch_factor)
    return DataLoader(**kwargs)


def restore_checkpoint(model: torch.nn.Module, cfg: Any, device: str) -> Path:
    # Resolve save path identical to Trainer logic
    save_dir = PROJECT_ROOT / cfg.save_path / cfg.dataset_name / cfg.model_config.name / "model"
    best_path = save_dir / "model_best.pth"
    last_path = save_dir / "model_last.pth"
    ckpt_path = best_path if best_path.exists() else last_path
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint found at {best_path} or {last_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    return ckpt_path


def _to_numpy(x):
    try:
        import numpy as np  # noqa: F401
    except Exception:
        pass
    if hasattr(x, 'detach'):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            return x.detach().cpu().tolist()
    try:
        import numpy as np
        return np.asarray(x)
    except Exception:
        return x


def run_eval_and_plot(cfg: Any, split: str = "test", interactive: bool = False) -> None:
    from nnlibrary.engines.eval import Evaluator

    device = cfg.device if hasattr(cfg, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
    amp_enable = getattr(cfg, 'amp_enable', True)
    amp_dtype_str = getattr(cfg, 'amp_dtype', 'float16')
    amp_dtype = torch.float16 if amp_dtype_str == 'float16' else torch.bfloat16

    # Build model and restore weights
    model = build_model(cfg).to(device)
    checkpoint_path = restore_checkpoint(model, cfg, device)
    model.eval()

    # Choose split
    if split == 'test':
        dl_cfg = cfg.dataset.test
        detailed = True
    elif split == 'val':
        dl_cfg = cfg.dataset.val
        detailed = True
    else:
        raise SystemExit("--split must be 'test' or 'val'")

    loader = build_dataloader(cfg, dl_cfg)

    # Loss (some Evaluator code expects a loss function, even if we only plot)
    import nnlibrary.utils.loss as loss_mod
    import torch.nn as nn
    lf_name = cfg.loss_fn.name
    lf_args = cfg.loss_fn.args
    if hasattr(loss_mod, lf_name):
        loss_fn = getattr(loss_mod, lf_name)(**lf_args)
    elif hasattr(nn, lf_name):
        loss_fn = getattr(nn, lf_name)(**lf_args)
    else:
        raise SystemExit(f"Loss function '{lf_name}' not found")

    class_names = cfg.dataset.info.get("class_names", [str(i) for i in range(cfg.dataset.info.get("num_classes", 0))])

    evaluator = Evaluator(
        dataloader=loader,
        loss_fn=loss_fn,
        device=device,
        amp_enable=amp_enable,
        amp_dtype=amp_dtype,
        class_names=class_names,
        detailed=detailed,
    )

    with torch.inference_mode():
        result = evaluator.eval(model=model)

    # Plot predicted vs true over sample index (proxy for time)
    y_true = result.get("y_true_seq")
    y_pred = result.get("y_pred_seq")
    if y_true is None or y_pred is None:
        print("Detailed sequences not returned; nothing to plot.")
        return

    # Convert to numpy for plotting
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    import numpy as np
    t = np.arange(len(y_true_np))

    # Save figures next to model artifacts
    fig_dir = PROJECT_ROOT / cfg.save_path / cfg.dataset_name / cfg.model_config.name / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    base_title = f"{cfg.model_config.name} | {cfg.dataset_name} | split={split} | ckpt={Path(checkpoint_path).name}"

    if interactive:
        # Plotly HTML export (no fallback; require plotly if flag is used)
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise SystemExit("Plotly is required for --interactive. Install with: pip install plotly")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_true_np, name="true", mode="lines", line=dict(width=1.2)))
        fig.add_trace(go.Scatter(x=t, y=y_pred_np, name="pred", mode="lines", line=dict(width=1.0), line_shape="hv"))
        fig.update_layout(title=base_title, xaxis_title="sample index", yaxis_title="class id")
        out_html = fig_dir / f"pred_vs_true_{split}.html"
        fig.write_html(out_html, include_plotlyjs="cdn")
        print(f"Saved interactive plot to: {out_html}")

    # Matplotlib PNG export
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(t, y_true_np, where='post', label='true', linewidth=1.2, alpha=0.9)
    ax.step(t, y_pred_np, where='post', label='pred', linewidth=1.0, alpha=0.9)
    ax.set_xlabel('sample index')
    ax.set_ylabel('class id')
    ax.set_title(base_title)
    ax.legend(loc='upper right')
    fig.tight_layout()
    out_png = fig_dir / f"pred_vs_true_{split}.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved plot as PNG to: {out_png}")


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    config_name = argv[1]
    split = 'test'
    interactive = False

    # Lightweight arg parsing
    args = argv[2:]
    i = 0
    while i < len(args):
        a = args[i]
        if a in ('--split', '-s'):
            if i + 1 >= len(args):
                print("Missing value for --split; expected 'test' or 'val'")
                sys.exit(1)
            split = args[i + 1]
            i += 2
            continue
        if a == '--interactive':
            interactive = True
            i += 1
            continue
        print(f"Unknown argument: {a}")
        sys.exit(1)

    cfg = load_cfg(config_name)
    run_eval_and_plot(cfg, split=split, interactive=interactive)


if __name__ == '__main__':
    main(sys.argv)
