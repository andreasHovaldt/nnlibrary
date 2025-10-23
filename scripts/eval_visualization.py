#!/usr/bin/env python3

from __future__ import annotations

import sys
import os
import importlib
import importlib.util
import pkgutil
import argparse
from types import ModuleType
from pathlib import Path
from typing import Any, Tuple

import torch
import matplotlib.pyplot as plt

# Make repo root importable
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Arg parser
parser = argparse.ArgumentParser(description="Helper script used for visualizing the evaluation of a trained model")
parser.add_argument("-n", "--config-name", type=str, required=True, help="Name of config, either as shorthand 'TCN-reg', as dotted module 'nnlibrary.configs.TCN-reg' or as file path '~/nnlibrary/configs/TCN-reg.py'.")
parser.add_argument("-s", "--split", type=str, default="test", choices=["train", "val", "test"], help="The split to evaluate the model on. Default: 'test'")
parser.add_argument("--interactive", action="store_true", help="Make an interactive plotly plot.")


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


def _extract_name_args(obj: Any) -> Tuple[str, dict]:
    """Extract (name, args) from either a BaseConfig-like object or a plain dict."""
    if hasattr(obj, 'name') and hasattr(obj, 'args'):
        return getattr(obj, 'name'), getattr(obj, 'args')
    if isinstance(obj, dict) and 'name' in obj and 'args' in obj:
        return obj['name'], obj['args']
    raise SystemExit(f"Unsupported config format: {type(obj)}. Expected attrs 'name'/'args' or a dict with those keys.")


def build_model(cfg: Any) -> torch.nn.Module:
    import nnlibrary.models as models
    model_name, model_args = _extract_name_args(cfg.model_config)
    if hasattr(models, model_name):
        model_cls = getattr(models, model_name)
        return model_cls(**model_args)
    raise SystemExit(f"Model '{model_name}' not found in nnlibrary.models")


def build_dataset(cfg: Any, dl_cfg: Any):
    import nnlibrary.datasets as datasets
    # Support BaseConfig or dict in dl_cfg.dataset
    ds_name, ds_args = _extract_name_args(dl_cfg.dataset)
    # If task is available, ensure consistency with dataset args if not set
    if 'task' not in ds_args and hasattr(cfg, 'task'):
        ds_args = {**ds_args, 'task': cfg.task}
    if hasattr(datasets, ds_name):
        return getattr(datasets, ds_name)(**ds_args)
    raise SystemExit(f"Dataset '{ds_name}' not found in nnlibrary.datasets")


def build_dataloader(cfg: Any, dl_cfg: Any) -> torch.utils.data.DataLoader:
    from torch.utils.data import DataLoader
    dataset = build_dataset(cfg, dl_cfg)

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

    # Resolve batch size like Trainer (fall back to cfg.eval_batch_size)
    bs = getattr(dl_cfg, 'batch_size', None)
    if bs is None and hasattr(cfg, 'eval_batch_size'):
        bs = int(cfg.eval_batch_size)
    elif bs is None:
        bs = 512

    kwargs = dict(
        dataset=dataset,
        batch_size=int(bs),
        shuffle=bool(dl_cfg.shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
    )
    if prefetch_factor is not None and num_workers > 0:
        kwargs['prefetch_factor'] = int(prefetch_factor)
    return DataLoader(**kwargs)


def _get_model_name(cfg: Any) -> str:
    name, _ = _extract_name_args(cfg.model_config)
    return name


def restore_checkpoint(model: torch.nn.Module, cfg: Any, device: str) -> Path:
    # Resolve save path identical to Trainer logic
    save_dir = PROJECT_ROOT / cfg.save_path / cfg.dataset_name / _get_model_name(cfg) / "model"
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
    from nnlibrary.engines.eval import ClassificationEvaluator

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
    elif split == 'train':
        dl_cfg = cfg.dataset.train
        detailed = True
    else:
        raise SystemExit("--split must be 'test', 'val' or 'train'")

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

    # Branch on task type
    task = getattr(cfg, 'task', 'undefined')

    # Save figures next to model artifacts
    fig_dir: Path = PROJECT_ROOT / cfg.save_path / cfg.dataset_name / _get_model_name(cfg) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation will be saved to: {fig_dir.resolve()}")
    base_title = f"{_get_model_name(cfg)} | {cfg.dataset_name} | split={split} | ckpt={Path(checkpoint_path).name}"

    if task == 'classification':
        from nnlibrary.engines.eval import ClassificationEvaluator
        class_names = cfg.dataset.info.get("class_names", [str(i) for i in range(cfg.dataset.info.get("num_classes", 0))])
        evaluator = ClassificationEvaluator(
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

        y_true = result.get("y_true")
        y_pred = result.get("y_pred")
        if y_true is None or y_pred is None:
            print("Detailed sequences not returned; nothing to plot.")
            return
        y_true_np = _to_numpy(y_true)
        y_pred_np = _to_numpy(y_pred)
        import numpy as np
        t = np.arange(len(y_true_np))

        # Interactive (optional)
        if interactive:
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

    elif task == 'regression':
        from nnlibrary.engines.eval import RegressionEvaluator
        output_names = cfg.dataset.info.get("class_names")
        evaluator = RegressionEvaluator(
            dataloader=loader,
            loss_fn=loss_fn,
            device=device,
            amp_enable=amp_enable,
            amp_dtype=amp_dtype,
            output_names=output_names,
            detailed=True,
            inverse_transform=None,
        )
        with torch.inference_mode():
            result = evaluator.eval(model=model)

        y_true = result.get("y_true_seq")
        y_pred = result.get("y_pred_seq")
        if y_true is None or y_pred is None:
            print("Detailed sequences not returned; nothing to plot.")
            return
        y_true_np = _to_numpy(y_true)
        y_pred_np = _to_numpy(y_pred)
        import numpy as np
        t = np.arange(y_true_np.shape[0])
        num_outputs = 1 if y_true_np.ndim == 1 else int(y_true_np.shape[1])
        names = output_names if (isinstance(output_names, list) and len(output_names) == num_outputs) else [f"Output {i}" for i in range(num_outputs)]

        # Save an individual PNG per output feature
        def _safe_name(s: str) -> str:
            return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)

        for i in range(num_outputs):
            y_t = y_true_np[:, i] if num_outputs > 1 else y_true_np
            y_p = y_pred_np[:, i] if num_outputs > 1 else y_pred_np
            name_i = names[i]
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(t, y_t, label='true', linewidth=1.2, alpha=0.9)
            ax.plot(t, y_p, label='pred', linewidth=1.0, alpha=0.9)
            ax.set_title(f"{base_title} | {name_i}")
            ax.set_xlabel('sample index')
            ax.set_ylabel('value')
            ax.legend(loc='upper right')
            fig.tight_layout()
            out_png = fig_dir / f"pred_vs_true_{split}_{i:02d}_{_safe_name(str(name_i))}.png"
            fig.savefig(out_png)
            plt.close(fig)
            print(f"Saved regression plot for '{name_i}' to: {out_png}")

        if interactive:
            try:
                import plotly.graph_objects as go
                from math import ceil
                # Build a small grid interactively (2 columns by default)
                cols = 2 if num_outputs > 1 else 1
                rows = int(np.ceil(num_outputs / cols))
                import plotly.subplots as sp
                fig = sp.make_subplots(rows=rows, cols=cols, subplot_titles=names)
                for i in range(num_outputs):
                    r = i // cols + 1
                    c = i % cols + 1
                    y_t = y_true_np[:, i] if num_outputs > 1 else y_true_np
                    y_p = y_pred_np[:, i] if num_outputs > 1 else y_pred_np
                    fig.add_trace(go.Scatter(x=t, y=y_t, name=f"true-{names[i]}", mode="lines", line=dict(width=1.2)), row=r, col=c)
                    fig.add_trace(go.Scatter(x=t, y=y_p, name=f"pred-{names[i]}", mode="lines", line=dict(width=1.0)), row=r, col=c)
                fig.update_layout(title=base_title, showlegend=False, height=350 * rows, width=700 * cols)
                out_html = fig_dir / f"pred_vs_true_{split}.html"
                fig.write_html(out_html, include_plotlyjs="cdn")
                print(f"Saved interactive regression plots to: {out_html}")
            except ImportError:
                print("Plotly not installed; skipping interactive regression plots.")

    else:
        raise SystemExit(f"Unknown task '{task}'. Expected 'classification' or 'regression'.")


def main() -> None:
    args = parser.parse_args()

    config_name = args.config_name
    split = args.split
    interactive = args.interactive

    cfg = load_cfg(config_name)
    run_eval_and_plot(cfg, split=split, interactive=interactive)


if __name__ == '__main__':
    main()
