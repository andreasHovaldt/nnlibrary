#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import pkgutil
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import torch
from torch.amp.grad_scaler import GradScaler

# Ensure project root is on path
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


def resolve_model(cfg: ModuleType) -> torch.nn.Module:
    """Instantiate model from cfg.model_config using nnlibrary.models registry.

    Expects cfg.model_config to be a dict-like with keys: name, args.
    """
    if not hasattr(cfg, "model_config"):
        raise SystemExit("Config is missing 'model_config' with keys {'name','args'}")
    model_config: Dict[str, Any] = dict(getattr(cfg, "model_config"))
    model_name: str = model_config["name"]
    model_args = model_config.get("args", {})
    if not isinstance(model_args, dict):
        # Some configs may use SimpleNamespace/BaseConfig; convert to dict
        model_args = dict(model_args)

    from nnlibrary import models as model_registry

    if not hasattr(model_registry, model_name):
        available = [k for k in dir(model_registry) if not k.startswith("_")]
        raise SystemExit(f"Model '{model_name}' not found in nnlibrary.models. Available: {', '.join(available)}")

    ModelCls = getattr(model_registry, model_name)
    return ModelCls(**model_args)


def pick_device(cfg: ModuleType) -> torch.device:
    if hasattr(cfg, "device"):
        dev_str = getattr(cfg, "device")
        if dev_str == "cuda" and not torch.cuda.is_available():
            logging.warning("Config requested CUDA but CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(dev_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_amp_dtype(cfg: ModuleType) -> torch.dtype:
    # Accept values like "float16", "fp16", "bfloat16", "bf16"
    dt_str = getattr(cfg, "amp_dtype", "float16").lower()
    if dt_str in ("float16", "fp16", "half"):
        return torch.float16
    if dt_str in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise SystemExit(f"Unsupported amp_dtype '{getattr(cfg, 'amp_dtype', None)}' in config. Use 'float16' or 'bfloat16'.")


def set_seed_if_any(cfg: ModuleType) -> None:
    seed = getattr(cfg, "seed", None)
    if seed is None:
        return
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def infer_shapes_from_args(model_args: Dict[str, Any]) -> tuple[int, int]:
    """Return (sequence_length, input_dim) from common arg keys with fallbacks."""
    seq_len = (
        model_args.get("sequence_length")
        or model_args.get("max_seq_length")
        or 32
    )
    in_dim = (
        model_args.get("input_dim")
        or model_args.get("feature_dim")
        or 32
    )
    return int(seq_len), int(in_dim)


def benchmark_training(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int, warmup: int, lr: float, amp_dtype: torch.dtype, device: torch.device) -> tuple[float, float]:
    """Return (fp32_time, amp_time). If AMP not supported on device, amp_time=nan."""
    loss_fn = torch.nn.MSELoss()

    def run(use_amp: bool) -> float:
        model_to_run = model
        model_to_run.train()
        opt = torch.optim.AdamW(model_to_run.parameters(), lr=lr, fused=True if device.type == "cuda" else False)
        scaler = GradScaler(device=str(device), enabled=(use_amp and device.type == "cuda" and amp_dtype == torch.float16))

        # Warmup
        for _ in range(warmup):
            opt.zero_grad(set_to_none=True)
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model_to_run(x)
                    loss = loss_fn(out, y)
                if amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            else:
                out = model_to_run(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model_to_run(x)
                    loss = loss_fn(out, y)
                if amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            else:
                out = model_to_run(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start

    # Clone initial weights for fair comparison
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # FP32
    model.load_state_dict(init_state)
    t_fp32 = run(use_amp=False)

    # AMP (CUDA only)
    if device.type == "cuda":
        model.load_state_dict(init_state)
        t_amp = run(use_amp=True)
    else:
        t_amp = float("nan")

    return t_fp32, t_amp


def main() -> None:
    parser = argparse.ArgumentParser(description="Test AMP speed for a config-defined model using synthetic data.")
    parser.add_argument("-n", "--config-name", required=True, type=str, help="Config name: short ('TCN-reg'), dotted module ('nnlibrary.configs.TCN-reg'), or a path to a config file.")
    parser.add_argument("--steps", type=int, default=1000, help="Timed training steps per mode (FP32/AMP). Default: 1000")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps before timing. Default: 100")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size; defaults to cfg.train_batch_size or 32.")
    args = parser.parse_args()

    cfg = load_cfg(args.config_name)
    set_seed_if_any(cfg)

    device = pick_device(cfg)
    amp_enabled = bool(getattr(cfg, "amp_enable", True))
    amp_dtype = parse_amp_dtype(cfg)

    # Instantiate model
    model = resolve_model(cfg)
    model.to(device)

    # Infer shapes and synthesize data
    model_args = dict(getattr(cfg, "model_config").get("args", {}))
    seq_len, in_dim = infer_shapes_from_args(model_args)
    batch_size = args.batch_size or int(getattr(cfg, "train_batch_size", 32))

    x = torch.randn(batch_size, seq_len, in_dim, device=device)
    # Probe output shape to build a matching target for MSE
    with torch.no_grad():
        model.eval()
        y_probe = model(x)
    target = torch.randn_like(y_probe)

    # Report environment
    print(f"using device: {device}")
    if device.type == "cuda":
        tf32_ok = torch.backends.cuda.matmul.allow_tf32
        f32_prec = torch.get_float32_matmul_precision()
        print(f"TF32 allow: {tf32_ok} | float32 matmul precision: {f32_prec}")
    print(f"AMP requested in cfg: {amp_enabled} | AMP dtype: {amp_dtype}")

    # Benchmark
    t_fp32, t_amp = benchmark_training(model, x, target, steps=args.steps, warmup=args.warmup, lr=float(getattr(cfg, "lr", 1e-3)), amp_dtype=amp_dtype, device=device)

    tokens = batch_size * seq_len * args.steps
    print(f"Train FP32 - time: {t_fp32:.2f}s | throughput: {tokens/t_fp32:.0f} tok/s")
    if device.type == "cuda":
        print(f"Train AMP  - time: {t_amp:.2f}s | throughput: {tokens/t_amp:.0f} tok/s | speedup vs FP32: {t_fp32/t_amp:.2f}x")
    else:
        print("Train AMP  - skipped (CUDA not available)")


if __name__ == "__main__":
    main()
