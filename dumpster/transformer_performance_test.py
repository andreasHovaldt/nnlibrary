from pathlib import Path
import sys; sys.path.append(str(Path(__file__).parent.parent))

import time
import argparse
import random
import torch
from torch.amp.grad_scaler import GradScaler
import numpy as np
from typing import TypedDict, Literal
from nnlibrary.models import TransformerRegression, TransformerRegression

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
set_seed()

# Types for constructor kwargs
class TransformerHyperparameters(TypedDict):
    input_dim: int
    output_dim: int
    dim_model: int
    num_heads: int
    num_layers: int
    dim_ff: int
    max_seq_length: int
    dropout: float
    pooling: Literal['last', 'mean', 'cls']

# CLI args to align with standalone AMP test behavior
parser = argparse.ArgumentParser(description="Transformer training AMP benchmark (basic vs optimized)")
parser.add_argument("--steps", type=int, default=1000, help="Timed training steps per mode (FP32/AMP). Default: 1000")
parser.add_argument("--warmup", type=int, default=100, help="Warmup steps before timing. Default: 100")
parser.add_argument("--batch-size", type=int, default=512, help="Batch size. Default: 32")
parser.add_argument("--seq-len", type=int, default=32, help="Sequence length (window). Default: 32")
parser.add_argument("--amp-dtype", choices=["auto","float16","bfloat16"], default="float16", help="AMP dtype to use. Default: float16")
parser.add_argument("--disable-tf32", action="store_true", help="Disable TF32 for a true FP32 baseline (sets highest precision)")
args = parser.parse_args()

# Dataset/model settings
INPUT_DIM = 37
OUTPUT_DIM = 5
SEQUENCE_LENGTH = int(args.seq_len)
BATCH_SIZE = int(args.batch_size)

# Test settings
TORCH_COMPILE = False
STEPS = int(args.steps)
WARMUP_STEPS = int(args.warmup)
LR = 1e-3
if args.amp_dtype == "float16":
    AMP_DTYPE = torch.float16
elif args.amp_dtype == "bfloat16":
    AMP_DTYPE = torch.bfloat16
else:
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# Model args
transformer_hyperparameters: TransformerHyperparameters = {
    'input_dim': INPUT_DIM,
    'output_dim': OUTPUT_DIM,
    'dim_model': 64,
    'num_heads': 4,
    'num_layers': 2,
    'dim_ff': 256,
    'max_seq_length': 32,
    'dropout': 0.1,
    'pooling': 'last',
}

# Basic implementation
try:
    model_basic = TransformerRegression(**transformer_hyperparameters)
    if TORCH_COMPILE:
        model_basic: TransformerRegression = torch.compile(model_basic) # type: ignore
except RuntimeError:
    print('Failed to compile model!')
    model_basic = TransformerRegression(**transformer_hyperparameters)
    
# Optimized implementation
try: 
    model_opt = TransformerRegression(**transformer_hyperparameters)
    if TORCH_COMPILE: 
        model_opt: TransformerRegression = torch.compile(model_opt)  # type: ignore
except RuntimeError:
    print('Failed to compile model!')
    model_opt = TransformerRegression(**transformer_hyperparameters)

# Benchmark
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device: ', device)
if device.type == 'cuda' and args.disable_tf32:
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision('highest')
    except Exception:
        pass
if device.type == 'cuda':
    print(f"TF32 allow: {torch.backends.cuda.matmul.allow_tf32} | float32 matmul precision: {torch.get_float32_matmul_precision()}")
print(f"AMP dtype: {AMP_DTYPE}")
model_basic.to(device)
model_opt.to(device)
x = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_DIM, device=device)
target = torch.randn(BATCH_SIZE, OUTPUT_DIM, device=device)

model_basic.train()
model_opt.train()

# Cache initial weights so each benchmark starts from the same point
init_state_basic = {k: v.detach().clone() for k, v in model_basic.state_dict().items()}
init_state_opt = {k: v.detach().clone() for k, v in model_opt.state_dict().items()}

def train_benchmark(model: torch.nn.Module, steps: int, warmup: int, use_amp: bool) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True if device.type == 'cuda' else False)
    scaler = GradScaler(device=str(device), enabled=use_amp and AMP_DTYPE == torch.float16 and device.type == 'cuda')

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                y = model(x)
                loss = torch.nn.functional.mse_loss(y, target)
            if AMP_DTYPE == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            y = model(x)
            loss = torch.nn.functional.mse_loss(y, target)
            loss.backward()
            optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                y = model(x)
                loss = torch.nn.functional.mse_loss(y, target)
            if AMP_DTYPE == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            y = model(x)
            loss = torch.nn.functional.mse_loss(y, target)
            loss.backward()
            optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter() - start

# -------- Training benchmarks --------
# FP32
model_basic.load_state_dict(init_state_basic)
model_opt.load_state_dict(init_state_opt)
basic_fp32_time = train_benchmark(model_basic, STEPS, WARMUP_STEPS, use_amp=False)
opt_fp32_time = train_benchmark(model_opt, STEPS, WARMUP_STEPS, use_amp=False)
print(f"Train FP32 - Basic: {basic_fp32_time:.2f}s | Optimized: {opt_fp32_time:.2f}s | Speedup: {basic_fp32_time / opt_fp32_time:.2f}x")

# AMP (CUDA only)
basic_amp_time = None
opt_amp_time = None
if device.type == 'cuda':
    model_basic.load_state_dict(init_state_basic)
    model_opt.load_state_dict(init_state_opt)
    basic_amp_time = train_benchmark(model_basic, STEPS, WARMUP_STEPS, use_amp=True)
    opt_amp_time = train_benchmark(model_opt, STEPS, WARMUP_STEPS, use_amp=True)
    print(f"Train AMP  - Basic: {basic_amp_time:.2f}s | Optimized: {opt_amp_time:.2f}s | Speedup: {basic_amp_time / opt_amp_time:.2f}x")
else:
    print("Train AMP: skipped (CUDA not available)")

# Simple throughput metric (tokens/sec)
tokens = BATCH_SIZE * SEQUENCE_LENGTH * STEPS
print(f"Throughput FP32 - Basic: {tokens/basic_fp32_time:.0f} tok/s | Optimized: {tokens/opt_fp32_time:.0f} tok/s")
if device.type == 'cuda' and basic_amp_time is not None and opt_amp_time is not None:
    print(f"Throughput AMP  - Basic: {tokens/basic_amp_time:.0f} tok/s | Optimized: {tokens/opt_amp_time:.0f} tok/s")