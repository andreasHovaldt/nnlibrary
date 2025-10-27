from pathlib import Path
import sys; sys.path.append(str(Path(__file__).parent.parent))

import time
import random
import torch
from torch.amp.grad_scaler import GradScaler
import numpy as np
from typing import TypedDict, Literal
from nnlibrary.models import TransformerRegression, TransformerRegressionOptimized

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

# Dataset settings
INPUT_DIM = 37
OUTPUT_DIM = 5
SEQUENCE_LENGTH = 32
BATCH_SIZE = 32

# Test settings
TORCH_COMPILE = False
STEPS = 1000
WARMUP_STEPS = 100
LR = 1e-3
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
    model_opt = TransformerRegressionOptimized(**transformer_hyperparameters)
    if TORCH_COMPILE: 
        model_opt: TransformerRegressionOptimized = torch.compile(model_opt)  # type: ignore
except RuntimeError:
    print('Failed to compile model!')
    model_opt = TransformerRegressionOptimized(**transformer_hyperparameters)

# Benchmark
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device: ', device)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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