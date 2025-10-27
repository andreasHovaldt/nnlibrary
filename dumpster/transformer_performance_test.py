from pathlib import Path
import sys; sys.path.append(str(Path(__file__).parent.parent))

import time
import random
import torch
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
INFERENCE_MODE = False
TORCH_COMPILE = False

# Model args (now precisely typed)
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
model_basic.to(device)              # avoid device mismatch
model_opt.to(device)
x = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_DIM, device=device)

# Toggle model mode based on INFERENCE_MODE
if INFERENCE_MODE:
    model_basic.eval()
    model_opt.eval()
else:
    model_basic.train()
    model_opt.train()

# Warmup both models to stabilize kernels/JIT
with (torch.inference_mode() if INFERENCE_MODE else torch.enable_grad()):
    for _ in range(10):
        _ = model_basic(x)
        _ = model_opt(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None

# Basic (inference)
y: torch.Tensor | None = None
start = time.perf_counter()
with (torch.inference_mode() if INFERENCE_MODE else torch.enable_grad()):
    for _ in range(1000):
        y = model_basic(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
basic_time = time.perf_counter() - start
assert y is not None
print(y.sum(dim=0))

# Optimized (inference)
out: torch.Tensor | None = None
start = time.perf_counter()
with (torch.inference_mode() if INFERENCE_MODE else torch.enable_grad()):
    for _ in range(1000):
        out = model_opt(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
opt_time = time.perf_counter() - start
assert out is not None
print(out.sum(dim=0))

print(f"Basic: {basic_time:.2f}s")
print(f"Optimized: {opt_time:.2f}s")
print(f"Speedup: {basic_time / opt_time:.2f}x")