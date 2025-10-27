from pathlib import Path
import sys; sys.path.append(str(Path(__file__).parent.parent))

import time
import torch
from typing import TypedDict, Literal
from nnlibrary.models import TransformerRegression, TransformerRegressionOptimized

# Types for constructor kwargs
class TransformerHyperparameters(TypedDict):
    input_dim: int
    output_dim: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    max_seq_length: int
    dropout: float
    pooling: Literal['last', 'mean', 'cls']

# Dataset settings
INPUT_DIM = 37
OUTPUT_DIM = 5
SEQUENCE_LENGTH = 32
BATCH_SIZE = 32

# Model args (now precisely typed)
transformer_hyperparameters: TransformerHyperparameters = {
    'input_dim': INPUT_DIM,
    'output_dim': OUTPUT_DIM,
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 2,
    'd_ff': 256,
    'max_seq_length': 32,
    'dropout': 0.1,
    'pooling': 'last',
}

# Basic implementation
model_basic = TransformerRegression(**transformer_hyperparameters)
model_basic = torch.compile(model_basic)

# Optimized implementation
model_opt = TransformerRegressionOptimized(**transformer_hyperparameters)
model_opt = torch.compile(model_opt)

# Benchmark
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device: ', device)
model_basic.to(device)              # avoid device mismatch
model_opt.to(device)
x = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_DIM, device=device)

# Warmup + accurate CUDA timing
for _ in range(10):
    y = model_basic(x); (y.mean()).backward(); model_basic.zero_grad(set_to_none=True)
torch.cuda.synchronize() if device.type == 'cuda' else None

start = time.time()
for _ in range(100):
    y = model_basic(x)
    y.mean().backward()
    model_basic.zero_grad(set_to_none=True)
torch.cuda.synchronize() if device.type == 'cuda' else None
basic_time = time.time() - start

# Optimized
start = time.time()
for _ in range(100):
    out = model_opt(x)
    out.mean().backward()
opt_time = time.time() - start

print(f"Basic: {basic_time:.2f}s")
print(f"Optimized: {opt_time:.2f}s")
print(f"Speedup: {basic_time / opt_time:.2f}x")

# Expected: 2-5x speedup!