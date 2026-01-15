# NNLIBRARY

High‑level framework for training and evaluating neural network architectures. The goal is to make training both classification and regression networks fast, easy, and consistent, while keeping the experimentation loop reproducible, observable and extensible.

Credit to [Pointcept](https://github.com/Pointcept/Pointcept), which this approach was inspired by.

---
## Overview

1. Configuration Design
	 - Each experiment is defined by a config module in `nnlibrary/configs/` (model, data paths, dataloader params, optimizer/scheduler, hooks, metrics).
	 - Usually each config imports a default base config ([`nnlibrary/configs/__default__.py`](./nnlibrary/configs/__default__.py)) which contains some of the general config settings.

2. Datasets & I/O
	 - Dataset configuration (paths, batch sizes, shuffle flags, optional transforms) is declared in the experiment config modules under `nnlibrary/configs/` (see the `dataset` and its `train/val/test` `DataLoaderConfig` entries).
	 - Concrete dataset classes are placed in `nnlibrary/datasets/` (e.g. `MpcDatasetHDF5` from [`nnlibrary/datasets/mpc_ahu.py`](./nnlibrary/datasets/mpc_ahu.py)). Each dataset should at the least implement `__len__` and the `__getitem__` for returning `(inputs, targets)`.
	 - To use a different dataset, add a new Dataset class in `nnlibrary/datasets/`, import it in [`nnlibrary/datasets/__init__.py`](./nnlibrary/datasets/__init__.py), and then you can reference it by name in the config’s dataset section.

3. Models
	 - Defined in `nnlibrary/models/` (e.g. `mlp.py`, `cnn.py`, `transformer.py`).
	 - Available architectures: MLP (`HVACMLP`), TCN (`TCNClassification`, `TCNRegression`), Transformer (`TransformerRegression`, `TransformerClassification`).
	 - New models can also be defined, but must be imported in [`nnlibrary/models/__init__.py`](./nnlibrary/__init__.py) to be used.

4. Training Engine
	 - Central `Trainer` (in [`nnlibrary/engines/train.py`](./nnlibrary/engines/train.py)) orchestrates: dataloaders, model, optimizer / scheduler, AMP, gradient clipping, hooks.
	 - Usually direct usage of `Trainer` is not needed, instead use the training script under [`scripts/train.py`](./scripts/train.py).
	 - Metrics & state exposed through a shared `info` dict so hooks remain decoupled.

5. Hooks
	 - A 'Hook' frame is present for interacting with multiple different stages of the training process, these can be seen under [`nnlibrary/engines/hooks.py`](./nnlibrary/engines/hooks.py).
     - The base hooks implement validation / test evaluation, checkpointing, timing instrumentation, logging to Weights & Biases (W&B) and TensorBoard, post‑test plotting.
	 - Easy to add custom logic (inherit `Hookbase` and register in config list).
    	 - If you want to keep the default hooks but also add new custom hooks in a config, add the following to your config:
            ```python
            hooks.append(CustomHookName)
            ```

6. Evaluation Abstractions
	 - `ClassificationEvaluator` and `RegressionEvaluator` return metric dicts (plus optional detailed artifacts like confusion matrix or prediction scatter/sequence plots).

7. Observability & Reproducibility
	 - W&B run grouping by dataset/model; TensorBoard summaries under `exp/<dataset>/<model>/tensorboard`.
	 - Deterministic config capture via run config; model checkpoints (`model_last.pth`, `model_best.pth`).

---
## Repository Layout

```
nnlibrary/
	configs/    # Experiment & model/training hyper‑parameter configs
	datasets/   # Dataset classes containing sample loading logic
	engines/    # Training engine, hooks, and evaluation classes
	models/     # Neural network architectures (MLP, TCN, Transformer)
	utils/      # Custom losses, schedulers, transforms, misc

scripts/
	train.py                # Entry point: dynamic config loading & training
	sweep.py                # Script for running WandB hyperparameter sweeps
	eval_visualization.py   # Post‑training evaluation and visualization
	export_onnx_model.py    # Export trained model checkpoints to ONNX format

exp/              # Auto‑generated experiment artifacts (checkpoints, figures, logs)
data/             # !! Here you should put the datasets !!
dumpster/         # Scratch/experimental scripts (not part of main workflow)
environment.yml   # Conda environment definition
.secrets/
	wandb         # !! File containing your WandB API key !!
```

---
## Quick Start

### 1. Environment
```bash
conda env create -f environment.yml
conda activate pytorch
```

### 1.1. WandB integration (Optional but strongly recommended)
Add your WandB API key in a file under the ```.secrets``` directory:
```bash
mkdir .secrets
echo '<wandb-api-key>' > .secrets/wandb
```

### 2. Data Layout
```
data/
	<dataset_range>/
		dataset-classification/ or dataset-regression/
			train.h5  val.h5  test.h5
			stats/
				metadata.json
				(optional) feature_means.npy / feature_stds.npy / target_mean.npy / target_std.npy
```

### 3. Train a Model
Use an existing config name (e.g. `MLP-cls`, `TCN-reg`, `transformer-reg`, etc.). The train script resolves short names, dotted paths, or file paths.

For running the config named "TCN-reg", placed at [`nnlibrary/configs/TCN-reg.py`](./nnlibrary/configs/TCN-reg.py):
```bash
python scripts/train.py -n TCN-reg # Shorthand
python scripts/train.py -n nnlibrary.configs.TCN-reg # Using module path
python scripts/train.py -n nnlibrary/configs/TCN-reg.py # Using relative or absolute path
```

Optional flags:
- `--logging` — Enable logger output
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` — Set logging verbosity (default: INFO)

Outputs (example):
```
exp/<dataset>/<model_name>/<run_name>/
	model/
        model_last.pth
	    model_best.pth
	tensorboard/
	figures/
	wandb/
```

### 3.1. Hyperparameter Sweeps
The framework supports WandB-powered hyperparameter sweeps. To use sweeps:

1. **Define a sweep configuration in your config file:**
```python
# At the end of your config (e.g., TCN-reg.py)
sweep_configuration = {
    "name": "TCN-reg-sweep",
    "method": "grid",  # or "random", "bayes"
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "num_epochs": {"values": [5, 10, 15, 20, 30]},
        "lr": {"values": [1e-2, 1e-3, 1e-4]},
        "train_batch_size": {"values": [256, 512, 1024, 2048]},
    },
}
```

2. **Run the sweep:**
```bash
python scripts/sweep.py -n TCN-reg
python scripts/sweep.py -n transformer-reg --logging --log-level DEBUG
```

**Requirements:**
- WandB must be enabled in the config (`enable_wandb = True`)
- A `sweep_configuration` dict must be defined in the config
- Setting a `seed` is recommended for reproducibility across sweep runs

**Notes:**
- Sweep results are saved under `exp/<dataset>/<model>/sweeps/<sweep_name>/`
- The sweep will override config values (e.g., `lr`, `num_epochs`) with values from the sweep search space
- Metrics logged depend on the task: regression logs `loss`, `mae`, `rmse`, `r2_score`; classification logs `loss`, `avg_sample_accuracy`, `avg_class_accuracy`



### 4. Evaluate & Visualize (WIP)
Visualize predictions from a trained model checkpoint:
```bash
python scripts/eval_visualization.py -n <config_name> -r <run_name> [--split train|val|test] [--interactive]
```

Example:
```bash
python scripts/eval_visualization.py -n TCN-reg -r witty-salamander-4 --split test
python scripts/eval_visualization.py -n transformer-reg -r clever-fox-2 --interactive
```

### 5. Export to ONNX
Export a trained checkpoint to ONNX format for deployment:
```bash
python scripts/export_onnx_model.py -n <config_name> -r <run_name> [--dynamic-batch] [--dynamic-seq] [--opset <VERSION_NUM>]
```

The exported ONNX model is saved under: `exp/<dataset>/<model_name>/<run_name>/onnx/`

Example:
```bash
python scripts/export_onnx_model.py -n TCN-reg -r witty-salamander-4
python scripts/export_onnx_model.py -n transformer-reg -r clever-fox-2 --dynamic-batch --opset 17
python scripts/export_onnx_model.py -n transformer-cls -r ugly-fork-42 --dynamic-batch --dynamic-seq
```
It is recommended to set the `--dynamic-batch` flag, since in deployment, the batch size would usually differ from the batch sized when trained. The `--dynamic-seq` should only be used if you know that the architecture supports a dynamic sequence length.

### 6. Inspect Metrics
* TensorBoard: `tensorboard --logdir exp/<dataset>/<model_name>/<run_name>/tensorboard`
* Weights & Biases: open the run page (if enabled in config).

---
## Extending
| Task | Where |
|------|-------|
| New model | Implement in `nnlibrary/models/`, import new model in [`/nnlibrary/model/__init__.py`](./nnlibrary/models/__init__.py), then reference in your config |
| Custom hook | Subclass `Hookbase` in [`nnlibrary/engines/hooks.py`](./nnlibrary/engines/hooks.py), add to config `hooks` list |
| New loss or scheduler | Add to [`nnlibrary/utils/loss.py`](./nnlibrary/utils/loss.py) or [`nnlibrary/utils/schedulers.py`](./nnlibrary/utils/schedulers.py) |
| New transform | Add to [`nnlibrary/utils/transforms.py`](./nnlibrary/utils/transforms.py) (e.g. target normalization/standardization) |
| New dataset | Implement in `nnlibrary/datasets/`, import in [`nnlibrary/datasets/__init__.py`](./nnlibrary/datasets/__init__.py), then reference in config |


## Minimal In‑Code Example
```python
from nnlibrary.engines import Trainer
import nnlibrary.configs.HVACModeMLP as cfg

trainer = Trainer(cfg=cfg)
trainer.train()      # trains, validates (if enabled), checkpoints
```
