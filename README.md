# MPC AHU Neural Network

High‑level framework for training and evaluating neural network surrogates of an existing Model Predictive Control (MPC) strategy governing HVAC Air Handling Units (AHUs) (e.g. `HVAC_SR1_PLC1`, `HVAC_SR1_PLC2`). The goal is to approximate the MPC policy / setpoint generation (classification or regression targets) with fast, deployable PyTorch models (MLP, TCN, etc.) while keeping the experimentation loop reproducible, observable and extensible.

For background on the underlying control problem see the MPC documentation: https://mpc-cv.docs.cern.ch/

---
## Framework overview

1. Configuration‑First Design
	 - Each experiment is defined by a config module in `nnlibrary/configs/` (model, data paths, dataloader params, optimizer/scheduler, hooks, metrics).
	 - Usually each config imports a default config (`nnlibrary/configs/__default__.py`) which contains some of the lesser important config settings
    	 - This approach is done because all configs must contain all the different arguments, but it is not always necessary to modify all settings per config basis

2. Datasets & I/O
	 - Dataset configuration (paths, batch sizes, shuffle flags, optional transforms) is declared in the experiment config modules under `nnlibrary/configs/` (see the `dataset` and its `train/val/test` `DataLoaderConfig` entries).
	 - Concrete dataset classes live in `nnlibrary/datasets/` (e.g. `MpcDatasetHDF5` from `nnlibrary/datasets/mpc_ahu.py`). Each dataset should at the least implements `__len__` and `__getitem__` to return `(inputs, targets)`.
	 - To use a different dataset, add a new Dataset class in `nnlibrary/datasets/`, import it in `nnlibrary/datasets/__init__.py`, and then you can reference it by name in the config’s dataset section.

3. Models
	 - Defined in `nnlibrary/models/` (e.g. `mlp.py`, `cnn.py`).
	 - New models can be defined, but must be imported in `nnlibrary/models/__init__.py` to be used.

4. Training Engine
	 - Central `Trainer` (in `nnlibrary/engines/train.py`) orchestrates: dataloaders, model, optimizer / scheduler, AMP autocast, hooks.
	 - Usually direct usage of `Trainer` is not recommended, instead use the training script under `scripts/train.py`
     - Metrics & state exposed through a shared `info` dict so hooks remain decoupled.

5. Hooks (Lifecycle Extensions)
	 - A 'Hook' frame is present for interacting with multiple different stages of the training process, these can be seen under `nnlibrary/engines/hooks.py`.
     - The base hooks implement validation / test evaluation, checkpointing, timing instrumentation, logging to Weights & Biases (W&B) and TensorBoard, post‑test plotting.
	 - Easy to add custom logic (inherit `Hookbase` and register in config list).
    	 - If you want to keep the default hooks in your custom config add the following to your config:
            ```python
            hooks.append(CustomHookName)
            ```

6. Evaluation Abstractions
	 - `ClassificationEvaluator` and `RegressionEvaluator` return consistent metric dicts (plus optional detailed artifacts like confusion matrix or prediction scatter/sequence plots).

7. Observability & Reproducibility
	 - W&B run grouping by dataset/model; TensorBoard summaries under `exp/<dataset>/<model>/tensorboard`.
	 - Deterministic config capture via run config; model checkpoints (`model_last.pth`, `model_best.pth`).

---
## Repository Layout (Essentials)

```
nnlibrary/
	configs/          # Experiment & model/training hyper‑parameter configs
	datasets/         # Dataset classes containing sample loading logic
	engines/          # Main scripts orchestrating the runtime
	models/           # Neural network architectures (MLP, TCN, etc.)
	utils/            # Custom losses, schedulers, operations (causal conv, standardize, etc.)

scripts/
	train.py                # Entry point: dynamic config loading & training
	eval_visualization.py   # Post‑training sequence prediction plots (WIP)

exp/                # Auto‑generated experiment artifacts (checkpoints, figures, logs)
data/               # Contains the dataset files
environment.yml     # Conda environment definition
```

---
## Quick Start

### 1. Environment
```bash
conda env create -f environment.yml
conda activate pytorch
```

### 2. Data Layout Expectation (This is to be changed)
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
Use an existing config name (e.g. `HVACModeMLP`, `TCN-reg`, etc.). The train script resolves short names, dotted paths, or file paths.

For running the config named "TCN-reg", residing at `nnlibrary/configs/TCN-reg.py`:
```bash
python scripts/train.py TCN-reg # Shorthand
python scripts/train.py nnlibrary.configs.TCN-reg # Using module path
python scripts/train.py nnlibrary/configs/TCN-reg.py # Using relative or absolute path
```

Outputs (example):
```
exp/<dataset>/<model_name>/
	model/
        model_last.pth
	    model_best.pth
	tensorboard/
	figures/
	wandb/
```

### 4. Evaluate & Visualize (Very WIP, only works for classification task)
```
Usage:
    python scripts/eval_visualization.py <config_name> [--split test|val] [--interactive]
```
```
python scripts/eval_visualization.py \
	TCN-reg \
    --split {val, test}     # Which split to evaluate on
	--interactive           # optional Plotly HTML timeline
```


### 5. Inspect Metrics
* TensorBoard: `tensorboard --logdir exp/<dataset>/<model_name>/tensorboard`
* Weights & Biases: open the run page (if enabled in config).

---
## Extending
| Task | Where |
|------|-------|
| New model | Implement in `nnlibrary/models/`, import new model in `/nnlibrary/model/__init__.py`, then reference in your config |
| Custom hook | Subclass `Hookbase` in `nnlibrary/engines/hooks.py`, add to config `hooks` list |
| New loss or scheduler | Add to `nnlibrary/utils/loss.py` or `utils/schedulers.py` |
| New dataset | Implement in `nnlibrary/datasets/`, import new dataset in `/nnlibrary/datasets/__init__.py`, then update config to point to it |s


## Minimal In‑Code Example
```python
from nnlibrary.engines import Trainer
import nnlibrary.configs.HVACModeMLP as cfg

trainer = Trainer(cfg=cfg)
trainer.train()      # trains, validates (if enabled), checkpoints
```

---
## Design Idea
* Single source of truth: configs hold knobs; trainer/hook code stays lean.
* Pluggable lifecycle: hooks avoid subclass explosion on the trainer.
* Evaluation symmetry: same evaluators for validation & test; consistent metric dict.
* Traceability: run artifacts self-contained under `exp/`.
* Modularization allows for easier debugging.


## Next Steps / TODO (High-Level)
* Add config template docs & typed validation
* Optional rng seed control
* CLI for quick metric table summary across runs