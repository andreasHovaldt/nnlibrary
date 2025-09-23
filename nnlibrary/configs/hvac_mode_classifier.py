import json
from pathlib import Path
from types import SimpleNamespace
from nnlibrary.engines import hooks as h
from .__base__ import BaseConfig, DataLoaderConfig


data_path = Path().cwd().resolve() / "data" / "14days_2025-09-04_2025-09-18" / "dataset"
dataset_metadata = json.loads((data_path / "stats" / "metadata.json").read_text())



#############################################
### General Config ##########################
#############################################

save_path = "exp/"

epochs = 10 # <-- Unused
start_epoch = 0
max_epoch = 3

lr = 1e-3

device = 'cpu'

amp_enable = True
amp_dtype = "float16"

enable_wandb = True
wandb_project_name = "nnlibrary"
wandb_key = None

hooks = [
    h.TestHook
]

# TODO CONFIGS
# seed = None
# weight = None



#############################################
### Model Config ############################
#############################################
model_config = BaseConfig(
    name = "HVACModeMLP",
    args = {
        "window_size": dataset_metadata["window"],
        "feature_dim": dataset_metadata["feature_dim"],
        "n_classes": dataset_metadata["num_classes"]
    }
)



#############################################
### Optimizer Config ########################
#############################################
optimizer = BaseConfig( # 'params' and 'lr' should not be passed in args
    name = "AdamW",
    args = {},
)



#############################################
### Scheduler Config ########################
#############################################
scheduler = BaseConfig( # 'optimizer' should not be passed in args
    name = "OneCycleLR",
    args={
        "epochs": max_epoch - start_epoch,
        "steps_per_epoch": dataset_metadata["train_samples"],
        "pct_start": 0.1, # % of time used for warmup
        "max_lr": lr,
        "anneal_strategy": "cos",
        "div_factor": 1e1, # 10.0,
        "final_div_factor": 1e3, # 1000.0,
    },
)



#############################################
### Loss function Config ####################
#############################################
loss_fn = BaseConfig(
    name = "CrossEntropyLoss",
    args={},
)



#############################################
### Dataset Config ##########################
#############################################
dataset = SimpleNamespace()
dataset.train = DataLoaderConfig(
    dataset = BaseConfig(
        name = "MpcDatasetHDF5",
        args = {
            "hdf5_file": data_path / "train.h5",
            "cache_in_memory": True,
            "verbose": True,
        }),
    batch_size = 512,
    shuffle = True,
)

dataset.val = DataLoaderConfig(
    dataset=BaseConfig(
        name="MpcDatasetHDF5",
        args={
            "hdf5_file": data_path / "val.h5",
            "cache_in_memory": True,
            "verbose": True,
        }),
    batch_size=512,
    shuffle=False,
)

dataset.test = DataLoaderConfig(
    dataset=BaseConfig(
        name="MpcDatasetHDF5",
        args={
            "hdf5_file": data_path / "test.h5",
            "cache_in_memory": True,
            "verbose": True,
        }),
    batch_size=512,
    shuffle=False,
)