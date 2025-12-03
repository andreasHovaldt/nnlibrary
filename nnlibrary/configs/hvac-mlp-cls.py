import json
from pathlib import Path
from types import SimpleNamespace
from nnlibrary.engines import hooks as h

from .__default__ import *
from .__base__ import BaseConfig, DataLoaderConfig



#############################################
### General Config ##########################
#############################################

dataset_name = "730days_2023-09-24_2025-09-23"
data_root = Path().cwd().resolve() / "data" / dataset_name / "dataset6-classification-normalized"
dataset_metadata = json.loads((data_root / "stats" / "metadata.json").read_text())

save_path = "exp/"
task = "classification"

num_epochs = 10
train_batch_size = 512
eval_batch_size = 512

lr = 1e-3

validation_metric_name = "loss"
validation_metric_higher_is_better = False

seed = 42



#############################################
### Model Config ############################
#############################################
model_config = BaseConfig(
    name = "HVACModeMLP",
    args = dict(
        window_size = dataset_metadata["temporal_settings"]["window"],
        feature_dim = dataset_metadata["dataset_info"]["feature_dim"],
        n_classes = dataset_metadata["dataset_info"]["num_classes"],
    )
)



#############################################
### Loss function Config ####################
#############################################
loss_fn = BaseConfig(
    name="CrossEntropyLoss",
    args=dict(
        weight = [0.25, 0.50, 10.00],
    )
)

# loss_fn = BaseConfig(
#     name="FocalLoss",
#     args=dict(
#         alpha = [0.27, 0.46, 2.28],
#         gamma = 2.0,
#     )
# )



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
    args=dict(
        pct_start = 0.1, # % of time used for warmup
        anneal_strategy = "cos",
        div_factor = 1e1, # 10.0,
        final_div_factor = 1e3, # 1000.0,
    ),
)



#############################################
### Dataset Config ##########################
#############################################
dataset = SimpleNamespace()
dataset.info = dict(
    num_classes = 3,
    class_names = [
        "econ",
        "cool",
        "heat",
    ],
    input_transforms = None,
    target_transforms = None,
)

dataset.train = DataLoaderConfig(
    dataset = dict(
        name = "MpcDatasetHDF5",
        args = dict(
            hdf5_file = data_root / "train.h5",
            task = task,
            cache_in_memory = True,
            verbose = True,
        )),
    shuffle = True,
)

dataset.val = DataLoaderConfig(
    dataset=dict(
        name="MpcDatasetHDF5",
        args=dict(
            hdf5_file = data_root / "val.h5",
            task = task,
            cache_in_memory = True,
            verbose = True,
        )),
    shuffle=False,
)

dataset.test = DataLoaderConfig(
    dataset=dict(
        name="MpcDatasetHDF5",
        args=dict(
            hdf5_file = data_root / "test.h5",
            task = task,
            cache_in_memory = True,
            verbose = True,
        )),
    shuffle=False,
)

# Sweep configuration - This is only needed if you want to run 'scripts/sweep.py'
# Define the search space
sweep_configuration = {
    "name": "mlp-cls-sweep",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "num_epochs": {"values": [5, 10, 15, 20, 30]},
        "lr": {"values": [1e-2, 1e-3, 1e-4]},
        "train_batch_size": {"values": [256, 512, 1024, 2048]},
    },
}