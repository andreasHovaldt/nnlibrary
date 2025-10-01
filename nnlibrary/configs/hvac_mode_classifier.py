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
data_root = Path().cwd().resolve() / "data" / dataset_name / "dataset"
dataset_metadata = json.loads((data_root / "stats" / "metadata.json").read_text())

save_path = "exp/"

num_epochs = 10
train_batch_size = 512
eval_batch_size = 512

lr = 1e-3

validation_metric_name = "avg_class_accuracy"

# hooks.append()

# TODO CONFIGS
# seed = None
# weight = None



#############################################
### Model Config ############################
#############################################
model_config = BaseConfig(
    name = "HVACModeMLP",
    args = dict(
        window_size = dataset_metadata["window"],
        feature_dim = dataset_metadata["feature_dim"],
        n_classes = dataset_metadata["num_classes"],
    )
)



#############################################
### Loss function Config ####################
#############################################
loss_fn = BaseConfig(
    name="FocalLoss",
    args=dict(
        alpha = [0.27, 0.46, 2.28],
        gamma = 2.0,
    )
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
    args=dict(
        pct_start = 0.1, # % of time used for warmup
        max_lr = lr,
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
    ]
)
dataset.train = DataLoaderConfig(
    dataset = BaseConfig(
        name = "MpcDatasetHDF5",
        args = dict(
            hdf5_file = data_root / "train.h5",
            cache_in_memory = True,
            verbose = True,
        )),
    batch_size = train_batch_size, # TODO: Move this away form the config file and use the cfg.train_batch_size param instead
    shuffle = True,
)

dataset.val = DataLoaderConfig(
    dataset=BaseConfig(
        name="MpcDatasetHDF5",
        args=dict(
            hdf5_file = data_root / "val.h5",
            cache_in_memory = True,
            verbose = True,
        )),
    batch_size=train_batch_size, # FIXME: ^^^^
    shuffle=False,
)

dataset.test = DataLoaderConfig(
    dataset=BaseConfig(
        name="MpcDatasetHDF5",
        args=dict(
            hdf5_file = data_root / "test.h5",
            cache_in_memory = True,
            verbose = True,
        )),
    batch_size=eval_batch_size, # FIXME: ^^^^
    shuffle=False,
)