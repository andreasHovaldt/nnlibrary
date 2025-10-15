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
data_root = Path().cwd().resolve() / "data" / dataset_name / "dataset-regression-normalized"
dataset_metadata = json.loads((data_root / "stats" / "metadata.json").read_text())

save_path = "exp/"
task = "regression"

num_epochs = 10
train_batch_size = 512
eval_batch_size = 512

lr = 1e-3

validation_metric_name = "loss"
validation_metric_higher_is_better = False

# TODO CONFIGS
# seed = None
# weight = None should be a path to a pretrained params dict



#############################################
### Model Config ############################
#############################################
model_config = BaseConfig(
    name = "TCNRegression",
    args = dict(
        input_dim=dataset_metadata["feature_dim"],
        sequence_length=dataset_metadata["window"],
        num_classes=dataset_metadata["num_classes"],
        regression_head_hidden_dim=64,
        hidden_layer_sizes=[64, 64, 128, 128],
        kernel_size=3,
        dropout=0.3,
        dropout_type="channel",
    )
)



#############################################
### Loss function Config ####################
#############################################
loss_fn = BaseConfig(
    name="MSELoss",
    args=dict()
)



#############################################
### Optimizer Config ########################
#############################################
optimizer = BaseConfig( # 'params' and 'lr' should not be passed in args
    name = "AdamW",
    args = dict(),
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
        "fan_speed_cmd_10001",
        "fresh_air_damper_cmd_10001",
        "setpoint_supply_air_mpc_10001",
        "setpoint_heating_mpc_10001",
        "setpoint_cooling_mpc_10001",
    ],
    standardize_target = False,
    normalize_target = True,
)
dataset.train = DataLoaderConfig(
    dataset = BaseConfig(
        name = "MpcDatasetHDF5",
        args = dict(
            hdf5_file = data_root / "train.h5",
            task = task,
            cache_in_memory = True,
            verbose = True,
        )),
    batch_size = train_batch_size,
    shuffle = True,
)

dataset.val = DataLoaderConfig(
    dataset=BaseConfig(
        name="MpcDatasetHDF5",
        args=dict(
            hdf5_file = data_root / "val.h5",
            task = task,
            cache_in_memory = True,
            verbose = True,
        )),
    batch_size=eval_batch_size,
    shuffle=False,
)

dataset.test = DataLoaderConfig(
    dataset=BaseConfig(
        name="MpcDatasetHDF5",
        args=dict(
            hdf5_file = data_root / "test.h5",
            task = task,
            cache_in_memory = True,
            verbose = True,
        )),
    batch_size=eval_batch_size,
    shuffle=False,
)