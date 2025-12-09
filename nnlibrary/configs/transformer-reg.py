import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from nnlibrary.engines import hooks as h

from .__default__ import *
from .__base__ import DataLoaderConfig



#############################################
### General Config ##########################
#############################################

dataset_name = "730days_2023-09-24_2025-09-23"
data_root = Path().cwd().resolve() / "data" / dataset_name / "dataset6-regression-normalized"
dataset_metadata = json.loads((data_root / "stats" / "metadata.json").read_text())

save_path = "exp/"
task = "regression"

num_epochs = 20
train_batch_size = 1024
eval_batch_size = 1024

lr = 1e-3

validation_metric_name = "loss"
validation_metric_higher_is_better = False

seed = 42



#############################################
### Model Config ############################
#############################################
model_config = dict(
    name = "TransformerRegression",
    args = dict(
        input_dim=dataset_metadata["dataset_info"]["feature_dim"],
        output_dim=dataset_metadata["dataset_info"]["num_classes"],
        dim_model = 64,
        num_heads = 4,
        num_layers = 2,
        dim_ff = 256,
        max_seq_length=dataset_metadata["temporal_settings"]["window"],
        dropout=0.1,
        pooling="last",
    )
)

#############################################
### Loss function Config ####################
#############################################
loss_fn = dict(
    name="MSELoss",
    args=dict()
)



#############################################
### Optimizer Config ########################
#############################################
optimizer = dict( # 'params' and 'lr' should not be passed in args
    name = "AdamW",
    args = dict(),
)



#############################################
### Scheduler Config ########################
#############################################
scheduler = dict( # 'optimizer' should not be passed in args
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
    num_classes = 4,
    class_names = [
        "fan_speed_cmd_10001",
        "fresh_air_damper_cmd_10001",
        "setpoint_heating_mpc_10001",
        "setpoint_cooling_mpc_10001",
    ],
    
    input_transforms = None,
    
    target_transforms = [
        # dict(
        #     name = 'Absolute2Relative', # FIXME: Theres a problem since the setpoints change during the dataset
        #     args = dict(
        #         zone_sp_min = float,
        #         zone_sp_max = float,
        #         heating_idx = 2,
        #         cooling_idx = 3,
        #     ),
        # ),
        
        dict(
            name = 'MinMaxNormalize',
            args = dict(
                min_vals=np.load(data_root / "stats" / "target_min.npy").astype(float).tolist(),
                max_vals=np.load(data_root / "stats" / "target_max.npy").astype(float).tolist(),
            ),
        ),
        
        # dict(
        #     name = 'Standardize',
        #     args = dict(
        #         mean=np.load(data_root / "stats" / "target_mean.npy").astype(float).tolist(),
        #         std=np.load(data_root / "stats" / "target_std.npy").astype(float).tolist(),
        #     ),
        # ),
    ]
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
    "name": "transformer-reg-sweep",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "num_epochs": {"values": [5, 10, 15, 20, 30]},
        "lr": {"values": [1e-2, 1e-3, 1e-4]},
        "train_batch_size": {"values": [256, 512, 1024, 2048]},
    },
}