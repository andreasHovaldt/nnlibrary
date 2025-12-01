import torch
from nnlibrary.engines import hooks as h

#############################################
### General Config ##########################
#############################################

save_path = "exp/"

timing = True # Whether to time the run

validate_model = True # Whether to validate the model while training
validation_metric_name = "loss" # The metric used to determine the best model
validation_metric_higher_is_better = False # Whether higher or lower is better for the defined validation_metric
validation_plot = False # Whether to create a plot based on the validation each epoch (Only available on wandb)

test_model = True # Whether to test the model post training

lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
amp_enable = True
amp_dtype = "float16"
clip_grad = None # Gradient clipping, None to disable, set to a float to enable, common range between 0.5 -> 5.0

enable_tensorboard = True
enable_wandb = True
wandb_group_name = None
wandb_project_name = "nnlibrary"
wandb_key = None

hooks = [
    h.ValidationHook, # <--- Needs to be before wandb and tensorboard hooks in the list
    h.TestHook,       # <----/
    h.WandbHook,
    h.TensorBoardHook,
    h.CheckpointerHook,
    h.TimingHook,
]

# Reproducibility controls
# If set to an integer, training will attempt to be deterministic across Python, NumPy, and PyTorch
# Note: some CUDA ops are inherently non-deterministic; enabling full determinism may impact performance
seed = None

# TODO: CONFIGS
# weight = None # should be a path to a pretrained params dict