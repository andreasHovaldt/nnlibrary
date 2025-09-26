import torch
from nnlibrary.engines import hooks as h

#############################################
### General Config ##########################
#############################################

save_path = "exp/"
validate_model = True # Whether to validate the model while training
test_model = True # Whether to test the model post training

lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
amp_enable = True
amp_dtype = "float16"

enable_tensorboard = False
enable_wandb = True
wandb_group_name = None
wandb_project_name = "nnlibrary"
wandb_key = None

hooks = [
    h.ValidationHook,
    h.WandbHook,
    h.TensorBoardHook,
    h.CheckpointerHook,
]

# TODO CONFIGS
# seed = None
# weight = None