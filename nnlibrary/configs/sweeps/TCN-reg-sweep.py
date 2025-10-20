import os
import wandb
from pathlib import Path


# Set wandb environment variable
wandb_key = None

# If no key in config, try to read from .secrets/wandb file
secrets_file = Path().cwd().resolve() / '.secrets' / 'wandb'
if secrets_file.exists():
    try:
        wandb_key = secrets_file.read_text().strip()
    except Exception as e:
        print(f"Warning: Could not read WandB key from {secrets_file}: {e}")
else:
    print(f"Could not find the wandb secrets file at: '{secrets_file}'")

# Set the environment variable if we found a key
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key
    print("WandB API key loaded successfully")
else:
    print("No WandB API key found - WandB logging has been disabled")


# Ensure multiple runs inits are allowed for the current runtime
wandb.setup(wandb.Settings(reinit="create_new"))
# When a new wandb.init is run, a new run is created while still allowing to log data to previous runs
# It is important to handle the runs with the specific run object and not use the general wandb.log function


# Helper to ensure directories exist for run artifacts
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# 1: Define objective/training function
def objective(config, run: wandb.Run):
    # Log to the active sweep run passed from main()
    score = 0.0
    for epoch in range(config.epochs):
        score += config.x**3 + config.y
        run.log({
            "epoch": epoch,
            "score": score,
        })
    return score

def main():
    out_dir = (Path(__file__).parent / 'sweep').resolve()
    _ensure_dir(out_dir)
    # Important: For sweeps, the agent provides project/name; ignore custom project warnings
    run = wandb.init(dir=out_dir)
    try:
        score = objective(run.config, run)
        run.log({"score": score})
    finally:
        run.finish()


# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
        "epochs": {"values": [5,]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
wandb.finish()






# import os
# import wandb
# from pathlib import Path


# # Set wandb environment variable
# wandb_key = None

# # If no key in config, try to read from .secrets/wandb file
# secrets_file = Path().cwd().resolve() / '.secrets' / 'wandb'
# if secrets_file.exists():
#     try:
#         wandb_key = secrets_file.read_text().strip()
#     except Exception as e:
#         print(f"Warning: Could not read WandB key from {secrets_file}: {e}")
# else:
#     print(f"Could not find the wandb secrets file at: '{secrets_file}'")

# # Set the environment variable if we found a key
# if wandb_key:
#     os.environ['WANDB_API_KEY'] = wandb_key
#     print("WandB API key loaded successfully")
# else:
#     print("No WandB API key found - WandB logging has been disabled")


# # Ensure multiple runs inits are allowed for the current runtime
# wandb.setup(wandb.Settings(reinit="create_new"))
# # When a new wandb.init is run, a new run is created while still allowing to log data to previous runs
# # It is important to handle the runs with the specific run object and not use the general wandb.log function


# def objective_wandb_run(epochs: int) -> wandb.Run:
#     return wandb.init(
#         project="sweep-test",
#         dir=(Path(__file__).parent / 'individual').resolve(),
#         config = dict(
#             dataset = f"sweep-dataset",
#             task = "sweep-task",
#             architecture = "sweep-achitecture",
#             model_name = "sweep-model",
            
#             epochs = epochs,
#             batch_size = 67,
#             learning_rate = 100,
            
#             loss_fn = "sweep-loss-fn",
#             optimizer = "sweep-optimizer",
#             scheduler = "sweep-scheduler",
#         )
#     )

# # 1: Define objective/training function
# def objective(config):
#     score = 0.0
#     with objective_wandb_run(epochs=config.epochs) as objective_run:
#         for epoch in range(config.epochs):
#             score += config.x**3 + config.y
#             objective_run.log(
#                 data = dict(
#                     epoch=epoch,
#                     score=score,
#                 )
#             )
#     return score

# def main():
#     with wandb.init(
#         project="sweep-test",
#         dir=(Path(__file__).parent / 'sweep').resolve(),
#     ) as run:
#         score = objective(run.config)
#         run.log({"score": score})


# # 2: Define the search space
# sweep_configuration = {
#     "method": "random",
#     "metric": {"goal": "minimize", "name": "score"},
#     "parameters": {
#         "x": {"max": 0.1, "min": 0.01},
#         "y": {"values": [1, 3, 7]},
#         "epochs": {"values": [5,]},
#     },
# }

# # 3: Start the sweep
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

# wandb.agent(sweep_id, function=main, count=10)
# wandb.finish()