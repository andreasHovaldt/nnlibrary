"""
Based on:
https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
"""

# Ensure project root is on path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
import torch
import onnx
from nnlibrary.models.cnn import TCN, TCNRegression, TCNResidualBlock


# Convert model to ONNX format
torch_model = TCNResidualBlock(
    in_channels=37,
    out_channels=128,
    kernel_size=3,
    dilation=2 ** 1,
    dropout=0.3,
    dropout_type='channel',
)

example_inputs = (torch.randn((512, 32, 37)),)
onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
assert onnx_program is not None, 'Failed to convert model to ONNX format!'


# Save the ONNX model
name = 'TCNResidualBlock'
save_path = PROJECT_ROOT / 'exp' / 'onnx_models' / f'{name}.onnx'
onnx_program.save(save_path)


# Load saved model to check if it was well formed
onnx_model = onnx.load(save_path)
onnx.checker.check_model(onnx_model)