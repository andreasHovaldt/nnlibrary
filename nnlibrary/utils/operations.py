import torch
from torchvision.transforms import v2
from typing import Any, Sequence


class Standardize(v2.Transform):
    """Per-feature standardization transform for tensors.

    Applies standardized values x' = (x - mean) / (std + eps) using PyTorch broadcasting.
    This implementation assumes the feature dimension is last and that ``mean`` and ``std``
    are 1D of length F matching the last dimension. For example: (T, F) or (N, T, F).

    Note: This class subclasses ``torchvision.transforms.v2.Transform``. The ``params`` argument
    is currently unused and present to match the v2 callable signature.

    Args:
        mean: Sequence of per-feature means with length F
        std: Sequence of per-feature standard deviations with length F
        eps: Small constant added to the denominator to avoid division by zero (default: 1e-8)

    Example:
        >>> # x: (batch, time, feat)
        >>> t = torch.randn(8, 100, 5)
        >>> tr = Standardize(mean=[0.0, 1.0, -0.5, 2.0, 3.0], std=[1.0, 2.0, 0.5, 4.0, 0.25])
        >>> y = tr(t)
        >>> y.shape
        torch.Size([8, 100, 5])
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float], eps: float = 1e-8):
        """Create a per-feature standardization transform.

        Args:
            mean: Per-feature means of length F (must match the last dimension of inputs)
            std: Per-feature standard deviations of length F (same length as ``mean``)
            eps: Small constant to avoid division by zero (default: 1e-8)
        """
        super().__init__()
        assert len(mean) == len(std), "Mean and std must have the same length!"
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.eps = eps

    def transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> Any:
        """Apply standardization to the input tensor.

        Assumes the last dimension of ``inpt`` is the feature dimension of size F, and that
        ``mean`` / ``std`` have shape (F,). Uses broadcasting to subtract and divide along that
        last dimension while preserving all other dimensions.

        Args:
            inpt: Input tensor with feature dimension last (e.g., (T, F), (N, T, F))
            params: Unused parameter dictionary required by ``v2.Transform`` interface

        Returns:
            A tensor with the same shape and dtype as ``inpt`` where per-feature standardization
            has been applied.
        """
        assert type(inpt) == torch.Tensor, f"'Standardize' transform expects inputs to be of type torch.Tensor, got {type(inpt)}"
        # Move mean/std to same device as input
        mean = self.mean.to(inpt.device)
        std = self.std.to(inpt.device)
        return ((inpt - mean) / (std + self.eps)).to(dtype=inpt.dtype, device=inpt.device)
    
    def inverse_transform(self, inpt: torch.Tensor) -> torch.Tensor:
        """Apply inverse standardization to the input tensor."""
        assert type(inpt) == torch.Tensor, f"'Standardize' transform expects inputs to be of type torch.Tensor, got {type(inpt)}"
        mean = self.mean.to(inpt.device)
        std = self.std.to(inpt.device)
        return ((inpt * (std + self.eps)) + mean).to(dtype=inpt.dtype, device=inpt.device)


class MinMaxNormalize(v2.Transform):
    """Per-feature Min-Max normalization transform for tensors.

    Maps each feature to the [0, 1] range using x' = (x - min) / (max - min + eps),
    assuming feature dimension is last. Works with shapes like (T, F) or (N, T, F).

    Args:
        min_vals: Sequence of per-feature minimum values of length F
        max_vals: Sequence of per-feature maximum values of length F
        eps: Small constant added to the denominator to avoid division by zero (default: 1e-8)

    Example:
        >>> t = torch.randn(8, 100, 5)
        >>> tr = MinMaxNormalize(min_vals=[-5]*5, max_vals=[5]*5)
        >>> y = tr(t)
        >>> y.shape
        torch.Size([8, 100, 5])
    """
    def __init__(self, min_vals: Sequence[float], max_vals: Sequence[float], eps: float = 1e-8):
        super().__init__()
        assert len(min_vals) == len(max_vals), "min_vals and max_vals must have the same length!"
        self.min_vals = torch.as_tensor(min_vals, dtype=torch.float32)
        self.max_vals = torch.as_tensor(max_vals, dtype=torch.float32)
        self.eps = eps

    def transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> Any:
        """Apply Min-Max normalization to the input tensor.

        Assumes feature dimension is last and min/max are length-F vectors.
        """
        assert type(inpt) == torch.Tensor, f"'MinMaxNormalize' expects torch.Tensor, got {type(inpt)}"
        minv = self.min_vals.to(inpt.device)
        maxv = self.max_vals.to(inpt.device)
        return ((inpt - minv) / ((maxv - minv) + self.eps)).to(dtype=inpt.dtype, device=inpt.device)

    def inverse_transform(self, inpt: torch.Tensor) -> torch.Tensor:
        """Invert Min-Max normalization back to original scale.

        x = inpt * (max - min + eps) + min
        """
        assert type(inpt) == torch.Tensor, f"'MinMaxNormalize' expects torch.Tensor, got {type(inpt)}"
        minv = self.min_vals.to(inpt.device)
        maxv = self.max_vals.to(inpt.device)
        return (inpt * ((maxv - minv) + self.eps) + minv).to(dtype=inpt.dtype, device=inpt.device)