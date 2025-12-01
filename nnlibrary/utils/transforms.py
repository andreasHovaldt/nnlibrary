import torch
from torchvision.transforms import v2
from typing import Any, Sequence, Optional


class TransformBase(v2.Transform):
    def __init__(self) -> None:
        super().__init__()
        
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Method to override for custom transforms."""
        raise NotImplementedError
    
    def inverse_transform(self, inpt: torch.Tensor) -> torch.Tensor:
        """Method to override for custom transforms."""
        raise NotImplementedError
    

class TransformComposer(TransformBase):
    """
    Class for handling multiple transforms
    """

    def __init__(
        self,
        transforms: Sequence[TransformBase]
    ) -> None:
        super().__init__()
        self.transforms = transforms

    def transform(self, inpt: torch.Tensor, params: Optional[dict[str, Any]] = None) -> torch.Tensor:
        for transform in self.transforms:
            inpt = transform(inpt)
        return inpt.to(dtype=inpt.dtype, device=inpt.device)

    def inverse_transform(self, inpt: torch.Tensor) -> torch.Tensor:
        # Apply inverse in reverse order to correctly undo the composition
        for transform in reversed(self.transforms):
            inpt = transform.inverse_transform(inpt)

        return inpt.to(dtype=inpt.dtype, device=inpt.device)
        

class Standardize(TransformBase):
    """Per-feature standardization transform for tensors.

    Applies standardized values x' = (x - mean) / (std + eps) using PyTorch broadcasting.
    This implementation assumes the feature dimension is last and that ``mean`` and ``std``
    are 1D of length F matching the last dimension. For example: (T, F) or (N, T, F).

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
        print("Using standardized targets!")

    def transform(self, inpt: torch.Tensor, params: Optional[dict[str, Any]] = None) -> Any:
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


class MinMaxNormalize(TransformBase):
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
        print("Using normalized targets!")

    def transform(self, inpt: torch.Tensor, params: Optional[dict[str, Any]] = None) -> Any:
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
    
    
class Absolute2Relative(TransformBase):
    """Transform y to relative heating/cooling setpoints using fixed zone bounds.

    Usage:
      - transform = Absolute2Relative(heating_idx=2, cooling_idx=3, zone_sp_min=18.0, zone_sp_max=24.0)
        y_rel = transform(y_abs)
      - y_abs = transform.inverse_transform(y_rel)

    Only the heating and cooling setpoint indices are transformed; other outputs are passed through.
    """

    def __init__(
        self,
        zone_sp_min: float,
        zone_sp_max: float,
        heating_idx: int = 2,
        cooling_idx: int = 3,
        clamp: bool = False,
    ) -> None:
        assert (zone_sp_max - zone_sp_min) > 1e-8, "Zone max must be bigger than zone min!"
        super().__init__()
        self.heating_idx = heating_idx
        self.cooling_idx = cooling_idx
        self.clamp = clamp
        self._zmin = torch.as_tensor(zone_sp_min, dtype=torch.float32)
        self._zmax = torch.as_tensor(zone_sp_max, dtype=torch.float32)

    def _get_bounds(self, inpt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zmin_t = self._zmin.to(dtype=inpt.dtype, device=inpt.device)
        zmax_t = self._zmax.to(dtype=inpt.dtype, device=inpt.device)
        zrange = torch.clamp(zmax_t - zmin_t, 1e-8)
        return zmin_t, zmax_t, zrange

    def transform(self, inpt: torch.Tensor, params: Optional[dict[str, Any]] = None) -> torch.Tensor:
        assert type(inpt) == torch.Tensor, f"'Absolute2Relative' expects torch.Tensor, got {type(inpt)}"
        assert self.heating_idx < inpt.shape[-1], f"Heating index is out of bounds, got shape {inpt.shape}"
        assert self.cooling_idx < inpt.shape[-1], f"Cooling index is out of bounds, got shape {inpt.shape}"
        
        out = inpt.clone()
        zmin, zmax, zrange = self._get_bounds(inpt)

        relative_heating_setpoint = (inpt[..., self.heating_idx] - zmin) / zrange
        relative_cooling_setpoint = (inpt[..., self.cooling_idx] - zmin) / zrange
        
        if self.clamp:
            relative_heating_setpoint = relative_heating_setpoint.clamp(0.0, 1.0)
            relative_cooling_setpoint = relative_cooling_setpoint.clamp(0.0, 1.0)
        
        out[..., self.heating_idx] = relative_heating_setpoint
        out[..., self.cooling_idx] = relative_cooling_setpoint

        return out.to(dtype=inpt.dtype, device=inpt.device)

    def inverse_transform(self, inpt: torch.Tensor) -> torch.Tensor:
        assert type(inpt) == torch.Tensor, f"'Absolute2Relative' expects torch.Tensor, got {type(inpt)}"
        assert self.heating_idx < inpt.shape[-1], f"Heating index is out of bounds, got shape {inpt.shape}"
        assert self.cooling_idx < inpt.shape[-1], f"Cooling index is out of bounds, got shape {inpt.shape}"
        
        out = inpt.clone()
        zmin, zmax, zrange = self._get_bounds(inpt)

        absolute_heating_setpoint = inpt[..., self.heating_idx] * zrange + zmin
        absolute_cooling_setpoint = inpt[..., self.cooling_idx] * zrange + zmin
        
        if self.clamp:
            absolute_heating_setpoint = absolute_heating_setpoint.clamp(zmin, zmax)
            absolute_cooling_setpoint = absolute_cooling_setpoint.clamp(zmin, zmax)
            
        out[..., self.heating_idx] = absolute_heating_setpoint
        out[..., self.cooling_idx] = absolute_cooling_setpoint

        return out.to(dtype=inpt.dtype, device=inpt.device)
