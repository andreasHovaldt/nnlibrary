import torch
import torch.nn as nn

class DilatedCausalConv(nn.Conv1d):
    """
    The dilated causal convolution operation as defined in Figure 1a of https://arxiv.org/pdf/1803.01271
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int | tuple[int], 
        # stride: int | tuple[int] = 1, 
        padding: int = 0, 
        dilation: int | tuple[int] = 1, 
        # groups: int = 1, 
        # bias: bool = True, 
        # padding_mode: str = "zeros", 
        device=None, 
        dtype=None
    ) -> None:
        super().__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            # stride=stride, 
            padding=padding, 
            dilation=dilation, 
            # groups=groups, 
            # bias=bias, 
            # padding_mode=padding_mode, 
            device=device, 
            dtype=dtype,
        )
        
        # The amount of futures which are discarded
        # The conv layer padding is symetrical, thus the padding is
        #  of course added on both sides of the time series sequence.
        #  This means the padding adds "future states" to the right 
        #  of the sequence, which must be removed.
        self.chomp_size = padding
    
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        # Perform Dilated and padded convolution
        x = super().forward(input)
        
        # Chomp the padded future states
        x = x if self.chomp_size == 0 else x[:, :, :-self.chomp_size]
        
        return x