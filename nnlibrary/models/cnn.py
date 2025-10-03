import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from nnlibrary.utils.operations import DilatedCausalConv


class TCN(nn.Module):
    """
    Temporal Convolutional Network

    https://arxiv.org/pdf/1803.01271
    
    """
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        num_classes: int,
        hidden_layer_sizes: list | None,
        kernel_size: int = 3,
        dropout: float = 0.3,
        dropout_type: str = "channel",
    ) -> None:
        """
        Temporal Convolutional Network (TCN) for sequence classification.

        Input shape is (batch, time, features). Internally this is transposed to (batch, features, time)
        so 1D convolutions operate along the time axis. The network stacks residual blocks with exponentially
        increasing dilations (1, 2, 4, ...). Each block has two dilated causal convolutions (with weight norm),
        ReLU and dropout, plus a residual connection (1x1 conv when channel sizes differ). A global average
        pool over time and a linear classifier produce the logits.

        Receptive field: for N residual blocks and kernel size K (same for all blocks) with dilations 2^i,
        the temporal receptive field is R = 1 + (K - 1) * (2^N - 1). At time index t (0-based), a unit can
        depend on up to min(R, t+1) past steps. If sequence_length >> R, no single position "sees" the entire
        window; however, because we apply global average pooling over time, all timesteps still contribute via
        different positions. If you instead use only the last time step output, consider choosing R >= sequence_length.

        Args:
            input_dim: Number of input features (C).
            sequence_length: Sequence/window length (L). Used to warn when L < receptive_field.
            num_classes: Number of output classes.
            hidden_layer_sizes: List of output channel sizes for each residual block. Defaults to [64, 64, 128, 128] when None.
            kernel_size: Temporal kernel size for all convolutions. Defaults to 3.
            dropout: Dropout probability used in each residual block.
            dropout_type: "feature" for element-wise dropout (nn.Dropout) or "channel" for channel-wise dropout over time
                          (nn.Dropout1d). Channel-wise is often preferred for temporal convs.

        Attributes:
            receptive_field (int): Computed temporal receptive field R as described above.
        """
        
        super().__init__()
        
        # Set default values
        if hidden_layer_sizes is None: hidden_layer_sizes = [64, 64, 128, 128]
        
        # Add input size to layer list
        channels = [input_dim] + hidden_layer_sizes
        
        # Compute receptive field for information; warn if sequence is shorter than receptive field
        # R = 1 + (K - 1) * sum_{i=0..N-1} 2^i = 1 + (K - 1) * (2^N - 1)
        # R=receptive field, K=kernel size, N=number of hidden layers
        
        num_blocks = len(hidden_layer_sizes)
        self.receptive_field = 1 + (kernel_size - 1) * (2**num_blocks - 1)
        if sequence_length is not None and sequence_length < self.receptive_field:
            warnings.warn(
                f"TCN receptive field ({self.receptive_field}) exceeds sequence_length ({sequence_length}). "
                "Outputs near the start of the sequence will rely on left padding (zeros).",
                RuntimeWarning,
            )
        
        # Build the residual blocks of the TCN
        self.tcn_residual_blocks = nn.ModuleList()
        for i in range(len(hidden_layer_sizes)):
            dilation = 2 ** i # As I understand, the dilation factor increases the receptive field of the conv operations
            self.tcn_residual_blocks.append(
                TCNResidualBlock(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    dropout_type=dropout_type,
                )
            )
        
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Linear(in_features=channels[-1], out_features=num_classes)
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.transpose(1, 2) # (batch, time, features) -> (batch, features, time)
        # Now the input represents the individual features over time
        # This lets the torch conv operation learn the relation between how each individual feature relates to the outcome through time
        # tldr: The conv layer convolves through the time dimension instead of the feature dimension
        
        # Encoder
        for residual_block in self.tcn_residual_blocks:
            x = residual_block(x)
        
        # Decoder
        x = self.global_pool(x).squeeze(-1)
        logits = self.classifier(x)
        
        return logits
        
        
        
    

class TCNResidualBlock(nn.Module):
    """
    The residual block of the Temporal Convolutional Network
    
    Taken from: https://arxiv.org/pdf/1803.01271
    
    Can be seen in figure (1)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        dilation: int, 
        dropout: float,
        dropout_type: str,
    ) -> None:
        super().__init__()
        
        # Here is a good visualization of what the different kernel size, padding and dilation parameters do to the convolution
        # https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        
        padding = (kernel_size - 1) * dilation
        
        self.dilated_causal_conv1 = nn.utils.parametrizations.weight_norm(DilatedCausalConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        ))
        
        self.dilated_causal_conv2 = nn.utils.parametrizations.weight_norm(DilatedCausalConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        ))
        
        self.relu = nn.ReLU()
        
        if dropout_type == "feature": self.dropout = nn.Dropout(p=dropout) # Feature (element-wise) dropout = finer-grained noise, could be harsher on temporal features
        elif dropout_type == "channel": self.dropout = nn.Dropout1d(p=dropout) # Channel-wise dropout = coarser, encourages redundancy across channels while keeping time structure intact.
        else: raise ValueError(f"Dropout type '{dropout_type}' is invalid!")
        
        # From Figure 1b text: A 1x1 convolution is added when residual input and output have different dimensions
        self.residual_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape -> (batch, features, time)
        
        residual = self.residual_conv(x)

        x = self.dilated_causal_conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.dilated_causal_conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return F.relu(x + residual)
    