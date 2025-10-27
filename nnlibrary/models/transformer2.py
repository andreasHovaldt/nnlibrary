import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerRegressionOptimized(nn.Module):
    """
    Optimized Transformer for HVAC time-series regression
    Uses PyTorch's efficient scaled_dot_product_attention
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128, # Internal representation dimension (model width), "How much information can the model remember about each timestep?"
        num_heads: int = 4, # Number of parallel attention heads, "How many different ways can the model look at relationships?"
        num_layers: int = 3, # Number of stacked transformer encoder layers (model depth), "How many times should the model refine its understanding?"
        d_ff: int = 512, # Feedforward network hidden dimension, "Internal processing capacity after attention"
        max_seq_length: int = 100, # Maximum sequence length (for positional encoding buffer), "What's the longest sequence I'll ever feed?"
        dropout: float = 0.1,
        pooling: str = 'last',
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayerOptimized(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        
        # Regression head
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_dim)
        
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) boolean mask (True = keep, False = mask)
        Returns:
            output: (batch, output_dim)
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Add CLS token if needed
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        
        # Pool sequence
        if self.pooling == 'last':
            x = x[:, -1, :]
        elif self.pooling == 'mean':
            if mask is not None:
                # Masked mean
                mask_expanded = mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                x = x.mean(dim=1)
        elif self.pooling == 'cls':
            x = x[:, 0, :]
        
        # Regression head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output


class EncoderLayerOptimized(nn.Module):
    """Optimized encoder layer using efficient attention"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        
        self.self_attn = MultiHeadAttentionOptimized(
            E_q=d_model,
            E_k=d_model,
            E_v=d_model,
            E_total=d_model,
            nheads=num_heads,
            dropout=dropout,
        )
        
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class MultiHeadAttentionOptimized(nn.Module):
    """
    Optimized multi-head attention using PyTorch's scaled_dot_product_attention
    Based on: https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    """
    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        
        # Check if Q, K, V have same embedding dim
        self._qkv_same_embed_dim = (E_q == E_k == E_v)
        
        if self._qkv_same_embed_dim:
            # Packed projection: single linear layer for Q, K, V (faster!)
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias)
        else:
            # Separate projections
            self.q_proj = nn.Linear(E_q, E_total, bias=bias)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias)
        
        self.out_proj = nn.Linear(E_total, E_q, bias=bias)
        
        assert E_total % nheads == 0, "E_total must be divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query: (N, L_q, E_q)
            key: (N, L_k, E_k)
            value: (N, L_v, E_v)
            attn_mask: (N, L_q, L_k) or (L_q, L_k) boolean mask
            is_causal: If True, apply causal mask
        Returns:
            output: (N, L_q, E_q)
        """
        # Step 1: Input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                # All same tensor: single packed projection
                qkv = self.packed_proj(query)
                query, key, value = torch.chunk(qkv, 3, dim=-1)
            else:
                # Same dim but different tensors: split weights
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias = k_bias = v_bias = None
                
                query = F.linear(query, q_weight, q_bias)
                key = F.linear(key, k_weight, k_bias)
                value = F.linear(value, v_weight, v_bias)
        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)
        
        # Step 2: Split heads
        # (N, L, E_total) -> (N, L, nheads, E_head) -> (N, nheads, L, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        
        # Handle attention mask
        # Convert boolean mask to additive mask for SDPA
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            # PyTorch SDPA expects: True = attend, False = mask
            # But it wants None or float mask, so we don't convert
            pass
        
        # Step 3: Scaled dot-product attention (OPTIMIZED!)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        # Step 4: Combine heads
        # (N, nheads, L, E_head) -> (N, L, nheads, E_head) -> (N, L, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        
        # Step 5: Output projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)] # type: ignore