import torch
import torch.nn as nn
import math


class TransformerRegression(nn.Module):
    """
    Transformer for time-series regression (HVAC control)
    Input: (batch, seq_len, input_dim) - sequence of sensor readings
    Output: (batch, output_dim) - actuator commands
    """
    def __init__(
        self, 
        input_dim,           # Number of input features per timestep
        output_dim,          # Number of outputs (5 for your HVAC)
        d_model=128,         # Model dimension
        num_heads=4,         # Number of attention heads
        num_layers=3,        # Number of transformer layers
        d_ff=512,            # Feedforward dimension
        max_seq_length=100,  # Maximum sequence length
        dropout=0.1,
        pooling='last'       # 'last', 'mean', or 'cls'
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        
        # Regression head
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_dim)
        self.relu = nn.ReLU()
        
        # Optional: learnable CLS token
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) optional padding mask
        Returns:
            output: (batch, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.dropout(self.positional_encoding(x))
        
        # Add CLS token if using
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # Create attention mask (optional)
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        else:
            attn_mask = None
        
        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attn_mask)
        
        # Pool sequence dimension
        if self.pooling == 'last':
            x = x[:, -1, :]  # Take last timestep
        elif self.pooling == 'mean':
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                sum_embeddings = (x * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                x = sum_embeddings / sum_mask.clamp(min=1e-9)
            else:
                x = x.mean(dim=1)
        elif self.pooling == 'cls':
            x = x[:, 0, :]  # Take CLS token
        
        # Regression head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
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