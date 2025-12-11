import torch
from typing import Optional

try:
    # Local imports (avoid circulars by importing within functions if needed)
    from nnlibrary.models.transformer import TransformerRegression, EncoderLayer as EncoderLayerBasic
    from nnlibrary.models.transformer import TransformerRegression, EncoderLayer as EncoderLayerOpt
except Exception:
    # Types will be resolved at runtime when functions are called
    TransformerRegression = object  # type: ignore
    TransformerRegression = object  # type: ignore


def _copy_linear(dst: torch.nn.Linear, src: torch.nn.Linear) -> None:
    assert dst.weight.shape == src.weight.shape, f"Linear weight shape mismatch: {dst.weight.shape} vs {src.weight.shape}"
    dst.weight.data.copy_(src.weight.data)
    if dst.bias is not None and src.bias is not None:
        assert dst.bias.shape == src.bias.shape
        dst.bias.data.copy_(src.bias.data)


def basic_to_optimized(basic_model, optimized_model) -> None:
    """
    Copy weights from TransformerRegression (basic) to TransformerRegressionOptimized.
    Assumes identical hyperparameters (d_model, num_heads, num_layers, d_ff, pooling).
    """
    # Input projection
    _copy_linear(optimized_model.input_projection, basic_model.input_projection)

    # Positional encoding buffer
    if hasattr(basic_model.positional_encoding, 'pe') and hasattr(optimized_model.positional_encoding, 'pe'):
        assert optimized_model.positional_encoding.pe.shape == basic_model.positional_encoding.pe.shape
        optimized_model.positional_encoding.pe.data.copy_(basic_model.positional_encoding.pe.data)

    # Encoder layers
    for i, (b_layer, o_layer) in enumerate(zip(basic_model.encoder_layers, optimized_model.encoder_layers)):
        # Self Attention: pack W_q, W_k, W_v into packed_proj
        b_attn = b_layer.self_attn
        o_attn = o_layer.self_attn

        d_model = b_attn.d_model
        # Pack Q, K, V into a single weight matrix (out, in) stacked along out-dim
        # Expected order: [Q; K; V]
        if hasattr(o_attn, 'packed_proj'):
            qkv_weight = torch.cat([
                b_attn.W_q.weight.data,
                b_attn.W_k.weight.data,
                b_attn.W_v.weight.data,
            ], dim=0)
            qkv_bias: Optional[torch.Tensor]
            if b_attn.W_q.bias is not None:
                qkv_bias = torch.cat([
                    b_attn.W_q.bias.data,
                    b_attn.W_k.bias.data,
                    b_attn.W_v.bias.data,
                ], dim=0)
            else:
                qkv_bias = None

            assert o_attn.packed_proj.weight.shape == qkv_weight.shape, \
                f"packed_proj weight mismatch: {o_attn.packed_proj.weight.shape} vs {qkv_weight.shape}"
            o_attn.packed_proj.weight.data.copy_(qkv_weight)
            if qkv_bias is not None and o_attn.packed_proj.bias is not None:
                assert o_attn.packed_proj.bias.shape == qkv_bias.shape
                o_attn.packed_proj.bias.data.copy_(qkv_bias)
        else:
            # Separate projections path
            _copy_linear(o_attn.q_proj, b_attn.W_q)
            _copy_linear(o_attn.k_proj, b_attn.W_k)
            _copy_linear(o_attn.v_proj, b_attn.W_v)

        # Output projection
        _copy_linear(o_attn.out_proj, b_attn.W_o)

        # Feed Forward
        _copy_linear(o_layer.feed_forward.fc1, b_layer.feed_forward.fc1)
        _copy_linear(o_layer.feed_forward.fc2, b_layer.feed_forward.fc2)

        # LayerNorms
        o_layer.norm1.weight.data.copy_(b_layer.norm1.weight.data)
        o_layer.norm1.bias.data.copy_(b_layer.norm1.bias.data)
        o_layer.norm2.weight.data.copy_(b_layer.norm2.weight.data)
        o_layer.norm2.bias.data.copy_(b_layer.norm2.bias.data)

    # Pooling-specific params
    if getattr(basic_model, 'pooling', 'last') == 'cls' and hasattr(basic_model, 'cls_token') and hasattr(optimized_model, 'cls_token'):
        assert optimized_model.cls_token.shape == basic_model.cls_token.shape
        optimized_model.cls_token.data.copy_(basic_model.cls_token.data)

    # Regression head
    _copy_linear(optimized_model.fc1, basic_model.fc1)
    _copy_linear(optimized_model.fc2, basic_model.fc2)


def optimized_to_basic(optimized_model, basic_model) -> None:
    """
    Copy weights from TransformerRegressionOptimized to TransformerRegression.
    Assumes identical hyperparameters.
    """
    # Input projection
    _copy_linear(basic_model.input_projection, optimized_model.input_projection)

    # Positional encoding buffer
    if hasattr(basic_model.positional_encoding, 'pe') and hasattr(optimized_model.positional_encoding, 'pe'):
        assert basic_model.positional_encoding.pe.shape == optimized_model.positional_encoding.pe.shape
        basic_model.positional_encoding.pe.data.copy_(optimized_model.positional_encoding.pe.data)

    # Encoder layers
    for i, (o_layer, b_layer) in enumerate(zip(optimized_model.encoder_layers, basic_model.encoder_layers)):
        o_attn = o_layer.self_attn
        b_attn = b_layer.self_attn

        # Unpack packed_proj into W_q, W_k, W_v
        if hasattr(o_attn, 'packed_proj'):
            q_weight, k_weight, v_weight = torch.chunk(o_attn.packed_proj.weight.data, 3, dim=0)
            # Weights
            assert b_attn.W_q.weight.shape == q_weight.shape
            assert b_attn.W_k.weight.shape == k_weight.shape
            assert b_attn.W_v.weight.shape == v_weight.shape
            b_attn.W_q.weight.data.copy_(q_weight)
            b_attn.W_k.weight.data.copy_(k_weight)
            b_attn.W_v.weight.data.copy_(v_weight)
            # Biases
            if o_attn.packed_proj.bias is not None:
                q_bias, k_bias, v_bias = torch.chunk(o_attn.packed_proj.bias.data, 3, dim=0)
                if b_attn.W_q.bias is not None:
                    b_attn.W_q.bias.data.copy_(q_bias)
                if b_attn.W_k.bias is not None:
                    b_attn.W_k.bias.data.copy_(k_bias)
                if b_attn.W_v.bias is not None:
                    b_attn.W_v.bias.data.copy_(v_bias)
        else:
            _copy_linear(b_attn.W_q, o_attn.q_proj)
            _copy_linear(b_attn.W_k, o_attn.k_proj)
            _copy_linear(b_attn.W_v, o_attn.v_proj)

        # Output projection
        _copy_linear(b_attn.W_o, o_attn.out_proj)

        # Feed Forward
        _copy_linear(b_layer.feed_forward.fc1, o_layer.feed_forward.fc1)
        _copy_linear(b_layer.feed_forward.fc2, o_layer.feed_forward.fc2)

        # LayerNorms
        b_layer.norm1.weight.data.copy_(o_layer.norm1.weight.data)
        b_layer.norm1.bias.data.copy_(o_layer.norm1.bias.data)
        b_layer.norm2.weight.data.copy_(o_layer.norm2.weight.data)
        b_layer.norm2.bias.data.copy_(o_layer.norm2.bias.data)

    # Pooling-specific params
    if getattr(basic_model, 'pooling', 'last') == 'cls' and hasattr(basic_model, 'cls_token') and hasattr(optimized_model, 'cls_token'):
        assert basic_model.cls_token.shape == optimized_model.cls_token.shape
        basic_model.cls_token.data.copy_(optimized_model.cls_token.data)

    # Regression head
    _copy_linear(basic_model.fc1, optimized_model.fc1)
    _copy_linear(basic_model.fc2, optimized_model.fc2)
