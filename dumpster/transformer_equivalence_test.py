from pathlib import Path
import sys; sys.path.append(str(Path(__file__).parent.parent))

import torch
import random
import numpy as np

from typing import TypedDict, Literal, Tuple
from nnlibrary.models import TransformerRegression, TransformerRegressionOptimized
from nnlibrary.models.transformer import MultiHeadAttention as BasicMHA
from nnlibrary.models.transformer2 import MultiHeadAttentionOptimized as OptMHA
from nnlibrary.utils.transformer_weight_mapping import basic_to_optimized, optimized_to_basic


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    # Test matrix: devices, pooling modes, shapes, AMP
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))

    pooling_modes: Tuple[Literal['last','mean','cls'], ...] = ('last','mean','cls')
    shapes = [ # Batch size, sequence length
        (1, 1),   # minimal
        (32, 16),  # small
        (512, 32),  # medium
    ]

    # Hyperparameters must match between models
    class HParams(TypedDict):
        input_dim: int
        output_dim: int
        dim_model: int
        num_heads: int
        num_layers: int
        dim_ff: int
        max_seq_length: int
        dropout: float
        pooling: Literal['last','mean','cls']

    base_hparams: HParams = {
        'input_dim': 37,
        'output_dim': 5,
        'dim_model': 64,
        'num_heads': 4,
        'num_layers': 2,
        'dim_ff': 256,
        'max_seq_length': 256,  # large enough to cover tested T
        'dropout': 0.1,
        'pooling': 'last',
    }

    def build_models(hp: HParams, device: torch.device):
        basic = TransformerRegression(**hp).to(device)  # type: ignore[call-arg]
        opt = TransformerRegressionOptimized(**hp).to(device)  # type: ignore[call-arg]
        # Align weights
        basic_to_optimized(basic, opt)
        basic.eval(); opt.eval()
        return basic, opt

    def forward_equiv(device: torch.device, pooling: str, B: int, T: int, use_amp: bool=False):
        hp: HParams = base_hparams.copy()
        hp['pooling'] = pooling  # type: ignore
        basic, opt = build_models(hp, device)
        x = torch.randn(B, T, hp['input_dim'], device=device)
        ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (use_amp and device.type=='cuda') else torch.inference_mode()
        with ctx:
            yb = basic(x)
            yo = opt(x)
        atol = 5e-3 if (use_amp and device.type=='cuda') else 1e-5
        rtol = atol
        ok = torch.allclose(yb, yo, atol=atol, rtol=rtol)
        diff = (yb - yo).abs().max().item()
        print(f"[FWD] device={device} pooling={pooling} BxT={B}x{T} amp={use_amp} : allclose={ok} max_diff={diff:.3e}")
        return ok

    def masked_mean_pooling_equiv(device: torch.device, B: int, T: int):
        # Mask affects pooling only in current impl; use pooling='mean'
        hp: HParams = base_hparams.copy()
        hp['pooling'] = 'mean'  # type: ignore
        basic, opt = build_models(hp, device)
        x = torch.randn(B, T, hp['input_dim'], device=device)
        # Random mask with at least one valid token per sequence
        mask = (torch.rand(B, T, device=device) > 0.2)
        mask[:, 0] = True
        with torch.inference_mode():
            yb = basic(x, mask=mask.float())  # basic pooling path expects float
            yo = opt(x, mask=mask)            # optimized pooling path supports bool
        ok = torch.allclose(yb, yo, atol=1e-5, rtol=1e-5)
        diff = (yb - yo).abs().max().item()
        print(f"[MASKED MEAN] device={device} BxT={B}x{T} allclose={ok} max_diff={diff:.3e}")
        return ok

    def grad_equiv_training(device: torch.device, pooling: str, B: int, T: int):
        # Compare gradients in training with dropout disabled
        hp: HParams = base_hparams.copy()
        hp['pooling'] = pooling  # type: ignore
        hp['dropout'] = 0.0      # type: ignore
        basic = TransformerRegression(**hp).to(device)  # type: ignore[call-arg]
        opt = TransformerRegressionOptimized(**hp).to(device)  # type: ignore[call-arg]
        basic_to_optimized(basic, opt)
        basic.train(); opt.train()
        x = torch.randn(B, T, hp['input_dim'], device=device)
        target = torch.randn(B, hp['output_dim'], device=device)
        basic.zero_grad(set_to_none=True)
        opt.zero_grad(set_to_none=True)
        yb = basic(x)
        yo = opt(x)
        lb = torch.nn.functional.mse_loss(yb, target)
        lo = torch.nn.functional.mse_loss(yo, target)
        lb.backward(); lo.backward()
        # Compare a few grads
        pairs = [
            ('input_projection.weight', basic.input_projection.weight.grad, opt.input_projection.weight.grad),
            ('enc0_ffn_fc2.weight', basic.encoder_layers[0].feed_forward.fc2.weight.grad, opt.encoder_layers[0].feed_forward.fc2.weight.grad),  # type: ignore[attr-defined]
            ('fc2.weight', basic.fc2.weight.grad, opt.fc2.weight.grad),
        ]
        ok_all = True
        for name, g1, g2 in pairs:
            if g1 is None or g2 is None:
                print(f"[GRAD] {name}: missing grad")
                ok_all = False
                continue
            ok = torch.allclose(g1, g2, atol=1e-5, rtol=1e-5)
            diff = (g1 - g2).abs().max().item()
            print(f"[GRAD] device={device} pooling={pooling} {name}: allclose={ok} max_diff={diff:.3e}")
            ok_all &= ok
        return ok_all

    def weight_roundtrip(device: torch.device):
        hp: HParams = base_hparams.copy()
        basic1 = TransformerRegression(**hp).to(device)  # type: ignore[call-arg]
        opt = TransformerRegressionOptimized(**hp).to(device)  # type: ignore[call-arg]
        # Copy basic1 -> opt -> basic2
        basic_to_optimized(basic1, opt)
        basic2 = TransformerRegression(**hp).to(device)  # type: ignore[call-arg]
        optimized_to_basic(opt, basic2)
        # Compare a few key parameter tensors
        keys = [
            ('input_projection.weight', basic1.input_projection.weight, basic2.input_projection.weight),
            ('fc2.weight', basic1.fc2.weight, basic2.fc2.weight),
            ('enc0_ffn_fc2.weight', basic1.encoder_layers[0].feed_forward.fc2.weight, basic2.encoder_layers[0].feed_forward.fc2.weight),  # type: ignore[attr-defined]
        ]
        ok_all = True
        for name, p1, p2 in keys:
            diff = (p1 - p2).abs().max().item()
            ok = torch.allclose(p1, p2, atol=1e-8, rtol=1e-8)
            print(f"[ROUNDTRIP] device={device} {name}: allclose={ok} max_diff={diff:.3e}")
            ok_all &= ok
        return ok_all

    # Reference comparisons vs torch.nn.MultiheadAttention
    def ref_compare_basic_mha(device: torch.device, B: int, T: int, d_model: int, num_heads: int):
        basic = BasicMHA(d_model=d_model, num_heads=num_heads).to(device)
        ref = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=0.0, batch_first=True).to(device)
        # Map weights basic -> ref
        with torch.no_grad():
            ref.in_proj_weight.copy_(torch.cat([basic.W_q.weight, basic.W_k.weight, basic.W_v.weight], dim=0))
            if ref.in_proj_bias is not None:
                q_bias = basic.W_q.bias if basic.W_q.bias is not None else torch.zeros(d_model, device=device)
                k_bias = basic.W_k.bias if basic.W_k.bias is not None else torch.zeros(d_model, device=device)
                v_bias = basic.W_v.bias if basic.W_v.bias is not None else torch.zeros(d_model, device=device)
                ref.in_proj_bias.copy_(torch.cat([q_bias, k_bias, v_bias], dim=0))
            ref.out_proj.weight.copy_(basic.W_o.weight)
            if ref.out_proj.bias is not None and basic.W_o.bias is not None:
                ref.out_proj.bias.copy_(basic.W_o.bias)
        # Inputs and mask
        x = torch.randn(B, T, d_model, device=device)
        keep_mask = (torch.rand(B, T, device=device) > 0.2)
        attn_mask_basic = keep_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T), True=keep
        key_padding_mask = ~keep_mask  # (B,T), True=mask (pad)
        basic.eval(); ref.eval()
        with torch.inference_mode():
            yb = basic(x, x, x, mask=attn_mask_basic)
            yr, _ = ref(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        ok = torch.allclose(yb, yr, atol=1e-5, rtol=1e-5)
        diff = (yb - yr).abs().max().item()
        print(f"[REF BASIC MHA] device={device} BxT={B}x{T} d_model={d_model} heads={num_heads} : allclose={ok} max_diff={diff:.3e}")
        return ok

    def ref_compare_optimized_mha(device: torch.device, B: int, T: int, d_model: int, num_heads: int):
        opt = OptMHA(E_q=d_model, E_k=d_model, E_v=d_model, E_total=d_model, num_heads=num_heads, dropout=0.0).to(device)
        ref = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=0.0, batch_first=True).to(device)
        # Map weights opt -> ref (packed)
        with torch.no_grad():
            if hasattr(opt, 'packed_proj'):
                ref.in_proj_weight.copy_(opt.packed_proj.weight)
                if ref.in_proj_bias is not None and opt.packed_proj.bias is not None:
                    ref.in_proj_bias.copy_(opt.packed_proj.bias)
            else:
                # Separate projections path
                ref.in_proj_weight.copy_(torch.cat([opt.q_proj.weight, opt.k_proj.weight, opt.v_proj.weight], dim=0))
                if ref.in_proj_bias is not None:
                    qb = opt.q_proj.bias if opt.q_proj.bias is not None else torch.zeros(d_model, device=device)
                    kb = opt.k_proj.bias if opt.k_proj.bias is not None else torch.zeros(d_model, device=device)
                    vb = opt.v_proj.bias if opt.v_proj.bias is not None else torch.zeros(d_model, device=device)
                    ref.in_proj_bias.copy_(torch.cat([qb, kb, vb], dim=0))
            ref.out_proj.weight.copy_(opt.out_proj.weight)
            if ref.out_proj.bias is not None and opt.out_proj.bias is not None:
                ref.out_proj.bias.copy_(opt.out_proj.bias)
        # Inputs and mask
        x = torch.randn(B, T, d_model, device=device)
        keep_mask = (torch.rand(B, T, device=device) > 0.2)
        # EncoderLayerOptimized converts keep->mask and expands to (B,1,1,T). Here we can pass boolean mask in that shape directly.
        attn_mask_opt = (~keep_mask).unsqueeze(1).unsqueeze(2)  # True = mask
        key_padding_mask = ~keep_mask
        opt.eval(); ref.eval()
        with torch.inference_mode():
            yo = opt(x, x, x, attn_mask=attn_mask_opt)
            yr, _ = ref(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        ok = torch.allclose(yo, yr, atol=1e-5, rtol=1e-5)
        diff = (yo - yr).abs().max().item()
        print(f"[REF OPT MHA ] device={device} BxT={B}x{T} d_model={d_model} heads={num_heads} : allclose={ok} max_diff={diff:.3e}")
        return ok

    # Execute tests
    overall_ok = True
    for device in devices:
        # Forward tests: pooling modes and shapes
        for pooling in pooling_modes:
            for (B, T) in shapes:
                overall_ok &= forward_equiv(device, pooling, B, T, use_amp=False)
                if device.type == 'cuda':
                    overall_ok &= forward_equiv(device, pooling, B, T, use_amp=True)
        # Masked mean pooling test (mask affects pooling only)
        for (B, T) in shapes:
            overall_ok &= masked_mean_pooling_equiv(device, B, T)
        # Reference MHA comparisons (basic and optimized) for a couple of settings
        for (B, T) in [(2, 16), (3, 8)]:
            overall_ok &= ref_compare_basic_mha(device, B=B, T=T, d_model=64, num_heads=4)
            overall_ok &= ref_compare_optimized_mha(device, B=B, T=T, d_model=64, num_heads=4)
        # Gradient parity in training (dropout=0)
        for pooling in pooling_modes:
            overall_ok &= grad_equiv_training(device, pooling, B=4, T=16)
        # Weight roundtrip
        overall_ok &= weight_roundtrip(device)

    print("\nSUMMARY: ", "PASS" if overall_ok else "FAIL")


if __name__ == '__main__':
    main()
