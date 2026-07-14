# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the mixed FP8/MXFP4 GEMM ``gemm_a8wfp4``.

Computes ``out = a @ w.T`` where the activation ``a`` (``[M, K]``) is quantized
to FP8 (``float8_e4m3fn``) with a per-token scale and the weight ``w``
(``[N, K]``) is quantized to MXFP4 (e2m1 values with e8m0 per-1x32 block scales
along K), matching the AMD runtime ``gemm_a8wfp4``:

- activation: one FP8 scale per row of ``a`` (``x_scale`` is ``[M, 1]``);
- weight: packed FP4 codes (``[N, K/2]``, two e2m1 values per byte) plus an e8m0
  scale per 32-element K block (``w_scale`` is ``[N, K/32]``, uint8);
- the GEMM dequantizes each operand (FP8 value times its row scale, FP4 value
  times its block scale), accumulates in fp32, and truncates to bf16.

``quantize_a8wfp4`` is the single source of the quantized operands and scales
(the harness reuses it to drive the real AMD runtime op), so the reference and
the hardware kernel operate on byte-identical inputs and differ only by fp32
accumulation order.
"""
import torch
import torch.nn as nn

# e4m3fn saturation magnitude used as the per-token activation scale divisor.
_FP8_MAX = 448.0
# MXFP4 block size along K (hardware-fixed) and the largest e2m1 magnitude.
_SCALE_GROUP_SIZE = 32
_FP4_MAX = 6.0

# MXFP4 (e2m1) decode table indexed by the 4-bit code (sign in bit 3).
_MXFP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _f32_to_e2m1_codes(x):
    """Round fp32 values to MXFP4 (e2m1) 4-bit codes, saturating out-of-range
    magnitudes and handling denormals (adapted from the torchao FP utilities)."""
    EBITS, MBITS = 2, 1
    EBITS_F32, MBITS_F32 = 8, 23
    F32_EXP_BIAS = (1 << (EBITS_F32 - 1)) - 1
    exp_bias = (1 << (EBITS - 1)) - 1
    max_int = (1 << (EBITS + MBITS)) - 1
    sign_mask = 1 << (EBITS + MBITS)
    magic_adder = (1 << (MBITS_F32 - MBITS - 1)) - 1
    max_normal = 2 ** ((1 << EBITS) - 1 - exp_bias) * (
        ((1 << (MBITS + 1)) - 1) / (2**MBITS)
    )
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (F32_EXP_BIAS - exp_bias) + (MBITS_F32 - MBITS) + 1
    denorm_mask_int = denorm_exp << MBITS_F32
    denorm_mask_float = torch.tensor(
        denorm_mask_int, dtype=torch.int32
    ).view(torch.float32)

    x = x.float().view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(
        torch.logical_not(saturate_mask), x < min_normal
    )
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - MBITS)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - MBITS)
    normal_x = normal_x.to(torch.uint8)

    codes = torch.full_like(x, max_int, dtype=torch.uint8)
    codes = torch.where(denormal_mask, denormal_x, codes)
    codes = torch.where(normal_mask, normal_x, codes)

    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - MBITS - EBITS)
    sign_lp = sign_lp.to(torch.uint8) & sign_mask
    return (codes | sign_lp).to(torch.uint8)


def _quantize_fp8_pertoken(a):
    """Per-token FP8 quantization of ``a`` (``[M, K]``).

    Returns the FP8 codes and the fp32 row scales (``[M, 1]``)."""
    af = a.float()
    amax = af.abs().amax(dim=-1, keepdim=True)
    scale = (amax / _FP8_MAX).clamp_min(1e-12)
    q = (af / scale).to(torch.float8_e4m3fn)
    return q, scale


def _quantize_mxfp4_blockscale(w):
    """Per-1x32 MXFP4 quantization of ``w`` (``[N, K]``).

    Returns packed FP4 codes (``[N, K/2]``, two e2m1 values per byte) and the
    e8m0 block scales (``[N, K/32]``, uint8)."""
    n, k = w.shape
    groups = k // _SCALE_GROUP_SIZE
    wf = w.float().view(n, groups, _SCALE_GROUP_SIZE)
    amax = wf.abs().amax(dim=-1)
    block_scale = (amax / _FP4_MAX).clamp_min(1e-12)
    e8m0 = (torch.log2(block_scale) + 127.0).round().clamp_(0, 127).to(torch.uint8)
    scale_dec = torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)
    codes = _f32_to_e2m1_codes(wf / scale_dec).view(n, k)
    packed = (codes[:, 1::2].to(torch.int32) << 4) | codes[:, ::2].to(torch.int32)
    packed = packed.to(torch.uint8)
    return packed, e8m0


def quantize_a8wfp4(a, w):
    """Quantize a high-precision activation/weight pair to the FP8/MXFP4 operands
    the deployed mixed GEMM consumes.

    Returns ``(x_fp8, x_scale, w_packed, w_scale)`` with the layouts the AMD
    runtime ``gemm_a8wfp4`` expects (``x_scale`` is ``[M, 1]`` fp32, ``w_packed``
    is ``[N, K/2]`` uint8, ``w_scale`` is ``[N, K/32]`` e8m0 uint8)."""
    x_fp8, x_scale = _quantize_fp8_pertoken(a)
    w_packed, w_scale = _quantize_mxfp4_blockscale(w)
    return x_fp8, x_scale, w_packed, w_scale


def _dequant_fp8(x_fp8, x_scale):
    return x_fp8.float() * x_scale


def _dequant_mxfp4(w_packed, w_scale):
    """Decode packed FP4 codes and e8m0 block scales to fp32 weights."""
    n, kh = w_packed.shape
    k = kh * 2
    codes = torch.empty(n, k, dtype=torch.uint8, device=w_packed.device)
    codes[:, ::2] = w_packed & 0xF
    codes[:, 1::2] = w_packed >> 4
    table = _MXFP4_VALUES.to(w_packed.device)
    values = table[codes.long()].view(n, k // _SCALE_GROUP_SIZE, _SCALE_GROUP_SIZE)
    scale_dec = torch.exp2(w_scale.float() - 127.0).unsqueeze(-1)
    return (values * scale_dec).view(n, k)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, w):
        x_fp8, x_scale, w_packed, w_scale = quantize_a8wfp4(a, w)
        x_f32 = _dequant_fp8(x_fp8, x_scale)
        w_f32 = _dequant_mxfp4(w_packed, w_scale)
        out = torch.matmul(x_f32, w_f32.transpose(0, 1))
        return out.to(torch.bfloat16)


def get_inputs():
    # Representative GPT-OSS QKV-projection shape (M, N, K) = (256, 5120, 2880);
    # the harness sweeps more shapes (K a multiple of the MXFP4 32-block).
    m, n, k = 256, 5120, 2880
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    return [a, w]


def get_init_inputs():
    return []
