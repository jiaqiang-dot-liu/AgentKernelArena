# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the MXFP4 GEMM ``gemm_a4w4`` (a4w4 blockscale).

Computes ``out = a @ w.T`` where the activation ``a`` (``[M, K]``) and weight
``w`` (``[N, K]``) are quantized to MXFP4 (``float4_e2m1fn_x2`` values with e8m0
per-1x32 block scales along K), matching the AMD runtime MXFP4 GEMM: each
operand is dequantized (e2m1 value times its e8m0 block scale), the GEMM
accumulates in fp32, and the result is truncated to bf16.

The MXFP4 (f32->e2m1 rounding, saturation, denormals) and e8m0 block-scale
numerics implemented here match AMD's reference quantizer bit-for-bit, so the
dequantized values are identical to the values the hardware GEMM consumes.
"""
import torch
import torch.nn as nn

# MXFP4 (e2m1) decode table indexed by the 4-bit code (sign in bit 3).
_MXFP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_BLOCK = 32
# log2 of the largest power of two <= F4E2M1_MAX(=6); the EVEN-mode e8m0 scale
# divides the per-block exponent by this power of two.
_FP4_TARGET_MAX_POW2 = 2
# int32 two's-complement representations of the bit constants used by the EVEN
# scale rounding (0x00200000 round bias, 0xFF800000 sign+exponent mask).
_EVEN_ROUND_BIAS = 0x00200000
_EXP_MASK_I32 = -8388608  # 0xFF800000 as signed int32


def _f32_to_e8m0_even(amax):
    """EVEN-mode (Quark even_round) per-block e8m0 scale, bit-for-bit matching
    the AMD runtime MXFP4 quantizer: round amax's mantissa to even, take
    ``floor(log2(.)) - 2``, clamp the unbiased exponent to ``[-127, 127]`` and
    re-bias to the uint8 e8m0 code."""
    u32 = amax.contiguous().view(torch.int32)
    rounded = (u32 + _EVEN_ROUND_BIAS) & _EXP_MASK_I32
    biased_exp = (rounded >> 23) & 0xFF
    unbiased = (biased_exp - 127 - _FP4_TARGET_MAX_POW2).clamp(-127, 127)
    return (unbiased + 127).to(torch.uint8)


def _e8m0_to_f32(scale_e8m0_biased):
    """Decode biased e8m0 exponents (uint8) back to fp32 power-of-two scales."""
    scale_e8m0_biased = scale_e8m0_biased.view(torch.uint8)
    zero_case = scale_e8m0_biased == 0
    nan_case = scale_e8m0_biased == 0xFF
    scale_f32 = scale_e8m0_biased.to(torch.int32) << 23
    scale_f32[zero_case] = 0x00400000
    scale_f32[nan_case] = 0x7F800001
    return scale_f32.view(torch.float32)


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


def _mxfp4_dequant(x):
    """MXFP4 per-1x32 e8m0 quantize+dequantize over the last dim, returning the
    fp32 values the hardware GEMM sees."""
    shape = x.shape
    xb = x.float().reshape(-1, _BLOCK)
    max_abs = torch.amax(torch.abs(xb), dim=1)
    scale_e8m0 = _f32_to_e8m0_even(max_abs)
    scale_f32 = _e8m0_to_f32(scale_e8m0).view(-1, 1)
    codes = _f32_to_e2m1_codes(xb / scale_f32)
    table = _MXFP4_VALUES.to(x.device)
    deq = table[codes.long()] * scale_f32
    return deq.reshape(shape)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, w):
        a_f32 = _mxfp4_dequant(a)
        w_f32 = _mxfp4_dequant(w)
        out = torch.matmul(a_f32, w_f32.transpose(0, 1))
        return out.to(torch.bfloat16)


def get_inputs():
    # Representative DeepSeek-V3 MXFP4 shape (M, N, K) = (128, 7168, 4608); the
    # harness sweeps more real shapes from dsv3_a4w4_blockscale_tuned_gemm.csv.
    m, n, k = 128, 7168, 4608
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    return [a, w]


def get_init_inputs():
    return []
