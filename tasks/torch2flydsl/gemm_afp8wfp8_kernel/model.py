# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the MXFP8/FP8 GEMM ``gemm_afp8wfp8``.

Computes ``out = a @ w.T`` where the activation ``a`` (``[M, K]``) is quantized
to MXFP8 (``float8_e4m3fn`` values with e8m0 per-1x32 block scales along K) and
the weight ``w`` (``[N, K]``) is quantized to FP8 with e8m0 128x128 block
scales, matching the AMD runtime ``gemm_afp8wfp8``:

- activation: FP8 codes (``[M, K]``) plus an e8m0 scale per 32-element K block
  (``x_scale`` is ``[M, K/32]``, uint8);
- weight: FP8 codes (``[N, K]``) plus an e8m0 scale per 128x128 block
  (``w_scale`` is ``[N/128, K/128]``, uint8);
- the GEMM dequantizes each operand (FP8 value times its block scale),
  accumulates in fp32, and truncates the result to bf16.

``quantize_afp8wfp8`` is the single source of the quantized operands and scales
(the harness reuses it to drive the real AMD runtime op), so the reference and
the hardware kernel operate on byte-identical inputs and differ only by fp32
accumulation order.
"""
import torch
import torch.nn as nn

# e4m3fn saturation magnitude used as the block-scale divisor on gfx950.
_FP8_MAX = 448.0
# MXFP8 activation block size along K (hardware-fixed).
_A_GROUP = 32
# FP8 weight block dims (hardware-fixed 128x128 e8m0 scaling).
_W_BLOCK_N = 128
_W_BLOCK_K = 128
# 0xFF800000 as a signed int32: keeps sign + 8-bit exponent (strips mantissa).
_E8M0_MASK_INT32 = -8388608


def _e8m0_to_f32(e8m0):
    """Decode unsigned-biased e8m0 (uint8) to fp32: ``value = 2^(b-127)``."""
    return torch.exp2((e8m0.to(torch.int32) - 127).to(torch.float32))


def _quantize_mxfp8_1x32(a):
    """Per-1x32 MXFP8 (E4M3 + E8M0) quantization of ``a`` (``[M, K]``).

    Matches the integer "round up to E8M0-representable power-of-two" sequence
    of the ``dynamic_mxfp8_quant`` kernel. Returns FP8 codes (``[M, K]``) and
    the e8m0 block scales (``[M, K/32]``, uint8)."""
    g = _A_GROUP
    x = a.float()
    m, k = x.shape
    ng = k // g
    x2 = x.reshape(m, ng, g)
    amax = x2.abs().amax(dim=-1, keepdim=True)

    amax_i32 = amax.contiguous().view(torch.int32)
    amax_i32 = (amax_i32 + 0x200000) & _E8M0_MASK_INT32
    amax_p2 = amax_i32.view(torch.float32)

    scale_unbiased = amax_p2.log2().floor() - 8
    scale_unbiased = torch.clamp(scale_unbiased, min=-127, max=127)
    e8m0 = (scale_unbiased.to(torch.int32) + 127).to(torch.uint8)
    quant_scale = torch.exp2(-scale_unbiased)

    qx = (x2 * quant_scale).reshape(m, k).to(torch.float8_e4m3fn)
    return qx, e8m0.reshape(m, ng)


def _quantize_fp8_block_e8m0(w):
    """Per-128x128 FP8 (E4M3 + E8M0) quantization of ``w`` (``[N, K]``).

    Returns FP8 codes (``[N, K]``) and the e8m0 block scales
    (``[N/128, K/128]``, uint8)."""
    n, k = w.shape
    sn, sk = n // _W_BLOCK_N, k // _W_BLOCK_K
    wb = w.float().view(sn, _W_BLOCK_N, sk, _W_BLOCK_K)
    amax = wb.abs().amax(dim=(1, 3))
    scale = (amax / _FP8_MAX).clamp_min(1e-12)
    unbiased = torch.ceil(torch.log2(scale)).clamp_(-127, 127)
    scale_dec = torch.exp2(unbiased).view(sn, 1, sk, 1)
    e8m0 = (unbiased.to(torch.int32) + 127).to(torch.uint8)
    q = (wb / scale_dec).reshape(n, k).to(torch.float8_e4m3fn)
    return q, e8m0


def quantize_afp8wfp8(a, w):
    """Quantize a high-precision activation/weight pair to the MXFP8/FP8
    operands the deployed GEMM consumes.

    Returns ``(x_fp8, x_scale, w_fp8, w_scale)`` with the layouts the AMD
    runtime ``gemm_afp8wfp8`` expects (``x_scale`` e8m0 ``[M, K/32]``,
    ``w_scale`` e8m0 ``[N/128, K/128]``)."""
    x_fp8, x_scale = _quantize_mxfp8_1x32(a)
    w_fp8, w_scale = _quantize_fp8_block_e8m0(w)
    return x_fp8, x_scale, w_fp8, w_scale


def _dequant_a(x_fp8, x_scale):
    scale = _e8m0_to_f32(x_scale).repeat_interleave(_A_GROUP, dim=1)
    return x_fp8.float() * scale


def _dequant_w(w_fp8, w_scale):
    scale = _e8m0_to_f32(w_scale)
    scale = scale.repeat_interleave(_W_BLOCK_N, dim=0).repeat_interleave(
        _W_BLOCK_K, dim=1
    )
    return w_fp8.float() * scale


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, w):
        x_fp8, x_scale, w_fp8, w_scale = quantize_afp8wfp8(a, w)
        x_f32 = _dequant_a(x_fp8, x_scale)
        w_f32 = _dequant_w(w_fp8, w_scale)
        out = torch.matmul(x_f32, w_f32.transpose(0, 1))
        return out.to(torch.bfloat16)


def get_inputs():
    # Representative shape (M, N, K) = (128, 1536, 4096); the harness sweeps more
    # shapes (N and K multiples of 128 for the 128x128 weight-scale layout).
    m, n, k = 128, 1536, 4096
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    return [a, w]


def get_init_inputs():
    return []
