# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the BF16/FP8-blockscale GEMM ``gemm_a16w8_blockscale``.

Computes ``out = a @ w.T`` where the activation ``a`` (``[M, K]``) is kept in
bf16 (no quantization) and the weight ``w`` (``[N, K]``) is quantized to FP8
(``float8_e4m3fn``) with one fp32 scale per 128x128 block, matching the AMD
runtime ``gemm_a16w8_blockscale`` (``prequant=False`` path):

- activation: bf16, consumed directly (``x`` is ``[M, K]``);
- weight: FP8 codes (``[N, K]``) plus an fp32 scale per 128x128 block
  (``w_scale`` is ``[ceil(N/128), ceil(K/128)]``);
- the GEMM upcasts the bf16 activation to fp32, dequantizes the weight (FP8
  value times its block scale), accumulates in fp32, and truncates to bf16.

``quantize_a16w8_blockscale`` is the single source of the kernel operands (the
harness reuses it to drive the real AMD runtime op), so the reference and the
hardware kernel operate on byte-identical inputs and differ only by fp32
accumulation order.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# e4m3fn saturation magnitude used as the block-scale divisor on gfx950.
_FP8_MAX = 448.0
# FP8 weight block dims (hardware-fixed 128x128 fp32 scaling).
_BLOCK_N = 128
_BLOCK_K = 128


def _quantize_fp8_block128(w):
    """Per-128x128 FP8 quantization of ``w`` (``[N, K]``) with fp32 block scales.

    Returns FP8 codes (``[N, K]``) and the fp32 block scales
    (``[ceil(N/128), ceil(K/128)]``)."""
    n, k = w.shape
    sn = (n + _BLOCK_N - 1) // _BLOCK_N
    sk = (k + _BLOCK_K - 1) // _BLOCK_K
    wf = w.float()
    wp = F.pad(wf, (0, sk * _BLOCK_K - k, 0, sn * _BLOCK_N - n))
    wb = wp.view(sn, _BLOCK_N, sk, _BLOCK_K)
    amax = wb.abs().amax(dim=(1, 3))
    scale = (amax / _FP8_MAX).clamp_min(1e-12)
    scale_full = (
        scale.repeat_interleave(_BLOCK_N, dim=0)
        .repeat_interleave(_BLOCK_K, dim=1)[:n, :k]
    )
    q = (wf / scale_full).to(torch.float8_e4m3fn)
    return q, scale


def quantize_a16w8_blockscale(a, w):
    """Quantize a high-precision activation/weight pair to the BF16/FP8-blockscale
    operands the deployed GEMM consumes.

    Returns ``(x_bf16, w_fp8, w_scale)`` with the layouts the AMD runtime
    ``gemm_a16w8_blockscale`` expects (``x_bf16`` is ``[M, K]`` bf16, ``w_fp8``
    is ``[N, K]`` FP8, ``w_scale`` is ``[ceil(N/128), ceil(K/128)]`` fp32)."""
    x_bf16 = a.to(torch.bfloat16)
    w_fp8, w_scale = _quantize_fp8_block128(w)
    return x_bf16, w_fp8, w_scale


def _dequant_w(w_fp8, w_scale):
    n, k = w_fp8.shape
    scale_full = (
        w_scale.repeat_interleave(_BLOCK_N, dim=0)
        .repeat_interleave(_BLOCK_K, dim=1)[:n, :k]
    )
    return w_fp8.float() * scale_full


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, w):
        x_bf16, w_fp8, w_scale = quantize_a16w8_blockscale(a, w)
        x_f32 = x_bf16.float()
        w_f32 = _dequant_w(w_fp8, w_scale)
        out = F.linear(x_f32, w_f32)
        return out.to(torch.bfloat16)


def get_inputs():
    # Representative shape (M, N, K) = (256, 2048, 2048); the harness sweeps more
    # shapes (N and K multiples of 128 for the 128x128 weight-scale layout).
    m, n, k = 256, 2048, 2048
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    return [a, w]


def get_init_inputs():
    return []
