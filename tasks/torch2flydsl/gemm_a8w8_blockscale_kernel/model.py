# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the FP8 block-scaled GEMM ``gemm_a8w8_blockscale``.

Computes ``out = a @ w.T`` where the activation ``a`` (``[M, K]``) and weight
``w`` (``[N, K]``) are quantized to FP8 (``float8_e4m3fn``) with block scales,
matching the AMD runtime FP8 block-scaled GEMM:

- activation: per-token, per-1x128 (along K) FP8 scale (``x_scale`` is
  ``[M, K/128]``);
- weight: per-128x128 block FP8 scale (``w_scale`` is ``[ceil(N/128),
  ceil(K/128)]``);
- the GEMM dequantizes each operand (FP8 value times its block scale),
  accumulates in fp32, and truncates the result to bf16.

The quantization here defines the exact FP8 operands the deployed kernel sees:
``quantize_a8w8_blockscale`` is the single source of the FP8 tensors and scales
(reused by the harness to drive the real AMD runtime op), so the reference and
the hardware kernel operate on byte-identical inputs and differ only by fp32
accumulation order.
"""
import torch
import torch.nn as nn

def _amd_fp8_dtype():
    """fp8 storage dtype the matching aiter op uses on the active GPU arch,
    mirroring ``aiter/utility/dtypes.py``: gfx942/CDNA3 -> ``float8_e4m3fnuz``
    (finite max 240); gfx950/CDNA4 and others -> ``float8_e4m3fn`` (max 448)."""
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        arch = ""
    return torch.float8_e4m3fnuz if arch == "gfx942" else torch.float8_e4m3fn


# Arch-selected FP8 type and its saturation magnitude (per-block scale divisor).
_FP8_DTYPE = _amd_fp8_dtype()
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
_BLOCK_N = 128
_BLOCK_K = 128


def _quantize_rowwise_1x128(a):
    """Per-token, per-1x128 (K) FP8 quantization of ``a`` (``[M, K]``).

    Returns the FP8 codes (``float8_e4m3fn``) and the fp32 scales
    (``[M, K/128]``)."""
    m, k = a.shape
    scale_k = (k + _BLOCK_K - 1) // _BLOCK_K
    af = a.float().view(m, scale_k, _BLOCK_K)
    amax = af.abs().amax(dim=-1)
    scale = (amax / _FP8_MAX).clamp_min(1e-12)
    q = (af / scale.unsqueeze(-1)).view(m, k).to(_FP8_DTYPE)
    return q, scale


def _quantize_blockwise_128x128(w):
    """Per-128x128 block FP8 quantization of ``w`` (``[N, K]``).

    Returns the FP8 codes (``float8_e4m3fn``) and the fp32 scales
    (``[ceil(N/128), ceil(K/128)]``)."""
    n, k = w.shape
    scale_n = (n + _BLOCK_N - 1) // _BLOCK_N
    scale_k = (k + _BLOCK_K - 1) // _BLOCK_K
    pad_n = scale_n * _BLOCK_N - n
    pad_k = scale_k * _BLOCK_K - k
    wf = w.float()
    if pad_n or pad_k:
        wf = torch.nn.functional.pad(wf, (0, pad_k, 0, pad_n))
    blocks = wf.view(scale_n, _BLOCK_N, scale_k, _BLOCK_K)
    amax = blocks.abs().amax(dim=(1, 3))
    scale = (amax / _FP8_MAX).clamp_min(1e-12)
    scale_full = scale.repeat_interleave(_BLOCK_N, 0).repeat_interleave(_BLOCK_K, 1)
    q = (wf / scale_full).to(_FP8_DTYPE)[:n, :k].contiguous()
    return q, scale


def quantize_a8w8_blockscale(a, w):
    """Quantize a high-precision activation/weight pair to the FP8 block-scaled
    operands the deployed GEMM consumes.

    Returns ``(x_fp8, x_scale, w_fp8, w_scale)`` with layouts matching the AMD
    runtime FP8 block-scaled GEMM."""
    x_fp8, x_scale = _quantize_rowwise_1x128(a)
    w_fp8, w_scale = _quantize_blockwise_128x128(w)
    return x_fp8, x_scale, w_fp8, w_scale


def _dequant_blockscale(x_fp8, x_scale, w_fp8, w_scale):
    """Dequantize the FP8 block-scaled operands and run the fp32 GEMM."""
    m, k = x_fp8.shape
    n = w_fp8.shape[0]
    scale_k = x_scale.shape[1]

    x = x_fp8.float().view(m, scale_k, _BLOCK_K) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    sn, sk = w_scale.shape
    w_scale_full = (
        w_scale.repeat_interleave(_BLOCK_N, 0).repeat_interleave(_BLOCK_K, 1)
    )[:n, :k]
    w = w_fp8.float() * w_scale_full

    return torch.matmul(x, w.transpose(0, 1))


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, w):
        x_fp8, x_scale, w_fp8, w_scale = quantize_a8w8_blockscale(a, w)
        out = _dequant_blockscale(x_fp8, x_scale, w_fp8, w_scale)
        return out.to(torch.bfloat16)


def get_inputs():
    # Representative DeepSeek-V3 shape (M, N, K) = (128, 512, 7168); the harness
    # sweeps more real shapes from a8w8_blockscale_tuned_gemm_ds_v3.csv.
    m, n, k = 128, 512, 7168
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    return [a, w]


def get_init_inputs():
    return []
