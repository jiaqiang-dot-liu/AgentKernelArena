# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the batched INT8 GEMM ``batched_gemm_a8w8``.

Computes a per-batch ``out[b] = (x[b] @ w[b].T) * (x_scale[b] @ w_scale[b])``
where the activation ``x`` (``[B, M, K]``) is quantized to INT8 with a per-token
scale and the weight ``w`` (``[B, N, K]``) is quantized to INT8 with a
per-output-channel scale, matching the AMD runtime batched a8w8 GEMM:

- activation scale ``x_scale`` is ``[B, M, 1]`` (one per row of each batch);
- weight scale ``w_scale`` is ``[B, 1, N]`` (one per output channel of each
  batch);
- the GEMM forms the INT8 product, accumulates in fp32, multiplies by the
  outer-product of the per-token and per-channel scales, and truncates to bf16.

``quantize_batched_a8w8`` is the single source of the INT8 operands and scales
(the harness reuses it to drive the real AMD runtime op), so the reference and
the hardware kernel operate on byte-identical inputs and differ only by the
accumulation precision (fp32 reference vs int32 hardware accumulate).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# INT8 saturation magnitude used as the per-row scale divisor.
_INT8_MAX = 127.0


def _quantize_rowwise_i8(t):
    """Per-row INT8 quantization of a ``[B, R, K]`` tensor.

    Returns the INT8 codes and the fp32 row scales (``[B, R, 1]``) such that
    ``t ~= codes.float() * scale``."""
    tf = t.float()
    amax = tf.abs().amax(dim=-1, keepdim=True)
    scale = (amax / _INT8_MAX).clamp_min(1e-12)
    q = (tf / scale).round().clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return q, scale


def quantize_batched_a8w8(x, w):
    """Quantize a high-precision batched activation/weight pair to the INT8
    operands the deployed batched GEMM consumes.

    Returns ``(x_i8, x_scale, w_i8, w_scale)`` where ``x_scale`` is ``[B, M, 1]``
    and ``w_scale`` is ``[B, 1, N]`` (the layouts the AMD runtime op expects)."""
    x_i8, x_scale = _quantize_rowwise_i8(x)
    w_i8, w_scale_rows = _quantize_rowwise_i8(w)
    w_scale = w_scale_rows.transpose(1, 2)
    return x_i8, x_scale, w_i8, w_scale


def _dequant_batched_matmul(x_i8, x_scale, w_i8, w_scale):
    """INT8 batched GEMM with fp32 accumulation and per-token/per-channel scale."""
    b, m, _ = x_i8.shape
    n = w_i8.shape[1]
    out = torch.empty(b, m, n, dtype=torch.float32, device=x_i8.device)
    for i in range(b):
        prod = F.linear(x_i8[i].float(), w_i8[i].float())
        out[i] = prod * torch.matmul(x_scale[i], w_scale[i])
    return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        x_i8, x_scale, w_i8, w_scale = quantize_batched_a8w8(x, w)
        out = _dequant_batched_matmul(x_i8, x_scale, w_i8, w_scale)
        return out.to(torch.bfloat16)


def get_inputs():
    # Representative batched a8w8 shape (B, M, N, K) = (16, 128, 1280, 8192); the
    # harness sweeps more real shapes from configs/a8w8_untuned_batched_gemm.csv.
    b, m, n, k = 16, 128, 1280, 8192
    x = torch.randn(b, m, k, dtype=torch.bfloat16)
    w = torch.randn(b, n, k, dtype=torch.bfloat16)
    return [x, w]


def get_init_inputs():
    return []
