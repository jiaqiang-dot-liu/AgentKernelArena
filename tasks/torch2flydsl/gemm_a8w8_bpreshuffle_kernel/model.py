# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference for the a8w8 b-preshuffle quantized GEMM.

Emulates the AMD-runtime semantics of the a8w8 b-preshuffle GEMM:

* per-token fp8 (e4m3) quantization of the activation ``x`` ``[M, K]``;
* per-output-channel fp8 quantization of the weight ``w`` ``[N, K]``;
* an fp32-accumulated matmul of the dequantized operands;
* scaling by the per-token and per-channel scales;
* truncation of the result to bf16.

The deployed kernel additionally stores the weight in a ``(16, 16)``
pre-shuffled layout for MFMA-friendly loads. That shuffle is a pure permutation
of the K elements and leaves the numeric result unchanged, so this reference
computes the mathematically-equivalent dense matmul. The harness applies the
real weight shuffle before invoking the FlyDSL kernel, exercising the layout
handling apples-to-apple against this reference.
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


FP8_DTYPE = _amd_fp8_dtype()


def pertoken_quant(x, quant_dtype=FP8_DTYPE):
    """Symmetric per-row (per-token / per-channel) fp8 quantization.

    Returns ``(q, scale)`` where ``q`` holds the quantized values in
    ``quant_dtype`` and ``scale`` is the fp32 per-row scale ``[rows, 1]``.
    """
    dtype_max = torch.finfo(quant_dtype).max
    x = x.to(torch.float32)
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / dtype_max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (x / scale).to(quant_dtype)
    return q, scale.to(torch.float32)


class Model(nn.Module):
    def __init__(self, out_dtype=torch.bfloat16):
        super().__init__()
        self.out_dtype = out_dtype

    def forward(self, x, weight):
        # Quantize both operands to fp8 (per-token activation, per-channel
        # weight), then dequantize implicitly by accumulating the raw fp8
        # products in fp32 and applying the scales after the reduction.
        xq, x_scale = pertoken_quant(x)
        wq, w_scale = pertoken_quant(weight)
        acc = torch.matmul(xq.float(), wq.float().transpose(-1, -2))
        out = acc * x_scale * w_scale.transpose(0, 1)
        return out.to(self.out_dtype)


def get_inputs():
    # Representative real shape (M, N, K) = (512, 5120, 1280); the harness
    # sweeps additional model shapes internally.
    m, n, k = 512, 5120, 1280
    x = torch.randn(m, k, dtype=torch.bfloat16)
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    return [x, weight]


def get_init_inputs():
    return []
