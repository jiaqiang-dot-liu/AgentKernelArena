# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for fused 2D RMSNorm + dynamic per-token FP8 quant.

The op normalizes each row of a ``[m, n]`` bf16 activation with RMSNorm and a
per-channel weight, then dynamically quantizes the normalized result to FP8
(E4M3) with a per-row (per-token) amax scale, matching the AMD runtime CK op
``rmsnorm2d_fwd_with_dynamicquant`` (``aiter.rmsnorm_quant`` with
``group_size=0``):

  normed[i, :] = x[i, :] * rsqrt(mean(x[i, :]^2) + eps) * weight   (fp32)
  scale[i]     = amax(|normed[i, :]|) / dtype_max    (fp32; all-zero row -> 1)
  out[i, :]    = round_to_fp8(normed[i, :] / scale[i])

AMD-runtime semantics (option-b): the fused CK kernel keeps the RMSNorm result
in fp32 registers (fp32 mean-of-squares reduction and weight multiply, NO
intermediate bf16 truncation) and quantizes directly from fp32, then the
per-token FP8 quant is the standard ``pertoken_quant`` path with an
arch-selected FP8 type (``aiter/utility/dtypes.py``): gfx942/CDNA3 uses
``float8_e4m3fnuz`` (finite max 240), gfx950/CDNA4 uses ``float8_e4m3fn``
(finite max 448), round-to-nearest-even. The returned scale is fp32 of shape
``[m, 1]``.

forward(input, weight) -> (output_fp8, scale_fp32)
  input  : [m, n]   bf16
  weight : [n]      bf16   (per-channel scale, gamma)
  output : [m, n]   fp8 e4m3
  scale  : [m, 1]   fp32
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


_FP8_DTYPE = _amd_fp8_dtype()


class Model(nn.Module):
    """RMSNorm + dynamic per-token FP8 quant. ``Model(eps)``; weight at call time."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.dtype_max = float(torch.finfo(_FP8_DTYPE).max)

    def forward(self, input, weight):
        xf = input.float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = xf * rstd * weight.float()
        amax = normed.abs().amax(dim=-1, keepdim=True)
        scale = amax / self.dtype_max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        out = (normed / scale).to(_FP8_DTYPE)
        return out, scale.to(torch.float32)


def get_inputs():
    m, n = 128, 4096
    torch.manual_seed(0)
    input = torch.randn(m, n, dtype=torch.bfloat16)
    weight = torch.randn(n, dtype=torch.bfloat16)
    return [input, weight]


def get_init_inputs():
    return [1e-5]
