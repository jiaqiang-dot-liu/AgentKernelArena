# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for fused 2D RMSNorm + SmoothQuant per-token INT8.

The op normalizes each row of a ``[m, n]`` bf16 activation with RMSNorm and a
per-channel weight, applies a per-channel smoothing scale ``x_scale`` ``[n]``,
then dynamically quantizes each row to INT8 with a per-row (per-token) amax
scale, matching the AMD runtime CK op ``rmsnorm2d_fwd_with_smoothquant``:

  normed[i, :] = x[i, :] * rsqrt(mean(x[i, :]^2) + eps) * weight   (fp32)
  hidden[i, :] = normed[i, :] * x_scale                            (fp32)
  scale[i]     = amax(|hidden[i, :]|) / 127   (fp32; all-zero row -> 1)
  out[i, :]    = int8(trunc_toward_zero(hidden[i, :] / scale[i]))

AMD-runtime semantics (option-b): the fused CK kernel keeps the RMSNorm result
in fp32 registers (fp32 mean-of-squares reduction and weight multiply, NO
intermediate bf16 truncation) and quantizes directly from fp32. The per-token
INT8 quant is the standard ``pertoken_quant`` smooth path: hidden divided by the
per-row scale and converted to int8 by truncation toward zero (saturated to
``[-128, 127]``), matching the device kernel's ``static_cast<int8_t>`` store.
The returned scale is fp32 of shape ``[m, 1]``.

forward(input, x_scale, weight) -> (output_int8, scale_fp32)
  input   : [m, n]   bf16
  x_scale : [n]      fp32   (per-channel smoothing scale)
  weight  : [n]      bf16   (per-channel RMSNorm scale, gamma)
  output  : [m, n]   int8
  scale   : [m, 1]   fp32
"""
import torch
import torch.nn as nn

_I8_MAX = 127.0


class Model(nn.Module):
    """RMSNorm + SmoothQuant per-token INT8. ``Model(eps)``; weight/x_scale at call time."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, x_scale, weight):
        xf = input.float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = xf * rstd * weight.float()
        hidden = normed * x_scale.float()
        amax = hidden.abs().amax(dim=-1, keepdim=True)
        scale = amax / _I8_MAX
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(hidden / scale, -128.0, 127.0)
        y = q.to(torch.int8)
        return y, scale.to(torch.float32)


def get_inputs():
    m, n = 128, 5120
    torch.manual_seed(0)
    input = torch.randn(m, n, dtype=torch.bfloat16)
    x_scale = torch.randn(n, dtype=torch.float32)
    weight = torch.randn(n, dtype=torch.bfloat16)
    return [input, x_scale, weight]


def get_init_inputs():
    return [1e-5]
