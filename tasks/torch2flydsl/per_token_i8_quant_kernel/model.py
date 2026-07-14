# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for dynamic per-token INT8 quantization.

Dynamically quantizes each row of a ``[m, n]`` bf16 activation to INT8 with a
per-row (per-token) amax scale, matching the AMD runtime op
``aiter.dynamic_per_token_scaled_quant`` (the HIP per-token path of
``aiter.pertoken_quant`` with no smoothing scale):

  scale[i]  = amax(|x[i, :]|) / 127        (fp32; all-zero row -> 1)
  out[i, :] = int8(trunc_toward_zero(x[i, :] / scale[i]))

AMD-runtime semantics (option-b): the reduction and division are done in fp32
and the result is converted to int8 by truncation toward zero (saturated to
``[-128, 127]``), matching the device kernel's ``static_cast<int8_t>`` store.
The returned scale is fp32 of shape ``[m, 1]``.

forward(input) -> (output_int8, scale_fp32)
  input  : [m, n]   bf16
  output : [m, n]   int8
  scale  : [m, 1]   fp32
"""
import torch
import torch.nn as nn

_I8_MAX = 127.0


class Model(nn.Module):
    """Dynamic per-token INT8 quantizer. ``Model()`` takes no hyperparams."""

    def __init__(self):
        super().__init__()

    def forward(self, input):
        x = input.float()
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = amax / _I8_MAX
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(x / scale, -128.0, 127.0)
        y = q.to(torch.int8)
        return y, scale.to(torch.float32)


def get_inputs():
    m, n = 128, 8192
    torch.manual_seed(0)
    return [torch.randn(m, n, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
