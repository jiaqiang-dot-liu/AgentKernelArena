# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for the fused ``gelu_tanh_and_mul`` gated activation.

The op consumes a row-major ``[m, 2 * d]`` tensor whose last dimension is the
concatenation of a gate half ``x`` and an up-projection half ``y``, and produces
``[m, d]`` with ``out = gelu_tanh(x) * y``. The activation is the tanh
approximation of GELU,

  gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

i.e. PyTorch's ``F.gelu(..., approximate="tanh")`` — matching the device functor
used by AMD's ``gelu_tanh_and_mul`` runtime op. The exact erf-based form belongs
to the separate ``gelu_and_mul`` op.

The numerics mirror AMD's runtime path: inputs and outputs are bf16, while the
GELU and the gate multiply are evaluated in fp32 and truncated to bf16 on store.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        d = input.shape[-1] // 2
        x, y = input.split([d, d], dim=-1)
        out = F.gelu(x.float(), approximate="tanh") * y.float()
        return out.to(torch.bfloat16)


def get_inputs():
    return [torch.randn(512, 8192, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
