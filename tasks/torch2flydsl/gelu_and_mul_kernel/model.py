# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for the fused ``gelu_and_mul`` gated activation.

The op consumes a row-major ``[m, 2 * d]`` tensor whose last dimension is the
concatenation of a gate half ``x`` and an up-projection half ``y``, and produces
``[m, d]`` with ``out = gelu(x) * y``. The activation is the exact (erf-based)
GELU, i.e. ``x * 0.5 * (1 + erf(x / sqrt(2)))`` — PyTorch's ``approximate="none"``
form — matching the device functor used by AMD's ``gelu_and_mul`` runtime op (the
tanh approximation belongs to the separate ``gelu_tanh_and_mul`` op).

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
        out = F.gelu(x.float(), approximate="none") * y.float()
        return out.to(torch.bfloat16)


def get_inputs():
    return [torch.randn(512, 8192, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
