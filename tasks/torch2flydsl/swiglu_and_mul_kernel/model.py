# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for the fused ``swiglu_and_mul`` gated activation.

The op consumes a row-major ``[m, 2 * d]`` tensor whose last dimension is the
concatenation of a gate half ``x`` and a linear half ``y``, and produces
``[m, d]`` using the GPT-OSS clamped SwiGLU:

    gate   = min(x, LIMIT)
    linear = clamp(y, -LIMIT, LIMIT)
    out    = gate * sigmoid(ALPHA * gate) * (linear + 1)

with ``ALPHA = 1.702`` and ``LIMIT = 7.0`` hard-coded in AMD's device kernel
(the ``swiglu_and_mul`` runtime op takes no ``limit`` argument). The numerics mirror the
runtime path: inputs and outputs are bf16, while the clamp, gating, and multiply
are evaluated in fp32 and truncated to bf16 on store.
"""
import torch
import torch.nn as nn

ALPHA = 1.702
LIMIT = 7.0


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        d = input.shape[-1] // 2
        x, y = input.split([d, d], dim=-1)
        gate = torch.clamp(x.float(), max=LIMIT)
        linear = torch.clamp(y.float(), min=-LIMIT, max=LIMIT)
        out = gate * torch.sigmoid(ALPHA * gate) * (linear + 1.0)
        return out.to(torch.bfloat16)


def get_inputs():
    return [torch.randn(512, 8192, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
