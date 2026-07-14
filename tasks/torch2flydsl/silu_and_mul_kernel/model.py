# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for the fused ``silu_and_mul`` gated activation.

The op consumes a row-major ``[m, 2 * d]`` tensor whose last dimension is the
concatenation of a gate half ``x`` and an up-projection half ``y``, and produces
``[m, d]`` with ``out = silu(x) * y``. When ``limit > 0`` it applies the GPT-OSS
clamp (``x`` upper-clamped to ``limit``, ``y`` clamped to ``[-limit, limit]``).

The numerics mirror AMD's ``silu_and_mul`` runtime op: inputs and
outputs are bf16, while the SiLU and the gate multiply are evaluated in fp32 and
truncated to bf16 on store. In the clamped path the upper-clamped gate is
re-truncated to bf16 before the activation, matching the device kernel's
``cast<float> -> fmin -> cast<bf16> -> silu`` sequence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, limit=0.0):
        super().__init__()
        self.limit = float(limit)

    def forward(self, input):
        d = input.shape[-1] // 2
        x, y = input.split([d, d], dim=-1)
        gate = x.float()
        up = y.float()
        if self.limit > 0.0:
            gate = torch.clamp(gate, max=self.limit).to(torch.bfloat16).float()
            up = torch.clamp(up, min=-self.limit, max=self.limit)
        out = F.silu(gate) * up
        return out.to(torch.bfloat16)


def get_inputs():
    return [torch.randn(512, 8192, dtype=torch.bfloat16)]


def get_init_inputs():
    return [0.0]
