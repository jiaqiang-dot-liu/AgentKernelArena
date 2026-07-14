# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for the ``gelu_fast`` elementwise activation.

The op applies the tanh approximation of GELU elementwise to a ``[m, n]``
activation tensor (no gating / no split), producing ``[m, n]``:

  gelu_fast(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

This is PyTorch's ``F.gelu(..., approximate="tanh")`` and matches the device
functor used by AMD's ``gelu_fast`` runtime op (the op_test validates the runtime
kernel against exactly this tanh-approximation reference).

The numerics mirror AMD's runtime path: inputs and outputs are bf16, while the
activation is evaluated in fp32 and truncated to bf16 on store.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        out = F.gelu(input.float(), approximate="tanh")
        return out.to(input.dtype)


def get_inputs():
    return [torch.randn(512, 8192, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
