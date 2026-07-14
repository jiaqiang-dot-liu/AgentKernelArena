# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for SmoothQuant dynamic per-token INT8 quantization.

SmoothQuant first applies a per-channel smoothing scale to the activations
(``x * x_scale``, ``x_scale`` of shape ``[n]``), then performs a dynamic
per-token (per-row) INT8 quantization of the smoothed tensor. The op returns
``(y, y_scale)`` where ``y`` is ``[m, n]`` int8 and ``y_scale`` is fp32 of
shape ``[m, 1]``.

AMD-runtime semantics (option-b): this mirrors the AMD runtime
``smoothquant_fwd`` / ``pertoken_quant`` (int8) op. The per-row scale is
``amax(|x * x_scale|) /
127`` (INT8 max), with an all-zero row keeping a scale of 1. The smoothed
values are divided by the per-row scale and converted to int8 by truncation
toward zero (saturated to ``[-128, 127]``), matching the device kernel's
``static_cast<int8_t>`` store.
"""
import torch
import torch.nn as nn

_I8_MAX = 127.0


class Model(nn.Module):
    """SmoothQuant per-token INT8 quantizer. ``Model()`` takes no hyperparams."""

    def __init__(self):
        super().__init__()

    def forward(self, input, x_scale):
        hidden = input.float() * x_scale.float()
        amax = hidden.abs().amax(dim=-1, keepdim=True)
        scale = amax / _I8_MAX
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        # Truncation toward zero (saturated), matching the device int8 store.
        q = torch.clamp(hidden / scale, -128.0, 127.0)
        y = q.to(torch.int8)
        return y, scale.to(torch.float32)


def get_inputs():
    m, n = 128, 5120
    return [
        torch.randn(m, n, dtype=torch.bfloat16),
        torch.randn(n, dtype=torch.float32),
    ]


def get_init_inputs():
    return []
