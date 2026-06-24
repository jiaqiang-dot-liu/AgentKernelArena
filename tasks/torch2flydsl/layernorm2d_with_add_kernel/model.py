# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""PyTorch reference for fused residual-add + 2D LayerNorm (bf16 in/out).

The op adds a residual to the input, returns that residual sum, and applies
row-wise LayerNorm (over the last dimension) with an affine weight and bias,
matching the AMD runtime CK op ``layernorm2d_fwd_with_add``:

  residual_out = x + residual                              (bf16 store)
  mean = mean(residual_out[i, :])
  var  = mean((residual_out[i, :] - mean)^2)
  out  = (residual_out[i, :] - mean) * rsqrt(var + eps) * weight + bias

Activations are bf16; the residual add is stored in bf16 (and returned), while
the mean/variance reduction and the normalization are done in fp32 and the
result is truncated back to bf16 on store. Mirrors the op_test ``run_torch``
(``residual_out = input + residual``; ``F.layer_norm`` with internal fp32
reduction).

forward(input, residual, weight, bias) -> (output, residual_out)
  input        : [m, n]   bf16
  residual     : [m, n]   bf16
  weight       : [n]      bf16   (affine scale, gamma)
  bias         : [n]      bf16   (affine shift, beta)
  output       : [m, n]   bf16
  residual_out : [m, n]   bf16
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Fused add + 2D LayerNorm. ``Model(eps)``; weight/bias supplied at call time."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, residual, weight, bias):
        residual_out = input + residual
        xf = residual_out.float()
        mean = xf.mean(-1, keepdim=True)
        var = (xf - mean).pow(2).mean(-1, keepdim=True)
        norm = (xf - mean) * torch.rsqrt(var + self.eps)
        out = norm * weight.float() + bias.float()
        return out.to(input.dtype), residual_out


def get_inputs():
    # Representative transformer hidden shape: m=128 tokens, n=8192 hidden.
    # CPU tensors (KernelBench convention); the consumer/harness relocates them.
    m, n = 128, 8192
    torch.manual_seed(0)
    input = torch.randn(m, n, dtype=torch.bfloat16)
    residual = torch.randn(m, n, dtype=torch.bfloat16)
    weight = torch.randn(n, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.bfloat16)
    return [input, residual, weight, bias]


def get_init_inputs():
    # Flat positional args for Model(eps).
    return [1e-5]
