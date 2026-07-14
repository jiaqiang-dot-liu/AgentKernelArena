# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference for fused residual-add + 2D RMSNorm (bf16, fp32 reduction).

Adds a residual to the input, then RMS-normalizes the sum with a per-channel
weight, matching the AMD runtime CK ``rmsnorm2d_fwd_with_add`` op:

  residual_out = input + residual
  out          = residual_out * rsqrt(mean(residual_out^2) + eps) * weight

The residual add and the mean-of-squares reduction / normalization are done in
fp32 and truncated back to bf16. ``residual_out`` (the pre-norm sum) is returned
because the runtime op writes it back for the next layer. Mirrors the op_test
``run_torch(..., residual=...)``.

forward(input, weight, residual) -> (output, residual_out)
  input        : [m, n]   bf16
  weight       : [n]      bf16   (per-channel scale, gamma)
  residual     : [m, n]   bf16
  output       : [m, n]   bf16   (normalized)
  residual_out : [m, n]   bf16   (input + residual, pre-norm)
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Fused add + 2D RMSNorm. ``Model(eps)``; weight supplied at call time."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, weight, residual):
        residual_out = input + residual
        rf = residual_out.float()
        rstd = torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = rf * rstd * weight.float()
        return out.to(input.dtype), residual_out.to(input.dtype)


def get_inputs():
    # Representative transformer hidden shape: m=128 tokens, n=4096 hidden.
    # CPU tensors (KernelBench convention); the consumer/harness relocates them.
    m, n = 128, 4096
    torch.manual_seed(0)
    input = torch.randn(m, n, dtype=torch.bfloat16)
    weight = torch.randn(n, dtype=torch.bfloat16)
    residual = torch.randn(m, n, dtype=torch.bfloat16)
    return [input, weight, residual]


def get_init_inputs():
    # Flat positional args for Model(eps).
    return [1e-5]
