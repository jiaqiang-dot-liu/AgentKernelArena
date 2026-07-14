# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference for 2D RMSNorm (bf16 in/out, fp32 reduction).

Row-wise RMS normalization over the last dimension followed by a per-channel
weight, matching the AMD runtime CK ``rmsnorm2d_fwd`` op:

  out[i, :] = x[i, :] * rsqrt(mean(x[i, :]^2) + eps) * weight

Activations are bf16; the mean-of-squares reduction and the normalization are
done in fp32 and the result is truncated back to bf16 on store. This mirrors
the op_test ``run_torch`` (which calls ``F.rms_norm`` with an internal fp32
reduction).

forward(input, weight) -> output
  input  : [m, n]   bf16   (row-major activations)
  weight : [n]      bf16   (per-channel scale, gamma)
  output : [m, n]   bf16
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """2D RMSNorm. ``Model(eps)``; weight is supplied at call time."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, weight):
        xf = input.float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = xf * rstd * weight.float()
        return out.to(input.dtype)


def get_inputs():
    # Representative transformer hidden shape: m=128 tokens, n=4096 hidden.
    # CPU tensors (KernelBench convention); the consumer/harness relocates them.
    m, n = 128, 4096
    torch.manual_seed(0)
    input = torch.randn(m, n, dtype=torch.bfloat16)
    weight = torch.randn(n, dtype=torch.bfloat16)
    return [input, weight]


def get_init_inputs():
    # Flat positional args for Model(eps).
    return [1e-5]
