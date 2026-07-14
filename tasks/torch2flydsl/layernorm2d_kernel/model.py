# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference for 2D LayerNorm (bf16 in/out, fp32 reduction).

Row-wise layer normalization over the last dimension with an affine weight and
bias, matching the AMD runtime CK ``layernorm2d_fwd`` op:

  mean = mean(x[i, :])
  var  = mean((x[i, :] - mean)^2)
  out  = (x[i, :] - mean) * rsqrt(var + eps) * weight + bias

Activations are bf16; the mean/variance reduction and the normalization are done
in fp32 and the result is truncated back to bf16 on store. Mirrors the op_test
``run_torch`` (which calls ``F.layer_norm`` with an internal fp32 reduction).

forward(input, weight, bias) -> output
  input  : [m, n]   bf16
  weight : [n]      bf16   (affine scale, gamma)
  bias   : [n]      bf16   (affine shift, beta)
  output : [m, n]   bf16
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """2D LayerNorm. ``Model(eps)``; weight/bias supplied at call time."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, weight, bias):
        xf = input.float()
        mean = xf.mean(-1, keepdim=True)
        var = (xf - mean).pow(2).mean(-1, keepdim=True)
        norm = (xf - mean) * torch.rsqrt(var + self.eps)
        out = norm * weight.float() + bias.float()
        return out.to(input.dtype)


def get_inputs():
    # Representative transformer hidden shape: m=128 tokens, n=8192 hidden.
    # CPU tensors (KernelBench convention); the consumer/harness relocates them.
    m, n = 128, 8192
    torch.manual_seed(0)
    input = torch.randn(m, n, dtype=torch.bfloat16)
    weight = torch.randn(n, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.bfloat16)
    return [input, weight, bias]


def get_init_inputs():
    # Flat positional args for Model(eps).
    return [1e-5]
