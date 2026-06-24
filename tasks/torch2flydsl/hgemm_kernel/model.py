# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference for half-precision GEMM.

Computes ``out = a @ b.T`` with fp32 accumulation, where ``a`` is ``[M, K]`` and
``b`` is ``[N, K]``. The result is cast back to the input dtype.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # fp32 accumulation, result cast back to the input dtype.
        return torch.matmul(a.float(), b.float().transpose(-1, -2)).to(a.dtype)


def get_inputs():
    # Representative shape (M, N, K) = (256, 256, 5120); the harness sweeps more.
    m, n, k = 256, 256, 5120
    a = torch.rand(m, k, dtype=torch.bfloat16)
    b = torch.rand(n, k, dtype=torch.bfloat16)
    return [a, b]


def get_init_inputs():
    return []
