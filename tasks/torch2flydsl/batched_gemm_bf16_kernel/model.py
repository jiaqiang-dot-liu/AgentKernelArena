# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the batched bf16 GEMM ``batched_gemm_bf16``.

Computes a per-batch ``out[b] = x[b] @ w[b].T`` where ``x`` is ``[B, M, K]`` and
``w`` is ``[B, N, K]``, matching the AMD runtime batched bf16 GEMM: each batch is
a bf16 GEMM with fp32 accumulation and a bf16 output (no quantization).
"""
import torch
import torch.nn as nn


def _batched_matmul(x, w):
    """Per-batch ``x[b] @ w[b].T`` accumulated in fp32."""
    return torch.bmm(x.float(), w.float().transpose(1, 2))


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        out = _batched_matmul(x, w)
        return out.to(torch.bfloat16)


def get_inputs():
    # Representative batched bf16 shape (B, M, N, K) = (16, 128, 1280, 8192); the
    # harness sweeps more real shapes from configs/bf16_untuned_batched_gemm.csv.
    b, m, n, k = 16, 128, 1280, 8192
    x = torch.randn(b, m, k, dtype=torch.bfloat16)
    w = torch.randn(b, n, k, dtype=torch.bfloat16)
    return [x, w]


def get_init_inputs():
    return []
