# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


class Standard_matrix_multiplication_(nn.Module):
    def __init__(self):
        super(Standard_matrix_multiplication_, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    # Standard GEMM (M,K) @ (K,N); escalate all three dims.
    for M, K, N in [(512, 1024, 768), (1024, 2048, 1024), (2048, 2048, 2048),
                    (2048, 4096, 2048), (4096, 4096, 4096)]:
        yield [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []  # No special initialization inputs needed
