# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


class Tall_skinny_matrix_multiplication_(nn.Module):
    def __init__(self):
        super(Tall_skinny_matrix_multiplication_, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


M = 16384 * 2
N = 16 * 2

def get_inputs():
    # Tall-skinny GEMM (M,N) @ (N,M) with M >> N; escalate M and N.
    for M, N in [(8192, 16), (16384, 16), (32768, 32), (65536, 16), (32768, 64)]:
        yield [torch.rand(M, N), torch.rand(N, M)]


def get_init_inputs():
    return []  # No special initialization inputs needed
