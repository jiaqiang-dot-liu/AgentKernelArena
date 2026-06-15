# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


class Matmul_with_irregular_shapes_(nn.Module):
    def __init__(self):
        super(Matmul_with_irregular_shapes_, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
