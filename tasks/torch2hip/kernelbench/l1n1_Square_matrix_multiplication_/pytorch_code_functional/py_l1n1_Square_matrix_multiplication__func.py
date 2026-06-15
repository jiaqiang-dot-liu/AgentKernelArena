# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


class Square_matrix_multiplication_(nn.Module):
    def __init__(self):
        super(Square_matrix_multiplication_, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
