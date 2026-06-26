# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.bmm(A, B)


class Batched_matrix_multiplication(nn.Module):
    def __init__(self):
        super(Batched_matrix_multiplication, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    # Batched GEMM (b,m,k) @ (b,k,n); escalate batch and inner dims.
    for b, m, k, n in [(16, 128, 256, 256), (32, 256, 256, 512), (64, 256, 512, 512),
                       (128, 256, 512, 1024), (128, 512, 512, 512)]:
        yield [torch.rand(b, m, k), torch.rand(b, k, n)]


def get_init_inputs():
    return []  # No special initialization inputs needed
