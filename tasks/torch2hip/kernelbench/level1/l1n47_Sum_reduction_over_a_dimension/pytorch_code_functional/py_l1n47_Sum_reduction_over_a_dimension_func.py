# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sum(x, dim=dim, keepdim=True)


class Sum_reduction_over_a_dimension(nn.Module):
    def __init__(self, dim: int):
        super(Sum_reduction_over_a_dimension, self).__init__()
        self.dim = dim

    def forward(self, x, fn=module_fn):
        return fn(x, self.dim)


batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_inputs():
    # Sum reduction over reduce_dim (fixed by get_init_inputs); escalate shape.
    for b, d1, d2 in [(32, 2048, 2047), (64, 2048, 2047), (128, 2048, 2047),
                      (64, 4096, 4095), (128, 1024, 1023)]:
        yield [torch.rand(b, d1, d2)]


def get_init_inputs():
    return [reduce_dim]
