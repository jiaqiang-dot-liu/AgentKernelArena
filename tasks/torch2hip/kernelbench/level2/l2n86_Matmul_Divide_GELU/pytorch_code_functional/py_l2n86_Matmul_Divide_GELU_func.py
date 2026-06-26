# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, weight, bias, divisor):
    x = F.linear(x, weight, bias)
    x = x / divisor
    x = F.gelu(x)
    return x


class Matmul_Divide_GELU(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x, fn=module_fn):
        return fn(x, self.linear.weight, self.linear.bias, self.divisor)


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    # input_size fixed, vary batch.
    for b in [256, 512, 1024, 2048]:
        yield [torch.rand(b, input_size)]


def get_init_inputs():
    return [input_size, output_size, divisor]
