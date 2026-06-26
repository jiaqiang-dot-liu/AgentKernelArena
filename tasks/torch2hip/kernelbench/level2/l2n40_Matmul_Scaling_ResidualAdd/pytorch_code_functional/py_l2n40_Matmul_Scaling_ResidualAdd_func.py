# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, weight, bias, scaling_factor):
    x = F.linear(x, weight, bias)
    original_x = x.clone().detach()
    x = x * scaling_factor
    x = x + original_x
    return x


class Matmul_Scaling_ResidualAdd(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias, self.scaling_factor)


batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    # in_features fixed, escalate batch.
    for b in [2048, 4096, 8192, 16384, 32768]:
        yield [torch.rand(b, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor]
