# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, weight, bias, scaling_factor):
    x = F.linear(x, weight, bias)
    x = x * torch.sigmoid(x)
    x = x * scaling_factor
    return x


class Matmul_Swish_Scaling(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias, self.scaling_factor)


batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    # in_features fixed, escalate batch.
    for b in [32, 64, 128, 256, 512]:
        yield [torch.rand(b, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor]
