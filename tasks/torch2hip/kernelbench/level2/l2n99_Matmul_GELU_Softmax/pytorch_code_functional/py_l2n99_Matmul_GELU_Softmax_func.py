# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, weight, bias):
    x = F.linear(x, weight, bias)
    x = F.gelu(x)
    x = F.softmax(x, dim=1)
    return x


class Matmul_GELU_Softmax(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, fn=module_fn):
        return fn(x, self.linear.weight, self.linear.bias)


batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    # in_features fixed, escalate batch.
    for b in [256, 512, 1024, 2048, 4096]:
        yield [torch.rand(b, in_features)]


def get_init_inputs():
    return [in_features, out_features]
