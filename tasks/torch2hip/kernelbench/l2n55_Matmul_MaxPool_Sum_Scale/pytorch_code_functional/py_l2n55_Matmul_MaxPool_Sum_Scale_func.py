# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, weight, bias, kernel_size, scale_factor):
    x = F.linear(x, weight, bias)
    x = F.max_pool1d(x.unsqueeze(1), kernel_size).squeeze(1)
    x = torch.sum(x, dim=1)
    x = x * scale_factor
    return x


class Matmul_MaxPool_Sum_Scale(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias, self.kernel_size, self.scale_factor)


batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
