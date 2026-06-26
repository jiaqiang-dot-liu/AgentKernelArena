# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, weight, bias, pool_kernel_size, scale_factor):
    x = F.linear(x, weight, bias)
    x = F.avg_pool1d(x.unsqueeze(1), pool_kernel_size).squeeze(1)
    x = F.gelu(x)
    x = x * scale_factor
    x = torch.max(x, dim=1).values
    return x


class Matmul_AvgPool_GELU_Scale_Max(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias, self.pool_kernel_size,
                  self.scale_factor)


batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    # in_features fixed, escalate batch.
    for b in [256, 512, 1024, 2048, 4096]:
        yield [torch.rand(b, in_features)]


def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
