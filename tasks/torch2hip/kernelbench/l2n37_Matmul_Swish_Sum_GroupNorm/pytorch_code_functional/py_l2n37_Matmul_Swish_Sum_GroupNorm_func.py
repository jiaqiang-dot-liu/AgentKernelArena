# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, weight, bias, extra_bias, gn_weight, gn_bias, num_groups):
    x = F.linear(x, weight, bias)
    x = torch.sigmoid(x) * x
    x = x + extra_bias
    x = F.group_norm(x, num_groups, gn_weight, gn_bias)
    return x


class Matmul_Swish_Sum_GroupNorm(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias, self.bias,
                  self.group_norm.weight, self.group_norm.bias, self.num_groups)


batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
