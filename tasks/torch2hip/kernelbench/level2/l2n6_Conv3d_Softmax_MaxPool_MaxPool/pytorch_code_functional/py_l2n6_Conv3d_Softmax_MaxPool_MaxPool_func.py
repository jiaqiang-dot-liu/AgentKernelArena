# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, conv_weight, conv_bias, pool_kernel_size):
    x = F.conv3d(x, conv_weight, conv_bias)
    x = torch.softmax(x, dim=1)
    x = F.max_pool3d(x, pool_kernel_size)
    x = F.max_pool3d(x, pool_kernel_size)
    return x


class Conv3d_Softmax_MaxPool_MaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.pool_kernel_size)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    # Conv3d pipeline; in_channels fixed, escalate batch (depth/spatial held viable).
    for b, d, h, w in [(16, 16, 32, 32), (32, 16, 32, 32), (64, 16, 32, 32),
                       (128, 16, 32, 32), (64, 16, 16, 16)]:
        yield [torch.rand(b, in_channels, d, h, w)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
