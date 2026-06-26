# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, conv_weight, conv_bias, scaling_factor, bias, pool_kernel_size):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = torch.tanh(x)
    x = x * scaling_factor
    x = x + bias
    x = F.max_pool2d(x, pool_kernel_size)
    return x


class Conv2d_Tanh_Scaling_BiasAdd_Max(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor,
                 bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.scaling_factor,
                  self.bias, self.pool_kernel_size)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

def get_inputs():
    # in_channels fixed; bias_shape independent of spatial, vary batch/spatial.
    for b, h, w in [(32, 128, 128), (64, 256, 256), (128, 64, 64), (16, 256, 256)]:
        yield [torch.rand(b, in_channels, h, w)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
