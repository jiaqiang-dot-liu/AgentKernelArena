# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, conv_weight, conv_bias, gn_weight, gn_bias, num_groups, scale,
              maxpool_kernel_size, clamp_min, clamp_max):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.group_norm(x, num_groups, gn_weight, gn_bias)
    x = x * scale
    x = F.max_pool2d(x, maxpool_kernel_size)
    x = torch.clamp(x, clamp_min, clamp_max)
    return x


class Conv2d_GroupNorm_Scale_MaxPool_Clamp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape,
                 maxpool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.num_groups = num_groups
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.group_norm.weight,
                  self.group_norm.bias, self.num_groups, self.scale,
                  self.maxpool_kernel_size, self.clamp_min, self.clamp_max)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128 
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    # in_channels fixed; scale_shape independent of spatial, escalate batch/spatial.
    for b, h, w in [(16, 64, 64), (32, 128, 128), (64, 128, 128), (128, 128, 128), (64, 256, 256)]:
        yield [torch.rand(b, in_channels, h, w)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
