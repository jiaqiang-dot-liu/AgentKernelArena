# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, conv_weight, conv_bias, subtract1_value, subtract2_value, kernel_size_pool):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = x - subtract1_value
    x = torch.tanh(x)
    x = x - subtract2_value
    x = F.avg_pool2d(x, kernel_size_pool)
    return x


class Conv2d_Subtract_Tanh_Subtract_AvgPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value,
                 subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.subtract1_value,
                  self.subtract2_value, self.kernel_size_pool)


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
