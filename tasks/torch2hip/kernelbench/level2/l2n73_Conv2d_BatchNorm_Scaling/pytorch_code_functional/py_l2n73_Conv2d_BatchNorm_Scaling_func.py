# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var,
              bn_eps, scaling_factor):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, training=False, eps=bn_eps)
    x = x * scaling_factor
    return x


class Conv2d_BatchNorm_Scaling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.bn.weight, self.bn.bias,
                  self.bn.running_mean, self.bn.running_var, self.bn.eps, self.scaling_factor)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0

def get_inputs():
    # in_channels fixed, vary batch/spatial.
    for b, h, w in [(32, 64, 64), (64, 128, 128), (128, 32, 32), (16, 128, 128)]:
        yield [torch.rand(b, in_channels, h, w)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
