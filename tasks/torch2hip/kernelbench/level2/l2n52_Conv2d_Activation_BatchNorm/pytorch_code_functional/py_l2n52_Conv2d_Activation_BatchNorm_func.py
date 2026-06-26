# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, bn_eps):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = torch.multiply(torch.tanh(F.softplus(x)), x)
    x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, training=False, eps=bn_eps)
    return x


class Conv2d_Activation_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.bn.weight, self.bn.bias,
                  self.bn.running_mean, self.bn.running_var, self.bn.eps)


batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    # in_channels fixed, escalate batch/spatial.
    for b, h, w in [(16, 64, 64), (32, 64, 64), (64, 64, 64), (32, 128, 128), (64, 128, 128)]:
        yield [torch.rand(b, in_channels, h, w)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
