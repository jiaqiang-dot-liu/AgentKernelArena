# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias, stride, padding,
              groups) -> torch.Tensor:
    return F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)


class conv_depthwise_2D_square_input_square_kernel(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super(conv_depthwise_2D_square_input_square_kernel, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                                padding=padding, groups=in_channels, bias=bias)

    def forward(self, x, fn=module_fn):
        c = self.conv2d
        return fn(x, c.weight, c.bias, c.stride, c.padding, c.groups)


# Test code
batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0

def get_inputs():
    # Depthwise conv2d; in_channels fixed, escalate batch/spatial.
    for b, h, w in [(4, 128, 128), (8, 256, 256), (16, 256, 256), (8, 512, 512), (16, 512, 512)]:
        yield [torch.rand(b, in_channels, h, w)]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
