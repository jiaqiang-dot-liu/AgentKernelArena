# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias, stride, padding,
              dilation, groups) -> torch.Tensor:
    return F.conv2d(x, weight, bias, stride=stride, padding=padding,
                    dilation=dilation, groups=groups)


class conv_standard_2D__square_input__square_kernel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(conv_standard_2D__square_input__square_kernel, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size),
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)

    def forward(self, x, fn=module_fn):
        c = self.conv2d
        return fn(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)


# Test code
batch_size = 16
in_channels = 16
out_channels = 128
kernel_size = 3
width = 1024
height = 1024

def get_inputs():
    # Conv2d; in_channels fixed by get_init_inputs, escalate batch/spatial.
    for b, h, w in [(4, 128, 128), (8, 256, 256), (16, 256, 256), (8, 512, 512), (16, 512, 512)]:
        yield [torch.rand(b, in_channels, h, w)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
