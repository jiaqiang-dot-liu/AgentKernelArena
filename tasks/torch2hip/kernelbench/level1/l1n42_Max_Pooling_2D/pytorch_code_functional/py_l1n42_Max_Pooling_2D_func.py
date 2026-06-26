# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, kernel_size, stride, padding, dilation) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation)


class Max_Pooling_2D(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(Max_Pooling_2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, fn=module_fn):
        return fn(x, self.kernel_size, self.stride, self.padding, self.dilation)


batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    # MaxPool2d is independent of channel count; vary batch/channels/spatial.
    for b, c, h, w in [(16, 32, 256, 256), (32, 64, 128, 128), (8, 16, 512, 512), (32, 64, 256, 256)]:
        yield [torch.rand(b, c, h, w)]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
