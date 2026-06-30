# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn

class Conv2d_InstanceNorm_Divide(nn.Module):
    """
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(Conv2d_InstanceNorm_Divide, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = x / self.divide_by
        return x

batch_size = 128
in_channels  = 64
out_channels = 128
height = width = 128
kernel_size = 3
divide_by = 2.0

def get_inputs():
    # in_channels fixed by get_init_inputs, escalate batch/spatial.
    for b, h, w in [(32, 64, 64), (64, 64, 64), (128, 64, 64), (64, 128, 128), (128, 128, 128)]:
        yield [torch.rand(b, in_channels, h, w)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
