# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn

class Max_Pooling_2D(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(Max_Pooling_2D, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool(x)

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    # MaxPool2d is independent of channel count; escalate batch/channels/spatial.
    for b, c, h, w in [(16, 32, 256, 256), (32, 64, 256, 256), (32, 64, 512, 512),
                       (8, 16, 512, 512), (64, 64, 128, 128)]:
        yield [torch.rand(b, c, h, w)]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
