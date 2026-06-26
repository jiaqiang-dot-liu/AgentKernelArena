# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn

class Softmax(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Softmax, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)

batch_size = 4096
dim = 393216

def get_inputs():
    # Row-wise softmax over dim=1.
    for b, d in [(1024, 8192), (2048, 16384), (4096, 32768), (512, 65536)]:
        yield [torch.rand(b, d)]


def get_init_inputs():
    return []  # No special initialization inputs needed
