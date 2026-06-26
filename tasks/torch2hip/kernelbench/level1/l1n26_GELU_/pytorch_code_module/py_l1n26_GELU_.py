# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn

class GELU_(nn.Module):
    """
    Simple model that performs a GELU activation.
    """
    def __init__(self):
        super(GELU_, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return torch.nn.functional.gelu(x)

batch_size = 4096
dim = 393216

def get_inputs():
    # Element-wise GELU over varied 2D shapes.
    for b, d in [(1024, 8192), (2048, 16384), (4096, 32768), (512, 65536)]:
        yield [torch.rand(b, d)]


def get_init_inputs():
    return []  # No special initialization inputs needed
