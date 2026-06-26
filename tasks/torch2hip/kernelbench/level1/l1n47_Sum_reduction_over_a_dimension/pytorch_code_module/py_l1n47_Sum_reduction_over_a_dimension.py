# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn

class Sum_reduction_over_a_dimension(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Sum_reduction_over_a_dimension, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return torch.sum(x, dim=self.dim, keepdim=True)

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_inputs():
    # Sum reduction over reduce_dim (fixed by get_init_inputs); escalate shape.
    for b, d1, d2 in [(32, 2048, 2047), (64, 2048, 2047), (128, 2048, 2047),
                      (64, 4096, 4095), (128, 1024, 1023)]:
        yield [torch.rand(b, d1, d2)]


def get_init_inputs():
    return [reduce_dim]
