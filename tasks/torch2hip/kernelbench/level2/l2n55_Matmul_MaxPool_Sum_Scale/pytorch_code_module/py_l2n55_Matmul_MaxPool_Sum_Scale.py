# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn

class Matmul_MaxPool_Sum_Scale(nn.Module):
    """
    Matmul_MaxPool_Sum_Scale that performs matrix multiplication, max pooling, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(Matmul_MaxPool_Sum_Scale, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.max_pool(x.unsqueeze(1)).squeeze(1)
        x = torch.sum(x, dim=1)
        x = x * self.scale_factor
        return x

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    # in_features fixed, vary batch.
    for b in [32, 64, 128, 256]:
        yield [torch.rand(b, in_features)]


def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
