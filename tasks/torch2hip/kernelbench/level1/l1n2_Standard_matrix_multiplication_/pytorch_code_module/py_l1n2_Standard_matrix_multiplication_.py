# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn

class Standard_matrix_multiplication_(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Standard_matrix_multiplication_, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    # Standard GEMM (M,K) @ (K,N); vary all three dims.
    for M, K, N in [(512, 1024, 768), (1024, 2048, 1024), (2048, 4096, 2048), (1536, 1536, 1536)]:
        yield [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []  # No special initialization inputs needed
