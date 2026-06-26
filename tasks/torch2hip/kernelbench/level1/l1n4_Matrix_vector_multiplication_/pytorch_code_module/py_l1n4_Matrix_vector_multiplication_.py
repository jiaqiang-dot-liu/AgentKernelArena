# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn

class Matrix_vector_multiplication_(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(Matrix_vector_multiplication_, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return torch.matmul(A, B)

M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

def get_inputs():
    # Matrix-vector (M,K) @ (K,1).
    for M, K in [(1024, 262144), (2048, 524288), (512, 1048576), (4096, 131072)]:
        yield [torch.rand(M, K), torch.rand(K, 1)]


def get_init_inputs():
    return []  # No special initialization inputs needed
