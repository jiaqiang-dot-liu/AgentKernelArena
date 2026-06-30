# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn

class Tall_skinny_matrix_multiplication_(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(Tall_skinny_matrix_multiplication_, self).__init__()

    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return torch.matmul(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    # Tall-skinny GEMM (M,N) @ (N,M) with M >> N; escalate M and N.
    for M, N in [(8192, 16), (16384, 16), (32768, 32), (65536, 16), (32768, 64)]:
        yield [torch.rand(M, N), torch.rand(N, M)]


def get_init_inputs():
    return []  # No special initialization inputs needed
