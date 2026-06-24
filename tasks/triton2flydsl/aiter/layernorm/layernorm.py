# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone LayerNorm (forward) Triton kernel.

Provenance: ported from aiter.ops.triton.normalization.norm (`layer_norm` ->
`_layernorm_forward`) and its device kernel `_layernorm_kernel`
(aiter.ops.triton._triton_kernels.normalization.norm). The quant / fused-add /
backward paths are dropped so the module depends only on `triton` + `torch`.

Op:
    y[i, :] = (x[i, :] - mean(x[i, :])) * rsqrt(var(x[i, :]) + eps) * w + b
with the mean / variance reduction done in fp32 and the result written in the
input dtype. One program normalizes a full row (block-strided loop over n_cols).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    # Pointers to matrices
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `x_row_stride` is
    # how much to increase `x_ptr` by to get the element one row down.
    x_row_stride,
    y_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    eps,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call layer_norm function
    below

    Applies Layer Normalization over a mini-batch of inputs.

    Key parameters:
    - X: The input tensor to be normalized with shape (M, N).
    - Y: The output tensor with the same shape as the input one.
    - W: The learnable weights tensor with shape (N, ).
    - B: The learnable bias tensor with shape (N, ).
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)

    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1

    # Calculate mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  # Unmasked loads
        _mean += x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    _mean += x_block
    mean = tl.sum(_mean, axis=0) / n_cols

    # Calculate variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  # Unmasked loads
        x_block = x_block - mean
        _var += x_block * x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.0)
    _var += x_block * x_block

    var = tl.sum(_var, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    # Write mean / rstd
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Normalize and store
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        w_block = tl.load(w_ptr + col_offsets)
        b_block = tl.load(b_ptr + col_offsets)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)
        y_block = (x_block - mean) * rstd
        y_block = y_block * w_block + b_block
        tl.store(y_ptr_start + col_offsets, y_block)

    # For last iteration, do masked load and store
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
    x_block = tl.load(x_ptr_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y_block = (x_block - mean) * rstd
    y_block = y_block * w_block + b_block
    tl.store(y_ptr_start + col_offsets, y_block, mask=mask)


def _layernorm_forward(
    y: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float = 1e-5,
):

    M, N = x.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    _layernorm_kernel[(M,)](
        x, y, weight, bias, mean, rstd, x.stride(0), y.stride(0), M, N, eps, BLOCK_SIZE
    )

    return


def layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Applies Layer Normalization over a mini-batch of inputs.

    Key parameters:
    - input: The input tensor to be normalized with shape (M, N).
    - weight: The learnable weights tensor with shape (N, ).
    - bias: The learnable bias tensor with shape (N, )
    - eps: A value added to the denominator for numerical stability.

    Returns:
    - Output: The output tensor with shape (M, N).
    """
    y = torch.empty_like(input)
    M = input.shape[0]
    mean = torch.empty((M,), dtype=torch.float32, device=input.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=input.device)
    _layernorm_forward(y, input, weight, bias, mean, rstd, eps)
    return y
