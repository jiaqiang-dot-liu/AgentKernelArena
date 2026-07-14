# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone fused residual-add + RMSNorm (forward) Triton kernel.

Provenance: ported from aiter.ops.triton.normalization.rmsnorm
(`rmsnorm2d_fwd_with_add` -> `_rmsnorm_forward_with_add`) and its device kernel
`_fused_add_rmsnorm_kernel`
(aiter.ops.triton._triton_kernels.normalization.rmsnorm). The quant / backward
paths and the `get_num_sms` helper are dropped/inlined so the module depends only
on `triton` + `torch`.

Op (the canonical pre-norm transformer-block residual step):
    residual_out = input + residual_in            (written back, input dtype)
    output       = rmsnorm(residual_out) * g       (fp32 reduce, input dtype out)
The sum-of-squares + scale are accumulated in fp32; a blocked persistent kernel
handles wide rows.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    res_in_ptr,
    res_out_ptr,
    g_ptr,
    rsigma_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call
    rmsnorm2d_fwd_with_add function below.

    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - Res_in: The tensor to be added to the Input tensor with shape (n_rows, n_cols).
    - Res_out: The tensor in which the addition result will be stored with shape (n_rows, n_cols).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride
            row_res_in_ptr = res_in_ptr + row_idx * input_row_stride
            row_res_out_ptr = res_out_ptr + row_idx * input_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs)
                res_in_ptrs = row_res_in_ptr + cols
                res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
                res_in = tl.load(res_in_ptrs)
                x += res_in
                # Stores residual_out
                res_out_ptrs = row_res_out_ptr + cols
                tl.store(res_out_ptrs, x.to(res_out_ptr.type.element_ty))

                x = x.to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            res_in_ptrs = row_res_in_ptr + cols
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            x += res_in
            # Stores residual_out
            res_out_ptrs = row_res_out_ptr + cols
            tl.store(res_out_ptrs, x.to(res_out_ptr.type.element_ty), mask=mask)

            x = x.to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                res_out_ptrs = row_res_out_ptr + cols
                res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
                x = tl.load(res_out_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            res_out_ptrs = row_res_out_ptr + cols
            x = tl.load(res_out_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            res_in_ptrs = res_in_ptr + row_idx * input_row_stride + col_offsets
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            row += res_in
            # Stores residual_out
            res_out_ptrs = res_out_ptr + row_idx * input_row_stride + col_offsets
            res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
            tl.store(res_out_ptrs, row.to(res_out_ptr.type.element_ty), mask=mask)
            row = row.to(tl.float32)

            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


def _get_num_sms():
    # Inlined from utils.device_info.get_num_sms (number of compute units).
    return torch.cuda.get_device_properties(0).multi_processor_count


def num_programs(x):
    return min(x.shape[0], _get_num_sms())


def block_size(x):
    return min(65536 // x.element_size(), triton.next_power_of_2(x.shape[1]))


def use_blocked(x):
    return x.shape[1] > block_size(x)


def _rmsnorm_forward_with_add(
    out: torch.Tensor,
    x: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    rsigma: torch.Tensor,
    epsilon: float,
):

    n_rows, n_cols = x.shape

    blk_size = block_size(x)
    USE_BLOCKED = use_blocked(x)
    NUM_PRGMS = num_programs(x)

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _fused_add_rmsnorm_kernel[grid](
        x,
        out,
        residual_in,
        residual_out,
        weight,
        rsigma,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        epsilon,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )


def rmsnorm2d_fwd_with_add(
    out: torch.Tensor,
    input: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
):
    """
    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result.

    Key parameters:
    - Out: The tensor where the output will be stored with shape (M, N).
    - Input: The input tensor to be normalized with shape (M, N).
    - Residual_in: The tensor to be added to the Input tensor with shape (M, N).
    - Residual_out: The tensor in which the addition result will be stored with shape (M, N).
    - Weight: The learnable weights tensor with shape (N, ).
    - Epsilon: A value added to the denominator for numerical stability.

    Returns:
    - Output: The output tensor with shape (M, N).
    """
    M = input.shape[0]
    rsigma = torch.empty((M,), dtype=torch.float32, device=input.device)
    _rmsnorm_forward_with_add(
        out, input, residual_in, residual_out, weight, rsigma, epsilon
    )
    return out
