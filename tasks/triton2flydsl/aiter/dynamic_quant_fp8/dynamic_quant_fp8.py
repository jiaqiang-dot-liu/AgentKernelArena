# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone fp8/int8 quantization Triton kernels (static + dynamic per-tensor/per-token).

Provenance: ported from aiter.ops.triton.quant.quant
(`static_per_tensor_quant_fp8_i8`, `dynamic_per_tensor_quant_fp8_i8`,
`dynamic_per_token_quant_fp8_i8`) and their device kernels
(`_static_per_tensor_quant_fp8_i8_kernel`, `_dynamic_per_tensor_quant_fp8_i8_kernel`,
`_dynamic_per_token_quant_fp8_i8_kernel`) in
aiter.ops.triton._triton_kernels.quant.quant. The kernels depend only on `triton`,
so they are copied verbatim and the host wrappers depend only on `triton` + `torch`.

Ops (each row of x [M, N] -> quantized qx + fp32 scale):
  - static_per_tensor:  qx = (x / scale_in).to(qdtype)              (caller scale)
  - dynamic_per_tensor: scale = amax(|x|) / DTYPE_MAX (atomic over rows), then static.
  - dynamic_per_token:  scale[m] = amax_n(|x[m,:]|) / DTYPE_MAX; qx[m] = x[m] / scale[m].
DTYPE_MAX is the max of the target dtype (int8 -> 127, fp8 e4m3 -> 240 on gfx942
e4m3fnuz / 448 on gfx950 e4m3fn). The quant dtype is chosen by the caller's qx tensor.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _static_per_tensor_quant_fp8_i8_kernel(
    qx_ptr,
    x_in_ptr,
    scale_in_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    scale = tl.load(scale_in_ptr)
    scale_recip = 1 / scale

    qx = (x * scale_recip).to(qx_ptr.dtype.element_ty)

    tl.store(qx_ptr + offs, qx, mask=mask)


@triton.jit
def _dynamic_per_tensor_quant_fp8_i8_kernel(
    x_in_ptr,
    scale_out_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x))
    tl.atomic_max(scale_out_ptr, m / DTYPE_MAX, sem="relaxed")


@triton.jit
def _dynamic_per_token_quant_fp8_i8_kernel(
    qx_ptr,
    scale_out_ptr,
    x_in_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x), axis=-1)
    scale_out = m.to(tl.float32) / DTYPE_MAX
    scale_recip = 1 / scale_out

    qx = x * scale_recip
    qx = qx.to(qx_ptr.dtype.element_ty)

    scale_offs = pid
    tl.store(scale_out_ptr + scale_offs, scale_out)

    tl.store(qx_ptr + offs, qx, mask=mask, cache_modifier=".cs")


def _dtype_max(qx: torch.Tensor):
    return (
        torch.finfo(qx.dtype).max
        if torch.is_floating_point(qx)
        else torch.iinfo(qx.dtype).max
    )


def static_per_tensor_quant_fp8_i8(
    qx: torch.Tensor, x_in: torch.Tensor, scale_in: torch.Tensor
):
    """Quantize x_in to fp8/int8 using the caller-provided per-tensor scale.

    Args:
        qx: Output tensor (same shape as x_in), fp8 or int8, caller-allocated.
        x_in: Input tensor of shape (M, N).
        scale_in: fp32 scale tensor of shape (1,).
    Returns:
        qx: Quantized output.
    """
    assert scale_in.numel() == 1
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_in, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )
    return qx


def dynamic_per_tensor_quant_fp8_i8(
    qx: torch.Tensor, x_in: torch.Tensor, scale_out: torch.Tensor
):
    """Compute a per-tensor scale (amax/DTYPE_MAX) then quantize x_in.

    Args:
        qx: Output tensor (same shape as x_in), fp8 or int8, caller-allocated.
        x_in: Input tensor of shape (M, N).
        scale_out: fp32 scale tensor of shape (1,), caller-allocated and zeroed.
    Returns:
        (qx, scale_out).
    """
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_tensor_quant_fp8_i8_kernel[grid](
        x_in,
        scale_out,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=_dtype_max(qx),
    )
    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )
    return qx, scale_out


def dynamic_per_token_quant_fp8_i8(
    qx: torch.Tensor,
    x_in: torch.Tensor,
    scale_out: torch.Tensor,
):
    """Compute a per-row (per-token) scale then quantize x_in.

    Args:
        qx: Output tensor (same shape as x_in), fp8 or int8, caller-allocated.
        x_in: Input tensor of shape (M, N).
        scale_out: fp32 scale tensor of shape (M,), caller-allocated.
    Returns:
        (qx, scale_out).
    """
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_token_quant_fp8_i8_kernel[grid](
        qx,
        scale_out,
        x_in,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=_dtype_max(qx),
    )
    return qx, scale_out
