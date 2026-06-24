# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone fused SiLU-and-mul Triton kernel.

Provenance: ported from aiter.ops.triton.activation (`fused_silu_mul`) and its
device kernel `fused_silu_mul_kernel` / `_silu_exp2`
(aiter.ops.triton._triton_kernels.activation). The mxfp4 / fp8-group quant
kernels and their `_mxfp4_quant_op` / `_fp8_quant_op` imports, plus the
`AiterTritonLogger`, are dropped so the module depends only on `triton` + `torch`.

Op:
    last size 2*d -> first d lanes through SiLU (exp2 form), elementwise-multiplied
    by the second d lanes; output last dim is d. Activation is computed in fp32 and
    truncated to the input dtype (same numerics as the MoE silu-fused GEMM path).
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_exp2(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def fused_silu_mul_kernel(
    inp_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride_in,
    col_stride_in,
    row_stride_out,
    col_stride_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    SiLU on the first half of the last dimension, multiply by the second half.
    Each row has 2 * n_cols input elements; writes n_cols outputs.
    2D grid: axis 0 tiles rows (BLOCK_M), axis 1 tiles columns (BLOCK_N).
    """
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    row_idx = m_pid * BLOCK_M + m_offs
    col_idx = n_pid * BLOCK_N + n_offs

    row_in = row_idx * row_stride_in
    row_out = row_idx * row_stride_out

    first_half_ptrs = inp_ptr + row_in[:, None] + col_idx[None, :] * col_stride_in
    second_half_ptrs = (
        inp_ptr + row_in[:, None] + (n_cols + col_idx)[None, :] * col_stride_in
    )
    out_ptrs = out_ptr + row_out[:, None] + col_idx[None, :] * col_stride_out

    mask = (row_idx < n_rows)[:, None] & (col_idx < n_cols)[None, :]
    a = tl.load(first_half_ptrs, mask=mask, other=0.0).to(tl.float32)
    silu_a = _silu_exp2(a).to(inp_ptr.dtype.element_ty)
    b = tl.load(second_half_ptrs, mask=mask, other=0.0)
    o = (silu_a * b).to(out_ptr.dtype.element_ty)
    tl.store(out_ptrs, o, mask=mask)


def fused_silu_mul(
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused SiLU-and-mul along the last dimension (same pattern as MoE silu-fused GEMM).

    ``x`` must be contiguous with even ``size(-1)``. For last size ``2 * d``, the first
    ``d`` lanes are passed through SiLU (``_silu_exp2``); the second ``d`` lanes are the
    multipliers. Output shape matches ``x`` except ``out.size(-1) == d``.

    Returns:
        ``out`` if provided, else a newly allocated tensor.
    """

    def _pick_block_n(d: int, n_rows: int) -> int:
        """Tile size along the reduced last dim (cap 1024); at least 32 for vectorization.

        Tuned on ROCm for MoE TP4 locals (GLM-4.7 ``d=384``, Kimi-K2.5 ``d=512``) and wide
        MoE activations: ``n_rows`` selects decode vs prefill N-tiling (see sweep in repo
        history / ``bench_moe.py -bench_silu_mul``).
        """
        n = max(d, 1)
        # Kimi-K2.5 TP4 (d=512): prefill favors one 512-wide N tile; decode keeps 256×2.
        if n == 512:
            return 512 if n_rows > 4096 else 256
        # GLM-4.7 TP4 (d=384): wider decode rows use 256×2; larger batches favor 128×3 N tiles.
        if n == 384:
            return 256 if n_rows <= 128 else 128
        upper = min(n, 1024)
        p = 1
        while p * 2 <= upper:
            p *= 2
        return max(32, p)

    def _pick_block_m(n_rows: int, block_n: int, d: int) -> int:
        """Row tile size: latency shapes use wide M tiles; prefill uses tuned (d, n_rows) pairs."""
        if n_rows <= 64:
            return min(32, max(4, triton.next_power_of_2(n_rows)))
        if d == 384 and n_rows > 128:
            return 32 if n_rows > 8192 else 8
        if d == 512 and n_rows > 4096:
            return 8
        if d == 512 and 128 < n_rows <= 4096:
            return 8
        if block_n >= 1024:
            return 8
        if block_n >= 512:
            return 8
        return 16

    def _pick_num_warps(n_rows: int, block_m: int, block_n: int) -> int:
        """ROCm: 8 warps for tiny full-wavefront decode tiles; 2 warps for larger tiles."""
        if n_rows <= 128 and block_m >= 16 and block_n >= 128:
            return 8
        return 2

    assert x.is_cuda, "fused_silu_mul requires a CUDA tensor"
    assert x.is_contiguous(), "x must be contiguous"
    last = x.size(-1)
    assert last % 2 == 0, "last dimension must be even (2 * d)"
    d = last // 2
    leading = x.shape[:-1]
    n_rows = x.numel() // (2 * d)
    if n_rows == 0:
        return (
            torch.empty(*leading, d, dtype=x.dtype, device=x.device)
            if out is None
            else out
        )

    if out is None:
        out = torch.empty(*leading, d, dtype=x.dtype, device=x.device)
    else:
        assert out.is_contiguous(), "out must be contiguous"
        assert out.shape == (*leading, d), "out shape must match x with last dim halved"
        assert out.dtype == x.dtype and out.device == x.device

    row_stride_in = 2 * d
    col_stride_in = 1
    row_stride_out = d
    col_stride_out = 1

    block_n = _pick_block_n(d, n_rows)
    block_m = _pick_block_m(n_rows, block_n, d)
    grid_m = triton.cdiv(n_rows, block_m)
    grid_n = triton.cdiv(d, block_n)
    num_warps = _pick_num_warps(n_rows, block_m, block_n)

    grid = (grid_m, grid_n)
    fused_silu_mul_kernel[grid](
        x,
        out,
        n_rows,
        d,
        row_stride_in,
        col_stride_in,
        row_stride_out,
        col_stride_out,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        waves_per_eu=0,
    )
    return out
