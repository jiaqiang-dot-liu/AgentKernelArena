# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused MoE (Mixture-of-Experts) kernel with MXFP4 weights.

Self-contained / inlined version of aiter.ops.triton.moe_op_mxfp4 — the wrapper
plus all triton kernels and helpers it depends on are pulled into this file so
the module has no dependency on aiter at import time.
"""

import logging
from typing import Any, Dict

import torch
import triton
import triton.language as tl


_LOGGER = logging.getLogger("AITER_TRITON")


# ============================================================================
# INLINED HELPERS (from aiter.ops.triton.utils)
# ============================================================================


# from aiter.ops.triton.utils.device_info.get_num_xcds
def get_num_xcds() -> int:
    # Currently, you can't query this programmatically.
    # For gfx942/gfx950 it's 8, so we hardcode that here.
    return 8


# from aiter.ops.triton.utils.types.torch_to_triton_dtype
torch_to_triton_dtype = {
    torch.float64: tl.float64,
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e4m3fnuz: tl.float8e4b8,
    torch.float8_e5m2: tl.float8e5,
    torch.float8_e5m2fnuz: tl.float8e5b16,
    torch.int64: tl.int64,
    torch.int32: tl.int32,
    torch.int16: tl.int16,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
}


def get_scaled_dot_format_string(dtype: tl.dtype) -> str:
    mapping = {
        tl.float16: "fp16",
        tl.bfloat16: "bf16",
        tl.uint8: "e2m1",
        tl.float8e4nv: "e4m3",
        tl.float8e5: "e5m2",
    }
    return mapping[dtype]


# ============================================================================
# INLINED TRITON KERNELS (from aiter.ops.triton._triton_kernels)
# ============================================================================


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _fused_moe_kernel_mxfp4(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    a_mx_scale_ptr,
    b_mx_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_amxm,
    stride_amxk,
    stride_bmxe,
    stride_bmxk,
    stride_bmxn,
    # Meta-parameters
    A_DTYPE_FORMAT: tl.constexpr,
    B_DTYPE_FORMAT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWIZZLE_MX_A: tl.constexpr,  # TODO add swizzle support
    SWIZZLE_MX_B: tl.constexpr,  # TODO add swizzle support
    NUM_XCDS: tl.constexpr,
):
    is_a_microscaled_format: tl.constexpr = a_mx_scale_ptr is not None
    is_b_microscaled_format: tl.constexpr = b_mx_scale_ptr is not None
    MX_PACK_DIVISOR: tl.constexpr = 32
    if is_a_microscaled_format:
        a_type: tl.constexpr = a_ptr.dtype.element_ty
        tl.static_assert(
            a_type == tl.uint8 or (a_type == tl.float8e4nv or a_type == tl.float8e5),
            "mx_weight_ptr must be 1 byte",
        )
        tl.static_assert(
            a_mx_scale_ptr.dtype.element_ty == tl.uint8, "a_mx_scale_ptr must be uint8"
        )
        tl.static_assert(
            BLOCK_SIZE_K % MX_PACK_DIVISOR == 0,
            "BLOCK_SIZE_K must be a multiple of MX_PACK_DIVISOR",
        )
    if is_b_microscaled_format:
        b_type: tl.constexpr = b_ptr.dtype.element_ty
        tl.static_assert(
            b_type == tl.uint8 or (b_type == tl.float8e4nv or b_type == tl.float8e5),
            "mx_weight_ptr must be 1 byte",
        )
        tl.static_assert(
            b_mx_scale_ptr.dtype.element_ty == tl.uint8, "b_mx_scale_ptr must be uint8"
        )
        tl.static_assert(
            BLOCK_SIZE_K % MX_PACK_DIVISOR == 0,
            "BLOCK_SIZE_K must be a multiple of MX_PACK_DIVISOR",
        )

    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return  # rest of the tiles are dummy paddings
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr + off_expert)
    offs_b_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_b_n = tl.max_contiguous(
        tl.multiple_of(offs_b_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N
    )

    if is_a_microscaled_format:
        A_PACK_DIVISOR: tl.constexpr = 2 if a_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // A_PACK_DIVISOR
        MX_SCALE_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR

        if SWIZZLE_MX_A:
            tl.static_assert(BLOCK_SIZE_M % 128 == 0)
            tl.static_assert(MX_SCALE_BLOCK_K_A % 4 == 0)
            PACKED_MX_BLOCK_A: tl.constexpr = (MX_SCALE_BLOCK_K_A // 4) * 32 * 4 * 4
            offs_inner = tl.arange(0, PACKED_MX_BLOCK_A)
            offs_scale_m = (
                pid_m * (BLOCK_SIZE_M // 128) + tl.arange(0, BLOCK_SIZE_M // 128)
            ) % N
            offs_scale_m = tl.max_contiguous(
                tl.multiple_of(offs_scale_m, BLOCK_SIZE_M // 128), BLOCK_SIZE_M // 128
            )

            a_mx_scale_ptrs = (
                a_mx_scale_ptr
                + offs_scale_m.to(tl.int64)[:, None] * stride_amxm
                + offs_inner[None, :]
            )
        else:
            offs_scale_ak = tl.arange(0, MX_SCALE_BLOCK_K_A)
            offs_scale_m = offs_token
            a_mx_scale_ptrs = (
                a_mx_scale_ptr
                + offs_scale_ak.to(tl.int64)[None, :] * stride_amxk
                + offs_scale_m.to(tl.int64)[:, None] // top_k * stride_amxm
            )
    else:
        a_mx_scale_ptrs = None
        A_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_A: tl.constexpr = 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K

    if is_b_microscaled_format:
        B_PACK_DIVISOR: tl.constexpr = 2 if b_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // B_PACK_DIVISOR
        MX_SCALE_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR

        b_mx_scale_ptr += off_expert * stride_bmxe

        if SWIZZLE_MX_B:
            tl.static_assert(BLOCK_SIZE_N % 128 == 0)
            tl.static_assert(MX_SCALE_BLOCK_K_B % 4 == 0)
            PACKED_MX_BLOCK_B: tl.constexpr = (MX_SCALE_BLOCK_K_B // 4) * 32 * 4 * 4
            offs_inner = tl.arange(0, PACKED_MX_BLOCK_B)
            offs_scale_n = (
                pid_n * (BLOCK_SIZE_N // 128) + tl.arange(0, BLOCK_SIZE_N // 128)
            ) % N
            offs_scale_n = tl.max_contiguous(
                tl.multiple_of(offs_scale_n, BLOCK_SIZE_N // 128), BLOCK_SIZE_N // 128
            )

            b_mx_scale_ptrs = (
                b_mx_scale_ptr
                + offs_scale_n.to(tl.int64)[:, None]
                * PACKED_MX_BLOCK_B
                * (K // MX_SCALE_BLOCK_K_B // (MX_PACK_DIVISOR // B_PACK_DIVISOR))
                + offs_inner[None, :]
            )
        else:
            offs_scale_bk = tl.arange(0, MX_SCALE_BLOCK_K_B)
            offs_scale_n = offs_b_n
            b_mx_scale_ptrs = (
                b_mx_scale_ptr
                + offs_scale_bk.to(tl.int64)[None, :] * stride_bmxk
                + offs_scale_n.to(tl.int64)[:, None] * stride_bmxn
            )
    else:
        b_mx_scale_ptrs = None
        B_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_B: tl.constexpr = 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K

    offs_a_k = tl.arange(0, PACKED_BLOCK_K_A)
    offs_b_k = tl.arange(0, PACKED_BLOCK_K_B)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_a_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, PACKED_BLOCK_K_A)):
        if EVEN_K:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None]
                & (offs_a_k[None, :] < (K - k * PACKED_BLOCK_K_A)),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_b_k[:, None] < (K - k * PACKED_BLOCK_K_B),
                other=0.0,
            )
        if is_a_microscaled_format or is_b_microscaled_format:
            if is_a_microscaled_format:
                mask_ak_scale = offs_scale_ak < (K - k * PACKED_BLOCK_K_A) // (
                    MX_PACK_DIVISOR // A_PACK_DIVISOR
                )
                a_mx_scales = tl.load(
                    a_mx_scale_ptrs, mask=mask_ak_scale[None, :], other=0.0
                )
            else:
                a_mx_scales = None
            mask_bk_scale = offs_scale_bk < (K - k * PACKED_BLOCK_K_B) // (
                MX_PACK_DIVISOR // B_PACK_DIVISOR
            )
            b_mx_scales = tl.load(
                b_mx_scale_ptrs, mask=mask_bk_scale[None, :], other=0.0
            )

            accumulator = tl.dot_scaled(
                a,
                a_mx_scales,
                A_DTYPE_FORMAT,
                b,
                b_mx_scales,
                B_DTYPE_FORMAT,
                acc=accumulator,
                fast_math=True,
            )

            if is_a_microscaled_format:
                if SWIZZLE_MX_A:
                    a_mx_scale_ptrs += MX_SCALE_BLOCK_K_A // 4 * stride_amxk
                else:
                    a_mx_scale_ptrs += MX_SCALE_BLOCK_K_A * stride_amxk
            if SWIZZLE_MX_B:
                b_mx_scale_ptrs += MX_SCALE_BLOCK_K_B // 4 * 512
            else:
                b_mx_scale_ptrs += MX_SCALE_BLOCK_K_B * stride_bmxk
        a_ptrs += PACKED_BLOCK_K_A * stride_ak
        b_ptrs += PACKED_BLOCK_K_B * stride_bk

    accumulator *= a_scale * b_scale
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ============================================================================
# WRAPPER
# ============================================================================


def fused_moe_mxfp4(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    A_mx_scale: torch.Tensor,
    B_mx_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    swizzle_mx_a: bool,
    swizzle_mx_b: bool,
    config: Dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """
    Fused MoE computation with MXFP4 (microscale FP4) quantization.

    Args:
        A (torch.Tensor): Input activations with shape (num_tokens, hidden_dim). FP4 or higher precision.
        B (torch.Tensor): Expert weights with shape (num_experts, hidden_dim, intermediate_dim). MXFP4 format.
        C (torch.Tensor): Output tensor with shape (num_tokens, top_k, intermediate_dim).
        A_scale (torch.Tensor): Per-tensor or per-group scale for A.
        B_scale (torch.Tensor): Per-group scale for B with shape (num_experts, ...).
        A_mx_scale (torch.Tensor): Microscale (E8M0) scale for A if A is MXFP4.
        B_mx_scale (torch.Tensor): Microscale (E8M0) scale for B.
        topk_weights (torch.Tensor): Routing weights for top-k experts with shape (num_tokens, top_k).
        topk_ids (torch.Tensor): Top-k expert IDs per token with shape (num_tokens, top_k).
        sorted_token_ids (torch.Tensor): Token IDs sorted by expert assignment.
        expert_ids (torch.Tensor): Expert ID for each sorted token.
        num_tokens_post_padded (torch.Tensor): Total tokens after block-size padding with shape (1,).
        mul_routed_weight (bool): Multiply output by routing weights.
        top_k (int): Number of experts per token.
        swizzle_mx_a (bool): Enable swizzled layout for A microscales.
        swizzle_mx_b (bool): Enable swizzled layout for B microscales.
        config (Dict[str, Any]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).
        compute_type (tl.dtype): Computation dtype for accumulation.

    Returns:
        None. Results written in-place to C.
    """
    _LOGGER.info(
        f"MOE_OP_MXFP4:  A={tuple(A.shape)}  B={tuple(B.shape)}  C={tuple(C.shape)} "
        + "A_scale={tuple(A_scale.shape)}  B_scale={tuple(B_scale.shape)} "
        + "A_mx_scale={tuple(A_mx_scale.shape)}  B_mx_scale={tuple(B_mx_scale.shape)} "
        + "topk_weights={tuple(topk_weights.shape)} sorted_token_ids={tuple(sorted_token_ids.shape)} "
        + "expert_ids={tuple(expert_ids.shape)} num_tokens_post_padded={tuple(num_tokens_post_padded.shape)} "
        + "top_k={top_k}"
    )
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    assert A_scale is not None
    assert B_scale is not None
    if A.dtype == torch.uint8:
        assert A_mx_scale is not None, "A_mx_scale should exist when A is mxfp4"
        A_mx_scale_strid_m, A_mx_scale_strid_k = A_mx_scale.stride()
    else:
        assert A_mx_scale is None, "A_mx_scale should not exist when A is not mxfp4"
        A_mx_scale_strid_m, A_mx_scale_strid_k = None, None
    # NOTE: Only supports B_mx_scale
    assert B_mx_scale is not None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    A_tl_dtype = torch_to_triton_dtype[A.dtype]
    A_DTYPE_FORMAT = get_scaled_dot_format_string(A_tl_dtype)
    B_tl_dtype = torch_to_triton_dtype[B.dtype]
    B_DTYPE_FORMAT = get_scaled_dot_format_string(B_tl_dtype)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    _fused_moe_kernel_mxfp4[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        A_mx_scale,
        B_mx_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        A.shape[1],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_mx_scale_strid_m,
        A_mx_scale_strid_k,
        B_mx_scale.stride(0),
        B_mx_scale.stride(2),
        B_mx_scale.stride(1),
        A_DTYPE_FORMAT=A_DTYPE_FORMAT,
        B_DTYPE_FORMAT=B_DTYPE_FORMAT,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        SWIZZLE_MX_A=swizzle_mx_a,  # TODO add swizzle support
        SWIZZLE_MX_B=swizzle_mx_b,  # TODO add swizzle support
        NUM_XCDS=get_num_xcds(),
        **config,
    )
