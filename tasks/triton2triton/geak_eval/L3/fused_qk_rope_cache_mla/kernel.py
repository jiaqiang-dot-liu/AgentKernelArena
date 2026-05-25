#!/usr/bin/env python3
"""
Fused QK RoPE + KV Cache Kernel for MLA — fused_qk_rope_cat_and_cache_mla.

Inlined from aiter so that this file has zero aiter dependency. The agent only
sees code that lives in this task workspace; anything imported from aiter would
be invisible to optimization.

Sources (verbatim, modulo: dropped @make_kernel_repr/repr= and replaced
AiterTritonLogger with logging.getLogger):
  - aiter/ops/triton/fusions/fused_kv_cache.py          (Python wrapper)
  - aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py  (Triton kernel)
  - aiter/ops/triton/_triton_kernels/rope/rope.py        (rotation helpers)
"""

import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

_LOGGER = logging.getLogger("AITER_TRITON")


# ============================================================================
# INLINED TRITON KERNELS
# ============================================================================


@triton.jit
def _get_neox_rotated_x_1D(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    x_rotated = tl.where(x_rotated_mask, x, -x)
    x_rotated = tl.reshape(x_rotated, (2, BLOCK_D_HALF))
    x_rotated = tl.flip(x_rotated, 1)
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    x_rotated = tl.flip(x_rotated, 0)
    return x_rotated


@triton.jit
def _get_gptj_rotated_x_1D(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    x_rotated = tl.where(x_rotated_mask, x, -x)
    x_rotated = tl.reshape(x_rotated, (BLOCK_D_HALF, 2))
    x_rotated = tl.flip(x_rotated, 1)
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    return x_rotated


@triton.jit
def _unit_rope(
    x_ptrs,
    cos,
    sin,
    d_pe_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    x_pe = tl.load(x_ptrs)

    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    x_pe = x_pe * cos + x_pe_rotated * sin

    return x_pe


@triton.jit
def _fused_qk_rope_cat_and_cache_mla_kernel(
    q_nope_ptr,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    decode_q_pe_out_ptr,
    k_pe_out_ptr,
    q_nope_zeros_out_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    B,
    B_slot,
    num_decode_toks_for_zeros,
    q_nope_stride_b,
    q_nope_stride_h,
    q_nope_stride_d,
    q_pe_stride_b,
    q_pe_stride_h,
    q_pe_stride_d,
    k_nope_stride_b,
    k_nope_stride_h,
    k_nope_stride_d,
    k_pe_stride_b,
    k_pe_stride_h,
    k_pe_stride_d,
    pos_stride_b,
    cos_stride_b,
    cos_stride_d,
    q_out_stride_b,
    q_out_stride_h,
    q_out_stride_d,
    decode_q_pe_out_stride_b,
    decode_q_pe_out_stride_h,
    decode_q_pe_out_stride_d,
    k_pe_out_stride_b,
    k_pe_out_stride_h,
    k_pe_out_stride_d,
    q_nope_zeros_out_stride_b,
    q_nope_zeros_out_stride_h,
    q_nope_zeros_out_stride_d,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_blk,
    kv_cache_stride_d,
    k_scale_ptr,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_DK_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1,
    SHUFFLED_KV_CACHE: tl.constexpr = False,
    OUTPUT_Q_NOPE_ZEROS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)

    d_nope_offs = tl.arange(0, BLOCK_D_nope).to(tl.int64)
    dk_nope_offs = tl.arange(0, BLOCK_DK_nope).to(tl.int64)
    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

    if pid < B * QH:
        pid_b = pid // QH
        pid_hq = pid % QH
        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
                # d_cos_mask = d_cos_offs < BLOCK_D_pe
            else:
                d_cos_offs = d_pe_offs // 2
                # d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe
        else:
            d_cos_offs = d_pe_offs
            # d_cos_mask = d_cos_offs < BLOCK_D_pe

        pos = tl.load(pos_ptr + pid_b * pos_stride_b)
        cos_offs = pos * cos_stride_b + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs)
        sin = tl.load(sin_ptr + cos_offs)

        q_nope_ptrs = (
            q_nope_ptr
            + pid_b * q_nope_stride_b
            + pid_hq * q_nope_stride_h
            + d_nope_offs * q_nope_stride_d
        )
        q_pe_ptrs = (
            q_pe_ptr
            + pid_b * q_pe_stride_b
            + pid_hq * q_pe_stride_h
            + d_pe_offs * q_pe_stride_d
        )
        q_out_ptrs = q_out_ptr + pid_b * q_out_stride_b + pid_hq * q_out_stride_h
        q_nope = tl.load(q_nope_ptrs)
        q_pe = _unit_rope(
            q_pe_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )
        tl.store(
            q_out_ptrs + d_nope_offs * q_out_stride_d,
            q_nope.to(q_out_ptr.dtype.element_ty),
        )
        tl.store(
            q_out_ptrs + (d_pe_offs + BLOCK_D_nope) * q_out_stride_d,
            q_pe.to(q_out_ptr.dtype.element_ty),
        )

        if pid < num_decode_toks_for_zeros * QH:
            decode_q_pe_out_ptrs = (
                decode_q_pe_out_ptr
                + pid_b * decode_q_pe_out_stride_b
                + pid_hq * decode_q_pe_out_stride_h
            )
            tl.store(
                decode_q_pe_out_ptrs + d_pe_offs * decode_q_pe_out_stride_d,
                q_pe.to(decode_q_pe_out_ptr.dtype.element_ty),
            )

        if OUTPUT_Q_NOPE_ZEROS:
            if pid < num_decode_toks_for_zeros * QH:
                z = tl.zeros(
                    (BLOCK_DK_nope,), dtype=q_nope_zeros_out_ptr.dtype.element_ty
                )
                tl.store(
                    q_nope_zeros_out_ptr
                    + pid_b * q_nope_zeros_out_stride_b
                    + pid_hq * q_nope_zeros_out_stride_h
                    + dk_nope_offs * q_nope_zeros_out_stride_d,
                    z,
                )

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if BLOCK_SIZE > 1:
                    pid_t_slot = pid_slot // BLOCK_SIZE
                    pid_blk = pid_slot % BLOCK_SIZE
                else:
                    pid_t_slot = pid_slot
                    pid_blk = 0
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                pid_hk = pid_hq // QH_PER_KH
                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs * k_nope_stride_d
                )
                k_pe_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)
                k_pe = _unit_rope(
                    k_pe_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))
                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                k_pe = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )

                if SHUFFLED_KV_CACHE:
                    if kv_cache_ptr.dtype.element_ty == tl.bfloat16:
                        K_WIDTH: tl.constexpr = 8
                    else:
                        K_WIDTH: tl.constexpr = 16
                    dk_nope_offs_shfl = tl.arange(0, BLOCK_DK_nope // K_WIDTH).to(
                        tl.int64
                    )
                    d_pe_offs_shfl = tl.arange(0, BLOCK_D_pe // K_WIDTH).to(tl.int64)
                    k_width_shfl = tl.arange(0, K_WIDTH).to(tl.int64)
                    k_nope = k_nope.reshape((BLOCK_DK_nope // K_WIDTH, K_WIDTH))
                    k_pe = k_pe.reshape((BLOCK_D_pe // K_WIDTH, K_WIDTH))

                    kv_cache_ptrs = (
                        kv_cache_ptr
                        + pid_t_slot * kv_cache_stride_b
                        + pid_hk * kv_cache_stride_h
                    )
                    kv_cache_nope_offs = (
                        (pid_blk // 16) * BLOCK_DK_nope * 16
                        + (pid_blk % 16) * K_WIDTH
                        + dk_nope_offs_shfl[:, None] * K_WIDTH * 16
                        + k_width_shfl[None, :]
                    ) * kv_cache_stride_d
                    kv_cache_pe_offs = (
                        (pid_blk // 16) * BLOCK_D_pe * 16
                        + (pid_blk % 16) * K_WIDTH
                        + d_pe_offs_shfl[:, None] * K_WIDTH * 16
                        + k_width_shfl[None, :]
                        + BLOCK_SIZE * BLOCK_DK_nope
                    ) * kv_cache_stride_d

                    tl.store(kv_cache_ptrs + kv_cache_nope_offs, k_nope)
                    tl.store(kv_cache_ptrs + kv_cache_pe_offs, k_pe)
                else:
                    kv_cache_ptrs = (
                        kv_cache_ptr
                        + pid_t_slot * kv_cache_stride_b
                        + pid_hk * kv_cache_stride_h
                    )
                    tl.store(kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope)
                    tl.store(
                        kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                        k_pe,
                    )
    else:
        pid = pid - B * QH + B * KH
        if pid < B_slot * KH:
            pid_b = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if BLOCK_SIZE > 1:
                    pid_t_slot = pid_slot // BLOCK_SIZE
                    pid_blk = pid_slot % BLOCK_SIZE
                else:
                    pid_t_slot = pid_slot
                    pid_blk = 0
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs * k_nope_stride_d
                )
                k_pe_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)
                k_pe = tl.load(k_pe_ptrs)
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))
                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                k_pe = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )

                if SHUFFLED_KV_CACHE:
                    if kv_cache_ptr.dtype.element_ty == tl.bfloat16:
                        K_WIDTH: tl.constexpr = 8
                    else:
                        K_WIDTH: tl.constexpr = 16
                    dk_nope_offs_shfl = tl.arange(0, BLOCK_DK_nope // K_WIDTH).to(
                        tl.int64
                    )
                    d_pe_offs_shfl = tl.arange(0, BLOCK_D_pe // K_WIDTH).to(tl.int64)
                    k_width_shfl = tl.arange(0, K_WIDTH).to(tl.int64)
                    k_nope = k_nope.reshape((BLOCK_DK_nope // K_WIDTH, K_WIDTH))
                    k_pe = k_pe.reshape((BLOCK_D_pe // K_WIDTH, K_WIDTH))

                    kv_cache_ptrs = (
                        kv_cache_ptr
                        + pid_t_slot * kv_cache_stride_b
                        + pid_hk * kv_cache_stride_h
                    )
                    kv_cache_nope_offs = (
                        (pid_blk // 16) * BLOCK_DK_nope * 16
                        + (pid_blk % 16) * K_WIDTH
                        + dk_nope_offs_shfl[:, None] * K_WIDTH * 16
                        + k_width_shfl[None, :]
                    ) * kv_cache_stride_d
                    kv_cache_pe_offs = (
                        (pid_blk // 16) * BLOCK_D_pe * 16
                        + (pid_blk % 16) * K_WIDTH
                        + d_pe_offs_shfl[:, None] * K_WIDTH * 16
                        + k_width_shfl[None, :]
                        + BLOCK_SIZE * BLOCK_DK_nope
                    ) * kv_cache_stride_d

                    tl.store(kv_cache_ptrs + kv_cache_nope_offs, k_nope)
                    tl.store(kv_cache_ptrs + kv_cache_pe_offs, k_pe)
                else:
                    kv_cache_ptrs = (
                        kv_cache_ptr
                        + pid_t_slot * kv_cache_stride_b
                        + pid_hk * kv_cache_stride_h
                    )
                    tl.store(kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope)
                    tl.store(
                        kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                        k_pe,
                    )


# ============================================================================
# PYTHON WRAPPER
# ============================================================================


def fused_qk_rope_cat_and_cache_mla(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    is_neox: bool,
    num_decode_toks_for_zeros: int = 0,
    apply_scale: bool = True,
    q_out: torch.Tensor = None,
    decode_q_pe_out: torch.Tensor = None,
    k_pe_out: torch.Tensor = None,
    q_out_dtype: torch.dtype = None,
    shuffled_kv_cache: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform RoPE on q_pe and k_pe and concat q_nope with q_pe and k_nope with k_pe along the last dimension
    the concatentaed k_nope and k_pe are copied to kv_cache inplace
    """
    _LOGGER.info(
        f"FUSED_QK_ROPE_CAT_AND_CACHE_MLA: q_nope={tuple(q_nope.shape)} q_pe={tuple(q_pe.shape)} k_nope={tuple(k_nope.shape)} k_pe={tuple(k_pe.shape)} "
        + f"pos={tuple(pos.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)} kv_cache={tuple(kv_cache.shape)} slot_mapping={tuple(slot_mapping.shape)}"
    )

    b, qh, d_nope = q_nope.shape
    b2, qh2, d_pe = q_pe.shape
    bk, kh, dk_nope = k_nope.shape
    bk2, kh2, dk2 = k_pe.shape
    block_size = 1
    if shuffled_kv_cache:
        b_cache, h_cache, block_size, d_cache = kv_cache.shape
    else:
        b_cache, h_cache, d_cache = kv_cache.shape
    (b_slot,) = slot_mapping.shape

    assert (
        b == b2 and bk == bk2 and b_slot <= bk and b <= bk
    ), "Q batch dimensions should be identical (b == b2), K batch dimensions should be identical (bk == bk2), slot_mapping should not exceed K batch size (b_slot <= bk), and Q batch should not exceed K batch (b <= bk)"
    assert qh == qh2, "Q head should be identical"
    assert kh == kh2 == h_cache, "K head should be identical"
    assert d_pe == dk2, "D dimension of q_pe and k_pe should be identical"
    assert (
        dk_nope + dk2 == d_cache
    ), "D dimension of k_nope and k_pe should be summed up to be the D dimension of kv_cache"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos.shape[-1]
    assert (d_freq == d_pe // 2) or (
        d_freq == d_pe
    ), "cos/sin last dim should be the same or half of the qk last dim"
    assert (
        num_decode_toks_for_zeros >= 0
    ), "num_decode_toks_for_zeros must be non-negative to avoid invalid tensor creation"
    if isinstance(k_scale, torch.Tensor):
        assert k_scale.numel() == 1, "k_scale should be a single-element torch.Tensor"
    reuse_freqs_front_part = d_freq == d_pe // 2

    if q_out is None:
        q_out = torch.empty(
            (b, qh, d_nope + d_pe),
            dtype=q_out_dtype if q_out_dtype is not None else q_nope.dtype,
            device=q_nope.device,
        )
    else:
        b_q_out, qh_q_out, d_q_out = q_out.shape
        assert (
            b == b_q_out and qh == qh_q_out and d_nope + d_pe == d_q_out
        ), "q_out shape mismatch"

    if decode_q_pe_out is None:
        decode_q_pe_out = torch.empty(
            (num_decode_toks_for_zeros, qh, d_pe),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
    else:
        b_decode_q_pe_out, qh_decode_q_pe_out, d_decode_q_pe_out = decode_q_pe_out.shape
        assert (
            num_decode_toks_for_zeros == b_decode_q_pe_out
            and qh == qh_decode_q_pe_out
            and d_pe == d_decode_q_pe_out
        ), "decode_q_pe_out shape mismatch"

    if k_pe_out is None:
        k_pe_out = torch.empty((bk, kh, d_pe), dtype=k_pe.dtype, device=k_pe.device)
    else:
        b_k_pe_out, hk_k_pe_out, d_k_pe_out = k_pe_out.shape
        assert (
            bk == b_k_pe_out and kh == hk_k_pe_out and d_pe == d_k_pe_out
        ), "k_pe_out shape mismatch, expected (bk, kh, d_pe)"

    q_nope_zeros_out = None
    if num_decode_toks_for_zeros > 0:
        q_nope_zeros_out = torch.empty(
            (num_decode_toks_for_zeros, qh, dk_nope),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

    if shuffled_kv_cache:
        kv_cache_stride_b = kv_cache.stride(0)
        kv_cache_stride_h = kv_cache.stride(1)
        kv_cache_stride_blk = kv_cache.stride(2)
        kv_cache_stride_d = kv_cache.stride(3)
    else:
        kv_cache_stride_b = kv_cache.stride(0)
        kv_cache_stride_h = kv_cache.stride(1)
        kv_cache_stride_blk = 0
        kv_cache_stride_d = kv_cache.stride(2)

    n_pid = b * qh + (b_slot - b) * kh
    grid = (n_pid, 1, 1)
    _fused_qk_rope_cat_and_cache_mla_kernel[grid](
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        pos,
        cos,
        sin,
        q_out,
        decode_q_pe_out,
        k_pe_out,
        q_nope_zeros_out,
        kv_cache,
        slot_mapping,
        b,
        b_slot,
        num_decode_toks_for_zeros,
        *q_nope.stride(),
        *q_pe.stride(),
        *k_nope.stride(),
        *k_pe.stride(),
        pos.stride(0),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        *decode_q_pe_out.stride(),
        *k_pe_out.stride(),
        q_nope_zeros_out.stride(0) if q_nope_zeros_out is not None else 0,
        q_nope_zeros_out.stride(1) if q_nope_zeros_out is not None else 0,
        q_nope_zeros_out.stride(2) if q_nope_zeros_out is not None else 0,
        kv_cache_stride_b,
        kv_cache_stride_h,
        kv_cache_stride_blk,
        kv_cache_stride_d,
        k_scale_ptr=k_scale,
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_nope=d_nope,
        BLOCK_DK_nope=dk_nope,
        BLOCK_D_pe=d_pe,
        BLOCK_D_HALF_pe=d_pe // 2,
        BLOCK_SIZE=block_size,
        SHUFFLED_KV_CACHE=shuffled_kv_cache,
        OUTPUT_Q_NOPE_ZEROS=(q_nope_zeros_out is not None),
        HAVE_K_SCALE=(k_scale is not None and apply_scale),
        num_warps=1,
    )

    if q_nope_zeros_out is None:
        # change q_nope_zeros_out from None to a tensor for torch compile
        q_nope_zeros_out = torch.empty(
            (num_decode_toks_for_zeros, qh, dk_nope),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
    return q_out, decode_q_pe_out, k_pe_out, q_nope_zeros_out
