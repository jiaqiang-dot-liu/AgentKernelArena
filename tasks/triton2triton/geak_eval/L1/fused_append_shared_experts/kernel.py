from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_append_shared_experts_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    M,  # total number of rows
    N_BASE,  # runtime scalar
    scale_factor,  # runtime scalar
    K: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid * BLOCK_M
    rows = row0 + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    # Vectorized load of K columns: [BLOCK_M, K]
    offs_k = tl.arange(0, K)
    in_offsets = rows[:, None] * K + offs_k[None, :]
    ids = tl.load(topk_ids_ptr + in_offsets, mask=row_mask[:, None], other=0)
    ws = tl.load(topk_weights_ptr + in_offsets, mask=row_mask[:, None], other=0.0)

    out_stride = K + S
    out_offsets = rows[:, None] * out_stride + offs_k[None, :]
    tl.store(out_ids_ptr + out_offsets, ids, mask=row_mask[:, None])
    tl.store(out_weights_ptr + out_offsets, ws, mask=row_mask[:, None])

    # Append shared experts: [BLOCK_M, S]
    offs_s = tl.arange(0, S)
    shared_ids = tl.cast(N_BASE + offs_s, ids.dtype)[None, :]
    shared_ws = tl.full([1, S], scale_factor, dtype=ws.dtype)

    out_s_offsets = rows[:, None] * out_stride + (K + offs_s[None, :])
    tl.store(out_ids_ptr + out_s_offsets, shared_ids, mask=row_mask[:, None])
    tl.store(out_weights_ptr + out_s_offsets, shared_ws, mask=row_mask[:, None])


# Pre-allocated output buffer cache - eliminates torch.cat and allocation kernels
_out_ids_buf = None
_out_ws_buf = None
_cache_m = 0
_cache_n = -1
_cache_s = 0
_cache_sf = None
_cache_k = 0
_cdiv = triton.cdiv


def fused_append_shared_experts(
    topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N=None
):
    global _out_ids_buf, _out_ws_buf, _cache_m, _cache_n, _cache_s, _cache_sf, _cache_k
    m, k = topk_ids.shape
    s = int(num_fused_shared_experts)
    if s <= 0:
        return topk_ids, topk_weights

    ks = k + s

    # Re-allocate output buffers only when needed (over-allocate for M)
    if (
        _out_ids_buf is None
        or m > _cache_m
        or k != _cache_k
        or s != _cache_s
        or N != _cache_n
        or scale_factor != _cache_sf
    ):
        alloc_m = max(m, 4096)
        device = topk_ids.device
        _out_ids_buf = torch.empty((alloc_m, ks), dtype=topk_ids.dtype, device=device)
        _out_ws_buf = torch.empty((alloc_m, ks), dtype=topk_weights.dtype, device=device)
        _cache_m = alloc_m
        _cache_n = N
        _cache_s = s
        _cache_sf = scale_factor
        _cache_k = k

    # Use sliced views of pre-allocated buffers
    out_ids = _out_ids_buf[:m]
    out_ws = _out_ws_buf[:m]

    # Single Triton kernel: copy K input columns + write S shared columns
    # One kernel launch instead of two PyTorch copy launches
    BLOCK_M = 64
    grid = (_cdiv(m, BLOCK_M),)
    _fused_append_shared_experts_kernel[grid](
        topk_ids, topk_weights,
        out_ids, out_ws,
        m, N, scale_factor,
        K=k, S=s, BLOCK_M=BLOCK_M,
    )

    return out_ids, out_ws
