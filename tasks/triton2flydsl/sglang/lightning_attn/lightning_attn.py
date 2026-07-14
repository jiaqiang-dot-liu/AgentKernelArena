# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/linear_attn.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone Triton kernel: MiniMax / Bailing lightning-attention decode step.

Extracted from sglang
  python/sglang/srt/layers/attention/linear/lightning_attn.py
    -> _linear_attn_decode_kernel  (@triton.jit)
    -> linear_decode_forward_triton (host wrapper)

Single-token linear-attention decode with a per-head exponentially-decayed KV
state (the MiniMax-Text-01 / Bailing linear-attention recurrence). For each
(batch, head) with cache slot `slot_idx[b]` and decay slope `slope_rate[h]`:

  kv_state_new = k (x) v + exp(-slope) * kv_state_old   # outer product, [D, Dv]
  out          = q . kv_state_new                        # contraction over D
  kv_state_old = kv_state_new                            # in-place state update

(slot_idx == -1 marks a padded batch entry and is skipped.) The KV state is kept
in fp32. The chunk-parallel prefill path (`lightning_attention` and the
`_attention` autograd.Function with its four diag/kv/reduce kernels) is the
heavier multi-kernel variant and is NOT part of this task. The einops `rearrange`
in the output reshape is inlined with a torch permute/reshape so the module
depends ONLY on `torch` / `triton`.

Public entry : linear_decode_forward_triton
@triton.jit  : _linear_attn_decode_kernel
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_attn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_cache_ptr,
    slope_rate,
    slot_idx,
    output_ptr,
    D: tl.constexpr,
    qkv_b_stride,
    qkv_h_stride,
    cache_b_stride,
    cache_h_stride,
    cache_d0_stride,
    cache_d1_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for linear attention decoding with KV cache.

    This kernel computes attention for a single token using the KV cache.
    """
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # head index
    pid_d = tl.program_id(2)  # dimension block index

    # Load slot index for the current batch
    slot_id = tl.load(slot_idx + pid_b)

    # Skip if slot_id is -1 (padding)
    if slot_id == -1:
        return

    batch_id = pid_b
    head_id = pid_h

    # Load decay rate for the current head
    ratio = tl.load(slope_rate + pid_h)

    # Calculate offsets for dimensions
    qk_d_offsets = tl.arange(0, D)
    v_d_offsets = tl.arange(0, BLOCK_SIZE) + pid_d * BLOCK_SIZE
    cache_d_offsets = (
        qk_d_offsets[:, None] * cache_d0_stride + v_d_offsets[None, :] * cache_d1_stride
    )

    # Calculate offsets for the current batch and head
    q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride

    cache_offset = slot_id * cache_b_stride + head_id * cache_h_stride

    # Create masks for loading tensors
    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D

    # Load query, key, and value tensors
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)

    # Compute key-value outer product
    kv_outer = k[:, None] * v[None, :]
    kv_mask = qk_mask[:, None] & v_mask[None, :]

    # Apply decay to previous KV cache
    ratio = tl.exp(-ratio)
    kv_ptr = kv_cache_ptr + cache_offset + cache_d_offsets
    kv_cache_old = tl.load(kv_ptr, mask=kv_mask, other=0.0)
    kv_outer = kv_outer + ratio * kv_cache_old

    # Compute attention output
    output = q[:, None].to(tl.float32) * kv_outer
    output = tl.sum(output, axis=0)

    # Update KV cache and store output
    tl.store(kv_ptr, kv_outer, mask=kv_mask)
    tl.store(output_ptr + q_offset + v_d_offsets, output, mask=v_mask)


def linear_decode_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_caches: torch.Tensor,
    slope_rate: torch.Tensor,
    slot_idx: torch.Tensor,
    BLOCK_SIZE: int = 32,
) -> torch.Tensor:
    """
    Perform linear attention decoding using Triton kernels.

    Args:
        q: Query tensor of shape [B, H, 1, D]
        k: Key tensor of shape [B, H, 1, D]
        v: Value tensor of shape [B, H, 1, D]
        kv_caches: Key-value cache tensor
        slope_rate: Decay rate tensor
        slot_idx: Slot indices for batches
        BLOCK_SIZE: Size of blocks for processing

    Returns:
        output: Attention output tensor
    """
    B, H, _, D = q.shape
    assert k.shape == (B, H, 1, D)
    assert v.shape == (B, H, 1, D)

    # Initialize output tensor
    output = torch.empty_like(q)

    # Set grid dimensions for the kernel
    grid = (B, H, D // BLOCK_SIZE)

    # Calculate strides for tensors
    qkv_b_stride = q.stride(0)
    qkv_h_stride = q.stride(1)

    cache_b_stride = kv_caches.stride(0)
    cache_h_stride = kv_caches.stride(1)
    cache_d0_stride = kv_caches.stride(2)
    cache_d1_stride = kv_caches.stride(3)

    # Launch the kernel
    _linear_attn_decode_kernel[grid](
        q,
        k,
        v,
        kv_caches,
        slope_rate,
        slot_idx,
        output,
        D,
        qkv_b_stride,
        qkv_h_stride,
        cache_b_stride,
        cache_h_stride,
        cache_d0_stride,
        cache_d1_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape output and return (einops rearrange "b h n d -> b n (h d)" inlined).
    output = output.permute(0, 2, 1, 3).reshape(B, 1, H * D)
    return output.squeeze(1).contiguous()
