# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference for fused QK-RMSNorm + GPT-J RoPE (bf16).

Fuses, for both Q and KV of a decode/prefill step:

  1. per-row RMSNorm (scope = head_dim D, eps = 1e-6), with a per-channel weight
     on KV (and optionally on Q);
  2. GPT-J pair-interleaved RoPE on the last ``rope_head_dim`` (RD) elements
     (the first ``D - RD`` un-rotated elements pass through).

The math is done in fp32 (RMSNorm + RoPE) and the result is truncated to bf16.

forward(q, kv, kv_weight, cos_cache, sin_cache, positions) -> (q_bf16, kv_bf16)
  q          : [T, H*D]            bf16   (Q activations, contiguous over H,D)
  kv         : [T, D]              bf16   (KV pre-norm; may be a strided slice)
  kv_weight  : [D]                 bf16   (per-channel RMSNorm gamma for KV)
  cos_cache  : [max_pos, RD/2]     bf16   (GPT-J RoPE cos table)
  sin_cache  : [max_pos, RD/2]     bf16   (GPT-J RoPE sin table)
  positions  : [T]                 int64  (per-token RoPE position index)
"""

import torch
import torch.nn as nn

_EPS = 1e-6


def _rope_tail_ref(x, cos2d, sin2d, pos, *, D, RD):
    """GPT-J pair-interleaved RoPE on the last RD dims (the rest pass through).

    cos/sin are indexed by per-token ``pos`` and shape ``(..., RD/2)``; each
    GPT-J pair (2k, 2k+1) shares ``cos[k]`` / ``sin[k]`` (REUSE_FREQS_FRONT).
    """
    NOPE = D - RD
    T = x.shape[0]
    leading = x.shape[1:-1]
    tail = x[..., NOPE:].reshape(T, *leading, RD // 2, 2)
    c = cos2d[pos].reshape(T, *((1,) * len(leading)), RD // 2)
    s = sin2d[pos].reshape(T, *((1,) * len(leading)), RD // 2)
    even, odd = tail[..., 0], tail[..., 1]
    new_e = even * c - odd * s
    new_o = even * s + odd * c
    tail_new = torch.stack([new_e, new_o], dim=-1).reshape(T, *leading, RD)
    return torch.cat([x[..., :NOPE], tail_new], dim=-1)


class Model(nn.Module):
    """Fused QK-RMSNorm + GPT-J RoPE (bf16). ``Model(H, D, RD, group_size)``.

    ``group_size`` (one of {32, 64, 128}) is the quant block width; it does not
    affect the bf16 math and is carried only to mirror the kernel's signature.
    """

    def __init__(self, num_q_heads, head_dim, rope_head_dim, group_size):
        super().__init__()
        self.H = num_q_heads
        self.D = head_dim
        self.RD = rope_head_dim
        self.group_size = group_size
        assert head_dim % 2 == 0 and rope_head_dim % 2 == 0
        assert rope_head_dim <= head_dim

    def forward(self, q, kv, kv_weight, cos_cache, sin_cache, positions):
        H, D, RD = self.H, self.D, self.RD
        T = q.shape[0]

        # --- RMSNorm in fp32 (scope = head_dim D, eps = 1e-6) ---
        q3 = q.view(T, H, D).float()
        kvf = kv.float()
        rstd_q = torch.rsqrt(q3.pow(2).mean(-1, keepdim=True) + _EPS)
        rstd_kv = torch.rsqrt(kvf.pow(2).mean(-1, keepdim=True) + _EPS)
        q_n = q3 * rstd_q  # Q has no per-channel weight
        kv_n = kvf * rstd_kv * kv_weight.float()  # KV carries per-channel gamma

        # --- GPT-J RoPE in fp32 on the RD tail ---
        cos2d = cos_cache.view(cos_cache.shape[0], cos_cache.shape[-1]).float()
        sin2d = sin_cache.view(sin_cache.shape[0], sin_cache.shape[-1]).float()
        q_roped = _rope_tail_ref(q_n, cos2d, sin2d, positions, D=D, RD=RD)
        kv_roped = _rope_tail_ref(kv_n, cos2d, sin2d, positions, D=D, RD=RD)

        # Truncate to bf16 (matches the kernel's bf16 store).
        return q_roped.to(torch.bfloat16), kv_roped.to(torch.bfloat16)


def _build_cos_sin(max_pos, RD, device="cpu"):
    """RoPE cos/sin tables, shape [max_pos, RD/2]."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, RD, 2, device=device).float() / RD))
    pos_range = torch.arange(max_pos, device=device).float()
    freqs = torch.einsum("i,j->ij", pos_range, inv_freq)
    cos = freqs.cos().to(torch.bfloat16).contiguous()
    sin = freqs.sin().to(torch.bfloat16).contiguous()
    return cos, sin


def get_inputs():
    # Representative decode shape: T=16 tokens, H=16 Q heads, D=512 head_dim,
    # RD=64 rope tail. CPU tensors (KernelBench convention; the consumer/harness
    # relocates to the GPU).
    H, D, RD = 16, 512, 64
    T = 16
    torch.manual_seed(0)

    max_pos = max(T, 64)
    cos, sin = _build_cos_sin(max_pos, RD)

    q = torch.randn(T, H * D, dtype=torch.bfloat16) * 0.1
    # kv is a strided slice of a wider [T, Q_LORA + D] tensor.
    Q_LORA = 1536
    qkv_a = torch.randn(T, Q_LORA + D, dtype=torch.bfloat16) * 0.1
    _, kv = torch.split(qkv_a, [Q_LORA, D], dim=-1)
    kv_weight = torch.randn(D, dtype=torch.bfloat16).abs() + 0.5
    positions = torch.randint(0, max_pos - 1, (T,), dtype=torch.int64)
    return [q, kv, kv_weight, cos, sin, positions]


def get_init_inputs():
    # Flat positional args for Model(num_q_heads, head_dim, rope_head_dim,
    # group_size).
    return [16, 512, 64, 64]
