# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference for jagged_dense_bmm_broadcast_add.

For each group ``b`` over its packed row slice ``[s, e)`` (from ``seq_offsets``)::

    Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
      (M_b x N)     (M_b x K)      (K x N)      (1 x N broadcast)

``Dense`` is supplied pre-transposed to ``(B, N, K)``, so the per-group matmul is
``Jagged[s:e] @ Dense[b].T`` with ``Dense[b]`` of shape ``(N, K)``. The math is
done in fp32 accumulation and the result is truncated back to bf16.

forward(jagged, dense, bias, seq_offsets) -> out  (all bf16 except seq_offsets)
  jagged       : (total_M, K)   bf16   packed (jagged) rows for all groups
  dense        : (B, N, K)      bf16   per-group dense weight, pre-transposed
  bias         : (B, N)         bf16   per-group broadcast bias
  seq_offsets  : (B + 1,)       int32  prefix-sum row offsets per group
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, jagged, dense, bias, seq_offsets):
        # jagged (total_M, K); dense (B, N, K); bias (B, N); seq_offsets (B+1,).
        L = jagged.shape[0]
        B, N, K = dense.shape
        out = torch.zeros((L, N), dtype=jagged.dtype, device=jagged.device)
        for b in range(B):
            s = int(seq_offsets[b].item())
            e = int(seq_offsets[b + 1].item())
            if e > s:
                # (M_b, K) @ (K, N) + (1, N) in fp32, truncate to bf16.
                out[s:e] = (
                    jagged[s:e].float() @ dense[b].float().transpose(-1, -2)
                    + bias[b].float()[None, :]
                ).to(jagged.dtype)
        return out


def _make_seq_offsets(m_per_group, device="cpu"):
    so = torch.zeros(len(m_per_group) + 1, dtype=torch.int32, device=device)
    for i, m in enumerate(m_per_group):
        so[i + 1] = so[i] + int(m)
    return so


def get_inputs():
    # Representative jagged shape: B=4 groups with varied per-group row counts
    # M_b summing to total_M, fixed N=K=128. CPU tensors (KernelBench
    # convention; the consumer/harness relocates to the GPU).
    torch.manual_seed(0)
    N, K = 128, 128
    m_per_group = [100, 128, 64, 200]  # varied, unaligned M_b
    B = len(m_per_group)
    total_M = sum(m_per_group)

    seq_offsets = _make_seq_offsets(m_per_group)
    jagged = torch.randn(total_M, K, dtype=torch.bfloat16)
    dense = torch.randn(B, N, K, dtype=torch.bfloat16)
    bias = torch.randn(B, N, dtype=torch.bfloat16)
    return [jagged, dense, bias, seq_offsets]


def get_init_inputs():
    # All operands are forward args, so Model(*get_init_inputs()) == Model().
    return []
