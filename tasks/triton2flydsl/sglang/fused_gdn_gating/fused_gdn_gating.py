"""
Standalone Triton kernel: fused Gated-DeltaNet (GDN) input gating.

Extracted from sglang
  python/sglang/srt/layers/attention/fla/fused_gdn_gating.py
    -> fused_gdn_gating_kernel
    -> fused_gdn_gating (host wrapper)

Computes the two per-head gating signals consumed by the Gated Delta Net
recurrence on the Qwen3-Next / Kimi-Linear serving path (decode step,
seq_len == 1). For projection outputs a, b and the learned A_log / dt_bias:

    g           = -exp(A_log) * softplus(a + dt_bias)   # forget gate (log space)
    beta_output = sigmoid(b)                             # delta write strength

where softplus is the numerically-stable, beta/threshold-parameterized form:

    softplus(x) = (1/beta) * log(1 + exp(beta*x))   if beta*x <= threshold
                = x                                  otherwise

`g` is accumulated and stored in fp32; `beta_output` is stored at `b`'s dtype.
Depends ONLY on `torch` and `triton`.

Public entry : fused_gdn_gating
@triton.jit  : fused_gdn_gating_kernel
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
# beta_output = b.sigmoid()
@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
    blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape
    seq_len = 1
    stride_a = a.stride(0)
    stride_b = b.stride(0)
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        stride_a,
        stride_b,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
