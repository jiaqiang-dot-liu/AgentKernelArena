"""
Standalone Triton kernel: fused dual-residual RMSNorm.

Extracted from sglang
  python/sglang/srt/layers/elementwise.py
    -> fused_dual_residual_rmsnorm_kernel  (@triton.jit)
    -> fused_dual_residual_rmsnorm         (host wrapper)

Computes, per row, the fused transformer sub-block
  mid    = residual + RMSNorm1(x) * weight1
  output = RMSNorm2(mid) * weight2
returning (output, mid). This is the post-attention / pre-MLP norm pattern used
by models that apply two back-to-back RMSNorms around a residual add (e.g.
sandwich / dual-residual norm layers). RMS is computed in fp32 over the hidden
dim; the first norm result is cast to the residual dtype before the add (so `mid`
is the new residual), and the final result is cast to the output dtype.

The `triton.autotune` variant, the `_is_hip` host-info query (inlined to True for
AMD Instinct), and the `FusedDualResidualRMSNorm` nn wrapper are dropped; only the
@triton.jit kernel and its deterministic-config host launcher are kept. Depends
ONLY on `torch` / `triton`.

Public entry : fused_dual_residual_rmsnorm
@triton.jit  : fused_dual_residual_rmsnorm_kernel
"""

import torch
import triton
import triton.language as tl

# AMD Instinct (gfx942/gfx950): max warps cap as in the upstream _is_hip branch.
_is_hip = True


@triton.jit
def fused_dual_residual_rmsnorm_kernel(
    output_ptr,
    mid_ptr,
    activ_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    r = tl.load(residual_ptr + input_start + offsets, mask=mask, other=0.0)
    w1_ = tl.load(weight1_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a2r = r + (a / rms * w1).to(r.dtype)
    tl.store(
        mid_ptr + input_start + offsets,
        a2r,
        mask=mask,
    )

    a2r = a2r.to(tl.float32)
    rms2 = tl.sqrt(tl.sum(a2r * a2r, axis=0) / hidden_dim + eps)

    w2_ = tl.load(weight2_ptr + offsets, mask=mask, other=0.0)
    w2 = w2_.to(tl.float32)

    tl.store(
        output_ptr + input_start + offsets,
        a2r / rms2 * w2,  # implicitly casts to output dtype here
        mask=mask,
    )


def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps):
    assert len(x.shape) == 2
    assert (
        x.shape == residual.shape and x.dtype == residual.dtype
    ), f"{x.shape=} {residual.shape=} {x.dtype=} {residual.dtype=}"
    output, mid = torch.empty_like(x), torch.empty_like(x)
    bs, hidden_dim = x.shape

    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    fused_dual_residual_rmsnorm_kernel[(bs,)](
        output,
        mid,
        x,
        residual,
        weight1,
        weight2,
        eps=eps,
        hidden_dim=hidden_dim,
        **config,
    )

    return output, mid
