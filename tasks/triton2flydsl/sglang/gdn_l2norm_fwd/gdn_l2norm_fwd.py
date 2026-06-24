"""
Standalone Triton kernel: L2 normalization (forward), last-dim normalize.

Extracted from sglang
  python/sglang/srt/layers/attention/fla/l2norm.py
    -> l2norm_fwd_kernel  (D <= 512 block-ptr path)
    -> l2norm_fwd_kernel1 (D  > 512 row path)
    -> l2norm_fwd (host wrapper)

Used in the Gated Delta Net (GDN) chunk-prefill pipeline to L2-normalize q/k
(`use_qk_l2norm_in_kernel`) on the Qwen3.5-35B-A3B serving path
(`l2norm_fwd_kernel`, ~3.7% of the GDN prefill pipeline GPU time).

Pure L2 norm along the last dim (NO mean subtraction):
    y = x / sqrt(sum(x*x, dim=-1) + eps)

Depends ONLY on `torch` and `triton`.

Public entry : l2norm_fwd
@triton.jit  : l2norm_fwd_kernel, l2norm_fwd_kernel1
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x, y, eps, NB=NB, T=T, D=D, BD=BD, BT=16, num_warps=8, num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x, y, eps=eps, D=D, BD=BD, num_warps=8, num_stages=3,
        )

    return y.view(x_shape_og)
