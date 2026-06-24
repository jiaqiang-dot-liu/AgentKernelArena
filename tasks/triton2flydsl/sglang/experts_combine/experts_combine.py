"""
Standalone Triton kernel: MoE / MLP experts-combine.

Extracted from sglang
  python/sglang/srt/layers/elementwise.py
    -> experts_combine_kernel    (@triton.jit)
    -> experts_combine_triton    (host wrapper)

Combines the routed-MoE branch output with the shared-MLP branch output, per
token, as
  out = (sum_k moe_hidden_states[:, k] + mlp_hidden_states) / sqrt(2)
where moe_hidden_states is [num_tokens, combine_k, hidden_dim] (the top-k expert
outputs that still need summing) or [num_tokens, hidden_dim] (already combined,
combine_k = 1) and mlp_hidden_states is [num_tokens, hidden_dim]. The 1/sqrt(2)
factor balances the variance of the two added branches.

The `register_custom_op` decorator (a torch.library wrapper) is dropped so the
function is a plain Python launcher; everything else is verbatim. Depends ONLY on
`torch` / `triton`.

Public entry : experts_combine_triton
@triton.jit  : experts_combine_kernel
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def experts_combine_kernel(
    out_hidden_states,
    moe_hidden_states,
    mlp_hidden_states,
    combine_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_index_mlp = pid * hidden_dim
    start_index_rmoe = pid * hidden_dim * combine_k
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    combine_k_offsets = tl.arange(0, combine_k)

    moe_x = tl.load(
        moe_hidden_states
        + start_index_rmoe
        + combine_k_offsets[:, None] * hidden_dim
        + offsets[None, :],
        mask=mask[None, :],
        other=0.0,
    )
    moe_x = tl.sum(moe_x, axis=0)
    mlp_x = tl.load(mlp_hidden_states + start_index_mlp + offsets, mask=mask, other=0.0)
    combined_x = (moe_x + mlp_x) / 1.4142135623730951

    tl.store(out_hidden_states + start_index_mlp + offsets, combined_x, mask=mask)


def experts_combine_triton(
    moe_hidden_states: torch.Tensor,
    mlp_hidden_states: torch.Tensor,
    output_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert moe_hidden_states.is_contiguous()
    assert mlp_hidden_states.is_contiguous()

    if len(moe_hidden_states.shape) == 2:
        combine_k = 1  # pre-combined
    else:
        combine_k = moe_hidden_states.shape[1]

    if output_buffer is None:
        out_hidden_states = torch.empty_like(mlp_hidden_states)
    else:
        flat_output_buffer = output_buffer.view(mlp_hidden_states.dtype).reshape(-1)
        assert flat_output_buffer.numel() >= mlp_hidden_states.numel()
        out_hidden_states = flat_output_buffer[: mlp_hidden_states.numel()].reshape(
            mlp_hidden_states.shape
        )

    bs, hidden_dim = mlp_hidden_states.shape

    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 1024)), 8), 4
        ),
    }

    experts_combine_kernel[(bs,)](
        out_hidden_states,
        moe_hidden_states,
        mlp_hidden_states,
        combine_k,
        hidden_dim,
        **config,
    )

    return out_hidden_states
