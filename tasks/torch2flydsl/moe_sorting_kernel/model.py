# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference for the MoE token-sorting op.

This is an integer counting-sort (histogram -> prefix-sum -> scatter) that
reorganizes router top-k selections into a per-expert, block-padded token
dispatch layout for batched expert GEMM. Correctness is an exact integer match
(no float tolerance for the integer outputs).

Given router top-k selections:
  * ``topk_ids``     : ``[M, topk]`` int32 -- expert id chosen for each
                       (token, slot). Each token's ``topk`` experts must be
                       unique (MoE router constraint), since one slot is stored
                       per (token, expert) pair.
  * ``topk_weights`` : ``[M, topk]`` float -- routing weight per (token, slot).

it produces:
  * ``sorted_token_ids``  : ``[max_num_tokens_padded]`` int32. Tokens grouped by
                            expert in ascending expert id; within an expert the
                            tokens are in ascending (token, slot) order. Each
                            entry is a packed id ``(slot << 24) | token_id``.
                            Each expert's run is padded up to a multiple of
                            ``block_size`` with the sentinel ``(topk << 24) | M``.
  * ``sorted_weights``    : ``[max_num_tokens_padded]`` float -- the routing
                            weight for each packed token (0.0 on padding slots).
  * ``sorted_expert_ids`` : ``[max_num_m_blocks]`` int32 -- the expert id for
                            each ``block_size`` block of ``sorted_token_ids``.
  * ``num_valid_ids``     : ``[2]`` int32 = ``[total_padded_token_count, M]``.

with
  ``max_num_tokens_padded = M*topk + num_experts*block_size - topk``
  ``max_num_m_blocks       = ceil(max_num_tokens_padded / block_size)``
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_experts, topk, block_size=32):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.block_size = block_size

    def forward(self, topk_ids, topk_weights):
        """Counting sort -> per-expert, block-padded packed token dispatch.

        Returns ``(sorted_token_ids, sorted_weights, sorted_expert_ids,
        num_valid_ids)``.
        """
        num_experts = self.num_experts
        block_size = self.block_size
        device = topk_ids.device
        M, topk = topk_ids.shape

        max_num_tokens_padded = M * topk + num_experts * block_size - topk
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

        sentinel = (topk << 24) | M
        sorted_token_ids = torch.full(
            (max_num_tokens_padded,), sentinel, dtype=torch.int32, device=device
        )
        sorted_weights = torch.zeros(
            (max_num_tokens_padded,), dtype=torch.float32, device=device
        )
        sorted_expert_ids = torch.full(
            (max_num_m_blocks,), -1, dtype=torch.int32, device=device
        )
        num_valid_ids = torch.zeros(2, dtype=torch.int32, device=device)

        ids_cursor = 0
        expert_ids_cursor = 0
        for eid in range(num_experts):
            # torch.where yields (token, slot) in row-major order == ascending
            # token then ascending slot, i.e. the within-expert scatter order.
            token_id, topk_pos = torch.where(topk_ids == eid)
            count = token_id.numel()
            if count == 0:
                continue
            num_blocks = (count + block_size - 1) // block_size
            padded = num_blocks * block_size
            packed = (topk_pos.to(torch.int32) << 24) | token_id.to(torch.int32)
            sorted_token_ids[ids_cursor : ids_cursor + count] = packed
            sorted_weights[ids_cursor : ids_cursor + count] = topk_weights[
                token_id, topk_pos
            ].to(torch.float32)
            ids_cursor += padded
            sorted_expert_ids[expert_ids_cursor : expert_ids_cursor + num_blocks] = eid
            expert_ids_cursor += num_blocks

        num_valid_ids[0] = ids_cursor
        num_valid_ids[1] = M
        return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids


def _gen_topk(M, topk, num_experts, seed=0):
    """Random router top-k with UNIQUE experts per token (MoE router constraint)."""
    g = torch.Generator().manual_seed(seed)
    topk_ids = torch.empty(M, topk, dtype=torch.int32)
    for t in range(M):
        perm = torch.randperm(num_experts, generator=g)[:topk]
        topk_ids[t] = perm.to(torch.int32)
    topk_weights = torch.rand(M, topk, generator=g, dtype=torch.float32)
    return topk_ids, topk_weights


def get_inputs():
    # Representative MoE routing shape (DeepSeek-R1 style): M=16, E=256, topk=8.
    topk_ids, topk_weights = _gen_topk(M=16, topk=8, num_experts=256, seed=42)
    return [topk_ids, topk_weights]


def get_init_inputs():
    # Flat positional args for Model(*get_init_inputs()), matching
    # Model.__init__(num_experts, topk, block_size).
    return [256, 8, 32]
