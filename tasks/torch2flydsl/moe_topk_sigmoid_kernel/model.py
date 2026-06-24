# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for sigmoid MoE top-k routing (Llama-4 Maverick).

Given a router gating output ``[num_tokens, num_experts]`` the op produces the
per-token top-k expert ids and routing weights exactly as AMD's fused
``aiter.topk_gating(score_func="sigmoid")`` kernel computes them:

  * ``topk_ids``      = the indices of the ``topk`` largest RAW gating values
    (selection is on the gating logits directly; there is no bias);
  * ``topk_weights``  = ``sigmoid(gating[topk_ids])`` -- the sigmoid is applied
    only to the selected logits, not used for selection.

No top-k renormalization is applied. Selection and the sigmoid are computed in
fp32; the op returns fp32 weights and int32 ids.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_experts, topk):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk

    def forward(self, gating_output):
        router_scores, router_indices = torch.topk(gating_output, self.topk, dim=-1)
        router_scores = torch.sigmoid(router_scores.float())
        return router_scores.float(), router_indices.to(torch.int32)


def selection_scores(gating_output, correction_bias=None):
    """Scores the kernel sorts by to pick the top-k (the raw gating logits).
    Shared with the harness for tie-aware routing comparison."""
    return gating_output.float()


def get_inputs():
    return [torch.randn(64, 128, dtype=torch.bfloat16)]


def get_init_inputs():
    # Model(num_experts, topk) -- Llama-4 Maverick routing.
    return [128, 1]
