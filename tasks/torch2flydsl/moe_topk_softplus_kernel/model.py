# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for sqrt-softplus MoE top-k routing (DeepSeek-V4-Pro).

Given a router gating output ``[num_tokens, num_experts]`` the op produces the
per-token top-k expert ids and routing weights exactly as AMD's fused
``aiter.topk_softplus`` (``aiter.topk_gating(score_func="sqrtsoftplus")``) kernel
computes them:

  * scores            = ``sqrt(softplus(gating))``;
  * selection scores  = ``scores + correction_bias``;
  * ``topk_ids``      = the indices of the ``topk`` largest selection scores;
  * ``topk_weights``  = the UN-biased ``scores`` gathered at those ids,
    optionally renormalized to sum to 1 across the top-k, then multiplied by
    ``routed_scaling_factor``.

All scoring and selection are computed in fp32; the op returns fp32 weights and
int32 ids, matching the kernel output contract.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_experts, topk, renormalize=True, route_scale=2.5):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.renormalize = renormalize
        self.route_scale = route_scale
        self.correction_bias = nn.Parameter(
            torch.randn(num_experts, dtype=torch.float32) * 0.1
        )

    def forward(self, gating_output):
        scores = F.softplus(gating_output.float()).sqrt()
        scores_biased = scores + self.correction_bias.float()
        topk_ids = scores_biased.topk(self.topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * self.route_scale
        return topk_weights.float(), topk_ids.to(torch.int32)


def selection_scores(gating_output, correction_bias):
    """Scores the kernel sorts by to pick the top-k (sqrt-softplus + bias).
    Shared with the harness for tie-aware routing comparison."""
    scores = F.softplus(gating_output.float()).sqrt()
    if correction_bias is not None and correction_bias.numel() > 0:
        scores = scores + correction_bias.float()
    return scores


def get_inputs():
    return [torch.randn(64, 256, dtype=torch.bfloat16)]


def get_init_inputs():
    # Model(num_experts, topk, renormalize, route_scale) -- DeepSeek-V4-Pro.
    return [256, 8, True, 2.5]
