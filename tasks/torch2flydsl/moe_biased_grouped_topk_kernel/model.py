# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for biased grouped MoE top-k routing (DeepSeek-V3).

DeepSeek-V3 routes each token in two stages over expert *groups*:

  * scores            = ``sigmoid(gating)``  (fp32);
  * selection scores  = ``scores + correction_bias``;
  * per-group score    = sum of the top-2 selection scores within each group;
  * pick the ``topk_group`` highest-scoring groups, mask out all other groups,
    then pick the ``topk`` experts by selection score among the kept groups;
  * ``topk_weights``  = the UN-biased ``scores`` gathered at the selected ids,
    optionally renormalized to sum to 1 across the top-k, then multiplied by
    ``routed_scaling_factor``.

This mirrors ``aiter.ops.topk.biased_grouped_topk_torch`` (the routed-scaling
factor is applied here, matching the fused ``biased_grouped_topk_hip`` op). All
scoring is fp32; the op returns fp32 weights and int32 ids.
"""
import torch
import torch.nn as nn


def grouped_route(gating_output, correction_bias, topk, renormalize,
                  num_expert_group, topk_group, route_scale):
    """DeepSeek-V3 biased grouped routing. Returns (topk_weights, topk_ids,
    masked_selection_scores) so the harness can reuse the masked scores for
    tie-aware comparison."""
    scores = gating_output.float().sigmoid()
    num_token = scores.shape[0]

    scores_for_choice = scores + correction_bias.float().unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * route_scale
    return topk_weights.float(), topk_ids.to(torch.int32), tmp_scores


class Model(nn.Module):
    def __init__(self, num_experts, topk, num_expert_group, topk_group,
                 renormalize=True, route_scale=2.5):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.renormalize = renormalize
        self.route_scale = route_scale
        self.correction_bias = nn.Parameter(
            torch.randn(num_experts, dtype=torch.float32) * 0.1
        )

    def forward(self, gating_output):
        topk_weights, topk_ids, _ = grouped_route(
            gating_output, self.correction_bias, self.topk, self.renormalize,
            self.num_expert_group, self.topk_group, self.route_scale,
        )
        return topk_weights, topk_ids


def selection_scores(gating_output, correction_bias, topk, num_expert_group,
                     topk_group, renormalize=True, route_scale=2.5):
    """Masked per-expert selection scores used to pick the final top-k (0 outside
    the selected groups). Shared with the harness for tie-aware comparison."""
    _, _, tmp_scores = grouped_route(
        gating_output, correction_bias, topk, renormalize,
        num_expert_group, topk_group, route_scale,
    )
    return tmp_scores


def get_inputs():
    return [torch.randn(64, 256, dtype=torch.bfloat16)]


def get_init_inputs():
    # Model(num_experts, topk, num_expert_group, topk_group, renormalize,
    # route_scale) -- DeepSeek-V3 routing (256 experts, 8 groups, top-4 groups,
    # top-8 experts).
    return [256, 8, 8, 4, True, 2.5]
