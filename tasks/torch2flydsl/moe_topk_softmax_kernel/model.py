# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for softmax MoE top-k routing (classic / DeepSeek-V3).

Given a router gating output ``[num_tokens, num_experts]`` the op produces the
per-token top-k expert ids and their routing weights, exactly as AMD's fused
``aiter.topk_gating(score_func="softmax")`` kernel computes them:

  * selection scores  = ``softmax(gating, dim=-1) + correction_bias``  (the bias
    is added AFTER softmax normalization, matching the kernel);
  * ``topk_ids``      = the indices of the ``topk`` largest selection scores;
  * ``topk_weights``  = the UN-biased softmax probabilities gathered at those
    ids, multiplied by ``routed_scaling_factor``.

Softmax is already normalized, so no top-k renormalization is applied. The
reference computes the softmax and selection in fp32, returns fp32 weights and
int32 ids, matching the kernel's output contract.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_experts, topk, route_scale=1.0, use_bias=True):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.route_scale = route_scale
        self.use_bias = use_bias
        if use_bias:
            self.correction_bias = nn.Parameter(
                torch.randn(num_experts, dtype=torch.float32) * 0.1
            )
        else:
            self.register_parameter("correction_bias", None)

    def forward(self, gating_output):
        scores = torch.softmax(gating_output.float(), dim=-1)
        if self.correction_bias is not None:
            scores_biased = scores + self.correction_bias.float()
        else:
            scores_biased = scores
        topk_ids = scores_biased.topk(self.topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids) * self.route_scale
        return topk_weights.float(), topk_ids.to(torch.int32)


def selection_scores(gating_output, correction_bias):
    """Scores the kernel sorts by to pick the top-k (softmax + bias). Shared with
    the harness for tie-aware routing comparison."""
    scores = torch.softmax(gating_output.float(), dim=-1)
    if correction_bias is not None and correction_bias.numel() > 0:
        scores = scores + correction_bias.float()
    return scores


def get_inputs():
    return [torch.randn(64, 256, dtype=torch.bfloat16)]


def get_init_inputs():
    # Model(num_experts, topk, route_scale, use_bias) -- DeepSeek-V3 routing.
    return [256, 8, 1.0, True]
