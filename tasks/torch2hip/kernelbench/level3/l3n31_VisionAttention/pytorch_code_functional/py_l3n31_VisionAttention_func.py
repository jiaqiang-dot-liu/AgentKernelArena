# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
              norm_weight, norm_bias, num_heads):
    B, C, H, W = x.shape
    x = x.view(B, C, H * W).permute(2, 0, 1)  # (L, N, E)
    L, N, E = x.shape
    head_dim = E // num_heads

    qkv = F.linear(x, in_proj_weight, in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(L, N * num_heads, head_dim).transpose(0, 1)
    k = k.reshape(L, N * num_heads, head_dim).transpose(0, 1)
    v = v.reshape(L, N * num_heads, head_dim).transpose(0, 1)

    attn = F.scaled_dot_product_attention(q, k, v)  # (N*nh, L, hd)
    attn = attn.transpose(0, 1).contiguous().view(L, N, E)
    attn_output = F.linear(attn, out_proj_weight, out_proj_bias)

    x = F.layer_norm(attn_output + x, (E,), norm_weight, norm_bias)
    x = x.permute(1, 2, 0).view(B, C, H, W)
    return x


class VisionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads

    def forward(self, x, fn=module_fn):
        return fn(x, self.attn.in_proj_weight, self.attn.in_proj_bias,
                  self.attn.out_proj.weight, self.attn.out_proj.bias,
                  self.norm.weight, self.norm.bias, self.num_heads)


embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    # num_channels == embed_dim is fixed; escalate batch and image size (sequence length).
    for b, h, w in [(2, 32, 32), (4, 32, 32), (2, 64, 64), (8, 32, 32), (2, 128, 128)]:
        yield [torch.rand(b, num_channels, h, w)]


def get_init_inputs():
    return [embed_dim, num_heads]
