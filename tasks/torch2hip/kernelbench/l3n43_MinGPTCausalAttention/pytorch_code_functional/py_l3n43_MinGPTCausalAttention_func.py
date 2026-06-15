# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias,
              mask_bias, n_head):
    B, T, C = x.size()
    qkv = F.linear(x, c_attn_weight, c_attn_bias)
    q, k, v = qkv.split(C, dim=2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(mask_bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = F.linear(y, c_proj_weight, c_proj_bias)
    return y


class MinGPTCausalAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                             .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, fn=module_fn):
        return fn(x, self.c_attn.weight, self.c_attn.bias,
                  self.c_proj.weight, self.c_proj.bias, self.bias, self.n_head)


batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
