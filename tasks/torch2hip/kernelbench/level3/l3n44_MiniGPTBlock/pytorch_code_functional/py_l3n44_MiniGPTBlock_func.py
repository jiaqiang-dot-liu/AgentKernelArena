# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                       (x + 0.044715 * torch.pow(x, 3.0))))


def _causal_self_attention(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, mask_bias, n_head):
    B, T, C = x.size()
    qkv = F.linear(x, c_attn_w, c_attn_b)
    q, k, v = qkv.split(C, dim=2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(mask_bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return F.linear(y, c_proj_w, c_proj_b)


def module_fn(x, ln1_w, ln1_b, attn_c_attn_w, attn_c_attn_b, attn_c_proj_w,
              attn_c_proj_b, attn_bias, n_head, ln2_w, ln2_b, mlp_cfc_w, mlp_cfc_b,
              mlp_cproj_w, mlp_cproj_b, n_embd):
    a = F.layer_norm(x, (n_embd,), ln1_w, ln1_b)
    x = x + _causal_self_attention(a, attn_c_attn_w, attn_c_attn_b, attn_c_proj_w,
                                   attn_c_proj_b, attn_bias, n_head)
    m = F.layer_norm(x, (n_embd,), ln2_w, ln2_b)
    h = F.linear(m, mlp_cfc_w, mlp_cfc_b)
    h = _new_gelu(h)
    h = F.linear(h, mlp_cproj_w, mlp_cproj_b)
    x = x + h
    return x


class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                           (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
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


class MiniGPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
            act=NewGELU(),
            dropout=nn.Dropout(resid_pdrop),
        ))
        self.n_embd = n_embd
        self.n_head = n_head

    def forward(self, x, fn=module_fn):
        return fn(x, self.ln_1.weight, self.ln_1.bias,
                  self.attn.c_attn.weight, self.attn.c_attn.bias,
                  self.attn.c_proj.weight, self.attn.c_proj.bias, self.attn.bias, self.n_head,
                  self.ln_2.weight, self.ln_2.bias,
                  self.mlp.c_fc.weight, self.mlp.c_fc.bias,
                  self.mlp.c_proj.weight, self.mlp.c_proj.bias, self.n_embd)


batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    # n_embd fixed; seq_len <= max_seqlen (1024). Escalate batch and seq_len.
    for b, s in [(16, 256), (32, 256), (64, 512), (128, 512), (16, 1024)]:
        yield [torch.rand(b, s, n_embd)]


def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
