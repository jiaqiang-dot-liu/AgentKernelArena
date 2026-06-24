# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""PyTorch reference for variable-length (thd) RoPE forward, bf16.

Rotary Position Embedding on a packed variable-length ``thd`` tensor (total
tokens T, heads h, head_dim d) delimited by ``cu_seqlens``, matching the AMD
runtime ``rope_thd_fwd`` op. Each sequence is rotated independently using the
within-sequence position to index the (uncached) frequency table ``freqs``; the
op computes cos/sin from ``freqs`` internally:

  for each sequence xi (length s):
    x_rot   = xi[..., rotary slice]
    x_pass  = xi[..., remaining]            (copied through un-rotated)
    cos,sin = cos(freqs[:s]), sin(freqs[:s])
    x_embed = x_rot * cos + rotate_half(x_rot) * sin
    out     = concat(x_embed, x_pass)       (order set by nope_first)

The rotation is computed in fp32 and truncated to bf16 on store, mirroring the
op_test ``ref_rope_thd_fwd(..., simulate_cached=False, comp_with_fp32=True)``
against which the runtime thd-RoPE op is validated. Two rotate styles are
supported (``rotate_style`` 0 = NEOX rotates the second half, 1 = GPT-J rotates
the odd/even pairs). When ``reuse_freqs_front_part`` is set, ``freqs`` holds one
entry per frequency pair and is expanded on use (NEOX repeats, GPT-J
interleaves), so the rotary dim is ``freqs.shape[-1] * 2``; otherwise it is
``freqs.shape[-1]``. ``nope_first`` places the un-rotated slice in front.

forward(input, cu_seqlens, freqs) -> output
  input      : [T, h, d]                 bf16   (packed thd activations)
  cu_seqlens : [num_seqs + 1]            int32  (prefix-sum sequence offsets)
  freqs      : [F, 1, 1, rotary_dim/r]   float  (uncached angle table, F >= max s)
  output     : [T, h, d]                 bf16

``rotate_style``: 0 = NEOX, 1 = GPT-J. ``r`` is 2 when
``reuse_freqs_front_part`` else 1.
"""
import torch
import torch.nn as nn

ROTATE_NEOX = 0
ROTATE_GPTJ = 1


def _rotate_half_neox(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_gptj(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class Model(nn.Module):
    """thd RoPE. ``Model(rotate_style, reuse_freqs_front_part, nope_first)``;
    cu_seqlens and freqs are supplied at call time."""

    def __init__(self, rotate_style=ROTATE_NEOX, reuse_freqs_front_part=True, nope_first=False):
        super().__init__()
        self.rotate_style = int(rotate_style)
        self.reuse_freqs_front_part = bool(reuse_freqs_front_part)
        self.nope_first = bool(nope_first)

    def _rope_sbhd(self, x, freqs):
        rotate_half = (
            _rotate_half_neox if self.rotate_style == ROTATE_NEOX else _rotate_half_gptj
        )
        rotary_dim = freqs.shape[-1] * (2 if self.reuse_freqs_front_part else 1)
        if self.nope_first:
            d = x.shape[-1]
            x_rot, x_pass = x[..., d - rotary_dim :], x[..., : d - rotary_dim]
        else:
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

        f = freqs
        if self.reuse_freqs_front_part:
            if self.rotate_style == ROTATE_NEOX:
                f = f.repeat([1] * (f.dim() - 1) + [2])
            else:
                f = f.repeat_interleave(2, dim=-1)
        cos, sin = f.cos(), f.sin()
        x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
        if self.nope_first:
            return torch.cat((x_pass, x_embed), dim=-1)
        return torch.cat((x_embed, x_pass), dim=-1)

    def forward(self, input, cu_seqlens, freqs):
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        f = freqs.float()
        outs = []
        for xi in torch.split(input, seqlens):
            s = xi.shape[0]
            x = xi.float().unsqueeze(1)  # [s, 1, h, d]
            out = self._rope_sbhd(x, f[:s])
            outs.append(out.squeeze(1))
        return torch.cat(outs).to(input.dtype)


def _build_freqs(max_pos, rotary_dim, reuse_freqs_front_part=True, device="cpu"):
    """Uncached angle table [max_pos, 1, 1, rotary_dim / ratio] (fp32).

    The per-position frequencies are the standard 1/10000^(2k/rotary_dim) RoPE
    schedule; the runtime op consumes these angles and computes cos/sin itself.
    """
    ratio = 2 if reuse_freqs_front_part else 1
    half = rotary_dim // ratio
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(max_pos, device=device).float()
    freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [max_pos, half]
    return freqs.view(max_pos, 1, 1, half).contiguous()


def get_inputs():
    # Packed thd: a few variable-length sequences (cu_seqlens), h heads,
    # d head_dim, full rotary (rotary_dim == d). CPU tensors (KernelBench
    # convention); the consumer/harness relocates them.
    cu = [0, 100, 228, 484, 712, 1024]
    h, d = 8, 128
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)
    t = cu[-1]
    torch.manual_seed(0)
    input = torch.randn(t, h, d, dtype=torch.bfloat16)
    freqs = _build_freqs(t, rotary_dim=d, reuse_freqs_front_part=True).to(torch.bfloat16)
    return [input, cu_seqlens, freqs]


def get_init_inputs():
    # Flat positional args for Model(rotate_style, reuse_freqs_front_part,
    # nope_first): NEOX (0), reuse front part True, nope_first False.
    return [0, True, False]
