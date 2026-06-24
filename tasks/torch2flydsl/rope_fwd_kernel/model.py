# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference for cached-cos/sin RoPE forward, sbhd layout (bf16).

Rotary Position Embedding on an ``sbhd`` tensor with precomputed (cached) cos and
sin tables, matching the AMD runtime cached-RoPE forward op:

  x_rot   = x[..., rotary slice]
  x_pass  = x[..., remaining]      (copied through un-rotated)
  x_embed = x_rot * cos + rotate_half(x_rot) * sin
  out     = concat(x_embed, x_pass)   (order set by nope_first)

The rotation is done in fp32 and truncated to bf16 on store. Two rotate styles
are supported (``rotate_style`` 0 = NEOX rotates the second half, 1 = GPT-J
rotates the odd/even pairs). When ``reuse_freqs_front_part`` is set the cached
cos/sin hold one entry per frequency pair and are expanded on use (NEOX repeats,
GPT-J interleaves), so the rotary dim is ``cos.shape[-1] * 2``; otherwise it is
``cos.shape[-1]``. ``nope_first`` places the un-rotated (no-position) slice in
front of the rotary slice instead of after it.

Primary variant (get_inputs / get_init_inputs): NEOX, reuse_freqs_front_part
True, nope_first False, full rotary (rotary_dim == head_dim). Mirrors the op_test
``ref_rope_sbhd_fwd(..., simulate_cached=True, comp_with_fp32=True)`` against
which the runtime cached-RoPE op is validated.

forward(input, cos, sin) -> output
  input  : [s, b, h, d]                 bf16   (sbhd activations)
  cos    : [s, 1, 1, rotary_dim/ratio]  bf16   (cached cos table)
  sin    : [s, 1, 1, rotary_dim/ratio]  bf16   (cached sin table)
  output : [s, b, h, d]                 bf16

``rotate_style``: 0 = NEOX, 1 = GPT-J. ``ratio`` is 2 when
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
    """Cached-cos/sin RoPE (sbhd). ``Model(rotate_style, reuse_freqs_front_part,
    nope_first)``; cos/sin caches are supplied at call time."""

    def __init__(self, rotate_style=ROTATE_NEOX, reuse_freqs_front_part=True, nope_first=False):
        super().__init__()
        self.rotate_style = int(rotate_style)
        self.reuse_freqs_front_part = bool(reuse_freqs_front_part)
        self.nope_first = bool(nope_first)

    def forward(self, input, cos, sin):
        x = input.float()
        rotate_half = (
            _rotate_half_neox if self.rotate_style == ROTATE_NEOX else _rotate_half_gptj
        )
        rotary_dim = cos.shape[-1] * (2 if self.reuse_freqs_front_part else 1)

        if self.nope_first:
            d = x.shape[-1]
            x_rot, x_pass = x[..., d - rotary_dim :], x[..., : d - rotary_dim]
        else:
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

        c = cos.float()
        s = sin.float()
        if self.reuse_freqs_front_part:
            if self.rotate_style == ROTATE_NEOX:
                c = c.repeat([1] * (c.dim() - 1) + [2])
                s = s.repeat([1] * (s.dim() - 1) + [2])
            else:
                c = c.repeat_interleave(2, dim=-1)
                s = s.repeat_interleave(2, dim=-1)

        x_embed = (x_rot * c) + (rotate_half(x_rot) * s)
        if self.nope_first:
            out = torch.cat((x_pass, x_embed), dim=-1)
        else:
            out = torch.cat((x_embed, x_pass), dim=-1)
        return out.to(input.dtype)


def _build_cos_sin(s, rotary_dim, reuse_freqs_front_part=True, device="cpu"):
    """Cached cos/sin tables of shape [s, 1, 1, rotary_dim / ratio] (bf16).

    The per-position frequencies are the standard 1/10000^(2k/rotary_dim) RoPE
    schedule; cos/sin are stored in bf16 to mirror the runtime cache dtype.
    """
    ratio = 2 if reuse_freqs_front_part else 1
    half = rotary_dim // ratio
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(s, device=device).float()
    freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [s, half]
    cos = freqs.cos().to(torch.bfloat16).view(s, 1, 1, half).contiguous()
    sin = freqs.sin().to(torch.bfloat16).view(s, 1, 1, half).contiguous()
    return cos, sin


def get_inputs():
    # Representative sbhd shape: s=2048 seq, b=2 batch, h=8 heads, d=128 head_dim.
    # Primary variant: NEOX, reuse_freqs_front_part=True, full rotary (rotary==d).
    # CPU tensors (KernelBench convention); the consumer/harness relocates them.
    s, b, h, d = 2048, 2, 8, 128
    torch.manual_seed(0)
    input = torch.randn(s, b, h, d, dtype=torch.bfloat16)
    cos, sin = _build_cos_sin(s, rotary_dim=d, reuse_freqs_front_part=True)
    return [input, cos, sin]


def get_init_inputs():
    # Flat positional args for Model(rotate_style, reuse_freqs_front_part,
    # nope_first): NEOX (0), reuse front part True, nope_first False.
    return [0, True, False]
