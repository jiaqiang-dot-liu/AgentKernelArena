# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""PyTorch reference for 2D-image RoPE forward (NEOX, bf16).

Rotary Position Embedding for a 2D image token grid, matching the AMD runtime
op ``aiter.rope_2d_fwd``. The input is ``[b, H * W, h, d]`` (sequence dim is the
flattened height x width grid). The head dim ``d`` is split in half: the first
half is rotated with the per-row (height) cos/sin tables and the second half
with the per-column (width) cos/sin tables:

  x = x.view(b, H, W, h, d); x1, x2 = x.chunk(2, dim=-1)
  x1 = x1 * cos_h + rotate_half(x1) * sin_h     (height rotary, broadcast over W)
  x2 = x2 * cos_w + rotate_half(x2) * sin_w     (width  rotary, broadcast over H)
  out = concat(x1, x2).view(b, H * W, h, d)

The rotation uses the NEOX rotate style (rotates the second half of the d/2 slice)
and is evaluated in fp32, truncated to bf16 on store. cos_h/sin_h are indexed by
height position and cos_w/sin_w by width position; each cache holds ``d / 2``
entries per position (no reuse-front-part). Mirrors the op_test
``ref_rope_2d_fwd(..., RotateStyle.NEOX)`` against which the runtime op is
validated (reuse_freqs_front_part=False, nope_first=False).

forward(input, cos_h, sin_h, cos_w, sin_w) -> output
  input             : [b, H * W, h, d]   bf16
  cos_h / sin_h     : [1, H, 1, d // 2]  bf16   (height cos/sin cache)
  cos_w / sin_w     : [1, W, 1, d // 2]  bf16   (width  cos/sin cache)
  output            : [b, H * W, h, d]   bf16
"""
import torch
import torch.nn as nn


def _rotate_half_neox(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Model(nn.Module):
    """2D-image RoPE (NEOX). ``Model(height, width)``; cos/sin caches at call time."""

    def __init__(self, height=16, width=16):
        super().__init__()
        self.height = int(height)
        self.width = int(width)

    def forward(self, input, cos_h, sin_h, cos_w, sin_w):
        x = input.float()
        b, s, h, d = x.shape
        x = x.view(b, self.height, self.width, h, d)
        x1, x2 = x.chunk(2, dim=-1)

        ch = cos_h.float()[:, : self.height].unsqueeze(2)
        sh = sin_h.float()[:, : self.height].unsqueeze(2)
        x1 = (x1 * ch) + (_rotate_half_neox(x1) * sh)

        cw = cos_w.float()[:, : self.width].unsqueeze(1)
        sw = sin_w.float()[:, : self.width].unsqueeze(1)
        x2 = (x2 * cw) + (_rotate_half_neox(x2) * sw)

        out = torch.cat([x1, x2], dim=-1).view(b, s, h, d)
        return out.to(input.dtype)


def _build_cos_sin_2d(npos, head_dim, device="cpu"):
    """Cos/sin cache of shape [1, npos, 1, head_dim / 2] (bf16).

    Standard 1/10000^(2k/half) RoPE frequency schedule over ``half = head_dim/2``
    pairs; cos/sin are stored in bf16 to mirror the runtime cache dtype.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(npos, device=device).float()
    freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [npos, half]
    cos = freqs.cos().to(torch.bfloat16).view(1, npos, 1, half).contiguous()
    sin = freqs.sin().to(torch.bfloat16).view(1, npos, 1, half).contiguous()
    return cos, sin


def get_inputs():
    # Representative 2D-image grid: b=2, H=W=16 (s=256), h=8 heads, d=128 head_dim.
    # CPU tensors (KernelBench convention); the consumer/harness relocates them.
    b, height, width, h, d = 2, 16, 16, 8, 128
    torch.manual_seed(0)
    input = torch.randn(b, height * width, h, d, dtype=torch.bfloat16)
    cos_h, sin_h = _build_cos_sin_2d(height, d)
    cos_w, sin_w = _build_cos_sin_2d(width, d)
    return [input, cos_h, sin_h, cos_w, sin_w]


def get_init_inputs():
    # Flat positional args for Model(height, width).
    return [16, 16]
