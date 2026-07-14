# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for MXFP8 (E4M3) per-1x32 dynamic quantization.

The op quantizes a ``[m, k]`` tensor to MXFP8: each contiguous block of 32
elements along the last dim shares one E8M0 (power-of-two) block scale, and the
32 values are stored as FP8 E4M3. The op returns ``(y, scale)`` where ``y`` is
``[m, k]`` ``float8_e4m3fn`` and ``scale`` is ``[m, k // 32]`` E8M0 bytes
(uint8).

AMD-runtime semantics (option-b): this is a bit-faithful port of the
``dynamic_mxfp8_quant`` Triton kernel quant logic. The block scale is derived
by the integer "round up to E8M0-representable power-of-two" sequence
``amax_i32 = (amax_i32 + 0x200000) & 0xFF800000`` then
``scale = floor(log2(amax_p2)) - 8``, clamped to ``[-127, 127]`` and biased by
127; values are divided by ``2**scale`` in fp32 and cast to FP8 E4M3 with
round-to-nearest-even. The E8M0 scale bytes are integer-only after the fp32
cast, so they match the kernel exactly; the FP8 codes match within one ULP of
the fp32->fp8 rounding.
"""
import torch
import torch.nn as nn

_BLOCK = 32
# 0xFF800000 as a signed int32: keeps sign + 8-bit exponent (strips mantissa).
_E8M0_MASK_INT32 = -8388608


class Model(nn.Module):
    """MXFP8 per-1x32 dynamic quantizer. ``Model(group_size=32)``."""

    def __init__(self, group_size=32):
        super().__init__()
        self.group_size = int(group_size)

    def forward(self, input):
        g = self.group_size
        x = input.float()
        M, K = x.shape
        Ng = K // g

        x2 = x.reshape(M, Ng, g)
        amax = x2.abs().amax(dim=-1, keepdim=True)  # (M, Ng, 1)

        amax_i32 = amax.contiguous().view(torch.int32)
        amax_i32 = (amax_i32 + 0x200000) & _E8M0_MASK_INT32
        amax_p2 = amax_i32.view(torch.float32)

        scale_unbiased = amax_p2.log2().floor() - 8
        scale_unbiased = torch.clamp(scale_unbiased, min=-127, max=127)
        scale_e8m0 = (scale_unbiased.to(torch.int32) + 127).to(torch.uint8)
        quant_scale = torch.exp2(-scale_unbiased)

        qx = (x2 * quant_scale).reshape(M, K)
        y = qx.to(torch.float8_e4m3fn)
        s = scale_e8m0.reshape(M, Ng)
        return y, s


def get_inputs():
    return [torch.randn(128, 1024, dtype=torch.bfloat16) * 4.0]


def get_init_inputs():
    return [32]
