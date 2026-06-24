# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for FP8 dynamic per-1x128 (per-group) quantization.

The op splits each row of a ``[m, n]`` activation tensor into contiguous groups
of 128 elements and quantizes each group independently into the FP8 E4M3 range
using a dynamic per-group amax scale. It returns the quantized FP8 values
together with the fp32 scales that dequantize them.

AMD-runtime semantics (option-b): the FP8 type is arch-selected to match the
aiter op (``aiter/utility/dtypes.py``): gfx942/CDNA3 uses ``float8_e4m3fnuz``
(finite max 240) and gfx950/CDNA4 uses OCP ``float8_e4m3fn`` (finite max 448).
For each 1x128 block the scale is ``amax(|x_block|) / dtype_max``; an all-zero
block keeps a scale of 1 to avoid division by zero. Values are rescaled in fp32
and cast to FP8 with round-to-nearest-even, exactly as the AMD runtime per-group
quant op (``get_hip_quant(QuantType.per_1x128)`` -> ``per_group_quant`` with
``group_size=128``) does. The returned scale is fp32 of shape ``[m, n // 128]``.

forward(input) -> (output_fp8, scale_fp32)
  input  : [m, n]          bf16   (n divisible by 128)
  output : [m, n]          fp8 e4m3
  scale  : [m, n // 128]   fp32
"""
import torch
import torch.nn as nn

def _amd_fp8_dtype():
    """fp8 storage dtype the matching aiter op uses on the active GPU arch,
    mirroring ``aiter/utility/dtypes.py``: gfx942/CDNA3 -> ``float8_e4m3fnuz``
    (finite max 240); gfx950/CDNA4 and others -> ``float8_e4m3fn`` (max 448)."""
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        arch = ""
    return torch.float8_e4m3fnuz if arch == "gfx942" else torch.float8_e4m3fn


_FP8_DTYPE = _amd_fp8_dtype()
_GROUP_SIZE = 128


class Model(nn.Module):
    """FP8 dynamic per-1x128 quantizer. ``Model()`` takes no hyperparameters."""

    def __init__(self):
        super().__init__()
        self.dtype_max = float(torch.finfo(_FP8_DTYPE).max)

    def forward(self, input):
        m, n = input.shape
        x = input.float().view(-1, _GROUP_SIZE)
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = amax / self.dtype_max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        y = (x / scale).to(_FP8_DTYPE).view(m, n)
        return y, scale.to(torch.float32).view(m, n // _GROUP_SIZE)


def get_inputs():
    return [torch.randn(128, 4096, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
