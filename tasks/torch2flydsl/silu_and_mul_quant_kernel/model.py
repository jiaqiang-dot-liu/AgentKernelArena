# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for fused ``silu_and_mul`` + per-group FP8 quant.

The op consumes a row-major ``[m, 2 * d]`` tensor whose last dimension is the
concatenation of a gate half ``x`` and an up-projection half ``y``, computes
``act = silu(x) * y`` (``[m, d]``), then dynamically quantizes ``act`` to FP8
(E4M3) with a per-group amax scale along the last dim, matching the AMD runtime
op ``aiter.silu_and_mul_quant``:

  act[i, :]              = silu(x[i, :]) * y[i, :]                  (fp32)
  scale[i, g]            = max(amax(|act_group[i, g]|), 1e-10) / dtype_max
  out[i, g, :]           = round_to_fp8(act_group[i, g] / scale[i, g])

When ``limit > 0`` the GPT-OSS clamp is applied (``x`` upper-clamped to
``limit``, ``y`` clamped to ``[-limit, limit]``) before the activation. The SiLU
and gate multiply are evaluated in fp32 and the per-group FP8 quant uses an
arch-selected FP8 type (``aiter/utility/dtypes.py``): gfx942/CDNA3 uses
``float8_e4m3fnuz`` (finite max 240), gfx950/CDNA4 uses ``float8_e4m3fn``
(finite max 448), round-to-nearest-even; the floor 1e-10 keeps an all-zero
group's scale finite.

forward(input) -> (output_fp8, scale_fp32)
  input  : [m, 2 * d]            bf16
  output : [m, d]                fp8 e4m3
  scale  : [m, d // group_size]  fp32
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Model(nn.Module):
    """silu_and_mul + per-group FP8 quant. ``Model(group_size, limit)``."""

    def __init__(self, group_size=128, limit=0.0):
        super().__init__()
        self.group_size = int(group_size)
        self.limit = float(limit)
        self.dtype_max = float(torch.finfo(_FP8_DTYPE).max)

    def forward(self, input):
        d = input.shape[-1] // 2
        x, y = input.split([d, d], dim=-1)
        gate = x.float()
        up = y.float()
        if self.limit > 0.0:
            gate = torch.clamp(gate, max=self.limit).to(torch.bfloat16).float()
            up = torch.clamp(up, min=-self.limit, max=self.limit)
        act = F.silu(gate) * up

        m = act.shape[0]
        gs = self.group_size
        ng = d // gs
        ag = act.view(m, ng, gs)
        amax = ag.abs().amax(dim=-1, keepdim=True)
        amax = torch.maximum(amax, torch.full_like(amax, 1e-10))
        scale = amax / self.dtype_max
        out = (ag / scale).to(_FP8_DTYPE).view(m, d)
        return out, scale.view(m, ng).to(torch.float32)


def get_inputs():
    m, n = 512, 8192
    torch.manual_seed(0)
    return [torch.randn(m, n, dtype=torch.bfloat16)]


def get_init_inputs():
    # Flat positional args for Model(group_size, limit).
    return [128, 0.0]
