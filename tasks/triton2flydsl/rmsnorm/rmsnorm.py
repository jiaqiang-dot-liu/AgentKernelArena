# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
"""Minimal, self-contained Triton RMSNorm forward — the source kernel to rewrite.

    RMSNorm(x)[i] = x[i] / sqrt(mean(x^2, dim=-1) + eps) * weight[i]

Row-wise: one program per row, fp32 accumulation of the sum of squares (the
numerically important part — accumulating in fp16 overflows for large inputs).
This is the READ-ONLY reference the FlyDSL port must match; it is also the live
correctness oracle + performance baseline used by the rewrite driver.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr, w_ptr, y_ptr,
    x_row_stride, y_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / n_cols
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
    y = x * inv_rms * w

    tl.store(y_ptr + row * y_row_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x, weight, eps: float = 1e-5):
    """RMSNorm forward. x: (M, N); weight: (N,). Returns y with x's shape/dtype."""
    assert x.is_cuda and weight.is_cuda, "inputs must be on CUDA/HIP device"
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    rms_norm_kernel[(M,)](
        x, weight, y,
        x.stride(0), y.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
