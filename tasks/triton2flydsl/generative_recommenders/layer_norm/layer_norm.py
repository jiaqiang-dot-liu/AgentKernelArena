# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-ignore-all-errors
"""Standalone multi-row LayerNorm forward (Triton).

Provenance: Meta generative-recommenders,
generative_recommenders/ops/triton/triton_layer_norm.py
(https://github.com/meta-recsys/generative-recommenders).

STANDALONE, FORWARD-only extraction depending only on `triton`/`torch`. Ports the
two multi-row LayerNorm forward kernels and a host wrapper:
  - `_layer_norm_fwd`          : unweighted layernorm (mean-subtract + rstd-scale),
  - `_weighted_layer_norm_fwd` : affine layernorm y = (x - mean) * rstd * w + b.
Each program normalizes BLOCK_N rows of a [N, D] activation over the feature dim D
(<= 64KB / element), computing mean and rstd in fp32 via `tl.make_block_ptr`.

The autograd `*Function` classes, all backward kernels (`_layer_norm_bwd_dx`,
`_weighted_layer_norm_bwd_dx`, `_layer_norm_bwd_dwdb`), the RMSNorm/Swish paths,
and the `maybe_register_custom_op` wrapper are dropped. The
`generative_recommenders.common` / `ops.utils` deps (`triton_autotune`,
`switch_to_contiguous_if_needed`, and the host GPU-capability gate) are inlined;
the gate resolves to the CDNA/AMD autotune configs on this target.
"""

from typing import List, Optional, Tuple

import torch

import triton
import triton.language as tl

try:
    # @manual=//triton:triton
    from triton.language.extra.libdevice import rsqrt as libdevice_rsqrt
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.cuda.libdevice import rsqrt as libdevice_rsqrt
    except ImportError:
        # pyre-ignore: Undefined import [21]
        # @manual=//triton:triton
        from triton.language.math import rsqrt as libdevice_rsqrt


# --- inlined switch_to_contiguous_if_needed (generative_recommenders.common) ---
def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# --- inlined triton_autotune (generative_recommenders.common fallback) ---
def triton_autotune(
    configs: List[triton.Config],
    key: List[str],
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
):
    kwargs = {}
    if prune_configs_by is not None:
        kwargs["prune_configs_by"] = prune_configs_by
    if reset_to_zero is not None:
        kwargs["reset_to_zero"] = reset_to_zero
    if restore_value is not None:
        kwargs["restore_value"] = restore_value
    return triton.autotune(configs=configs, key=key, **kwargs)


def _get_layer_norm_fwd_configs() -> List[triton.Config]:
    """Generate autotune configs for multi-row LayerNorm kernels."""
    configs = []
    block_ns = [1, 2, 4, 8]  # CDNA/AMD autotune block counts
    for BLOCK_N in block_ns:
        for num_warps in [1, 2, 4, 8]:
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


@triton_autotune(
    configs=_get_layer_norm_fwd_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def _layer_norm_fwd(
    X,
    Y,
    Mean,
    Rstd,
    N,
    D,
    eps,
    stride_x,
    stride_y,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N, D),
        strides=(stride_x, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    Y_block_ptr = tl.make_block_ptr(
        base=Y,
        shape=(N, D),
        strides=(stride_y, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x_block, axis=1) / D
        if TRAINING:
            tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
    else:
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)

    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)

    if COMPUTE_MEAN_AND_RSTD:
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = 1 / tl.sqrt(var + eps)
        if TRAINING:
            tl.store(Rstd + rows, rstd, row_mask)
    else:
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd

    tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


@triton_autotune(
    configs=_get_layer_norm_fwd_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def _weighted_layer_norm_fwd(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N,
    D,
    eps,
    stride_x,
    stride_y,
    IS_SWISH: tl.constexpr,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
):
    # Get the block ID and calculate starting row
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    # Load weight and bias once (shared across all rows in this block)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)

    # Create block pointers for X and Y
    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N, D),
        strides=(stride_x, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    Y_block_ptr = tl.make_block_ptr(
        base=Y,
        shape=(N, D),
        strides=(stride_y, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )

    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x_block, axis=1) / D
        if TRAINING:
            tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
    else:
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)

    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)

    if COMPUTE_MEAN_AND_RSTD:
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = libdevice_rsqrt(var + eps)
        if TRAINING:
            tl.store(Rstd + rows, rstd, row_mask)
    else:
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    y = y * w[None, :] + b[None, :]

    if IS_SWISH:
        y = tl.sigmoid(y) * x_block

    tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


def triton_weighted_layer_norm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward multi-row LayerNorm.

    When `weight`/`bias` are provided, computes the affine
    y = (x - mean) / sqrt(var + eps) * weight + bias; otherwise the unweighted
    y = (x - mean) / sqrt(var + eps). Returns (y, mean, rstd).
    """
    assert x.dim() == 2, f"x.dim() == {x.dim()}, expected 2"
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    learnable = weight is not None
    if learnable:
        assert bias is not None and weight is not None
        assert weight.dim() == 1
        assert bias.dim() == 1
        assert weight.numel() == D
        assert bias.numel() == D

    y = torch.empty_like(x)
    out_mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    out_rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if N == 0:
        return y, out_mean, out_rstd

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
    if learnable:
        _weighted_layer_norm_fwd[grid](
            x,
            y,
            weight,
            bias,
            out_mean,
            out_rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            IS_SWISH=False,
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=True,
        )
    else:
        _layer_norm_fwd[grid](
            x,
            y,
            out_mean,
            out_rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=True,
        )

    return y, out_mean, out_rstd


def triton_layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Forward-only public entry: returns the normalized activation y."""
    y, _, _ = triton_weighted_layer_norm_fwd(x, weight, bias, eps)
    return y
