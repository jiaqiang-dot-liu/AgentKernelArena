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

# pyre-unsafe
"""Standalone jagged x dense batched matmul with broadcast bias add (Triton).

Provenance: Meta generative-recommenders,
generative_recommenders/ops/triton/triton_jagged.py
(https://raw.githubusercontent.com/meta-recsys/generative-recommenders/main/generative_recommenders/ops/triton/triton_jagged.py).

This is a STANDALONE, FORWARD-only extraction depending only on `triton`/`torch`.
The `generative_recommenders.common` / `ops.utils` deps it needed
(`triton_autotune`, `autotune_max_seq_len`, `fine_grained_autotune_max_seq_len`,
`switch_to_contiguous_if_needed`) are inlined below. The optional
vendor-specific fused-library path (`torch.ops.load_library(...)` branches) is
removed and the portable `@triton.jit` path is kept, which this AMD/CDNA
(gfx950) build always uses. The autograd
backward (`_JaggedDenseBmmAddFunction.backward` and the jagged/dense/bias bwd
kernels) is dropped -- this extraction is forward-only.
"""

from typing import List, Tuple

import torch

import triton
import triton.language as tl


# --- inlined switch_to_contiguous_if_needed (generative_recommenders.common) ---
def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# --- inlined power-of-2 / autotune-key helpers (generative_recommenders.common) ---
def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def prev_power_of_2(x: int) -> int:
    out = next_power_of_2(x)
    return out // 2 if out > x else out


STATIC_MAX_SEQ_LENS: List[int] = []
USE_RUNTIME_MAX_SEQ_LEN: bool = False


def autotune_max_seq_len(runtime_max_seq_len: int) -> int:
    if USE_RUNTIME_MAX_SEQ_LEN:
        return prev_power_of_2(runtime_max_seq_len)
    else:
        if STATIC_MAX_SEQ_LENS == []:
            return 1
        for max_len in STATIC_MAX_SEQ_LENS:
            if max_len >= runtime_max_seq_len:
                return max_len
        return STATIC_MAX_SEQ_LENS[-1]


_FINE_GRAINED_BUCKETS: List[int] = [
    1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768,
    40960, 49152, 65536, 81920, 98304,
]


def fine_grained_autotune_max_seq_len(runtime_max_seq_len: int) -> int:
    if USE_RUNTIME_MAX_SEQ_LEN:
        for bucket in _FINE_GRAINED_BUCKETS:
            if runtime_max_seq_len <= bucket:
                return bucket
        return _FINE_GRAINED_BUCKETS[-1]
    else:
        if STATIC_MAX_SEQ_LENS == []:
            return 1
        for max_len in STATIC_MAX_SEQ_LENS:
            if max_len >= runtime_max_seq_len:
                return max_len
        return STATIC_MAX_SEQ_LENS[-1]


# --- inlined triton_autotune (generative_recommenders.common fallback) ---
# Thin wrapper over the public triton.autotune decorator (the upstream fallback
# constructs triton.runtime.autotuner.Autotuner directly; the public decorator
# is equivalent and stable across Triton/ROCm versions).
def triton_autotune(
    configs: List[triton.Config],
    key: List[str],
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    warmup: int = 25,
    rep: int = 100,
):
    kwargs = {}
    if prune_configs_by is not None:
        kwargs["prune_configs_by"] = prune_configs_by
    if reset_to_zero is not None:
        kwargs["reset_to_zero"] = reset_to_zero
    if restore_value is not None:
        kwargs["restore_value"] = restore_value
    return triton.autotune(configs=configs, key=key, **kwargs)


def _get_bmm_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [64, 128]:
        for BLOCK_N in [64, 128, 256]:
            for BLOCK_K in [32, 64]:
                for num_stages in [3, 5]:
                    for num_warps in [4, 8]:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": BLOCK_M,
                                    "BLOCK_N": BLOCK_N,
                                    "BLOCK_K": BLOCK_K,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


@triton_autotune(
    configs=_get_bmm_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "N", "K", "ELEMENTWISE", "HAS_BIAS"],
)
@triton.jit
def jagged_dense_bmm_broadcast_add_kernel(
    seq_offsets,
    Jagged,
    Dense,
    Bias,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    N,
    K,
    stride_jm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_bias_b,
    stride_om,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ELEMENTWISE: tl.constexpr,
):
    """
    Computing bmm Out = Jagged x Dense + Bias
    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), Bias has shape (B, N), and Out has shape (sum_B(M_i), N)
    """

    off_n = tl.program_id(0)
    off_m = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Jagged += (seq_start + start_m) * stride_jm
    Dense += off_b.to(tl.int64) * stride_db
    Out += seq_start * stride_om

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < (seq_len - start_m)) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    if HAS_BIAS:
        if ELEMENTWISE:
            Bias += (seq_start + start_m) * stride_bias_b
            bias_ptrs = Bias + offs_m[:, None] * stride_bias_b + offs_n[None, :]
            bias = tl.load(
                bias_ptrs,
                mask=(offs_m[:, None] < (seq_len - start_m)) & (offs_n[None, :] < N),
                other=0.0,
            )
            accumulator += bias.to(tl.float32)
        else:
            bias_ptrs = Bias + off_b.to(tl.int64) * stride_bias_b + offs_n
            bias = tl.load(bias_ptrs, mask=offs_n < N)
            accumulator += bias[None, :].to(tl.float32)

    out = accumulator.to(Out.dtype.element_ty)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    Out += start_m * stride_om
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < (seq_len - start_m)) & (offs_n[None, :] < N),
    )


def triton_jagged_dense_bmm_add_fwd(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
    elementwise: bool = False,
) -> Tuple[torch.Tensor, int, int, int]:
    jagged = switch_to_contiguous_if_needed(jagged)
    bias = switch_to_contiguous_if_needed(bias)
    L, K = jagged.shape
    B, _, N = dense.shape
    out = torch.empty((L, N), dtype=jagged.dtype, device=jagged.device)

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_N"]),
        triton.cdiv(max_seq_len, meta["BLOCK_M"]),
        B,
    )

    jagged_dense_bmm_broadcast_add_kernel[grid](
        seq_offsets=seq_offsets,
        Jagged=jagged,
        Dense=dense,
        Bias=bias,
        Out=out,
        AUTOTUNE_MAX_SEQ_LEN=fine_grained_autotune_max_seq_len(max_seq_len),
        N=N,
        K=K,
        stride_jm=jagged.stride(0),
        stride_db=dense.stride(0),
        stride_dk=dense.stride(1),
        stride_dn=dense.stride(2),
        stride_bias_b=bias.stride(0),
        stride_om=out.stride(0),
        HAS_BIAS=True,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        ELEMENTWISE=elementwise,
    )

    return out, B, K, N


def triton_jagged_dense_bmm_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
    elementwise: bool = False,
) -> torch.Tensor:
    """
    Computing bmm Out = Jagged x Dense + Bias
    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), Bias has shape (B, N) or (sum_B(M_i), N) depending on Elementwise, and Out has shape (sum_B(M_i), N)
    """
    # Forward-only: the autograd Function (`_JaggedDenseBmmAddFunction`) and its
    # backward kernels from the source are dropped; this calls the Triton fwd.
    out, _, _, _ = triton_jagged_dense_bmm_add_fwd(
        max_seq_len, seq_offsets, jagged, dense, bias, elementwise
    )
    return out
