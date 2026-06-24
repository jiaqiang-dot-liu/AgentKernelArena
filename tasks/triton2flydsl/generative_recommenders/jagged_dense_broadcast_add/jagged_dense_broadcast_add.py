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
"""Standalone jagged + dense broadcast add (Triton).

Provenance: Meta generative-recommenders,
generative_recommenders/ops/triton/triton_jagged.py
(https://github.com/meta-recsys/generative-recommenders).

STANDALONE, FORWARD-only extraction depending only on `triton`/`torch`. Computes
Out = Jagged + Dense, where Jagged is [sum_B(N_i), D] (variable-length per batch,
indexed via the `seq_offsets` [B + 1] prefix-sum), Dense is the per-batch
broadcast operand [B, D], and Out is [sum_B(N_i), D]. This is the additive
sibling of the already-built `jagged_dense_bmm_broadcast_add` task (same file).

Only the portable `@triton.jit jagged_dense_broadcast_add_kernel` and a host
forward wrapper are ported. The `generative_recommenders.common` deps
(`triton_autotune`, `autotune_max_seq_len`, `switch_to_contiguous_if_needed`) are
inlined. The autograd backward (`jagged_reduce_sum` + the bwd of
`_JaggedDenseBroadcastAddFunction`) is dropped -- this extraction is forward-only.
"""

from typing import List

import torch

import triton
import triton.language as tl


# --- inlined switch_to_contiguous_if_needed (generative_recommenders.common) ---
def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# --- inlined power-of-2 / autotune-key helpers (generative_recommenders.common) ---
def next_power_of_2(n: int) -> int:
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


def _get_jagged_dense_broadcast_add_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [16, 32, 64]:
        for num_stages in [1, 2]:
            for num_warps in [2, 4, 8]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_N": BLOCK_N,
                        },
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )
    return configs


@triton_autotune(
    configs=_get_jagged_dense_broadcast_add_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def jagged_dense_broadcast_add_kernel(
    seq_offsets,
    Jagged,
    Dense,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    D,
    stride_jn,
    stride_db,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Computing Out = Jagged + Dense
    JaggedA has shape (sum_B(N_i), D), Dense has shape (B, D), and Out has shape (sum_B(N_i), D)
    """

    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    Jagged += seq_start * stride_jn
    Dense += off_b * stride_db
    Out += seq_start * stride_on
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    jagged_ptrs = Jagged + offs_n[:, None] * stride_jn + offs_d[None, :]
    dense_ptrs = Dense + offs_d
    out_ptrs = Out + offs_n[:, None] * stride_jn + offs_d[None, :]
    for d in range(0, D, BLOCK_D):
        jg = tl.load(
            jagged_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_n[:, None] < seq_len) & ((d + offs_d)[None, :] < D),
        )
        dn = tl.load(dense_ptrs, mask=d + offs_d < D)
        out = jg + dn[None, :]
        tl.store(
            out_ptrs,
            out,
            mask=(offs_n[:, None] < seq_len) & ((d + offs_d)[None, :] < D),
        )
        dense_ptrs += BLOCK_D
        jagged_ptrs += BLOCK_D
        out_ptrs += BLOCK_D


def triton_jagged_dense_broadcast_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    """
    Computing Out = Jagged + Dense
    Jagged has shape (sum_B(N_i), D), Dense has shape (B, D), Out has shape (sum_B(N_i), D).

    Forward-only: the autograd Function (`_JaggedDenseBroadcastAddFunction`) and its
    backward kernel (`jagged_reduce_sum`) from the source are dropped.
    """
    jagged = switch_to_contiguous_if_needed(jagged)
    dense = switch_to_contiguous_if_needed(dense)
    L, D = jagged.shape
    B, _ = dense.shape
    out = torch.empty_like(jagged)

    grid = lambda meta: (  # noqa E731
        B,
        triton.cdiv(max_seq_len, meta["BLOCK_N"]),
    )
    BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
    jagged_dense_broadcast_add_kernel[grid](
        seq_offsets=seq_offsets,
        Jagged=jagged,
        Dense=dense,
        Out=out,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        D=D,
        stride_jn=jagged.stride(0),
        stride_db=dense.stride(0),
        stride_on=out.stride(0),
        BLOCK_D=BLOCK_D,
    )
    return out
