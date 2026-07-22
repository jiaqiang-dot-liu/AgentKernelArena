# FlyDSL (`flydsl2flydsl`) Tasks

## Prerequisites

These tasks require [FlyDSL](https://github.com/ROCm/FlyDSL) and a ROCm-enabled AMD GPU.

Recommended setup from the repository root:

```bash
make docker-setup-flydsl
```

This installs FlyDSL into the Docker runtime's persistent pip user-base when the
selected image does not already provide it, then verifies the import and ROCm
PyTorch GPU availability.

For manual inspection, enter the supported runtime first:

```bash
make docker-shell
python3 -c "import flydsl; print(flydsl.__version__)"
```

## Hardware support

The suite has active tasks for **AMD MI300X = gfx942 (CDNA3)** and
**AMD MI355X = gfx950 (CDNA4)**. Per-task hardware support is machine-readable
in each `config.yaml`; the runner selects the tasks matching the current GPU.
See **[`ARCH_SUPPORT.md`](ARCH_SUPPORT.md)** for the full matrix.

`pa_decode_fp8_kernel` is a normal active task. It uses the framework-provided
`aiter` package for FP8 KV quantization and paged-attention metadata/reduce, just
as tasks may use other runtime dependencies; it has no task-specific dependency
probe or skip condition.

`fp8_gemm_4wave_kernel` and `fp8_gemm_8wave_kernel` are active
**gfx950/CDNA4-only** tasks. They emit the CDNA4-only 16B `buffer_load_lds`
intrinsic, so the runner excludes them on gfx942 through `required_arch: gfx950`.

## Compute pattern (L1 / L2 / L3)

Examples are grouped by **compute pattern** (not by any other “difficulty” scale):

- **L1** — Elementwise or single per-row reduction; threads work independently.
- **L2** — No matrix multiply, but requires cross-thread cooperation via shared
  memory (LDS) or a fused multi-step pass.
- **L3** — Contains a matrix multiply (MFMA): GEMM or attention, with software
  pipelining, double-buffered LDS, split-K, or paged / FP8 KV-cache.

### L1

| Task | Reason |
|------|--------|
| `softmax_kernel` | Numerically stable softmax, register-buffered per-row reduction, exp2 fast path. No matmul, no cross-thread cooperation. |
| `rmsnorm_kernel` | RMSNorm with float32 accumulation. Per-row reduction; the multiple kernels are just dtype variants. |

### L2

| Task | Reason |
|------|--------|
| `layernorm_kernel` | LayerNorm with shared-memory (LDS) reduction and fused `x*scale+bias` epilogue. No matmul. |
| `fused_rope_cache_kernel` | Fused rotary embedding + KV-cache write; cross-lane `ds_bpermute` shuffles, vectorized buffer_load/store. No matmul. |
| `topk_gating_softmax_kernel` | Fused MoE gating softmax + top-K + optional renormalize + token_expert_indices. |
| `moe_sorting_kernel` | MoE token/expert sorting (oneshot + multiphase); CDNA-focused. |
| `silu_and_mul_fq_kernel` | Fused activation (SiLU/SwiGLU) + optional quant + sorted scales for split-K MoE stage-1. |

### L3

| Task | Reason |
|------|--------|
| `flash_attn_func_kernel` | Fused multi-head attention: online softmax, MFMA32 GEMM, DMA-to-LDS, software-pipelined QK/PV. |
| `hgemm_splitk_kernel` | Half-precision GEMM with split-K, double-buffered LDS, pre-shuffled B. |
| `pa_decode_fp8_kernel` | Paged-attention decode with FP8 KV-cache and multi-partition reduce; most complex kernel. Active as a normal framework task. |
| `blockscale_preshuffle_gemm_kernel` | FP8 blockscale GEMM with preshuffled B and MFMA epilogue. |
| `fp8_gemm_4wave_kernel` | Active FP8 GEMM (4-wave) with row scales. **gfx950/CDNA4-only**; selected through `required_arch: gfx950`. |
| `fp8_gemm_8wave_kernel` | Active FP8 GEMM (8-wave) with row scales, ported from HipKittens CDNA4. **gfx950/CDNA4-only**; selected through `required_arch: gfx950`. |
| `preshuffle_gemm_v2_kernel` | Preshuffle GEMM v2 (layout API; fp8/fp16/bf16). |
| `pa_decode_swa_kernel` | Paged-attention decode for sliding-window (partitioned) paths. |

## Vendored FlyDSL helper modules

Each task is fully self-contained: the FlyDSL helper modules it needs are
vendored inside that task's own `kernels/` subfolder and imported via the
`kernels.` path (e.g. `kernels_common.py`, `tensor_shim.py`, `mfma_epilogues.py`,
`fp8_gemm_utils.py`, `layout_utils.py`, `moe_common.py`). There is no shared
top-level `kernels/` folder.

## Benchmark config

The benchmark config (`config_geak_flydsl.yaml`) that lists these tasks (grouped by
the same L1/L2/L3 pattern) lives on the **`geak-flydsl-common-benchmark`** branch.
