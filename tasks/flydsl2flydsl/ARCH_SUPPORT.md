# FlyDSL `flydsl2flydsl` — Architecture Support Matrix

**Pinned FlyDSL version:** `v0.2.0` (commit `28a18d328b4882c999864b2df2f8f9fe3fcc8b47`)
**Benchmark hardware:** AMD MI300X = **gfx942 (CDNA3)** and AMD MI355X =
**gfx950 (CDNA4)**

All kernel sources in this task suite are pinned to FlyDSL **v0.2.0**. This suite
contains the gfx942-relevant subset of the v0.2.0 `kernels/`. Kernels that target
other architectures (RDNA4/gfx1250, RDNA/RDNA3) are not runnable on MI300X and
have been removed from this suite; they are preserved unchanged on the
`flydsl2flydsl-skip-tasks-parked` branch for a later pass. The two CDNA4/gfx950
FP8 GEMMs remain here as active tasks selected only on matching hardware.

Architecture-specific tasks carry a machine-readable `platform_support` block;
tasks without that optional block run normally on the selected platform:

```yaml
platform_support:
  required_arch: gfx942        # or gfx950
  runnable_on_gfx942: true
  status: active
```

---

## ✅ Active on MI300X (gfx942) — benchmarked when gfx942 is selected (13)

| Task | Source (`kernels/…`) | Pattern |
|------|----------------------|---------|
| `softmax_kernel` | softmax_kernel.py | L1 reduction |
| `rmsnorm_kernel` | rmsnorm_kernel.py | L1 reduction |
| `layernorm_kernel` | layernorm_kernel.py | L1 reduction |
| `fused_rope_cache_kernel` | fused_rope_cache_kernel.py | L2 fused |
| `silu_and_mul_fq_kernel` | silu_and_mul_fq.py | L2 fused + quant |
| `topk_gating_softmax_kernel` | topk_gating_softmax_kernel.py | L2 MoE gating |
| `moe_sorting_kernel` | moe_sorting_kernel.py | L2 MoE sort |
| `blockscale_preshuffle_gemm_kernel` | blockscale_preshuffle_gemm.py | L3 GEMM (fp8 blockscale) |
| `preshuffle_gemm_v2_kernel` | preshuffle_gemm_v2.py | L3 GEMM (preshuffle) |
| `hgemm_splitk_kernel` | hgemm_splitk.py | L3 GEMM (split-K) |
| `flash_attn_func_kernel` | flash_attn_func.py | L3 attention |
| `pa_decode_swa_kernel` | pa_decode_swa.py | L3 paged-attn decode (SWA) |
| `pa_decode_fp8_kernel` | pa_decode_fp8.py | L3 paged-attn decode (FP8); normal framework task using the provided `aiter` runtime |

## ✅ Active on MI355X (gfx950) — benchmarked when gfx950 is selected (3)

`pa_decode_fp8_kernel` runs as a normal task on gfx950. The following two GEMMs
are gfx950-only:

These two FP8 GEMMs are ported from the HipKittens CDNA4 kernels and emit the
**CDNA4-only 16-byte `buffer_load_lds` intrinsic** (global→LDS direct DMA). The
gfx942 (CDNA3) LLVM backend cannot legalize that operand and aborts at codegen
with `LLVM ERROR: Do not know how to expand this operator's operand!` (process
exits 134). Their `config.yaml` therefore uses `required_arch: gfx950`,
`runnable_on_gfx942: false`, and `status: active`. The runner selects them on
gfx950 and excludes them on gfx942 before workspace setup.

| Task | Source (`kernels/…`) | Scope |
|------|----------------------|-------|
| `pa_decode_fp8_kernel` | pa_decode_fp8.py | Active without a task-specific architecture or dependency skip |
| `fp8_gemm_4wave_kernel` | fp8_gemm_4wave.py | Active on gfx950; 16B `buffer_load_lds` is CDNA4-only |
| `fp8_gemm_8wave_kernel` | fp8_gemm_8wave.py | Active on gfx950; 16B `buffer_load_lds` is CDNA4-only |

### Parked (RDNA4/gfx1250 or RDNA) — moved out of this PR

The following gfx1250 / RDNA kernels are not runnable on MI300X (gfx942) and have
been removed from this suite. They are preserved unchanged on the
`flydsl2flydsl-skip-tasks-parked` branch (along with the shared top-level
`kernels/` helpers they depend on) and can be revisited in a later pass:

- `gemm_fp8fp4_gfx1250_kernel` (gfx1250 — FP8/FP4 WMMA)
- `wmma_gemm_gfx1250_kernel` (gfx1250 — WMMA matrix ops)
- `moe_gemm_2stage_mxscale_gfx1250_kernel` (gfx1250 — MXFP-scale MoE GEMM)
- `moe_gemm_2stage_wmma_gfx1250_kernel` (gfx1250 — WMMA MoE GEMM)
- `rdna3_f16_gemm_kernel` (rdna3 — RDNA3 WMMA f16 GEMM)
- `rdna_f16_gemm_kernel` (rdna — RDNA-only f16 GEMM)
- `rdna_fp8_preshuffle_gemm_kernel` (rdna — RDNA-only fp8 preshuffle GEMM)

---

## Notes
- Each task is self-contained: the FlyDSL helper modules it needs are vendored
  under that task's own `kernels/` subfolder. There is no shared top-level
  `kernels/` folder in this suite.
- `gfx950` literals appearing alongside `gfx942` are feature-gates (e.g. HW LDS
  transpose, K16 MFMA, 16B LDS DMA). Most kernels fall back to a gfx942 path and
  still run on MI300X — **but `fp8_gemm_4wave` / `fp8_gemm_8wave` do NOT**: they
  unconditionally emit the CDNA4-only 16B `buffer_load_lds` and abort at codegen
  on gfx942 (see the gfx950 section above). They remain active tasks, but are
  selected only on gfx950.
