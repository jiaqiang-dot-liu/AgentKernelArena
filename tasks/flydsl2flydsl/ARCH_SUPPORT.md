# FlyDSL `flydsl2flydsl` — Architecture Support Matrix

**Pinned FlyDSL version:** `v0.2.0` (commit `28a18d328b4882c999864b2df2f8f9fe3fcc8b47`)
**Benchmark hardware:** AMD MI300X = **gfx942 (CDNA3)**
**Container image:** `flydsl-v0.2.0-rocm7.2.4.sqsh`

All kernel sources in this task suite are pinned to FlyDSL **v0.2.0**. This suite
contains the gfx942-relevant subset of the v0.2.0 `kernels/`. Kernels that target
other architectures (RDNA4/gfx1250, RDNA/RDNA3) are not runnable on MI300X and
have been removed from this suite; they are preserved unchanged on the
`flydsl2flydsl-skip-tasks-parked` branch for a later pass. The two CDNA4/gfx950
FP8 GEMMs remain here, catalogued as `status: skip`.

Each task's `config.yaml` carries a machine-readable `platform_support` block:

```yaml
platform_support:
  required_arch: gfx942        # or gfx950
  runnable_on_gfx942: true     # false => catalogued only, not benchmarked
  status: active               # or skip
  skip_reason: ...             # present when status: skip
```

---

## ✅ Active on MI300X (gfx942) — benchmarked by default (12)

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

## 🟠 Runnable on gfx942 with external runtime — skipped by default (1)

`pa_decode_fp8_kernel` targets gfx942 and passes compile/correctness/performance
in an `aiter`-enabled runtime, but it imports the external AMD `aiter` package
for fp8 KV quantization and paged-attention metadata/reduce helpers. The default
FlyDSL validation image does not ship that dependency, so the task is
`status: skip` and excluded from the default gate.

| Task | Source (`kernels/…`) | Why skipped by default |
|------|----------------------|------------------------|
| `pa_decode_fp8_kernel` | pa_decode_fp8.py | Requires external `aiter` runtime; self-containment check intentionally fails without it |

## 🔴 NOT runnable on MI300X (gfx942) — catalogued, skipped (2)

Present in this suite as `status: skip`. Require CDNA4/gfx950.

### Requires CDNA4 (gfx950)

These two FP8 GEMMs are ported from the HipKittens CDNA4 kernels and emit the
**CDNA4-only 16-byte `buffer_load_lds` intrinsic** (global→LDS direct DMA). The
gfx942 (CDNA3) LLVM backend cannot legalize that operand and aborts at codegen
with `LLVM ERROR: Do not know how to expand this operator's operand!` (process
exits 134). `compile_command` passes because it does not trigger full codegen;
the crash surfaces during `--correctness`. Their `config.yaml` is therefore
`required_arch: gfx950`, `runnable_on_gfx942: false`, `status: skip`.

| Task | Source (`kernels/…`) | Requires | Why not gfx942 |
|------|----------------------|----------|----------------|
| `fp8_gemm_4wave_kernel` | fp8_gemm_4wave.py | gfx950 | 16B `buffer_load_lds` (CDNA4-only); CDNA3 backend cannot legalize → LLVM codegen abort |
| `fp8_gemm_8wave_kernel` | fp8_gemm_8wave.py | gfx950 | 16B `buffer_load_lds` (CDNA4-only); CDNA3 backend cannot legalize → LLVM codegen abort |

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
  on gfx942 (see the gfx950 section above). They are gfx950-only.
