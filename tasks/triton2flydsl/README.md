# Triton -> FlyDSL Rewrite (`triton2flydsl`) Tasks

These tasks evaluate an agent's ability to **rewrite an existing Triton kernel
into an equivalent FlyDSL kernel** and then optimize it, rather than optimizing
the kernel in its original language.

FlyDSL is a fine-grained, JIT-compiled MLIR-based DSL for AMD GPUs that patches
directly into inference frameworks without heavyweight prebuilt libraries.

## How it differs from other task types

- Input (`source_file_path`) is the original **Triton** kernel.
- The produced/optimized kernel is in **FlyDSL**.
- The original Triton kernel is used as BOTH the correctness oracle and the
  performance baseline: `speedup = triton_ms / flydsl_ms`. The score answers
  "did rewriting to FlyDSL actually help?".

## Execution path

`triton2flydsl` tasks are driven by KernelForge's `forge-rewrite-by-flydsl` layer
(a thin front-end that ports the kernel to FlyDSL and then reuses `forge-loop` to
optimize it). See the KernelForge `forge-rewrite-by-flydsl` design for the
Arena -> forge-rewrite-by-flydsl -> forge-loop chain.

## Prerequisites

Requires [FlyDSL](https://github.com/ROCm/FlyDSL) and a ROCm-enabled AMD GPU:

```bash
make setup-flydsl
```

## Tasks

| Task | Difficulty | Source | Description |
|------|-----------|--------|-------------|
| `softmax` | Easy | rocmbench Triton `softmax_kernel_online` | Online, numerically stable row-softmax |

## Task config schema (the `rewrite` block)

Beyond the standard task fields, `triton2flydsl` tasks carry a `rewrite` block
consumed by `forge-rewrite-by-flydsl` (target language is always FlyDSL):

```yaml
rewrite:
  op_name: softmax          # FlyDSL module must expose build_<op_name>_module(...)
  source_entry: softmax     # host callable ref(x)->y used as live oracle + baseline (optional)
  shapes:                   # (M, N, dtype) tuples driving correctness + benchmark
    - {M: 8192, N: 8192, dtype: fp16}
```
