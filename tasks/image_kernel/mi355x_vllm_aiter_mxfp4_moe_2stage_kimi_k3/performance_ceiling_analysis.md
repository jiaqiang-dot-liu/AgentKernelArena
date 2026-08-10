# Performance ceiling analysis

This analysis follows
[`performance-ceiling-methodology.md`](../../../docs/reference/performance-ceiling-methodology.md)
and [`benchmark-methodology.md`](../../../docs/reference/benchmark-methodology.md).

## Result

- Bound classification: `mixed-bound`.
- Mean ideal case latency: `0.278601693 ms`.
- Measured task mean latency: `1.052133249 ms`.
- Measured efficiency relative to the ideal model: `26.48%`.

The prefill case is compute-bound by BF16 matrix work. The decode case is
memory-bound by the FP4 weights and E8M0 scales of the selected experts.

## Operator and scored workload

The timed operator is one `fused_moe` call with two logically serial GEMMs:

1. Stage 1 computes gate and up projections, then applies SiTUv2.
2. Stage 2 computes the down projection, applies route weights, and reduces the
   16 routes of each token.

The common per-rank dimensions are:

- model dimension \(H=3584\);
- intermediate dimension \(I=384\);
- experts \(E=896\);
- top-k \(K_t=16\);
- BF16 activations and outputs;
- packed MXFP4 weights with one E8M0 scale byte per 32 weight values.

The two scored cases are:

- `kimi-k3-prefill-flydsl-k003-k006`: \(T=7211\), \(R=T K_t=115376\);
- `kimi-k3-decode-graph-k001-k002`: \(T=62\), \(R=992\).

The deterministic inputs selected 896 and 619 active experts, respectively.

## Measurement environment

- GPU: AMD Instinct MI355X/gfx950, 256 CUs.
- Runtime: ROCm 7.2.3, PyTorch 2.11.0 development build.
- Profiler: `rocprofv3 1.1.0`.
- HBM ceiling: `8.0 TB/s`.
- BF16 matrix ceiling: `2.5166 PFLOP/s`.
- FP32 vector ceiling: `157.2864 TFLOP/s`.

The advertised `10.0663 PFLOP/s` MXFP4 matrix peak is not used. The A16W4
FlyDSL path calls `unpack_b_mxfp4_bf16` and issues
`mfma_f32_16x16x32_bf16`; the matrix service rate is therefore the BF16 peak.

A one-workgroup FP32 fill was warmed 10 times and traced for 50 further
dispatches through the same PyTorch/HIP stack. The minimum stable device
duration was `1.76 us`, which is used as the single-dispatch floor.

## Correctness and uninstrumented benchmark

All 14 M-bucket correctness cases passed. For the scored cases:

- prefill: cosine `0.999927`, relative norm error `0.0121`;
- decode: cosine `0.999962`, relative norm error `0.0088`.

Uninstrumented CUDA Graph timing reported:

- `kimi-k3-prefill-flydsl-k003-k006`: `1.829492599 ms`;
- `kimi-k3-decode-graph-k001-k002`: `0.274773900 ms`.

Their arithmetic mean is `1.052133249 ms`.

## Profiling evidence

`rocprofv3` observed six device dispatches per prefill invocation: four MoE
sorting kernels followed by `moe_gemm1_0` and `moe_gemm2_0`. Mean target
durations were:

- sorting kernels in total: `36.314 us`;
- stage 1: `1093.584 us`;
- stage 2: `729.280 us`.

Decode used two sorting kernels and the same two logical GEMM stages:

- sorting kernels in total: `10.740 us`;
- stage 1: `161.702 us`;
- stage 2: `84.141 us`.

Counter collection produced:

- prefill stage 1: `32.204%` mean `MfmaUtil`, `0.022%`
  `MemUnitStalled`, `23.344%` occupancy;
- prefill stage 2: `23.692%` mean `MfmaUtil`, `0.449%`
  `MemUnitStalled`, `36.662%` occupancy;
- decode stage 1: `28.276%` mean `MfmaUtil`, `0.017%`
  `MemUnitStalled`, `31.680%` occupancy;
- decode stage 2: `25.439%` mean `MfmaUtil`, `0.141%`
  `MemUnitStalled`, `39.801%` occupancy.

The large prefill case spends almost all traced time in the BF16 MFMA stages.
The decode GEMMs are close to the service time of more than 1.3 GB of selected
expert weights and scales.

## Ideal execution model

The ideal implementation uses one token/expert-centric dispatch. Sorting is not
a semantic output and may be folded into route processing. Stage 1 intermediates
may remain on chip and stage 2 may directly accumulate route-weighted output,
so the model does not charge global intermediate traffic or a second dispatch.

This is an optimistic legal fusion, not a description of the current FlyDSL
implementation.

### Arithmetic

For \(R=T K_t\), the two matrix stages require:

\[
F_{\mathrm{matrix}} =
4RHI + 2RHI = 6RHI
\]

SiTUv2, route weighting, and final top-k reduction are represented by:

\[
F_{\mathrm{vector,proxy}} = 10RI + (R-T)H
\]

The \(10RI\) term counts the ordinary SiTUv2 work, three SFU evaluations per
intermediate element as one proxy operation each, and one pre-stage-2 route
weight multiplication. Weighting before the linear down projection is
algebraically equivalent and cheaper than weighting its \(H\)-wide output.
The remaining \((R-T)H\) additions reduce 16 routes per token. Tanh and sigmoid
are SFU operations, so their modeled contribution remains a proxy; a materially
slower SFU implementation would require recalibration.

The compute service time is:

\[
T_{\mathrm{compute}} =
\frac{F_{\mathrm{matrix}}}{2.5166\times10^{15}}
+
\frac{F_{\mathrm{vector,proxy}}}{157.2864\times10^{12}}
\]

### Semantic traffic

Let \(A\) be the number of active experts. The minimum weight traffic is:

\[
B_{\mathrm{weights}} =
A H I
+\left(\frac{A H I}{16}\right)
+\left(\frac{A H I}{2}\right)
+\left(\frac{A H I}{32}\right)
= \frac{51}{32} AHI
\]

The terms are stage-1 packed FP4 weights, stage-1 E8M0 scales, stage-2 packed
FP4 weights, and stage-2 E8M0 scales. Only semantic scale groups are counted;
physical padding in the persisted W2 scale layout is excluded.

Including BF16 input and output plus FP32 route IDs and weights:

\[
B = 4TH + \frac{51}{32}AHI + 8R
\]

The ideal lower bound is:

\[
T_{\mathrm{ideal}} =
0.00176\ \mathrm{ms}
+10^3\max\left(
T_{\mathrm{compute}},
\frac{B}{8.0\times10^{12}}
\right)\ \mathrm{ms}
\]

## Per-case calculation

### Prefill

- \(R=115376\), \(A=896\).
- Matrix work: `952,721,473,536` operations.
- Vector/SFU proxy: `830,707,200` operations.
- Semantic traffic: `2,069,593,472` bytes.
- Matrix service: `0.378574852 ms`.
- Vector/SFU proxy service: `0.005281494 ms`.
- Memory service: `0.258699184 ms`.
- Ideal latency: `0.385616347 ms`.
- Classification: `compute-bound`.
- Measured efficiency: `21.08%`.

### Decode

- \(R=992\), \(A=619\).
- Matrix work: `8,191,475,712` operations.
- Vector/SFU proxy: `7,142,400` operations.
- Semantic traffic: `1,358,616,320` bytes.
- Matrix service: `0.003254977 ms`.
- Vector/SFU proxy service: `0.000045410 ms`.
- Memory service: `0.169827040 ms`.
- Ideal latency: `0.171587040 ms`.
- Classification: `memory-bound`.
- Measured efficiency: `62.45%`.

## Interpretation and validity

The decode case is already much closer to its semantic traffic floor than the
prefill case. Prefill has room in MFMA scheduling, stage fusion, sorting
elimination, and intermediate traffic.

The model assumes each active expert's required weight and scale data is fetched
from HBM once. A realizable single-dispatch kernel may trade this reuse against
parallelism and on-chip capacity, so the value is deliberately optimistic.
Recompute the ceiling if the scored cases, active-expert distribution, timing
boundary, A16W4 instruction path, benchmark method, or GPU stack changes.
