# Performance ceiling analysis

This analysis applies the
[performance ceiling methodology](../../../docs/reference/performance-ceiling-methodology.md)
to `dynamic_per_tensor_quant` on MI355X/gfx950.

## Result

- Bound classification: `latency-bound`
- Mean ideal case latency: `0.001105537 ms`
- Measured task mean latency: `0.006058409 ms`
- Measured efficiency relative to the ideal model: `18.25%`

The ideal value is a lower-bound model for a legal fused implementation. It is
not an observed best-known implementation.

## Operator and workload

The operator converts a BF16 input tensor to FP8 and writes one FP32 tensor-wide
scale. The scored cases are:

- `mixtral-k006`: shape `(64, 4096)`, 262,144 elements.
- `mixtral-k002`: shape `(64, 512)`, 32,768 elements.
- `mixtral-k004`: shape `(64, 2, 1792)`, 229,376 elements.

The current AITER implementation launches three serialized kernels:

1. `initializeScale` clears the output scale.
2. `data_to_scale_kernel` reads the BF16 input and reduces its absolute maximum.
3. `scaled_quant_kernel` reads the BF16 input again and writes FP8 output.

CUDA Graph timing amortizes host submission overhead, but it does not fuse these
three device kernel nodes.

## Measurement environment

- GPU architecture: MI355X/gfx950.
- Compute units: 256 across 8 XCCs.
- HBM3E bandwidth used by the ideal model: 8.0 TB/s.
- Benchmark method: task CUDA Graph timing.
- Profiler: `rocprofv3`.
- Timing boundary: one end-to-end `dynamic_per_tensor_quant` invocation.

The module was built and warmed before profiling. Kernel trace and hardware
counters were collected in separate profiler passes so incompatible counter
groups did not share one collection.

## Uninstrumented benchmark

The task benchmark reported:

- `mixtral-k006`: `0.005704542 ms`.
- `mixtral-k002`: `0.005842532 ms`.
- `mixtral-k004`: `0.006628153 ms`.

Their arithmetic mean is:

\[
\frac{0.005704542 + 0.005842532 + 0.006628153}{3}
= 0.006058409\ \mathrm{ms}
\]

All three correctness cases passed before the timings were accepted.

## Kernel trace

`rocprofv3` observed eight invocations of each current stage per case. The
average device kernel durations were:

- `mixtral-k006`: `1.325 us` initialization, `2.230 us` scale reduction, and
  `2.070 us` quantization; stage sum `5.625 us`.
- `mixtral-k002`: `1.345 us` initialization, `2.135 us` scale reduction, and
  `1.870 us` quantization; stage sum `5.350 us`.
- `mixtral-k004`: `1.205 us` initialization, `3.115 us` scale reduction, and
  `2.270 us` quantization; stage sum `6.590 us`.

The stage sums closely track the uninstrumented CUDA Graph latency. This confirms
that the measured time is inside the device execution graph rather than Python or
HIP host submission overhead.

Across the 24 profiled `initializeScale` dispatches, the minimum observed
duration was `1.04 us`. This value is the calibrated single-dispatch latency
floor:

\[
T_{\mathrm{dispatch}} = 0.00104\ \mathrm{ms}
\]

## Counter evidence

The largest case, `mixtral-k006`, was used for counter classification:

- Maximum `VALUBusy`: `0.381%`.
- Maximum `MemUnitStalled`: `0.018%`.
- No MFMA work is present.
- The two data-processing grids contain 64 workgroups on a 256-CU device.

Compute and memory pipelines are both far from saturation. The small grids,
serialized short kernels, and sub-microsecond resource service terms identify
device latency as the dominant limit.

## Ideal execution model

The lower bound allows a legal single-dispatch cooperative implementation:

1. Each workgroup loads its assigned BF16 elements once and retains them on chip.
2. Workgroups reduce the tensor-wide maximum and synchronize.
3. The retained values are converted and written as FP8.
4. One FP32 scale is written.

This removes the current scale-initialization launch and the second BF16 input
read. It does not assume case-ID dispatch or known input values.

For \(N\) elements, minimum semantic I/O is:

\[
B_{\mathrm{semantic}} = 2N + N + 4 = 3N + 4\ \mathrm{bytes}
\]

The terms are one BF16 read, one FP8 write, and one FP32 scale write. The
operation contains no matrix work. This model treats its arithmetic service term
as no larger than the semantic memory service term, supported by the low
`VALUBusy` measurement. If conversion or reduction throughput is later shown to
exceed the memory term, the ceiling must be revised. The launch-aware lower bound
is:

\[
T_{\mathrm{ideal}} =
T_{\mathrm{dispatch}}
+ \frac{B_{\mathrm{semantic}}}{8.0 \times 10^9\ \mathrm{bytes/ms}}
\]

## Per-case calculation

For `mixtral-k006`:

\[
B = 3 \times 262144 + 4 = 786436\ \mathrm{bytes}
\]

\[
T_{\mathrm{ideal}} =
0.00104 + \frac{786436}{8.0 \times 10^9}
= 0.001138305\ \mathrm{ms}
\]

For `mixtral-k002`:

\[
B = 3 \times 32768 + 4 = 98308\ \mathrm{bytes}
\]

\[
T_{\mathrm{ideal}} =
0.00104 + \frac{98308}{8.0 \times 10^9}
= 0.001052289\ \mathrm{ms}
\]

For `mixtral-k004`:

\[
B = 3 \times 229376 + 4 = 688132\ \mathrm{bytes}
\]

\[
T_{\mathrm{ideal}} =
0.00104 + \frac{688132}{8.0 \times 10^9}
= 0.001126017\ \mathrm{ms}
\]

The equal-weight task ceiling is:

\[
\texttt{mean\_case\_latency\_ms} =
\frac{0.001138305 + 0.001052289 + 0.001126017}{3}
= 0.001105537\ \mathrm{ms}
\]

The measured implementation reaches:

\[
\frac{0.001105537}{0.006058409} \times 100\%
= 18.25\%
\]

## Interpretation

`latency-bound` means that irreducible device dispatch and synchronization
latency is larger than ideal compute or HBM service time. It does not mean that
the benchmark is dominated by host launch overhead.

The `0.001105537 ms` value assumes that a correct single-dispatch implementation
is legal. If the task is changed to require the existing three-stage structure,
the ceiling must be recalculated with at least three device dispatch floors.

If a valid implementation measures below any recorded per-case ideal latency,
the ceiling is invalid and must be recomputed rather than clamping efficiency to
100%.
