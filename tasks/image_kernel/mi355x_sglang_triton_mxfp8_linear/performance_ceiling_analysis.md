# Performance ceiling analysis

## Result

- Bound classification: `mixed-bound`.
- Mean ideal case latency: `0.017078265 ms`.
- Measured task mean latency: `0.120717750 ms`.
- Measured efficiency relative to the ideal model: `14.15%`.

## Scope and environment

The timed operator is one `_run_mxfp8_linear_kernel` call. Activation
quantization is outside the timing boundary. The kernel reads FP8-E4M3
activations and weights, reads one UE8M0 byte per 32 values for each operand,
accumulates in FP32, and writes BF16 output.

- GPU: AMD Instinct MI355X/gfx950, 256 CUs.
- Runtime: ROCm 7.2, PyTorch 2.9.1.
- Profiler: `rocprofv3 1.1.0`.
- HBM ceiling: `8.0 TB/s`.
- Dense MXFP8 matrix ceiling: `5.0332 PFLOP/s`, not the BF16
  `2.5 PFLOP/s` ceiling. Source: [AMD Instinct MI355X GPU brochure](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf).

A one-workgroup FP32 zero-fill kernel was traced 50 times through the same
PyTorch/HIP path. Its minimum stable device duration was `1.08 us`, which is
used as the single-dispatch floor.

## Measurement and profiling evidence

All six correctness cases passed with relative error `0.0002`. CUDA Graph
timing reported:

- `qkv_proj-decode-m1`: `0.029638235 ms`.
- `qkv_proj-decode-m64`: `0.029759569 ms`.
- `qkv_proj-prefill-m16384`: `0.339848369 ms`.
- `o_proj-decode-m1`: `0.006940919 ms`.
- `o_proj-decode-m64`: `0.010153938 ms`.
- `o_proj-prefill-m16384`: `0.307965471 ms`.

`rocprofv3` observed one target dispatch per invocation. Mean target-kernel
durations were `31.470`, `32.280`, `385.459`, `9.780`, `11.265`, and
`336.513 us` in the case order above. On `qkv_proj-prefill-m16384`, mean
`MfmaUtil` was `14.768%` and mean `MemUnitStalled` was `0.043%`. This confirms
scaled-MFMA work without evidence that HBM stalls limit the large prefill case.

## Ideal model

For shape \((M,N,K)\), minimum matrix work is:

\[
F = 2MNK
\]

Minimum semantic traffic is:

\[
B =
MK + \frac{MK}{32}
+ NK + \frac{NK}{32}
+ 2MN
\]

The terms are FP8 activation values, activation UE8M0 scales, FP8 weights,
weight UE8M0 scales, and BF16 output. The scales are not folded into the FP8
value traffic.

The per-case lower bound is:

\[
T_{\mathrm{ideal}} =
0.00108
+ \max\left(
\frac{F}{5.0332 \times 10^{12}},
\frac{B}{8.0 \times 10^9}
\right)\ \mathrm{ms}
\]

## Per-case results

- `qkv_proj-decode-m1`: `15,728,640` operations and `8,118,976` bytes;
  compute `0.000003125 ms`, memory `0.001014872 ms`, ideal
  `0.002094872 ms`; dispatch/memory crossover.
- `qkv_proj-decode-m64`: `1,006,632,960` operations and `8,679,424` bytes;
  compute `0.000199999 ms`, memory `0.001084928 ms`, ideal
  `0.002164928 ms`; dispatch/memory crossover.
- `qkv_proj-prefill-m16384`: `257,698,037,760` operations and
  `153,862,144` bytes; compute `0.051199642 ms`, memory `0.019232768 ms`,
  ideal `0.052279642 ms`; compute-bound.
- `o_proj-decode-m1`: `12,582,912` operations and `6,501,408` bytes;
  compute `0.000002500 ms`, memory `0.000812676 ms`, ideal
  `0.001892676 ms`; latency-bound.
- `o_proj-decode-m64`: `805,306,368` operations and `7,342,080` bytes;
  compute `0.000159999 ms`, memory `0.000917760 ms`, ideal
  `0.001997760 ms`; dispatch/memory crossover.
- `o_proj-prefill-m16384`: `206,158,430,208` operations and
  `225,116,160` bytes; compute `0.040959714 ms`, memory `0.028139520 ms`,
  ideal `0.042039714 ms`; compute-bound.

The arithmetic mean of the six ideal values is `0.017078265 ms`. Decode and
prefill have different limiting resources, so the task-level classification is
`mixed-bound`.
