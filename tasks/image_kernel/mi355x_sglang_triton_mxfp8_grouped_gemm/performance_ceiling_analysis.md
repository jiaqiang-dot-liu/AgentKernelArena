# Performance ceiling analysis

## Result

- Bound classification: `mixed-bound`.
- Mean ideal case latency: `0.159286152 ms`.
- Measured task mean latency: `0.873795944 ms`.
- Measured efficiency relative to the ideal model: `18.23%`.

## Scope and environment

The timed operator contains two serial grouped GEMMs:

1. GEMM1 uses \(R=T \times 4\), \(N=2I\), \(K=H\), gathers one activation row
   for each route, and writes BF16.
2. GEMM2 uses \(R=T \times 4\), \(N=H\), \(K=I\), applies one FP32 route
   weight, and writes FP32 per-route output.

MoE alignment, activation quantization, SwiGLU, and final top-k reduction are
outside the timing boundary.

- GPU: AMD Instinct MI355X/gfx950, 256 CUs.
- Runtime: ROCm 7.2, PyTorch 2.9.1.
- Profiler: `rocprofv3 1.1.0`.
- HBM ceiling: `8.0 TB/s`.
- Dense MXFP8 matrix ceiling: `5.0332 PFLOP/s`, not the BF16
  `2.5 PFLOP/s` ceiling. Source: [AMD Instinct MI355X GPU brochure](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf).
- Dispatch floor: `1.08 us`, calibrated from 50 one-workgroup PyTorch/HIP
  dispatches in this container.

## Measurement and profiling evidence

All three correctness cases passed with relative error from `0.0270` to
`0.0273`. CUDA Graph timing reported:

- `decode-t1`: `0.046506119 ms`.
- `decode-t64`: `0.225618380 ms`.
- `prefill-t16384`: `2.349263334 ms`.

The deterministic generated routes selected `4`, `110`, and `128` experts and
produced `256`, `7,040`, and `69,376` post-padding route slots.

For the two target dispatches, `rocprofv3` measured:

- `decode-t1`: GEMM1 `39.436 us`, GEMM2 `5.805 us`.
- `decode-t64`: GEMM1 `142.646 us`, GEMM2 `72.151 us`.
- `prefill-t16384`: GEMM1 `1181.949 us`, GEMM2 `991.456 us`.

On `prefill-t16384`, GEMM1 showed `12.111%` mean `MfmaUtil` and `0.037%`
mean `MemUnitStalled`; GEMM2 showed `7.094%` and `0.615%`, respectively.
The larger GEMM2 memory-stall signal is consistent with its FP32 route-output
traffic.

Each current launcher also emits an output zero-fill kernel, so the measured
region has four device nodes: fill, GEMM1, fill, GEMM2. Zero-fill is not
semantically required because every valid route row is overwritten. The ideal
model retains the two required grouped-GEMM dispatches and removes both fills.

## Two-stage ideal model

Let \(A\) be the number of active experts, \(P\) the number of post-padding
slots, \(R=4T\), \(H=6144\), and \(I=384\). Routing metadata read by each stage
is:

\[
B_{\mathrm{meta}} = 4P + 4\left\lceil\frac{P}{64}\right\rceil + 4
\]

This is the padded sorted-token array, one expert ID per 64-row block, and the
post-padding count.

GEMM1 work and traffic are:

\[
F_1 = 4RHI
\]

\[
B_1 =
\frac{33}{32}\left(TH + A(2I)H\right)
+ 4RI + B_{\mathrm{meta}}
\]

GEMM2 work and traffic are:

\[
F_2 = 2RHI
\]

\[
B_2 =
\frac{33}{32}\left(RI + AHI\right)
+ 4RH + 4R + B_{\mathrm{meta}}
\]

The factor \(33/32\) counts one UE8M0 byte for every 32 FP8 values. The
`4RI` GEMM1 term is BF16 \([R,2I]\) output. The `4RH` GEMM2 term is FP32
\([R,H]\) output.

The two stages are serialized:

\[
T_{\mathrm{ideal}} =
\left[
0.00108 + \max\left(\frac{F_1}{5.0332\times10^{12}},
\frac{B_1}{8.0\times10^9}\right)
\right]
+
\left[
0.00108 + \max\left(\frac{F_2}{5.0332\times10^{12}},
\frac{B_2}{8.0\times10^9}\right)
\right]\ \mathrm{ms}
\]

Aggregating both stages before applying one `max` would incorrectly allow
GEMM1 compute service to overlap GEMM2 memory service.

## Per-case results

- `decode-t1`: GEMM1 has `37,748,736` operations and `19,477,716` bytes,
  giving `0.003514715 ms`; GEMM2 has `18,874,368` operations and `9,833,044`
  bytes, giving `0.002309131 ms`. Total ideal latency is `0.005823845 ms`;
  memory-bound.
- `decode-t64`: GEMM1 has `2,415,919,104` operations and `536,092,604`
  bytes, giving `0.068091576 ms`; GEMM2 has `1,207,959,552` operations and
  `274,055,100` bytes, giving `0.035336887 ms`. Total ideal latency is
  `0.103428463 ms`; memory-bound.
- `prefill-t16384`: GEMM1 has `618,475,290,624` operations and `827,608,308`
  bytes, giving `0.123959141 ms` and is compute-bound. GEMM2 has
  `309,237,645,312` operations and `1,948,536,052` bytes, giving
  `0.244647007 ms` and is memory-bound. Total ideal latency is
  `0.368606147 ms`; mixed-bound.

The arithmetic mean of the three ideal values is `0.159286152 ms`.

## Contract limitation

The benchmark materializes GEMM2 input during setup, so the timed GEMM2 does
not consume the timed GEMM1 output. This analysis follows the declared task
contract that requires both launches. If the scored contract is instead
interpreted only from the returned GEMM2 tensor, GEMM1 is not enforced and the
two-stage ceiling would not be valid.
