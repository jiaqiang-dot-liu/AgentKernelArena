# Performance ceiling analysis

This analysis follows
[`performance-ceiling-methodology.md`](../../../docs/reference/performance-ceiling-methodology.md)
and [`benchmark-methodology.md`](../../../docs/reference/benchmark-methodology.md).

## Result

- Bound classification: `mixed-bound`.
- Mean ideal case latency: `0.105989627 ms`.
- Measured task mean latency: `0.693609661 ms`.
- Measured efficiency relative to the ideal model: `15.28%`.

Packed decode is memory-bound by FP32 state-cache traffic. Chunk cases have a
compute-dominant optimistic service roofline, but are qualitatively
`latency-bound` because the recurrence, serial kernel group, and limited
parallelism prevent the current implementation from approaching that roof.

## Operator and workload

KDA applies the following gated delta recurrence for every sequence, head, and
token:

\[
\begin{aligned}
q_t &\leftarrow \operatorname{l2norm}(q_t)/\sqrt{D} \\
k_t &\leftarrow \operatorname{l2norm}(k_t) \\
g_t &\leftarrow -5\,\operatorname{sigmoid}
  \left(\exp(A_{\log})(g^{raw}_t+b_g)\right) \\
S_t &\leftarrow S_{t-1}\odot\exp(g_t) \\
v'_t &\leftarrow
  \left(v_t-S_t k_t\right)\operatorname{sigmoid}(\beta^{raw}_t) \\
S_t &\leftarrow S_t+v'_t k_t^\mathsf{T} \\
o_t &\leftarrow S_t q_t
\end{aligned}
\]

All cases use \(H=12\) local heads and \(D=128\) for both key and value
dimensions. Q, K, V, and output are BF16. Gates, beta, and recurrent state are
FP32.

The scored cases are:

- packed decode: 62 independent one-token sequences;
- chunk prefill: one sequence with 1080 or 7211 tokens;
- long chunk headroom: one sequence with 16384 or 32768 tokens.

## Measurement environment

- GPU: AMD Instinct MI355X/gfx950, 256 CUs.
- Runtime: ROCm 7.2.3, PyTorch 2.11.0 development build.
- Profiler: `rocprofv3 1.1.0`.
- HBM ceiling: `8.0 TB/s`.
- FP32 vector ceiling: `157.2864 TFLOP/s`.

A one-workgroup FP32 fill was warmed 10 times and traced for 50 further
dispatches through the same PyTorch/HIP stack. The minimum stable device
duration was `1.76 us`.

## Correctness and uninstrumented benchmark

All five cases passed the independent FP64 recurrence reference:

- decode: cosine `0.999999`, normalized maximum error `0.0027`;
- chunk cases: cosine `0.999992`, normalized maximum error from `0.0052` to
  `0.0078`.

Uninstrumented CUDA Graph timing reported:

- `kda-decode-packed-k007`: `0.019342236 ms`;
- `kda-prefill-chunk-t7211`: `0.423641801 ms`;
- `kda-prefill-chunk-t1080`: `0.111238536 ms`;
- `kda-long-chunk-t16384`: `0.951685731 ms`;
- `kda-long-chunk-t32768`: `1.962140000 ms`.

Their arithmetic mean is `0.693609661 ms`.

CUDA Graph capture repeats `_run` on the same input object. Because KDA updates
state in place, cases with `benchmark_effective_repeats > 1` measure a chained
steady-state sequence rather than independent state snapshots. The per-call
semantic state traffic is unchanged, but cache residency can be more favorable
than a cold invocation. This is part of the scored harness contract.

## Profiling evidence

Packed decode is one
`fused_recurrent_kda_packed_decode_kernel` dispatch. Under tracing its mean
device duration was `29.751 us`. Counter collection reported:

- `38.552%` mean `VALUBusy`;
- `0%` `MfmaUtil`;
- `0.279%` `MemUnitStalled`;
- `36.662%` occupancy.

The chunk path uses nine device dispatches per invocation: two L2-normalization
kernels, gate cumsum, intra-chunk construction, inter-chunk solve, W/U
recomputation, recurrent state propagation, output construction, and one fill.
The summed mean durations of those timed dispatches were:

- T=1080: `118.791 us`;
- T=7211: `425.838 us`;
- T=16384: `951.416 us`;
- T=32768: `1901.699 us`.

The recurrent state-propagation kernel
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` was the largest component:

- T=1080: `43.681 us`;
- T=7211: `246.876 us`;
- T=16384: `587.746 us`;
- T=32768: `1198.159 us`.

For T=1080, this kernel had `1.805%` mean occupancy, `1.343%` `MfmaUtil`,
`2.980%` `VALUBusy`, and `0.124%` `MemUnitStalled`. At T=32768 it still had
only `2.275%` occupancy and `1.735%` `MfmaUtil`. The low compute and memory
signals together with the serial recurrence identify insufficient parallelism
and dependency latency, not HBM saturation, as the current chunk limitation.

## Ideal service model

The numerical ceiling uses a direct, fused recurrence as the legal ideal
implementation. Q/K normalization, gate evaluation, state update, and output
generation are combined into one dispatch. Intermediate tensors produced by
the current chunk algorithm are not semantic outputs and are not charged.

This produces an optimistic hardware-service lower bound. It does not assume
that the whole-device FP32 peak is realizable across a long dependent
recurrence; the qualitative chunk classification remains `latency-bound`.

### Arithmetic proxy

The dominant state work per token and head is:

- state decay: \(D^2\);
- state-vector product: \(2D^2\);
- outer-product state update: \(2D^2\);
- output state-vector product: \(2D^2\).

Q/K normalization, gate, and beta add lower-order vector work. Ordinary FP32
operations and special-function evaluations are tracked as:

\[
F_{\mathrm{ordinary}} = TH(7D^2+9D+1)
\]

\[
F_{\mathrm{SFU}} = TH(2D+3)+H
\]

For the roofline calculation, one SFU evaluation is counted as one proxy
operation:

\[
F_{\mathrm{proxy}} = TH(7D^2+11D+4)+H
\]

This convention understates the latency of exp, sigmoid, reciprocal square
root, and reduction dependencies. It is retained to make the optimistic lower
bound explicit and reproducible.

### Semantic traffic

For \(N\) sequences:

\[
B =
12THD
+4TH
+4H(D+1)
+8NHD^2
+B_{\mathrm{metadata}}
\]

The terms are:

- BF16 Q/K/V reads plus BF16 output write: \(8THD\);
- FP32 raw gate read: \(4THD\);
- FP32 raw beta: \(4TH\);
- FP32 \(A_{\log}\) and gate bias: \(4H(D+1)\);
- FP32 initial-state read and final-state write: \(8NHD^2\);
- state indices for decode or sequence offsets for chunk mode.

The per-case lower bound is:

\[
T_{\mathrm{ideal}} =
0.00176\ \mathrm{ms}
+10^3\max\left(
\frac{F_{\mathrm{proxy}}}{157.2864\times10^{12}},
\frac{B}{8.0\times10^{12}}
\right)\ \mathrm{ms}
\]

## Per-case calculation

### Packed decode, 62 sequences

- Arithmetic proxy: `86,378,412` operations.
- Semantic traffic: `98,669,768` bytes.
- Compute service: `0.000549179 ms`.
- Memory service: `0.012333721 ms`.
- Ideal latency: `0.014093721 ms`.
- Classification: `memory-bound`.
- Measured efficiency: `72.87%`.

### Chunk, T=1080

- Arithmetic proxy: `1,504,656,012` operations.
- Semantic traffic: `21,537,464` bytes.
- Compute service: `0.009566345 ms`.
- Memory service: `0.002692183 ms`.
- Ideal latency: `0.011326345 ms`.
- Classification: `latency-bound`.
- Measured efficiency: `10.18%`.

### Chunk, T=7211

- Arithmetic proxy: `10,046,365,212` operations.
- Semantic traffic: `134,838,344` bytes.
- Compute service: `0.063873070 ms`.
- Memory service: `0.016854793 ms`.
- Ideal latency: `0.065633070 ms`.
- Classification: `latency-bound`.
- Measured efficiency: `15.49%`.

### Chunk, T=16384

- Arithmetic proxy: `22,826,188,812` operations.
- Semantic traffic: `304,355,384` bytes.
- Compute service: `0.145125000 ms`.
- Memory service: `0.038044423 ms`.
- Ideal latency: `0.146885000 ms`.
- Classification: `latency-bound`.
- Measured efficiency: `15.43%`.

### Chunk, T=32768

- Arithmetic proxy: `45,652,377,612` operations.
- Semantic traffic: `607,131,704` bytes.
- Compute service: `0.290250000 ms`.
- Memory service: `0.075891463 ms`.
- Ideal latency: `0.292010000 ms`.
- Classification: `latency-bound`.
- Measured efficiency: `14.88%`.

## Interpretation and validity

Decode is close to its FP32 state-traffic floor; its main opportunity is
reducing state movement or improving cache reuse. Chunk KDA has a larger gap
because the current blockwise parallel algorithm materializes intermediates and
serializes nine kernels, while its dominant state propagation still exposes
very little device parallelism.

The chunk number is a service-time lower bound, not a calibrated prediction of
an attainable one-kernel implementation. A measured implementation below the
recorded value, or evidence that SFU/reduction latency is an unavoidable larger
term, invalidates the proxy and requires recalculation. The ceiling must also be
recomputed if the recurrence, state dtype, timing boundary, cases, benchmark
method, or GPU stack changes.
