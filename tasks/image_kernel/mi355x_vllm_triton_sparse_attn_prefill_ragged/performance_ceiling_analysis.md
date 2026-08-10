# Performance ceiling analysis

## Result

- Classification: `mixed-bound`
- Mean ideal latency: `0.075736640 ms`
- Measured mean latency: `2.174253802 ms`
- Ceiling efficiency: `3.48%`

## Profiling evidence

All three cases passed correctness and CUDA Graph timing. `rocprofv3` measured
the target kernel at `5836.966 us`, `967.392 us`, and `93.183 us` for query
counts 7211, 1073, and 64. On the largest case, mean MFMA utilization was
`16.954%` and mean memory-stall time was `0.041%`.

## Model

For query count \(Q\), KV-pool size \(N_{kv}\), 64 heads, head dimension 512,
and top-k 512:

\[
F = 4Q \times 64 \times 512 \times 512
\]

\[
B = 4Q \times 64 \times 512
  + 2N_{kv} \times 512
  + 4Q \times 512 + 4(Q+1)
\]

The terms cover Q read plus output write, one pool-resident latent-KV read,
indices, and indptr. The 4–8 MB KV pools fit in the shared last-level cache, so
the ideal semantic model reuses pool rows across queries. The ideal latency is:

\[
T = 0.00104 + \max(F/2.5{\times}10^{12}, B/8.0{\times}10^9)
\]

The per-case results are `0.194608807`, `0.029843124`, and `0.002757987 ms`.
The two large cases are compute-bound; the 64-query case is at the
compute/memory/latency crossover. Without cross-query KV reuse, the observed
implementation can behave memory-bound, hence the task-level `mixed-bound`
classification.

## Why the task is mixed-bound

- `dsv4-flash-prefill-sq7211` and `sq1073` are compute-bound when selected KV
  pool rows remain cache-resident and are reused across query heads and queries.
- `dsv4-flash-prefill-sq64` is at the compute, memory, and dispatch-latency
  crossover.
- An implementation that repeatedly fetches gathered KV rows from HBM moves the
  large cases toward memory-bound behavior.

The label therefore describes both the per-case crossover and the operator's
sensitivity to legally achievable KV reuse.
