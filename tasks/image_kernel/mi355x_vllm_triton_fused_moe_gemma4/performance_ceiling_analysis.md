# Performance ceiling analysis

## Result

- Classification: `mixed-bound`
- Mean ideal latency: `0.109817744 ms`
- Measured mean latency: `0.460481549 ms`
- Ceiling efficiency: `23.85%`

## Profiling evidence

All cases passed correctness and CUDA Graph timing. Each invocation launches
`fused_moe_kernel` twice. Mean per-launch target times were `75.821`, `94.063`,
and `423.752 us` for 64, 1080, and 7218 tokens. The largest case showed
`25.686%` mean MFMA utilization and `0.142%` mean memory-stall time.

## Model

For \(M\) tokens, top-k 8, hidden size 2816, and per-rank intermediate size 352:

\[
F = 6M \times 8 \times 2816 \times 352
\]

The semantic byte model includes BF16 weights for the experts actually selected
by each deterministic route tensor, plus input, output, and routing metadata.
Profiling found 124, 128, and 128 active experts. The ideal model permits a
token-centric one-dispatch implementation of both GEMMs and top-k reduction.

At 8 TB/s HBM and 2.5 PFLOP/s BF16 matrix peak, ideal latencies are
`0.093315200`, `0.097727552`, and `0.138410481 ms`. Decode and 1080-token
prefill are weight-memory-bound; 7218-token prefill is compute-bound, making the
task `mixed-bound`.

## Why the task is mixed-bound

- The 64-token decode case selects 124 experts; selected BF16 expert weights
  dominate its small GEMMs.
- The 1080-token case selects all 128 experts and remains memory-bound.
- At 7218 tokens, the two expert GEMMs contain enough work to exceed the minimum
  expert-weight service time, making the case compute-bound.

The classification changes with routed token count even though the expert
geometry is unchanged.
