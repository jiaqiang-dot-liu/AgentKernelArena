# Performance ceiling analysis

## Result

- Classification: `memory-bound`
- Mean ideal latency: `0.053289573 ms`
- Measured mean latency: `0.089158454 ms`
- Ceiling efficiency: `59.77%`

## Profiling evidence

All seven cases passed correctness and CUDA Graph timing. The current operator
launches an MFMA16 partial-attention kernel followed by a reduction kernel.
Combined target-kernel times range from `56.09 us` to `130.37 us`. The largest
Qwen case showed `1.450%` mean MFMA utilization and `0.133%` mean memory-stall
time.

## Model

For each case:

\[
F = 4B H_q L D
\]

\[
Bytes = 4B H_q D + 4B H_{kv} L D
\]

The model reads BF16 Q once, writes BF16 output once, reads BF16 K/V cache once
per KV head, and includes partition scratch traffic. The task contract fixes the
partial-attention plus reduction launches, so the ideal retains two dispatches.

At 8 TB/s HBM and 2.5 PFLOP/s BF16 matrix peak, ideal case latencies are
`0.036308288`, `0.053356864`, `0.070405440`, `0.035972416`, `0.069799232`,
`0.036476224`, and `0.070708544 ms`. HBM service dominates every geometry.
