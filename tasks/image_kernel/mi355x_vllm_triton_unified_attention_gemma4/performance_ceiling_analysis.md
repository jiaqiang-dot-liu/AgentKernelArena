# Performance ceiling analysis

## Result

- Classification: `memory-bound`
- Mean ideal latency: `0.030501057 ms`
- Measured mean latency: `0.087319104 ms`
- Ceiling efficiency: `34.93%`

## Profiling evidence

All cases passed correctness and CUDA Graph timing. Target-kernel times were
`64.701` and `64.845 us` for the two sliding-window cases, and `77.441` and
`156.666 us` for full attention at context 1024 and 2048. The largest full case
showed `6.065%` mean MFMA utilization and `0.005%` mean memory-stall time.

## Model

For each case:

\[
F = 4B H_q L D
\]

\[
Bytes = 4B H_q D + 4B H_{kv} L D
\]

The model reads BF16 Q once, writes BF16 output once, and reads shared BF16 K/V
once per KV head. Sliding attention uses \(L=1024\) for both context lengths.
With one dispatch, 8 TB/s HBM, and 2.5 PFLOP/s BF16 matrix peak, ideal latencies
are `0.034662081`, `0.034662081`, `0.017950401`, and `0.034729665 ms`. The HBM
term dominates every case.
