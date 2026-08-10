# Performance ceiling analysis

## Result

- Classification: `memory-bound`
- Mean ideal latency: `0.017854145 ms`
- Measured mean latency: `0.147522710 ms`
- Ceiling efficiency: `12.10%`

## Profiling evidence

All three cases passed correctness and CUDA Graph timing. Target-kernel times
were `76.756`, `148.726`, and `226.462 us` at context lengths 1024, 2048, and
3072. The largest case showed `1.655%` mean MFMA utilization and `0.001%` mean
memory-stall time.

## Model

For 64 sequences, four query heads, one KV head, and head dimension 256:

\[
F = 4 \times 64 \times 4 \times L \times 256
\]

\[
B = 4 \times 64 \times 4 \times 256
  + 4 \times 64 \times 1 \times L \times 256
\]

The first byte term is BF16 Q read plus output write. The second is one ideal
BF16 K/V-cache read shared by the four GQA heads. With a 1.04 us dispatch floor,
8 TB/s HBM, and 2.5 PFLOP/s BF16 matrix peak, the ideal case latencies are
`0.009463489`, `0.017854145`, and `0.026244801 ms`. HBM service dominates.
