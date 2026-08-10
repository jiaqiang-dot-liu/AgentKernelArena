# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.173026087 ms`.
- Bound classification: `mixed-bound`.
- Baseline mean latency: `2.056249977 ms`.
- Baseline reaches `8.41%` of the modeled ceiling.

### Why mixed-bound

The 7211-token case is compute-bound. The 1080- and 64-token cases are
memory-bound by packed INT4 expert weights and BF16 group scales. The
deterministic routes select 384, 384, and 282 experts, so the small case has a
smaller but still dominant weight bank.

## Proof approach

1. Validate and benchmark all three cases; trace both WNA16 expert-GEMM
   dispatches and collect MFMA/memory counters.
2. Count active experts from the actual deterministic route IDs.
3. For tokens \(M\), top-k \(K=8\), hidden \(H=7168\), and intermediate
   \(I=256\), matrix work is:

   \[
   F = 6MKHI
   \]

4. Semantic traffic includes selected packed INT4 W1/W2 weights, both BF16 scale
   banks, activation, output, and route IDs/weights. Model a legal token-centric
   fused dispatch using `2.5 PFLOP/s` BF16 service, `8.0 TB/s` HBM, and a
   `1.04 us` dispatch floor.
5. The ideal case latencies are `0.255099060`, `0.153555008`, and
   `0.110424192 ms`; their arithmetic mean is `0.173026087 ms`.
