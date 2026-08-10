# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.109817744 ms`.
- Bound classification: `mixed-bound`.
- Baseline mean latency: `0.460481549 ms`.
- Baseline reaches `23.85%` of the modeled ceiling.

### Why mixed-bound

The 64-token decode and 1080-token prefill cases are memory-bound by selected
BF16 expert weights. At 7218 tokens, matrix work exceeds minimum weight service
time, so the case becomes compute-bound. The deterministic routes select 124,
128, and 128 experts respectively.

## Proof approach

1. Validate and benchmark the three token counts; trace both
   `fused_moe_kernel` launches and surrounding MoE stages.
2. Measure distinct active experts from each deterministic routing tensor so the
   ideal does not charge unused expert weights.
3. For tokens \(M\), top-k \(K=8\), hidden \(H=2816\), and intermediate
   \(I=352\), count the two expert GEMMs:

   \[
   F = 6MKHI
   \]

4. Count selected BF16 expert weights, activation, output, and routing metadata.
   Model a legal token-centric fused dispatch using `2.5 PFLOP/s` BF16,
   `8.0 TB/s` HBM, and a `1.04 us` dispatch floor.
5. The ideal case latencies are `0.093315200`, `0.097727552`, and
   `0.138410481 ms`; their arithmetic mean gives the reported ceiling.
