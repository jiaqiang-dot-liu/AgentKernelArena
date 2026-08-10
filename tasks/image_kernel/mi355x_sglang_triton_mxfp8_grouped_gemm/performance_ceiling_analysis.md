# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.159286152 ms`.
- Bound classification: `mixed-bound`.
- Baseline mean latency: `0.873795944 ms`.
- Baseline reaches `18.23%` of the modeled ceiling.

### Why mixed-bound

Decode cases are memory-bound by active-expert MXFP8 weights and UE8M0 scales.
For prefill, GEMM1 is compute-bound while GEMM2 is memory-bound because it writes
large FP32 per-route output. The two stages are serialized, so their different
resource limits cannot overlap.

## Proof approach

1. Validate and benchmark token counts 1, 64, and 16384; record deterministic
   active experts and padded route slots.
2. Trace two target grouped-GEMM dispatches and remove only non-semantic
   zero-fill nodes from the ideal model.
3. With routed rows \(R=4T\), hidden \(H=6144\), and intermediate \(I=384\):

   \[
   F_1=4RHI,\qquad F_2=2RHI
   \]

   Stage traffic counts selected FP8 values, one UE8M0 scale per 32 values,
   routing metadata, BF16 GEMM1 output, and FP32 GEMM2 output.

4. Evaluate each serial stage independently with `5.0332 PFLOP/s` MXFP8,
   `8.0 TB/s` HBM, and a `1.08 us` dispatch floor, then add the stage times.
5. Ideal total latencies are `0.005823845`, `0.103428463`, and
   `0.368606147 ms`; their mean is `0.159286152 ms`.
