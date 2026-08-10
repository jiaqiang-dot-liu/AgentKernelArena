# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.052599100 ms`.
- Bound classification: `mixed-bound`.
- Baseline mean latency: `0.400270797 ms`.
- Baseline reaches `13.14%` of the modeled ceiling.

### Why mixed-bound

The 64-token, 4096-wide case is latency-bound because its semantic memory
service is below the dispatch floor. The other cases are memory-bound, although
both 64-token cases remain near the latency/memory crossover because their grids
cannot use all 256 CUs. No case is compute-bound under the MI355X FP32 roofline.

## Proof approach

1. Validate and benchmark all four cases, then trace the post-mix, prenorm GEMM,
   and pre-mix TileLang kernels.
2. For token count \(T\), hidden width \(H\), `hc_mult=4`, and 20 Sinkhorn
   iterations, derive minimum semantic traffic:

   \[
   B = 20TH + 160T + 384H + 108
   \]

3. Count FP32 roofline work, including post/pre mixing, the 24-output projection,
   reductions, and Sinkhorn:

   \[
   F = T(244H + 1390)
   \]

4. Model a legal one-dispatch fused implementation using `157.3 TFLOP/s` FP32,
   `8.0 TB/s` HBM, and a `1.04 us` dispatch floor:

   \[
   T_i = 0.00104+\max(F_i/157.3{\times}10^9,\ B_i/8.0{\times}10^9)
   \]

5. The ideal case latencies are `0.002532238`, `0.130749418`, `0.001893262`,
   and `0.075221482 ms`.
