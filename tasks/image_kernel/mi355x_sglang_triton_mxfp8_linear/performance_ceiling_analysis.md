# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.017078265 ms`.
- Bound classification: `mixed-bound`.
- Baseline mean latency: `0.120717750 ms`.
- Baseline reaches `14.15%` of the modeled ceiling.

### Why mixed-bound

The two large prefill GEMMs are compute-bound. Decode with \(M=1\) or \(64\) is
latency-bound or at the dispatch/memory crossover because the fixed weight and
UE8M0 scale traffic dominates the small matrix workload.

## Proof approach

1. Validate and benchmark six QKV/O projection cases; trace one
   `_mxfp8_linear_kernel` dispatch per invocation.
2. Calibrate a `1.08 us` dispatch floor in the same SGLang container.
3. For shape \((M,N,K)\), count:

   \[
   F=2MNK
   \]

   \[
   B=MK+MK/32+NK+NK/32+2MN
   \]

   The byte terms are FP8 activations/weights, UE8M0 scales, and BF16 output.

4. Use the MI355X dense MXFP8 ceiling `5.0332 PFLOP/s` and HBM `8.0 TB/s`:

   \[
   T_i=0.00108+\max(F_i/5.0332{\times}10^{12},B_i/8.0{\times}10^9)
   \]

5. The arithmetic mean of the six per-case lower bounds is
   `0.017078265 ms`.
