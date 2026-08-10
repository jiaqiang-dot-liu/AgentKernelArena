# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.278601693 ms`.
- Bound classification: `mixed-bound`.
- Baseline mean latency: `1.052133249 ms`.
- Baseline reaches `26.48%` of the modeled ceiling.

### Why mixed-bound

The 7211-token prefill case is compute-bound by the two expert matrix stages.
The 62-token decode case is memory-bound by FP4 weights and E8M0 scales for 619
selected experts. The prefill route selects all 896 experts.

## Proof approach

1. Validate all 14 dispatch buckets and benchmark the two scored cases.
2. Trace sorting plus both FlyDSL GEMM stages and count active experts from the
   deterministic route tensors.
3. Model a legal token/expert-centric fused implementation. For routed rows
   \(R=16T\), hidden \(H=3584\), and intermediate \(I=384\):

   \[
   F_{\mathrm{matrix}}=6RHI
   \]

   Vector work includes SiTUv2, route weighting, and top-k reduction.

4. Minimum weight traffic for \(A\) active experts is:

   \[
   B_{\mathrm{weights}}=\frac{51}{32}AHI
   \]

   Add BF16 input/output and FP32 route IDs/weights.

5. Use the actual BF16 MFMA service rate `2.5166 PFLOP/s` used after MXFP4
   unpacking, FP32 vector peak `157.2864 TFLOP/s`, HBM `8.0 TB/s`, and a
   `1.76 us` dispatch floor. The two per-case lower bounds average to
   `0.278601693 ms`.
