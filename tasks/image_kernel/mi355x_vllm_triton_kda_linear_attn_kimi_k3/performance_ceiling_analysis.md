# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.105989627 ms`.
- Bound classification: `mixed-bound`.
- Baseline mean latency: `0.693609661 ms`.
- Baseline reaches `15.28%` of the modeled ceiling.

### Why mixed-bound

Packed decode is memory-bound by FP32 recurrent-state traffic. Chunk cases have
a compute-dominant optimistic service roofline, but are practically
latency-bound by the token recurrence, nine serialized kernels, and very low
parallel occupancy. A single compute- or memory-bound label would hide this
dependency limitation.

## Proof approach

1. Validate packed decode and four chunk lengths against the independent FP64
   recurrence reference, then collect CUDA Graph baselines.
2. Trace the one-kernel decode path and the nine-kernel chunk path. Chunk
   profiling showed only about `1.8–2.3%` occupancy in the dominant recurrent
   state-propagation kernel.
3. Model a legal fused recurrence over \(T\) tokens, \(H=12\) heads, and
   \(D=128\):

   \[
   F_{\mathrm{proxy}}=TH(7D^2+11D+4)+H
   \]

   Semantic traffic counts BF16 Q/K/V/output, FP32 gate and beta, model
   parameters, and initial/final FP32 state.

4. Use `157.2864 TFLOP/s` FP32 vector peak, `8.0 TB/s` HBM, and a
   task-local `1.76 us` dispatch calibration. The numerical result is an
   optimistic hardware-service lower bound; recurrence latency is preserved in
   the qualitative classification.
5. Compute all five per-case lower bounds and take their arithmetic mean,
   yielding `0.105989627 ms`.
