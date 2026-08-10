# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.017854145 ms`.
- Bound classification: `memory-bound`.
- Baseline mean latency: `0.147522710 ms`.
- Baseline reaches `12.10%` of the modeled ceiling.

### Why memory-bound

The BF16 K/V-cache read grows linearly with context length and dominates both
matrix service and the single device dispatch. GQA allows four query heads to
share one ideal KV-head read, but the resulting arithmetic intensity remains far
below the MI355X BF16/HBM ridge point.

## Proof approach

1. Validate and CUDA-Graph benchmark context lengths 1024, 2048, and 3072.
2. Trace the single `kernel_paged_attention_2d` dispatch and collect MFMA and
   memory-stall evidence.
3. For 64 sequences, four query heads, one KV head, and head size 256:

   \[
   F=4\times64\times4\times L\times256
   \]

   Semantic bytes include BF16 Q/output, one shared K/V read per KV head, page
   IDs, sequence lengths, and prefix metadata.

4. Apply a `1.04 us` dispatch floor, `2.5 PFLOP/s` BF16 peak, and `8.0 TB/s`
   HBM:

   \[
   T_i=0.00104+\max(F_i/2.5{\times}10^{12},B_i/8.0{\times}10^9)
   \]

5. Ideal latencies are `0.009463489`, `0.017854145`, and `0.026244801 ms`.
