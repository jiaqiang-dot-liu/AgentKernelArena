# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.053289573 ms`.
- Bound classification: `memory-bound`.
- Baseline mean latency: `0.089158454 ms`.
- Baseline reaches `59.77%` of the modeled ceiling.

### Why memory-bound

Decode repeatedly streams the paged BF16 K/V cache. For every tested GQA
geometry, ideal HBM service is much larger than BF16 matrix service. The launch
contract also requires a partition kernel and a reduction kernel, but their two
dispatch floors remain smaller than K/V traffic at context lengths 1024–2048.

## Proof approach

1. Validate all seven cases and measure the uninstrumented CUDA Graph latency.
2. Trace `paged_attention_ll4mi_QKV_mfma16_kernel` and
   `paged_attention_ll4mi_reduce_kernel`; retain two mandatory dispatches and
   include partition scratch traffic.
3. For batch \(B\), query heads \(H_q\), KV heads \(H_{kv}\), context \(L\), and
   head dimension \(D\), count core attention work as:

   \[
   F = 4BH_qLD
   \]

   Semantic traffic includes Q/output, K/V pages, page metadata, partial output,
   max, and softmax sums.

4. Evaluate each stage with MI355X dense BF16 peak `2.5 PFLOP/s`, HBM
   `8.0 TB/s`, and dispatch floor `1.04 us`; serialized stage times are added.
5. The seven ideal latencies are `0.036308288`, `0.053356864`, `0.070405440`,
   `0.035972416`, `0.069799232`, `0.036476224`, and `0.070708544 ms`.
