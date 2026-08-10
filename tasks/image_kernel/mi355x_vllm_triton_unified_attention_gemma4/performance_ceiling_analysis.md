# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.030501057 ms`.
- Bound classification: `memory-bound`.
- Baseline mean latency: `0.087319104 ms`.
- Baseline reaches `34.93%` of the modeled ceiling.

### Why memory-bound

All four decode cases stream more BF16 K/V data than their matrix-core service
time. Sliding-window cases attend exactly 1024 tokens even at context 2048; full
attention scales with the complete context. The one-dispatch floor is smaller
than ideal K/V service in every geometry.

## Proof approach

1. Validate and benchmark both Gemma4 head geometries at contexts 1024 and 2048.
2. Trace the single 2D `kernel_unified_attention` path; no reduction companion is
   present.
3. For sequences \(B\), query heads \(H_q\), KV heads \(H_{kv}\), attended
   length \(L\), and head dimension \(D\):

   \[
   F=4BH_qLD
   \]

   Semantic bytes include Q/output, shared K/V, and page metadata.

4. Apply `2.5 PFLOP/s` BF16, `8.0 TB/s` HBM, and a `1.04 us` dispatch floor.
5. Ideal latencies are `0.034662081`, `0.034662081`, `0.017950401`, and
   `0.034729665 ms`; HBM service dominates each case.
