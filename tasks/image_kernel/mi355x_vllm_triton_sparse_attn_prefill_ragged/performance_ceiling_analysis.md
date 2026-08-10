# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.075736640 ms`.
- Bound classification: `mixed-bound`.
- Baseline mean latency: `2.174253802 ms`.
- Baseline reaches `3.48%` of the modeled ceiling.

### Why mixed-bound

With pool-resident latent-KV reuse, the 7211- and 1073-query cases are
compute-bound. The 64-query case is at the compute/memory/dispatch crossover.
Implementations that repeatedly fetch gathered KV rows instead of reusing the
4–8 MB KV pool move the large cases toward memory-bound behavior.

## Proof approach

1. Validate and benchmark query counts 7211, 1073, and 64; trace the single
   `_sparse_attn_prefill_ragged_kernel` dispatch.
2. For queries \(Q\), heads \(H=64\), head dimension \(D=512\), top-k
   \(K=512\), and KV-pool size \(N_{kv}\):

   \[
   F=4QHKD
   \]

3. Minimum semantic traffic counts Q/output, one pool-resident KV read, sparse
   indices, and indptr:

   \[
   B=4QHD+2N_{kv}D+4QK+4(Q+1)
   \]

4. Apply `2.5 PFLOP/s` BF16, `8.0 TB/s` HBM, and a `1.04 us` dispatch floor.
5. Ideal latencies are `0.194608807`, `0.029843124`, and `0.002757987 ms`.
   Their equal-weight mean is the reported task ceiling.
