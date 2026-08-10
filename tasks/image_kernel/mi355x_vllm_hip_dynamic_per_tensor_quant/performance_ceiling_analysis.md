# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.001105537 ms`.
- Bound classification: `latency-bound`.
- Baseline mean latency: `0.006058409 ms`.
- Baseline reaches `18.25%` of the modeled ceiling.

### Why latency-bound

The current operator uses three short device kernels. CUDA Graph replay removes
repeated host submission overhead, but it does not fuse those device nodes.
`rocprofv3` measured a `1.04 us` device dispatch floor, while the largest case
needs less than `0.10 us` of ideal HBM service. `VALUBusy` stayed below `0.381%`
and `MemUnitStalled` below `0.018%`, so neither compute nor HBM throughput is the
dominant lower bound.

## Proof approach

1. Run correctness and the task's CUDA Graph benchmark for all three cases.
2. Trace `initializeScale`, `data_to_scale_kernel`, and `scaled_quant_kernel`
   with `rocprofv3`; use the minimum observed trivial-kernel duration as the
   device latency floor.
3. Model a legal one-dispatch fused implementation. For \(N\) elements, minimum
   semantic traffic is one BF16 read, one FP8 write, and one FP32 scale:

   \[
   B = 3N + 4
   \]

4. Use the MI355X HBM ceiling of `8.0 TB/s`:

   \[
   T_i = 0.00104 + B_i/(8.0\times10^9)\ \mathrm{ms}
   \]

5. The three ideal case latencies are `0.001138305`, `0.001052289`, and
   `0.001126017 ms`. Their arithmetic mean is `0.001105537 ms`.
