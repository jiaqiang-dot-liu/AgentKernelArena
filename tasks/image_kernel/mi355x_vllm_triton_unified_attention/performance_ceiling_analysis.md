# Performance ceiling analysis

## Conclusion

- Ideal mean case latency: `0.014833230 ms`.
- Bound classification: `mixed-bound`.
- Baseline mean latency: `0.031308046 ms`.
- Baseline reaches `47.37%` of the modeled ceiling.

### Why mixed-bound

The MiniMax and Gemma cases are memory-bound by KV and partial-output traffic.
The smaller GPT-OSS and Mixtral cases sit at the memory/two-dispatch latency
crossover. Every scored case requires partial attention followed by
`reduce_segments`; CUDA Graph replay does not remove either device stage.

## Proof approach

1. Validate and benchmark all five geometries; trace
   `kernel_unified_attention_3d` and `reduce_segments`.
2. Retain two mandatory dispatches. The legal ideal reduces the segment count
   from the source-selected 16 to one, but does not remove the reduction stage.
3. Count BF16 Q/output, BF16 or FP8 K/V, page metadata, and the minimum
   partial-output/max/sum scratch traffic. Core attention work is:

   \[
   F=4BH_qLD
   \]

4. Evaluate serialized stages with `2.5 PFLOP/s` BF16, `8.0 TB/s` HBM, and
   `1.04 us` per dispatch.
5. Ideal latencies are `0.010619778`, `0.035834242`, `0.019253634`,
   `0.004229506`, and `0.004228994 ms`.
