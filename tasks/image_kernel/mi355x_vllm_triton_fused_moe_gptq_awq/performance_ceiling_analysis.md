# Performance ceiling analysis

## Result

- Classification: `mixed-bound`
- Mean ideal latency: `0.173026087 ms`
- Measured mean latency: `2.056249977 ms`
- Ceiling efficiency: `8.41%`

## Profiling evidence

All cases passed correctness and CUDA Graph timing. Each invocation launches
`fused_moe_kernel_gptq_awq` twice. Mean target-kernel time per launch was
`1314.305`, `1043.850`, and `649.261 us` for 7211, 1080, and 64 tokens. The
largest case showed `16.435%` mean MFMA utilization and `0.169%` mean
memory-stall time.

## Model

The two expert GEMMs perform:

\[
F = 6M \times 8 \times 7168 \times 256
\]

The semantic bytes include packed INT4 weights and BF16 scales for the experts
actually selected by each deterministic route tensor, plus activation, output,
and routing metadata. Profiling found 384, 384, and 282 active experts.

The ideal model permits a token-centric one-dispatch implementation that performs
both GEMMs and top-k reduction internally. At 8 TB/s HBM and 2.5 PFLOP/s BF16
matrix peak it gives:

- 7211 tokens: `0.255099060 ms`, compute-bound.
- 1080 tokens: `0.153555008 ms`, memory-bound.
- 64 tokens: `0.110424192 ms`, memory-bound.

The task is `mixed-bound` because the dominant roofline term changes with token
count.

## Why the task is mixed-bound

- At 7211 tokens, matrix work exceeds minimum expert-weight service time, so the
  case is compute-bound.
- At 1080 tokens, the same 384-expert INT4 weight and scale bank dominates the
  smaller matrix workload.
- At 64 tokens, only 282 experts are active, but their packed weights and scales
  still dominate the very small GEMMs.

The task-level label must not be replaced by the bound of its largest case.
