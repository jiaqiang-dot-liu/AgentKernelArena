# Performance ceiling analysis

## Result

- Classification: `mixed-bound`
- Mean ideal latency: `0.052599100 ms`
- Measured mean latency: `0.400270797 ms`
- Ceiling efficiency: `13.14%`

## Profiling evidence

All four cases passed correctness and CUDA Graph timing. Every invocation
contains post-mix, prenorm GEMM, and pre-mix kernels. Their summed target-kernel
times track the task latency: approximately `47.45 us` and `1034.63 us` for the
7168-wide cases, and `33.98 us` and `518.67 us` for the 4096-wide cases. The
largest case showed no reported MFMA utilization and `0.877%` mean memory-stall
time.

## Model

For token count \(T\), hidden width \(H\), and `hc_mult=4`, the fused semantic
traffic model is:

\[
B = 20TH + 160T + 384H + 108
\]

It counts each required input and output once, including BF16 residual and layer
data, FP32 mix tensors, and the FP32 prenorm matrix. FP32 roofline work covering
post-mix, the 24-output prenorm GEMM, reductions, pre-mix, and 20 Sinkhorn
iterations is:

\[
F = T(244H + 1390)
\]

Using one ideal fused dispatch, 8 TB/s HBM, and 157.3 TFLOP/s FP32 peak gives
`0.002532238`, `0.130749418`, `0.001893262`, and `0.075221482 ms`. The
`deepseek-flash-k021` case is latency-bound; the other cases are memory-bound,
with the small cases close to the latency/memory crossover.

## Why the task is mixed-bound

- `deepseek-flash-k021` is latency-bound because its semantic memory service
  time is below the single-dispatch floor.
- `deepseek-pro-k014` is memory-bound by the strict formula but remains close to
  the latency/memory crossover.
- Both 7211-token cases are memory-bound; their large residual and output tensors
  dominate the FP32 projection work.

The task mixes latency-limited and memory-limited cases; it is not
compute-bound under the MI355X FP32 roofline.
