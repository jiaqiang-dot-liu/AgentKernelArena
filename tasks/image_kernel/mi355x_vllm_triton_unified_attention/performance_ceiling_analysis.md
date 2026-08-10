# Performance ceiling analysis

## Result

- Classification: `mixed-bound`
- Mean ideal latency: `0.014833230 ms`
- Measured mean latency: `0.031308046 ms`
- Ceiling efficiency: `47.37%`

## Profiling evidence

All five cases passed correctness and CUDA Graph timing. The current 3D path
launches `kernel_unified_attention_3d` followed by `reduce_segments`; their mean
combined target times range from `15.43 us` to `57.91 us`. The representative
Gemma case showed `2.830%` mean MFMA utilization and `0.082%` mean memory-stall
time.

## Model

For each decode case:

\[
F = 4B H_q L D
\]

\[
Bytes = 4B H_q D + 2B H_{kv} L D \times bytes_{kv}
\]

Q and output are BF16. KV uses two bytes except for the Mixtral FP8-cache case,
which uses one. The task contract requires the partial-attention and
`reduce_segments` launches. The ideal model retains two dispatches but reduces
the legal segment count from the current 16 to one.

At 8 TB/s HBM and 2.5 PFLOP/s BF16 matrix peak, ideal latencies are
`0.010619778`, `0.035834242`, `0.019253634`, `0.004229506`, and
`0.004228994 ms`. The first three cases are memory-bound; the two smallest are
at the memory/latency crossover, giving a task-level `mixed-bound` label.

## Why the task is mixed-bound

- `minimax-k004`, `gemma-k002`, and `gemma-k006` are memory-bound by KV and
  partial-output traffic.
- `gptoss-k020` and `mixtral-k031` have much smaller head dimensions. Their
  memory service time is comparable to the two mandatory device dispatch floors.
- Every scored case retains both `kernel_unified_attention_3d` and
  `reduce_segments`; CUDA Graph replay does not remove either device stage.

The task therefore spans memory-bound and latency/memory-crossover cases.
