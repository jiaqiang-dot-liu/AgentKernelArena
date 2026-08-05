# mi355x-rocm-paged-attention-decode-flydsl-rewrite-20260801

FlyDSL **rewrite** counterpart to `mi355x_vllm_hip_paged_attention_decode`. Same
operator, same seven session-derived cases — but the target language is FlyDSL:
AITER's HIP `paged_attention_rocm` becomes the correctness oracle and the speedup
baseline, and the agent must produce an equivalent FlyDSL kernel instead of
editing the HIP source.

Keeping the case list byte-identical to the sibling task is deliberate: it makes
same-language optimization and cross-language rewrite directly comparable on one
operator.

## How it differs from the sibling task

| | `..._paged_attention_decode` | this task |
| --- | --- | --- |
| Edit surface | AITER HIP sources (`pa_kernels.cuh`, …) | `kernel.py` (FlyDSL) |
| AITER source | the thing being optimized | protected oracle + baseline |
| Forge subcommand | `forge-loop` | `forge-rewrite-by-flydsl` |
| Speedup means | optimized HIP vs pristine HIP | FlyDSL vs AITER HIP |

## The contract

`kernel.py` must expose:

```python
build_paged_attention_decode_module(
    num_q_heads, num_kv_heads, head_size, block_size) -> launch_fn
launch_fn(out, query, key_cache, value_cache, block_tables, seq_lens, scale)
```

BF16 throughout, with `x = 16 // itemsize = 8`:

| Tensor | Shape |
| --- | --- |
| `query` | `(num_seqs, num_q_heads, head_size)` |
| `key_cache` | `(num_blocks, num_kv_heads, head_size // x, block_size, x)` |
| `value_cache` | `(num_blocks, num_kv_heads, head_size, block_size)` |
| `block_tables` | `(num_seqs, max_blocks_per_seq)` int32 |
| `seq_lens` | `(num_seqs,)` int32 |
| `out` | `(num_seqs, num_q_heads, head_size)`, written in place |

Decode paged attention, one query token per sequence: for sequence `s` and query
head `h`, attend over the first `seq_lens[s]` KV positions gathered through
`block_tables[s]`, softmax scale `scale`, GQA sharing
`num_q_heads // num_kv_heads` query heads per KV head. No ALiBi, no sliding
window, no FP8 KV cache.

Unlike the AITER source, the port is **not** handed `exp_sums` / `max_logits` /
`tmp_out` scratch, so whether to split the KV dimension and how to reduce is
entirely the implementation's choice.

`scripts/forge_driver.py` is the authoritative statement of all of this. It is
self-contained on purpose — the rewrite pipeline embeds the driver text in the
port agent's prompt (capped at 8 KB), so a driver that merely delegates to
`task_runner.py` would tell the agent nothing about the operator's I/O.

## Two valid states

Both the arena harness and the driver work with or without a port, which is what
makes baseline and optimized runs comparable:

- no `kernel.py`, or the seeded `NotImplementedError` stub — measure AITER; this
  is the baseline run;
- a working `kernel.py` — measure the FlyDSL port.

Arena correctness always validates whichever implementation is under test
against an **fp32 torch reference**. That is strictly stronger than comparing the
port to AITER, because it also catches a bug the port would inherit by imitating
the source. The driver additionally gates the port against the live AITER output
on the SNR threshold the rewrite pipeline enforces.

## Timing

Arena scoring uses the shared CUDA-graph helper, same as every other
image_kernel task. The driver carries its own compact CUDA-graph timer (it runs
standalone at workspace root, so it cannot use the injected helper) with an event
fallback: a port that syncs with the host — a `.item()`, a python-side length
read — cannot be graph-captured, and that is a legitimate if slower
implementation, so it degrades to per-launch event timing instead of failing the
whole benchmark. The chosen method is printed as `# timing: ...`.

## Forge integration

`config.yaml` carries a `rewrite:` block. `agents/forge/launch_agent.py` reads it
and dispatches `kernel-agents forge-rewrite-by-flydsl` instead of `forge-loop`;
tasks without the block are untouched. Everything else — seeding, perf-helper
injection, the three arena commands — stays a normal `image_kernel` task, so no
`src/` change was needed.

## Verified locally

On MI355X/gfx950, workspace materialized through
`src.preprocessing.setup_workspace`:

```text
compile      paged_attention_rocm compile smoke (aiter): PASS
correctness  PASS x7 (aiter)
performance  0.0556 / 0.0754 / 0.1260 / 0.0561 / 0.1139 / 0.0566 / 0.1186 ms
             benchmark_method=cuda_graph on every case

driver --ref-bench-mode   7x case_ms, median 0.0742 ms, # timing: cuda_graph
driver --profile-run      exit 0
driver (no kernel.py)     allclose: False, no crash
driver (seeded stub)      allclose: False, no crash
```

The post-port paths were exercised with a throwaway pure-PyTorch stand-in for
`kernel.py` (scratch workspace only, never shipped): correctness against the
AITER oracle gave **SNR 51.6–51.9 dB, allclose True on all seven cases**, so the
30 dB port gate has ample margin for this operator. That stand-in also surfaced
and confirmed the graph-capture fallback described above.

## End-to-end run

A full run through `main.py` with the forge agent on MI355X/gfx950
(`claude-opus-5`) exercised the whole chain:

```text
arena          task discovery -> workspace -> aiter_meta seeding
arena baseline 7 cases, average 0.1012 ms          (task_runner, AITER)
launcher       "task targets flydsl - dispatching forge-rewrite-by-flydsl"
pipeline       ingest -> seeded skeleton kernel.py -> using task driver
source baseline 0.0744 ms (full suite)             (driver --ref-bench-mode)
PORT           OK on attempt 1/3, SNR 51.62 dB
interim        flydsl 0.402904 ms vs source 0.074388 ms -> 0.185x
OPTIMIZE       forge-loop --fellow flydsl-fellow, iteration 1 reached
```

The port is genuine FlyDSL: one wavefront per (sequence, query head), a two-pass
softmax with the scores staged in LDS and fp32 accumulation, then a weighted
value reduction with the output dimension partitioned across lanes.

Two results worth reading carefully:

- **SNR 51.62 dB matches the PyTorch stand-in exactly.** That is the BF16 output
  quantization floor, so the port is numerically correct rather than barely
  clearing the 30 dB gate.
- **The ported kernel starts 5.4x slower than AITER.** That is expected — PORT is
  a correctness-only phase — and it is why the pipeline hands off to OPTIMIZE.
  Contrary to the scope note below, the port itself was not the obstacle here.

The in-session gate also demonstrably works: edit 6 was blocked with
`validation driver_error` (the candidate kernel faulted the GPU), the agent
recovered, and edit 8 was allowed.

The run was stopped once OPTIMIZE reached iteration 1, so there is no final
optimized speedup number. Everything before that is clean, with no errors in any
preceding stage.

## Scope note

FlyDSL is an MLIR-level DSL — see `tasks/flydsl2flydsl/flash_attn_func_kernel`
for what a hand-written attention kernel looks like there. Porting a full paged
attention (three GQA geometries, paged KV gather, split-K reduction) looked like
a hard enough target that the PORT phase might not clear it at all.

That turned out to be too pessimistic: the run above ported on the first
attempt. The real difficulty is on the other side — the port lands 5.4x slower
than the hand-tuned AITER assembly, so the interesting question this task poses
is whether OPTIMIZE can close a 5x gap, not whether the port is reachable. Keep
that in mind before narrowing the case list; the two GQA 4:1 cases remain the
natural reduction if it is ever needed.

Expected runtime image:

```text
harbor.crusoe.primus-safe.amd.com/sync/vllm-openai-rocm:v0.24.0
```
