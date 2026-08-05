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

End to end, the real `forge-rewrite-by-flydsl` pipeline was driven with the argv
the launcher builds, and **ingest and seed are confirmed working**: the spec
resolves `builder_symbol` to `build_paged_attention_decode_module`, matching what
the driver and task runner expect, and the pipeline wrote the seeded `kernel.py`.

**Not verified:** the PORT and OPTIMIZE stages. Both spawn an LLM session, and
the container used for this work has no Anthropic gateway configured
(`ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` unset). This is an environment
limitation, not a known defect.

## Scope note

FlyDSL is an MLIR-level DSL — see `tasks/flydsl2flydsl/flash_attn_func_kernel`
for what a hand-written attention kernel looks like there. Porting a full paged
attention (three GQA geometries, paged KV gather, split-K reduction) is a hard
target, and the PORT phase may well fail. That is a property of the benchmark,
not a defect: the task's job is to pose the problem faithfully. If the port
proves unreachable in practice, the natural narrowing is to keep only the two
GQA 4:1 cases and widen again once that works.

Expected runtime image:

```text
harbor.crusoe.primus-safe.amd.com/sync/vllm-openai-rocm:v0.24.0
```
