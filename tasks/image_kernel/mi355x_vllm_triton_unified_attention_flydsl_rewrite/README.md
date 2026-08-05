# mi355x-ci-unified-attention-flydsl-rewrite-20260720

FlyDSL **rewrite** counterpart to `mi355x_vllm_triton_unified_attention`. Same
operator, same five session-derived cases — but the target language is FlyDSL:
AITER's Triton `unified_attention` becomes the correctness oracle and the speedup
baseline, and the agent must produce an equivalent FlyDSL kernel instead of
editing the Triton source.

Keeping the case list identical to the sibling task is deliberate: it makes
same-language optimization and cross-language rewrite directly comparable on one
operator.

## How it differs from the sibling task

| | `..._triton_unified_attention` | this task |
| --- | --- | --- |
| Edit surface | AITER Triton `unified_attention.py` | `kernel.py` (FlyDSL) |
| Triton source | the thing being optimized | protected oracle + baseline |
| Forge subcommand | `forge-loop` | `forge-rewrite-by-flydsl` |
| Speedup means | optimized Triton vs pristine Triton | FlyDSL vs Triton |

## The contract

`kernel.py` must expose:

```python
build_unified_attention_decode_module(
    num_q_heads, num_kv_heads, head_size, block_size) -> launch_fn
launch_fn(out, query, key_cache, value_cache, block_tables, seq_lens,
          scale, k_scale, v_scale)
```

| Tensor | Shape / dtype |
| --- | --- |
| `query` | `(num_seqs, num_q_heads, head_size)` bf16 |
| `key_cache` | `(num_blocks, block_size, num_kv_heads, head_size)`, bf16 or `float8_e4m3fn` |
| `value_cache` | same shape and dtype as `key_cache` |
| `block_tables` | `(num_seqs, pages_per_seq)` int32 |
| `seq_lens` | `(num_seqs,)` int32 |
| `k_scale` / `v_scale` | `(1,)` fp32 dequant scales; 1.0 when the cache is bf16 |
| `out` | `(num_seqs, num_q_heads, head_size)` bf16, written in place |

Decode attention, one query token per sequence: for sequence `s` and query head
`h`, attend over the first `seq_lens[s]` KV positions gathered through
`block_tables[s]`, softmax scale `scale`, GQA sharing
`num_q_heads // num_kv_heads` query heads per KV head. An fp8 KV cache is
dequantized by the matching scale before use. No sliding window, no ALiBi, no
logit softcap.

`scripts/forge_driver.py` is the authoritative statement of all of this. It is
self-contained on purpose — the rewrite pipeline embeds the driver text in the
port agent's prompt (capped at 8 KB), so a driver that merely delegates to
`task_runner.py` would tell the agent nothing about the operator's I/O.

## The geometry spread is the hard part

Unlike the paged-attention rewrite task, where every case shares `head_size` 128
and `block_size` 16, these five cases deliberately span a wide range:

| Case | Q/KV heads | head_size | KV dtype | Session share |
| --- | --- | --- | --- | --- |
| `minimax-k004` | 12 / 2 | 128 | bf16 | 16.412% |
| `gemma-k002` | 8 / 4 | 256 | bf16 | 19.643% |
| `gemma-k006` | 8 / 1 | 512 | bf16 | 3.973% |
| `gptoss-k020` | 8 / 1 | 64 | bf16 | 7.648% |
| `mixtral-k031` | 4 / 1 | 128 | **fp8** | 5.764% |

That is head_size 64 through 512, GQA ratios from 2:1 to 8:1, and one fp8 KV
cache — all served by one implementation. A port specialized to a single shape
will fail the suite, so the contract takes the geometry as builder arguments.

## Two valid states

Both the arena harness and the driver work with or without a port, which is what
makes baseline and optimized runs comparable:

- no `kernel.py`, or the seeded `NotImplementedError` stub — measure Triton; this
  is the baseline run;
- a working `kernel.py` — measure the FlyDSL port.

Arena correctness always validates whichever implementation is under test
against an **fp32 torch reference**. That is strictly stronger than comparing the
port to Triton, because it also catches a bug the port would inherit by imitating
the source. The driver additionally gates the port against the live Triton output
on the SNR threshold the rewrite pipeline enforces.

## Timing

Arena scoring uses the shared CUDA-graph helper, same as every other
image_kernel task. The driver carries its own compact CUDA-graph timer (it runs
standalone at workspace root, so it cannot use the injected helper) with an event
fallback: a port that syncs with the host cannot be graph-captured, and that is a
legitimate if slower implementation, so it degrades rather than failing the whole
benchmark. The chosen method is printed as `# timing: ...`.

## Forge integration

`config.yaml` carries a `rewrite:` block. `agents/forge/launch_agent.py` reads it
and dispatches `kernel-agents forge-rewrite-by-flydsl` instead of `forge-loop`;
tasks without the block are untouched. Everything else — seeding, perf-helper
injection, the three arena commands — stays a normal `image_kernel` task.

Only the `aiter` package is seeded, so `_configure` symlinks the installed
`aiter_meta` beside it; `aiter.utility.aiter_types` resolves `aiter_enum.h`
relative to the seeded package's parent and would otherwise fail.

## Verified locally

On MI355X/gfx950, workspace materialized through
`src.preprocessing.setup_workspace`:

```text
compile      unified_attention compile smoke (triton): PASS
correctness  PASS x5 (triton)
performance  0.0261 / 0.0572 / 0.0469 / 0.0133 / 0.0137 ms
             benchmark_method=cuda_graph on every case

driver --ref-bench-mode  5x case_ms, median 0.0258 ms, # timing: cuda_graph
driver (no kernel.py)    allclose: False, no crash
driver --bench-mode      errors cleanly when kernel.py is absent
```

The post-port paths were exercised with a throwaway pure-PyTorch stand-in for
`kernel.py` (scratch workspace only, never shipped): correctness against the
Triton oracle gave **SNR 53.3–54.4 dB, allclose True on all five cases including
the fp8 one**, so the 30 dB port gate has ample margin here too. That stand-in
also confirmed the graph-capture event fallback.

The launcher was checked to detect this task's `rewrite` block (`op_name`
`unified_attention_decode`, five shapes forwarded) while the sibling task is
still routed to `forge-loop`.

A full `main.py` end-to-end run was done for the paged-attention rewrite task
rather than this one; the pipeline plumbing is shared, and see that task's README
for the observed PORT/OPTIMIZE behaviour.

Expected runtime image:

```text
harbor.crusoe.primus-safe.amd.com/sync/vllm-openai-rocm:v0.24.0
```
