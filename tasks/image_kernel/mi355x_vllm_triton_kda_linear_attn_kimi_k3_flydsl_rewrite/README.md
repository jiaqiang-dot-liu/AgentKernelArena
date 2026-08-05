# mi355x-kimi-k3-kda-linear-attn-flydsl-rewrite-20260728

FlyDSL **rewrite** variant of `mi355x_vllm_triton_kda_linear_attn_kimi_k3`. Same
operator, same session, same five cases — but the target language is FlyDSL, so
the vendored Triton implementation is the correctness oracle and the speedup
baseline instead of the edit surface, and the agent produces `kernel.py`.

`config.yaml` carries a `rewrite:` block, which is what makes
`agents/forge/launch_agent.py` dispatch `kernel-agents forge-rewrite-by-flydsl`
instead of `forge-loop`. Everything else — seeding, perf-helper injection, the
arena compile/correctness/performance commands — is a normal `image_kernel` task.

## Verification status

**Not yet exercised on a GPU.** This task needs the Kimi-K3 vLLM build
(`0.1.dev19253+g5f76ae224`, ROCm 7.2.3), which vendors the KDA tree under
`vllm/models/kimi_k3/amd/ops/third_party/kda` and the FLA ops under
`vllm/third_party/flash_linear_attention`. Neither exists in upstream vLLM
0.24.0, so on an image without them every mode fails at
`ModuleNotFoundError: No module named 'vllm.models.kimi_k3'`.

What *was* verified on gfx950, by standing a torch implementation of the FlyDSL
contract in for `kernel.py` and the float64 golden in for the Triton oracle:

| checked | result |
|---|---|
| golden vs. a contract-conformant implementation, all 5 cases | `cos = 0.999999`, `rel_max_err <= 0.0035` |
| input construction matches the sibling Triton task | chunk `t1080` output norm `1.473`, same as that task's recorded `1.473997` |
| decode strided views into `mixed_qkv` | `q/k/v.stride() == (4608, 128, 1)`, i.e. `3 * H * D` rows |
| driver correctness contract | emits `SNR:` and `allclose:`; state reset between the two implementations holds (state SNR ~142 dB) |
| driver rejects a port that skips the state write-back | `allclose: False`, state SNR `-12.6 dB` |
| driver rejects a port that ignores `state_indices` | `allclose: False`, decode SNR `6.2 dB` |
| arena correctness rejects the same missing state write-back | `final state normalized max err 2.2088 too high` |
| arena performance with the injected CUDA-graph helpers | runs; falls back to event timing for a host-syncing implementation, as designed |
| pre-port state (no `kernel.py`) | driver correctness prints `allclose: False`; `--bench-mode` exits 1 with a clear message |
| launcher dispatch | `forge-rewrite-by-flydsl`, anchor `fused_recurrent.py`, all 5 shapes forwarded |

Still unverified: the two Triton oracle calls themselves, `--ref-bench-mode`
baselines, and the end-to-end run through KernelForge. Those need the K3 image.

## One FlyDSL entry for two source entries

The source exposes two entries, and the session's hot kernels sit behind both:

- **k007** `fused_recurrent_kda_packed_decode_kernel` (decode; 48.4 ms, 2.11% GPU),
  launched only by `fused_recurrent_kda_packed_decode` from the non-spec decode
  branch. The sibling `fused_recurrent_kda` launches a *different* kernel on the
  speculative-decode branch and K3 sets `num_nextn_predict_layers=0`, so it never
  runs — targeting it would benchmark dead code.
- the **prefill chunk group** (`chunk_kda_fwd_*`, `chunk_gated_delta_rule_fwd_h_*`,
  `kda_gate_chunk_cumsum_*`, `l2norm_*`, `solve_tril`, ~43.7 ms) reached through
  `chunk_kda_with_fused_gate`.

They evaluate the **same** gated delta-rule recurrence — one token-serial, one
blockwise. Asking a port to reproduce both kernel decompositions in FlyDSL inside
the PORT budget is not a realistic target; reproducing the recurrence is. So
`kernel.py` implements a single varlen entry driven by `cu_seqlens` and
`state_indices`, and decode is the degenerate case of one token per segment:

```
build_kda_linear_attn_module(num_heads, head_dim, chunk_size) -> launch_fn
launch_fn(out, q, k, v, raw_g, raw_beta, A_log, dt_bias,
          state, state_indices, cu_seqlens, scale, lower_bound)
```

This is not a softened target. `--ref-bench-mode` still calls both real Triton
entries with their native signatures, so the baseline is the framework's actual
code path, and the same recurrence that a token-serial port evaluates in T steps
the source evaluates in T/64 blocked steps. That gap *is* the task: a port good
enough to clear the correctness gate at 320 tokens will be far off the baseline
at T=32768, which is where the OPTIMIZE phase has to earn its speedup.

`scripts/forge_driver.py` is self-contained and is embedded verbatim in the port
agent's prompt, so it is the authoritative statement of the layouts.

## Semantics

Per segment `n`, with `S = state[state_indices[n]]` and `t` over
`[cu_seqlens[n], cu_seqlens[n+1])`:

```
g   = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))   # safe-gate branch
qn  = l2norm(q_t) * scale ;  kn = l2norm(k_t)                 # eps 1e-6
S  *= exp(g)                     # decay per k-column
vt  = v_t - S @ kn               # delta-rule "remove old value"
vt *= sigmoid(raw_beta_t)
S  += outer(vt, kn)
out_t = S @ qn
```

`S` is fp32 and is left updated in place — decode feeds it straight back in on
the next step.

### Three contract details that are easy to get wrong

All read off the kernel sources, not assumed:

1. **`lower_bound = -5.0` is not a clamp — it selects a different gate
   function.** Without the bound the source computes
   `-exp(A_log) * softplus(raw_g + dt_bias)`; K3 never takes that branch
   (`fused_recurrent.py:513-521`, `chunk.py:507-515`).
2. **`raw_beta` is pre-sigmoid.** The source applies `sigmoid` internally
   (`fused_recurrent.py:525`, `chunk.py:470`), so pre-applying it would square
   the gate.
3. **In the decode case `q/k/v` are strided views into the packed `mixed_qkv`
   block**, so their row stride is `3 * H * D`, not `H * D`. Handing the port
   contiguous copies instead would have given it memory traffic the Triton
   kernel does not pay, and biased the comparison.

Also: `A_log` is 1-D of length `local_num_heads` and `dt_bias` is
`local_num_heads * head_dim`; decode `state_indices` start at 1 because slot 0 is
the NULL slot (`fused_recurrent.py:481`).

## Cases

Identical to `mi355x_vllm_triton_kda_linear_attn_kimi_k3`, down to the seeds and
the tensor generation order, so the two tasks time the same numbers and the
Triton-optimization and FlyDSL-rewrite results are directly comparable. Dims are
the session's per-rank TP=8 shapes: `num_heads=12`, `head_dim=128`,
`chunk_size=64`.

| id | mode | seqs x len | source |
|---|---|---|---|
| `kda-decode-packed-k007` | packed_decode | 62 x 1 | reconstructed (slice concurrency conc62) |
| `kda-prefill-chunk-t7211` | chunk | 1 x 7211 | trace |
| `kda-prefill-chunk-t1080` | chunk | 1 x 1080 | trace |
| `kda-long-chunk-t16384` | chunk | 1 x 16384 | extrapolated headroom |
| `kda-long-chunk-t32768` | chunk | 1 x 32768 | extrapolated headroom |

Profiling is a single-shape probe pinned to `kda-decode-packed-k007` — the one
case with an individual hot-kernel id — so the profiled kernel never drifts
between runs.

Each case carries its own benchmark budget. The long cases deliberately allow few
graph repeats: a token-serial port at T=32768 is orders of magnitude slower than
the blocked source, and the rewrite pipeline benches the whole suite under a
single 600 s timeout.

## Two gates, deliberately different

- **`scripts/forge_driver.py`** (the rewrite pipeline's PORT gate) compares the
  port to the *live Triton output* and reports SNR, which is what
  `--snr-threshold 30` acts on.
- **`scripts/task_runner.py`** (the arena gate) compares whichever implementation
  is under test to an *independent float64 golden* transcribed from
  `fused_recurrent_kda_packed_decode_kernel`. That is strictly stronger: it also
  catches a bug the port would inherit by imitating the source. Tolerance is
  `cos > 0.999` and normalized max error `< 0.03`.

Both check the **final recurrent state**, not just `out`. With one token per
segment `out` only observes the state through the q projection, so a state error
orthogonal to `q` would otherwise pass — and it would corrupt every subsequent
decode step. The arena gate checks the state for the FlyDSL path only, where the
contract fixes the update to `state[state_indices[n]]`; how the Triton source
surfaces its final state is the oracle's own business.

Correctness caps the token count at 320 (still 5 chunks at `chunk_size=64`, so
the cross-chunk path is exercised) because the golden is an O(T) float64 loop.
Unlike the sibling task the *segment* count is not capped: a rewrite gets varlen
segment indexing wrong far more easily than the in-tree source does, and the
decode case's 62 extra loop steps cost nothing.

## Run

```
python3 scripts/task_runner.py compile       # smoke: one KDA call
python3 scripts/task_runner.py correctness   # float64 parity, both modes
python3 scripts/task_runner.py performance   # CUDA-graph timed -> build/performance_report.json

python3 scripts/forge_driver.py                  # SNR / allclose vs live Triton
python3 scripts/forge_driver.py --ref-bench-mode # the speedup baseline
python3 scripts/forge_driver.py --bench-mode     # the FlyDSL port
```

With no `kernel.py` the runner measures the Triton source, which is the baseline
run; with a working `kernel.py` it measures the port.
