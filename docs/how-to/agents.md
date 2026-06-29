# Configure agents and models

AgentKernelArena evaluates one agent per run. The agent is selected by the
`agent.template` field in `config.yaml`. This page lists the supported agents,
explains how models and providers are configured, and describes how to use the
arena for A/B testing.

## Supported agents

| `agent.template` | Description |
| --- | --- |
| `cursor` | Cursor Agent CLI |
| `claude_code` | Anthropic Claude Code CLI |
| `codex` | OpenAI Codex CLI |
| `swe_agent` | SWE-agent |
| `openevolve` | OpenEvolve (GEAK) evolutionary search |
| `geak_optimagentv2` | GEAK OptimAgent v2 |
| `geak_hip` | GEAK HIP agent |
| `geak_ourllm_kernel2kernel` | GEAK OurLLM kernel-to-kernel agent |
| `single_llm_call` | A single LLM call (no agent loop) |
| `task_validator` | Task quality validator (see [Validate tasks](task-validator.md)) |

Select one in `config.yaml`:

```yaml
agent:
  template: claude_code
```

Each agent lives under `agents/<agent_name>/` and is registered into a shared
registry, so the framework loads only the agent you select.

## Models and providers

The model an agent uses is configured by that agent's own integration (its CLI
configuration or an `agent_config.yaml` under `agents/<agent_name>/`).
Supported providers include:

- **OpenAI** (for example, GPT-5)
- **Anthropic Claude** (for example, Opus and Sonnet families)
- **OpenRouter** — access to a broad set of hosted models
- **Local vLLM** — a self-hosted model served on port `30001` (`make vllm`)

Export the matching API keys before running:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## A/B testing and ablation studies

Beyond ranking different agents and models, AgentKernelArena works as a
controlled A/B testing harness for agent-side capabilities — a new MCP server,
skill, prompt strategy, or tool integration.

Run the **same task set twice**, once with the capability enabled and once
without, then compare the standardized scores:

```bash
# Baseline
python main.py --run-suffix baseline

# With the new capability enabled in the agent configuration
python main.py --run-suffix with_capability
```

Both runs land in the same workspace directory with distinct run names, so the
[visualization dashboard](visualization.md) can show them side-by-side. Because
the tasks, environment, prompts, and scoring are held constant, score deltas
reflect the impact of the capability under test.

## Add a new agent

To integrate a custom agent:

1. Create `agents/<your_agent>/` with a launch function decorated with
   `@register_agent("<your_agent>")`.
2. Add an entry to the `AgentType` enum and an import branch in
   `src/module_registration.py`.
3. Wire the agent into the prompt builder and post-processing handler in
   `src/module_registration.py` if it needs the standard behavior.

See the development section of the repository `README.md` for the full template.
