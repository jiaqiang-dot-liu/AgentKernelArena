import logging
import os


def resolve_num_parallel(
    eval_config: dict,
    agent_config: dict,
    gpu_ids: str,
) -> int:
    """Resolve GEAK --num-parallel for the run.

    Decouples worker count from GPU count so the new GEAK gwiab-scheduler
    can run multiple subagents per GPU. Priority:
      1. GEAK_NUM_PARALLEL env var
      2. num_parallel in eval_config (top-level config.yaml)
      3. num_parallel in agent_config (agents/<name>/agent_config.yaml)
      4. len(gpu_ids) — historical default
    """
    raw = (
        os.environ.get("GEAK_NUM_PARALLEL")
        or eval_config.get("num_parallel")
        or (agent_config.get("num_parallel") if agent_config else None)
    )
    if raw is not None:
        try:
            n = int(raw)
            if n < 1:
                raise ValueError("must be >= 1")
            return n
        except (TypeError, ValueError) as e:
            logging.getLogger(__name__).warning(
                f"Invalid num_parallel={raw!r} ({e}); falling back to len(gpu_ids)"
            )
    return max(1, len(gpu_ids.split(",")))
