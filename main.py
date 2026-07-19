# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import re
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from src.tasks import get_task_config, is_flydsl_rewrite
from src.preprocessing import setup_workspace, setup_rocm_env, is_task_complete
from src.module_registration import AgentType, load_agent_launcher, load_post_processing_handler
from src.evaluator import measure_baseline, evaluate_kernel, write_task_result, write_rewrite_task_result
from src.runtime_env import apply_subprocess_python_path
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness


parser = argparse.ArgumentParser(description="arguments for AgentKernelArena")
parser.add_argument("--config_name", type=str, default="config.yaml",help="the config of AgentKernelArena, default set to config. \
                    You can set different tasks in different config yaml file in order to run multi evaluation task in one folder.")
parser.add_argument("--run-suffix", type=str, default=None,
                    help="Suffix appended to the run directory name, e.g. --run-suffix composer2_hip → run_20260416_120000_composer2_hip")
parser.add_argument("--resume-run", type=str, default=None,
                    help="Resume an existing run by specifying the run directory name (e.g., run_20250115_143022)")
parser.add_argument("--resume-latest", action="store_true",
                    help="Resume the most recent run in the workspace")

def main() -> None:
    """Main entry point for AgentKernelArena framework."""
    args = parser.parse_args()

    # Load config.yaml
    with open(args.config_name, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configuration
    tasks = config['tasks']  # Now directly a list
    agent_string = config['agent']['template']
    target_gpu_model = config['target_gpu_model']

    log_directory = config['log_directory']
    workspace_directory_prefix = config['workspace_directory_prefix']

    # Convert agent string to AgentType enum
    try:
        agent = AgentType.from_string(agent_string)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Build workspace directory name
    workspace_directory_name = f"{workspace_directory_prefix}_{target_gpu_model}_{agent.value}"    
    project_root = Path(__file__).resolve().parent
    workspace_directory = (project_root / workspace_directory_name).resolve()

    if args.run_suffix and not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_suffix):
        print("Error: --run-suffix may only contain letters, numbers, dot, underscore, and dash")
        return

    # Handle resume functionality
    resume_mode = False
    if args.resume_run:
        # Resume specific run
        run_directory_name = args.resume_run
        run_directory = workspace_directory / run_directory_name
        if not run_directory.exists():
            print(f"Error: Run directory does not exist: {run_directory}")
            return
        resume_mode = True
        # Extract the YYYYMMDD_HHMMSS timestamp from run directory name.
        # The name may include a suffix: run_20260429_194009_claude_opus_hip
        # Task directories use only the timestamp portion, not the suffix.
        m = re.match(r"^run_(\d{8}_\d{6})", run_directory_name)
        if m:
            timestamp = m.group(1)
        else:
            print(f"Error: Invalid run directory name format: {run_directory_name}. Expected format: run_YYYYMMDD_HHMMSS[_suffix]")
            return
    elif args.resume_latest:
        # Resume latest run
        # Find all run directories and get the most recent one
        run_dirs = sorted([d for d in workspace_directory.iterdir()
                          if d.is_dir()
                          and d.name.startswith("run_")
                          and not d.name.endswith("_heldout")],
                         key=lambda x: x.name, reverse=True)
        if not run_dirs:
            print(f"Error: No run directories found in {workspace_directory}")
            return
        run_directory = run_dirs[0]
        run_directory_name = run_directory.name
        resume_mode = True
        # Extract the YYYYMMDD_HHMMSS timestamp (same regex as --resume-run)
        m = re.match(r"^run_(\d{8}_\d{6})", run_directory_name)
        if m:
            timestamp = m.group(1)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        # Create new run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = f"_{args.run_suffix}" if args.run_suffix else ""
        run_directory_name = f"run_{timestamp}{suffix}"
        run_directory = workspace_directory / run_directory_name
        run_directory.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    log_filename = f"{target_gpu_model}_{agent.value}_{timestamp}{log_suffix}.log"
    log_path = log_dir / log_filename

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("AgentKernelArena Framework Started")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Agent: {agent.value}")
    logger.info(f"Target Architecture: {target_gpu_model}")
    logger.info(f"Workspace Directory: {workspace_directory}")
    logger.info(f"Run Directory: {run_directory}")
    if resume_mode:
        logger.info(f"RESUME MODE: Resuming existing run {run_directory_name}")
    else:
        logger.info(f"NEW RUN: Creating new run {run_directory_name}")

    python_path = apply_subprocess_python_path()
    logger.info(f"Subprocess Python environment: {python_path}")

    # Set PYTORCH_ROCM_ARCH based on target_gpu_model before any task runs
    setup_rocm_env(target_gpu_model, logger)

    # Load agent launcher
    try:
        agent_launcher = load_agent_launcher(agent, logger)
    except Exception as e:
        logger.error(f"Failed to load agent launcher: {e}")
        return


    # Get task config
    if 'all' in tasks:
        task_config_dict = get_task_config()
    else:
        task_config_dict = {}
        for category in tasks:
            task_config_dict.update(get_task_config(category=category))

    # Filter out completed tasks if resuming
    if resume_mode:
        original_task_count = len(task_config_dict)
        tasks_to_run = {}
        skipped_tasks = []
        
        for task_name, task_config_dir in task_config_dict.items():
            if is_task_complete(run_directory, task_name, timestamp):
                skipped_tasks.append(task_name)
                logger.info(f"Skipping completed task: {task_name}")
            else:
                tasks_to_run[task_name] = task_config_dir
        
        task_config_dict = tasks_to_run
        
        logger.info(f"Resume mode: {len(skipped_tasks)} tasks already completed, {len(task_config_dict)} tasks remaining")
        if skipped_tasks:
            logger.info(f"Skipped tasks: {skipped_tasks}")
        if len(task_config_dict) == 0:
            logger.info("All tasks are already completed. Nothing to run.")
            return

    logger.info(f"Found {len(task_config_dict)} tasks to execute")
    logger.info(f"Tasks: {list(task_config_dict.keys())}")

    # Collect workspace paths for post-processing
    workspace_paths = []

    # Run tasks
    for idx, (task_name, task_config_dir) in enumerate(task_config_dict.items(), 1):
        logger.info("=" * 80)
        logger.info(f"Task {idx}/{len(task_config_dict)}: {task_name}")
        logger.info("=" * 80)
        
        try:
            # Setup workspace
            workspace_path = setup_workspace(task_config_dir, run_directory, timestamp, logger, task_name=task_name)
            
            # Load task config for evaluation
            with open(task_config_dir, 'r') as f:
                task_config = yaml.safe_load(f)
            
            task_type = task_config.get('task_type', '')

            # The task_validator inspects the task and writes its own
            # validation_report.yaml; it does not optimize the kernel. Skip the whole
            # benchmark pipeline for it (baseline compile/measure, optimized
            # evaluation, and perf plots) — those would only re-measure the unchanged
            # kernel and emit a meaningless ~1.0x task_result.yaml plus plots. The
            # validator runs its own compile/correctness/performance checks.
            is_validator = (agent == AgentType.TASK_VALIDATOR)
            # A rewrite task (source language != target, e.g. triton2flydsl) is driven
            # by forge-rewrite: it ports the source kernel into FlyDSL, then reuses
            # forge-loop to optimize it, leaving the best kernel in the workspace. Skip
            # the generic baseline/evaluate pipeline; like torch2hip, Arena scores the
            # FINAL kernel by running the task's own driver commands (it does NOT trust
            # any agent result JSON). flydsl2flydsl (src == dst) is NOT a rewrite: it
            # optimizes an existing FlyDSL kernel via the generic path.
            is_rewrite = (not is_validator) and is_flydsl_rewrite(task_type)

            baseline_cases = []
            if is_validator:
                logger.info("task_validator run: skipping baseline/evaluation/perf-plot benchmark pipeline")
            elif is_rewrite:
                logger.info("triton2flydsl rewrite run: skipping generic baseline; Arena will score the final kernel via the task driver (the source is the driver's oracle + baseline)")
            elif task_type == 'torch2hip':
                logger.info("torch2hip task: skipping baseline compilation, measuring PyTorch baseline directly...")
                baseline_cases = measure_baseline(workspace_path, task_config, logger)
            else:
                from src.evaluator import evaluate_compilation
                logger.info("Compiling original kernel for baseline measurement...")
                pass_compilation, comp_error = evaluate_compilation(workspace_path, task_config, logger)
                if not pass_compilation:
                    logger.warning(f"Baseline compilation failed: {comp_error}")
                    logger.warning("Baseline measurement will be skipped")
                    baseline_cases = []
                else:
                    logger.info("Measuring baseline performance...")
                    baseline_cases = measure_baseline(workspace_path, task_config, logger)

            harness_snapshot = snapshot_workspace_harness(workspace_path)

            # Launch agent (agent should only generate optimized kernel)
            logger.info(f"Launching agent: {agent.value}")

            # For agentic approaches (cursor, claude_code, etc.)
            result = agent_launcher(
                eval_config=config,
                task_config_dir=task_config_dir,
                workspace=str(workspace_path)
            )

            logger.info(f"Agent execution completed")

            if is_validator:
                pass
            elif is_rewrite:
                # forge-rewrite (agent) ported + optimized the FlyDSL kernel, leaving
                # the best one in the workspace. Protect the harness, then let Arena
                # score the FINAL kernel by running the task's driver commands
                # (correctness + bench + source baseline) -- Arena is the scoring
                # authority, like torch2hip; it does NOT trust an agent result JSON.
                verify_workspace_harness(harness_snapshot)
                write_rewrite_task_result(workspace_path, task_config, task_name, agent.value, logger)
            else:
                verify_workspace_harness(harness_snapshot)
                # Agents work inside the task workspace and could accidentally
                # modify generated perf helpers. Re-materialize from src/tools/perf/
                # immediately before scoring so benchmark methodology stays
                # canonical.
                materialize_perf_helpers_in_workspace(workspace_path, logger=logger)

                # Centralized evaluation of optimized kernel
                logger.info("Running centralized evaluation...")
                evaluation_results = evaluate_kernel(
                    workspace_path,
                    task_config,
                    baseline_cases,
                    logger
                )

                # Write standardized task_result.yaml
                write_task_result(
                    workspace_path,
                    evaluation_results,
                    baseline_cases,
                    task_name,
                    agent.value,
                    logger
                )

            logger.info(f"Task {task_name} completed successfully")

            # Add workspace path to list for post-processing
            workspace_paths.append(str(workspace_path))

        except Exception as e:
            logger.error(f"Task {task_name} failed with error: {e}", exc_info=True)
            # Still add workspace path even if task failed (for post-processing to record failure)
            if 'workspace_path' in locals():
                workspace_paths.append(str(workspace_path))
            continue

    # Run post-processing to generate report
    logger.info("=" * 80)
    logger.info("Running Post-Processing")
    logger.info("=" * 80)

    try:
        post_processing_handler = load_post_processing_handler(agent, logger)
        post_processing_handler(workspace_paths, logger)
    except NotImplementedError as e:
        logger.warning(f"Post-processing skipped: {e}")
    except Exception as e:
        logger.error(f"Post-processing failed: {e}", exc_info=True)

    logger.info("=" * 80)
    logger.info("AgentKernelArena Framework Completed")
    logger.info("=" * 80)



if __name__ == "__main__":
    main()
